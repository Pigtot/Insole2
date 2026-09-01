"""Tests for the Winkler contact model.

These are correctness gates, not smoke tests: each one checks a property the
model must have for its output to mean anything.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.registration.contact import (  # noqa: E402
    dome_profile, plantar_profile_from_mesh, series_stiffness, solve_contact,
)

SHAPE = (24, 56)
CELL_A = 0.003 ** 2
LOAD = 700.0


def uniform_K(E_pa=10e6, h=0.010, tissue=None):
    return series_stiffness(np.full(SHAPE, E_pa), h, tissue_pa_per_m=tissue)


# --- stiffness combination ------------------------------------------------
def test_series_stiffness_is_softer_than_either_spring():
    k_insole = series_stiffness(np.full(SHAPE, 10e6), 0.010, tissue_pa_per_m=None)
    both = series_stiffness(np.full(SHAPE, 10e6), 0.010, tissue_pa_per_m=2e6)
    assert np.all(both < k_insole)
    assert np.all(both < 2e6)


def test_series_stiffness_rejects_bad_geometry():
    with pytest.raises(ValueError):
        series_stiffness(np.ones(SHAPE), 0.0)
    with pytest.raises(ValueError):
        series_stiffness(np.ones(SHAPE), 0.01, tissue_pa_per_m=-1)


# --- the three correctness gates -----------------------------------------
def test_flat_foot_on_uniform_insole_gives_uniform_pressure():
    sol = solve_contact(np.zeros(SHAPE), uniform_K(), load_n=LOAD,
                        cell_area_m2=CELL_A, allow_tilt=False)
    assert np.ptp(sol.pressure_pa) == pytest.approx(0.0, abs=1e-6)
    assert sol.contact_mask.all()


def test_integrated_pressure_equals_applied_load():
    for prof in (np.zeros(SHAPE), dome_profile(SHAPE, 0.01)):
        sol = solve_contact(prof, uniform_K(), load_n=LOAD, cell_area_m2=CELL_A,
                            allow_tilt=False)
        assert sol.total_force_n == pytest.approx(LOAD, rel=0.01)
        assert sol.converged


def test_dome_concentrates_pressure_at_its_lowest_point():
    """Hertz-like: a curved indenter must load its centre, not its rim."""
    sol = solve_contact(dome_profile(SHAPE, 0.02), uniform_K(), load_n=LOAD,
                        cell_area_m2=CELL_A, allow_tilt=False)
    peak = np.unravel_index(sol.pressure_pa.argmax(), SHAPE)
    centre = ((SHAPE[0] - 1) / 2, (SHAPE[1] - 1) / 2)
    assert abs(peak[0] - centre[0]) <= 1 and abs(peak[1] - centre[1]) <= 1
    assert sol.contact_mask.sum() < sol.pressure_pa.size


# --- contact physics ------------------------------------------------------
def test_pressure_is_never_negative():
    """Unilateral contact: the insole pushes, it never pulls."""
    sol = solve_contact(dome_profile(SHAPE, 0.03), uniform_K(), load_n=LOAD,
                        cell_area_m2=CELL_A, allow_tilt=False)
    assert np.all(sol.pressure_pa >= 0)
    assert np.all(sol.indentation_m >= 0)


def test_lifted_regions_carry_no_load():
    prof = np.zeros(SHAPE)
    prof[:, :10] = 0.05                       # a 50 mm step: cannot be reached
    sol = solve_contact(prof, uniform_K(), load_n=LOAD, cell_area_m2=CELL_A,
                        allow_tilt=False)
    assert sol.pressure_pa[:, :10].max() == 0.0
    assert sol.total_force_n == pytest.approx(LOAD, rel=0.01)


def test_softer_insole_spreads_load_and_lowers_peak():
    prof = dome_profile(SHAPE, 0.01)
    stiff = solve_contact(prof, uniform_K(40e6), load_n=LOAD, cell_area_m2=CELL_A,
                          allow_tilt=False)
    soft = solve_contact(prof, uniform_K(5e6), load_n=LOAD, cell_area_m2=CELL_A,
                         allow_tilt=False)
    assert soft.contact_area_m2 > stiff.contact_area_m2
    assert soft.peak_pressure_pa < stiff.peak_pressure_pa


def test_target_cop_is_respected():
    """Prescribing the COP is what makes design comparisons fair."""
    target = (6.0, 14.0)
    sol = solve_contact(np.zeros(SHAPE), uniform_K(), load_n=LOAD,
                        cell_area_m2=CELL_A, target_cop=target)
    cu, cv = sol.center_of_pressure()
    assert cu == pytest.approx(target[0], abs=1.0)
    assert cv == pytest.approx(target[1], abs=1.0)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        solve_contact(np.zeros(SHAPE), uniform_K(), load_n=-1, cell_area_m2=CELL_A)
    with pytest.raises(ValueError):
        solve_contact(np.zeros((4, 4)), uniform_K(), load_n=LOAD, cell_area_m2=CELL_A)


def test_summary_reports_force_error():
    sol = solve_contact(np.zeros(SHAPE), uniform_K(), load_n=LOAD, cell_area_m2=CELL_A,
                        allow_tilt=False)
    s = sol.summary()
    assert s["force_error_pct"] < 1.0
    assert s["peak_pressure_kpa"] > 0
    assert s["contact_area_cm2"] > 0


# --- plantar profile ------------------------------------------------------
def test_plantar_profile_is_zero_at_its_lowest_point():
    rng = np.random.default_rng(0)
    verts = rng.uniform(-0.05, 0.05, (500, 3))
    verts[:, 0] *= 3.0                        # make one axis clearly the long one
    prof = plantar_profile_from_mesh(verts, np.arange(len(verts)), (12, 28))
    assert prof.min() == pytest.approx(0.0)
    assert np.all(prof >= 0)


def test_plantar_profile_rejects_empty_selection():
    with pytest.raises(ValueError):
        plantar_profile_from_mesh(np.zeros((10, 3)), np.array([], dtype=int), (8, 8))


# --- bottoming out -------------------------------------------------------
def test_bottoming_out_is_reported():
    """Too soft and too thin: the insole crushes flat and stops cushioning."""
    prof = dome_profile(SHAPE, 0.004)
    E = np.full(SHAPE, 0.15e6)
    h = 0.004
    sol = solve_contact(prof, series_stiffness(E, h), load_n=LOAD, cell_area_m2=CELL_A,
                        allow_tilt=False, thickness_m=h,
                        insole_stiffness_pa_per_m=E / h)
    assert sol.bottomed_fraction > 0.0


def test_stiff_insole_does_not_bottom_out():
    prof = dome_profile(SHAPE, 0.004)
    E = np.full(SHAPE, 40e6)
    h = 0.010
    sol = solve_contact(prof, series_stiffness(E, h), load_n=LOAD, cell_area_m2=CELL_A,
                        allow_tilt=False, thickness_m=h,
                        insole_stiffness_pa_per_m=E / h)
    assert sol.bottomed_fraction == 0.0


def test_densification_limit_uses_the_insole_share_not_total_indentation():
    """Springs in series share displacement. Charging the whole indentation to the
    insole (an earlier bug) makes bottoming look far worse than it is."""
    prof = np.zeros(SHAPE)
    E = np.full(SHAPE, 2.0e6)
    h = 0.010
    K = series_stiffness(E, h)                       # includes soft tissue
    soft_tissue = solve_contact(prof, K, load_n=LOAD, cell_area_m2=CELL_A,
                                allow_tilt=False, thickness_m=h,
                                insole_stiffness_pa_per_m=E / h)
    # Same insole, no tissue in series -> the insole takes ALL the indentation,
    # so it must be at least as bottomed as when tissue shares the load.
    K_only = series_stiffness(E, h, tissue_pa_per_m=None)
    no_tissue = solve_contact(prof, K_only, load_n=LOAD, cell_area_m2=CELL_A,
                              allow_tilt=False, thickness_m=h,
                              insole_stiffness_pa_per_m=E / h)
    assert no_tissue.bottomed_fraction >= soft_tissue.bottomed_fraction


def test_bottoming_requires_thickness_information():
    """Without a thickness there is nothing to bottom out against."""
    sol = solve_contact(np.zeros(SHAPE), uniform_K(), load_n=LOAD,
                        cell_area_m2=CELL_A, allow_tilt=False)
    assert sol.bottomed_fraction == 0.0
