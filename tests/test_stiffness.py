"""Tests for the lattice stiffness measurement.

The central test is that the solver reproduces a known answer: a solid block
must return the modulus it was given. Everything else in this module -- and the
whole contact model downstream -- is only meaningful if that holds.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.lattice.stiffness import (  # noqa: E402
    TPU_MODULUS_PA, TPU_POISSON, effective_modulus, fit_gibson_ashby,
    load_bearing_component, voxel_hex_mesh,
)

SPACING = (0.01 / 6, 0.01 / 6, 0.01 / 6)


# --- meshing --------------------------------------------------------------
def test_voxel_mesh_has_one_element_per_occupied_voxel():
    occ = np.zeros((4, 4, 4), bool)
    occ[1:3, 1:3, :] = True
    m = voxel_hex_mesh(occ, SPACING)
    assert m.t.shape[1] == int(occ.sum())
    assert m.t.shape[0] == 8


def test_voxel_mesh_elements_have_positive_volume():
    """Wrong corner ordering silently inverts elements and corrupts stiffness."""
    occ = np.ones((3, 3, 3), bool)
    m = voxel_hex_mesh(occ, SPACING)
    p = m.p[:, m.t]                                  # (3, 8, n_elems)
    # local 0 -> (0,0,0), 2 -> (1,0,0), 1 -> (0,1,0), 3 -> (0,0,1)
    e1 = p[:, 2] - p[:, 0]
    e2 = p[:, 1] - p[:, 0]
    e3 = p[:, 3] - p[:, 0]
    vol = np.einsum("ij,ij->j", np.cross(e1, e2, axis=0), e3)
    assert np.all(np.abs(vol) > 0)


def test_empty_occupancy_is_rejected():
    with pytest.raises(ValueError):
        voxel_hex_mesh(np.zeros((3, 3, 3), bool), SPACING)


# --- connectivity ---------------------------------------------------------
def test_load_bearing_component_drops_floating_islands():
    occ = np.zeros((5, 5, 5), bool)
    occ[2, 2, :] = True          # a column spanning bottom to top
    occ[0, 0, 2] = True          # a floating speck
    keep = load_bearing_component(occ)
    assert keep[2, 2, 0] and keep[2, 2, -1]
    assert not keep[0, 0, 2]


def test_non_spanning_material_yields_no_stiffness():
    """Material that never bridges base to top carries no load -- the physically
    correct answer is zero, not a singular solve."""
    occ = np.zeros((4, 4, 4), bool)
    occ[:, :, 1] = True          # a floating slab, touching neither face
    r = effective_modulus(occ, SPACING)
    assert r.effective_modulus_pa == 0.0


# --- the anchor test ------------------------------------------------------
def test_solid_block_recovers_its_own_modulus():
    occ = np.ones((6, 6, 6), bool)
    r = effective_modulus(occ, SPACING, E_s=TPU_MODULUS_PA, nu=TPU_POISSON)
    assert r.effective_modulus_pa == pytest.approx(TPU_MODULUS_PA, rel=0.02)
    assert r.relative_density == pytest.approx(1.0)


def test_modulus_scales_linearly_with_base_material():
    """Linear elasticity: doubling E_s must double E_eff exactly."""
    occ = np.ones((4, 4, 4), bool)
    a = effective_modulus(occ, SPACING, E_s=20e6).effective_modulus_pa
    b = effective_modulus(occ, SPACING, E_s=40e6).effective_modulus_pa
    assert b == pytest.approx(2 * a, rel=1e-6)


def test_modulus_is_independent_of_applied_strain():
    """Also linearity -- a nonlinear result here would mean a solver bug."""
    occ = np.ones((4, 4, 4), bool)
    a = effective_modulus(occ, SPACING, strain=0.005).effective_modulus_pa
    b = effective_modulus(occ, SPACING, strain=0.02).effective_modulus_pa
    assert a == pytest.approx(b, rel=1e-6)


def test_removing_material_reduces_stiffness():
    full = np.ones((6, 6, 6), bool)
    holed = full.copy()
    holed[2:4, 2:4, :] = False
    e_full = effective_modulus(full, SPACING).effective_modulus_pa
    e_holed = effective_modulus(holed, SPACING).effective_modulus_pa
    assert 0 < e_holed < e_full


# --- power-law fit --------------------------------------------------------
def test_gibson_ashby_recovers_a_known_power_law():
    rho = np.linspace(0.2, 0.8, 10)
    fit = fit_gibson_ashby(rho, 0.6 * rho ** 1.7)
    assert fit.C == pytest.approx(0.6, rel=1e-6)
    assert fit.n == pytest.approx(1.7, rel=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


def test_gibson_ashby_is_monotonic_and_labels_regime():
    fit = fit_gibson_ashby(np.linspace(0.2, 0.8, 6), 0.5 * np.linspace(0.2, 0.8, 6) ** 2)
    assert fit(0.5) > fit(0.3)
    assert fit.regime in {"stretch-dominated", "mixed", "bending-dominated"}
    assert "rho^" in fit.describe()


def test_power_law_fit_needs_enough_points():
    with pytest.raises(ValueError):
        fit_gibson_ashby(np.array([0.5]), np.array([0.2]))
