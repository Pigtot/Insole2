"""Tests for the implicit lattice engine and the pressure->material mappings."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.lattice.implicit import (  # noqa: E402
    TPMS_FUNCTIONS, calibrate_density_vs_thickness, generate_graded_tpms, gyroid,
)
from visole.lattice.mapping import (  # noqa: E402
    DensityMapping, DensityToThickness, normalise_pressure,
)

SMALL = dict(bounds_mm=((0.0, 16.0), (0.0, 16.0), (0.0, 8.0)),
             cell_size_mm=8.0, resolution=8)


# --- the implicit surface -------------------------------------------------
def test_gyroid_matches_its_definition():
    rng = np.random.default_rng(0)
    x, y, z = rng.normal(size=(3, 200))
    expected = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    assert np.allclose(gyroid(x, y, z), expected)


def test_gyroid_is_2pi_periodic():
    rng = np.random.default_rng(1)
    p = rng.normal(size=(3, 50))
    shifted = p + np.array([[2 * np.pi], [0.0], [0.0]])
    assert np.allclose(gyroid(*p), gyroid(*shifted))


@pytest.mark.parametrize("name", sorted(TPMS_FUNCTIONS))
def test_every_topology_is_signed(name):
    """A usable level set must take both signs, or there is no surface."""
    g = np.linspace(0, 2 * np.pi, 24)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    f = TPMS_FUNCTIONS[name](X, Y, Z)
    assert f.min() < 0 < f.max()


# --- density calibration --------------------------------------------------
def test_density_increases_monotonically_with_thickness():
    cal = calibrate_density_vs_thickness(thicknesses=np.linspace(0.1, 1.2, 8), resolution=20)
    d = np.asarray(cal["relative_density"])
    assert np.all(np.diff(d) > 0)
    assert 0 < d[0] < d[-1] <= 1.0


def test_thickness_inversion_round_trips():
    cal = calibrate_density_vs_thickness(resolution=20)
    inv = DensityToThickness(cal)
    for t, d in zip(cal["thickness"], cal["relative_density"]):
        assert inv(d) == pytest.approx(t, abs=0.06)


# --- generation -----------------------------------------------------------
def test_uniform_coupon_is_watertight_and_connected():
    lat = generate_graded_tpms(constant_thickness=0.5, **SMALL)
    assert lat.watertight
    assert lat.n_components == 1
    assert len(lat.mesh.faces) > 100


def test_thicker_lattice_is_denser():
    thin = generate_graded_tpms(constant_thickness=0.35, **SMALL)
    thick = generate_graded_tpms(constant_thickness=0.75, **SMALL)
    assert thick.relative_density > thin.relative_density


def test_very_thin_lattice_is_reported_as_fragmented_not_silently_accepted():
    """The failure mode that matters: a part that looks fine by density but is
    actually thousands of disconnected islands."""
    lat = generate_graded_tpms(constant_thickness=0.12, **SMALL)
    assert lat.n_components > 1


def test_graded_thickness_actually_varies_the_geometry():
    def ramp(X, Y):
        return 0.3 + 0.5 * (X - X.min()) / max(np.ptp(X), 1e-9)

    graded = generate_graded_tpms(thickness_fn=ramp, **SMALL)
    assert np.ptp(graded.thickness_field) > 0.4
    uniform = generate_graded_tpms(constant_thickness=0.3, **SMALL)
    assert graded.relative_density > uniform.relative_density


def test_thickness_fn_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        generate_graded_tpms(thickness_fn=lambda X, Y: np.zeros((3, 3)), **SMALL)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        generate_graded_tpms(topology="not_a_surface", **SMALL)
    with pytest.raises(ValueError):
        generate_graded_tpms(bounds_mm=((0, 0), (0, 10), (0, 10)), cell_size_mm=8.0)
    with pytest.raises(ValueError):
        generate_graded_tpms(thickness_fn=lambda X, Y: -np.ones_like(X), **SMALL)


def test_summary_reports_both_density_estimators_without_reconciling_them():
    s = generate_graded_tpms(constant_thickness=0.5, **SMALL).summary()
    assert s["relative_density_voxel_count"] > 0
    assert s["relative_density_from_mesh_volume"] > 0
    assert "connected" in s


def test_export_writes_a_readable_stl(tmp_path):
    import trimesh

    lat = generate_graded_tpms(constant_thickness=0.5, **SMALL)
    path = tmp_path / "c.stl"
    lat.export_stl(path)
    assert path.stat().st_size > 0
    assert len(trimesh.load(path).faces) == len(lat.mesh.faces)


# --- pressure -> material mapping -----------------------------------------
def test_stiffen_and_soften_are_opposite_hypotheses():
    p = np.linspace(0, 1, 11)
    stiff = DensityMapping("stiffen")(p)
    soft = DensityMapping("soften")(p)
    assert stiff[-1] > stiff[0]
    assert soft[-1] < soft[0]
    assert np.allclose(stiff + soft, stiff[0] + soft[0])


def test_uniform_hypothesis_ignores_pressure():
    out = DensityMapping("uniform")(np.linspace(0, 1, 7))
    assert np.ptp(out) == 0


def test_mapping_stays_inside_its_density_bounds():
    m = DensityMapping("stiffen", rho_min=0.2, rho_max=0.5)
    out = m(np.linspace(-2, 3, 50))       # deliberately out of range
    assert out.min() >= 0.2 - 1e-9
    assert out.max() <= 0.5 + 1e-9


def test_default_rho_min_is_above_the_connectivity_floor():
    """Guards the measured floor: below ~0.19 the gyroid sheet fragments."""
    assert DensityMapping().rho_min >= 0.19


def test_gamma_shapes_the_response():
    p = np.linspace(0, 1, 21)
    assert np.all(DensityMapping(gamma=2.0)(p) <= DensityMapping(gamma=1.0)(p) + 1e-12)


def test_invalid_mapping_parameters_rejected():
    for kwargs in ({"hypothesis": "melt"}, {"rho_min": 0.6, "rho_max": 0.3},
                   {"gamma": 0.0}, {"rho_max": 1.5}):
        with pytest.raises(ValueError):
            DensityMapping(**kwargs)


def test_normalise_uses_a_shared_reference_so_designs_are_comparable():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([0.0, 2.0, 4.0])
    assert np.allclose(normalise_pressure(a, reference=4.0), [0, 0.25, 0.5])
    assert np.allclose(normalise_pressure(b, reference=4.0), [0, 0.5, 1.0])
    # without a shared reference both would normalise to the same thing
    assert np.allclose(normalise_pressure(a), normalise_pressure(b))


def test_normalise_handles_an_all_zero_field():
    assert np.all(normalise_pressure(np.zeros(5)) == 0)


# --- foot-shaped insole ---------------------------------------------------
def _footprint(shape=(16, 36)):
    nu, nv = shape
    u, v = np.meshgrid(np.linspace(-1, 1, nu), np.linspace(0, 1, nv), indexing="ij")
    return (u ** 2 / (0.85 - 0.35 * np.cos(np.pi * v)) ** 2) < 1


def test_insole_lattice_is_clipped_to_the_footprint():
    """A rectangular block would waste material and be the wrong part."""
    from visole.lattice.implicit import generate_insole_lattice

    fp = _footprint()
    lat = generate_insole_lattice(fp, np.full(fp.shape, 0.6), resolution=4)
    block = generate_insole_lattice(np.ones_like(fp), np.full(fp.shape, 0.6), resolution=4)
    assert lat.mesh.volume < block.mesh.volume


def test_insole_lattice_is_watertight_and_connected_with_a_rim():
    from visole.lattice.implicit import generate_insole_lattice

    fp = _footprint()
    lat = generate_insole_lattice(fp, np.full(fp.shape, 0.6), resolution=4, rim_mm=2.5)
    assert lat.watertight
    assert lat.n_components == 1


def test_graded_thickness_changes_insole_density():
    from visole.lattice.implicit import generate_insole_lattice

    fp = _footprint()
    thin = generate_insole_lattice(fp, np.full(fp.shape, 0.35), resolution=4)
    thick = generate_insole_lattice(fp, np.full(fp.shape, 0.85), resolution=4)
    assert thick.relative_density > thin.relative_density


def test_substantial_components_are_never_silently_discarded():
    """A shelled part is a sealed box: marching cubes emits an inner AND an outer
    surface. Dropping the 'second component' there would destroy the geometry."""
    from visole.lattice.implicit import generate_insole_lattice

    fp = _footprint()
    shelled = generate_insole_lattice(fp, np.full(fp.shape, 0.6), resolution=4,
                                      rim_mm=2.5, shell_mm=1.5)
    assert shelled.discarded_fragment_mm3 == 0.0
    assert shelled.mesh.volume > 0


def test_insole_rejects_mismatched_inputs():
    from visole.lattice.implicit import generate_insole_lattice

    with pytest.raises(ValueError):
        generate_insole_lattice(_footprint(), np.zeros((4, 4)), resolution=4)
    with pytest.raises(ValueError):
        generate_insole_lattice(np.zeros((8, 8), bool), np.full((8, 8), 0.6), resolution=4)
