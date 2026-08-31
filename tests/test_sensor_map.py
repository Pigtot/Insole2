"""Tests for insole sensor geometry.

The left/right numbering hazard is the reason this file exists: the two insoles
share pad geometry by mirroring but use different sensor indices, so a test
suite that only checked shapes would miss the bug that matters.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.pressure.sensor_map import (  # noqa: E402
    LEFT_RIGHT_CORRESPONDENCE,
    N_SENSORS,
    SensorMap,
    load_sensor_map,
)

MAPS_PRESENT = (REPO / "data/raw/insole_gaitrite/sensors_map_left.svg").exists()
requires_maps = pytest.mark.skipif(not MAPS_PRESENT, reason="sensor maps not downloaded")


def test_correspondence_is_a_bijection_on_0_31():
    assert sorted(LEFT_RIGHT_CORRESPONDENCE) == list(range(N_SENSORS))
    assert sorted(LEFT_RIGHT_CORRESPONDENCE.values()) == list(range(N_SENSORS))


def test_no_sensor_index_is_shared_between_feet():
    """The headline hazard: left[i] is never the same site as right[i]."""
    same = [r for r, l in LEFT_RIGHT_CORRESPONDENCE.items() if r == l]
    assert same == [], f"indices {same} would be dangerously easy to assume symmetric"


@requires_maps
@pytest.mark.parametrize("side", ["left", "right"])
def test_map_parses_to_32_sensors(side):
    m = load_sensor_map(side)
    assert isinstance(m, SensorMap)
    assert m.xy.shape == (N_SENSORS, 2)
    assert len(m.pads) == N_SENSORS
    assert m.side == side


@requires_maps
@pytest.mark.parametrize("side", ["left", "right"])
def test_normalised_coordinates_span_unit_square(side):
    m = load_sensor_map(side)
    assert m.norm_xy.min() == pytest.approx(0.0, abs=1e-9)
    assert m.norm_xy.max() == pytest.approx(1.0, abs=1e-9)


@requires_maps
def test_homologous_sensors_share_normalised_coordinates():
    """The correspondence table must agree with the parsed geometry exactly."""
    r, l = load_sensor_map("right"), load_sensor_map("left")
    ridx = list(LEFT_RIGHT_CORRESPONDENCE.keys())
    lidx = [LEFT_RIGHT_CORRESPONDENCE[i] for i in ridx]
    assert np.allclose(r.norm_xy[ridx], l.norm_xy[lidx], atol=1e-9)


@requires_maps
def test_pads_have_consistent_area():
    """All 32 pads are the same physical sensor; wildly varying area means a
    mis-chained outline leaked into the pad list."""
    from visole.pressure.sensor_map import _polygon_area

    for side in ("left", "right"):
        areas = np.array([_polygon_area(p) for p in load_sensor_map(side).pads])
        assert areas.std() / areas.mean() < 0.01


@requires_maps
def test_to_other_foot_round_trips():
    r, l = load_sensor_map("right"), load_sensor_map("left")
    v = np.random.default_rng(0).normal(size=N_SENSORS)
    assert np.array_equal(l.to_other_foot(r.to_other_foot(v)), v)


@requires_maps
def test_to_other_foot_moves_values_to_homologous_sites():
    r, l = load_sensor_map("right"), load_sensor_map("left")
    v = np.arange(N_SENSORS, dtype=float)
    as_left = r.to_other_foot(v)
    for right_id, left_id in LEFT_RIGHT_CORRESPONDENCE.items():
        assert as_left[left_id] == v[right_id]


@requires_maps
def test_to_other_foot_rejects_wrong_width():
    r = load_sensor_map("right")
    with pytest.raises(ValueError):
        r.to_other_foot(np.zeros(31))


@requires_maps
def test_to_other_foot_supports_batched_input():
    r = load_sensor_map("right")
    batch = np.random.default_rng(1).normal(size=(5, N_SENSORS))
    out = r.to_other_foot(batch)
    assert out.shape == batch.shape
    for right_id, left_id in LEFT_RIGHT_CORRESPONDENCE.items():
        assert np.array_equal(out[:, left_id], batch[:, right_id])


@requires_maps
def test_both_feet_have_identical_region_counts():
    """Mirrored geometry must yield the same number of sensors per region."""
    rl = {k: len(v) for k, v in load_sensor_map("right").region_indices().items()}
    ll = {k: len(v) for k, v in load_sensor_map("left").region_indices().items()}
    assert rl == ll


@requires_maps
def test_midfoot_is_sparse_as_the_layout_implies():
    """The medial arch does not contact, so the midfoot carries few sensors.
    If this ever fails, the heel<->toe axis has probably been flipped."""
    for side in ("left", "right"):
        groups = load_sensor_map(side).region_indices()
        assert len(groups["midfoot"]) < len(groups["heel"])
        assert len(groups["midfoot"]) < len(groups["metatarsal"])


def test_bad_side_rejected():
    with pytest.raises(ValueError):
        load_sensor_map("port")
