"""Tests for the canonical plantar frame.

The property that matters: both feet land in ONE frame with anatomy, not sensor
index, deciding position. And interpolation must never be mistakable for
measurement -- hence the support mask.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.pressure.canonical_foot import (  # noqa: E402
    CANONICAL_REGIONS, CanonicalField, both_feet_to_canonical, canonical_grid,
    canonical_to_sensors, lift_to_canonical, sensor_uv,
)
from visole.pressure.sensor_map import LEFT_RIGHT_CORRESPONDENCE, load_sensor_map  # noqa: E402

MAPS = (REPO / "data/raw/insole_gaitrite/sensors_map_left.svg").exists()
requires_maps = pytest.mark.skipif(not MAPS, reason="sensor maps not downloaded")


def test_grid_cell_centres_lie_inside_the_unit_square():
    U, V = canonical_grid((4, 6))
    assert U.shape == V.shape == (4, 6)
    assert 0 < U.min() and U.max() < 1
    assert 0 < V.min() and V.max() < 1


def test_regions_tile_the_heel_to_toe_axis_without_gaps():
    bounds = [(lo, hi) for _, lo, hi in CANONICAL_REGIONS]
    assert bounds[0][0] == 0.0
    for (_, hi), (lo, _) in zip(bounds, bounds[1:]):
        assert hi == lo
    assert bounds[-1][1] >= 1.0


def test_source_must_be_a_declared_provenance():
    ok = CanonicalField(np.zeros((2, 2)), np.ones((2, 2), bool), "counts",
                        "measured_sensor", "left")
    assert ok.supported_fraction == 1.0
    with pytest.raises(ValueError, match="source must be"):
        CanonicalField(np.zeros((2, 2)), np.ones((2, 2), bool), "counts",
                       "vibes", "left")


def test_mismatched_support_shape_rejected():
    with pytest.raises(ValueError, match="same shape"):
        CanonicalField(np.zeros((2, 2)), np.ones((3, 3), bool), "c",
                       "synthetic", "left")


@requires_maps
def test_lift_produces_the_requested_grid_and_a_partial_support_mask():
    smap = load_sensor_map("left")
    f = lift_to_canonical(np.ones(32), smap, grid=(24, 56))
    assert f.values.shape == (24, 56)
    # Support must be neither empty nor the whole grid: a foot is not a rectangle.
    assert 0.2 < f.supported_fraction < 1.0


@requires_maps
def test_support_shrinks_as_the_radius_shrinks():
    smap = load_sensor_map("left")
    wide = lift_to_canonical(np.ones(32), smap, support_radius=0.30)
    tight = lift_to_canonical(np.ones(32), smap, support_radius=0.05)
    assert tight.supported_fraction < wide.supported_fraction


@requires_maps
def test_masked_view_hides_unsupported_cells():
    f = lift_to_canonical(np.ones(32), load_sensor_map("left"))
    m = f.masked()
    assert np.ma.is_masked(m)
    assert m.count() == int(f.support.sum())


@requires_maps
def test_constant_input_lifts_to_a_constant_field():
    f = lift_to_canonical(np.full(32, 7.0), load_sensor_map("right"))
    assert np.allclose(f.values, 7.0)


@requires_maps
def test_lift_then_sample_round_trips_close_to_the_original():
    """If this drifts, the canonical frame is distorting the measurements."""
    smap = load_sensor_map("left")
    rng = np.random.default_rng(0)
    v = rng.uniform(0, 2000, 32)
    back = canonical_to_sensors(lift_to_canonical(v, smap), smap)
    assert np.abs(back - v).mean() < 0.10 * np.ptp(v)


@requires_maps
def test_wrong_channel_count_rejected():
    with pytest.raises(ValueError):
        lift_to_canonical(np.ones(31), load_sensor_map("left"))


@requires_maps
def test_homologous_sensors_occupy_the_same_canonical_position():
    """The whole point of one shared frame: anatomy decides position, not index."""
    l_uv, r_uv = sensor_uv(load_sensor_map("left")), sensor_uv(load_sensor_map("right"))
    for right_id, left_id in LEFT_RIGHT_CORRESPONDENCE.items():
        assert np.allclose(r_uv[right_id], l_uv[left_id], atol=1e-9)


@requires_maps
def test_both_feet_loaded_at_homologous_sites_give_matching_canonical_fields():
    """Load the same anatomical site on each foot; the canonical fields must agree
    even though the two insoles number that site differently."""
    right_id, left_id = 0, LEFT_RIGHT_CORRESPONDENCE[0]
    lv = np.zeros(32); lv[left_id] = 1000.0
    rv = np.zeros(32); rv[right_id] = 1000.0
    fl, fr = both_feet_to_canonical(lv, rv)
    assert np.allclose(fl.values, fr.values, atol=1e-6)
    assert np.unravel_index(fl.values.argmax(), fl.values.shape) == \
           np.unravel_index(fr.values.argmax(), fr.values.shape)


@requires_maps
def test_naive_index_matching_would_have_been_wrong():
    """Guards the trap: using the SAME index on both feet puts the load in
    different anatomical places."""
    v = np.zeros(32); v[0] = 1000.0
    fl, fr = both_feet_to_canonical(v, v)
    assert not np.allclose(fl.values, fr.values, atol=1e-6)


@requires_maps
def test_region_means_respond_to_where_load_is_placed():
    smap = load_sensor_map("left")
    groups = smap.region_indices()
    v = np.zeros(32)
    v[groups["heel"]] = 1000.0
    means = lift_to_canonical(v, smap).region_means()
    assert means["heel"] > means["toes"]
