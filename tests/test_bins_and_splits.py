"""Tests for the pressure-bin representation and participant splits."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.data.splits import Split, leave_one_subject_out, participant_split  # noqa: E402
from visole.pressure.bins import PressureBins  # noqa: E402


# --- bins -----------------------------------------------------------------
def test_default_layout_has_nine_ordered_bins():
    b = PressureBins.log_spaced()
    assert b.n_bins == 9
    assert b.edges[0] == 0.0
    assert np.all(np.diff(b.edges) > 0)


def test_encode_is_monotonic_in_value():
    b = PressureBins.log_spaced()
    v = np.array([0, 0.5, 2, 20, 200, 2000, 99999.0])
    idx = b.encode(v)
    assert np.all(np.diff(idx) >= 0)
    assert idx[0] == 0
    assert idx[-1] == b.n_bins - 1, "values above the top edge must land in the last bin"


def test_zero_maps_to_the_no_contact_bin():
    b = PressureBins.log_spaced()
    assert b.encode(np.zeros(5)).tolist() == [0] * 5


def test_encode_preserves_shape():
    b = PressureBins.log_spaced()
    assert b.encode(np.zeros((7, 32))).shape == (7, 32)


def test_negative_values_rejected():
    """Negative counts mean the caller forgot to baseline-correct and clip."""
    b = PressureBins.log_spaced()
    with pytest.raises(ValueError, match="negative"):
        b.encode(np.array([-1.0]))


def test_decode_lands_inside_its_bin():
    b = PressureBins.log_spaced()
    for i in range(1, b.n_bins - 1):
        rep = b.decode(np.array([i]))[0]
        assert b.edges[i] <= rep < b.edges[i + 1]


def test_decode_of_contact_bin_is_zero():
    b = PressureBins.log_spaced()
    assert b.decode(np.array([0]))[0] == 0.0


def test_one_hot_is_one_hot():
    b = PressureBins.log_spaced()
    oh = b.one_hot(np.array([0.0, 5.0, 2000.0]))
    assert oh.shape == (3, b.n_bins)
    assert np.all(oh.sum(-1) == 1)
    assert np.array_equal(oh.argmax(-1), b.encode(np.array([0.0, 5.0, 2000.0])))


def test_quantisation_error_is_reported_not_hidden():
    b = PressureBins.log_spaced()
    v = np.random.default_rng(0).uniform(0, 3000, size=2000)
    q = b.quantisation_error(v)
    assert q["mae"] > 0, "binning is lossy; a zero here would mean the metric is broken"
    assert q["median_ae"] <= q["p95_ae"] <= q["max_ae"]


def test_bin_distance_matrix_is_ordinal():
    b = PressureBins.log_spaced()
    d = b.bin_distance_matrix()
    assert d.shape == (b.n_bins, b.n_bins)
    assert np.all(np.diag(d) == 0)
    assert d[0, -1] == b.n_bins - 1
    assert d[0, 2] > d[0, 1], "distant bin confusions must cost more than adjacent ones"


def test_malformed_edges_rejected():
    with pytest.raises(ValueError):
        PressureBins(np.array([0.0, 5.0, 3.0]))       # not increasing
    with pytest.raises(ValueError):
        PressureBins(np.array([1.0, 5.0]))            # must start at 0
    with pytest.raises(ValueError):
        PressureBins.log_spaced(n_bins=1)


# --- splits ---------------------------------------------------------------
PARTICIPANTS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]


def test_split_groups_are_disjoint():
    s = participant_split(PARTICIPANTS, seed=0)
    assert set(s.train) & set(s.val) == set()
    assert set(s.train) & set(s.test) == set()
    assert set(s.val) & set(s.test) == set()
    assert sorted(s.all_participants, key=lambda p: int(p[1:])) == PARTICIPANTS


def test_overlapping_split_is_rejected_at_construction():
    """The leakage guard must fire even if a split is hand-written."""
    with pytest.raises(ValueError, match="leaks identity"):
        Split(name="bad", train=["P1", "P2"], val=["P2"], test=["P3"], seed=0)


def test_split_is_deterministic_for_a_seed():
    a = participant_split(PARTICIPANTS, seed=7)
    b = participant_split(PARTICIPANTS, seed=7)
    assert (a.train, a.val, a.test) == (b.train, b.val, b.test)


def test_different_seeds_can_differ():
    a = participant_split(PARTICIPANTS, seed=0)
    b = participant_split(PARTICIPANTS, seed=99)
    assert (a.train, a.val, a.test) != (b.train, b.val, b.test)


def test_split_round_trips_through_json(tmp_path):
    s = participant_split(PARTICIPANTS, seed=1)
    loaded = Split.load(s.save(tmp_path / "s.json"))
    assert (loaded.train, loaded.val, loaded.test, loaded.seed) == (
        s.train, s.val, s.test, s.seed)


def test_too_few_participants_rejected():
    with pytest.raises(ValueError):
        participant_split(["P1", "P2"])


def test_loso_holds_out_each_participant_exactly_once():
    folds = leave_one_subject_out(PARTICIPANTS)
    assert len(folds) == len(PARTICIPANTS)
    assert sorted(f.test[0] for f in folds) == sorted(PARTICIPANTS)
    for f in folds:
        assert f.test[0] not in f.train and f.test[0] not in f.val


def test_saved_project_split_is_participant_disjoint():
    """Guards the split actually committed to data/splits/."""
    path = REPO / "data" / "splits" / "participant_holdout.json"
    if not path.exists():
        pytest.skip("project split not generated yet")
    s = Split.load(path)          # construction re-runs the leakage check
    assert len(s.test) > 0 and len(s.train) > len(s.test)
