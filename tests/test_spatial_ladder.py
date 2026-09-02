"""Tests for the spatial-resolution ladder metric.

These are correctness gates, not smoke tests. The whole claim of
`score_contrast` is that it separates two models that a per-sensor skill score
cannot tell apart:

  * a model that knows *where* load sits, and
  * a model that only knows *how much* total load there is.

If the share metric does not separate them on synthetic data where the answer
is known by construction, then it cannot be trusted on real predictions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from visole.pressure.sensor_map import load_sensor_map  # noqa: E402
from evaluate_spatial_ladder import (  # noqa: E402
    NULL_PREFIX, build_contrasts, score_contrast,
)

RNG = np.random.default_rng(0)
N = 400


def _synthetic(n=N):
    """Targets whose heel/forefoot split varies independently of total load.

    Total load and the heel share are drawn separately, so a model that only
    tracks total load has no way to recover the share -- which is exactly the
    property the metric must detect.
    """
    total = RNG.uniform(500, 3000, n)
    heel_share = RNG.uniform(0.2, 0.8, n)
    y = np.zeros((n, 64))
    y[:, :20] = (total * heel_share)[:, None] / 20        # "heel" channels
    y[:, 20:] = (total * (1 - heel_share))[:, None] / 44  # "forefoot" channels
    return y, total, heel_share


A, B = np.arange(20), np.arange(20, 64)


def test_perfect_model_scores_one_on_both():
    y, _, _ = _synthetic()
    s = score_contrast(y, y.copy(), y, A, B)
    assert s["skill_absolute"] == pytest.approx(1.0, abs=1e-9)
    assert s["skill_share"] == pytest.approx(1.0, abs=1e-9)


def test_total_load_only_model_scores_zero_on_share():
    """The central gate.

    This model knows total load exactly but always distributes it in the mean
    pattern. It must look good on the absolute metric and worthless on share --
    if it does not, the metric cannot support any claim about spatial
    resolution.
    """
    y, total, _ = _synthetic()
    mean_pattern = y.mean(0) / y.mean(0).sum()
    pred = total[:, None] * mean_pattern[None, :]

    s = score_contrast(y, pred, y, A, B)
    assert s["skill_absolute"] > 0.3, "should track the absolute heel load well"
    assert abs(s["skill_share"]) < 0.05, "must not appear to resolve the split"
    # and it gives itself away by predicting a nearly constant share
    assert s["share_pred_sd"] < 1e-6 < s["share_true_sd"]


def test_share_is_negative_when_the_split_is_inverted():
    """A model that gets the anatomy backwards must score worse than nothing."""
    y, _, _ = _synthetic()
    pred = y.copy()
    pred[:, :20], pred[:, 20:] = y[:, 20:].sum(1)[:, None] / 20, y[:, :20].sum(1)[:, None] / 44
    assert score_contrast(y, pred, y, A, B)["skill_share"] < 0.0


def test_low_load_frames_are_excluded_from_the_share():
    """Near-airborne frames have a noise-dominated ratio and must be dropped."""
    y, _, _ = _synthetic()
    y[:100] *= 1e-4  # both feet effectively off the ground
    assert score_contrast(y, y.copy(), y, A, B)["n_frames_share"] == len(y) - 100


def test_contrasts_are_disjoint_and_anatomically_paired():
    contrasts = build_contrasts(seed=0, null_repeats=3)
    midfoot = set(load_sensor_map("left").region_indices()["midfoot"])
    assert midfoot, "region definitions changed"

    for name, (a, b, _) in contrasts.items():
        assert not set(a.tolist()) & set(b.tolist()), f"{name}: sides overlap"
        covered = set((a % 32).tolist()) | set((b % 32).tolist())
        # heel-vs-forefoot is deliberately a *partial* partition: the midfoot is
        # neither, and lumping it into either side would blur the contrast.
        expected = set(range(32)) - (midfoot if name == "heel_forefoot" else set())
        assert covered == expected, f"{name}: unexpected sensor coverage"

    # Anatomical contrasts take the same sensors from both feet; the right half
    # is already re-indexed into left numbering, so `i` and `i + 32` are
    # homologous sites and must never be separated.
    for name in ("heel_forefoot", "medial_lateral"):
        a, b, _ = contrasts[name]
        for side in (a, b):
            lo = set(side[side < 32].tolist())
            hi = set((side[side >= 32] - 32).tolist())
            assert lo == hi, f"{name}: feet are not treated symmetrically"


def test_null_controls_differ_from_each_other():
    """Pooling identical partitions would be a control in name only."""
    contrasts = build_contrasts(seed=0, null_repeats=5)
    nulls = [tuple(contrasts[n][0].tolist())
             for n in contrasts if n.startswith(NULL_PREFIX)]
    assert len(nulls) == 5
    assert len(set(nulls)) == 5
