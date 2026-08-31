"""Tests for evaluation metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.evaluation.metrics import (  # noqa: E402
    center_of_pressure, contact_metrics, cop_error, grouped_report,
    regression_metrics, skill_score,
)

RNG = np.random.default_rng(0)


def test_perfect_prediction_has_zero_error():
    y = RNG.uniform(0, 100, (50, 32))
    m = regression_metrics(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["total_load_mae"] == pytest.approx(0.0)


def test_shape_and_dimension_errors_are_caught():
    with pytest.raises(ValueError, match="shape mismatch"):
        regression_metrics(np.zeros((4, 3)), np.zeros((4, 5)))
    with pytest.raises(ValueError, match="expected"):
        regression_metrics(np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError, match="empty"):
        regression_metrics(np.zeros((0, 3)), np.zeros((0, 3)))


def test_correlation_is_nan_not_zero_for_a_constant_series():
    """r is undefined for a constant series; reporting 0.0 would be a claim."""
    y = RNG.uniform(0, 10, (30, 4))
    const = np.ones_like(y)
    assert np.isnan(regression_metrics(y, const)["total_load_r"])


def test_skill_is_one_for_perfect_zero_for_baseline_negative_for_worse():
    y = RNG.uniform(0, 100, (60, 8))
    base = np.tile(y.mean(0), (60, 1))
    assert skill_score(y, y, base)["skill_vs_baseline"] == pytest.approx(1.0)
    assert skill_score(y, base, base)["skill_vs_baseline"] == pytest.approx(0.0)
    worse = skill_score(y, base + 500, base)
    assert worse["skill_vs_baseline"] < 0
    assert worse["beats_baseline"] is False


def test_total_load_metrics_respond_to_a_pure_offset():
    y = RNG.uniform(0, 10, (20, 5))
    m = regression_metrics(y, y + 1.0)
    assert m["total_load_mae"] == pytest.approx(5.0)


# --- contact --------------------------------------------------------------
def test_contact_perfect_and_inverted():
    t = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    assert contact_metrics(t, t)["accuracy"] == 1.0
    assert contact_metrics(t, ~t)["accuracy"] == 0.0


def test_balanced_accuracy_exposes_a_constant_predictor():
    t = np.array([1] * 90 + [0] * 10, dtype=bool)
    m = contact_metrics(t, np.ones_like(t))
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["balanced_accuracy"] == pytest.approx(0.5)


def test_contact_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        contact_metrics(np.ones(4, bool), np.ones(5, bool))


# --- centre of pressure ---------------------------------------------------
def test_cop_of_a_single_loaded_sensor_is_that_sensor():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    v = np.array([[0.0, 5.0, 0.0]])
    assert np.allclose(center_of_pressure(v, xy)[0], [10.0, 0.0])


def test_cop_of_equal_load_is_the_centroid():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert np.allclose(center_of_pressure(np.ones((1, 2)), xy)[0], [5.0, 0.0])


def test_cop_is_nan_when_unloaded_not_silently_zero():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert np.isnan(center_of_pressure(np.zeros((1, 2)), xy)).all()


def test_cop_channel_count_must_match_sensor_count():
    with pytest.raises(ValueError):
        center_of_pressure(np.ones((2, 5)), np.zeros((3, 2)))


def test_cop_error_ignores_unloaded_frames():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    y = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    e = cop_error(y, y, xy)
    assert e["n_valid"] == 2
    assert e["cop_mae_units"] == pytest.approx(0.0)


# --- grouping -------------------------------------------------------------
def test_grouped_report_splits_by_label():
    y = RNG.uniform(0, 10, (40, 3))
    g = np.array(["A"] * 20 + ["B"] * 20)
    rep = grouped_report(y, y, g)
    assert set(rep["by_group"]) == {"A", "B"}
    assert rep["pooled"]["mae"] == pytest.approx(0.0)


def test_grouped_report_rejects_mismatched_group_length():
    with pytest.raises(ValueError, match="one label per sample"):
        grouped_report(np.zeros((4, 2)), np.zeros((4, 2)), np.array(["a", "b"]))
