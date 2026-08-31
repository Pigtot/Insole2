"""Metrics for plantar-loading prediction.

Design rules
------------
* No single headline number. A model can look good on total load while being
  useless per sensor, so both are always reported.
* Everything is reported **per group** (participant, condition, foot) as well as
  pooled, because a pooled mean hides which subjects fail.
* Skill is measured against the mean-predictor baseline, not against zero. An
  R^2-style skill score below 0 means "worse than predicting the average", which
  is the outcome that actually matters early on.
"""

from __future__ import annotations

import numpy as np

__all__ = ["regression_metrics", "skill_score", "contact_metrics",
           "center_of_pressure", "cop_error", "grouped_report"]


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.ndim != 2:
        raise ValueError(f"expected (n_samples, n_channels), got {y_true.shape}")
    if y_true.size == 0:
        raise ValueError("empty arrays")
    return y_true, y_pred


def _mean_ignoring_nan(values) -> float:
    """Mean of the defined entries, or NaN if none are defined.

    A constant predictor makes every per-channel correlation undefined; numpy's
    nanmean warns and returns NaN there. We return NaN deliberately and quietly
    -- "no correlation is defined" is the correct answer, not 0.0.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")


def regression_metrics(y_true, y_pred) -> dict:
    """MAE/RMSE overall, per channel, and on the summed total load."""
    y_true, y_pred = _check(y_true, y_pred)
    err = y_pred - y_true
    per_channel_mae = np.abs(err).mean(axis=0)
    per_channel_rmse = np.sqrt((err ** 2).mean(axis=0))

    total_true = y_true.sum(axis=1)
    total_pred = y_pred.sum(axis=1)
    total_err = total_pred - total_true

    denom = float(np.ptp(y_true)) or 1.0
    return {
        "n_samples": int(y_true.shape[0]),
        "n_channels": int(y_true.shape[1]),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "normalised_mae": float(np.abs(err).mean() / denom),
        "per_channel_mae_min": float(per_channel_mae.min()),
        "per_channel_mae_median": float(np.median(per_channel_mae)),
        "per_channel_mae_max": float(per_channel_mae.max()),
        "total_load_mae": float(np.abs(total_err).mean()),
        "total_load_rmse": float(np.sqrt((total_err ** 2).mean())),
        "total_load_r": _safe_corr(total_true, total_pred),
        "mean_channel_r": _mean_ignoring_nan([
            _safe_corr(y_true[:, c], y_pred[:, c]) for c in range(y_true.shape[1])
        ]),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, or NaN when a series is constant (r is undefined, not 0)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def skill_score(y_true, y_pred, y_baseline) -> dict:
    """Fraction of the baseline's squared error that the model removes.

    1.0 is perfect, 0.0 is exactly as good as the baseline, and **negative means
    worse than the baseline** -- the number that stops a bad model being sold as
    a good one.
    """
    y_true, y_pred = _check(y_true, y_pred)
    _, y_baseline = _check(y_true, y_baseline)
    sse_model = float(((y_pred - y_true) ** 2).sum())
    sse_base = float(((y_baseline - y_true) ** 2).sum())
    total_model = float(((y_pred.sum(1) - y_true.sum(1)) ** 2).sum())
    total_base = float(((y_baseline.sum(1) - y_true.sum(1)) ** 2).sum())
    return {
        "skill_vs_baseline": 1.0 - sse_model / sse_base if sse_base > 0 else float("nan"),
        "skill_total_load": 1.0 - total_model / total_base if total_base > 0 else float("nan"),
        "beats_baseline": bool(sse_model < sse_base),
    }


def contact_metrics(true_contact, pred_contact) -> dict:
    """Binary stance/swing agreement.

    Balanced accuracy is reported alongside accuracy: stance occupies roughly
    half of a walking clip, but for other conditions a constant predictor can
    score high plain accuracy.
    """
    t = np.asarray(true_contact).astype(bool).ravel()
    p = np.asarray(pred_contact).astype(bool).ravel()
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    tp = int((t & p).sum()); tn = int((~t & ~p).sum())
    fp = int((~t & p).sum()); fn = int((t & ~p).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    return {
        "accuracy": (tp + tn) / t.size,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": (2 * prec * sens / (prec + sens)) if prec and sens and (prec + sens) else float("nan"),
        "positive_rate_true": float(t.mean()),
        "positive_rate_pred": float(p.mean()),
    }


def center_of_pressure(values, sensor_xy) -> np.ndarray:
    """Pressure-weighted centroid, (n_samples, 2). NaN where nothing is loaded."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    xy = np.asarray(sensor_xy, dtype=float)
    if values.shape[1] != xy.shape[0]:
        raise ValueError(f"{values.shape[1]} channels vs {xy.shape[0]} sensor positions")
    w = np.clip(values, 0, None)
    total = w.sum(axis=1)
    out = np.full((values.shape[0], 2), np.nan)
    ok = total > 0
    out[ok] = (w[ok] @ xy) / total[ok, None]
    return out


def cop_error(y_true, y_pred, sensor_xy) -> dict:
    """Distance between true and predicted centre of pressure, loaded frames only."""
    a = center_of_pressure(y_true, sensor_xy)
    b = center_of_pressure(y_pred, sensor_xy)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if not ok.any():
        return {"cop_mae_units": float("nan"), "n_valid": 0}
    d = np.linalg.norm(a[ok] - b[ok], axis=1)
    return {
        "cop_mae_units": float(d.mean()),
        "cop_median_units": float(np.median(d)),
        "cop_p90_units": float(np.percentile(d, 90)),
        "n_valid": int(ok.sum()),
    }


def grouped_report(y_true, y_pred, groups, y_baseline=None) -> dict:
    """Metrics per group label, plus pooled. Groups is a per-sample label array."""
    y_true, y_pred = _check(y_true, y_pred)
    groups = np.asarray(groups)
    if groups.shape[0] != y_true.shape[0]:
        raise ValueError("groups must have one label per sample")
    report = {"pooled": regression_metrics(y_true, y_pred), "by_group": {}}
    if y_baseline is not None:
        report["pooled"].update(skill_score(y_true, y_pred, y_baseline))
    for g in sorted(set(groups.tolist()), key=str):
        m = groups == g
        if m.sum() < 2:
            continue
        entry = regression_metrics(y_true[m], y_pred[m])
        if y_baseline is not None:
            entry.update(skill_score(y_true[m], y_pred[m], np.asarray(y_baseline)[m]))
        report["by_group"][str(g)] = entry
    return report
