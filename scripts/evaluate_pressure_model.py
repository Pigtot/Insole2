#!/usr/bin/env python
"""Evaluate and plot saved baseline predictions.

    envs/core/bin/python scripts/evaluate_pressure_model.py

Reads experiments/pressure_baseline/outputs/baseline_predictions.npz and writes
a figure comparing predictions against measured pressure. Kept separate from
training so plots can be regenerated without refitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visole.evaluation.metrics import regression_metrics  # noqa: E402

OUT = REPO / "experiments" / "pressure_baseline" / "outputs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default=str(OUT / "baseline_predictions.npz"))
    ap.add_argument("--report", default=str(OUT / "baseline_report.json"))
    args = ap.parse_args()

    z = np.load(args.predictions, allow_pickle=True)
    y, p, b = z["y_true"], z["y_pred"], z["y_base"]
    participant, condition = z["participant"], z["condition"]
    report = json.loads(Path(args.report).read_text())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    # 1. total load over a slice of test samples
    ax = axes[0, 0]
    n = min(600, len(y))
    ax.plot(y[:n].sum(1), lw=1.4, label="measured", color="k")
    ax.plot(p[:n].sum(1), lw=1.2, label="Baseline 1 (kinematics)", color="tab:red")
    ax.plot(b[:n].sum(1), lw=1.0, ls="--", label="Baseline 0 (mean)", color="tab:blue")
    ax.set_title("Total load over test samples", fontsize=10)
    ax.set_xlabel("test sample"); ax.set_ylabel("counts")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2. predicted vs measured, total load
    ax = axes[0, 1]
    ax.scatter(y.sum(1), p.sum(1), s=3, alpha=0.25, color="tab:red")
    lo, hi = float(y.sum(1).min()), float(y.sum(1).max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    r = report["baseline_1_kinematics_ridge"]["total_load_r"]
    ax.set_title(f"Total load: predicted vs measured (r = {r:.3f})", fontsize=10)
    ax.set_xlabel("measured"); ax.set_ylabel("predicted"); ax.grid(alpha=0.3)

    # 3. per-channel MAE
    ax = axes[0, 2]
    mae_b1 = np.abs(p - y).mean(0)
    mae_b0 = np.abs(b - y).mean(0)
    idx = np.arange(len(mae_b1))
    ax.bar(idx - 0.2, mae_b0, width=0.4, label="Baseline 0", color="tab:blue")
    ax.bar(idx + 0.2, mae_b1, width=0.4, label="Baseline 1", color="tab:red")
    ax.axvline(31.5, color="k", lw=0.8)
    ax.text(15, ax.get_ylim()[1] * 0.95, "left foot", ha="center", fontsize=8)
    ax.text(47, ax.get_ylim()[1] * 0.95, "right foot", ha="center", fontsize=8)
    ax.set_title("Per-channel MAE (counts)", fontsize=10)
    ax.set_xlabel("channel (right re-indexed to left numbering)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # 4. skill by participant
    ax = axes[1, 0]
    bp = report["baseline_1_by_participant"]
    names = sorted(bp)
    vals = [bp[k]["skill_vs_baseline"] for k in names]
    ax.bar(names, vals, color=["tab:green" if v > 0 else "tab:red" for v in vals])
    ax.axhline(0, color="k", lw=1)
    ax.set_title("Skill vs mean predictor, by test participant\n"
                 "(0 = no better than the average; negative = worse)", fontsize=10)
    ax.set_ylabel("skill"); ax.grid(alpha=0.3, axis="y")

    # 5. skill by condition
    ax = axes[1, 1]
    bc = report["baseline_1_by_condition"]
    names = sorted(bc)
    vals = [bc[k]["skill_vs_baseline"] for k in names]
    ax.bar(names, vals, color=["tab:green" if v > 0 else "tab:red" for v in vals])
    ax.axhline(0, color="k", lw=1)
    ax.set_title("Skill vs mean predictor, by walking condition", fontsize=10)
    ax.set_ylabel("skill"); ax.grid(alpha=0.3, axis="y")

    # 6. contact task
    ax = axes[1, 2]
    ct = report.get("contact_task", {})
    if ct:
        sides = sorted(ct)
        x = np.arange(len(sides))
        ax.bar(x - 0.2, [ct[s]["balanced_accuracy"] for s in sides], width=0.4,
               label="model (balanced acc.)", color="tab:green")
        ax.bar(x + 0.2, [ct[s]["majority_class_accuracy"] for s in sides], width=0.4,
               label="majority class", color="0.6")
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(sides)
        ax.set_ylim(0, 1)
        ax.set_title("Contact task: is the foot on the ground?\n"
                     "(labels from GAITRite, coverage-masked)", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    else:
        ax.axis("off")

    fig.suptitle("Milestone 3 — video → plantar loading, participant-disjoint test set "
                 "(units: baseline-corrected counts, NOT kPa)", fontsize=12)
    path = OUT / "baseline_evaluation.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)

    m = regression_metrics(y, p)
    print(f"test samples {len(y)}, participants {sorted(set(participant.tolist()))}")
    print(f"pooled MAE {m['mae']:.1f}  total-load r {m['total_load_r']:.3f}")
    print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
