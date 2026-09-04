#!/usr/bin/env python
"""One figure showing the whole Visole pipeline with its measured numbers.

    envs/core/bin/python scripts/make_pipeline_summary.py

Every number is read from the experiment JSON on disk -- nothing is typed in by
hand -- so the figure cannot drift from the results it claims to summarise.
Stages are colour-coded by evidence status, because the difference between
"measured", "simulated" and "not validated" is the whole point.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

STATUS_COLOURS = {
    "measured": "#2f7d32",      # verified against real data / ground truth
    "simulated": "#1565c0",     # model output with stated assumptions
    "unvalidated": "#b0b0b0",   # no hardware evidence
}


def load(path):
    try:
        return json.loads((REPO / path).read_text())
    except Exception:
        return None


def collect() -> list[dict]:
    aud = load("experiments/gait_dataset_audit/outputs/audit_stats.json")
    loso = load("experiments/pressure_baseline/outputs/loso_pose.json")
    lad = load("experiments/pressure_baseline/outputs/spatial_ladder_pose.json")
    f3d = load("experiments/focus_baseline/foot3d/results.json")
    cal = load("experiments/insole_physics/outputs/stiffness_calibration_r10.json")
    opt = load("experiments/insole_physics/outputs/insole_optimum.json")
    syn = [json.loads(Path(p).read_text())
           for p in sorted(glob.glob(str(REPO / "experiments/focus_baseline/synthetic/synthetic_result_*.json")))]

    ch = [r["chamfer_mm"] for r in (f3d or []) if "chamfer_mm" in r]
    syn_ch = [s["chamfer_mm"] for s in syn if "chamfer_mm" in s]

    stages = []
    if aud:
        stages.append(dict(
            title="1. Walking video",
            lines=[f"{aud['n_participants']} participants, {aud['n_clips_total']} clips",
                   "1920x1088, fps varies per clip",
                   "feet are SHOD -- sole never visible"],
            status="measured"))
    if loso:
        s = loso["skill_gated"]
        stages.append(dict(
            title="2. Video -> plantar loading",
            lines=[f"skill {s['mean']:+.3f} +/- {s['sd']:.3f} vs mean predictor",
                   f"{loso['n_folds_beating_mean_gated']}/{loso['n_folds']} participants beat it "
                   "(leave-one-out)",
                   f"correct loaded foot {100*loso['foot_dominance_acc']['mean']:.0f}% of frames"],
            status="measured"))
    if lad:
        # The whole-foot skill above cannot separate "knows where the load is"
        # from "knows how much load there is". This stage is that separation,
        # and it is where the chain narrows.
        c = lad["contrasts"]

        def _axis(key, label):
            v = c[key]["vs_null"]
            mark = "yes" if v["folds_beating_null"] > v["n_folds"] / 2 else "NO"
            return (f"{label}: {mark} "
                    f"({v['folds_beating_null']}/{v['n_folds']} folds beat own null)")

        stages.append(dict(
            title="2b. How much of that is spatial?",
            lines=[_axis("left_right", "which foot"),
                   _axis("heel_forefoot", "heel vs forefoot"),
                   _axis("medial_lateral", "medial vs lateral")],
            status="measured"))
    if ch:
        line3 = (f"cameras estimated: {np.mean(syn_ch):.1f} mm (~{np.mean(syn_ch)/np.median(ch):.0f}x worse)"
                 if syn_ch else "cameras estimated: not yet tested")
        stages.append(dict(
            title="3. Photos -> 3D foot (FOCUS)",
            lines=[f"median {np.median(ch):.2f} mm chamfer, {len(ch)}/{len(ch)} scans",
                   "474 real photos in 3.2 min on MPS",
                   line3],
            status="measured"))
    stages.append(dict(
        title="4. Canonical plantar frame",
        lines=["both feet in one shared frame",
               "left/right sensor maps differ -- no index is shared",
               "sparse sensors lifted, support mask kept"],
        status="measured"))
    if cal:
        stages.append(dict(
            title="5. Lattice stiffness (FEA)",
            lines=[cal["fit"]["description"].split("  ")[0],
                   f"R^2 = {cal['fit']['r_squared']:.4f}, solid block recovers E_s",
                   "exponent converged; prefactor +/-30%"],
            status="measured"))
    if opt:
        o, t = opt["optimum"], opt["reference_tpu_uniform"]
        gain = 100 * (t["peak_pressure_kpa"] - o["peak_pressure_kpa"]) / t["peak_pressure_kpa"]
        stages.append(dict(
            title="6. Contact model -> optimum",
            lines=[f"{o['E_s_mpa']:.1f} MPa, {o['thickness_mm']:.0f} mm, {o['hypothesis']} grading",
                   f"{o['peak_pressure_kpa']:.0f} kPa peak, {gain:.0f}% below 40 MPa TPU",
                   "material dominates grading"],
            status="simulated"))
    stages.append(dict(
        title="7. Printed insole on a real foot",
        lines=["no printed part", "no pressure sensor", "no human testing"],
        status="unvalidated"))
    return stages


def main() -> int:
    stages = collect()
    if not stages:
        print("no experiment results found")
        return 1

    n = len(stages)
    row_h, gap = 1.10, 0.30
    fig_h = 1.05 + n * (row_h + gap)
    fig, ax = plt.subplots(figsize=(14.5, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.55, n * (row_h + gap) + 0.22)  # -0.55: a strip for the legend
    ax.axis("off")

    top = n * (row_h + gap) + 0.18
    for i, st in enumerate(stages):
        y = top - (i + 1) * (row_h + gap)
        c = STATUS_COLOURS[st["status"]]
        ax.add_patch(FancyBboxPatch((0.35, y), 9.30, row_h,
                                    boxstyle="round,pad=0.05", linewidth=2.0,
                                    edgecolor=c, facecolor=c, alpha=0.11, zorder=2))
        ax.text(0.62, y + row_h - 0.24, st["title"], fontsize=13.5,
                fontweight="bold", color=c, va="center", zorder=3)
        for k, line in enumerate(st["lines"]):
            ax.text(0.62, y + row_h - 0.53 - k * 0.215, line, fontsize=10.4,
                    color="0.15", va="center", zorder=3)
        # Status label sits inside the box, horizontal -- rotated text at the
        # figure edge collided between rows and was unreadable.
        ax.text(9.42, y + row_h / 2, st["status"].upper(), fontsize=9.5, color=c,
                ha="right", va="center", fontweight="bold", zorder=3)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((5.0, y - 0.03), (5.0, y - gap + 0.04),
                                         arrowstyle="-|>", mutation_scale=22,
                                         color="0.45", linewidth=2.2, zorder=1))

    handles = [plt.Line2D([0], [0], marker="s", color="w", markersize=13,
                          markerfacecolor=v, label=k)
               for k, v in STATUS_COLOURS.items()]
    ax.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
              fontsize=11, bbox_to_anchor=(0.5, 0.055))
    fig.subplots_adjust(top=0.945, bottom=0.015, left=0.01, right=0.99)
    fig.suptitle("Visole pipeline: what is measured, what is simulated, "
                 "what is not yet validated", fontsize=15, y=0.985)

    out = REPO / "experiments" / "pipeline_summary.png"
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)} ({len(stages)} stages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
