#!/usr/bin/env python
"""PoC 1 -- one real 32-channel sample -> ordered pressure bins, with sensor map.

    envs/core/bin/python scripts/poc_pressure_bins.py --clip P1/FP/1

Deliberately small: it proves the representation round-trips on real data and
that the sensor geometry we recovered places the channels sensibly. It makes no
claim about predicting pressure.
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
from matplotlib import colors as mcolors  # noqa: E402

from visole.data.insole_gaitrite import InsoleGaitRite  # noqa: E402
from visole.pressure.bins import PressureBins  # noqa: E402
from visole.pressure.sensor_map import load_sensor_map  # noqa: E402
from visole.visualization.pressure_plots import scatter_sensors  # noqa: E402

OUT = REPO / "experiments" / "pressure_baseline" / "outputs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", default="P1/FP/1")
    ap.add_argument("--n-bins", type=int, default=9)
    args = ap.parse_args()

    p, c, k = args.clip.split("/")
    clip = InsoleGaitRite().get(p, c, k)
    bins = PressureBins.log_spaced(n_bins=args.n_bins)

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"clip": clip.key, "n_bins": bins.n_bins,
                    "units": bins.units, "edges": bins.edges.tolist(), "sides": {}}

    fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
    cmap = plt.get_cmap("viridis", bins.n_bins)
    norm = mcolors.BoundaryNorm(np.arange(bins.n_bins + 1) - 0.5, bins.n_bins)

    for row, side in enumerate(("left", "right")):
        arr = clip.pressure(side, baseline_correct=True)
        smap = load_sensor_map(side)
        peak_i = int(arr.sum(1).argmax())          # most loaded instant
        sample = arr[peak_i]
        idx = bins.encode(sample)
        qe = bins.quantisation_error(arr)
        report["sides"][side] = {
            "n_samples": int(arr.shape[0]),
            "peak_frame": peak_i,
            "peak_total_counts": float(arr[peak_i].sum()),
            "bin_histogram_whole_clip": np.bincount(
                bins.encode(arr).ravel(), minlength=bins.n_bins).tolist(),
            "quantisation_error_counts": qe,
            "sample_values": sample.round(1).tolist(),
            "sample_bins": idx.tolist(),
        }

        ax = axes[row, 0]
        sc = scatter_sensors(ax, smap, sample, vmax=float(arr.max()), size=170)
        ax.set_title(f"{side.upper()} raw counts at peak load\n"
                     f"(frame {peak_i}, 32 measured channels)", fontsize=9)
        fig.colorbar(sc, ax=ax, fraction=0.05, label="baseline-corrected counts")

        ax = axes[row, 1]
        for pad in smap.pads:
            ax.plot(pad[:, 0], pad[:, 1], lw=0.4, color="0.8", zorder=1)
        for chain in smap.outline:
            ax.plot(chain[:, 0], chain[:, 1], lw=0.8, color="0.45", zorder=1)
        sc2 = ax.scatter(smap.xy[:, 0], smap.xy[:, 1], c=idx, s=170, cmap=cmap,
                         norm=norm, zorder=3, edgecolors="k", linewidths=0.4)
        for (x, y), b in zip(smap.xy, idx):
            ax.text(x, y, str(int(b)), fontsize=6, ha="center", va="center",
                    color="w", zorder=4)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(f"{side.upper()} ordered bin index (0..{bins.n_bins - 1})", fontsize=9)
        fig.colorbar(sc2, ax=ax, fraction=0.05, ticks=range(bins.n_bins), label="bin")

        ax = axes[row, 2]
        hist = report["sides"][side]["bin_histogram_whole_clip"]
        ax.bar(range(bins.n_bins), hist, color=[cmap(i) for i in range(bins.n_bins)],
               edgecolor="k", linewidth=0.4)
        ax.set_xlabel("bin"); ax.set_ylabel("count (all samples x 32 sensors)")
        ax.set_title(f"{side.upper()} bin occupancy over the clip\n"
                     f"round-trip MAE {qe['mae']:.1f} counts "
                     f"({100 * qe['normalised_mae']:.2f}% of range)", fontsize=9)
        ax.set_xticks(range(bins.n_bins)); ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"PoC 1 - {clip.key}: 32 measured channels as {bins.n_bins} ordered bins "
                 f"(units: baseline-corrected sensor counts, NOT kPa)", fontsize=11)
    fig_path = OUT / f"poc_bins_{p}_{c}_{k}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    (OUT / f"poc_bins_{p}_{c}_{k}.json").write_text(json.dumps(report, indent=2))
    print(bins.describe())
    for side, r in report["sides"].items():
        q = r["quantisation_error_counts"]
        print(f"\n{side}: peak frame {r['peak_frame']}, total {r['peak_total_counts']:.0f} counts")
        print(f"  bin histogram: {r['bin_histogram_whole_clip']}")
        print(f"  binning round-trip MAE {q['mae']:.2f} counts "
              f"(median {q['median_ae']:.2f}, p95 {q['p95_ae']:.2f})")
    print(f"\nwrote {fig_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
