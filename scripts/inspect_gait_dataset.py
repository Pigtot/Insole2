#!/usr/bin/env python
"""Milestone 2 -- audit the Insole-GAITRite dataset and produce evidence.

    envs/core/bin/python scripts/inspect_gait_dataset.py --max-clips 60
    envs/core/bin/python scripts/inspect_gait_dataset.py --clip P1/FP/1

Writes JSON statistics and figures under experiments/gait_dataset_audit/outputs/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visole.data.insole_gaitrite import (  # noqa: E402
    NOMINAL_INSOLE_HZ, PRESSURE_COLUMNS, InsoleGaitRite, estimate_baseline,
)
from visole.pressure.sensor_map import load_sensor_map  # noqa: E402
from visole.visualization.pressure_plots import (  # noqa: E402
    interpolated_display_image, scatter_sensors,
)

OUT = REPO / "experiments" / "gait_dataset_audit" / "outputs"


def survey(ds: InsoleGaitRite, max_clips: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    clips = list(ds.clips)
    sample = rng.sample(clips, min(max_clips, len(clips)))

    fps_counter: Counter = Counter()
    res_counter: Counter = Counter()
    ratios, durations, n_missing = [], [], Counter()
    vmin, vmax = np.inf, -np.inf
    baselines, sump_resid, nan_total, row_total = [], [], 0, 0

    for c in sample:
        if not c.left_csv.exists() or not c.right_csv.exists():
            n_missing["insole_csv"] += 1
            continue
        df = c.insole("left")
        arr = df[PRESSURE_COLUMNS].to_numpy(float)
        row_total += len(arr)
        nan_total += int(np.isnan(arr).sum())
        vmin, vmax = min(vmin, arr.min()), max(vmax, arr.max())
        baselines.append(float(np.percentile(arr, 5)))
        if "sumP" in df:
            sump_resid.append(float(np.abs(arr.sum(1) - df["sumP"].to_numpy(float)).max()))

        vi = c.video_info
        if vi is None:
            n_missing["video"] += 1
            continue
        fps_counter[round(vi.fps, 2)] += 1
        res_counter[f"{vi.width}x{vi.height}"] += 1
        durations.append(vi.duration_s)
        if vi.duration_s > 0:
            ratios.append((len(arr) / NOMINAL_INSOLE_HZ) / vi.duration_s)

    return {
        "n_clips_total": len(clips),
        "n_clips_sampled": len(sample),
        "participants": ds.participants,
        "n_participants": len(ds.participants),
        "absent_ids_in_P1_P24": [f"P{i}" for i in range(1, 25)
                                 if f"P{i}" not in set(ds.participants)],
        "clips_by_condition": ds.counts_by_condition(),
        "video_fps_values": dict(sorted(fps_counter.items())),
        "video_resolutions": dict(res_counter),
        "video_duration_s": _stats(durations),
        "insole_over_video_duration_ratio": _stats(ratios),
        "pressure_value_min": None if vmin is np.inf else float(vmin),
        "pressure_value_max": None if vmax is -np.inf else float(vmax),
        "unloaded_baseline_p5": _stats(baselines),
        "max_abs_sumP_minus_channel_sum": max(sump_resid) if sump_resid else None,
        "nan_count": nan_total,
        "rows_checked": row_total,
        "missing_modalities": dict(n_missing),
    }


def _stats(v) -> dict | None:
    if not len(v):
        return None
    a = np.asarray(v, dtype=float)
    return {"min": float(a.min()), "median": float(np.median(a)),
            "max": float(a.max()), "n": int(a.size)}


def synchronised_figure(clip, out_path: Path, frame_fraction: float = 0.5) -> Path:
    """Video frame + both feet's signals + sensor-map state at the same instant."""
    import cv2

    vi = clip.video_info
    t = frame_fraction * vi.duration_s
    frame = clip.frame_at_time(t)

    left = clip.pressure("left", baseline_correct=True)
    right = clip.pressure("right", baseline_correct=True)
    tl = clip.pressure_time("left")
    tr = clip.pressure_time("right")
    il = int(np.clip(np.searchsorted(tl, t), 0, len(tl) - 1))
    ir = int(np.clip(np.searchsorted(tr, t), 0, len(tr) - 1))
    vmax = float(max(left.max(), right.max()))

    fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.5, 1, 1])

    ax = fig.add_subplot(gs[0, :3])
    if frame is not None:
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{clip.key}   video frame at t = {t:.2f} s "
                 f"({vi.width}x{vi.height}, {vi.fps:.2f} fps)", fontsize=10)
    ax.axis("off")

    axn = fig.add_subplot(gs[0, 3])
    axn.axis("off")
    axn.text(0, 1, "\n".join([
        "Measured: 32 channels/foot",
        f"insole {NOMINAL_INSOLE_HZ:g} Hz (nominal)",
        f"video {vi.fps:.2f} fps (measured)",
        f"n samples L/R: {len(left)}/{len(right)}",
        f"video {vi.duration_s:.2f} s",
        f"insole {len(left)/NOMINAL_INSOLE_HZ:.2f} s",
        "",
        "units: RAW SENSOR COUNTS",
        "baseline-corrected, NOT kPa",
        "participants wore own shoes",
    ]), va="top", ha="left", fontsize=8.5, family="monospace")

    for row, (sig, tt, idx, name) in enumerate(
            [(left, tl, il, "LEFT"), (right, tr, ir, "RIGHT")], start=1):
        axs = fig.add_subplot(gs[row, :2])
        axs.plot(tt, sig.sum(1), lw=1.6, color="tab:blue")
        axs.axvline(t, color="crimson", lw=1.2, ls="--")
        axs.set_ylabel(f"{name}\ntotal load (counts)", fontsize=9)
        axs.set_xlim(tt[0], tt[-1])
        if row == 2:
            axs.set_xlabel("time within clip (s)")
        axs.grid(alpha=0.3)

        axh = fig.add_subplot(gs[row, 2])
        axh.imshow(sig.T, aspect="auto", origin="lower", cmap="inferno",
                   extent=[tt[0], tt[-1], -0.5, 31.5], vmax=vmax)
        axh.axvline(t, color="w", lw=1.0, ls="--")
        axh.set_ylabel("sensor index", fontsize=8)
        axh.set_title(f"{name}: 32 channels", fontsize=8)

        smap = load_sensor_map(name.lower())
        axm = fig.add_subplot(gs[row, 3])
        img = interpolated_display_image(smap, sig[idx])
        x, y = smap.xy[:, 0], smap.xy[:, 1]
        pad = 0.06 * max(np.ptp(x), np.ptp(y))
        axm.imshow(img, origin="lower", cmap="inferno", vmin=0, vmax=vmax, alpha=0.55,
                   extent=[x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad])
        scatter_sensors(axm, smap, sig[idx], vmax=vmax, size=70)
        axm.set_title(f"{name} at t (dots = measured;\nfield = display-only interp.)",
                      fontsize=7)

    fig.suptitle("Insole-GAITRite audit: synchronised video and plantar signals",
                 fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--max-clips", type=int, default=60)
    ap.add_argument("--clip", default="P1/FP/1", help="participant/condition/id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ds = InsoleGaitRite(args.root)
    print(f"indexed {len(ds)} clips, {len(ds.participants)} participants")
    if args.dry_run:
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    stats = survey(ds, args.max_clips)
    (OUT / "audit_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: stats[k] for k in
                      ("n_participants", "clips_by_condition", "video_fps_values",
                       "insole_over_video_duration_ratio", "nan_count")}, indent=2))

    p, c, k = args.clip.split("/")
    fig = synchronised_figure(ds.get(p, c, k), OUT / f"sync_{p}_{c}_{k}.png")
    print(f"wrote {fig.relative_to(REPO)}")
    print(f"wrote {(OUT / 'audit_stats.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
