#!/usr/bin/env python
"""Cache 2D body-pose keypoints for walking clips (feeds Baseline 1b).

    envs/core/bin/python scripts/extract_pose.py --clips-per-cell 3 --stride 3

Resumable: clips already cached are skipped. Raw keypoints are cached rather
than derived features, so the feature definition can change without re-running
the network.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# macOS framework Python has no CA bundle; torchvision's weight download needs one.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:  # pragma: no cover
    pass

from visole.compute import device as vdev  # noqa: E402
from visole.data.insole_gaitrite import InsoleGaitRite, WALKING_CONDITIONS  # noqa: E402
from visole.models.pose_features import extract_clip_pose, load_pose_model  # noqa: E402

CACHE = REPO / "data" / "processed" / "pose"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-per-cell", type=int, default=3,
                    help="clips per (participant, condition)")
    ap.add_argument("--stride", type=int, default=3, help="video frame stride")
    ap.add_argument("--min-size", type=int, default=480)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ds = InsoleGaitRite()
    clips, per_cell = [], {}
    for clip in ds.filter(conditions=WALKING_CONDITIONS, require_all_modalities=True):
        key = (clip.participant, clip.condition)
        if per_cell.get(key, 0) >= args.clips_per_cell:
            continue
        per_cell[key] = per_cell.get(key, 0) + 1
        clips.append(clip)
    if args.max_clips:
        clips = clips[:args.max_clips]

    todo = [c for c in clips
            if not (CACHE / f"{c.participant}_{c.condition}_{c.clip_id}.npz").exists()]
    print(f"{len(clips)} clips selected, {len(todo)} still to process")
    if args.dry_run:
        return 0
    if not todo:
        print("nothing to do")
        return 0

    device = vdev.resolve_device(args.device)
    print(f"device: {device}")
    model = load_pose_model(device, min_size=args.min_size)
    CACHE.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    frames_done = 0
    with vdev.MPSFallbackRecorder() as rec:
        for i, clip in enumerate(todo, 1):
            seq = extract_clip_pose(clip, model, device, stride=args.stride)
            if seq is None:
                print(f"  skip {clip.key} (no video)")
                continue
            np.savez_compressed(
                CACHE / f"{clip.participant}_{clip.condition}_{clip.clip_id}.npz",
                times=seq.times, keypoints=seq.keypoints,
                scores=seq.scores, detected=seq.detected)
            frames_done += len(seq.times)
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                rate = frames_done / max(el, 1e-9)
                eta = (len(todo) - i) * (el / i)
                print(f"  {i}/{len(todo)} clips, {frames_done} frames, "
                      f"{rate:.1f} fps, ETA {eta / 60:.1f} min", flush=True)
    if rec.had_fallback:
        print("MPS fallbacks observed (logged, not hidden):")
        for op in rec.operators[:10]:
            print("   ", op[:150])
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {CACHE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
