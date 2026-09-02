#!/usr/bin/env python
"""Bake a worked example so the dashboard shows real output without an upload.

    envs/core/bin/python scripts/make_demo_run.py --clip P9/NP/1

Runs the full pipeline on a clip from the published Insole-GAITRite dataset
(Zenodo 10.5281/zenodo.19662017, CC-BY-4.0) and stores the stage previews and
numbers under experiments/demo_run/ so the site can display them immediately.

That dataset clip is the honest choice of reference video: the loading model was
trained on this camera and setup, so the demo shows the pipeline working *in
distribution*. On arbitrary phone footage the same stages run but the loading
prediction is extrapolating -- which the dashboard says on the stage itself.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.data.insole_gaitrite import InsoleGaitRite  # noqa: E402
from visole.pipeline.live import LivePipeline  # noqa: E402

OUT = REPO / "experiments" / "demo_run"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="P9/NP/1")
    ap.add_argument("--max-frames", type=int, default=44)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    p, c, k = args.clip.split("/")
    clip = InsoleGaitRite().get(p, c, k)
    if clip.video_path is None:
        print(f"no video for {args.clip}")
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print(f"running the pipeline on {clip.key} ...")
    stages = LivePipeline(clip.video_path, OUT, max_frames=args.max_frames,
                          stride=args.stride).run()

    failed = [s.title for s in stages if s.status == "failed"]
    for s in stages:
        size = ""
        if s.preview:
            kb = (OUT / s.preview).stat().st_size / 1024
            size = f"  {s.preview_w}x{s.preview_h}  {kb:.0f} KB"
        print(f"  {s.title:<26}{s.status:<9}{(s.seconds or 0):>5.1f}s{size}")

    # The STL is large and regenerable; the site does not display it.
    stl = OUT / "insole.stl"
    stl_mb = stl.stat().st_size / 1e6 if stl.exists() else 0.0
    if stl.exists():
        stl.unlink()

    meta = {
        "source": "Insole-GAITRite dataset, Zenodo 10.5281/zenodo.19662017 (CC-BY-4.0)",
        "clip": clip.key,
        "video": f"{clip.video_info.width}x{clip.video_info.height}, "
                 f"{clip.video_info.fps:.1f} fps, {clip.video_info.duration_s:.1f} s",
        "in_distribution": True,
        "note": ("A dataset clip, so the loading model is operating on the camera and "
                 "setup it was trained for. On other footage the same stages run but "
                 "the loading stage is extrapolating."),
        "stl_mb_generated_then_discarded": round(stl_mb, 1),
        "stages": [s.as_dict() for s in stages],
    }
    (OUT / "demo.json").write_text(json.dumps(meta, indent=2))

    total_kb = sum(f.stat().st_size for f in OUT.glob("*.jpg")) / 1024
    total_kb += sum(f.stat().st_size for f in OUT.glob("*.png")) / 1024
    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'all stages completed'}")
    print(f"previews total {total_kb:.0f} KB -> {OUT.relative_to(REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
