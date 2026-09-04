#!/usr/bin/env python
"""Bake a worked example so the dashboard shows real output without an upload.

    envs/core/bin/python scripts/make_demo_run.py --clip P9/NP/1
    envs/core/bin/python scripts/make_demo_run.py --video data/raw/sample_videos/treadmill_walk.mp4

Runs the full pipeline and stores the stage previews and numbers under
experiments/ so the site and the README can display them immediately.

Two reference videos, and the difference between them is the point:

--clip  a clip from the published Insole-GAITRite dataset (Zenodo
        10.5281/zenodo.19662017, CC-BY-4.0). The loading model was trained on
        this camera and setup, so this shows the pipeline working *in
        distribution*, against footage that has measured pressure beside it.

--video any file. `scripts/download_data.py sample_videos` fetches a CC-BY
        treadmill clip from Wikimedia Commons for exactly this purpose. The same
        stages run, but the loading stage is extrapolating -- a different person,
        a different camera, a different viewpoint. That is recorded as
        in_distribution=false in demo.json, and the dashboard and README say so
        on the stage itself rather than in a footnote.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.data import registry  # noqa: E402
from visole.data.insole_gaitrite import InsoleGaitRite, probe_video  # noqa: E402
from visole.pipeline.live import LivePipeline  # noqa: E402

OOD_NOTE = (
    "Video the loading model was never trained on. Pose detection, the canonical "
    "frame, the density mapping, the contact solve and the geometry are honest on "
    "any footage -- they are deterministic given their input. The loading stage is "
    "extrapolating: treat its output as illustrative of the pipeline, not as a "
    "measurement of the person in the video."
)


def _source_credit(video: Path) -> tuple[str, str | None]:
    """Attribution for a video, taken from the registry when it declares one.

    CC-BY obliges us to credit the clip wherever its output is shown, so the
    credit travels in demo.json rather than being retyped into the README.
    """
    spec = registry.REGISTRY.get("sample_videos")
    if spec and video.resolve().parent == spec.dest_path.resolve():
        return f"{spec.title} ({spec.licence})", "\n".join(spec.notes)
    return f"user-supplied video: {video.name}", None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--clip", help="dataset clip, e.g. P9/NP/1 (default)")
    src.add_argument("--video", type=Path, help="any video file")
    ap.add_argument("--out", type=Path,
                    help="output directory (default depends on the source)")
    ap.add_argument("--max-frames", type=int, default=44)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    if args.video is not None:
        video = args.video
        if not video.exists():
            print(f"no such video: {video}")
            print("hint: envs/core/bin/python scripts/download_data.py sample_videos")
            return 1
        out = args.out or REPO / "experiments" / "demo_external"
        info = probe_video(video)
        if info is None:
            print(f"could not read {video}")
            return 1
        source, notes = _source_credit(video)
        label, in_distribution, note = video.name, False, OOD_NOTE
    else:
        clip = InsoleGaitRite().get(*(args.clip or "P9/NP/1").split("/"))
        if clip.video_path is None:
            print(f"no video for {clip.key}")
            return 1
        video, info = clip.video_path, clip.video_info
        out = args.out or REPO / "experiments" / "demo_run"
        source = ("Insole-GAITRite dataset, Zenodo 10.5281/zenodo.19662017 (CC-BY-4.0)")
        notes, label, in_distribution = None, clip.key, True
        note = ("A dataset clip, so the loading model is operating on the camera and "
                "setup it was trained for. On other footage the same stages run but "
                "the loading stage is extrapolating.")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"running the pipeline on {label} ...")
    stages = LivePipeline(video, out, max_frames=args.max_frames,
                          stride=args.stride).run()

    failed = [s.title for s in stages if s.status == "failed"]
    for s in stages:
        size = ""
        if s.preview:
            kb = (out / s.preview).stat().st_size / 1024
            size = f"  {s.preview_w}x{s.preview_h}  {kb:.0f} KB"
        print(f"  {s.title:<26}{s.status:<9}{(s.seconds or 0):>5.1f}s{size}")

    # The STL is large and regenerable; the site does not display it.
    stl = out / "insole.stl"
    stl_mb = stl.stat().st_size / 1e6 if stl.exists() else 0.0
    if stl.exists():
        stl.unlink()

    meta = {
        "source": source,
        "clip": label,
        "video": f"{info.width}x{info.height}, {info.fps:.1f} fps, "
                 f"{info.duration_s:.1f} s",
        "in_distribution": in_distribution,
        "note": note,
        "stl_mb_generated_then_discarded": round(stl_mb, 1),
        "stages": [s.as_dict() for s in stages],
    }
    if notes:
        meta["attribution"] = notes
    (out / "demo.json").write_text(json.dumps(meta, indent=2))

    total_kb = sum(f.stat().st_size for f in out.glob("*.jpg")) / 1024
    total_kb += sum(f.stat().st_size for f in out.glob("*.png")) / 1024
    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'all stages completed'}")
    print(f"previews total {total_kb:.0f} KB -> {out.relative_to(REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
