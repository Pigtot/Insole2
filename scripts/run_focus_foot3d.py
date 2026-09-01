#!/usr/bin/env python
"""Run FOCUS on the real Foot3D multiview benchmark, with ground-truth cameras.

    envs/focus/bin/python scripts/run_focus_foot3d.py --scan 0035
    envs/focus/bin/python scripts/run_focus_foot3d.py --all --max-views 24

Foot3D ships a per-scan ``colmap.json`` that is *already* in the format FOCUS's
own ``colmap2pytorch3d`` produces (same ``camera`` keys, same per-image
``R``/``C``/``T``). So COLMAP never has to run: we split that file per view and
hand FOCUS known-good cameras. That removes the stage that failed on synthetic
renders and isolates what we actually want to measure -- reconstruction quality.

Devices follow adapters/focus_mac: TOC prediction on MPS, all PyTorch3D
geometry on CPU (it segfaults on MPS).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from adapters import focus_mac  # noqa: E402

focus_mac.configure()

import numpy as np  # noqa: E402

FOOT3D = REPO / "data/raw/foot3d_multiview/OneDrive_1_9-1-2026"
OUT_ROOT = REPO / "experiments/focus_baseline/foot3d"
TOC_MODEL = focus_mac.FOCUS_ROOT / "data/toc_model/densedepth_toc_predictor.pth"


def prepare_views(scan_dir: Path, work: Path, max_views: int | None) -> tuple[list, list]:
    """Predict TOC for a scan's images and attach ground-truth calibration."""
    from FOCUS.data import io
    from FOCUS.toc_prediction import predict as toc

    calib = json.loads((scan_dir / "colmap.json").read_text())
    camera = calib["camera"]
    by_stem = {Path(i["pth"]).stem: i for i in calib["images"]}

    imgs, keys = io.load_images_from_dir(scan_dir / "rgb")
    if max_views:
        imgs, keys = imgs[:max_views], keys[:max_views]

    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    toc.predict_toc(imgs, TOC_MODEL, work, keys, batch_size=4)
    pred_s = time.time() - t0

    written = 0
    for k in keys:
        entry = by_stem.get(k)
        if entry is None:
            continue
        view_data = {**camera, "R": entry["R"], "T": entry["T"], "C": entry["C"]}
        (work / k / "colmap.json").write_text(json.dumps(view_data))
        written += 1
    return keys, [pred_s, written]


def evaluate(pred_mesh_path: Path, gt_mesh_path: Path, n_samples: int = 10000) -> dict:
    """Chamfer distance using FOCUS's own evaluation protocol.

    Uses upstream ``FOCUS.eval.eval``: both meshes are cropped at z = 0.1 m
    (``cutoff_slice_FIND``) before comparison, because the Foot3D ground-truth
    scans include the leg up to ~200 mm while FOCUS only reconstructs the foot.
    Comparing uncropped meshes measures the missing leg, not reconstruction
    error. Distances are point-to-**surface** (``trimesh.proximity``), matching
    the paper, not point-to-point nearest neighbour.

    Runs entirely on CPU -- no pytorch3d, so the MPS segfault issue never arises.
    """
    import trimesh
    from FOCUS.eval.eval import angle_between_normals, cutoff_slice_FIND, sample

    def load_cropped(path):
        m = trimesh.load(path, process=False, force="mesh")
        return cutoff_slice_FIND(m)

    gt = load_cropped(gt_mesh_path)
    pred = load_cropped(pred_mesh_path)
    if len(gt.faces) == 0 or len(pred.faces) == 0:
        return {"eval_error": "empty mesh after z=0.1 crop"}

    np.random.seed(0)
    gt_pts, gt_norm = sample(gt, n_samples)
    pred_pts, pred_norm = sample(pred, n_samples)

    _, d_pred2gt, face_x = trimesh.proximity.closest_point(gt, pred_pts)
    _, d_gt2pred, face_y = trimesh.proximity.closest_point(pred, gt_pts)

    n_pred2gt = angle_between_normals(gt.face_normals[face_x], pred_norm)
    n_gt2pred = angle_between_normals(pred.face_normals[face_y], gt_norm)

    return {
        "chamfer_mm": float((d_pred2gt.mean() + d_gt2pred.mean()) / 2 * 1000),
        "pred_to_gt_mean_mm": float(d_pred2gt.mean() * 1000),
        "gt_to_pred_mean_mm": float(d_gt2pred.mean() * 1000),
        "pred_to_gt_median_mm": float(np.median(d_pred2gt) * 1000),
        "gt_to_pred_median_mm": float(np.median(d_gt2pred) * 1000),
        "normal_error_deg": float(np.nanmean(np.concatenate([n_pred2gt, n_gt2pred]))),
        "n_pred_faces_cropped": int(len(pred.faces)),
        "n_gt_faces_cropped": int(len(gt.faces)),
    }


def run_scan(scan: str, max_views: int | None, overwrite: bool) -> dict:
    from FOCUS.data.dataset import load_views
    from FOCUS.fusion import fuse
    from FOCUS.fusion.hyperparameters import FusionHyperparameters

    scan_dir = FOOT3D / scan
    work = OUT_ROOT / scan
    if work.exists() and overwrite:
        shutil.rmtree(work)

    result: dict = {"scan": scan}
    keys, (pred_s, n_calib) = prepare_views(scan_dir, work, max_views)
    result["n_images"] = len(keys)
    result["n_calibrated"] = n_calib
    result["toc_seconds"] = round(pred_s, 1)
    print(f"  {len(keys)} views, {n_calib} calibrated, TOC {pred_s:.1f}s")

    views = load_views(work)
    result["n_views_used"] = len(views)
    if len(views) < 3:
        result["error"] = "too few usable views"
        return result

    t0 = time.time()
    hp = FusionHyperparameters(is_world_space=True)
    fuse.fuse(views, work, hyperparameters=hp)
    result["fusion_seconds"] = round(time.time() - t0, 1)

    meshes = sorted(work.glob("*.obj")) + sorted(work.glob("**/*.obj"))
    meshes = [m for m in meshes if "colmap" not in m.name]
    if not meshes:
        result["error"] = "no mesh produced"
        return result
    pred = meshes[0]
    result["mesh"] = str(pred.relative_to(REPO))
    try:
        result.update(evaluate(pred, scan_dir / "mesh.obj"))
    except Exception as exc:
        result["eval_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", default="0035")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true", default=True)
    args = ap.parse_args()

    if not FOOT3D.exists():
        print(f"Foot3D not found at {FOOT3D}"); return 1

    scans = sorted(p.name for p in FOOT3D.iterdir()
                   if p.is_dir() and p.name.isdigit()) if args.all else [args.scan]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for s in scans:
        print(f"\n=== scan {s} ===")
        try:
            r = run_scan(s, args.max_views, args.overwrite)
        except Exception as exc:
            r = {"scan": s, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAILED {r['error']}")
        results.append(r)
        if "chamfer_mm" in r:
            print(f"  chamfer {r['chamfer_mm']:.2f} mm  "
                  f"(pred->gt {r['pred_to_gt_mean_mm']:.2f}, gt->pred {r['gt_to_pred_mean_mm']:.2f})")

    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))
    ok = [r for r in results if "chamfer_mm" in r]
    if ok:
        ch = np.array([r["chamfer_mm"] for r in ok])
        print(f"\n{len(ok)}/{len(results)} scans reconstructed")
        print(f"chamfer: mean {ch.mean():.2f} mm, median {np.median(ch):.2f}, "
              f"range {ch.min():.2f}-{ch.max():.2f}")
    print(f"wrote {(OUT_ROOT / 'results.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
