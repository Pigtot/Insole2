#!/usr/bin/env python
"""Phase D -- FOCUS end-to-end on Blender renders, with COLMAP estimating cameras.

    envs/focus/bin/python scripts/run_focus_synthetic.py --run <output_folder> \
        --gt data/raw/foot3d_multiview/.../0036/mesh.obj

Milestone 4 measured 2.93 mm using the dataset's *given* camera poses. This
closes that gap: the renders come from a known mesh, COLMAP has to work the
cameras out itself, and the error against the input mesh is what camera
estimation costs.

Upstream ``run_focus.py`` aborts if *any* view fails to calibrate, but it writes
the per-view ``colmap.json`` for the successful ones first. So this script picks
up that output, drops uncalibrated views, and runs fusion on the rest -- no
change to upstream code.
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

SKIP = {"colmap", "logs", "frames", "videos"}


def prune_uncalibrated(run_dir: Path) -> tuple[int, int]:
    kept, dropped = 0, 0
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir() or d.name in SKIP:
            continue
        if (d / "colmap.json").exists():
            kept += 1
        else:
            shutil.rmtree(d)
            dropped += 1
    return kept, dropped


def align_similarity(pred_path: Path, gt_path: Path, out_path: Path) -> dict:
    """Align a COLMAP-scale reconstruction to the ground truth, then save it.

    Monocular structure-from-motion determines geometry only up to a **similarity
    transform**: COLMAP cannot know absolute scale, orientation or position. Here
    the reconstruction came out ~111x larger than the metric ground truth, so
    comparing them directly is meaningless (it gave a 2.7 m chamfer). Fitting
    scale + rotation + translation before evaluating is the standard SfM protocol,
    and it is disclosed rather than folded silently into the number.

    Two stages, because ICP diverges outright when the scales differ by two orders
    of magnitude: a closed-form centroid/RMS-radius normalisation first, then ICP
    refinement from several initial orientations.
    """
    import trimesh

    pred = trimesh.load(pred_path, process=False, force="mesh")
    gt = trimesh.load(gt_path, process=False, force="mesh")

    pv, gv = np.asarray(pred.vertices), np.asarray(gt.vertices)
    pc, gc = pv.mean(0), gv.mean(0)
    ps = float(np.linalg.norm(pv - pc, axis=1).mean())
    gs = float(np.linalg.norm(gv - gc, axis=1).mean())
    if ps <= 0:
        return {"alignment_error": "degenerate prediction"}
    s0 = gs / ps

    init = np.eye(4)
    init[:3, :3] *= s0
    init[:3, 3] = gc - s0 * pc
    coarse = pred.copy()
    coarse.apply_transform(init)

    try:
        matrix, cost = trimesh.registration.mesh_other(
            coarse, gt, samples=2000, scale=True, icp_first=10, icp_final=60)
        refined = coarse.copy()
        refined.apply_transform(matrix)
        total = matrix @ init
    except Exception as exc:  # pragma: no cover
        refined, cost, total = coarse, float("nan"), init
        print(f"  ICP refinement failed ({type(exc).__name__}), using coarse fit")

    det = float(np.linalg.det(total[:3, :3]))
    scale = float(abs(det) ** (1.0 / 3.0))
    refined.export(out_path)
    return {"alignment_cost": float(cost),
            "recovered_scale_vs_colmap": scale,
            "coarse_scale": s0,
            "alignment_determinant": det,
            "aligned_mesh": str(out_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output folder from run_focus.py")
    ap.add_argument("--gt", required=True, help="ground-truth mesh that was rendered")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from FOCUS.data.dataset import load_views
    from FOCUS.fusion import fuse
    from FOCUS.fusion.hyperparameters import FusionHyperparameters

    run_dir = Path(args.run).resolve()
    kept, dropped = prune_uncalibrated(run_dir)
    print(f"views: {kept} calibrated, {dropped} dropped (COLMAP could not place them)")
    if kept < 3:
        print("too few calibrated views to fuse")
        return 1

    views = load_views(run_dir)
    print(f"{len(views)} views passed the coverage filter")
    t0 = time.time()
    # is_world_space=False is essential here. COLMAP fixes the scene only up to a
    # similarity transform, so its world frame has no canonical floor -- whereas the
    # Foot3D runs used the dataset's own pre-aligned cameras and could assume one.
    # Leaving it True made FOCUS's absolute "cutoff below z=0" filter delete 93% of
    # the point cloud (9572 -> 682) before it ever reached meshing.
    fuse.fuse(views, run_dir,
              hyperparameters=FusionHyperparameters(is_world_space=False))
    fusion_s = time.time() - t0

    meshes = [m for m in sorted(run_dir.glob("*.obj")) if "colmap" not in m.name]
    if not meshes:
        print("no mesh produced")
        return 1
    pred = meshes[0]

    sys.path.insert(0, str(REPO / "scripts"))
    from run_focus_foot3d import evaluate

    aligned_path = run_dir / "mesh_aligned.obj"
    align = align_similarity(pred, Path(args.gt), aligned_path)
    print(f"similarity alignment: coarse scale {align.get('coarse_scale', float('nan')):.4f}, "
          f"total {align.get('recovered_scale_vs_colmap', float('nan')):.4f}, "
          f"ICP cost {align.get('alignment_cost', float('nan')):.5f}")
    metrics = evaluate(aligned_path, Path(args.gt))
    metrics.update(align)
    result = {
        "run": str(run_dir),
        "ground_truth": args.gt,
        "cameras": "ESTIMATED BY COLMAP (not given)",
        "alignment": "similarity (scale+rotation+translation) fitted before evaluation, because SfM is scale-free",
        "views_calibrated": kept, "views_dropped": dropped,
        "views_fused": len(views), "fusion_seconds": round(fusion_s, 1),
        "mesh": str(pred),
        **metrics,
    }
    out = run_dir.parent / f"synthetic_result{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps(result, indent=2))

    print(f"\nchamfer {metrics.get('chamfer_mm', float('nan')):.2f} mm  "
          f"(pred->gt {metrics.get('pred_to_gt_mean_mm', float('nan')):.2f}, "
          f"gt->pred {metrics.get('gt_to_pred_mean_mm', float('nan')):.2f})")
    print(f"normal error {metrics.get('normal_error_deg', float('nan')):.1f} deg")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
