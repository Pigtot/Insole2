#!/usr/bin/env python
"""Leave-one-subject-out evaluation -- real error bars for Milestone 3.

    envs/core/bin/python scripts/evaluate_loso.py --features pose

The 14/4/4 holdout split gave a single number from four test participants, with
one of them (P23) well below the others. That is too few people to know whether
the result is stable. This runs 22 folds: every participant is the test set
once, so the spread across people is measured rather than guessed.

Data is assembled once and re-split per fold -- feature extraction is cached, so
the cost is 22 ridge solves, not 22 passes over the video.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.data.insole_gaitrite import InsoleGaitRite, WALKING_CONDITIONS  # noqa: E402
from visole.evaluation.metrics import cop_error, regression_metrics, skill_score  # noqa: E402
from visole.pressure.sensor_map import load_sensor_map  # noqa: E402
from train_pressure_baseline import (  # noqa: E402
    FEATURE_SOURCES, clip_samples, fit_contact_gated, fit_ridge,
    predict_contact_gated, predict_ridge,
)

OUT = REPO / "experiments" / "pressure_baseline" / "outputs"


def assemble_all(ds, frame_stride, max_per_participant, right_map, source):
    """Every usable walking clip, once."""
    X, Y, P, C, K, V = [], [], [], [], [], []
    counts: dict[str, int] = {}
    clips = ds.filter(conditions=WALKING_CONDITIONS, require_all_modalities=True)
    t0 = time.time()
    for i, clip in enumerate(clips):
        if counts.get(clip.participant, 0) >= max_per_participant:
            continue
        got = clip_samples(clip, frame_stride, right_map, source)
        if got is None:
            continue
        x, y, p, c, k, v = got
        X.append(x); Y.append(y); P.append(p); C.append(c); K.append(k); V.append(v)
        counts[clip.participant] = counts.get(clip.participant, 0) + 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(clips)} clips, {sum(len(a) for a in X)} samples, "
                  f"{time.time() - t0:.0f}s", flush=True)
    if not X:
        raise RuntimeError("no usable clips")
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(P),
            np.concatenate(C), np.concatenate(K), np.concatenate(V))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", choices=sorted(FEATURE_SOURCES), default="pose")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--max-clips-per-participant", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=10.0,
                    help="fixed alpha; tuning inside every fold would need a third split")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ds = InsoleGaitRite()
    right_map = load_sensor_map("right")
    left_xy = load_sensor_map("left").xy
    source = FEATURE_SOURCES[args.features]

    print(f"assembling all walking clips ({args.features} features)...")
    X, Y, P, C, K, V = assemble_all(ds, args.frame_stride,
                                    args.max_clips_per_participant, right_map, source)
    participants = sorted(set(P.tolist()), key=lambda s: int(s[1:]))
    print(f"total {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(participants)} participants")
    if args.dry_run:
        return 0

    folds = []
    for held in participants:
        te = P == held
        tr = ~te
        if te.sum() < 50 or tr.sum() < 200:
            print(f"  {held}: too few samples, skipped")
            continue
        Xtr, Ytr = X[tr], Y[tr]
        Xte, Yte = X[te], Y[te]

        base = np.tile(Ytr.mean(0), (len(Yte), 1))
        direct = np.clip(predict_ridge(fit_ridge(Xtr, Ytr, args.alpha), Xte), 0, None)
        gated_model = fit_contact_gated(Xtr, Ytr, K[tr], V[tr], args.alpha)
        gated = np.clip(predict_contact_gated(gated_model, Xte), 0, None)

        row = {
            "participant": held,
            "n_test": int(te.sum()),
            "b0": regression_metrics(Yte, base),
            "direct": {**regression_metrics(Yte, direct),
                       **skill_score(Yte, direct, base)},
            "gated": {**regression_metrics(Yte, gated),
                      **skill_score(Yte, gated, base)},
            "cop_direct": cop_error(Yte[:, :32], direct[:, :32], left_xy),
            "cop_gated": cop_error(Yte[:, :32], gated[:, :32], left_xy),
        }
        # which foot carries more load
        row["foot_dominance_acc"] = float(np.mean(
            (Yte[:, :32].sum(1) > Yte[:, 32:].sum(1)) ==
            (gated[:, :32].sum(1) > gated[:, 32:].sum(1))))
        folds.append(row)
        print(f"  {held:>4}  n={row['n_test']:5}  direct {row['direct']['skill_vs_baseline']:+.3f}"
              f"   gated {row['gated']['skill_vs_baseline']:+.3f}"
              f"   MAE {row['gated']['mae']:7.1f}", flush=True)

    def spread(key, sub):
        v = np.array([f[sub][key] for f in folds], dtype=float)
        return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
                "min": float(np.nanmin(v)), "median": float(np.nanmedian(v)),
                "max": float(np.nanmax(v))}

    summary = {
        "features": args.features,
        "alpha": args.alpha,
        "n_folds": len(folds),
        "n_samples_total": int(X.shape[0]),
        "skill_direct": spread("skill_vs_baseline", "direct"),
        "skill_gated": spread("skill_vs_baseline", "gated"),
        "mae_gated": spread("mae", "gated"),
        "total_load_r_gated": spread("total_load_r", "gated"),
        "n_folds_beating_mean_direct": int(sum(
            f["direct"]["skill_vs_baseline"] > 0 for f in folds)),
        "n_folds_beating_mean_gated": int(sum(
            f["gated"]["skill_vs_baseline"] > 0 for f in folds)),
        "foot_dominance_acc": {
            "mean": float(np.mean([f["foot_dominance_acc"] for f in folds])),
            "sd": float(np.std([f["foot_dominance_acc"] for f in folds])),
            "min": float(np.min([f["foot_dominance_acc"] for f in folds])),
        },
        "folds": folds,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"loso_{args.features}.json"
    path.write_text(json.dumps(summary, indent=2))

    sd, sg = summary["skill_direct"], summary["skill_gated"]
    print(f"\n=== leave-one-subject-out, {len(folds)} folds ===")
    print(f"skill direct : mean {sd['mean']:+.3f} +/- {sd['sd']:.3f}   "
          f"range {sd['min']:+.3f} .. {sd['max']:+.3f}")
    print(f"skill gated  : mean {sg['mean']:+.3f} +/- {sg['sd']:.3f}   "
          f"range {sg['min']:+.3f} .. {sg['max']:+.3f}")
    print(f"folds beating the mean predictor: direct "
          f"{summary['n_folds_beating_mean_direct']}/{len(folds)}, "
          f"gated {summary['n_folds_beating_mean_gated']}/{len(folds)}")
    fd = summary["foot_dominance_acc"]
    print(f"which-foot-is-loaded accuracy: {fd['mean']:.3f} +/- {fd['sd']:.3f} "
          f"(worst {fd['min']:.3f})")
    print(f"\nwrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
