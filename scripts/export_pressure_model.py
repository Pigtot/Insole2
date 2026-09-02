#!/usr/bin/env python
"""Fit the loading model on ALL participants and persist it for inference.

    envs/core/bin/python scripts/export_pressure_model.py

The evaluation numbers in Milestone 3 come from participant-disjoint splits and
stay that way. This is a separate, deployment-only fit over every participant,
for running on new video where there is nothing to hold out.

Saved as plain arrays (ridge weights, logistic gate coefficients, scaler
statistics) rather than pickled estimator objects, so the file cannot break on a
scikit-learn upgrade and can be read without importing anything.

**The model is trained on one fixed camera in one room.** Applying it to an
arbitrary phone video is out of distribution and its output should be treated as
illustrative, not as a measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from visole.data.insole_gaitrite import InsoleGaitRite  # noqa: E402
from visole.pressure.sensor_map import load_sensor_map  # noqa: E402
from train_pressure_baseline import FEATURE_SOURCES, fit_ridge  # noqa: E402
from evaluate_loso import assemble_all  # noqa: E402

OUT = REPO / "models"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="pose", choices=sorted(FEATURE_SOURCES))
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--max-clips-per-participant", type=int, default=12)
    ap.add_argument("--frame-stride", type=int, default=1)
    args = ap.parse_args()

    ds = InsoleGaitRite()
    right_map = load_sensor_map("right")
    source = FEATURE_SOURCES[args.features]
    print(f"assembling all walking clips ({args.features})...")
    X, Y, P, C, K, V = assemble_all(ds, args.frame_stride,
                                    args.max_clips_per_participant, right_map, source)
    parts = sorted(set(P.tolist()), key=lambda s: int(s[1:]))
    print(f"{X.shape[0]} samples, {X.shape[1]} features, {len(parts)} participants")

    ridge = fit_ridge(X, Y, args.alpha)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    gates = {}
    for j, side in enumerate(("left", "right")):
        m = V[:, j]
        y = K[m, j]
        sc = StandardScaler().fit(X[m])
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[m]), y)
        gates[side] = dict(mean=sc.mean_, scale=sc.scale_,
                           coef=clf.coef_.ravel(), intercept=clf.intercept_)
        acc = clf.score(sc.transform(X[m]), y)
        print(f"  {side} contact gate: train accuracy {acc:.3f} on {m.sum()} samples")

    mags = {}
    for j, side in enumerate(("left", "right")):
        cols = slice(0, 32) if side == "left" else slice(32, 64)
        m = V[:, j] & K[:, j]
        mags[side] = fit_ridge(X[m], Y[m, cols], args.alpha)
        print(f"  {side} magnitude head: {m.sum()} contact samples")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "ridge_mu": ridge["mu"], "ridge_sd": ridge["sd"],
        "ridge_W": ridge["W"], "ridge_ymu": ridge["ymu"],
    }
    for side in ("left", "right"):
        for k, v in gates[side].items():
            payload[f"gate_{side}_{k}"] = v
        for k in ("mu", "sd", "W", "ymu"):
            payload[f"mag_{side}_{k}"] = mags[side][k]
    np.savez_compressed(OUT / "pressure_model.npz", **payload)

    meta = {
        "features": args.features, "alpha": args.alpha,
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "participants": parts,
        "trained_on": "ALL participants -- deployment fit, not an evaluation",
        "evaluation": ("Held-out performance is +0.514 +/- 0.119 skill over 22 "
                       "leave-one-subject-out folds; see experiments/pressure_baseline."),
        "limitation": ("Trained on one fixed camera in one room with shod feet. "
                       "Applying it to arbitrary phone video is OUT OF DISTRIBUTION "
                       "and the output is illustrative, not a measurement."),
        "units": "baseline-corrected raw sensor counts, NOT kPa",
        "output": "64 channels: left 32, then right 32 re-indexed into left numbering",
    }
    (OUT / "pressure_model.json").write_text(json.dumps(meta, indent=2))
    size = (OUT / "pressure_model.npz").stat().st_size / 1e6
    print(f"\nwrote models/pressure_model.npz ({size:.1f} MB) + pressure_model.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
