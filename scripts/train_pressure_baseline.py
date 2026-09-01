#!/usr/bin/env python
"""Milestone 3 -- Baseline 0 (mean) and Baseline 1 (kinematics) for plantar loading.

    envs/core/bin/python scripts/train_pressure_baseline.py --max-clips-per-participant 8

Question: does ordinary video carry information about plantar loading, beyond
what predicting the training average already gives you?

Targets are 64 channels: the left foot's 32 sensors, then the right foot's 32
**re-indexed into the left foot's numbering** so channel i and channel 32+i are
the same anatomical site. The two insoles do not share numbering, so this
re-indexing is required for the halves to be comparable.

Evaluation is participant-disjoint, from the committed split.
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
from visole.data.splits import Split  # noqa: E402
from visole.evaluation.metrics import (  # noqa: E402
    contact_metrics, cop_error, grouped_report, regression_metrics, skill_score,
)
from visole.models.video_features import (  # noqa: E402
    add_temporal_context, extract_clip_features,
)
from visole.pressure.sensor_map import load_sensor_map  # noqa: E402

OUT = REPO / "experiments" / "pressure_baseline" / "outputs"
CACHE = REPO / "data" / "processed" / "video_features"
OFFSETS = (-6, -3, 0, 3, 6)


POSE_CACHE = REPO / "data" / "processed" / "pose"


def motion_source(clip):
    """Coarse whole-body motion features (Baseline 1a)."""
    cache = CACHE / f"{clip.participant}_{clip.condition}_{clip.clip_id}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["times"], z["features"]
    cf = extract_clip_features(clip)
    if cf is None:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, times=cf.times, features=cf.features)
    return cf.times, cf.features


def pose_source(clip):
    """Limb kinematics from cached 2D pose (Baseline 1b).

    Only reads the cache -- run scripts/extract_pose.py first. Clips without a
    cache are skipped rather than silently falling back to weaker features.
    """
    from visole.models.pose_features import PoseSequence, pose_to_features

    cache = POSE_CACHE / f"{clip.participant}_{clip.condition}_{clip.clip_id}.npz"
    if not cache.exists():
        return None
    z = np.load(cache)
    seq = PoseSequence(times=z["times"], keypoints=z["keypoints"],
                       scores=z["scores"], detected=z["detected"])
    return seq.times, pose_to_features(seq)


FEATURE_SOURCES = {"motion": motion_source, "pose": pose_source}


def clip_samples(clip, frame_stride: int, right_map, source) -> tuple | None:
    """Features and 64-channel targets for one clip, aligned in clip time."""
    got = source(clip)
    if got is None:
        return None
    times, feats = got
    if len(times) < 3:
        return None

    ctx = add_temporal_context(feats, OFFSETS)
    valid = np.isfinite(ctx).all(axis=1)
    if valid.sum() < 10:
        return None

    left = clip.pressure("left", baseline_correct=True)
    right = clip.pressure("right", baseline_correct=True)
    tl = clip.pressure_time("left")
    tr = clip.pressure_time("right")
    # right foot re-indexed into left numbering so the halves are homologous
    right_canonical = right_map.to_other_foot(right)

    idx = np.arange(len(times))[valid][::frame_stride]
    if len(idx) == 0:
        return None
    t = times[idx]
    y_left = np.stack([np.interp(t, tl, left[:, c]) for c in range(32)], axis=1)
    y_right = np.stack([np.interp(t, tr, right_canonical[:, c]) for c in range(32)], axis=1)
    y = np.concatenate([y_left, y_right], axis=1)

    # Contact labels from GAITRite, in clip time. Skip clips without them so the
    # contact task is never trained on guessed labels.
    cl = clip.contact_labels("left", t)
    cr = clip.contact_labels("right", t)
    if cl is None or cr is None:
        return None
    # Second array marks where GAITRite actually covers the foot; outside the
    # walkway the labels are meaningless (see Clip.contact_labels).
    contact = np.stack([cl[0], cr[0]], axis=1)
    contact_valid = np.stack([cl[1], cr[1]], axis=1)

    return (ctx[idx], y, np.full(len(idx), clip.participant),
            np.full(len(idx), clip.condition), contact, contact_valid)


def assemble(ds, participants, frame_stride, max_per_participant, right_map, label, source):
    X, Y, P, C, K, V = [], [], [], [], [], []
    # Quota is per (participant, condition). A per-participant quota silently
    # fills up with whichever condition sorts first -- the first run of this
    # experiment covered FP only and looked like a complete result.
    per_count: dict[tuple, int] = {}
    clips = ds.filter(participants=participants, conditions=WALKING_CONDITIONS,
                      require_all_modalities=True)
    t0 = time.time()
    for i, clip in enumerate(clips):
        key = (clip.participant, clip.condition)
        if per_count.get(key, 0) >= max_per_participant:
            continue
        got = clip_samples(clip, frame_stride, right_map, source)
        if got is None:
            continue
        x, y, p, c, k, kv = got
        X.append(x); Y.append(y); P.append(p); C.append(c); K.append(k); V.append(kv)
        per_count[key] = per_count.get(key, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i + 1}/{len(clips)} clips scanned, "
                  f"{sum(len(a) for a in X)} samples, {time.time() - t0:.0f}s", flush=True)
    if not X:
        raise RuntimeError(f"no usable clips for {label}")
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(P),
            np.concatenate(C), np.concatenate(K), np.concatenate(V))


def fit_ridge(X, Y, alpha: float):
    """Closed-form ridge on standardised features with an intercept."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    ymu = Y.mean(0)
    A = Xs.T @ Xs
    reg = alpha * np.eye(A.shape[0])
    reg[-1, -1] = 0.0                      # never penalise the intercept
    W = np.linalg.solve(A + reg, Xs.T @ (Y - ymu))
    return {"mu": mu, "sd": sd, "W": W, "ymu": ymu}


def predict_ridge(model, X):
    Xs = (X - model["mu"]) / model["sd"]
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    return Xs @ model["W"] + model["ymu"]


def fit_contact_gated(X, Y, contact, valid, alpha: float):
    """Contact-gated predictor, after HOPE (arXiv 2608.06192).

    HOPE predicts hand pressure as ``p = c * p_tilde`` -- a contact probability
    multiplied by a magnitude -- so that "no contact implies no pressure" is
    enforced by the architecture rather than left for the model to discover.

    That structure fits plantar loading even better than it fits hands: a foot in
    swing is *exactly* zero across all 32 channels for roughly half of every
    walking clip. A single linear map cannot represent a hard zero and instead
    smears a compromise across both phases.

    Two differences from HOPE, both forced by our data:
      * the gate is per **foot**, not per vertex -- a foot is on the ground or it
        is not, and we have exactly two feet;
      * the magnitude head is fitted on contact frames only, so it learns "given
        this foot is down, how is load distributed" without being pulled toward
        zero by swing frames.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gates = {}
    for j, side in enumerate(("left", "right")):
        m = valid[:, j]
        y = contact[m, j]
        if m.sum() < 50 or len(set(y.tolist())) < 2:
            gates[side] = None
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        clf.fit(X[m], y)
        gates[side] = clf

    # Magnitude head: contact frames only, per foot half of the target.
    mags = {}
    for j, side in enumerate(("left", "right")):
        cols = slice(0, 32) if side == "left" else slice(32, 64)
        m = valid[:, j] & contact[:, j]
        if m.sum() < 50:
            mags[side] = None
            continue
        mags[side] = fit_ridge(X[m], Y[m, cols], alpha)
    return {"gates": gates, "mags": mags}


def predict_contact_gated(model, X):
    out = np.zeros((len(X), 64))
    for j, side in enumerate(("left", "right")):
        cols = slice(0, 32) if side == "left" else slice(32, 64)
        gate, mag = model["gates"][side], model["mags"][side]
        if mag is None:
            continue
        magnitude = np.clip(predict_ridge(mag, X), 0, None)
        if gate is None:
            out[:, cols] = magnitude
            continue
        prob = gate.predict_proba(X)[:, 1][:, None]
        out[:, cols] = prob * magnitude
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=str(REPO / "data/splits/participant_holdout.json"))
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--max-clips-per-participant", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=None, help="ridge alpha; default: tuned on val")
    ap.add_argument("--features", choices=sorted(FEATURE_SOURCES), default="motion",
                    help="motion = coarse whole-body (1a); pose = limb kinematics (1b)")
    ap.add_argument("--tag", default=None, help="suffix for output filenames")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source = FEATURE_SOURCES[args.features]
    tag = args.tag or args.features
    ds = InsoleGaitRite()
    split = Split.load(args.split)
    right_map = load_sensor_map("right")
    left_map = load_sensor_map("left")
    print(f"split: train {len(split.train)} / val {len(split.val)} / test {len(split.test)} participants")
    if args.dry_run:
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, ps in (("train", split.train), ("val", split.val), ("test", split.test)):
        data[name] = assemble(ds, ps, args.frame_stride, args.max_clips_per_participant,
                              right_map, name, source)
        conds = sorted(set(data[name][3].tolist()))
        print(f"{name}: X{data[name][0].shape} Y{data[name][1].shape} conditions={conds}")

    Xtr, Ytr, Ptr, Ctr, Ktr, Vtr = data["train"]
    Xva, Yva, _, _, Kva, Vva = data["val"]
    Xte, Yte, Pte, Cte, Kte, Vte = data["test"]

    # --- Baseline 0: predict the training mean ---------------------------
    base_te = np.tile(Ytr.mean(0), (len(Yte), 1))
    b0 = regression_metrics(Yte, base_te)

    # --- Baseline 1: ridge on kinematic features -------------------------
    if args.alpha is None:
        best, best_alpha = None, None
        for a in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7):
            m = fit_ridge(Xtr, Ytr, a)
            sc = float(((predict_ridge(m, Xva) - Yva) ** 2).mean())
            print(f"  alpha={a:>8.0f}  val MSE={sc:,.1f}")
            if best is None or sc < best:
                best, best_alpha = sc, a
        alpha = best_alpha
    else:
        alpha = args.alpha
    print(f"chosen alpha={alpha}")

    model = fit_ridge(Xtr, Ytr, alpha)
    pred_te = np.clip(predict_ridge(model, Xte), 0, None)
    b1 = regression_metrics(Yte, pred_te)
    b1.update(skill_score(Yte, pred_te, base_te))

    # --- contact-gated variant (HOPE-style) -------------------------------
    gated = fit_contact_gated(Xtr, Ytr, Ktr, Vtr, alpha)
    pred_gated = np.clip(predict_contact_gated(gated, Xte), 0, None)
    b2 = regression_metrics(Yte, pred_gated)
    b2.update(skill_score(Yte, pred_gated, base_te))
    cop_b2 = cop_error(Yte[:, :32], pred_gated[:, :32], left_map.xy)

    xy = left_map.xy
    cop_b0 = cop_error(Yte[:, :32], base_te[:, :32], xy)
    cop_b1 = cop_error(Yte[:, :32], pred_te[:, :32], xy)

    by_participant = grouped_report(Yte, pred_te, Pte, base_te)["by_group"]
    by_condition = grouped_report(Yte, pred_te, Cte, base_te)["by_group"]

    # --- contact task: can video tell whether a foot is on the ground? -----
    # Labels come from GAITRite footfall events converted to clip time, which is
    # an independent reference system -- not from the insole signal we predict.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    contact_results = {}
    for j, side in enumerate(("left", "right")):
        mtr, mte = Vtr[:, j], Vte[:, j]      # keep only GAITRite-covered samples
        ytr, yte = Ktr[mtr, j], Kte[mte, j]
        xtr, xte = Xtr[mtr], Xte[mte]
        if len(set(ytr.tolist())) < 2 or len(yte) < 10:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=0.1))
        clf.fit(xtr, ytr)
        pred = clf.predict(xte)
        m = contact_metrics(yte, pred)
        m["n_train"] = int(len(ytr))
        m["n_test"] = int(len(yte))
        m["gaitrite_coverage_test"] = float(mte.mean())
        # A majority-class predictor is the honest floor for a binary task.
        majority = np.full_like(yte, bool(ytr.mean() > 0.5))
        m["majority_class_accuracy"] = float((majority == yte).mean())
        m["beats_majority"] = bool(m["accuracy"] > m["majority_class_accuracy"])
        contact_results[side] = m

    report = {
        "task": "video frame -> 64 channels (left 32 + right 32 re-indexed to left numbering)",
        "feature_set": args.features,
        "units": "baseline-corrected raw sensor counts (NOT kPa)",
        "split": {"train": split.train, "val": split.val, "test": split.test},
        "n_samples": {k: int(len(v[0])) for k, v in data.items()},
        "n_features": int(Xtr.shape[1]),
        "n_context_offsets": list(OFFSETS),
        "ridge_alpha": alpha,
        "baseline_0_mean_predictor": b0,
        "baseline_1_kinematics_ridge": b1,
        "baseline_2_contact_gated": b2,
        "cop_error_left_foot": {"baseline_0": cop_b0, "baseline_1": cop_b1,
                                "baseline_2_gated": cop_b2},
        "baseline_1_by_participant": by_participant,
        "baseline_1_by_condition": by_condition,
        "contact_task": contact_results,
    }
    (OUT / f"baseline_report_{tag}.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / f"baseline_predictions_{tag}.npz",
                        y_true=Yte, y_pred=pred_te, y_gated=pred_gated, y_base=base_te,
                        participant=Pte, condition=Cte)

    print("\n=== TEST (participant-disjoint) ===")
    print(f"{'':28}{'Baseline 0':>14}{'B1 direct':>14}{'B2 gated':>14}")
    for key, label in (("mae", "MAE (counts)"), ("rmse", "RMSE (counts)"),
                       ("total_load_mae", "total-load MAE"),
                       ("total_load_r", "total-load r"),
                       ("mean_channel_r", "mean per-channel r")):
        print(f"{label:28}{b0[key]:>14.3f}{b1[key]:>14.3f}{b2[key]:>14.3f}")
    print(f"{'COP error (svg units)':28}{cop_b0['cop_mae_units']:>14.1f}"
          f"{cop_b1['cop_mae_units']:>14.1f}{cop_b2['cop_mae_units']:>14.1f}")
    print(f"\nskill vs mean predictor  B1 direct: {b1['skill_vs_baseline']:+.4f}"
          f"   (total load {b1['skill_total_load']:+.4f})")
    print(f"skill vs mean predictor  B2 gated : {b2['skill_vs_baseline']:+.4f}"
          f"   (total load {b2['skill_total_load']:+.4f})")
    print(f"beats the mean predictor: direct={b1['beats_baseline']} gated={b2['beats_baseline']}")
    print("\nby test participant:")
    for p, m in sorted(by_participant.items()):
        print(f"  {p:>5}  skill {m['skill_vs_baseline']:+.4f}   MAE {m['mae']:8.2f}")
    print("\nby condition:")
    for c, m in sorted(by_condition.items()):
        print(f"  {c:>5}  skill {m['skill_vs_baseline']:+.4f}   MAE {m['mae']:8.2f}")
    print("\ncontact task (stance vs swing, labels from GAITRite):")
    for side, m in contact_results.items():
        print(f"  {side:>5}  acc {m['accuracy']:.3f}  balanced {m['balanced_accuracy']:.3f}"
              f"  majority {m['majority_class_accuracy']:.3f}"
              f"  beats majority: {m['beats_majority']}")
    print(f"\nwrote {(OUT / f'baseline_report_{tag}.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
