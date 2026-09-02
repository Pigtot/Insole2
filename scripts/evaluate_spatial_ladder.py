#!/usr/bin/env python
"""Does the pose model have real spatial resolution, or just total-load tracking?

    envs/core/bin/python scripts/evaluate_spatial_ladder.py --features pose

Milestone 3 reports +0.514 skill on all 64 channels. That number cannot
distinguish two very different models:

  (a) the model knows *where* load sits on the foot, or
  (b) the model only tracks *how much* total load there is, and spreads it over
      the sensors in the average pattern.

Model (b) scores well on any per-sensor metric, because total load explains most
of the variance in every channel at once. The whole-foot skill score is
therefore not evidence of spatial resolution, and neither is a good regional
correlation.

The discriminating test is to **divide total load out**. For a spatial contrast
(heel vs forefoot, medial vs lateral, ...) score two things:

  absolute  the contrast's summed load, against a mean predictor
  share     the same contrast as a *fraction* of the load it belongs to,
            against a baseline that always predicts the training-set mean share

A model that only knows total load predicts a near-constant share, so it scores
~0.0 on the share test however well it does on the absolute one. The gap
between the two columns is the answer.

`null_random` is a control: an arbitrary equal-size partition of the sensors
with no anatomical meaning. If an anatomical contrast does not beat it, the
anatomy is not what the model is responding to.

Units are baseline-corrected raw sensor counts throughout. Not kPa -- no
calibration to physical pressure exists for this dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.data.insole_gaitrite import InsoleGaitRite, WALKING_CONDITIONS  # noqa: E402
from visole.pressure.sensor_map import load_sensor_map  # noqa: E402
from evaluate_loso import assemble_all  # noqa: E402
from train_pressure_baseline import (  # noqa: E402
    FEATURE_SOURCES, fit_contact_gated, predict_contact_gated,
)

OUT = REPO / "experiments" / "pressure_baseline" / "outputs"

#: Frames where the denominator of a share is this small are dropped: with both
#: feet near-airborne the ratio is dominated by sensor noise, not by anatomy.
MIN_DENOM_COUNTS = 200.0

#: Contrast names starting with this are pooled into the null control band.
NULL_PREFIX = "null_random_"


def build_contrasts(seed: int = 0, null_repeats: int = 10) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """(a_channels, b_channels, description) over the 64-channel target vector.

    Channels are left[0:32] + right[32:64], with the right foot already
    re-indexed into left numbering by `train_pressure_baseline`, so one map's
    geometry describes both halves.
    """
    m = load_sensor_map("left")
    regions = m.region_indices()
    medial = np.where(m.norm_xy[:, 0] < 0.5)[0]
    lateral = np.where(m.norm_xy[:, 0] >= 0.5)[0]
    heel = np.array(regions["heel"])
    forefoot = np.array(sorted(regions["metatarsal"] + regions["toes"]))

    def both(idx):
        """Same anatomical sensors on both feet."""
        return np.concatenate([idx, idx + 32])

    out = {
        "left_right": (np.arange(32), np.arange(32, 64),
                       "which foot carries the load"),
        "heel_forefoot": (both(heel), both(forefoot),
                          f"heel ({heel.size}) vs metatarsal+toes ({forefoot.size}), both feet"),
        "medial_lateral": (both(medial), both(lateral),
                           f"medial ({medial.size}) vs lateral ({lateral.size}), both feet"),
    }
    # A single random partition is one coin flip. Draw several and pool them, so
    # "beats the control" is a claim about the distribution of arbitrary splits
    # rather than about one lucky or unlucky draw.
    rng = np.random.default_rng(seed)
    for r in range(null_repeats):
        perm = rng.permutation(32)
        out[f"{NULL_PREFIX}{r}"] = (
            both(perm[:16]), both(perm[16:]),
            f"control {r}: arbitrary 16/16 split, no anatomical meaning")
    return out


def score_contrast(y_true, y_pred, y_train, a, b) -> dict:
    """Absolute and share skill for one spatial contrast.

    Skill is 1 - SSE(model)/SSE(baseline): 0.0 means no better than the
    baseline, negative means worse.
    """
    ta, tb = y_true[:, a].sum(1), y_true[:, b].sum(1)
    pa, pb = y_pred[:, a].sum(1), y_pred[:, b].sum(1)

    # absolute: predict the A-side load, baseline is the training mean
    base_a = y_train[:, a].sum(1).mean()
    sse_abs = ((pa - ta) ** 2).sum()
    sse_abs_base = ((base_a - ta) ** 2).sum()

    # share: predict A / (A + B), baseline is the training mean share
    denom_tr = y_train[:, a].sum(1) + y_train[:, b].sum(1)
    ok_tr = denom_tr > MIN_DENOM_COUNTS
    base_share = float((y_train[ok_tr][:, a].sum(1) / denom_tr[ok_tr]).mean()) \
        if ok_tr.any() else np.nan

    denom_t = ta + tb
    ok = denom_t > MIN_DENOM_COUNTS
    n_used = int(ok.sum())
    if n_used < 30 or not np.isfinite(base_share):
        return {"skill_absolute": np.nan, "skill_share": np.nan, "n_frames_share": n_used}

    share_t = ta[ok] / denom_t[ok]
    denom_p = pa[ok] + pb[ok]
    share_p = np.divide(pa[ok], denom_p, out=np.full(n_used, base_share),
                        where=denom_p > 0)
    sse_sh = ((share_p - share_t) ** 2).sum()
    sse_sh_base = ((base_share - share_t) ** 2).sum()

    return {
        "skill_absolute": float(1 - sse_abs / sse_abs_base) if sse_abs_base > 0 else np.nan,
        "skill_share": float(1 - sse_sh / sse_sh_base) if sse_sh_base > 0 else np.nan,
        "share_true_mean": float(share_t.mean()),
        "share_pred_mean": float(share_p.mean()),
        "share_true_sd": float(share_t.std()),
        "share_pred_sd": float(share_p.std()),
        "n_frames_share": n_used,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=sorted(FEATURE_SOURCES), default="pose")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--max-clips-per-participant", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0, help="null-control partitions")
    ap.add_argument("--null-repeats", type=int, default=10,
                    help="arbitrary partitions pooled into the control band")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    contrasts = build_contrasts(args.seed, args.null_repeats)
    named = [n for n in contrasts if not n.startswith(NULL_PREFIX)]
    nulls = [n for n in contrasts if n.startswith(NULL_PREFIX)]
    ds = InsoleGaitRite()
    right_map = load_sensor_map("right")

    print(f"assembling walking clips ({args.features} features)...", flush=True)
    X, Y, P, C, K, V = assemble_all(ds, args.frame_stride,
                                    args.max_clips_per_participant, right_map,
                                    FEATURE_SOURCES[args.features])
    participants = sorted(set(P.tolist()), key=lambda s: int(s[1:]))
    print(f"{X.shape[0]} samples, {X.shape[1]} features, {len(participants)} participants")
    for name in named:
        a, b, why = contrasts[name]
        print(f"  {name:16s} {a.size:3d} vs {b.size:3d} channels -- {why}")
    print(f"  {'null control':16s} {len(nulls)} arbitrary 32 vs 32 partitions, pooled")
    if args.dry_run:
        return 0

    folds = []
    for held in participants:
        te = P == held
        tr = ~te
        if te.sum() < 50 or tr.sum() < 200:
            print(f"  {held}: too few samples, skipped")
            continue
        model = fit_contact_gated(X[tr], Y[tr], K[tr], V[tr], args.alpha)
        pred = np.clip(predict_contact_gated(model, X[te]), 0, None)
        row = {"participant": held, "n_test": int(te.sum())}
        for name, (a, b, _) in contrasts.items():
            row[name] = score_contrast(Y[te], pred, Y[tr], a, b)
        folds.append(row)
        null_mean = float(np.nanmean([row[n]["skill_share"] for n in nulls]))
        print(f"  {held:>4}  " + "  ".join(
            f"{n}={row[n]['skill_share']:+.3f}" for n in named)
            + f"  null={null_mean:+.3f}", flush=True)

    def spread(names, key):
        """Pooled over folds, and over partitions when `names` is the null band."""
        v = np.array([f[n][key] for f in folds for n in names], dtype=float)
        per_fold = np.array([np.nanmean([f[n][key] for n in names]) for f in folds],
                            dtype=float)
        return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
                "min": float(np.nanmin(v)), "median": float(np.nanmedian(v)),
                "max": float(np.nanmax(v)),
                "n_folds_positive": int((per_fold > 0).sum())}

    def vs_null(name):
        """Paired against the same fold's own null band.

        Pairing by participant matters: folds differ a lot in difficulty, so an
        unpaired comparison of two means mixes that variance in. Here each
        contrast is compared only against controls computed on the same person.
        """
        margin = np.array([f[name]["skill_share"]
                           - np.nanmean([f[n]["skill_share"] for n in nulls])
                           for f in folds], dtype=float)
        wins = int((margin > 0).sum())
        return {"folds_beating_null": wins, "n_folds": len(margin),
                "median_margin": float(np.nanmedian(margin)),
                "sign_test_p": min(1.0, 2 * sum(
                    math.comb(len(margin), k)
                    for k in range(min(wins, len(margin) - wins) + 1)
                ) / 2 ** len(margin))}

    summary = {
        "question": "is whole-foot skill spatial resolution, or total-load tracking?",
        "features": args.features,
        "alpha": args.alpha,
        "n_folds": len(folds),
        "n_samples_total": int(X.shape[0]),
        "units": "baseline-corrected raw sensor counts (NOT kPa -- no calibration)",
        "evidence": "predicted (model output, participant-disjoint LOSO)",
        "min_denom_counts": MIN_DENOM_COUNTS,
        "null_repeats": len(nulls),
        "contrasts": {
            **{n: {"description": contrasts[n][2],
                   "n_a": int(contrasts[n][0].size), "n_b": int(contrasts[n][1].size),
                   "absolute": spread([n], "skill_absolute"),
                   "share": spread([n], "skill_share"),
                   "vs_null": vs_null(n)}
               for n in named},
            "null_control": {
                "description": f"{len(nulls)} arbitrary 16/16 sensor partitions, pooled",
                "n_a": 32, "n_b": 32,
                "absolute": spread(nulls, "skill_absolute"),
                "share": spread(nulls, "skill_share")},
        },
        "folds": folds,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"spatial_ladder_{args.features}.json"
    path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'contrast':16s} {'absolute':>17s} {'share':>17s} {'beats own null':>15s} {'p':>9s}")
    print("-" * 78)
    for n in named:
        c = summary["contrasts"][n]
        ab, sh, vn = c["absolute"], c["share"], c["vs_null"]
        print(f"{n:16s} {ab['mean']:+.3f} +/-{ab['sd']:.3f} {sh['mean']:+.3f} +/-{sh['sd']:.3f} "
              f"{vn['folds_beating_null']:8d}/{vn['n_folds']:<5d} {vn['sign_test_p']:9.2e}")
    nc = summary["contrasts"]["null_control"]
    print(f"{'null_control':16s} {nc['absolute']['mean']:+.3f} +/-{nc['absolute']['sd']:.3f} "
          f"{nc['share']['mean']:+.3f} +/-{nc['share']['sd']:.3f} {'--':>14s} {'--':>9s}")
    print("\nA contrast shows spatial resolution only if its *share* skill beats the\n"
          "controls computed on the same participant. Losing to an arbitrary sensor\n"
          "partition means the model is not resolving that anatomical axis.")
    print(f"\nwrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
