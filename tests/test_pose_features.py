"""Tests for pose-derived kinematic features.

These run without the pose network: they build synthetic PoseSequence objects,
which is the only way to test the maths deterministically.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.models.pose_features import (  # noqa: E402
    KP_INDEX, POSE_FEATURE_NAMES, PoseSequence, pose_to_features,
)

NAME_TO_COL = {n: i for i, n in enumerate(POSE_FEATURE_NAMES)}


def make_seq(n=10, scale=1.0, ankle_offset=(0.0, 0.0), fps=30.0):
    """A standing figure at a given apparent scale."""
    kp = np.zeros((n, 17, 2))
    for i in range(n):
        kp[i, KP_INDEX["left_shoulder"]] = (-10 * scale, -50 * scale)
        kp[i, KP_INDEX["right_shoulder"]] = (10 * scale, -50 * scale)
        kp[i, KP_INDEX["left_hip"]] = (-8 * scale, 0.0)
        kp[i, KP_INDEX["right_hip"]] = (8 * scale, 0.0)
        kp[i, KP_INDEX["left_knee"]] = (-8 * scale, 30 * scale)
        kp[i, KP_INDEX["right_knee"]] = (8 * scale, 30 * scale)
        kp[i, KP_INDEX["left_ankle"]] = ((-8 + ankle_offset[0]) * scale,
                                         (60 + ankle_offset[1]) * scale)
        kp[i, KP_INDEX["right_ankle"]] = (8 * scale, 60 * scale)
    return PoseSequence(times=np.arange(n) / fps, keypoints=kp,
                        scores=np.full((n, 17), 5.0), detected=np.ones(n, bool))


def test_feature_matrix_shape():
    f = pose_to_features(make_seq(12))
    assert f.shape == (12, len(POSE_FEATURE_NAMES))


def test_features_are_scale_invariant():
    """The critical property: apparent size varies ~7x along the walkway, so
    a near and a far subject in the same pose must give the same features."""
    near = pose_to_features(make_seq(scale=4.0))
    far = pose_to_features(make_seq(scale=1.0))
    # torso_len is deliberately raw (it encodes distance); everything else is normalised.
    cols = [i for n, i in NAME_TO_COL.items() if n != "torso_len"]
    assert np.allclose(near[5, cols], far[5, cols], atol=1e-9)


def test_torso_length_still_records_apparent_scale():
    near = pose_to_features(make_seq(scale=4.0))
    far = pose_to_features(make_seq(scale=1.0))
    assert near[5, NAME_TO_COL["torso_len"]] > far[5, NAME_TO_COL["torso_len"]]


def test_ankle_separation_tracks_a_moved_foot():
    base = pose_to_features(make_seq())
    stepped = pose_to_features(make_seq(ankle_offset=(-20.0, 0.0)))
    assert (stepped[5, NAME_TO_COL["ankle_dist"]]
            > base[5, NAME_TO_COL["ankle_dist"]])


def test_stationary_subject_has_near_zero_ankle_speed():
    f = pose_to_features(make_seq())
    assert abs(f[5, NAME_TO_COL["l_ankle_speed"]]) < 1e-6
    assert abs(f[5, NAME_TO_COL["r_ankle_speed"]]) < 1e-6


def test_ankle_speed_ratio_identifies_the_moving_foot():
    """The most direct swing/stance cue: one ankle moving, the other planted."""
    n = 8
    kp = make_seq(n).keypoints.copy()
    for i in range(n):
        kp[i, KP_INDEX["left_ankle"]] = (-8 + 4 * i, 60)   # left swings
    seq = PoseSequence(times=np.arange(n) / 30.0, keypoints=kp,
                       scores=np.full((n, 17), 5.0), detected=np.ones(n, bool))
    f = pose_to_features(seq)
    assert f[4, NAME_TO_COL["ankle_speed_ratio"]] > 0.9   # +1 => left is moving


def test_undetected_frames_are_nan_not_zero():
    """Zeros would be read as a real pose at the origin."""
    seq = make_seq(6)
    det = seq.detected.copy()
    det[2] = False
    seq = PoseSequence(seq.times, seq.keypoints, seq.scores, det)
    f = pose_to_features(seq)
    assert np.isnan(f[2]).all()
    assert np.isfinite(f[3]).any()


def test_zero_torso_does_not_produce_infinities():
    """A degenerate detection must yield NaN, never a huge finite number that
    would silently dominate a fitted model."""
    seq = make_seq(4)
    kp = seq.keypoints.copy()
    kp[:, KP_INDEX["left_shoulder"]] = kp[:, KP_INDEX["left_hip"]]
    kp[:, KP_INDEX["right_shoulder"]] = kp[:, KP_INDEX["right_hip"]]
    f = pose_to_features(PoseSequence(seq.times, kp, seq.scores, seq.detected))
    assert not np.isinf(f).any()


def test_keypoint_indices_match_coco_layout():
    assert KP_INDEX["left_ankle"] == 15 and KP_INDEX["right_ankle"] == 16
    assert KP_INDEX["left_hip"] == 11 and KP_INDEX["right_hip"] == 12
