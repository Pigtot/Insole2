"""Limb kinematics from 2D body pose -- the UnderPressure-style feature set.

Why this exists
---------------
The coarse whole-body features in :mod:`visole.models.video_features` failed:
on held-out participants they gave a skill of +0.009 against a mean predictor
and *chance* accuracy (0.51 balanced) on stance-vs-swing. A diagnostic showed
the most-separating feature differed from clip to clip, which is the signature
of clip-specific coincidence rather than a transferable signal. A person-sized
bounding box simply cannot express which leg is loaded.

This module extracts per-limb keypoints instead -- ankles, knees, hips -- which
is the information UnderPressure uses to regress ground reaction force.

Scale normalisation is the critical part. Apparent size varies roughly sevenfold
along the walkway, so every spatial quantity is divided by the subject's own
torso length in that frame. Without this, the features encode distance from the
camera and the model learns the room, not the gait.

Model: torchvision Keypoint R-CNN (COCO, 17 keypoints). Benchmarked on this
machine at 0.104 s/frame on MPS with ``min_size=480`` versus 1.14 s/frame on CPU
at the default size -- MPS is worth using here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO keypoint indices used here.
KP_INDEX = {
    "nose": 0, "left_shoulder": 5, "right_shoulder": 6,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

POSE_FEATURE_NAMES = (
    "l_ankle_x", "l_ankle_y", "r_ankle_x", "r_ankle_y",
    "l_knee_x", "l_knee_y", "r_knee_x", "r_knee_y",
    "ankle_sep_x", "ankle_sep_y", "ankle_dist",
    "l_ankle_vx", "l_ankle_vy", "r_ankle_vx", "r_ankle_vy",
    "l_ankle_speed", "r_ankle_speed", "ankle_speed_ratio",
    "l_knee_angleish", "r_knee_angleish",
    "torso_len", "hip_y", "hip_vx", "hip_vy",
    "l_ankle_score", "r_ankle_score",
)


@dataclass(frozen=True)
class PoseSequence:
    times: np.ndarray        # (n,)
    keypoints: np.ndarray    # (n, 17, 2) pixels, NaN where no person found
    scores: np.ndarray       # (n, 17)
    detected: np.ndarray     # (n,) bool


def load_pose_model(device, min_size: int = 480, box_score_thresh: float = 0.7):
    """Load Keypoint R-CNN onto ``device``.

    Note for macOS framework Python: downloading the weights needs a CA bundle.
    Set ``SSL_CERT_FILE`` to ``certifi.where()`` first or the fetch fails with
    CERTIFICATE_VERIFY_FAILED.
    """
    from torchvision.models.detection import (
        KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn,
    )

    model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_score_thresh=box_score_thresh,
        min_size=min_size, max_size=int(min_size * 1.9),
    ).eval()
    return model.to(device)


def extract_clip_pose(clip, model, device, *, stride: int = 4, width: int = 960,
                      height: int = 544, max_frames: int | None = None) -> PoseSequence | None:
    """Run pose estimation over a clip, taking every ``stride`` frames."""
    import cv2
    import torch

    if clip.video_path is None or clip.video_info is None:
        return None
    fps = clip.video_info.fps
    cap = cv2.VideoCapture(str(clip.video_path))
    frames, idxs = [], []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                frames.append(cv2.resize(frame, (width, height)))
                idxs.append(i)
                if max_frames and len(frames) >= max_frames:
                    break
            i += 1
    finally:
        cap.release()
    if not frames:
        return None

    n = len(frames)
    kps = np.full((n, 17, 2), np.nan)
    scs = np.zeros((n, 17))
    det = np.zeros(n, dtype=bool)
    sx = clip.video_info.width / width          # back to original pixel scale
    sy = clip.video_info.height / height

    with torch.inference_mode():
        for j, frame in enumerate(frames):
            x = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = x.permute(2, 0, 1).float().div(255).to(device)
            out = model([x])[0]
            if len(out["boxes"]) == 0:
                continue
            # Highest-scoring detection: one participant walks the walkway.
            best = int(out["scores"].argmax())
            k = out["keypoints"][best].detach().cpu().numpy()[:, :2]
            kps[j] = k * np.array([sx, sy])
            scs[j] = out["keypoints_scores"][best].detach().cpu().numpy()
            det[j] = True

    return PoseSequence(times=np.asarray(idxs) / fps, keypoints=kps, scores=scs, detected=det)


def pose_to_features(seq: PoseSequence, fps_equivalent: float | None = None) -> np.ndarray:
    """Turn a pose sequence into scale-normalised kinematic features.

    All spatial quantities are expressed in units of the subject's torso length
    in the same frame, and measured relative to the hip midpoint, so a person
    near the camera and the same person far away produce comparable numbers.
    """
    kp, sc, det = seq.keypoints, seq.scores, seq.detected
    n = len(kp)
    feats = np.full((n, len(POSE_FEATURE_NAMES)), np.nan)

    hip = np.nanmean(kp[:, [KP_INDEX["left_hip"], KP_INDEX["right_hip"]], :], axis=1)
    sho = np.nanmean(kp[:, [KP_INDEX["left_shoulder"], KP_INDEX["right_shoulder"]], :], axis=1)
    torso = np.linalg.norm(sho - hip, axis=1)
    torso = np.where(torso > 1e-3, torso, np.nan)      # avoid dividing by ~0

    la = kp[:, KP_INDEX["left_ankle"], :]
    ra = kp[:, KP_INDEX["right_ankle"], :]
    lk = kp[:, KP_INDEX["left_knee"], :]
    rk = kp[:, KP_INDEX["right_knee"], :]

    def rel(p):
        return (p - hip) / torso[:, None]

    rla, rra, rlk, rrk = rel(la), rel(ra), rel(lk), rel(rk)
    dt = np.diff(seq.times, prepend=seq.times[0] - (seq.times[1] - seq.times[0] if n > 1 else 1))
    dt = np.where(dt > 1e-6, dt, np.nan)

    def vel(p):
        v = np.full_like(p, np.nan)
        v[1:] = (p[1:] - p[:-1]) / dt[1:, None]
        return v

    vla, vra = vel(rla), vel(rra)
    vhip = vel(hip / torso[:, None])

    feats[:, 0:2] = rla
    feats[:, 2:4] = rra
    feats[:, 4:6] = rlk
    feats[:, 6:8] = rrk
    feats[:, 8:10] = rla - rra
    feats[:, 10] = np.linalg.norm(rla - rra, axis=1)
    feats[:, 11:13] = vla
    feats[:, 13:15] = vra
    sl = np.linalg.norm(vla, axis=1)
    sr = np.linalg.norm(vra, axis=1)
    feats[:, 15] = sl
    feats[:, 16] = sr
    # Which foot is moving faster -- the most direct swing/stance cue available.
    feats[:, 17] = (sl - sr) / (sl + sr + 1e-6)
    feats[:, 18] = np.linalg.norm(rla - rlk, axis=1)
    feats[:, 19] = np.linalg.norm(rra - rrk, axis=1)
    feats[:, 20] = torso
    feats[:, 21] = hip[:, 1] / np.nanmax(np.abs(hip[:, 1]) + 1e-6)
    feats[:, 22:24] = vhip
    feats[:, 24] = sc[:, KP_INDEX["left_ankle"]]
    feats[:, 25] = sc[:, KP_INDEX["right_ankle"]]

    feats[~det] = np.nan
    return feats
