"""Motion/kinematic features from the walkway video.

What these are
--------------
The camera is fixed, so a per-clip median frame is a good background estimate.
Subtracting it isolates the walking person; from that silhouette we take
position, size, and how much the *lower* part of the body (the feet) is moving.

What these are not
------------------
These are **not** plantar appearance features. The plantar surface is never
visible and every participant is shod (see the dataset audit). They describe
where the person is and how they are moving -- the UnderPressure premise
(motion carries loading information), not the PressureVision premise.

They are also not a pose estimator. Body-part keypoints would be a stronger
kinematic description; this is the cheap, dependency-free version that runs over
the whole corpus in minutes, and it establishes whether the signal is there at
all before anything heavier is justified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Names are stable and index-aligned with the feature matrix columns.
BASE_FEATURE_NAMES = (
    "cx", "cy", "bbox_h", "bbox_w", "bbox_area", "fg_fraction",
    "foot_motion", "body_motion", "cx_vel", "cy_vel", "h_vel",
    "foot_motion_d", "t_norm",
)


@dataclass(frozen=True)
class ClipFeatures:
    times: np.ndarray          # (n_frames,) seconds from clip start
    features: np.ndarray       # (n_frames, len(BASE_FEATURE_NAMES))
    names: tuple
    fps: float
    n_frames: int

    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.features).all(axis=1)


def extract_clip_features(clip, *, width: int = 320, height: int = 181,
                          fg_threshold: float = 25.0, foot_frac: float = 0.30,
                          max_frames: int | None = None) -> ClipFeatures | None:
    """Compute per-frame motion features for one clip.

    Decoding is done once at low resolution; the whole corpus fits in minutes.
    """
    import cv2

    if clip.video_path is None:
        return None
    info = clip.video_info
    if info is None or not info.fps:
        return None

    cap = cv2.VideoCapture(str(clip.video_path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (width, height))
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    if len(frames) < 3:
        return None

    stack = np.stack(frames)
    background = np.median(stack, axis=0)          # fixed camera -> median is the room
    foreground = np.abs(stack - background)

    n = len(stack)
    feats = np.full((n, len(BASE_FEATURE_NAMES)), np.nan)
    prev = None
    for i in range(n):
        mask = foreground[i] > fg_threshold
        fg_fraction = float(mask.mean())
        # Take the largest connected blob, not the global extent: scattered
        # lighting noise in the corners otherwise stretches the bounding box
        # across the whole frame and destroys every size and position feature.
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if n_lab < 2:
            prev = None
            continue
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[largest, cv2.CC_STAT_AREA] < 50:
            prev = None
            continue
        x0 = int(stats[largest, cv2.CC_STAT_LEFT])
        y0 = int(stats[largest, cv2.CC_STAT_TOP])
        x1 = x0 + int(stats[largest, cv2.CC_STAT_WIDTH]) - 1
        y1 = y0 + int(stats[largest, cv2.CC_STAT_HEIGHT]) - 1
        cx = float(centroids[largest][0]) / width
        cy = float(centroids[largest][1]) / height
        fg_fraction = float(stats[largest, cv2.CC_STAT_AREA]) / (width * height)
        bbox_h = (y1 - y0) / height
        bbox_w = (x1 - x0) / width

        foot_motion = body_motion = np.nan
        if i > 0:
            diff = np.abs(stack[i] - stack[i - 1])
            band = int(y1 - (y1 - y0) * foot_frac)
            foot_region = diff[band:y1 + 1, x0:x1 + 1]
            body_region = diff[y0:y1 + 1, x0:x1 + 1]
            foot_motion = float(foot_region.mean()) if foot_region.size else np.nan
            body_motion = float(body_region.mean()) if body_region.size else np.nan

        row = [cx, cy, bbox_h, bbox_w, bbox_h * bbox_w, fg_fraction,
               foot_motion, body_motion, np.nan, np.nan, np.nan, np.nan,
               i / max(n - 1, 1)]
        if prev is not None:
            row[8] = (cx - prev[0]) * info.fps
            row[9] = (cy - prev[1]) * info.fps
            row[10] = (bbox_h - prev[2]) * info.fps
            if np.isfinite(foot_motion) and np.isfinite(prev[6]):
                row[11] = (foot_motion - prev[6]) * info.fps
        feats[i] = row
        prev = row

    times = np.arange(n) / info.fps
    return ClipFeatures(times=times, features=feats, names=BASE_FEATURE_NAMES,
                        fps=float(info.fps), n_frames=n)


def add_temporal_context(features: np.ndarray, offsets=(-6, -3, 0, 3, 6)) -> np.ndarray:
    """Stack features from neighbouring frames.

    Instantaneous appearance cannot express gait phase; a short window can. Edges
    are handled by clamping to the first/last valid index rather than zero-padding,
    which would inject fake stationarity at clip boundaries.
    """
    n = features.shape[0]
    idx = np.arange(n)
    cols = [features[np.clip(idx + o, 0, n - 1)] for o in offsets]
    return np.concatenate(cols, axis=1)


def context_feature_names(names=BASE_FEATURE_NAMES, offsets=(-6, -3, 0, 3, 6)) -> tuple:
    return tuple(f"{nm}@{o:+d}" for o in offsets for nm in names)
