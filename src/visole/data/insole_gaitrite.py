"""Loader and index for the Insole-GAITRite dataset (Zenodo 10.5281/zenodo.19662017).

Audited facts this module encodes (see experiments/gait_dataset_audit/README.md)
------------------------------------------------------------------------------
* 22 participants are released (P1..P24 with P18 and P22 absent), 794 clips:
  SP 228, NP 219, FP 217, STAND 65, SITDOWN 65. Never hardcode 23.
* Video is 1920x1088. **Frame rate varies per clip** -- both ~60 fps and
  variable ~28.9-30 fps clips exist. The Zenodo description's "30 fps" is
  nominal. Always read fps from the file; never assume.
* Insole CSVs have no timestamp column. Sampling is nominally 64 Hz and clips
  are pre-trimmed so insole duration matches video duration within ~1.5%.
* Pressure channels are integers with a **non-zero unloaded baseline that
  varies per clip** (5th percentile ranged 19..187 across sampled files).
  Baseline must be estimated per clip, not assumed constant.
* ``sumP`` is exactly the raw sum of the 32 channels *including* that baseline
  (verified to 0.000 residual). It is not a zeroed load measure.
* Units are raw sensor counts. NOT kPa. No subject-specific calibration was
  performed and participants wore their own footwear.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "data" / "raw" / "insole_gaitrite" / "dataset"

N_SENSORS = 32
PRESSURE_COLUMNS = [f"PressureSensor {i}" for i in range(N_SENSORS)]
IMU_COLUMNS = ["AccelerometerX", "AccelerometerY", "AccelerometerZ",
               "GyroscopeX", "GyroscopeY", "GyroscopeZ"]
DERIVED_COLUMNS = ["copX", "copY", "sumP"]

WALKING_CONDITIONS = ("SP", "NP", "FP")
POSTURE_CONDITIONS = ("STAND", "SITDOWN")
ALL_CONDITIONS = WALKING_CONDITIONS + POSTURE_CONDITIONS

#: Nominal insole rate (Hz). The files carry no timestamps, so this is the
#: providers' stated rate, not a measured one.
NOMINAL_INSOLE_HZ = 64.0

#: Metronome-guided cadences (steps/min); NP is self-selected.
CONDITION_CADENCE = {"SP": 70.0, "NP": None, "FP": 127.0}


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float


@dataclass(frozen=True)
class Clip:
    """One segmented trial: video + both insoles (+ GAITRite for walking)."""

    participant: str
    condition: str
    clip_id: str
    directory: Path

    @property
    def key(self) -> str:
        return f"{self.participant}/{self.condition}/{self.clip_id}"

    @property
    def is_walking(self) -> bool:
        return self.condition in WALKING_CONDITIONS

    @cached_property
    def video_path(self) -> Path | None:
        vids = sorted(self.directory.glob("*.mp4"))
        return vids[0] if vids else None

    @property
    def left_csv(self) -> Path:
        return self.directory / "L.csv"

    @property
    def right_csv(self) -> Path:
        return self.directory / "R.csv"

    @property
    def sync_path(self) -> Path | None:
        p = self.directory / "sync_auto.json"
        return p if p.exists() else None

    @property
    def gaitrite_path(self) -> Path | None:
        p = self.directory / "gaitrite_test.csv"
        return p if p.exists() else None

    def has_all_modalities(self) -> bool:
        return bool(self.video_path) and self.left_csv.exists() and self.right_csv.exists()

    # -- signals ----------------------------------------------------------
    def insole(self, side: str) -> pd.DataFrame:
        """Raw insole dataframe for 'left' or 'right'."""
        path = self.left_csv if side.lower().startswith("l") else self.right_csv
        return pd.read_csv(path)

    def pressure(self, side: str, baseline_correct: bool = False) -> np.ndarray:
        """(n_samples, 32) array of raw sensor counts.

        With ``baseline_correct=True`` the per-sensor unloaded offset is
        removed (see :func:`estimate_baseline`) and values are clipped at 0.
        The result is still raw counts, only zeroed -- never call it kPa.
        """
        arr = self.insole(side)[PRESSURE_COLUMNS].to_numpy(dtype=float)
        if baseline_correct:
            arr = np.clip(arr - estimate_baseline(arr), 0.0, None)
        return arr

    @cached_property
    def video_info(self) -> VideoInfo | None:
        return probe_video(self.video_path) if self.video_path else None

    def sync(self) -> dict | None:
        if self.sync_path is None:
            return None
        return json.loads(self.sync_path.read_text())

    def gaitrite(self) -> pd.DataFrame | None:
        """GAITRite per-footfall events.

        Semicolon-delimited with embedded newlines inside quoted fields, so it
        needs a real CSV parser -- a line-based read silently corrupts it.
        """
        if self.gaitrite_path is None:
            return None
        with self.gaitrite_path.open(newline="", encoding="utf-8", errors="replace") as fh:
            rows = list(csv.reader(fh, delimiter=";", quotechar='"'))
        if not rows:
            return None
        header, *body = rows
        body = [r for r in body if len(r) == len(header)]
        return pd.DataFrame(body, columns=header)

    def insole_duration_s(self, side: str = "left") -> float:
        return len(self.insole(side)) / NOMINAL_INSOLE_HZ

    def iter_frames(self, stride: int = 1, max_frames: int | None = None):
        """Stream video frames lazily. Never loads a whole clip into RAM."""
        import cv2

        if self.video_path is None:
            return
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            idx = 0
            yielded = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    yield idx, frame
                    yielded += 1
                    if max_frames and yielded >= max_frames:
                        break
                idx += 1
        finally:
            cap.release()

    def frame_at_time(self, t_s: float):
        """Read the single frame nearest ``t_s`` seconds into the clip."""
        import cv2

        if self.video_path is None:
            return None
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(t_s, 0.0) * 1000.0)
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    def pressure_time(self, side: str = "left") -> np.ndarray:
        """Sample times in seconds from clip start, at the nominal rate."""
        return np.arange(len(self.insole(side))) / NOMINAL_INSOLE_HZ


def estimate_baseline(arr: np.ndarray, percentile: float = 5.0) -> np.ndarray:
    """Per-sensor unloaded offset for one clip.

    A low percentile of each channel over the clip: for walking, every sensor
    spends much of the clip unloaded. For STAND this assumption is weak (the
    foot never leaves the ground), so treat standing baselines with suspicion.
    """
    if arr.ndim != 2 or arr.shape[1] != N_SENSORS:
        raise ValueError(f"expected (n, {N_SENSORS}) array, got {arr.shape}")
    return np.percentile(arr, percentile, axis=0)


def probe_video(path: str | Path) -> VideoInfo | None:
    """Read true video properties. fps varies per clip, so always measure."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json",
             str(path)],
            capture_output=True, text=True, timeout=60,
        )
        j = json.loads(out.stdout)
        st = j["streams"][0]
        num, den = st["r_frame_rate"].split("/")
        fps = float(num) / float(den)
        n = int(st.get("nb_frames") or 0)
        dur = float(j["format"]["duration"])
        return VideoInfo(path, int(st["width"]), int(st["height"]), fps, n, dur)
    except (FileNotFoundError, KeyError, ValueError, IndexError, json.JSONDecodeError):
        pass
    try:  # fallback if ffprobe is unavailable
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return VideoInfo(path, w, h, fps, n, (n / fps) if fps else 0.0)
    except Exception:
        return None


_PID = re.compile(r"^P(\d+)$")


class InsoleGaitRite:
    """Index over the extracted dataset tree."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else DEFAULT_ROOT
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.root} not found. Run: "
                "envs/core/bin/python scripts/download_data.py insole_gaitrite --yes"
            )
        self.clips: list[Clip] = self._index()

    def _index(self) -> list[Clip]:
        clips: list[Clip] = []
        for pdir in sorted(self.root.iterdir(), key=_participant_sort_key):
            if not (pdir.is_dir() and _PID.match(pdir.name)):
                continue
            for cdir in sorted(pdir.iterdir()):
                if not cdir.is_dir() or cdir.name not in ALL_CONDITIONS:
                    continue
                for kdir in sorted(cdir.iterdir(), key=lambda p: _int_or_inf(p.name)):
                    if kdir.is_dir() and kdir.name.isdigit():
                        clips.append(Clip(pdir.name, cdir.name, kdir.name, kdir))
        return clips

    # -- views -------------------------------------------------------------
    @property
    def participants(self) -> list[str]:
        return sorted({c.participant for c in self.clips}, key=lambda s: int(s[1:]))

    def filter(self, participants: Sequence[str] | None = None,
               conditions: Sequence[str] | None = None,
               require_all_modalities: bool = False) -> list[Clip]:
        out = self.clips
        if participants is not None:
            keep = set(participants)
            out = [c for c in out if c.participant in keep]
        if conditions is not None:
            keep = set(conditions)
            out = [c for c in out if c.condition in keep]
        if require_all_modalities:
            out = [c for c in out if c.has_all_modalities()]
        return out

    def get(self, participant: str, condition: str, clip_id: str | int) -> Clip:
        key = f"{participant}/{condition}/{clip_id}"
        for c in self.clips:
            if c.key == key:
                return c
        raise KeyError(f"no clip {key}")

    def counts_by_condition(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.clips:
            out[c.condition] = out.get(c.condition, 0) + 1
        return out

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self) -> Iterator[Clip]:
        return iter(self.clips)


def _participant_sort_key(p: Path):
    m = _PID.match(p.name)
    return (0, int(m.group(1))) if m else (1, 0)


def _int_or_inf(s: str) -> float:
    return int(s) if s.isdigit() else float("inf")
