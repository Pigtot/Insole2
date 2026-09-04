"""Declarative registry of external datasets and model assets.

Every download Visole performs is declared here first, with its verified size,
checksum, licence and the reason Visole needs it. Nothing is fetched by an
ad-hoc URL buried in a script.

Sizes and checksums below were read from the providers' own metadata APIs on
2026-08-30 and are asserted against the downloaded bytes -- see
``scripts/download_data.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"

#: Downloads larger than this require an explicit opt-in flag (project policy).
LARGE_DOWNLOAD_THRESHOLD_BYTES = 5 * 1024**3


@dataclass(frozen=True)
class RemoteFile:
    key: str
    url: str
    size_bytes: int | None = None
    md5: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    files: tuple[RemoteFile, ...]
    dest: str
    licence: str
    source_url: str
    doi: str | None = None
    purpose: str = ""
    verified_on: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dest_path(self) -> Path:
        return DATA_ROOT / self.dest

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes or 0 for f in self.files)

    @property
    def is_large(self) -> bool:
        return self.total_bytes > LARGE_DOWNLOAD_THRESHOLD_BYTES


_ZENODO_19662017 = "https://zenodo.org/api/records/19662017/files/{key}/content"

INSOLE_GAITRITE = DatasetSpec(
    name="insole_gaitrite",
    title=(
        "An Open Insole-Based Plantar Pressure Dataset at Varying Cadences "
        "Compared Against GAITRite"
    ),
    doi="10.5281/zenodo.19662017",
    source_url="https://zenodo.org/records/19662017",
    licence="CC-BY-4.0",
    dest="raw/insole_gaitrite",
    verified_on="2026-08-30",
    purpose=(
        "Primary video+pressure feasibility dataset (RQ2). Synchronised anonymised "
        "video (30 fps), 32-channel instrumented insoles (64 Hz) and GAITRite "
        "reference (180 Hz) across slow/normal/fast walking, standing and sitting."
    ),
    files=(
        RemoteFile(
            key="insole-gaitrite-dataset.zip",
            url=_ZENODO_19662017.format(key="insole-gaitrite-dataset.zip"),
            size_bytes=1_686_755_134,
            md5="33063fb8b638c87085a157c44fc6eb1c",
        ),
        RemoteFile(
            key="sensors_map_left.svg",
            url=_ZENODO_19662017.format(key="sensors_map_left.svg"),
            size_bytes=60_822,
            md5="63df62e73e86aba36297309e8bab4420",
        ),
        RemoteFile(
            key="sensors_map_right.svg",
            url=_ZENODO_19662017.format(key="sensors_map_right.svg"),
            size_bytes=58_191,
            md5="e9584f8b290e6c80bcc9c74d116f446b",
        ),
    ),
    notes=(
        "Participants wore their OWN FOOTWEAR: the video shows shod feet, not bare "
        "plantar surfaces. This constrains what RGB can plausibly reveal about "
        "pressure and must be stated in every result derived from it.",
        "No subject-specific insole calibration was performed. Channel units are "
        "NOT established as kPa -- treat as raw sensor output until verified.",
        "Participant IDs are pseudonymous and NON-CONTIGUOUS (P1..P24, with gaps); "
        "23 participants were recorded originally and only those meeting quality "
        "criteria were released. Count participants, never assume the number.",
        "sync_auto.json and GAITRite CSVs exist only for walking clips (FP/NP/SP); "
        "STAND and SITDOWN clips have insole signals and video only.",
    ),
)

_COMMONS = "https://upload.wikimedia.org/wikipedia/commons"

SAMPLE_VIDEOS = DatasetSpec(
    name="sample_videos",
    title="Freely licensed gait video for demonstrating the pipeline on outside footage",
    doi=None,
    source_url="https://commons.wikimedia.org/wiki/Category:Gait",
    licence="CC-BY-2.0",
    dest="raw/sample_videos",
    verified_on="2026-09-03",
    purpose=(
        "A walking clip that is NOT from the training dataset, so the chain can be "
        "shown running end to end on footage anyone can fetch. Stages 1-3 and 5-8 "
        "are honest on any video; the loading stage is extrapolating and says so."
    ),
    files=(
        RemoteFile(
            key="treadmill_walk.ogv",
            url=(f"{_COMMONS}/1/1d/A-novel-walking-speed-estimation-scheme-and-its-"
                 "application-to-treadmill-control-for-gait-1743-0003-9-62-S1.ogv"),
            size_bytes=3_798_720,
            md5="a5913d51284fb8054d42062c7ecd5806",
        ),
    ),
    notes=(
        "ATTRIBUTION REQUIRED (CC BY 2.0). treadmill_walk.ogv is Additional file 1 of "
        "Yoon J, Park H-S, Damiano DL, 'A novel walking speed estimation scheme and its "
        "application to treadmill control for gait rehabilitation', Journal of "
        "NeuroEngineering and Rehabilitation 9:62 (2012), doi:10.1186/1743-0003-9-62, "
        "via Wikimedia Commons.",
        "Theora/.ogv at 720x480, 29.97 fps, 45 s. download_data.py transcodes a 10 s "
        "window to H.264 treadmill_walk.mp4, which is what the pipeline reads.",
        "OUT OF DISTRIBUTION on purpose. The loading model was trained on a fixed room "
        "camera with participants walking toward or away from it; this is a sagittal "
        "treadmill view of a different person. The camera also crops the head, so the "
        "torso-length normalisation the pose features rely on is degraded.",
    ),
)

REGISTRY: dict[str, DatasetSpec] = {d.name: d for d in (INSOLE_GAITRITE, SAMPLE_VIDEOS)}


def get(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]
