"""Sensor geometry for the Insole-GAITRite instrumented insoles.

Parses the dataset's own ``sensors_map_{left,right}.svg`` into sensor positions,
so sensor locations come from the providers' drawing rather than from a
hand-typed table.

What the SVGs actually contain (audited 2026-08-30)
---------------------------------------------------
* Both files were exported from the same ``Insole_RightSensors-brd.dxf``. The
  *left* file draws it inside a group with ``matrix(-1,0,0,1,613.96011,0)`` --
  a horizontal mirror. That is the anatomically correct way to make a left
  insole, but it means the raw path coordinates in the left file are pre-mirror
  and MUST be transformed. The numeric text labels sit outside that group and
  are already in final coordinates.
* Both files' ``sodipodi:docname`` is swapped relative to the filename and both
  descriptions say "RightSensors". These are Inkscape save-as leftovers. The
  geometry, not the metadata, establishes handedness (see below).
* Each file contains 32 closed pad outlines of near-identical area (~3157 svg
  units^2), recovered by chaining the 262 open Bezier segments end-to-end, plus
  an open insole outline. Sensor centroids come from those pads; the text
  labels only assign the index (a clean bijection, nearest match <=27 units vs
  a runner-up >=49 units).

The finding that matters most
-----------------------------
The left and right insoles use **different sensor numbering**. The pad geometry
is a pixel-exact mirror (max deviation 0.00 units), but not one of the 32
indices denotes the same anatomical site on both feet. ``LEFT_RIGHT_CORRESPONDENCE``
records the true mapping. Never assume ``left[i]`` and ``right[i]`` are
homologous -- they never are.

Units
-----
Positions are SVG user units, and normalised foot coordinates derived from
them. They are **not** millimetres: the drawing has no verified physical scale,
and the cohort wore insoles spanning EU 36-43, so one map cannot be
millimetre-true for every participant.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

SVG_NS = "{http://www.w3.org/2000/svg}"
N_SENSORS = 32

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAP_DIR = REPO_ROOT / "data" / "raw" / "insole_gaitrite"

#: Right sensor index -> left sensor index at the mirrored (homologous) site.
#: Derived geometrically; the match is exact (0.00 units) and bijective.
LEFT_RIGHT_CORRESPONDENCE: dict[int, int] = {
    0: 15, 1: 12, 2: 13, 3: 14, 4: 10, 5: 8, 6: 11, 7: 9,
    8: 4, 9: 5, 10: 6, 11: 7, 12: 19, 13: 1, 14: 2, 15: 3,
    16: 30, 17: 28, 18: 29, 19: 31, 20: 24, 21: 25, 22: 26, 23: 27,
    24: 20, 25: 21, 26: 22, 27: 23, 28: 16, 29: 18, 30: 17, 31: 0,
}

#: Boundaries along the normalised heel->toe axis. Geometric, not clinical:
#: chosen from the sensor layout (dense wide forefoot band, a single narrow
#: lateral midfoot column, a heel cluster). Treat as a working definition.
REGION_BOUNDS: tuple[tuple[str, float, float], ...] = (
    ("heel", 0.00, 0.28),
    ("midfoot", 0.28, 0.55),
    ("metatarsal", 0.55, 0.82),
    ("toes", 0.82, 1.01),
)


# --------------------------------------------------------------------------
# minimal SVG path handling (only M/m L/l C/c Z/z occur in these files)
# --------------------------------------------------------------------------
_TOKENS = re.compile(r"[MmLlCcZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _flatten_path(d: str, samples: int = 8) -> list[tuple[float, float]]:
    """Flatten a path's ``d`` attribute to a polyline."""
    toks = _TOKENS.findall(d)
    i = 0
    cur = (0.0, 0.0)
    pts: list[tuple[float, float]] = []
    cmd: str | None = None

    def num() -> float:
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    while i < len(toks):
        if re.fullmatch(r"[MmLlCcZz]", toks[i]):
            cmd = toks[i]
            i += 1
            if cmd in ("Z", "z"):
                continue
        if cmd in ("M", "m"):
            x, y = num(), num()
            cur = (x, y) if cmd == "M" else (cur[0] + x, cur[1] + y)
            pts.append(cur)
            cmd = "L" if cmd == "M" else "l"  # subsequent pairs are implicit lineto
        elif cmd in ("L", "l"):
            x, y = num(), num()
            cur = (x, y) if cmd == "L" else (cur[0] + x, cur[1] + y)
            pts.append(cur)
        elif cmd in ("C", "c"):
            p0 = cur
            ctrl = []
            for _ in range(3):
                x, y = num(), num()
                ctrl.append((x, y) if cmd == "C" else (cur[0] + x, cur[1] + y))
            p1, p2, p3 = ctrl
            for k in range(1, samples + 1):
                s = k / samples
                u = 1 - s
                pts.append((
                    u**3 * p0[0] + 3 * u * u * s * p1[0] + 3 * u * s * s * p2[0] + s**3 * p3[0],
                    u**3 * p0[1] + 3 * u * u * s * p1[1] + 3 * u * s * s * p2[1] + s**3 * p3[1],
                ))
            cur = p3
        else:
            i += 1
    return pts


def _polygon_area(pts) -> float:
    a = 0.0
    for k in range(len(pts)):
        x1, y1 = pts[k]
        x2, y2 = pts[(k + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _polygon_centroid(pts) -> tuple[float, float]:
    a = cx = cy = 0.0
    for k in range(len(pts)):
        x1, y1 = pts[k]
        x2, y2 = pts[(k + 1) % len(pts)]
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a) < 1e-9:  # degenerate: fall back to the mean point
        arr = np.asarray(pts)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    a *= 0.5
    return cx / (6 * a), cy / (6 * a)


def _chain_into_loops(segments, tol: float = 1.0):
    """Join open segments end-to-end into maximal chains."""
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    used = [False] * len(segments)
    chains = []
    for i, seg in enumerate(segments):
        if used[i]:
            continue
        used[i] = True
        chain = list(seg)
        grew = True
        while grew:
            grew = False
            for j, other in enumerate(segments):
                if used[j]:
                    continue
                if dist(chain[-1], other[0]) < tol:
                    chain += other[1:]
                elif dist(chain[-1], other[-1]) < tol:
                    chain += other[::-1][1:]
                elif dist(chain[0], other[-1]) < tol:
                    chain = other[:-1] + chain
                elif dist(chain[0], other[0]) < tol:
                    chain = other[::-1][:-1] + chain
                else:
                    continue
                used[j] = True
                grew = True
        chains.append(chain)
    closed = [c for c in chains if dist(c[0], c[-1]) < tol and _polygon_area(c) > 100.0]
    open_ = [c for c in chains if c not in closed]
    return closed, open_


#: Subtrees that define reusable templates rather than drawn content. Paths
#: inside them are not part of the insole and must not be chained into outlines
#: (doing so scatters stray marks across every plot).
_NON_RENDERED = {SVG_NS + t for t in ("defs", "symbol", "marker", "pattern", "clipPath")}


def _renderable_paths(node):
    """Yield <path> elements that are actually drawn, skipping template defs."""
    for child in node:
        if child.tag in _NON_RENDERED:
            continue
        if child.tag == SVG_NS + "path":
            yield child
        else:
            yield from _renderable_paths(child)


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorMap:
    """Sensor geometry for one insole.

    Attributes
    ----------
    side : 'left' or 'right'
    xy   : (32, 2) pad centroids in SVG user units (y increases downward)
    norm_xy : (32, 2) normalised foot coordinates --
        x in [0,1] medial(0) -> lateral(1); y in [0,1] heel(0) -> toe(1)
    pads : list of 32 closed polygons, index-aligned with `xy`
    """

    side: str
    xy: np.ndarray
    norm_xy: np.ndarray
    pads: tuple
    outline: tuple
    source: Path

    def __post_init__(self) -> None:
        if self.xy.shape != (N_SENSORS, 2):
            raise ValueError(f"expected ({N_SENSORS}, 2) positions, got {self.xy.shape}")

    def regions(self) -> list[str]:
        """Coarse plantar region per sensor, from the normalised heel->toe axis."""
        out = []
        for _, y in self.norm_xy:
            for name, lo, hi in REGION_BOUNDS:
                if lo <= y < hi:
                    out.append(name)
                    break
            else:  # pragma: no cover - bounds cover [0,1]
                out.append("unknown")
        return out

    def region_indices(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {name: [] for name, _, _ in REGION_BOUNDS}
        for i, r in enumerate(self.regions()):
            groups.setdefault(r, []).append(i)
        return groups

    def to_other_foot(self, values: np.ndarray) -> np.ndarray:
        """Reindex a 32-vector to the other foot's numbering, site-for-site.

        This is the ONLY correct way to compare or pool the two feet.
        """
        values = np.asarray(values)
        if values.shape[-1] != N_SENSORS:
            raise ValueError(f"expected last axis {N_SENSORS}, got {values.shape}")
        out = np.empty_like(values)
        if self.side == "right":
            for r, l in LEFT_RIGHT_CORRESPONDENCE.items():
                out[..., l] = values[..., r]
        else:
            for r, l in LEFT_RIGHT_CORRESPONDENCE.items():
                out[..., r] = values[..., l]
        return out


def parse_sensor_map_svg(path: str | Path, side: str | None = None) -> SensorMap:
    """Parse one ``sensors_map_*.svg`` into a :class:`SensorMap`."""
    path = Path(path)
    if side is None:
        side = "left" if "left" in path.stem.lower() else "right"
    root = ET.parse(path).getroot()

    # A mirror transform may sit on the group holding the paths (left file only).
    transform = None
    for g in root.iter(SVG_NS + "g"):
        t = g.get("transform")
        if t and any(c.tag == SVG_NS + "path" for c in g):
            vals = [float(v) for v in re.findall(r"[-+]?[\d.]+", t)]
            if len(vals) == 6:
                transform = vals

    segments = []
    for p in _renderable_paths(root):
        pts = _flatten_path(p.get("d") or "")
        if transform:
            a, b, c, d, e, f = transform
            pts = [(a * x + c * y + e, b * x + d * y + f) for x, y in pts]
        if len(pts) >= 2:
            segments.append(pts)

    pads, outline = _chain_into_loops(segments)
    if len(pads) != N_SENSORS:
        raise ValueError(
            f"{path.name}: recovered {len(pads)} pad outlines, expected {N_SENSORS}. "
            "The SVG structure may have changed; re-audit before trusting positions."
        )
    centroids = [_polygon_centroid(p) for p in pads]

    # Text labels carry the sensor index; they are outside any mirrored group.
    labels: dict[int, tuple[float, float]] = {}
    for t in root.iter(SVG_NS + "text"):
        txt = "".join(ts.text or "" for ts in t.iter(SVG_NS + "tspan")).strip()
        if not txt.isdigit():
            continue
        anchored = [e for e in t.iter(SVG_NS + "tspan") if e.get("x")]
        el = anchored[0] if anchored else t
        labels[int(txt)] = (float(el.get("x")), float(el.get("y")))

    if sorted(labels) != list(range(N_SENSORS)):
        raise ValueError(f"{path.name}: labels {sorted(labels)} != 0..{N_SENSORS - 1}")

    xy = np.zeros((N_SENSORS, 2))
    claimed: set[int] = set()
    for sid, lab in labels.items():
        k = min(range(len(centroids)),
                key=lambda k: math.hypot(centroids[k][0] - lab[0], centroids[k][1] - lab[1]))
        if k in claimed:
            raise ValueError(f"{path.name}: pad {k} claimed by two labels; assignment ambiguous")
        claimed.add(k)
        xy[sid] = centroids[k]

    return SensorMap(
        side=side,
        xy=xy,
        norm_xy=_normalise(xy, side),
        pads=tuple(np.asarray(p) for p in pads),
        outline=tuple(np.asarray(o) for o in outline),
        source=path,
    )


def _normalise(xy: np.ndarray, side: str) -> np.ndarray:
    """Map SVG units to normalised foot coordinates.

    y: SVG y grows downward and the forefoot is at small y, so heel->toe is a
       flip. x: for the *right* map, small x is medial (the midfoot sensors sit
       on the large-x lateral border, because the medial arch does not contact).
       The left map is its mirror, so its x axis runs the other way.
    """
    x, y = xy[:, 0], xy[:, 1]
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    nx = (x - x0) / (x1 - x0)
    ny = 1.0 - (y - y0) / (y1 - y0)  # heel (large svg y) -> 0, toe -> 1
    if side == "left":
        nx = 1.0 - nx  # so 0 = medial, 1 = lateral on both feet
    return np.column_stack([nx, ny])


@lru_cache(maxsize=4)
def load_sensor_map(side: str, map_dir: str | Path | None = None) -> SensorMap:
    """Load the left or right sensor map from the downloaded dataset."""
    side = side.lower()
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    directory = Path(map_dir) if map_dir else DEFAULT_MAP_DIR
    return parse_sensor_map_svg(directory / f"sensors_map_{side}.svg", side=side)
