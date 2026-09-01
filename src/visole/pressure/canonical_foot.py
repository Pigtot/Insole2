"""Canonical plantar coordinates -- one frame that every pressure source maps into.

Motivation
----------
Three recent hand-pressure papers converge on the same structural idea:

* **HOPE** (arXiv 2608.06192) predicts per-vertex pressure on the MANO mesh
  rather than in sensor coordinates, and lifts three *heterogeneous* supervision
  sources -- a 16x16 tactile glove, a planar Sensel array, and distance-based
  contact from interaction datasets -- into that one shared vertex space.
* **EgoPressDiff** predicts pressure in a **UV** map: the surface unwrapped to a
  2D texture, later warped back onto the 3D mesh.
* **WristPP** regresses per-vertex pressure on the hand mesh.

None of them predict in camera or sensor coordinates. The reason is that a
shared body-centred frame is what lets different sensors, viewpoints and
subjects be compared and combined at all.

Visole needs exactly this. Its sources will be:

======================  ==========================================
source                  nature
======================  ==========================================
insole (now)            32 sparse channels, raw counts, per foot
optical pedobarograph   dense, calibrated -- future hardware
video-estimated         predicted, not measured
synthetic               simulated
======================  ==========================================

The canonical frame is a normalised plantar rectangle:

* ``u`` in [0, 1]: **medial (0) to lateral (1)**
* ``v`` in [0, 1]: **heel (0) to toe (1)**

Both feet map into the *same* frame, which is only correct because the
left/right sensor correspondence was established geometrically -- see
:mod:`visole.pressure.sensor_map`. The two insoles do not share numbering, so a
naive mapping would mirror one foot's anatomy.

Honesty rule
------------
Lifting 32 sparse channels onto a dense grid does **not** create measurements.
:func:`lift_to_canonical` therefore returns a ``support`` mask marking cells that
actually have a nearby sensor. Everything outside it is interpolation, and any
loss computed over the grid must be masked by it -- the same discipline HOPE uses
when it masks its pressure loss on contact-only samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sensor_map import SensorMap, load_sensor_map

DEFAULT_GRID = (24, 56)          # (u, v) -- feet are much longer than wide

#: Plantar regions in canonical v (heel->toe). Geometric working definition.
CANONICAL_REGIONS = (
    ("heel", 0.00, 0.28),
    ("midfoot", 0.28, 0.55),
    ("metatarsal", 0.55, 0.82),
    ("toes", 0.82, 1.01),
)


@dataclass(frozen=True)
class CanonicalField:
    """A pressure field in canonical plantar coordinates."""

    values: np.ndarray        # (nu, nv)
    support: np.ndarray       # (nu, nv) bool -- True where a sensor is nearby
    units: str
    source: str               # measured_sensor | estimated_video | optical_measured | synthetic
    side: str

    def __post_init__(self) -> None:
        if self.values.shape != self.support.shape:
            raise ValueError("values and support must have the same shape")
        valid = {"measured_sensor", "estimated_video", "optical_measured", "synthetic"}
        if self.source not in valid:
            raise ValueError(f"source must be one of {sorted(valid)}, got {self.source!r}")

    @property
    def supported_fraction(self) -> float:
        return float(self.support.mean())

    def masked(self) -> np.ndarray:
        """Values with unsupported cells masked -- safe to display or score."""
        return np.ma.masked_where(~self.support, self.values)

    def region_means(self) -> dict[str, float]:
        """Mean over supported cells per plantar region."""
        nu, nv = self.values.shape
        v = (np.arange(nv) + 0.5) / nv
        out = {}
        for name, lo, hi in CANONICAL_REGIONS:
            band = (v >= lo) & (v < hi)
            sel = self.support[:, band]
            vals = self.values[:, band]
            out[name] = float(vals[sel].mean()) if sel.any() else float("nan")
        return out


def canonical_grid(grid: tuple[int, int] = DEFAULT_GRID) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre (u, v) coordinate arrays of shape ``grid``."""
    nu, nv = grid
    u = (np.arange(nu) + 0.5) / nu
    v = (np.arange(nv) + 0.5) / nv
    return np.meshgrid(u, v, indexing="ij")


def sensor_uv(smap: SensorMap) -> np.ndarray:
    """(32, 2) sensor positions in canonical (u, v).

    ``SensorMap.norm_xy`` is already normalised medial->lateral and heel->toe for
    both feet, so this is the identity plus a name. It exists so callers depend
    on the canonical contract rather than on the sensor map's internals.
    """
    return smap.norm_xy.copy()


def lift_to_canonical(values: np.ndarray, smap: SensorMap, *,
                      grid: tuple[int, int] = DEFAULT_GRID,
                      support_radius: float = 0.16,
                      power: float = 2.0,
                      units: str = "baseline_corrected_sensor_counts",
                      source: str = "measured_sensor") -> CanonicalField:
    """Lift 32 sparse channels onto the canonical plantar grid.

    Inverse-distance weighting, with a ``support`` mask marking cells within
    ``support_radius`` (in canonical units) of a real sensor. This mirrors HOPE's
    treatment of a 16x16 taxel grid: spread each reading over nearby surface, then
    keep an explicit record of where the data actually is.

    The interpolated values outside ``support`` are **not measurements** and must
    not be scored or reported as such.
    """
    values = np.asarray(values, dtype=float)
    if values.shape != (len(smap.xy),):
        raise ValueError(f"expected {len(smap.xy)} channel values, got {values.shape}")
    if not 0 < support_radius <= 1.5:
        raise ValueError("support_radius should be a small canonical distance")

    U, V = canonical_grid(grid)
    uv = sensor_uv(smap)
    d2 = (U[..., None] - uv[:, 0]) ** 2 + (V[..., None] - uv[:, 1]) ** 2
    w = 1.0 / np.power(d2 + 1e-9, power / 2.0)
    field = (w * values).sum(-1) / w.sum(-1)
    support = np.sqrt(d2.min(-1)) <= support_radius
    return CanonicalField(values=field, support=support, units=units,
                          source=source, side=smap.side)


def canonical_to_sensors(field: CanonicalField, smap: SensorMap) -> np.ndarray:
    """Sample a canonical field back at the 32 sensor locations.

    Round-tripping ``lift -> sample`` is a useful self-check: it should return
    close to the original readings, and :func:`lift_to_canonical` is only as
    trustworthy as that round trip.
    """
    nu, nv = field.values.shape
    uv = sensor_uv(smap)
    iu = np.clip((uv[:, 0] * nu).astype(int), 0, nu - 1)
    iv = np.clip((uv[:, 1] * nv).astype(int), 0, nv - 1)
    return field.values[iu, iv]


def both_feet_to_canonical(left_values: np.ndarray, right_values: np.ndarray,
                           **kwargs) -> tuple[CanonicalField, CanonicalField]:
    """Lift both feet into the same canonical frame.

    Correct only because the left/right sensor correspondence is established
    geometrically. ``SensorMap.norm_xy`` already places homologous sensors at
    identical canonical coordinates (verified to 0.0 difference), so anatomy --
    not sensor index -- decides where a value lands.
    """
    return (lift_to_canonical(left_values, load_sensor_map("left"), **kwargs),
            lift_to_canonical(right_values, load_sensor_map("right"), **kwargs))
