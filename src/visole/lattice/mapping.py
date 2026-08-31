"""Pressure -> material-parameter mappings, as competing hypotheses.

Whether a high-pressure region of an insole should become *stiffer* (to spread
load) or *softer* (to cushion the peak) is an open engineering question. This
project does not assume an answer. Both directions are implemented, they are
selected by name, and the choice is recorded in every export so that a result
can never be quietly attributed to the wrong mapping.

Nothing here predicts an outcome. These functions only convert a normalised
pressure field into a lattice parameter; whether the resulting insole actually
redistributes load is a question for physical testing that has not been done.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Hypotheses about how local pressure should drive local material properties.
MAPPINGS = {
    "stiffen": "higher pressure -> higher relative density (spread the load)",
    "soften": "higher pressure -> lower relative density (cushion the peak)",
    "uniform": "ignore pressure -> constant density (control condition)",
}


@dataclass(frozen=True)
class DensityMapping:
    """Maps normalised pressure in [0,1] to relative density."""

    hypothesis: str = "stiffen"
    #: Below rho ~= 0.19 the gyroid sheet stops being a single connected solid
    #: and breaks into thousands of islands -- measured by voxel-counting
    #: connectivity at cell_size 8 mm (see docs/lattice_stack_decision.md).
    #: Keep rho_min above that floor or the part is unprintable.
    rho_min: float = 0.20
    rho_max: float = 0.55
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.hypothesis not in MAPPINGS:
            raise ValueError(f"unknown hypothesis {self.hypothesis!r}; have {sorted(MAPPINGS)}")
        if not 0 < self.rho_min < self.rho_max < 1:
            raise ValueError("need 0 < rho_min < rho_max < 1")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")

    def __call__(self, normalised_pressure) -> np.ndarray:
        p = np.clip(np.asarray(normalised_pressure, dtype=float), 0.0, 1.0)
        shaped = p ** self.gamma
        if self.hypothesis == "uniform":
            return np.full_like(p, 0.5 * (self.rho_min + self.rho_max))
        if self.hypothesis == "soften":
            shaped = 1.0 - shaped
        return self.rho_min + (self.rho_max - self.rho_min) * shaped

    def describe(self) -> str:
        return (f"{self.hypothesis}: {MAPPINGS[self.hypothesis]} "
                f"(rho {self.rho_min}..{self.rho_max}, gamma={self.gamma})")


def normalise_pressure(field, *, reference: float | None = None) -> np.ndarray:
    """Scale a pressure field to [0,1].

    ``reference`` fixes the denominator across designs so that two insoles are
    comparable; without it each field is scaled by its own max and the designs
    are not on a common scale.
    """
    f = np.asarray(field, dtype=float)
    ref = float(reference) if reference is not None else float(f.max())
    if ref <= 0:
        return np.zeros_like(f)
    return np.clip(f / ref, 0.0, 1.0)


class DensityToThickness:
    """Invert a measured density-vs-thickness curve.

    Built from :func:`visole.lattice.implicit.calibrate_density_vs_thickness`,
    so the inversion rests on measured voxel fractions rather than a closed-form
    guess about TPMS sheet density.
    """

    def __init__(self, calibration: dict) -> None:
        t = np.asarray(calibration["thickness"], dtype=float)
        d = np.asarray(calibration["relative_density"], dtype=float)
        order = np.argsort(d)
        self._d = d[order]
        self._t = t[order]
        if not np.all(np.diff(self._d) >= 0):  # pragma: no cover - sorted above
            raise ValueError("density curve must be monotonic in thickness")

    @property
    def density_range(self) -> tuple[float, float]:
        return float(self._d.min()), float(self._d.max())

    def __call__(self, density) -> np.ndarray:
        d = np.asarray(density, dtype=float)
        lo, hi = self.density_range
        return np.interp(np.clip(d, lo, hi), self._d, self._t)
