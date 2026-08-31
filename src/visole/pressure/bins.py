"""Ordered pressure bins -- the PressureVision++ representation, in our units.

PressureVision++ treats pressure estimation as classification into nine
logarithmically spaced bins rather than scalar regression, and penalises large
bin errors more than small ones. We reuse the *structure* of that idea.

We do not reuse their bin edges. Theirs are in kPa from a calibrated sensor.
Ours are in **baseline-corrected raw sensor counts**, because the Insole-GAITRite
insoles were used without subject-specific calibration and no counts->kPa mapping
is established. Calling these kPa would be fabrication.

Layout: bin 0 is the below-contact bin; bins 1..n-1 are log-spaced up to the top
edge, and the topmost bin is open-ended so outliers cannot fall outside.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Default top edge, from the 99.9th percentile of baseline-corrected counts
#: over a 40-clip walking sample (869,696 values). Re-derive if the corpus changes.
DEFAULT_TOP_EDGE = 3100.0
#: Below this many baseline-corrected counts we call it no contact.
DEFAULT_CONTACT_THRESHOLD = 1.0


@dataclass(frozen=True)
class PressureBins:
    """Ordered bins over a non-negative signal.

    ``edges`` has length ``n_bins`` and holds the *lower* edge of each bin.
    edges[0] is always 0 (the below-contact bin).
    """

    edges: np.ndarray
    units: str = "baseline_corrected_sensor_counts"

    def __post_init__(self) -> None:
        e = np.asarray(self.edges, dtype=float)
        if e.ndim != 1 or e.size < 2:
            raise ValueError("need at least 2 bin edges")
        if not np.all(np.diff(e) > 0):
            raise ValueError("bin edges must be strictly increasing")
        if e[0] != 0.0:
            raise ValueError("edges[0] must be 0 (the below-contact bin)")
        object.__setattr__(self, "edges", e)

    @property
    def n_bins(self) -> int:
        return int(self.edges.size)

    @classmethod
    def log_spaced(cls, n_bins: int = 9, contact_threshold: float = DEFAULT_CONTACT_THRESHOLD,
                   top_edge: float = DEFAULT_TOP_EDGE, **kw) -> "PressureBins":
        """Bin 0 = no contact; bins 1..n-1 log-spaced from threshold to top_edge."""
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if not 0 < contact_threshold < top_edge:
            raise ValueError("need 0 < contact_threshold < top_edge")
        upper = np.geomspace(contact_threshold, top_edge, n_bins - 1)
        return cls(np.concatenate([[0.0], upper]), **kw)

    def encode(self, values) -> np.ndarray:
        """Map values to bin indices in [0, n_bins-1]. Shape is preserved."""
        v = np.asarray(values, dtype=float)
        if np.any(v < 0):
            raise ValueError("negative values -- baseline-correct and clip first")
        # searchsorted with 'right' puts v exactly on an edge into the upper bin.
        return np.clip(np.searchsorted(self.edges, v, side="right") - 1,
                       0, self.n_bins - 1).astype(np.int64)

    def decode(self, indices) -> np.ndarray:
        """Representative value per bin: geometric midpoint, 0 for the contact bin.

        Decoding is lossy by construction. Use :meth:`quantisation_error` to
        report how lossy before drawing conclusions from binned predictions.
        """
        idx = np.clip(np.asarray(indices), 0, self.n_bins - 1)
        reps = np.zeros(self.n_bins)
        for i in range(1, self.n_bins):
            lo = self.edges[i]
            hi = self.edges[i + 1] if i + 1 < self.n_bins else self.edges[-1] * (
                self.edges[-1] / self.edges[-2])
            reps[i] = float(np.sqrt(lo * hi))
        return reps[idx]

    def quantisation_error(self, values) -> dict:
        """Round-trip error introduced purely by binning -- the price of the
        representation, and a floor on any binned model's accuracy."""
        v = np.asarray(values, dtype=float)
        rt = self.decode(self.encode(v))
        err = np.abs(rt - v)
        denom = max(float(np.ptp(v)), 1e-9)
        return {
            "mae": float(err.mean()),
            "median_ae": float(np.median(err)),
            "p95_ae": float(np.percentile(err, 95)),
            "max_ae": float(err.max()),
            "normalised_mae": float(err.mean() / denom),
        }

    def one_hot(self, values) -> np.ndarray:
        idx = self.encode(values)
        out = np.zeros(idx.shape + (self.n_bins,), dtype=np.float32)
        np.put_along_axis(out, idx[..., None], 1.0, axis=-1)
        return out

    def bin_distance_matrix(self) -> np.ndarray:
        """|i - j| over bins, for an ordinal loss that punishes distant errors
        more than adjacent ones (the PressureVision++ 'structure-aware' idea)."""
        i = np.arange(self.n_bins)
        return np.abs(i[:, None] - i[None, :]).astype(np.float32)

    def describe(self) -> str:
        rows = []
        for i in range(self.n_bins):
            hi = self.edges[i + 1] if i + 1 < self.n_bins else np.inf
            label = "no contact" if i == 0 else ""
            rows.append(f"  bin {i}: [{self.edges[i]:8.2f}, {hi:8.2f})  {label}")
        return f"PressureBins({self.n_bins} bins, units={self.units})\n" + "\n".join(rows)
