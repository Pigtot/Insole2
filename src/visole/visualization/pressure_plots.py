"""Plotting helpers for insole pressure.

Interpolated images produced here are **display only**. There are 32 measured
channels per foot; interpolation invents nothing and adds no measurements. Every
function that interpolates says so in its name or docstring, and the plots label
it on the figure itself.
"""

from __future__ import annotations

import numpy as np

from ..pressure.sensor_map import SensorMap


def scatter_sensors(ax, smap: SensorMap, values: np.ndarray, *, vmax: float | None = None,
                    cmap: str = "inferno", size: float = 260, show_ids: bool = False):
    """Plot the 32 measured channels at their true pad locations."""
    values = np.asarray(values, dtype=float)
    if values.shape != (len(smap.xy),):
        raise ValueError(f"expected {len(smap.xy)} values, got {values.shape}")
    for pad in smap.pads:
        ax.plot(pad[:, 0], pad[:, 1], lw=0.4, color="0.75", zorder=1)
    for chain in smap.outline:
        ax.plot(chain[:, 0], chain[:, 1], lw=0.8, color="0.45", zorder=1)
    sc = ax.scatter(smap.xy[:, 0], smap.xy[:, 1], c=values, s=size,
                    cmap=cmap, vmin=0, vmax=vmax, zorder=3,
                    edgecolors="k", linewidths=0.4)
    if show_ids:
        for i, (x, y) in enumerate(smap.xy):
            ax.text(x, y, str(i), fontsize=5, ha="center", va="center",
                    color="white", zorder=4)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # SVG y grows downward; keep the drawing upright
    ax.axis("off")
    return sc


def interpolated_display_image(smap: SensorMap, values: np.ndarray, *, grid: int = 160,
                               power: float = 2.0, mask_radius: float | None = 90.0):
    """Inverse-distance-weighted image for DISPLAY ONLY.

    This is not a measurement and must never be described as one, nor fed to a
    model as if it were dense ground truth. It exists so a 32-value vector is
    legible to a human.
    """
    values = np.asarray(values, dtype=float)
    x, y = smap.xy[:, 0], smap.xy[:, 1]
    pad = 0.06 * max(np.ptp(x), np.ptp(y))
    gx = np.linspace(x.min() - pad, x.max() + pad, grid)
    gy = np.linspace(y.min() - pad, y.max() + pad, grid)
    GX, GY = np.meshgrid(gx, gy)
    d2 = (GX[..., None] - x) ** 2 + (GY[..., None] - y) ** 2
    w = 1.0 / np.power(d2 + 1e-6, power / 2.0)
    img = (w * values).sum(-1) / w.sum(-1)
    if mask_radius is not None:
        # Hide regions far from any sensor: extrapolating there would suggest
        # knowledge the 32 channels do not contain.
        img = np.ma.masked_where(np.sqrt(d2.min(-1)) > mask_radius, img)
    return img
