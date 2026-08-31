#!/usr/bin/env python
"""PoC 2 -- pressure-graded TPMS coupons, exported as STL.

    envs/core/bin/python scripts/generate_lattice_demo.py --resolution 10

Generates the comparison set the physical experiment will eventually need:

    A. solid            reference block
    B. uniform lattice  constant thickness (control)
    C. graded stiffen   high pressure -> higher relative density
    D. graded soften    high pressure -> lower relative density

C and D are competing hypotheses. Which one actually offloads is unknown and
cannot be settled by generating geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visole.lattice.implicit import (  # noqa: E402
    calibrate_density_vs_thickness, generate_graded_tpms,
)
from visole.lattice.mapping import (  # noqa: E402
    DensityMapping, DensityToThickness, normalise_pressure,
)

OUT = REPO / "experiments" / "lattice_demo" / "outputs"


def synthetic_pressure(X: np.ndarray, Y: np.ndarray, bounds) -> np.ndarray:
    """A synthetic 2D pressure gradient: two Gaussian 'load spots'.

    Explicitly synthetic. It stands in for a predicted plantar field so the
    geometry pipeline can be exercised; it is not a measurement.
    """
    (x0, x1), (y0, y1) = bounds[0], bounds[1]
    u = (X - x0) / (x1 - x0)
    v = (Y - y0) / (y1 - y0)
    heel = np.exp(-(((u - 0.30) ** 2 + (v - 0.28) ** 2) / (2 * 0.030)))
    fore = np.exp(-(((u - 0.68) ** 2 + (v - 0.72) ** 2) / (2 * 0.045)))
    return 0.9 * heel + 1.0 * fore


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=int, default=10, help="voxels per unit cell")
    ap.add_argument("--cell-size", type=float, default=8.0)
    ap.add_argument("--size", type=float, nargs=3, default=[48.0, 48.0, 12.0])
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sx, sy, sz = args.size
    bounds = ((0.0, sx), (0.0, sy), (0.0, sz))
    OUT.mkdir(parents=True, exist_ok=True)

    calib = calibrate_density_vs_thickness(cell_size_mm=args.cell_size)
    d2t = DensityToThickness(calib)
    print(f"measured density range for gyroid sheet: "
          f"{d2t.density_range[0]:.3f}..{d2t.density_range[1]:.3f}")
    if args.dry_run:
        return 0

    # Common normalisation reference so designs are comparable.
    probe_x = np.linspace(0, sx, 64)
    probe_y = np.linspace(0, sy, 64)
    PX, PY = np.meshgrid(probe_x, probe_y, indexing="ij")
    p_ref = float(synthetic_pressure(PX, PY, bounds).max())

    designs, records = {}, {}

    # A. solid reference -------------------------------------------------
    import trimesh

    solid = trimesh.creation.box(extents=[sx, sy, sz])
    solid.apply_translation([sx / 2, sy / 2, sz / 2])
    solid.export(OUT / "A_solid.stl")
    records["A_solid"] = {
        "description": "solid reference block", "relative_density_voxel_count": 1.0,
        "relative_density_from_mesh_volume": float(solid.volume) / (sx * sy * sz),
        "volume_mm3": float(solid.volume), "watertight": bool(solid.is_watertight),
        "n_faces": int(len(solid.faces)), "n_vertices": int(len(solid.vertices)),
    }

    # B. uniform lattice (control) ---------------------------------------
    mid_density = 0.5 * (DensityMapping().rho_min + DensityMapping().rho_max)
    t_uniform = float(d2t(mid_density))
    lat = generate_graded_tpms(bounds_mm=bounds, cell_size_mm=args.cell_size,
                               constant_thickness=t_uniform, resolution=args.resolution)
    lat.export_stl(OUT / "B_uniform.stl")
    designs["B_uniform"] = lat
    records["B_uniform"] = {"description": f"uniform lattice, t={t_uniform:.3f}",
                            "target_relative_density": mid_density, **lat.summary()}

    # C / D. graded, competing hypotheses --------------------------------
    for tag, hypothesis in (("C_graded_stiffen", "stiffen"), ("D_graded_soften", "soften")):
        mapping = DensityMapping(hypothesis=hypothesis, gamma=args.gamma)

        def thickness_fn(X, Y, _m=mapping):
            p = normalise_pressure(synthetic_pressure(X, Y, bounds), reference=p_ref)
            return d2t(_m(p))

        lat = generate_graded_tpms(bounds_mm=bounds, cell_size_mm=args.cell_size,
                                   thickness_fn=thickness_fn, resolution=args.resolution)
        lat.export_stl(OUT / f"{tag}.stl")
        designs[tag] = lat
        records[tag] = {"description": mapping.describe(), "hypothesis": hypothesis,
                        "gamma": args.gamma, **lat.summary()}

    report = {
        "note": ("Synthetic pressure field. Geometry only -- no claim is made that any "
                 "design reduces plantar pressure. That requires physical testing."),
        "calibration": calib,
        "designs": records,
    }
    (OUT / "lattice_demo.json").write_text(json.dumps(report, indent=2))

    # figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True)
    P = normalise_pressure(synthetic_pressure(PX, PY, bounds), reference=p_ref)
    im = axes[0].imshow(P.T, origin="lower", cmap="inferno", extent=[0, sx, 0, sy])
    axes[0].set_title("synthetic normalised pressure\n(input, not a measurement)", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].plot(calib["relative_density"], calib["thickness"], "o-")
    axes[1].set_xlabel("relative density (voxel-counted)")
    axes[1].set_ylabel("thickness parameter t")
    axes[1].set_title("measured density -> thickness\n(inverted for grading)", fontsize=9)
    axes[1].grid(alpha=0.3)

    for ax, tag in zip(axes[2:], ("C_graded_stiffen", "D_graded_soften")):
        tf = designs[tag].thickness_field
        im = ax.imshow(tf.T, origin="lower", cmap="viridis", extent=[0, sx, 0, sy])
        ax.set_title(f"{tag}\nthickness field t(x,y)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("PoC 2 - pressure -> material parameter -> graded gyroid lattice", fontsize=12)
    fig.savefig(OUT / "lattice_demo.png", dpi=120)
    plt.close(fig)

    print(f"\n{'design':<18}{'rho(voxel)':>12}{'rho(mesh)':>11}{'faces':>9}  watertight")
    for k, v in records.items():
        print(f"{k:<18}{v.get('relative_density_voxel_count', float('nan')):>12.4f}"
              f"{(v.get('relative_density_from_mesh_volume') or float('nan')):>11.4f}"
              f"{v.get('n_faces', 0):>9}  {v.get('watertight')}")
    print(f"\nwrote {len(list(OUT.glob('*.stl')))} STL files to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
