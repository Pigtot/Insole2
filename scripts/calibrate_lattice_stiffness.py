#!/usr/bin/env python
"""Phase A -- measure lattice stiffness by numerical compression test.

    envs/core/bin/python scripts/calibrate_lattice_stiffness.py --resolution 14
    envs/core/bin/python scripts/calibrate_lattice_stiffness.py --convergence

Produces the density -> stiffness curve that the contact model needs, plus the
solver-correctness evidence without which the curve is just numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.lattice.stiffness import (  # noqa: E402
    TPU_MODULUS_PA, TPU_POISSON, calibrate_stiffness_vs_density, effective_modulus,
)

OUT = REPO / "experiments" / "insole_physics" / "outputs"


def solid_block_check(n: int = 6) -> dict:
    """A solver that cannot recover a known modulus is not evidence."""
    occ = np.ones((n, n, n), dtype=bool)
    r = effective_modulus(occ, (0.01 / n,) * 3)
    ratio = r.effective_modulus_pa / TPU_MODULUS_PA
    return {"E_eff_pa": r.effective_modulus_pa, "E_s_pa": TPU_MODULUS_PA,
            "ratio": ratio, "passes": bool(abs(ratio - 1.0) < 0.02)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolution", type=int, default=14, help="voxels per unit cell")
    ap.add_argument("--cells", type=int, default=3)
    ap.add_argument("--cell-size", type=float, default=8.0)
    ap.add_argument("--modulus-mpa", type=float, default=TPU_MODULUS_PA / 1e6)
    ap.add_argument("--convergence", action="store_true",
                    help="also run a coarser resolution to size discretisation error")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    E_s = args.modulus_mpa * 1e6

    print("solver check: solid block must recover E_s")
    chk = solid_block_check()
    print(f"  E_eff/E_s = {chk['ratio']:.5f}  {'PASS' if chk['passes'] else 'FAIL'}")
    if not chk["passes"]:
        print("aborting: solver does not reproduce a known answer")
        return 1

    # Thicknesses chosen to spread relative density evenly above the measured
    # connectivity floor (rho ~ 0.19), not linearly in t.
    thicknesses = np.linspace(0.30, 1.10, 9)

    print(f"\ncalibrating gyroid, {args.cells}^3 cells at resolution {args.resolution}, "
          f"E_s={args.modulus_mpa:.0f} MPa")
    t0 = time.time()
    cal = calibrate_stiffness_vs_density(thicknesses, cell_size_mm=args.cell_size,
                                         cells=args.cells, resolution=args.resolution,
                                         E_s=E_s, nu=TPU_POISSON, verbose=True)
    cal["solid_block_check"] = chk
    cal["elapsed_s"] = round(time.time() - t0, 1)
    print(f"\n{cal['fit']['description']}")

    if args.convergence:
        coarse_res = max(6, args.resolution // 2)
        print(f"\nconvergence: repeating at resolution {coarse_res}")
        coarse = calibrate_stiffness_vs_density(thicknesses, cell_size_mm=args.cell_size,
                                                cells=args.cells, resolution=coarse_res,
                                                E_s=E_s, nu=TPU_POISSON, verbose=True)
        fine_n, coarse_n = cal["fit"]["n"], coarse["fit"]["n"]
        fine_C, coarse_C = cal["fit"]["C"], coarse["fit"]["C"]
        cal["convergence"] = {
            "coarse_resolution": coarse_res,
            "coarse_fit": coarse["fit"],
            "delta_n": abs(fine_n - coarse_n),
            "delta_C_rel": abs(fine_C - coarse_C) / max(fine_C, 1e-9),
        }
        print(f"\n  fine   n={fine_n:.3f} C={fine_C:.3f}")
        print(f"  coarse n={coarse_n:.3f} C={coarse_C:.3f}")
        print(f"  |dn|={cal['convergence']['delta_n']:.3f}  "
              f"dC/C={cal['convergence']['delta_C_rel']:.3f}")

    path = OUT / f"stiffness_calibration_r{args.resolution}.json"
    path.write_text(json.dumps(cal, indent=2))
    print(f"\nwrote {path.relative_to(REPO)}  ({cal['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
