#!/usr/bin/env python
"""Which 3D-printing FILAMENT should the insole be printed in?

    envs/core/bin/python scripts/optimize_filament_insole.py

The earlier optimiser reported a base-material modulus (1.5 MPa), which reads as
"use foam". That was the wrong frame for a printed part. This one searches only
over **real TPU filament grades**, keeps the lattice above its measured
printability floor, and reports the best achievable design and what it costs to
use a harder, easier-to-print grade.

Shore A hardness is converted to Young's modulus with Gent's empirical relation.
It is a correlation with roughly +/-30% scatter, and printed lattices are
anisotropic, so grades are indicative -- the honest way to fix this is a
compression test on a printed coupon.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from visole.lattice.mapping import normalise_pressure  # noqa: E402
from visole.registration.contact import SOFT_TISSUE_STIFFNESS_PA_PER_M  # noqa: E402
from optimize_insole import PRINTABLE_RHO_MIN, evaluate_design  # noqa: E402
from simulate_insole_designs import (  # noqa: E402
    load_calibration, measured_pressure_field, plantar_profile,
)

OUT = REPO / "experiments" / "insole_physics" / "outputs"

#: Commercially available TPU/TPE filament grades. Printability notes reflect the
#: consensus in printing guides: below ~70A filament is "gummy" and needs a direct
#: drive extruder and slow speeds; 95A is the easy default.
FILAMENTS = [
    (60, "very soft TPE - direct drive, slow, hard to print"),
    (70, "soft TPU/TPE - direct drive, slow"),
    (75, "soft TPU - direct drive"),
    (80, "soft TPU"),
    (85, "TPU - commonly recommended for insoles"),
    (90, "TPU - easy"),
    (95, "TPU - the usual default, easiest"),
]


def gent_modulus_mpa(shore_a: float) -> float:
    """Gent's empirical Shore A -> Young's modulus relation for elastomers."""
    return 0.0981 * (56 + 7.62336 * shore_a) / (0.137505 * (254 - 2.54 * shore_a))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="P1/FP/1")
    ap.add_argument("--side", default="left")
    ap.add_argument("--load-n", type=float, default=700.0)
    ap.add_argument("--grid", type=int, nargs=2, default=[24, 56])
    ap.add_argument("--max-thickness-mm", type=float, default=14.0,
                    help="wearable limit; the unconstrained optimum wanted 20 mm")
    args = ap.parse_args()

    cal = sorted(OUT.glob("stiffness_calibration_*.json"))
    if not cal:
        print("run scripts/calibrate_lattice_stiffness.py first")
        return 1
    fit = load_calibration(cal[-1])
    grid = tuple(args.grid)

    field, clip_key, frame = measured_pressure_field(args.clip, args.side, grid)
    measured = np.nan_to_num(np.asarray(field.masked(), dtype=float), nan=0.0)
    norm = normalise_pressure(measured)
    profile = plantar_profile(grid)
    tot = measured.sum()
    iu, iv = np.indices(measured.shape)
    cop = ((measured * iu).sum() / tot, (measured * iv).sum() / tot)
    cell_area = (0.100 / grid[0]) * (0.260 / grid[1])

    thicknesses = [t for t in (0.006, 0.008, 0.010, 0.012, 0.014, 0.020)
                   if t * 1000 <= args.max_thickness_mm + 1e-9]
    rho_maxes = [0.35, 0.45, 0.55, 0.70]

    rows = []
    for shore, note in FILAMENTS:
        E = gent_modulus_mpa(shore)
        for h in thicknesses:
            for rmax in rho_maxes:
                for hyp in ("uniform", "soften"):
                    r = evaluate_design(E * 1e6, h, hyp, PRINTABLE_RHO_MIN, rmax, 1.0,
                                        norm, profile, fit, cell_area, args.load_n,
                                        cop, SOFT_TISSUE_STIFFNESS_PA_PER_M)
                    r.update({"shore_a": shore, "bulk_modulus_mpa": E, "note": note})
                    rows.append(r)

    feasible = [r for r in rows if r["bottomed_fraction"] <= 0.02 and r["converged"]
                and r["force_error_pct"] < 1.0]
    print(f"stiffness law: {fit.describe()}")
    print(f"driving field: {clip_key} {args.side} frame {frame}, {args.load_n:.0f} N")
    print(f"printability floor rho >= {PRINTABLE_RHO_MIN}, "
          f"max thickness {args.max_thickness_mm:.0f} mm")
    print(f"{len(feasible)}/{len(rows)} designs feasible\n")

    print(f"{'Shore':>6}{'bulk MPa':>10}{'best kPa':>10}{'h(mm)':>7}{'rho_max':>9}"
          f"{'grading':>9}   printability")
    best_per = {}
    for shore, note in FILAMENTS:
        cand = [r for r in feasible if r["shore_a"] == shore]
        if not cand:
            print(f"{shore:>6}{gent_modulus_mpa(shore):>10.1f}"
                  f"{'  none feasible':>26}   {note}")
            continue
        b = min(cand, key=lambda r: r["peak_pressure_kpa"])
        best_per[shore] = b
        print(f"{shore:>6}{b['bulk_modulus_mpa']:>10.1f}{b['peak_pressure_kpa']:>10.1f}"
              f"{b['thickness_mm']:>7.0f}{b['rho_max']:>9.2f}{b['hypothesis']:>9}   {note}")

    if not best_per:
        print("nothing feasible")
        return 1
    overall = min(best_per.values(), key=lambda r: r["peak_pressure_kpa"])
    ref95 = best_per.get(95)

    print(f"\nbest printable design: Shore {overall['shore_a']}A, "
          f"{overall['thickness_mm']:.0f} mm, rho {overall['rho_min']:.2f}-{overall['rho_max']:.2f}, "
          f"{overall['hypothesis']} -> {overall['peak_pressure_kpa']:.0f} kPa")
    if ref95:
        d = 100 * (ref95["peak_pressure_kpa"] - overall["peak_pressure_kpa"]) / ref95["peak_pressure_kpa"]
        print(f"  vs the easy default (Shore 95A, {ref95['peak_pressure_kpa']:.0f} kPa): "
              f"{d:.0f}% lower peak pressure")
    if 85 in best_per:
        b85 = best_per[85]
        d = 100 * (b85["peak_pressure_kpa"] - overall["peak_pressure_kpa"]) / b85["peak_pressure_kpa"]
        print(f"  vs Shore 85A ({b85['peak_pressure_kpa']:.0f} kPa), the usual insole "
              f"recommendation: {d:.0f}% lower")

    report = {
        "note": ("Simulation. Shore->modulus via Gent's empirical relation (~+/-30% "
                 "scatter); printed lattices are anisotropic. A compression test on a "
                 "printed coupon would replace the conversion."),
        "stiffness_law": fit.describe(),
        "printable_rho_min": PRINTABLE_RHO_MIN,
        "max_thickness_mm": args.max_thickness_mm,
        "best_per_grade": {str(k): v for k, v in best_per.items()},
        "overall_best": overall,
        "all_results": rows,
    }
    (OUT / "filament_choice.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {(OUT / 'filament_choice.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
