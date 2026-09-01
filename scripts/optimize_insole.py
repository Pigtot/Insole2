#!/usr/bin/env python
"""Find the insole design that minimises simulated peak plantar pressure.

    envs/core/bin/python scripts/optimize_insole.py

Searches base modulus x thickness x grading hypothesis x density range, using the
real reconstructed plantar profile and a real measured pressure field, and reports
where the optimum sits and what binds it.

Two competing effects create an interior optimum:

* too **stiff** -> the insole barely conforms, load concentrates, peak pressure high;
* too **soft**  -> the insole compresses past densification ("bottoms out") and
  transmits load like the rigid sole beneath it.

Everything here is simulation under the Winkler model in ``docs/physics_model.md``.
The optimum is optimal *for that model*, not for a real foot.
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

from visole.lattice.mapping import DensityMapping, normalise_pressure  # noqa: E402
from visole.registration.contact import (  # noqa: E402
    SOFT_TISSUE_STIFFNESS_PA_PER_M, series_stiffness, solve_contact,
)
from simulate_insole_designs import (  # noqa: E402
    load_calibration, measured_pressure_field, plantar_profile,
)

OUT = REPO / "experiments" / "insole_physics" / "outputs"
PRINTABLE_RHO_MIN = 0.20     # measured gyroid connectivity floor


def evaluate_design(E_s_pa, thickness_m, hypothesis, rho_min, rho_max, gamma,
                    norm, profile, fit, cell_area, load_n, target_cop, tissue):
    if hypothesis == "solid":
        rho = np.ones_like(norm)
    else:
        rho = DensityMapping(hypothesis=hypothesis, rho_min=rho_min,
                             rho_max=rho_max, gamma=gamma)(norm)
    E_eff = np.where(rho >= 1.0, E_s_pa, fit(rho) * E_s_pa)
    k_ins = E_eff / thickness_m
    K = series_stiffness(E_eff, thickness_m, tissue_pa_per_m=tissue)
    sol = solve_contact(profile, K, load_n=load_n, cell_area_m2=cell_area,
                        target_cop=target_cop,
                        stiffness_ratio=float(np.median(k_ins) / tissue),
                        thickness_m=thickness_m, insole_stiffness_pa_per_m=k_ins)
    return {
        "E_s_mpa": E_s_pa / 1e6, "thickness_mm": thickness_m * 1000,
        "hypothesis": hypothesis, "rho_min": rho_min, "rho_max": rho_max,
        "gamma": gamma, "rho_mean": float(rho.mean()),
        **sol.summary(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="P1/FP/1")
    ap.add_argument("--side", default="left")
    ap.add_argument("--load-n", type=float, default=700.0)
    ap.add_argument("--grid", type=int, nargs=2, default=[24, 56])
    ap.add_argument("--tissue-pa-per-m", type=float, default=SOFT_TISSUE_STIFFNESS_PA_PER_M)
    ap.add_argument("--max-bottomed", type=float, default=0.02,
                    help="reject designs with more than this fraction bottomed out")
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
    target_cop = ((measured * iu).sum() / tot, (measured * iv).sum() / tot)
    cell_area = (0.100 / grid[0]) * (0.260 / grid[1])

    print(f"stiffness law: {fit.describe()}")
    print(f"driving field: {clip_key} {args.side} frame {frame}, load {args.load_n:.0f} N")
    print(f"printability floor: rho >= {PRINTABLE_RHO_MIN}\n")

    moduli = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 20.0, 40.0, 80.0]
    thicknesses = [0.004, 0.006, 0.008, 0.010, 0.014, 0.020, 0.025]
    hypotheses = ["uniform", "stiffen", "soften"]

    rows = []
    for E in moduli:
        for h in thicknesses:
            for hyp in hypotheses:
                rows.append(evaluate_design(
                    E * 1e6, h, hyp, PRINTABLE_RHO_MIN, 0.55, 1.0,
                    norm, profile, fit, cell_area, args.load_n, target_cop,
                    args.tissue_pa_per_m))
    rows.append(evaluate_design(40e6, 0.010, "solid", 1.0, 1.0, 1.0, norm, profile,
                                fit, cell_area, args.load_n, target_cop,
                                args.tissue_pa_per_m))

    feasible = [r for r in rows if r["bottomed_fraction"] <= args.max_bottomed
                and r["converged"] and r["force_error_pct"] < 1.0]
    print(f"{len(feasible)}/{len(rows)} designs feasible "
          f"(converged, <{100*args.max_bottomed:.0f}% bottomed out)")
    if not feasible:
        print("no feasible design")
        return 1

    feasible.sort(key=lambda r: r["peak_pressure_kpa"])
    best = feasible[0]

    print(f"\n{'rank':>5}{'E(MPa)':>8}{'h(mm)':>7}{'grading':>10}{'peak kPa':>10}"
          f"{'area cm2':>10}{'bottomed':>10}")
    for i, r in enumerate(feasible[:10], 1):
        print(f"{i:>5}{r['E_s_mpa']:>8.2f}{r['thickness_mm']:>7.0f}{r['hypothesis']:>10}"
              f"{r['peak_pressure_kpa']:>10.1f}{r['contact_area_cm2']:>10.1f}"
              f"{100*r['bottomed_fraction']:>9.0f}%")

    # what a plain TPU choice would have given, for contrast
    tpu = [r for r in rows if abs(r["E_s_mpa"] - 40) < 1e-6 and r["hypothesis"] == "uniform"]
    tpu_best = min(tpu, key=lambda r: r["peak_pressure_kpa"]) if tpu else None
    solid = [r for r in rows if r["hypothesis"] == "solid"]

    print(f"\noptimum: E = {best['E_s_mpa']:.2f} MPa, h = {best['thickness_mm']:.0f} mm, "
          f"{best['hypothesis']} grading -> {best['peak_pressure_kpa']:.1f} kPa")
    if tpu_best:
        gain = 100 * (tpu_best["peak_pressure_kpa"] - best["peak_pressure_kpa"]) / tpu_best["peak_pressure_kpa"]
        print(f"  vs uniform 40 MPa TPU ({tpu_best['peak_pressure_kpa']:.1f} kPa): {gain:.0f}% lower")
    if solid:
        gain = 100 * (solid[0]["peak_pressure_kpa"] - best["peak_pressure_kpa"]) / solid[0]["peak_pressure_kpa"]
        print(f"  vs solid insole      ({solid[0]['peak_pressure_kpa']:.1f} kPa): {gain:.0f}% lower")

    # is the optimum pinned to a search boundary?
    at_edge = []
    if best["E_s_mpa"] <= min(moduli) + 1e-9:
        at_edge.append("softest modulus searched")
    if best["thickness_mm"] >= max(thicknesses) * 1000 - 1e-9:
        at_edge.append("thickest insole searched")
    if best["thickness_mm"] <= min(thicknesses) * 1000 + 1e-9:
        at_edge.append("thinnest insole searched")
    print("\nboundary check: " + ("optimum is INTERIOR (a genuine trade-off)"
                                 if not at_edge else
                                 "optimum sits at the " + ", ".join(at_edge)
                                 + " -- widen the search before trusting it"))

    report = {
        "note": ("Simulation under the Winkler model. Optimal for that model only. "
                 "Peak pressures in kPa are simulated; the measured field supplies "
                 "only the normalised load shape and COP, in raw sensor counts."),
        "driving_clip": clip_key, "side": args.side, "frame": frame,
        "load_n": args.load_n, "tissue_pa_per_m": args.tissue_pa_per_m,
        "printable_rho_min": PRINTABLE_RHO_MIN,
        "stiffness_law": fit.describe(),
        "search": {"moduli_mpa": moduli, "thicknesses_m": thicknesses,
                   "hypotheses": hypotheses},
        "n_evaluated": len(rows), "n_feasible": len(feasible),
        "optimum": best, "top10": feasible[:10],
        "reference_tpu_uniform": tpu_best, "reference_solid": solid[0] if solid else None,
        "optimum_at_search_boundary": at_edge,
        "all_results": rows,
    }
    (OUT / "insole_optimum.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {(OUT / 'insole_optimum.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
