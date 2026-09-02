#!/usr/bin/env python
"""Milestone 9 -- the whole Visole chain, end to end, in one command.

    envs/core/bin/python scripts/run_visole_pipeline.py --clip P1/FP/1 --side left

    measured plantar pressure          (raw sensor counts)
        -> canonical plantar frame     (both feet, one frame)
        -> foot outline + profile      (FOCUS/FIND sole surface)
        -> density map                 (pressure -> material, chosen hypothesis)
        -> measured stiffness law      (E*/Es from FEA)
        -> contact check               (Winkler; does it bottom out?)
        -> foot-shaped graded lattice  (STL, printable)

Design parameters default to the optimum found by ``scripts/optimize_insole.py``
rather than being hardcoded, so this script follows the evidence instead of
restating it.

**Nothing here is validated on a real foot.** The geometry is real and printable;
the claim that it improves anything is not.
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
sys.path.insert(0, str(REPO / "scripts"))

from visole.lattice.implicit import (  # noqa: E402
    calibrate_density_vs_thickness, generate_insole_lattice,
)
from visole.lattice.mapping import (  # noqa: E402
    DensityMapping, DensityToThickness, normalise_pressure,
)
from visole.registration.contact import (  # noqa: E402
    SOFT_TISSUE_STIFFNESS_PA_PER_M, series_stiffness, solve_contact,
)
from simulate_insole_designs import (  # noqa: E402
    load_calibration, measured_pressure_field, plantar_profile,
)

OUT = REPO / "experiments" / "integrated_demo"
PHYS = REPO / "experiments" / "insole_physics" / "outputs"


def load_optimum() -> dict:
    p = PHYS / "insole_optimum.json"
    if not p.exists():
        return {"E_s_mpa": 2.5, "thickness_mm": 10.0, "hypothesis": "soften",
                "source": "fallback default (optimiser has not been run)"}
    o = json.loads(p.read_text())["optimum"]
    o["source"] = "scripts/optimize_insole.py"
    return o


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="P1/FP/1")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--grid", type=int, nargs=2, default=[24, 56])
    ap.add_argument("--load-n", type=float, default=700.0)
    ap.add_argument("--aggregate", default="peak", choices=["peak", "pti", "frame"],
                    help="how to reduce the clip to one pressure field")
    ap.add_argument("--resolution", type=int, default=5, help="voxels per lattice cell")
    ap.add_argument("--cell-size", type=float, default=8.0)
    ap.add_argument("--rim-mm", type=float, default=2.5)
    ap.add_argument("--modulus-mpa", type=float, default=None)
    ap.add_argument("--thickness-mm", type=float, default=None)
    ap.add_argument("--hypothesis", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    opt = load_optimum()
    E_mpa = args.modulus_mpa if args.modulus_mpa is not None else opt["E_s_mpa"]
    h_mm = args.thickness_mm if args.thickness_mm is not None else opt["thickness_mm"]
    hyp = args.hypothesis or opt["hypothesis"]
    print(f"design: {E_mpa:.2f} MPa base, {h_mm:.0f} mm thick, '{hyp}' grading")
    print(f"        (from {opt['source']})\n")

    grid = tuple(args.grid)

    # 1 -- measured pressure, lifted into the canonical plantar frame
    field, clip_key, frame = measured_pressure_field(args.clip, args.side, grid,
                                                     aggregate=args.aggregate)
    measured = np.nan_to_num(np.asarray(field.masked(), dtype=float), nan=0.0)
    norm = normalise_pressure(measured)
    print(f"1. pressure   {clip_key} {args.side}, {frame}, "
          f"{measured.max():.0f} counts peak")

    # 2 -- foot outline and profile from the FOCUS/FIND sole surface
    profile = plantar_profile(grid)
    sentinel = profile.max()
    footprint = profile < sentinel - 1e-9
    print(f"2. foot       {footprint.sum()}/{footprint.size} canonical cells under the foot "
          f"({100*footprint.mean():.0f}%), arch {1000*profile[footprint].max():.1f} mm")

    # 3 -- pressure -> relative density -> level-set thickness
    stiff_cal = sorted(PHYS.glob("stiffness_calibration_*.json"))
    if not stiff_cal:
        print("missing stiffness calibration; run scripts/calibrate_lattice_stiffness.py")
        return 1
    fit = load_calibration(stiff_cal[-1])
    mapping = DensityMapping(hypothesis=hyp)
    rho = np.where(footprint, mapping(norm), mapping.rho_min)
    d2t = DensityToThickness(calibrate_density_vs_thickness(cell_size_mm=args.cell_size))
    t_field = d2t(rho)
    print(f"3. material   rho {rho[footprint].min():.2f}-{rho[footprint].max():.2f}, "
          f"{fit.describe().split('  ')[0]}")

    # 4 -- does this design actually work under load?
    E_eff = fit(rho) * (E_mpa * 1e6)
    k_ins = E_eff / (h_mm / 1000.0)
    K = series_stiffness(E_eff, h_mm / 1000.0)
    cell_area = (0.100 / grid[0]) * (0.260 / grid[1])
    tot = measured.sum()
    iu, iv = np.indices(measured.shape)
    cop = ((measured * iu).sum() / tot, (measured * iv).sum() / tot)
    sol = solve_contact(profile, K, load_n=args.load_n, cell_area_m2=cell_area,
                        target_cop=cop, thickness_m=h_mm / 1000.0,
                        insole_stiffness_pa_per_m=k_ins)
    s = sol.summary()
    ok = s["bottomed_fraction"] <= 0.02 and sol.converged
    print(f"4. contact    peak {s['peak_pressure_kpa']:.0f} kPa, "
          f"contact {s['contact_area_cm2']:.0f} cm2, "
          f"bottomed {100*s['bottomed_fraction']:.0f}%  -> {'OK' if ok else 'REJECTED'}")
    if not ok:
        print("   design bottoms out under load; not exporting geometry")
        return 1

    # 5 -- foot-shaped graded lattice
    t0 = time.time()
    lat = generate_insole_lattice(footprint, t_field, height_mm=h_mm,
                                  cell_size_mm=args.cell_size,
                                  resolution=args.resolution, rim_mm=args.rim_mm)
    ls = lat.summary()
    stl = OUT / f"insole_{args.clip.replace('/', '_')}_{args.side}.stl"
    lat.export_stl(stl)
    print(f"5. geometry   {ls['n_faces']} faces, rho {ls['relative_density_voxel_count']:.3f}, "
          f"watertight={ls['watertight']}, connected={ls['connected']} "
          f"({time.time()-t0:.0f}s)")
    if ls["discarded_fragment_mm3"] > 0:
        print(f"   note: discarded {ls['discarded_fragment_mm3']:.1f} mm3 of "
              "disconnected material so the part is printable")

    report = {
        "note": ("Geometry is real and printable. Every pressure figure in kPa is "
                 "SIMULATED under the Winkler model; the measured insole data is in "
                 "raw sensor counts and supplies only the load shape and COP. No "
                 "claim is made that this insole improves anything on a real foot."),
        "design": {"base_modulus_mpa": E_mpa, "thickness_mm": h_mm,
                   "grading": hyp, "source": opt["source"]},
        "pressure_source": {"clip": clip_key, "side": args.side, "frame": frame,
                            "units": "raw sensor counts"},
        "foot": {"footprint_cells": int(footprint.sum()),
                 "arch_height_mm": float(1000 * profile[footprint].max())},
        "stiffness_law": fit.describe(),
        "contact": s,
        "geometry": ls,
        "stl": str(stl.relative_to(REPO)),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (OUT / "pipeline_result.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / "pipeline_fields.npz", measured=measured, profile=profile,
                        footprint=footprint, rho=rho, thickness=t_field,
                        simulated_pressure_pa=sol.pressure_pa)
    print(f"\nwrote {stl.relative_to(REPO)}  ({stl.stat().st_size/1e6:.1f} MB)")
    print(f"      {(OUT / 'pipeline_result.json').relative_to(REPO)}")
    print(f"total {report['elapsed_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
