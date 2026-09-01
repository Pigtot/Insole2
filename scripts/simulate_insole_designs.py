#!/usr/bin/env python
"""Phase C -- does a pressure-graded insole actually redistribute load?

    envs/core/bin/python scripts/simulate_insole_designs.py --sweep-modulus

Runs the four designs from ``scripts/generate_lattice_demo.py`` through the
Winkler contact model, driven by a **real measured** plantar pressure field and
a **real reconstructed** foot profile:

    A  solid          rho = 1
    B  uniform        rho = const                (control)
    C  stiffen        high pressure -> higher rho
    D  soften         high pressure -> lower rho

Every design carries the same load through the same centre of pressure, so the
only thing that differs is the *distribution*. Whatever the answer is -- including
"the designs are indistinguishable" -- it gets reported.

**This is simulation.** Pressures are in kPa because the load and material are
chosen inputs; the measured insole data that supplies the load's *shape* stays in
raw sensor counts, and the two are never mixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.data.insole_gaitrite import InsoleGaitRite  # noqa: E402
from visole.lattice.mapping import DensityMapping, normalise_pressure  # noqa: E402
from visole.lattice.stiffness import GibsonAshby, TPU_POISSON  # noqa: E402
from visole.pressure.canonical_foot import lift_to_canonical  # noqa: E402
from visole.pressure.sensor_map import load_sensor_map  # noqa: E402
from visole.registration.contact import (  # noqa: E402
    SOFT_TISSUE_STIFFNESS_PA_PER_M, plantar_profile_from_mesh, series_stiffness,
    solve_contact,
)

OUT = REPO / "experiments" / "insole_physics" / "outputs"
FIND_DIR = REPO / "external" / "FOCUS" / "data" / "find"

DESIGNS = {
    "A_solid": None,                 # rho = 1 everywhere
    "B_uniform": "uniform",
    "C_stiffen": "stiffen",
    "D_soften": "soften",
}


def load_calibration(path: Path) -> GibsonAshby:
    d = json.loads(path.read_text())
    f = d["fit"]
    rho = np.array([p["relative_density"] for p in d["points"]])
    ratio = np.array([p["E_ratio"] for p in d["points"]])
    return GibsonAshby(C=f["C"], n=f["n"], rho=rho, E_ratio=ratio,
                       r_squared=f["r_squared"])


def measured_pressure_field(clip_key: str, side: str, grid):
    """Peak-load instant of a real walking clip, lifted to canonical coordinates."""
    p, c, k = clip_key.split("/")
    clip = InsoleGaitRite().get(p, c, k)
    arr = clip.pressure(side, baseline_correct=True)
    frame = int(arr.sum(1).argmax())
    smap = load_sensor_map(side)
    field = lift_to_canonical(arr[frame], smap, grid=grid)
    return field, clip.key, frame


def plantar_profile(grid) -> np.ndarray:
    """Sole surface of the FIND template, which ships inside the FOCUS checkout."""
    import trimesh

    mesh = trimesh.load(FIND_DIR / "template.obj", process=False, force="mesh")
    sole_faces = np.load(FIND_DIR / "templ_sole_faces.npy")
    sole_verts = np.unique(mesh.faces[sole_faces])
    return plantar_profile_from_mesh(np.asarray(mesh.vertices), sole_verts, grid)


def run_design(name, hypothesis, norm_pressure, profile, fit, *, E_s, thickness,
               load_n, cell_area, target_cop, rho_min, rho_max, gamma, tissue):
    if hypothesis is None:
        rho = np.ones_like(norm_pressure)
    else:
        rho = DensityMapping(hypothesis=hypothesis, rho_min=rho_min,
                             rho_max=rho_max, gamma=gamma)(norm_pressure)
    E_eff = np.where(rho >= 1.0, E_s, fit(rho) * E_s)
    K = series_stiffness(E_eff, thickness, tissue_pa_per_m=tissue)
    # How far the insole is from mattering at all: if it is far stiffer than the
    # soft tissue, the series stiffness pins to the tissue and no amount of
    # grading can move the pressure distribution.
    ratio = float(np.median(E_eff / thickness) / tissue) if tissue else float("inf")
    sol = solve_contact(profile, K, load_n=load_n, cell_area_m2=cell_area,
                        target_cop=target_cop, stiffness_ratio=ratio)
    out = {"design": name, "hypothesis": hypothesis or "solid",
           "rho_mean": float(rho.mean()), "rho_min": float(rho.min()),
           "rho_max": float(rho.max()),
           "E_eff_mean_mpa": float(E_eff.mean() / 1e6), **sol.summary()}
    return out, sol, rho


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", default=None,
                    help="stiffness calibration JSON (default: newest in outputs/)")
    ap.add_argument("--clip", default="P1/FP/1")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--grid", type=int, nargs=2, default=[24, 56])
    ap.add_argument("--load-n", type=float, default=700.0, help="nominal body weight")
    ap.add_argument("--thickness-mm", type=float, default=10.0)
    ap.add_argument("--modulus-mpa", type=float, default=40.0)
    ap.add_argument("--rho-min", type=float, default=0.20)
    ap.add_argument("--rho-max", type=float, default=0.55)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--tissue-pa-per-m", type=float, default=SOFT_TISSUE_STIFFNESS_PA_PER_M)
    ap.add_argument("--sweep-modulus", action="store_true",
                    help="repeat across TPU modulus 20-80 MPa and thickness 6-14 mm")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cal_path = Path(args.calibration) if args.calibration else None
    if cal_path is None:
        found = sorted(OUT.glob("stiffness_calibration_*.json"))
        if not found:
            print("no stiffness calibration found. Run:\n"
                  "  envs/core/bin/python scripts/calibrate_lattice_stiffness.py")
            return 1
        cal_path = found[-1]
    fit = load_calibration(cal_path)
    print(f"stiffness law: {fit.describe()}   [{cal_path.name}]")

    grid = tuple(args.grid)
    field, clip_key, frame = measured_pressure_field(args.clip, args.side, grid)
    measured = field.masked() if hasattr(field, "masked") else np.asarray(field)
    measured = np.nan_to_num(np.asarray(measured, dtype=float), nan=0.0)
    norm = normalise_pressure(measured)
    profile = plantar_profile(grid)

    tot = measured.sum()
    iu, iv = np.indices(measured.shape)
    target_cop = ((measured * iu).sum() / tot, (measured * iv).sum() / tot)
    print(f"driving field: {clip_key} {args.side}, peak-load frame {frame}, "
          f"COP=({target_cop[0]:.1f}, {target_cop[1]:.1f})")

    # Physical cell size from a nominal foot: 100 mm wide x 260 mm long.
    cell_area = (0.100 / grid[0]) * (0.260 / grid[1])

    combos = [(args.modulus_mpa, args.thickness_mm)]
    if args.sweep_modulus:
        # Spans foam-like (1 MPa) to rigid TPU (80 MPa). The soft end is included
        # deliberately: at 40 MPa the lattice turns out to be far stiffer than
        # plantar tissue, so the sweep has to reach the regime where the insole
        # can actually influence the pressure distribution.
        combos = [(E, h) for E in (1.0, 5.0, 20.0, 40.0, 80.0) for h in (6.0, 14.0)]

    all_rows, fields = [], {}
    for E_mpa, h_mm in combos:
        for name, hyp in DESIGNS.items():
            row, sol, rho = run_design(
                name, hyp, norm, profile, fit,
                E_s=E_mpa * 1e6, thickness=h_mm / 1000.0, load_n=args.load_n,
                cell_area=cell_area, target_cop=target_cop,
                rho_min=args.rho_min, rho_max=args.rho_max, gamma=args.gamma,
                tissue=args.tissue_pa_per_m)
            row.update({"E_s_mpa": E_mpa, "thickness_mm": h_mm})
            all_rows.append(row)
            if (E_mpa, h_mm) == (args.modulus_mpa, args.thickness_mm):
                fields[name] = {"pressure_pa": sol.pressure_pa, "rho": rho}

    report = {
        "note": ("Simulation. Winkler foundation, linear elastic TPU, rigid foot with "
                 "a compliant tissue layer. Pressures in kPa are simulated outputs; "
                 "the measured insole field supplies only the normalised load shape "
                 "and the COP, and remains in raw sensor counts."),
        "calibration": cal_path.name,
        "stiffness_law": fit.describe(),
        "driving_clip": clip_key, "side": args.side, "frame": frame,
        "load_n": args.load_n, "cell_area_m2": cell_area,
        "target_cop": list(target_cop),
        "tissue_pa_per_m": args.tissue_pa_per_m,
        "results": all_rows,
    }
    (OUT / "insole_designs.json").write_text(json.dumps(report, indent=2))
    if fields:
        np.savez_compressed(OUT / "insole_design_fields.npz",
                            profile=profile, measured=measured,
                            **{f"{k}_p": v["pressure_pa"] for k, v in fields.items()},
                            **{f"{k}_rho": v["rho"] for k, v in fields.items()})

    # --- report -----------------------------------------------------------
    base = {(r["E_s_mpa"], r["thickness_mm"]): r for r in all_rows
            if r["design"] == "B_uniform"}
    print(f"\n{'E(MPa)':>7}{'h(mm)':>7}{'k_ins/k_tis':>12}{'design':>12}"
          f"{'peak kPa':>10}{'area cm2':>10}{'vs uniform':>12}")
    for r in all_rows:
        b = base[(r["E_s_mpa"], r["thickness_mm"])]
        rel = 100 * (r["peak_pressure_kpa"] - b["peak_pressure_kpa"]) / b["peak_pressure_kpa"]
        print(f"{r['E_s_mpa']:>7.1f}{r['thickness_mm']:>7.0f}"
              f"{r['stiffness_ratio_insole_over_tissue']:>12.1f}{r['design']:>12}"
              f"{r['peak_pressure_kpa']:>10.1f}"
              f"{r['contact_area_cm2']:>10.1f}{rel:>+11.1f}%")

    print("\npeak-pressure change vs uniform lattice, across all conditions:")
    for d in DESIGNS:
        if d == "B_uniform":
            continue
        vals = [100 * (r["peak_pressure_kpa"]
                       - base[(r["E_s_mpa"], r["thickness_mm"])]["peak_pressure_kpa"])
                / base[(r["E_s_mpa"], r["thickness_mm"])]["peak_pressure_kpa"]
                for r in all_rows if r["design"] == d]
        v = np.array(vals)
        print(f"  {d:>12}: mean {v.mean():+6.1f}%   range {v.min():+6.1f}..{v.max():+6.1f}%")
    print("\nwhere does grading start to matter?  (|peak change| vs uniform)")
    print(f"  {'E(MPa)':>7}{'h(mm)':>7}{'k_ins/k_tis':>12}{'stiffen':>10}{'soften':>9}")
    for key in sorted(base):
        rows = [r for r in all_rows if (r["E_s_mpa"], r["thickness_mm"]) == key]
        b = base[key]
        d = {r["design"]: 100 * (r["peak_pressure_kpa"] - b["peak_pressure_kpa"])
             / b["peak_pressure_kpa"] for r in rows}
        print(f"  {key[0]:>7.1f}{key[1]:>7.0f}"
              f"{b['stiffness_ratio_insole_over_tissue']:>12.1f}"
              f"{d.get('C_stiffen', 0):>+9.1f}%{d.get('D_soften', 0):>+8.1f}%")

    print(f"\nwrote {(OUT / 'insole_designs.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
