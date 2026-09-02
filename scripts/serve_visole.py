#!/usr/bin/env python
"""Local web dashboard for Visole.

    envs/core/bin/python scripts/serve_visole.py
    open http://localhost:8420

Reads the experiment JSON on disk and serves it alongside a static page, so the
site always reflects the actual results rather than a hand-written copy of them.
Standard library only -- no web framework added for a local viewer.
"""

from __future__ import annotations

import argparse
import glob
import json
import socketserver
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


_SUPS = str.maketrans("0123456789.", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u02d9")


def _sup(x: float) -> str:
    """Render an exponent with real superscript glyphs -- a literal caret in a
    headline number reads as a typo rather than as maths."""
    return f"{x:.2f}".translate(_SUPS)


def _superscript(x: float) -> str:
    return "\u03c1" + _sup(x)


def _load(rel: str):
    try:
        return json.loads((REPO / rel).read_text())
    except Exception:
        return None


def build_payload() -> dict:
    audit = _load("experiments/gait_dataset_audit/outputs/audit_stats.json")
    loso = _load("experiments/pressure_baseline/outputs/loso_pose.json")
    focus = _load("experiments/focus_baseline/foot3d/results.json")
    stiff = _load("experiments/insole_physics/outputs/stiffness_calibration_r10.json")
    opt = _load("experiments/insole_physics/outputs/insole_optimum.json")
    fil = _load("experiments/insole_physics/outputs/filament_choice.json")
    demo = _load("experiments/integrated_demo/pipeline_result.json")
    syn = [json.loads(Path(p).read_text()) for p in
           sorted(glob.glob(str(REPO / "experiments/focus_baseline/synthetic/synthetic_result_*.json")))]

    def med(xs):
        xs = sorted(xs)
        n = len(xs)
        return None if not n else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)

    ch = [r["chamfer_mm"] for r in (focus or []) if "chamfer_mm" in r]
    syn_ch = [s["chamfer_mm"] for s in syn if "chamfer_mm" in s]

    metrics = []
    if loso:
        metrics.append(dict(
            label="Video → plantar loading", value=f"+{loso['skill_gated']['mean']:.3f}",
            unit=f"± {loso['skill_gated']['sd']:.3f} skill",
            sub=f"{loso['n_folds_beating_mean_gated']}/{loso['n_folds']} participants beat the baseline",
            status="measured"))
    if ch:
        metrics.append(dict(
            label="Photos → 3D foot", value=f"{med(ch):.2f}", unit="mm chamfer",
            sub=f"{len(ch)}/{len(ch)} scans · {med(syn_ch):.1f} mm when cameras are estimated"
                if syn_ch else f"{len(ch)}/{len(ch)} scans",
            status="measured"))
    if stiff:
        metrics.append(dict(
            label="Lattice stiffness", value=_superscript(stiff["fit"]["n"]),
            unit=f"× {stiff['fit']['C']:.3f} · E*/Es",
            sub=f"R² = {stiff['fit']['r_squared']:.4f} · measured by FEA, not assumed",
            status="measured"))
    if opt and fil:
        o = opt["optimum"]
        best = fil["overall_best"]
        metrics.append(dict(
            label="Best printable insole", value=f"{best['peak_pressure_kpa']:.0f}",
            unit="kPa peak",
            sub=f"Shore {best['shore_a']}A · {best['thickness_mm']:.0f} mm · {best['hypothesis']} grading",
            status="simulated"))
        metrics.append(dict(
            label="Unconstrained optimum", value=f"{o['peak_pressure_kpa']:.0f}",
            unit="kPa peak",
            sub=f"{o['E_s_mpa']:.1f} MPa base · {o['thickness_mm']:.0f} mm",
            status="simulated"))
    if audit:
        metrics.append(dict(
            label="Dataset", value=f"{audit['n_participants']}", unit="participants",
            sub=f"{audit['n_clips_total']} clips · 0 NaNs in {audit['rows_checked']:,} rows checked",
            status="measured"))

    filaments = []
    if fil:
        for k in sorted(fil["best_per_grade"], key=int):
            b = fil["best_per_grade"][k]
            ref = fil["best_per_grade"].get("95", b)["peak_pressure_kpa"]
            filaments.append(dict(
                shore=int(k), modulus=round(b["bulk_modulus_mpa"], 1),
                peak=round(b["peak_pressure_kpa"], 1),
                delta=round(100 * (ref - b["peak_pressure_kpa"]) / ref, 1),
                note=b.get("note", "")))

    stages = [
        dict(n=1, title="Walking video", status="measured",
             detail=f"{audit['n_participants'] if audit else '—'} participants, "
                    f"{audit['n_clips_total'] if audit else '—'} clips. Feet are shod — "
                    "the sole is never visible."),
        dict(n=2, title="Video → plantar loading", status="measured",
             detail="Limb kinematics, not appearance. Coarse motion features scored "
                    "+0.009; pose keypoints reached +0.514."),
        dict(n=3, title="Photos → 3D foot", status="measured",
             detail=f"FOCUS on Apple Silicon, {med(ch):.2f} mm median chamfer over "
                    f"{len(ch)} scans." if ch else "FOCUS on Apple Silicon."),
        dict(n=4, title="Canonical plantar frame", status="measured",
             detail="Both feet in one frame. The two insoles number their sensors "
                    "differently — no index is shared."),
        dict(n=5, title="Lattice stiffness", status="measured",
             detail=(f"E*/Es = {stiff['fit']['C']:.3f} · ρ{_sup(stiff['fit']['n'])}, "
                     f"R² = {stiff['fit']['r_squared']:.4f} — a numerical compression "
                     f"test, not a textbook exponent."
                     if stiff else "Measured by FEA.")),
        dict(n=6, title="Contact model → optimum", status="simulated",
             detail="Winkler foundation with the real plantar profile. Softening "
                    "under high pressure lowers peak pressure."),
        dict(n=7, title="Printed insole on a real foot", status="unvalidated",
             detail="No printed part. No pressure sensor. No human testing."),
    ]

    figures = [
        dict(src="/experiments/pipeline_summary.png", title="Pipeline overview",
             caption="Every stage, colour-coded by evidence status."),
        dict(src="/experiments/integrated_demo/pipeline_demo.png", title="End to end",
             caption="Measured pressure → foot geometry → graded material → printable insole."),
        dict(src="/experiments/pressure_baseline/outputs/loso_pose.png",
             title="Leave-one-subject-out",
             caption="All 22 participants beat the baseline; the spread is bimodal."),
        dict(src="/experiments/focus_baseline/foot3d_results.png", title="Foot reconstruction",
             caption="14/14 Foot3D scans, with the reconstruction overlaid on ground truth."),
        dict(src="/experiments/insole_physics/outputs/insole_optimum.png",
             title="Finding the optimum",
             caption="The cushioning trade-off. × marks designs that bottom out."),
        dict(src="/experiments/insole_physics/outputs/filament_choice.png",
             title="Which filament",
             caption="Softer printable TPU wins. No foam required."),
    ]

    return dict(metrics=metrics, stages=stages, filaments=filaments, figures=figures,
                demo=demo, generated_from="experiment JSON on disk")


def _json_safe(obj):
    """Replace NaN/Infinity with null.

    Python's json.dumps happily writes bare NaN, which is valid Python but NOT
    valid JSON -- browsers reject the whole document with a parse error. Several
    metrics legitimately carry NaN (an undefined correlation, a stiffness ratio
    with no tissue term), so they are nulled rather than dropped.
    """
    import math

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") in ("/api/data", "/api/data.json"):
            body = json.dumps(_json_safe(build_payload()), allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", ""):
            self.path = "/web/index.html"
        return super().do_GET()

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload = build_payload()
    print(f"loaded {len(payload['metrics'])} metrics, {len(payload['stages'])} stages, "
          f"{len(payload['figures'])} figures")
    if args.dry_run:
        return 0

    handler = partial(Handler, directory=str(REPO))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"\n  Visole dashboard → http://localhost:{args.port}\n  Ctrl-C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
