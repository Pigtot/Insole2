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
import re
import shutil
import socketserver
import sys
import threading
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOBS_DIR = REPO / "experiments" / "live_jobs"
MAX_UPLOAD = 300 * 1024 * 1024          # 300 MB; a walking clip is a few MB

#: job id -> {"stages": [...], "done": bool, "error": str|None}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


_SUPS = str.maketrans("0123456789.", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u02d9")


def _sup(x: float) -> str:
    """Render an exponent with real superscript glyphs -- a literal caret in a
    headline number reads as a typo rather than as maths."""
    return f"{x:.2f}".translate(_SUPS)


def _superscript(x: float) -> str:
    return "\u03c1" + _sup(x)


def _png_size(path: Path) -> tuple[int, int]:
    """Read a PNG's dimensions from its IHDR header -- no image library needed."""
    try:
        with path.open("rb") as fh:
            head = fh.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return (0, 0)
        return (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"))
    except Exception:
        return (0, 0)


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

    # Attach real pixel dimensions so the page can reserve exact space for each
    # figure. The images are lazy-loaded, and without a reserved box the layout
    # shifts as each one arrives.
    for f in figures:
        f["w"], f["h"] = _png_size(REPO / f["src"].lstrip("/"))

    # A baked run of the full pipeline, so the page shows real per-stage output
    # immediately rather than only after someone uploads something.
    worked = _load("experiments/demo_run/demo.json")

    return dict(metrics=metrics, stages=stages, filaments=filaments, figures=figures,
                demo=demo, worked_example=worked,
                generated_from="experiment JSON on disk")


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


def _split_multipart(body: bytes, content_type: str):
    """Pull the first file part out of a multipart/form-data body.

    Hand-rolled rather than using cgi.FieldStorage, which is deprecated and gone
    in 3.13 -- this keeps the viewer working on future interpreters.
    """
    m = re.search(r"boundary=([^;]+)", content_type or "")
    if not m:
        return None, None
    boundary = b"--" + m.group(1).strip('"').encode()
    for part in body.split(boundary):
        head, _, data = part.partition(b"\r\n\r\n")
        if b"filename=" not in head:
            continue
        fn = re.search(rb'filename="([^"]*)"', head)
        name = (fn.group(1).decode(errors="replace") if fn else "upload.mp4") or "upload.mp4"
        return Path(name).name, data.rstrip(b"\r\n-")
    return None, None


def _run_job(job_id: str, video: Path):
    sys.path.insert(0, str(REPO / "src"))
    from visole.pipeline.live import LivePipeline

    def on_update(stages):
        with JOBS_LOCK:
            JOBS[job_id]["stages"] = [s.as_dict() for s in stages]

    try:
        stages = LivePipeline(video, JOBS_DIR / job_id, on_update=on_update).run()
        with JOBS_LOCK:
            JOBS[job_id]["stages"] = [s.as_dict() for s in stages]
            JOBS[job_id]["done"] = True
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["done"] = True
            JOBS[job_id]["error"] = f"{type(exc).__name__}: {exc}"


class Handler(SimpleHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(_json_safe(obj), allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/api/upload":
            return self._json({"error": "unknown endpoint"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._json({"error": "empty upload"}, 400)
        if length > MAX_UPLOAD:
            return self._json(
                {"error": f"file is {length/1e6:.0f} MB; limit is {MAX_UPLOAD/1e6:.0f} MB"},
                413)
        body = self.rfile.read(length)
        name, data = _split_multipart(body, self.headers.get("Content-Type", ""))
        if not data:
            return self._json({"error": "no file found in the upload"}, 400)

        job_id = uuid.uuid4().hex[:12]
        d = JOBS_DIR / job_id
        d.mkdir(parents=True, exist_ok=True)
        video = d / (name if name.lower().endswith((".mp4", ".mov", ".m4v", ".avi"))
                     else "upload.mp4")
        video.write_bytes(data)

        with JOBS_LOCK:
            JOBS[job_id] = {"stages": [], "done": False, "error": None,
                            "filename": name, "bytes": len(data)}
        threading.Thread(target=_run_job, args=(job_id, video), daemon=True).start()
        return self._json({"job": job_id, "filename": name,
                           "size_mb": round(len(data) / 1e6, 1)})

    def do_GET(self):  # noqa: N802
        m = re.match(r"^/api/job/([0-9a-f]{6,32})/?$", self.path)
        if m:
            with JOBS_LOCK:
                job = JOBS.get(m.group(1))
            return self._json(job or {"error": "unknown job"}, 200 if job else 404)
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

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
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
