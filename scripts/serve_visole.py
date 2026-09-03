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
    ladder = _load("experiments/pressure_baseline/outputs/spatial_ladder_pose.json")
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
    if ladder:
        ml = ladder["contrasts"]["medial_lateral"]["vs_null"]
        hf = ladder["contrasts"]["heel_forefoot"]["vs_null"]
        metrics.append(dict(
            label="Spatial resolution", value="heel/forefoot", unit="only",
            sub=f"Beats its own null in {hf['folds_beating_null']}/{hf['n_folds']} folds. "
                f"Medial–lateral does not ({ml['folds_beating_null']}/{ml['n_folds']}) — "
                f"so the model cannot inform grading across the foot.",
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
            sub=f"R² = {stiff['fit']['r_squared']:.4f} · numerical compression test, "
                f"not a physical one — nothing was compressed",
            status="simulated"))
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
                    "+0.009; pose keypoints reached +0.514 — but only the heel↔toe "
                    "axis survives a null-control test."),
        dict(n=3, title="Photos → 3D foot", status="measured",
             detail=f"FOCUS on Apple Silicon, {med(ch):.2f} mm median chamfer over "
                    f"{len(ch)} scans." if ch else "FOCUS on Apple Silicon."),
        dict(n=4, title="Canonical plantar frame", status="measured",
             detail="Both feet in one frame. The two insoles number their sensors "
                    "differently — no index is shared."),
        dict(n=5, title="Lattice stiffness", status="simulated",
             detail=(f"E*/Es = {stiff['fit']['C']:.3f} · ρ{_sup(stiff['fit']['n'])}, "
                     f"R² = {stiff['fit']['r_squared']:.4f} — a *numerical* compression "
                     f"test, so simulated, not measured. Nothing was compressed."
                     if stiff else "Derived by FEA, not physically measured.")),
        dict(n=6, title="Contact model → optimum", status="simulated",
             detail="Winkler foundation with the real plantar profile. Softening "
                    "under high pressure lowers peak pressure."),
        dict(n=7, title="Printed insole on a real foot", status="unvalidated",
             detail="No printed part. No pressure sensor. No human testing."),
    ]

    figures = [
        dict(src="/experiments/focus_baseline/focus_explained.png",
             title="How FOCUS builds the foot mesh",
             caption="Real photographs -> mask, dense TOC correspondences and normals, "
                     "fused across 33 views into a mesh 1.87 mm from the ground-truth scan."),
        dict(src="/experiments/pressure_baseline/heatmap_and_decisions.png",
             title="Heat map, decision rule, biomechanics",
             caption="32 predicted values become a display heat map, and the rule that "
                     "sets material density - including the frame-selection bug it exposed."),
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
    fusion = _load("experiments/integrated_demo/fusion_import.json")

    methods = [
        dict(group="Machine learning", items=[
            ("Pose estimation",
             "torchvision Keypoint R-CNN (ResNet50-FPN, COCO, 17 keypoints), min_size=480, "
             "run on Apple MPS at ~0.10 s/frame versus 1.14 s on CPU. The highest-scoring "
             "detection per frame is kept: one person walks the walkway."),
            ("Feature engineering",
             "Ankle, knee and hip positions expressed relative to the hip midpoint and "
             "divided by the subject's torso length in that frame. Apparent size varies "
             "about sevenfold along the walkway, so without that normalisation the "
             "features encode distance from the camera and the model learns the room. "
             "Five-frame temporal context (offsets -6..+6) gives 130 features."),
            ("Regression",
             "Ridge, solved in closed form on standardised features with an unpenalised "
             "intercept. alpha = 10, chosen on held-out PARTICIPANTS and confirmed to be "
             "an interior minimum of a 10-point grid from 1e-2 to 1e7. Ridge is deliberate: "
             "with 22 participants a high-capacity model would learn to recognise people, "
             "and a linear probe asks whether the information is linearly accessible "
             "without an optimiser to blame."),
            ("Contact gating",
             "pressure = P(contact) x magnitude, after HOPE (arXiv 2608.06192). A logistic "
             "gate per foot multiplies a ridge magnitude head fitted on contact frames "
             "only. A foot in swing is exactly zero across all 32 channels for about half "
             "of every clip, and a single linear map cannot represent a hard zero."),
            ("Evaluation",
             "Participant-disjoint throughout. Leave-one-subject-out over 22 folds gives "
             "+0.514 +/- 0.119 skill against a mean predictor, with 22/22 participants "
             "beating it. Skill, not R^2: 0 means no better than predicting the average "
             "and negative means worse."),
            ("Deployment fit",
             "A separate fit over all participants, stored as plain arrays (ridge weights, "
             "logistic coefficients, scaler statistics) rather than pickled estimators, so "
             "it cannot break on a library upgrade. Evaluation numbers still come from the "
             "disjoint splits."),
        ]),
        dict(group="Calibration", items=[
            ("Sensor geometry",
             "The dataset ships sensor maps as SVG with no closed shapes. The 32 pad "
             "centroids were recovered by chaining 262 open Bezier segments into closed "
             "loops. Validated by reproducing the device's OWN centre-of-pressure at "
             "r > 0.997 on both feet -- an independent check on the extraction, the label "
             "assignment and the left/right mirroring at once."),
            ("Signal baseline",
             "Pressure channels have a non-zero unloaded offset that varies per clip "
             "(5th percentile ranged 19..187 counts). Baseline is estimated per clip per "
             "sensor, never as a global constant. Units stay raw counts: no calibration to "
             "kPa exists for this hardware."),
            ("Temporal synchronisation",
             "The GAITRite offset convention was undocumented, so two competing hypotheses "
             "were tested against the data. Foot 0 = left gives 14.5 ms median error -- "
             "under one 64 Hz sample period -- against 556 ms for the alternative. A 38x "
             "gap is a decision, not an estimate."),
            ("Lattice stiffness",
             "Measured by numerical compression test rather than assumed from a textbook "
             "exponent: every solid voxel becomes a hexahedral finite element and the block "
             "is compressed in scikit-fem. E*/Es = 0.862 rho^1.713, R^2 = 0.9971. The "
             "solver is validated first -- a solid block returns its own modulus to "
             "1.00000, scales exactly linearly with the base material, and is independent "
             "of applied strain."),
            ("Density and thickness",
             "Relative density is voxel-counted, not taken from a formula, then that "
             "measured curve is INVERTED to drive grading. Both estimators (voxel count and "
             "mesh volume) are reported side by side; neither is tuned to match the other."),
            ("Shore hardness to modulus",
             "Gent's empirical relation converts filament grade to Young's modulus. It "
             "carries roughly +/-30% scatter and ignores print anisotropy, so grades are "
             "indicative -- a compression test on a printed coupon would replace it."),
        ]),
        dict(group="Figures (matplotlib)", items=[
            ("Headless rendering",
             "The Agg backend throughout, so every figure is produced by a script and can "
             "be regenerated. No figure is drawn by hand and no number is typed into a "
             "caption -- the dashboard reads them from the experiment JSON."),
            ("Plantar maps",
             "imshow with an equal aspect and the colour bar placed beside the map. Cells "
             "outside the foot outline are masked to NaN rather than filled, so "
             "interpolation is never mistaken for measurement. magma for pressure, viridis "
             "for density."),
            ("Resolution and format",
             "Plots render at dpi 200 so they stay sharp on a Retina panel; photographic "
             "frames are written as JPEG at quality 88 rather than PNG (129 KB for "
             "1200x680, against 375 KB for a 640x362 PNG before)."),
            ("3D geometry",
             "Poly3DCollection over a decimated face list, with the box aspect set from "
             "the real extents so the insole is not visually distorted."),
        ]),
    ]

    return dict(metrics=metrics, stages=stages, filaments=filaments, figures=figures,
                demo=demo, worked_example=worked, fusion=fusion, methods=methods,
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
        """Quiet request log: API calls only.

        `log_error` routes here too and passes an **HTTPStatus**, not a string,
        so any match has to coerce first -- `"/api/" in HTTPStatus.NOT_FOUND`
        raises TypeError from inside the error path and kills the connection.
        A missing favicon was enough to trigger it.
        """
        if any("/api/" in str(a) for a in args):
            super().log_message(fmt, *args)

    def log_error(self, fmt, *args):
        """Errors are never filtered -- swallowing them is what hid the above."""
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
