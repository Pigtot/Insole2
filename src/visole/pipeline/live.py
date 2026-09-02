"""Run the Visole chain on an arbitrary video, reporting each stage as it lands.

Used by the local dashboard so a user can drop in a clip and watch the pipeline
work through it. Every stage writes a preview image and a short set of numbers,
and calls back before starting the next one.

Read this before believing any output
-------------------------------------
The loading model was trained on **one fixed camera, in one room, with shod
feet**, from a 22-participant dataset. Running it on an arbitrary phone video is
out of distribution. What comes back is *illustrative of the pipeline*, not a
measurement of the person in the video, and the dashboard says so on the stage
itself rather than in a footnote.

Stages that are honest regardless of the video: pose detection, and everything
downstream of the pressure field (density mapping, contact mechanics, geometry)
which is deterministic given its input.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "models" / "pressure_model.npz"

STAGE_TITLES = [
    ("video", "Video"),
    ("pose", "Pose detection"),
    ("features", "Kinematic features"),
    ("pressure", "Predicted plantar loading"),
    ("canonical", "Canonical plantar frame"),
    ("density", "Pressure → material"),
    ("contact", "Contact simulation"),
    ("geometry", "Graded insole"),
]


@dataclass
class Stage:
    key: str
    title: str
    status: str = "pending"          # pending | running | done | failed | skipped
    detail: str = ""
    metrics: dict = field(default_factory=dict)
    preview: str | None = None
    preview_w: int = 0
    preview_h: int = 0
    seconds: float | None = None
    caveat: str | None = None

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("key", "title", "status", "detail", "metrics", "preview",
                 "preview_w", "preview_h", "seconds", "caveat")}


class LivePipeline:
    """Runs the chain, calling ``on_update`` whenever a stage changes state."""

    def __init__(self, video_path: Path, out_dir: Path,
                 on_update: Callable[[list[Stage]], None] | None = None,
                 max_frames: int = 48, stride: int = 4):
        self.video = Path(video_path)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.on_update = on_update or (lambda s: None)
        self.max_frames = max_frames
        self.stride = stride
        self.stages = [Stage(k, t) for k, t in STAGE_TITLES]
        self._by_key = {s.key: s for s in self.stages}
        self.grid = (24, 56)

    # -- plumbing ---------------------------------------------------------
    def _emit(self):
        self.on_update(self.stages)

    def _start(self, key: str) -> tuple[Stage, float]:
        s = self._by_key[key]
        s.status = "running"
        self._emit()
        return s, time.time()

    def _done(self, s: Stage, t0: float, detail: str, metrics=None, preview=None,
              caveat=None):
        s.status = "done"
        s.detail = detail
        s.metrics = metrics or {}
        if preview is not None:
            s.preview = preview
            s.preview_w, s.preview_h = self._dims(self.out / preview)
        else:
            s.preview = None
        s.caveat = caveat
        s.seconds = round(time.time() - t0, 1)
        self._emit()

    def _fail(self, s: Stage, t0: float, msg: str):
        s.status = "failed"
        s.detail = msg
        s.seconds = round(time.time() - t0, 1)
        self._emit()

    # Previews are rendered at 2x their display size so they stay sharp on a
    # Retina panel, and photographs are written as JPEG -- a photographic frame
    # as PNG was 375 KB where JPEG is ~40 KB for the same visible quality.
    PREVIEW_DPI = 200
    JPEG_QUALITY = 88

    def _fig(self, name: str):
        import matplotlib
        matplotlib.use("Agg")
        return self.out / f"{name}.png"

    def _save_photo(self, name: str, bgr) -> Path:
        """Write a photographic frame as JPEG at a sensible display width."""
        import cv2

        target_w = 1200
        h, w = bgr.shape[:2]
        if w > target_w:
            bgr = cv2.resize(bgr, (target_w, int(h * target_w / w)),
                             interpolation=cv2.INTER_AREA)
        path = self.out / f"{name}.jpg"
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
        return path

    def _save_plot(self, fig, name: str) -> Path:
        import matplotlib.pyplot as plt

        path = self.out / f"{name}.png"
        fig.savefig(path, dpi=self.PREVIEW_DPI, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
        return path

    @staticmethod
    def _dims(path: Path) -> tuple[int, int]:
        """Pixel size of a PNG or JPEG, read from the file header."""
        try:
            data = path.read_bytes()
        except Exception:
            return (0, 0)
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return (int.from_bytes(data[16:20], "big"),
                    int.from_bytes(data[20:24], "big"))
        if data[:2] == b"\xff\xd8":                      # JPEG: walk the segments
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                              0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    return (int.from_bytes(data[i + 7:i + 9], "big"),
                            int.from_bytes(data[i + 5:i + 7], "big"))
                i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
        return (0, 0)

    def _plantar_plot(self, field, name: str, label: str, cmap: str):
        """Plantar maps in a LANDSCAPE frame.

        Rendered portrait these are ~0.52 aspect, and a fixed display height
        squeezed them to 120 px wide -- too narrow to read. Pairing the map with
        its colour bar and label in a landscape figure keeps the foot large while
        the card stays a sensible shape.
        """
        import matplotlib.pyplot as plt

        fig, (ax, cax) = plt.subplots(
            1, 2, figsize=(4.6, 3.4), gridspec_kw={"width_ratios": [1, .06]})
        im = ax.imshow(np.asarray(field, dtype=float).T, origin="lower",
                       aspect="equal", cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("medial → lateral", fontsize=8, color="#666")
        ax.set_ylabel("heel → toe", fontsize=8, color="#666")
        for sp in ax.spines.values():
            sp.set_edgecolor("#ccc")
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=8, color="#666")
        cb.ax.tick_params(labelsize=7)
        fig.tight_layout()
        return self._save_plot(fig, name)

    # -- stages -----------------------------------------------------------
    def run(self) -> list[Stage]:
        try:
            info = self._stage_video()
            seq = self._stage_pose(info)
            feats = self._stage_features(seq)
            pred = self._stage_pressure(feats)
            field = self._stage_canonical(pred)
            rho, tfield, footprint, profile = self._stage_density(field)
            ok = self._stage_contact(rho, profile, field)
            self._stage_geometry(footprint, tfield, ok)
        except Exception as exc:  # surface the failure, do not swallow it
            for s in self.stages:
                if s.status in ("pending", "running"):
                    s.status = "failed"
                    s.detail = f"{type(exc).__name__}: {exc}"
            self._emit()
        return self.stages

    def _stage_video(self):
        s, t0 = self._start("video")
        import cv2
        from visole.data.insole_gaitrite import probe_video

        info = probe_video(self.video)
        if info is None:
            raise RuntimeError("could not read the video")
        cap = cv2.VideoCapture(str(self.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, info.n_frames // 3))
        okf, frame = cap.read()
        cap.release()
        prev = self._save_photo("00_video", frame) if okf else None
        self._done(s, t0,
                   f"{info.width}×{info.height}, {info.fps:.1f} fps, {info.duration_s:.1f} s",
                   {"frames": info.n_frames, "fps": round(info.fps, 2)},
                   prev.name if prev else None)
        return info

    def _stage_pose(self, info):
        s, t0 = self._start("pose")
        import cv2
        import matplotlib.pyplot as plt
        from visole.compute.device import resolve_device
        from visole.models.pose_features import extract_clip_pose, load_pose_model

        class _Shim:
            video_path = self.video
            video_info = info

        device = resolve_device("auto")
        model = load_pose_model(device)
        seq = extract_clip_pose(_Shim(), model, device, stride=self.stride,
                                max_frames=self.max_frames)
        if seq is None or not seq.detected.any():
            raise RuntimeError("no person detected in the video")

        j = int(np.argmax(seq.detected))
        cap = cv2.VideoCapture(str(self.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seq.times[j] * info.fps))
        okf, frame = cap.read()
        cap.release()
        prev = None
        if okf:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(6, 6 * info.height / info.width))
            ax.imshow(rgb)
            k = seq.keypoints[j]
            good = seq.scores[j] > 2.0
            ax.scatter(k[good, 0], k[good, 1], s=44, c="#00e0a0",
                       edgecolors="black", linewidths=.7, zorder=3)
            for a, b in ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
                         (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)):
                if good[a] and good[b]:
                    ax.plot(k[[a, b], 0], k[[a, b], 1], lw=2.2, c="#00e0a0", zorder=2)
            ax.axis("off")
            buf = self.out / "_pose_tmp.png"
            fig.savefig(buf, dpi=self.PREVIEW_DPI, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            prev = self._save_photo("01_pose", cv2.imread(str(buf)))
            buf.unlink(missing_ok=True)
        rate = 100 * seq.detected.mean()
        self._done(s, t0, f"{int(seq.detected.sum())} of {len(seq.detected)} sampled "
                          f"frames had a person detected ({rate:.0f}%)",
                   {"frames": len(seq.detected), "detection_rate": round(rate, 1)},
                   prev.name if prev else None)
        return seq

    def _stage_features(self, seq):
        s, t0 = self._start("features")
        from visole.models.pose_features import pose_to_features
        from visole.models.video_features import add_temporal_context

        feats = pose_to_features(seq)
        ctx = add_temporal_context(feats, (-6, -3, 0, 3, 6))
        valid = np.isfinite(ctx).all(axis=1)
        if valid.sum() < 3:
            raise RuntimeError("too few usable frames after feature extraction")
        self._done(s, t0,
                   f"{ctx.shape[1]} features per frame, {int(valid.sum())} usable frames. "
                   "Ankle, knee and hip positions normalised by torso length.",
                   {"n_features": int(ctx.shape[1]), "usable_frames": int(valid.sum())})
        return ctx[valid]

    def _stage_pressure(self, X):
        s, t0 = self._start("pressure")
        import matplotlib.pyplot as plt

        if not MODEL.exists():
            raise RuntimeError("no trained model; run scripts/export_pressure_model.py")
        m = np.load(MODEL)

        def ridge(pref, Xa):
            Xs = (Xa - m[f"{pref}_mu"]) / m[f"{pref}_sd"]
            Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
            return Xs @ m[f"{pref}_W"] + m[f"{pref}_ymu"]

        out = np.zeros((len(X), 64))
        for j, side in enumerate(("left", "right")):
            cols = slice(0, 32) if side == "left" else slice(32, 64)
            z = ((X - m[f"gate_{side}_mean"]) / m[f"gate_{side}_scale"]) @ \
                m[f"gate_{side}_coef"] + m[f"gate_{side}_intercept"]
            prob = 1.0 / (1.0 + np.exp(-z))
            out[:, cols] = prob[:, None] * np.clip(ridge(f"mag_{side}", X), 0, None)
        pred = np.clip(out, 0, None)

        peak = int(pred.sum(1).argmax())
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.plot(pred[:, :32].sum(1), lw=1.8, label="left", color="#3a7bd5")
        ax.plot(pred[:, 32:].sum(1), lw=1.8, label="right", color="#d5723a")
        ax.axvline(peak, color="#888", ls="--", lw=1)
        ax.set_xlabel("frame"); ax.set_ylabel("total load\n(raw counts)", fontsize=9)
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.25)
        fig.tight_layout()
        prev = self._save_plot(fig, "03_pressure")

        self._done(s, t0,
                   f"Peak total load at frame {peak}. Units are raw sensor counts, not kPa.",
                   {"peak_frame": peak, "peak_counts": round(float(pred[peak].sum()), 1)},
                   prev.name,
                   caveat="The model was trained on one fixed camera in one room with "
                          "shod feet. On any other video this is out of distribution — "
                          "illustrative of the pipeline, not a measurement of this person.")
        return pred[peak]

    def _stage_canonical(self, pred64):
        s, t0 = self._start("canonical")
        import matplotlib.pyplot as plt
        from visole.pressure.canonical_foot import lift_to_canonical
        from visole.pressure.sensor_map import load_sensor_map

        smap = load_sensor_map("left")
        field = lift_to_canonical(pred64[:32], smap, grid=self.grid)
        vals = np.nan_to_num(np.asarray(field.masked(), dtype=float), nan=0.0)
        prev = self._plantar_plot(vals, "04_canonical", "predicted load (raw counts)",
                                  "magma")
        self._done(s, t0, "32 predicted channels lifted onto a shared plantar grid.",
                   {"grid": f"{self.grid[0]}×{self.grid[1]}"}, prev.name)
        return vals

    def _stage_density(self, vals):
        s, t0 = self._start("density")
        import matplotlib.pyplot as plt
        from visole.lattice.mapping import DensityMapping, normalise_pressure
        from visole.lattice.implicit import calibrate_density_vs_thickness
        from visole.lattice.mapping import DensityToThickness
        import sys as _sys
        _sys.path.insert(0, str(REPO / "scripts"))
        from simulate_insole_designs import plantar_profile

        profile = plantar_profile(self.grid)
        footprint = profile < profile.max() - 1e-9
        norm = normalise_pressure(vals)
        mapping = DensityMapping(hypothesis="soften")
        rho = np.where(footprint, mapping(norm), mapping.rho_min)
        d2t = DensityToThickness(calibrate_density_vs_thickness())
        tfield = d2t(rho)

        prev = self._plantar_plot(np.where(footprint, rho, np.nan), "05_density",
                                  "relative density", "viridis")
        self._done(s, t0,
                   "Soften mapping: higher predicted pressure → lower lattice density.",
                   {"rho_min": round(float(rho[footprint].min()), 3),
                    "rho_max": round(float(rho[footprint].max()), 3)}, prev.name)
        return rho, tfield, footprint, profile

    def _stage_contact(self, rho, profile, vals):
        s, t0 = self._start("contact")
        import json
        import matplotlib.pyplot as plt
        from visole.lattice.stiffness import GibsonAshby
        from visole.registration.contact import series_stiffness, solve_contact

        cal = json.loads((REPO / "experiments/insole_physics/outputs"
                          "/stiffness_calibration_r10.json").read_text())["fit"]
        fit = GibsonAshby(C=cal["C"], n=cal["n"], rho=np.array([.3]),
                          E_ratio=np.array([.1]), r_squared=1.0)
        E_s, h = 5.5e6, 0.014                      # Shore 70A, 14 mm
        E_eff = fit(rho) * E_s
        K = series_stiffness(E_eff, h)
        cell = (0.100 / self.grid[0]) * (0.260 / self.grid[1])
        tot = vals.sum()
        iu, iv = np.indices(vals.shape)
        cop = ((vals * iu).sum() / tot, (vals * iv).sum() / tot) if tot > 0 else None
        sol = solve_contact(profile, K, load_n=700.0, cell_area_m2=cell,
                            target_cop=cop, thickness_m=h,
                            insole_stiffness_pa_per_m=E_eff / h)
        sm = sol.summary()
        ok = sm["bottomed_fraction"] <= 0.02 and sol.converged

        prev = self._plantar_plot(sol.pressure_pa / 1e3, "06_contact",
                                  "simulated pressure (kPa)", "magma")

        self._done(s, t0,
                   f"Shore 70A, 14 mm. Peak {sm['peak_pressure_kpa']:.0f} kPa over "
                   f"{sm['contact_area_cm2']:.0f} cm². "
                   + ("Accepted." if ok else "REJECTED — bottoms out under load."),
                   {"peak_kpa": round(sm["peak_pressure_kpa"], 1),
                    "contact_cm2": round(sm["contact_area_cm2"], 1),
                    "bottomed_pct": round(100 * sm["bottomed_fraction"], 1)},
                   prev.name,
                   caveat="Simulated under a Winkler foundation model with a nominal "
                          "700 N load. kPa here is a model output, not a measurement.")
        return ok

    def _stage_geometry(self, footprint, tfield, ok):
        s, t0 = self._start("geometry")
        if not ok:
            s.status = "skipped"
            s.detail = "Skipped: the contact stage rejected this design."
            self._emit()
            return
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from visole.lattice.implicit import generate_insole_lattice

        lat = generate_insole_lattice(footprint, tfield, height_mm=14.0,
                                      resolution=4, rim_mm=2.5)
        stl = self.out / "insole.stl"
        lat.export_stl(stl)
        ls = lat.summary()

        V = np.asarray(lat.mesh.vertices); F = np.asarray(lat.mesh.faces)
        step = max(1, len(F) // 20000)
        ex = V.max(0) - V.min(0)
        fig = plt.figure(figsize=(9.6, 3.6))
        for k, (el, az, ttl) in enumerate(
                ((90, -90, "top"), (12, -90, "side"), (34, -62, "oblique"))):
            ax = fig.add_subplot(1, 3, k + 1, projection="3d")
            ax.add_collection3d(Poly3DCollection(V[F[::step]], facecolor="#5b7fa6",
                                                 edgecolor="none"))
            ax.set_xlim(V[:, 0].min(), V[:, 0].max())
            ax.set_ylim(V[:, 1].min(), V[:, 1].max())
            ax.set_zlim(V[:, 2].min(), V[:, 2].max())
            ax.set_box_aspect(tuple(ex)); ax.view_init(elev=el, azim=az)
            ax.set_axis_off(); ax.set_title(ttl, fontsize=9, color="#666")
        fig.tight_layout()
        prev = self._save_plot(fig, "07_geometry")

        self._done(s, t0,
                   f"{ls['n_faces']:,} faces, watertight={ls['watertight']}, "
                   f"connected={ls['connected']}. STL ready to slice.",
                   {"faces": ls["n_faces"], "volume_cm3": round((ls["volume_mm3"] or 0) / 1000, 1),
                    "watertight": ls["watertight"]},
                   prev.name)
