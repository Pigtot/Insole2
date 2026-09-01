# Visole implementation plan — where each part runs

Written before significant model code, per the project brief. Reflects the
machine audit (`reports/system_report.md`) and the dataset audit
(`experiments/gait_dataset_audit/README.md`).

**Machine:** Apple M1 Max, 10 cores (8P/2E), 64 GB unified memory, 1.1 TB free,
macOS 26.5.2 arm64, PyTorch 2.13.0, MPS available and verified by a live matmul
agreeing with CPU to 1.1e-7. No CUDA, and none will be installed.

---

## Execution targets

| Component | Runs on | Why |
| --- | --- | --- |
| Device selection, checkpoint loading | CPU | `visole.compute.device`; checkpoints always map to CPU first, then move |
| Insole CSV / index / splits | CPU (pandas, NumPy) | Tiny; 794 clips of a few hundred rows |
| Video decode, frame streaming | CPU (ffmpeg/OpenCV) | Hardware decode is inside ffmpeg; frames are streamed, never bulk-loaded |
| Sensor-map parsing, COP validation | CPU | Pure geometry, milliseconds |
| Baseline 0 (mean predictor) | CPU | No parameters |
| Baseline 1 (kinematics → loading) | CPU, maybe MPS | Small MLP/GBM; CPU is likely faster than MPS at this size |
| Pose/keypoint extraction | MPS, CPU fallback | Batch 1, streamed |
| Baseline 2 (CNN on foot crops) | **MPS**, batch 1–8 | MobileNetV3/ResNet18 at 224 px |
| Baseline 3 (temporal GRU/TCN) | **MPS** | Short sequences; keep sequence length ≤ 64 |
| Implicit TPMS lattice, marching cubes | CPU (NumPy + scikit-image) | Not a GPU workload at this size |
| Lattice FEA (scikit-fem) | CPU (scipy sparse) | Pure Python, arm64-native. ~55 s per unit-cell solve at resolution 10 |
| Winkler contact solve | CPU (scipy.optimize) | 3 unknowns; milliseconds |
| Fusion 360 | **GUI only, no headless** | v2704.1.53 installed. `adsk.sim` is scriptable but solves in the cloud. Cross-check only, not a pipeline stage |
| Mesh handling, STL/STEP export | CPU (trimesh, later CadQuery) | — |
| FOCUS TOC predictor | **MPS** (verified) | 90 ms/image, 20.2x CPU, matches CPU to 3.2e-6 |
| PyTorch3D geometry | **CPU (mandatory)** | `knn_points` / `sample_points_from_meshes` **segfault** on MPS, and the ATen fallback does not cover them |
| COLMAP | CPU, external binary | 4.1.1 Homebrew, no CUDA |
| nTop | **Not available** | Export contract only (CSV/NPY/JSON + mesh + transform) |

### MPS rules

`PYTORCH_ENABLE_MPS_FALLBACK=1` is set by `visole.compute.device` **before** it
imports torch, because PyTorch reads it at backend init — setting it later is a
no-op. `describe_device()` reports `mps_fallback_effective` so a late import is
visible rather than silently ineffective. `MPSFallbackRecorder` captures the
per-operator fallback warnings so fallbacks are logged, not hidden.

Autocast is a **no-op on MPS** by default (`compute.device.autocast`). MPS
autocast support varies by PyTorch version and silently wrong numerics are worse
than being slow. Enable per experiment, only after verifying against CPU.

Memory discipline: start at batch 1, small images, `--max-samples` / `--max-frames`
on every heavy script, `num_workers` 0–2 on macOS, and reduce batch/resolution on
OOM. Never disable MPS memory limits.

---

## Milestone status

| # | Milestone | Status |
| --- | --- | --- |
| 0 | Machine audit | **Done** — `reports/system_report.md` |
| 1 | Repository foundation | **Done** — 69 tests passing |
| 2 | Public data audit | **Done** — `experiments/gait_dataset_audit/` |
| 3 | Pressure baseline | **Done** — pose kinematics reach skill +0.46 vs mean predictor |
| 4 | FOCUS inference | **Done** — 14/14 Foot3D scans, median **2.93 mm** chamfer, 474 images in 3.2 min |
| 5 | MapAnything comparator | Optional, after 4 |
| 6 | `PressureField` contract | Partly — bins, sensor map, splits, canonical plantar frame exist |
| 7 | Registration simulation | Not started |
| 8 | Lattice demonstration | **PoC done** + **physics**: measured stiffness law, Winkler contact, A/B/C/D compared |
| 9 | Integrated demo | Not started |

---

## The finding that changes the plan

The audit established that participants are **shod** and the feet occupy roughly
50 px to a few hundred px in a fixed room camera. The plantar surface is never
visible.

So the ordering of Baselines 1 and 2 is **not** "simple warm-up, then the real
model". On this dataset:

- **Baseline 1 (kinematics → loading) is the scientifically motivated primary
  method**, following UnderPressure. Motion is the signal that is actually present.
- **Baseline 2 (CNN on crops)** is a genuine test — but of *shoe and limb
  appearance*, and it must be described that way.
- Baseline 0 (predict the training mean) decides whether either learned anything.

The interesting result is the **comparison**, and a negative result is publishable
evidence: "ordinary video predicts total loading but not per-sensor distribution"
would be a real finding about what vision can observe.

## Next session (Milestone 3)

1. Extract gait-phase labels from GAITRite `HeelOn`/`HeelOff`/`ToeOn`/`ToeOff`
   — free, reliable weak labels — after confirming which `Foot` value is which.
2. Baseline 0 on the committed participant split (14 train / 4 val / 4 test).
3. Baseline 1 from motion features.
4. Metrics: per-sensor MAE/RMSE, total-load error, COP error, contact accuracy,
   reported **per participant, per condition, per foot** — never a single number.
5. Compare scalar regression against ordered-bin classification (an experiment,
   not an assumption).

Not before the above: temporal models, domain adaptation, dense decoders.

## Standing constraints

- Never split frames randomly; splits are by participant and the `Split` class
  raises on overlap.
- Never call raw counts kPa.
- Never call interpolated pixels measurements.
- Never assume `left[i]` and `right[i]` are the same anatomical site — they never are.
- Never assume 30 fps; read it per clip.
- Never claim a lattice reduces plantar pressure without physical testing.
- Record failures. A method that does not work is a result.
