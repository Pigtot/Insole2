# FOCUS on Apple Silicon — port record

**Verdict: FOCUS runs on this Mac.** Every module imports, the neural stage runs
on MPS **20× faster than CPU** and agrees with CPU to 3.2e-6, COLMAP works
without CUDA, and PyTorch3D builds from source. One upstream patch was needed,
for NumPy 2 rather than for macOS.

Machine: Apple M1 Max, 64 GB, macOS 26.5.2 · torch 2.13.0 · pytorch3d 0.7.8 ·
Python 3.12.1 · FOCUS commit `b98f569` · verified 2026-08-31.

```bash
envs/focus/bin/python scripts/run_focus_demo.py --check
```

```bash
envs/focus/bin/python scripts/run_focus_demo.py --benchmark
```

---

## 1. The headline: put the network on MPS, the geometry on CPU

| Stage | Device | Measured |
| --- | --- | --- |
| TOC predictor (72.4M params, EfficientNet-B5) | **MPS** | **90 ms/image** vs 1821 ms CPU — **20.2×** |
| PyTorch3D geometry | **CPU (mandatory)** | see §2 — MPS *segfaults* |
| COLMAP | CPU (external binary) | 4.1.1, built without CUDA |
| Video decode | CPU (ffmpeg) | — |

MPS output matches CPU on every head:

| head | relative difference |
| --- | --- |
| `TOC` | 3.07e-06 |
| `norm_xyz` | 2.91e-06 |
| `TOC_unc_log_var` | 3.23e-06 |
| `hm` | 7.70e-14 |
| `footedness` | 1.85e-07 |

Batch size barely matters — 91.7 ms/image at batch 1 falls only to 77.2 ms at
batch 16 (~16 %). **Use batch 4**: near-peak throughput at a quarter of the
memory. The upstream default of 10 is also fine.

## 2. Two PyTorch3D ops *segfault* on MPS

Not "raise", not "fall back" — the process dies with SIGSEGV (exit 139):

| op | CPU | MPS |
| --- | --- | --- |
| `Transform3d` compose | OK | OK |
| `PerspectiveCameras` | OK | OK |
| `Meshes.verts_normals_packed` | OK | OK |
| `rasterize_meshes` | OK | OK |
| **`knn_points`** | OK | **SIGSEGV** |
| **`sample_points_from_meshes`** | OK | **SIGSEGV** |

**`PYTORCH_ENABLE_MPS_FALLBACK=1` does not save you here.** That mechanism
covers ATen operators; these are third-party kernels inside pytorch3d's own
compiled `_C` extension, which the ATen fallback never sees. Setting the
variable is still worthwhile for everything else, but it is not protection.

Where this matters: both ops are used **only inside the FOUND submodule** —
chamfer distance, FIND training losses, 3D evaluation — and **not** in FOCUS
core (`toc_prediction`, `calibration`, `fusion`). So the FOCUS-SfM path never
touches them. FOCUS-O, which optimises the FIND model, does.

`adapters/focus_mac.assert_geometry_on_cpu()` raises a readable error instead of
letting the process die.

## 3. Installing PyTorch3D

There are no macOS arm64 wheels; build from source. It takes about 9 minutes and
needs no special flags beyond installing torch first.

```bash
git clone https://github.com/facebookresearch/pytorch3d.git external/pytorch3d-src
cd external/pytorch3d-src && git checkout 81d82980bc82fd605f27cca87f89ba08af94db3d
MAX_JOBS=8 envs/focus/bin/python -m pip install --no-build-isolation .
```

The build is CPU-only (no CUDA kernels), which is exactly what we need. FOCUS's
own `setup.py` already anticipates Mac users: it sets `MAX_JOBS=1` and offers
`SKIP_PYTORCH3D=1`. With 64 GB, `MAX_JOBS=8` is fine and much faster.

Install FOCUS itself without letting it drag in its pinned stack:

```bash
SKIP_PYTORCH3D=1 envs/focus/bin/python -m pip install --no-deps --editable external/FOCUS
```

## 4. Problems hit, and their fixes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `CERTIFICATE_VERIFY_FAILED` importing `FOCUS.optim` | The model builder downloads EfficientNet-B5 weights at import; the python.org build ships no CA bundle | `SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")` — done by `adapters/focus_mac.configure()` |
| `'numpy.ndarray' object has no attribute 'tostring'` | Removed in NumPy 2.0 | [patch](../patches/focus_numpy2_tostring.patch), see [upstream_modifications.md](upstream_modifications.md) |
| `ModuleNotFoundError: pyximport` | The matching module compiles Cython on the fly | `pip install Cython` |
| `ModuleNotFoundError: pymeshlab` / `ffmpeg` | Not in the minimal set | install both |
| Model rejects input: "size of tensor a (4) must match b (3)" | `predict_toc`'s docstring says "RGBA"; the loader actually produces **RGB** via `COLOR_BGR2RGB`. The docstring is stale | pass 3-channel RGB |
| `colmap` installed but `dyld: libx265.216.dylib not loaded` | Homebrew ffmpeg linked against an older x265 soname | `brew reinstall ffmpeg` |

### A self-inflicted one, recorded honestly

Running `brew reinstall x265` to fix the COLMAP dyld error **upgraded x265 to
`.217` and broke a previously working ffmpeg**, which the gait pipeline depends
on. `brew reinstall ffmpeg` fixed both. The lesson: reinstall the *dependent*
(ffmpeg), not the dependency (x265). Verified afterwards that `probe_video()`
and all 117 tests still pass.

## 5. Minimal dependency set

Upstream `requirements.txt` pins ~90 packages including `blendersynth` (Blender),
`dash`, `Flask`, `OpenEXR` — visualisation and synthetic-data tooling not needed
for inference. Installed instead:

```
torch torchvision pytorch3d numpy scipy matplotlib opencv-python scikit-learn
trimesh pyyaml timm efficientnet-pytorch pretrainedmodels geffnet munch tqdm
shapely pyquaternion Cython pymeshlab open3d ffmpeg-python certifi gdown
```

`pip` reports version conflicts against upstream's pins (`trimesh==4.5.2`,
`typing_extensions==4.12.2`, …). They are **not fatal** — every module imports
and the model runs — but they are unverified deviations and are recorded here
rather than silently accepted.

Note `open3d` **is** required despite the pymeshlab conflict: FOCUS calls it in a
**separate subprocess** (`utils/o3d_outlier_removal.py`) precisely so the two
libraries never share a process. Upstream comment: *"Importing this and open3D
together cause some errors on M1 mac with poisson reconstruction."* The authors
hit this before us.

## 6. What is verified, and what is not

**Verified**
- All FOCUS modules import, including `optim`.
- TOC prediction runs end-to-end on rendered foot images, producing masks,
  normals, dense correspondences, uncertainties and footedness.
- MPS is 20× faster than CPU and numerically equivalent.
- COLMAP calibrates 8 of 12 views from synthetic renders.

**Not verified**
- **Reconstruction accuracy.** No end-to-end mesh was produced, and no metric
  comparison against ground truth exists.

The blocker is **test data, not the port**. Validation images came from
rendering the shipped FIND template, and a flat-shaded synthetic foot is out of
distribution for a predictor trained on SynFoot plus real photographs: 4 of 16
views fell below the 2 % mask-coverage threshold and COLMAP could not calibrate
4 of the remaining 12. Raising `--num_colmap_matches` to 12,000 did not help.

**Real evaluation needs the Foot3D `Multiview` dataset** (474 calibrated images,
14 scans), which is released through a Google form:
<https://forms.gle/7eZh67UXMZYcM11M7>. Someone must request it by hand — select
`Multiview`. Until then, no claim about FOCUS's accuracy on this machine is
supported.

## 7. A useful discovery for Visole

`external/FOCUS/data/find/` ships the FIND foot template in-repo, no download:

- `template.obj` — 44,395 vertices, 88,786 faces, watertight, 264 × 97 × 150 mm
- **`templ_sole_faces.npy` — 19,340 faces (21.8 %) labelling the plantar surface**,
  covering 10,197 vertices
- `keypoints.csv` — big toe, 2nd–little toe, heel, lower heel, medial/lateral
  extrema, arch1–3

This is exactly the anchor Milestones 6–7 need: a canonical foot with a labelled
sole and named landmarks. It is also the natural target for the HOPE-style
per-vertex pressure representation — pressure predicted on sole vertices rather
than in camera coordinates.
