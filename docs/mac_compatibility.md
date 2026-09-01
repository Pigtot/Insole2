# Mac / Apple Silicon compatibility

Machine: Apple M1 Max (8P+2E), 64 GB unified memory, macOS 26.5.2, arm64.
Verified 2026-08-30 — see [reports/system_report.md](../reports/system_report.md).

Status values: **works** (run here), *untested* (not yet attempted),
**blocked** (known not to work on this machine).

## Core stack

| Component | Status | Note |
| --- | --- | --- |
| PyTorch 2.13.0 | **works** | MPS built *and* available; verified by a live 256×256 matmul agreeing with CPU to 1.1e-7 relative error |
| `torch.backends.mps` | **works** | `resolve_device("auto")` → `mps` |
| CUDA | **blocked** by design | `cuda_available=False`. Never install; `resolve_device("cuda")` raises rather than downgrading silently |
| NumPy 2.5.2 | **works** | Note: `ndarray.ptp()` was removed — use `np.ptp(x)` |
| pandas 3.0.5, SciPy 1.18.1 | **works** | |
| OpenCV 5.0.0.93 | **works** | Video decode and frame streaming |
| scikit-image 0.26.0 | **works** | `marching_cubes` emits a NumPy 2.5 deprecation warning from upstream; filtered in `pyproject.toml` |
| trimesh 5.0.0 | **works** | Watertight STL export verified |
| matplotlib 3.11.1 | **works** | `Agg` backend in scripts |
| ffmpeg / ffprobe | **works** | Note: `-vsync` is gone in this build — use `-fps_mode` |
| Homebrew, git 2.51 | **works** | |
| COLMAP | **works** | 4.1.1 via Homebrew, built **without CUDA**. Needed `brew reinstall ffmpeg` to fix a dyld x265 soname mismatch |
| CadQuery / LatticeQuery | *not installed* | Milestone 8. Visole's own lattice engine depends on neither |
| PyTorch3D (for FOCUS) | **works, CPU-only** | 0.7.8 built from source in ~9 min, `MAX_JOBS=8`. **`knn_points` and `sample_points_from_meshes` SEGFAULT on MPS** — keep all pytorch3d geometry on CPU. See [focus_mac_port.md](focus_mac_port.md) |
| FOCUS (TOC predictor) | **works on MPS** | 90 ms/image vs 1821 ms CPU (**20.2×**), matches CPU to 3.2e-6. Upstream already selects MPS itself |
| torchvision Keypoint R-CNN | **works on MPS** | 2D body pose. **0.104 s/frame on MPS vs 1.14 s/frame on CPU** (min_size=480 vs default) -- roughly 11x, the clearest MPS win so far |
| certifi | **required** | macOS framework Python ships no CA bundle; see below |
| MLX | *not installed* | Only for new models written from scratch, never for porting upstream research |
| nTop | **blocked** | No Apple Silicon build. Export contract kept so a Windows box could consume the field |

## The MPS fallback ordering trap

`PYTORCH_ENABLE_MPS_FALLBACK` is read when PyTorch initialises the MPS backend,
during `import torch`. **Setting it afterwards does nothing.**

`visole.compute.device` sets it before its own `import torch`, and records
whether torch had already been imported by someone else first. `describe_device()`
exposes `mps_fallback_effective`, so a too-late import is visible instead of
silently ineffective.

Rule: **import `visole.compute.device` before `torch`.**

Fallbacks themselves are captured by `MPSFallbackRecorder`, which records
PyTorch's per-operator warnings. PyTorch emits each once per process, so this
records first occurrences, not per-call counts — the honest limit of the
supported Python-level approach.

## Autocast

`compute.device.autocast()` is a **no-op on MPS**. MPS autocast coverage varies
across PyTorch versions, and silently wrong numerics cost more than the speed is
worth. Enable per experiment, only after checking against CPU.

## Memory discipline

64 GB is generous, but the discipline holds because it catches bugs, not just
OOMs: batch size 1 first, small images, stream video frames rather than loading
clips, `num_workers` 0–2 on macOS, `empty_cache()` between experiments, and
`--max-samples` / `--max-frames` on every heavy script. On OOM reduce batch,
resolution or sequence length — never disable PyTorch's MPS memory limits.

## Gotchas hit while building this

| Symptom | Cause | Fix |
| --- | --- | --- |
| Process dies with exit 139, no traceback | `knn_points` / `sample_points_from_meshes` given an MPS tensor. **`PYTORCH_ENABLE_MPS_FALLBACK` does not cover third-party compiled kernels**, only ATen ops | Keep pytorch3d geometry on CPU |
| `CERTIFICATE_VERIFY_FAILED` on import | python.org build has no CA bundle; a model downloads weights at import | Set `SSL_CERT_FILE` to `certifi.where()` |
| `'numpy.ndarray' has no attribute 'tostring'` | Removed in NumPy 2.0 | `.tobytes()` |
| `dyld: libx265.216.dylib not loaded` | Homebrew ffmpeg linked to an older x265 soname | `brew reinstall ffmpeg` — reinstall the *dependent*, not the dependency |
| `AttributeError: 'numpy.ndarray' object has no attribute 'ptp'` | NumPy 2.x removed the method | `np.ptp(x)` |
| `Unrecognized option 'vsync'` | Removed in this ffmpeg | `-fps_mode passthrough` |
| pytest hanging forever | Re-emitting warnings while iterating the live `catch_warnings` list — it appends to the list being iterated | Snapshot the list and exit the context first |
| Stray marks outside the insole outline | `ElementTree.iter()` also returns `<path>` elements inside `<defs>`/`<symbol>`/`<marker>` | Walk the tree and skip non-rendered subtrees |
| `python3` resolves to a PlatformIO venv | It is first on `PATH` | Use `envs/core/bin/python` explicitly; the venv is built from the 3.12 framework build |
| `CERTIFICATE_VERIFY_FAILED` downloading pretrained weights | The macOS **framework** Python build has no CA bundle, so `urllib` (and therefore `torch.hub`) cannot verify TLS. `curl` works because it uses the system keychain, which masks the problem until the first model download | `pip install certifi` and set `SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")`. `scripts/extract_pose.py` sets it automatically |
| `RuntimeWarning: Mean of empty slice` | `np.nanmean` over all-NaN input — every per-channel correlation is undefined for a constant predictor | Return NaN explicitly; NaN is the right answer, 0.0 would be a false claim |
