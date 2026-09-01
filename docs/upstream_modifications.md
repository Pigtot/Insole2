# Upstream modifications

Every change made to third-party code, why it was needed, and how to reapply it.
Upstream checkouts otherwise stay pristine so they can be updated cleanly.

## FOCUS — `array.tostring()` removed in NumPy 2

| | |
| --- | --- |
| Repository | [OllieBoyne/FOCUS](https://github.com/OllieBoyne/FOCUS) |
| Commit | `b98f5699c9148434d3ee16c3dd4d436d2560b822` |
| File | `FOCUS/calibration/colmap_database.py`, line 145 |
| Patch | [`patches/focus_numpy2_tostring.patch`](../patches/focus_numpy2_tostring.patch) |

**Why.** `ndarray.tostring()` was deprecated in NumPy 1.19 and **removed in
2.0**. FOCUS's vendored COLMAP database helper still calls it, so writing the
camera table raises `AttributeError` and the whole SfM path dies before COLMAP
is invoked.

**Change.** One line, `array.tostring()` → `array.tobytes()`. The two are exact
synonyms for this use — `tobytes()` has existed since NumPy 1.9 and `tostring()`
was only ever an alias for it. Behaviour is unchanged on old and new NumPy alike.

**Not Mac-specific.** This breaks on any machine with NumPy ≥ 2.

Reapply after a fresh clone:

```bash
git -C external/FOCUS apply ../../patches/focus_numpy2_tostring.patch
```

## FOCUS — OpenMP guard aborts the process on Apple Silicon

| | |
| --- | --- |
| File | `FOCUS/utils/remeshing.py`, `_setup_meshlab()` |
| Patch | [`patches/focus_mac_openmp_meshlab.patch`](../patches/focus_mac_openmp_meshlab.patch) |

**Why.** The guard is

```python
if os.environ.get("KMP_DUPLICATE_LIB_OK", "True"):   # always true
    os.environ["KMP_DUPLICATE_LIB_OK"] = "False"
```

`os.environ.get` returns a string and every non-empty string is truthy --
including `"False"` -- so the branch fires unconditionally and *disables*
duplicate-OpenMP tolerance. On this machine torch's `libomp` is already
initialised in-process when pymeshlab loads its own, so libomp calls `abort()`
at the first Poisson reconstruction: no traceback, process dead.

**Change.** Set `KMP_DUPLICATE_LIB_OK=TRUE` instead. Safe here because the same
function pins `OMP_NUM_THREADS=1` and the Poisson call passes `threads=1`, so
the two runtimes never schedule work concurrently. Verified by reconstructing
all 14 Foot3D scans with sane, symmetric chamfer distances.

## Everything else is an adapter, not a modification

The remaining Mac accommodations live outside the upstream tree, in
[`adapters/focus_mac.py`](../adapters/focus_mac.py):

- setting `SSL_CERT_FILE` to certifi's bundle before import,
- setting `PYTORCH_ENABLE_MPS_FALLBACK` before torch loads,
- the device policy (network on MPS, PyTorch3D geometry on CPU),
- `assert_geometry_on_cpu()`, which raises rather than letting an MPS tensor
  segfault pytorch3d's compiled kernels.

None of these touch upstream files. See [focus_mac_port.md](focus_mac_port.md).
