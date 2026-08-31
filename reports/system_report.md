# Visole system report

Generated: `2026-08-31T06:20:39+00:00`  
Interpreter: `/Users/lufan/Insole2/envs/core/bin/python`

## Verdict

- OK: MPS backend available and verified with a live matmul.
- OK: no CUDA, as expected. Never call CUDA APIs here.
- OK: 64.0 GB unified memory.
- OK: 1129.5 GB free disk.
- PENDING: COLMAP not installed (needed later for FOCUS-SfM).

## Raw values

| key | value |
| --- | --- |
| `timestamp_utc` | `2026-08-31T06:20:39+00:00` |
| `repo` | `/Users/lufan/Insole2` |
| `machine` | `arm64` |
| `system` | `Darwin` |
| `os_release` | `26.5.2` |
| `os_build` | `25F84` |
| `cpu_brand` | `Apple M1 Max` |
| `cpu_cores_logical` | `10` |
| `cpu_cores_physical` | `10` |
| `cpu_perf_cores` | `8` |
| `cpu_eff_cores` | `2` |
| `ram_total_gb` | `64.0` |
| `disk_total_gb` | `1858.2` |
| `disk_free_gb` | `1129.5` |
| `python_executable` | `/Users/lufan/Insole2/envs/core/bin/python` |
| `python_version` | `3.12.1` |
| `torch_version` | `2.13.0` |
| `torchvision_version` | `0.28.0` |
| `numpy_version` | `2.5.2` |
| `mps_built` | `True` |
| `mps_available` | `True` |
| `cuda_available` | `False` |
| `selected_device` | `mps` |
| `mps_fallback_env` | `1` |
| `torch_num_threads` | `8` |
| `colmap` | `False` |
| `git_version` | `git version 2.51.0` |
| `homebrew` | `True` |
| `mps_smoke_test` | `{"ran": true, "matmul_ok": true, "relative_error_vs_cpu": 1.1393486002247934e-07, "agrees_with_cpu": true}` |

## How to reproduce

```bash
envs/core/bin/python scripts/doctor.py
```
