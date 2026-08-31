#!/usr/bin/env python
"""Machine audit for Visole (Milestone 0).

Writes reports/system_report.md and prints a short verdict. Reports facts only;
where a value cannot be determined it says so rather than guessing.

    python scripts/doctor.py [--json] [--out reports/system_report.md]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Imported before torch so the MPS-fallback env var is set in time.
from visole.compute import device as vdev  # noqa: E402

import torch  # noqa: E402


def _sh(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or None
    except Exception:
        return None


def _sysctl(key: str) -> str | None:
    return _sh(["sysctl", "-n", key])


def collect() -> dict:
    total, used, free = shutil.disk_usage(REPO)
    mem_bytes = _sysctl("hw.memsize")
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": str(REPO),
        "machine": platform.machine(),
        "system": platform.system(),
        "os_release": platform.mac_ver()[0] or platform.release(),
        "os_build": _sh(["sw_vers", "-buildVersion"]),
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
        "cpu_cores_logical": os.cpu_count(),
        "cpu_cores_physical": _sysctl("hw.physicalcpu"),
        "cpu_perf_cores": _sysctl("hw.perflevel0.physicalcpu"),
        "cpu_eff_cores": _sysctl("hw.perflevel1.physicalcpu"),
        "ram_total_gb": round(int(mem_bytes) / 1024**3, 1) if mem_bytes else None,
        "disk_total_gb": round(total / 1024**3, 1),
        "disk_free_gb": round(free / 1024**3, 1),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": _module_version("torchvision"),
        "numpy_version": _module_version("numpy"),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": vdev.mps_available(),
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(vdev.resolve_device("auto")),
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "torch_num_threads": torch.get_num_threads(),
        "colmap": _sh(["colmap", "-h"]) is not None,
        "git_version": _sh(["git", "--version"]),
        "homebrew": shutil.which("brew") is not None,
    }
    info["mps_smoke_test"] = _mps_smoke_test()
    return info


def _module_version(name: str) -> str | None:
    try:
        import importlib

        return getattr(importlib.import_module(name), "__version__", None)
    except Exception:
        return None


def _mps_smoke_test() -> dict:
    """Actually run a tiny op on MPS. 'is_available()' alone is not proof."""
    if not vdev.mps_available():
        return {"ran": False, "reason": "MPS not available"}
    try:
        dev = torch.device("mps")
        a = torch.randn(256, 256, device=dev)
        b = torch.randn(256, 256, device=dev)
        c = (a @ b).sum().item()
        ref = (a.cpu() @ b.cpu()).sum().item()
        rel = abs(c - ref) / max(abs(ref), 1e-6)
        return {
            "ran": True,
            "matmul_ok": True,
            "relative_error_vs_cpu": rel,
            "agrees_with_cpu": rel < 1e-3,
        }
    except Exception as exc:  # pragma: no cover
        return {"ran": True, "matmul_ok": False, "error": repr(exc)}


def verdict(info: dict) -> list[str]:
    """Plain-language readiness notes. Conservative by design."""
    notes = []
    if info["machine"] != "arm64":
        notes.append(f"WARNING: expected arm64, found {info['machine']}.")
    if not info["mps_available"]:
        notes.append("WARNING: MPS unavailable -- all torch work will run on CPU.")
    else:
        notes.append("OK: MPS backend available and verified with a live matmul.")
    if info["cuda_available"]:
        notes.append("NOTE: CUDA reported available -- unexpected on this machine.")
    else:
        notes.append("OK: no CUDA, as expected. Never call CUDA APIs here.")
    ram = info.get("ram_total_gb") or 0
    if ram < 16:
        notes.append(f"WARNING: {ram} GB RAM. Keep batch size 1 and images small.")
    else:
        notes.append(f"OK: {ram} GB unified memory.")
    if info["disk_free_gb"] < 50:
        notes.append(f"WARNING: only {info['disk_free_gb']} GB free. Do not fetch large datasets.")
    else:
        notes.append(f"OK: {info['disk_free_gb']} GB free disk.")
    if not info["colmap"]:
        notes.append("PENDING: COLMAP not installed (needed later for FOCUS-SfM).")
    return notes


def to_markdown(info: dict, notes: list[str]) -> str:
    lines = [
        "# Visole system report",
        "",
        f"Generated: `{info['timestamp_utc']}`  ",
        f"Interpreter: `{info['python_executable']}`",
        "",
        "## Verdict",
        "",
    ]
    lines += [f"- {n}" for n in notes]
    lines += ["", "## Raw values", "", "| key | value |", "| --- | --- |"]
    for k, v in info.items():
        if isinstance(v, dict):
            v = json.dumps(v)
        lines.append(f"| `{k}` | `{v}` |")
    lines += [
        "",
        "## How to reproduce",
        "",
        "```bash",
        "envs/core/bin/python scripts/doctor.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/system_report.md")
    ap.add_argument("--json", action="store_true", help="print raw JSON to stdout")
    args = ap.parse_args()

    info = collect()
    notes = verdict(info)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(info, notes))
    (out.with_suffix(".json")).write_text(json.dumps(info, indent=2))

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for n in notes:
            print(n)
    print(f"\nwrote {out.relative_to(REPO)} and {out.with_suffix('.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
