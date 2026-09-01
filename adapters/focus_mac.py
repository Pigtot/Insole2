"""Apple Silicon adapter for FOCUS (github.com/OllieBoyne/FOCUS).

Upstream is **not modified**. Everything needed to run FOCUS on this machine is
here, so the checkout in ``external/FOCUS`` stays a clean upstream tree.

What this adapter knows, all of it measured on an M1 Max / macOS 26.5 /
torch 2.13 / pytorch3d 0.7.8 (see docs/focus_mac_port.md):

1. **The network belongs on MPS.** The TOC predictor (72.4M params,
   EfficientNet-B5 backbone) runs at 90 ms/image on MPS versus 1600 ms on CPU --
   17.8x -- and agrees with CPU to ~5e-6 relative error on every output head.

2. **PyTorch3D geometry belongs on CPU, and this is not a preference.**
   ``knn_points`` and ``sample_points_from_meshes`` **segfault** (SIGSEGV) when
   handed MPS tensors. They do not raise; the process dies. Nor does
   ``PYTORCH_ENABLE_MPS_FALLBACK=1`` help: that only covers ATen operators, and
   these are third-party compiled kernels in pytorch3d's own ``_C`` extension,
   which the ATen fallback never sees.

3. **SSL certificates must be pointed at certifi.** FOCUS's model builder pulls
   pretrained EfficientNet weights from GitHub at import time, and the
   python.org framework build ships no usable CA bundle, so the import dies with
   CERTIFICATE_VERIFY_FAILED.

Call :func:`configure` **before importing torch or FOCUS**.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FOCUS_ROOT = REPO_ROOT / "external" / "FOCUS"
FOCUS_ENV_PYTHON = REPO_ROOT / "envs" / "focus" / "bin" / "python"

#: PyTorch3D geometry must run here. See point 2 above -- MPS segfaults.
GEOMETRY_DEVICE = "cpu"

#: Ops verified to crash the process on MPS rather than raise.
MPS_UNSAFE_OPS = ("pytorch3d.ops.knn_points", "pytorch3d.ops.sample_points_from_meshes")

_configured = False


def configure(verbose: bool = False) -> dict:
    """Set the environment FOCUS needs on macOS. Safe to call more than once."""
    global _configured
    info: dict[str, object] = {}

    if "torch" in sys.modules and not _configured:
        info["warning"] = (
            "torch was imported before configure(); PYTORCH_ENABLE_MPS_FALLBACK "
            "is read at MPS backend init and will not take effect now."
        )

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    try:  # point SSL at certifi so the EfficientNet download works
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        info["ssl_cert_file"] = certifi.where()
    except ImportError:
        info["ssl_cert_file"] = None

    if FOCUS_ROOT.exists() and str(FOCUS_ROOT) not in sys.path:
        sys.path.insert(0, str(FOCUS_ROOT))
    info["focus_root"] = str(FOCUS_ROOT)

    _configured = True
    if verbose:
        for k, v in info.items():
            print(f"  {k}: {v}")
    return info


def network_device(prefer: str | None = None) -> str:
    """Device for the TOC predictor: MPS when available, else CPU."""
    import torch

    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def assert_geometry_on_cpu(*tensors) -> None:
    """Guard before any pytorch3d geometry call.

    Raises a readable error instead of letting the process segfault, which is
    what actually happens if an MPS tensor reaches pytorch3d's ``_C`` kernels.
    """
    for t in tensors:
        dev = getattr(t, "device", None)
        if dev is not None and dev.type == "mps":
            raise RuntimeError(
                "PyTorch3D geometry received an MPS tensor. On Apple Silicon this "
                "segfaults the process rather than raising (knn_points, "
                "sample_points_from_meshes). Move to CPU first: .to('cpu')."
            )


def preflight() -> dict:
    """Check every part of the port and report, without running a full pipeline."""
    configure()
    import torch

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "network_device": network_device(),
        "geometry_device": GEOMETRY_DEVICE,
    }
    try:
        import pytorch3d
        from pytorch3d import _C  # noqa: F401

        report["pytorch3d"] = pytorch3d.__version__
        report["pytorch3d_compiled_ext"] = True
    except Exception as exc:
        report["pytorch3d"] = f"MISSING ({type(exc).__name__})"
        report["pytorch3d_compiled_ext"] = False

    import shutil

    report["colmap"] = shutil.which("colmap") is not None
    report["ffmpeg"] = shutil.which("ffmpeg") is not None
    report["toc_model"] = (FOCUS_ROOT / "data/toc_model/densedepth_toc_predictor.pth").exists()
    report["find_template"] = (FOCUS_ROOT / "data/find/template.obj").exists()

    modules = ["FOCUS.toc_prediction.predict", "FOCUS.fusion.fuse",
               "FOCUS.calibration.run_colmap", "FOCUS.data.dataset",
               "FOCUS.utils.camera", "FOCUS.optim.optim"]
    ok = {}
    for m in modules:
        try:
            __import__(m)
            ok[m] = True
        except Exception as exc:
            ok[m] = f"{type(exc).__name__}: {str(exc)[:60]}"
    report["modules"] = ok
    return report
