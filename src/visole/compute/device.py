"""Central device abstraction for Visole.

No other module in Visole may assume CUDA. Everything routes through here.

Ordering note (important, and easy to get wrong)
------------------------------------------------
``PYTORCH_ENABLE_MPS_FALLBACK`` is read by PyTorch when the MPS backend is
first initialised, which happens during ``import torch``. Setting it after
torch has been imported has no effect. This module therefore sets a default
value for it *before* it imports torch, and records whether torch had already
been imported by someone else first, so :func:`describe_device` can tell the
truth about whether the fallback is actually active.

Practical rule: import ``visole.compute.device`` before ``torch``.
"""

from __future__ import annotations

import os
import platform
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable

# --- must happen before `import torch` -------------------------------------
_TORCH_WAS_ALREADY_IMPORTED = "torch" in sys.modules
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# ---------------------------------------------------------------------------

import torch  # noqa: E402

__all__ = [
    "get_device",
    "resolve_device",
    "describe_device",
    "to_device",
    "safe_load",
    "autocast",
    "MPSFallbackRecorder",
    "empty_cache",
]

_VALID_SPECS = ("auto", "mps", "cuda", "cpu")


def resolve_device(spec: str | torch.device | None = "auto") -> torch.device:
    """Resolve a device spec to a concrete, *available* ``torch.device``.

    ``auto`` prefers MPS, then CUDA, then CPU. An explicit request for a
    backend that is not available raises rather than silently downgrading --
    silently running on CPU when the caller asked for a GPU is how benchmark
    numbers become wrong.
    """
    if isinstance(spec, torch.device):
        return spec
    if spec is None:
        spec = "auto"
    spec = str(spec).lower().strip()

    if spec == "auto":
        if mps_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if spec.startswith("mps"):
        if not mps_available():
            raise RuntimeError(
                "MPS requested but unavailable. "
                f"built={torch.backends.mps.is_built()} "
                f"available={torch.backends.mps.is_available()}"
            )
        return torch.device(spec)

    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. On Apple Silicon use 'mps' or 'cpu'."
            )
        return torch.device(spec)

    if spec.startswith("cpu"):
        return torch.device("cpu")

    raise ValueError(f"Unknown device spec {spec!r}; expected one of {_VALID_SPECS}")


def mps_available() -> bool:
    """True only if the MPS backend is both built and usable."""
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_built() and backend.is_available())
    except Exception:  # pragma: no cover - defensive on odd builds
        return False


def get_device(spec: str | torch.device | None = "auto") -> torch.device:
    """Convenience alias for :func:`resolve_device`."""
    return resolve_device(spec)


def describe_device(device: torch.device | None = None) -> dict[str, Any]:
    """Structured description of the compute environment, for experiment logs."""
    device = device or resolve_device("auto")
    fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    return {
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "mps_available": mps_available(),
        "cuda_available": torch.cuda.is_available(),
        "mps_fallback_env": fallback_env,
        # If torch was imported before us, our setdefault may have been too late.
        "mps_fallback_effective": (
            bool(fallback_env == "1") and not _TORCH_WAS_ALREADY_IMPORTED
        ),
        "torch_imported_before_visole": _TORCH_WAS_ALREADY_IMPORTED,
        "num_threads": torch.get_num_threads(),
    }


def to_device(obj: Any, device: torch.device | str) -> Any:
    """Recursively move tensors / modules / containers to ``device``."""
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    if isinstance(obj, (torch.Tensor, torch.nn.Module)):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, device) for v in obj]
        return type(obj)(moved) if not isinstance(obj, tuple) else tuple(moved)
    return obj


def safe_load(path: str | os.PathLike, device: str | torch.device = "cpu", **kwargs: Any) -> Any:
    """Load a checkpoint without assuming it was saved on the current backend.

    Upstream research checkpoints are routinely saved from CUDA machines. Always
    map to CPU first (default) and move the model afterwards -- that is the only
    reliably portable order.
    """
    map_location = resolve_device(device) if not isinstance(device, torch.device) else device
    kwargs.setdefault("map_location", map_location)
    return torch.load(path, **kwargs)


class MPSFallbackRecorder:
    """Record PyTorch's 'operator not supported on MPS' fallback warnings.

    PyTorch emits a ``UserWarning`` naming each aten operator that silently
    falls back to CPU. We capture those warnings rather than claiming to hook
    the dispatcher -- this is the honest, supported way to see fallbacks from
    Python. Each operator is reported once per process by PyTorch, so this
    records first occurrences, not a per-call count.
    """

    _PATTERN = "not currently supported on the MPS backend"

    def __init__(self) -> None:
        self.operators: list[str] = []
        self._ctx: Any = None

    def __enter__(self) -> "MPSFallbackRecorder":
        self._ctx = warnings.catch_warnings(record=True)
        self._caught = self._ctx.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc: Any) -> None:
        # Snapshot before leaving the context: `self._caught` is the live list
        # that catch_warnings appends to, so re-emitting while iterating it
        # would append to the list being iterated and never terminate.
        caught = list(self._caught)
        self._ctx.__exit__(*exc)
        for w in caught:
            msg = str(w.message)
            if self._PATTERN in msg:
                self.operators.append(msg.strip())
            else:  # re-emit anything unrelated so we do not swallow real warnings
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    @property
    def had_fallback(self) -> bool:
        return bool(self.operators)


def autocast(device: torch.device | str, enabled: bool = True):
    """Device-aware autocast context.

    MPS autocast support is incomplete across PyTorch versions, so we default
    to a no-op on MPS rather than producing subtly wrong numerics. Enable it
    deliberately, per experiment, once verified.
    """
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "mps" and enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cpu", enabled=False)


def empty_cache(device: torch.device | str = "auto") -> None:
    """Release cached GPU memory on whichever backend is in use."""
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


@dataclass
class DeviceReport:
    """Snapshot of device state attached to experiment logs."""

    info: dict[str, Any] = field(default_factory=describe_device)

    def as_markdown(self) -> str:
        return "\n".join(f"- **{k}**: `{v}`" for k, v in self.info.items())
