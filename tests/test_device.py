"""Tests for the device abstraction.

These must pass on any backend -- that is the point of the abstraction.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visole.compute import device as vdev  # noqa: E402
import torch  # noqa: E402


def test_auto_resolves_to_available_backend():
    dev = vdev.resolve_device("auto")
    assert dev.type in {"mps", "cuda", "cpu"}
    if dev.type == "mps":
        assert vdev.mps_available()
    if dev.type == "cuda":
        assert torch.cuda.is_available()


def test_cpu_always_resolvable():
    assert vdev.resolve_device("cpu").type == "cpu"


def test_unavailable_backend_raises_rather_than_downgrading():
    """A silent downgrade to CPU would corrupt benchmark timings."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
            vdev.resolve_device("cuda")
    if not vdev.mps_available():
        with pytest.raises(RuntimeError, match="MPS requested but unavailable"):
            vdev.resolve_device("mps")


def test_bad_spec_raises():
    with pytest.raises(ValueError):
        vdev.resolve_device("tpu")


def test_passthrough_of_device_object():
    d = torch.device("cpu")
    assert vdev.resolve_device(d) is d


def test_to_device_handles_nested_containers():
    dev = vdev.resolve_device("cpu")
    payload = {"a": torch.zeros(2), "b": [torch.ones(3), (torch.zeros(1),)], "c": "not a tensor"}
    moved = vdev.to_device(payload, dev)
    assert moved["a"].device.type == "cpu"
    assert moved["b"][0].device.type == "cpu"
    assert moved["b"][1][0].device.type == "cpu"
    assert moved["c"] == "not a tensor"
    assert isinstance(moved["b"][1], tuple)


def test_describe_device_reports_required_keys():
    info = vdev.describe_device()
    for key in ("device", "torch_version", "mps_available", "cuda_available",
                "mps_fallback_effective"):
        assert key in info


def test_safe_load_maps_to_cpu_by_default(tmp_path):
    ckpt = tmp_path / "m.pt"
    torch.save({"w": torch.randn(4, 4)}, ckpt)
    loaded = vdev.safe_load(ckpt, weights_only=True)
    assert loaded["w"].device.type == "cpu"


def test_empty_cache_is_safe_on_every_backend():
    vdev.empty_cache("cpu")
    vdev.empty_cache("auto")


def test_autocast_is_a_noop_on_non_cuda():
    """We do not enable MPS autocast until numerics are verified per-experiment."""
    dev = vdev.resolve_device("cpu")
    with vdev.autocast(dev):
        x = torch.randn(4, 4)
        assert (x @ x).dtype == torch.float32


def test_mps_fallback_recorder_passes_through_unrelated_warnings():
    import warnings

    with pytest.warns(UserWarning, match="unrelated"):
        with vdev.MPSFallbackRecorder() as rec:
            warnings.warn("an unrelated warning", UserWarning)
    assert not rec.had_fallback


@pytest.mark.skipif(not vdev.mps_available(), reason="MPS not available")
def test_mps_matches_cpu_on_a_real_op():
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    ref = a @ b
    got = (a.to("mps") @ b.to("mps")).cpu()
    assert torch.allclose(ref, got, atol=1e-4)
