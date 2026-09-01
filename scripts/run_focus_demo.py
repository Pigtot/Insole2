#!/usr/bin/env python
"""FOCUS on Apple Silicon: preflight check, benchmark, and demo runner.

    envs/focus/bin/python scripts/run_focus_demo.py --check
    envs/focus/bin/python scripts/run_focus_demo.py --benchmark
    envs/focus/bin/python scripts/run_focus_demo.py --predict <image_dir> <out_dir>

Must run in the FOCUS environment (envs/focus), not the core one -- FOCUS pins
its own dependency stack and is deliberately isolated.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from adapters import focus_mac  # noqa: E402  (configures env before torch)

focus_mac.configure()

import numpy as np  # noqa: E402
import torch  # noqa: E402

TOC_MODEL = focus_mac.FOCUS_ROOT / "data/toc_model/densedepth_toc_predictor.pth"


def cmd_check() -> int:
    report = focus_mac.preflight()
    ok = True
    print("FOCUS Apple Silicon preflight\n" + "-" * 46)
    for k, v in report.items():
        if k == "modules":
            continue
        print(f"  {k:24} {v}")
        if k in ("pytorch3d_compiled_ext", "toc_model") and not v:
            ok = False
    print("  modules:")
    for m, status in report["modules"].items():
        print(f"    {'OK  ' if status is True else 'FAIL'} {m}"
              + ("" if status is True else f"  {status}"))
        ok = ok and status is True
    print("-" * 46)
    print("READY" if ok else "NOT READY -- see failures above")
    return 0 if ok else 1


def cmd_benchmark(n: int = 5) -> int:
    from FOCUS.toc_prediction.model import FootPredictorModel
    from FOCUS.toc_prediction.predict import _preprocess_image

    if not TOC_MODEL.exists():
        print(f"missing {TOC_MODEL}"); return 1

    rng = np.random.default_rng(0)
    x = _preprocess_image((rng.random((720, 540, 3)) * 255).astype(np.uint8))[None]

    outs, times = {}, {}
    for dev in ("cpu", focus_mac.network_device()):
        if dev in times:
            continue
        model = FootPredictorModel.load(str(TOC_MODEL), device=dev).eval()
        xd = x.to(dev)
        with torch.no_grad():
            model(xd)
            _sync(dev)
            ts = []
            for _ in range(n):
                t0 = time.time(); out = model(xd); _sync(dev); ts.append(time.time() - t0)
        times[dev] = float(np.median(ts))
        outs[dev] = {k: v.detach().cpu() for k, v in out.items()}
        print(f"  {dev:4}  {times[dev] * 1000:7.1f} ms/image")
        del model
        if dev == "mps":
            torch.mps.empty_cache()

    if "cpu" in times and "mps" in times:
        print(f"\n  MPS speedup: {times['cpu'] / times['mps']:.1f}x")
        print("\n  numerical agreement (MPS vs CPU):")
        worst = 0.0
        for k in outs["cpu"]:
            a, b = outs["cpu"][k], outs["mps"][k]
            rel = (a - b).abs().max().item() / max(a.abs().max().item(), 1e-8)
            worst = max(worst, rel)
            print(f"    {k:18} rel {rel:.2e}")
        print(f"  worst relative difference: {worst:.2e} "
              f"({'OK' if worst < 1e-3 else 'INVESTIGATE'})")
    return 0


def _sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()


def cmd_predict(src: Path, out: Path, batch_size: int) -> int:
    from FOCUS.data import io
    from FOCUS.toc_prediction import predict as toc

    imgs, keys = io.load_images_from_dir(src)
    if not imgs:
        print(f"no images in {src}"); return 1
    print(f"{len(imgs)} images -> {focus_mac.network_device()}")
    t0 = time.time()
    toc.predict_toc(imgs, TOC_MODEL, out, keys, batch_size=batch_size)
    dt = time.time() - t0
    print(f"\n{len(imgs)} images in {dt:.1f}s ({dt / len(imgs) * 1000:.0f} ms/image)")

    import cv2

    print(f"\n{'view':10}{'mask %':>9}")
    for k in keys:
        m = cv2.imread(str(out / k / "mask.png"), 0)
        if m is not None:
            print(f"{k:10}{100 * (m > 127).mean():>8.1f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="preflight the install")
    ap.add_argument("--benchmark", action="store_true", help="time MPS vs CPU and compare outputs")
    ap.add_argument("--predict", nargs=2, metavar=("SRC", "OUT"))
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    if args.check:
        return cmd_check()
    if args.benchmark:
        return cmd_benchmark()
    if args.predict:
        return cmd_predict(Path(args.predict[0]), Path(args.predict[1]), args.batch_size)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
