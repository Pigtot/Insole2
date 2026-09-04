#!/usr/bin/env python
"""Fetch a registered dataset, with resume, checksum verification and a size gate.

    envs/core/bin/python scripts/download_data.py --list
    envs/core/bin/python scripts/download_data.py insole_gaitrite [--verify-only]
    envs/core/bin/python scripts/download_data.py sample_videos

Policy: anything above the registry's large-download threshold refuses to run
without --yes, and every download refuses to start if it would not fit on disk.

A dataset may register a PREPARE step, run once the bytes verify, for work that
has to happen before the files are usable -- transcoding, trimming, extraction.
It is keyed by dataset name so the registry itself stays declarative.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# macOS framework Python ships no CA bundle, and urllib would fail the TLS
# handshake against every provider. Same fix as scripts/extract_pose.py.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:  # pragma: no cover
    pass

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.data import registry  # noqa: E402


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


#: Wikimedia serves 403 to the default Python-urllib agent, and its policy asks
#: for a descriptive one. Harmless everywhere else.
USER_AGENT = "visole/0.1 (research project; https://github.com/) urllib"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while block := resp.read(1 << 20):
            out.write(block)
            done += len(block)
            if total:
                pct = 100 * done / total
                print(f"\r  {human(done)} / {human(total)} ({pct:5.1f}%)", end="", flush=True)
    print()
    tmp.replace(dest)


def _prepare_sample_videos(spec) -> int:
    """Transcode the Commons .ogv into the 10 s H.264 window the pipeline reads.

    Two reasons this is not optional. Theora is not what the dashboard's upload
    path accepts, and ``extract_clip_pose`` never seeks -- it stops once it has
    ``max_frames`` samples, so a 45 s source would only ever expose its opening
    seconds. Trimming here picks the window instead of leaving it to chance.
    """
    if not shutil.which("ffmpeg"):
        print("  ffmpeg not found; skipping transcode. brew install ffmpeg")
        return 1

    src = spec.dest_path / "treadmill_walk.ogv"
    dest = spec.dest_path / "treadmill_walk.mp4"
    if dest.exists():
        print(f"[have] {dest.name}")
        return 0

    print(f"[prep] {dest.name}  (6 s -> 16 s, H.264)")
    cmd = ["ffmpeg", "-v", "error", "-ss", "6", "-t", "10", "-i", str(src),
           "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
           "-pix_fmt", "yuv420p", "-y", str(dest)]
    if subprocess.run(cmd, check=False).returncode != 0:
        print("  ffmpeg failed")
        return 1
    print(f"  wrote {human(dest.stat().st_size)}")
    return 0


#: Post-download work, keyed by dataset name. Runs only after the bytes verify.
PREPARE = {"sample_videos": _prepare_sample_videos}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", help="registry name")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="check existing files' checksums, download nothing")
    ap.add_argument("--yes", action="store_true", help="approve a large download")
    args = ap.parse_args()

    if args.list or not args.dataset:
        for spec in registry.REGISTRY.values():
            print(f"{spec.name:20} {human(spec.total_bytes):>10}  {spec.licence:12} {spec.title}")
            print(f"{'':20} doi:{spec.doi}  -> data/{spec.dest}")
        return 0

    spec = registry.get(args.dataset)
    print(f"{spec.title}\n  doi:{spec.doi}  licence:{spec.licence}")
    print(f"  total {human(spec.total_bytes)} -> {spec.dest_path.relative_to(REPO)}")
    for n in spec.notes:
        print(f"  ! {n}")

    free = shutil.disk_usage(REPO).free
    print(f"  disk free: {human(free)}")
    if not args.verify_only:
        if spec.total_bytes * 2 > free:
            print("REFUSING: not enough free disk for the archive plus its extraction.")
            return 2
        if spec.is_large and not args.yes:
            print(f"REFUSING: {human(spec.total_bytes)} exceeds the large-download "
                  "threshold. Re-run with --yes if this is intended.")
            return 2

    failures = 0
    for f in spec.files:
        dest = spec.dest_path / f.key
        if dest.exists() and f.size_bytes and dest.stat().st_size == f.size_bytes:
            print(f"[have] {f.key}")
        elif args.verify_only:
            print(f"[MISSING] {f.key}")
            failures += 1
            continue
        else:
            print(f"[get ] {f.key}")
            download(f.url, dest)

        if f.size_bytes and dest.stat().st_size != f.size_bytes:
            print(f"  SIZE MISMATCH: {dest.stat().st_size} != {f.size_bytes}")
            failures += 1
            continue
        if f.md5:
            got = md5sum(dest)
            ok = got == f.md5
            print(f"  md5 {'OK' if ok else 'MISMATCH'}: {got}")
            failures += 0 if ok else 1

    print("\nAll files verified." if not failures else f"\n{failures} problem(s).")
    if not failures and not args.verify_only and spec.name in PREPARE:
        failures += PREPARE[spec.name](spec)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
