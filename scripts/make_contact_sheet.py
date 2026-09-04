#!/usr/bin/env python
"""Lay every pipeline stage out on one sheet, from a directory make_demo_run wrote.

    envs/core/bin/python scripts/make_contact_sheet.py experiments/demo_run
    envs/core/bin/python scripts/make_contact_sheet.py experiments/demo_external

Reads demo.json and the numbered previews beside it. Every word on the sheet --
the stage titles, the measured numbers, the caveats, the source and its licence
-- comes from that JSON, so the sheet cannot drift from the run it depicts.

The stage previews have aspect ratios from 0.9 to 2.9, so the row heights are
computed from the images themselves rather than fixed in advance. A uniform grid
spends most of its area on whitespace around the portrait plantar fields.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]

CAVEAT_COLOUR = "#a4560c"
MUTED = "#555555"

WIDTH_IN = 13.0
MARGIN_IN = 0.4
COL_GAP_IN = 0.35
ROW_GAP_IN = 0.30
TITLE_IN = 0.30          #: height reserved above each image for its heading
LINE_IN = 0.165          #: height of one wrapped caption line
CHARS_PER_INCH = 10.5    #: at the caption font size, for wrapping

#: Which stages share a row, and how much of the content width that row may
#: use. The wide panels are capped: drawn full-bleed, a 2.9:1 mesh render is
#: three times the height of the plantar fields and dominates the sheet.
ROWS = [(("video", "pose"), 1.00),
        (("features",), 1.00),
        (("pressure",), 0.62),
        (("canonical", "density", "contact"), 1.00),
        (("geometry",), 0.72)]


def _wrapped(text: str, width_in: float) -> list[str]:
    if not text:
        return []
    return textwrap.wrap(text, max(20, int(width_in * CHARS_PER_INCH)))


def _measure(stage: dict, cell_w: float) -> tuple[float, float]:
    """Height of a stage's image and of its caption block, in inches."""
    w, h = stage.get("preview_w") or 0, stage.get("preview_h") or 0
    img = cell_w * h / w if w and h else 0.0
    lines = len(_wrapped(stage.get("detail") or "", cell_w))
    lines += len(_wrapped(stage.get("caveat") or "", cell_w))
    if stage.get("caveat"):
        lines += 0.4                      # a blank half-line before the caveat
    return img, lines * LINE_IN + 0.10


def build(root: Path, out: Path) -> Path:
    meta = json.loads((root / "demo.json").read_text())
    stages = {s["key"]: s for s in meta["stages"]}
    in_dist = meta.get("in_distribution", False)

    content_w = WIDTH_IN - 2 * MARGIN_IN
    note = _wrapped(meta.get("note", ""), content_w)
    header_in = 1.34 + LINE_IN * len(note)

    # Pass one: measure every row so the figure is exactly as tall as it needs.
    plan = []
    for keys, width_frac in ROWS:
        row_w = content_w * width_frac
        cell_w = (row_w - COL_GAP_IN * (len(keys) - 1)) / len(keys)
        cells = [(stages[k],) + _measure(stages[k], cell_w) for k in keys]
        img_h = max(c[1] for c in cells)
        cap_h = max(c[2] for c in cells)
        plan.append((cells, cell_w, img_h, cap_h))

    height_in = (header_in + sum(TITLE_IN + r[2] + r[3] for r in plan)
                 + ROW_GAP_IN * len(plan) + MARGIN_IN)

    fig = plt.figure(figsize=(WIDTH_IN, height_in), facecolor="white")

    def fx(x_in: float) -> float:
        return x_in / WIDTH_IN

    def fy(y_in: float) -> float:
        return 1.0 - y_in / height_in

    fig.text(fx(MARGIN_IN), fy(0.45),
             "Every stage of the pipeline, run end to end on "
             + ("a clip from the training dataset" if in_dist
                else "video the model has never seen"),
             fontsize=17, va="baseline")
    fig.text(fx(MARGIN_IN), fy(0.70),
             f"{meta['clip']}  ·  {meta['video']}\n{meta['source']}",
             fontsize=9, color=MUTED, va="top", linespacing=1.6)
    ny = 1.32
    for line in note:
        fig.text(fx(MARGIN_IN), fy(ny), line, fontsize=8.5,
                 color="#111111" if in_dist else CAVEAT_COLOUR, va="baseline")
        ny += LINE_IN

    # Pass two: draw.
    y = header_in
    n = 0
    for cells, cell_w, img_h, cap_h in plan:
        for i, (stage, own_img, _own_cap) in enumerate(cells):
            n += 1
            x = MARGIN_IN + i * (cell_w + COL_GAP_IN)
            fig.text(fx(x), fy(y + TITLE_IN - 0.09),
                     f"{n:02d}  {stage['title']}", fontsize=11.5,
                     fontweight="bold", va="baseline")

            if own_img:
                # Centre a narrower image (a portrait plot) inside its cell.
                ax = fig.add_axes([fx(x), fy(y + TITLE_IN + own_img),
                                   cell_w / WIDTH_IN, own_img / height_in])
                ax.imshow(plt.imread(root / stage["preview"]))
                ax.axis("off")

            ty = y + TITLE_IN + img_h + 0.16
            for line in _wrapped(stage.get("detail") or "", cell_w):
                fig.text(fx(x), fy(ty), line, fontsize=8.5, color=MUTED,
                         va="baseline")
                ty += LINE_IN
            if stage.get("caveat"):
                ty += LINE_IN * 0.4
                for line in _wrapped(stage["caveat"], cell_w):
                    fig.text(fx(x), fy(ty), line, fontsize=8.5,
                             color=CAVEAT_COLOUR, va="baseline")
                    ty += LINE_IN
        y += TITLE_IN + img_h + cap_h + ROW_GAP_IN

    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path,
                    help="a directory written by scripts/make_demo_run.py")
    ap.add_argument("--out", type=Path, help="default: <run_dir>/contact_sheet.png")
    args = ap.parse_args()

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else REPO / p

    root = _resolve(args.run_dir)
    if not (root / "demo.json").exists():
        print(f"no demo.json in {root} -- run scripts/make_demo_run.py first")
        return 1

    out = _resolve(args.out) if args.out else root / "contact_sheet.png"
    out = build(root, out)
    print(f"wrote {out.relative_to(REPO)}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
