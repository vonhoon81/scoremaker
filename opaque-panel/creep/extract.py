#!/usr/bin/env python3
"""Extract the tab windows from the Creep lesson video into .npy alpha maps.

The tab sits on an opaque black panel pinned to the bottom of the frame -- nothing
bleeds through it, so unlike the earlier videos there is no background to subtract.
What this one does instead is switch layouts: for most of the lesson there is a single
full-width panel, but through "Chorus 1 (both guitars)" the frame splits in two and each
half carries its own panel showing a different part of the score. So a frame holds one
or two independent tab windows, and each window steps forward a couple of bars at a
time while overlapping the window before it.

  1. scan the band at SCAN_FPS and, per frame, decide single or split layout
  2. cut the timeline into runs of constant layout; within a run each panel is its own
     stream of pages, and a page boundary is a jump in the panel's ink mask
  3. per page take the per-pixel temporal maximum of full-res crops -- the marks are
     white and static, the playhead sweeping over them only ever darkens
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)  # the repo root, for common.py and the source videos

import common  # noqa: E402
VIDEO = os.path.join(ROOT, 'Radiohead - Creep (Guitar lesson with TAB).mp4')
WORK = common.work("sm5")

W, H = 1920, 1080
BAND_Y, BAND_H = 612, 468  # the black tab panel; below it is the frame edge
# in split layout a white rule runs down these columns between the two panels
DIV_X0, DIV_X1 = 958, 970
BORDER = 2

SCAN_FPS = 8
SCAN_W = 960
SPLIT_COVER = 0.5  # fraction of the band height the divider must be bright over

MARK = 130  # a tab mark is white (255) on a panel that sits at 11
FLIP = 0.012  # a step moves 2 bars of marks; inside a page frames differ by <0.002
MIN_PAGE = 0.9  # seconds; anything shorter is a step transition, not a page
INK_PRESENT = 0.004  # below this the panel is blank (title card, intro count-in)
MUSIC_END = 206.0  # the closing chapter is a patron roll on the same black panel

SAMPLES = 10  # full-res frames combined per page
TRIM = 0.18  # fraction of the page dropped at each end, to clear the step
LO, HI = 80, 205  # panel grey mapped to alpha 0..1; the panel itself sits at 11 and
# carries a faint beat grid around 60, which this drops without eating the marks


def run(cmd):
    subprocess.run(cmd, check=True)


def band_frames(fps, width, out):
    """Band crops at `fps`, `width` px wide, as a sorted list of paths."""
    if not os.path.isdir(out) or not os.listdir(out):
        os.makedirs(out, exist_ok=True)
        run(["ffmpeg", "-v", "error", "-i", VIDEO, "-vf",
             f"fps={fps},crop={W}:{BAND_H}:0:{BAND_Y},scale={width}:-1",
             f"{out}/f%05d.png", "-y"])
    return sorted(glob.glob(f"{out}/*.png"))


# ---------------------------------------------------------------- stage 1: layout


def is_split(g):
    """True if the bright vertical rule between the two panels is present."""
    sx = g.shape[1] / W
    strip = g[:, round(DIV_X0 * sx):max(round(DIV_X1 * sx), round(DIV_X0 * sx) + 1)]
    return (strip.max(axis=1) > MARK).mean() > SPLIT_COVER


def regions(split):
    """Panel x-ranges for a layout, already inset past the rule between panels."""
    if split:
        return [(BORDER, DIV_X0 - BORDER), (DIV_X1 + BORDER, W - BORDER)]
    return [(BORDER, W - BORDER)]


def layouts():
    """[(first frame, last frame, split?)] over the scan, in scan-frame indices."""
    fs = band_frames(SCAN_FPS, SCAN_W, f"{WORK}/scan")
    G = np.stack([np.asarray(Image.open(f).convert("L")) for f in fs]).astype(np.int16)
    flags = [is_split(g) for g in G]
    out, start = [], 0
    for i in range(1, len(flags)):
        if flags[i] != flags[i - 1]:
            out.append((start, i - 1, flags[i - 1]))
            start = i
    out.append((start, len(flags) - 1, flags[-1]))
    # a lone frame straddling the cross-fade is not a layout of its own
    return [(a, b, s) for a, b, s in out if (b - a + 1) / SCAN_FPS >= 1.0], G


# ---------------------------------------------------------------- stage 2: pages


def pages():
    """Every tab window shown, as (t0, t1, region x0, region x1, layout index)."""
    lays, G = layouts()
    sx = G.shape[2] / W
    out = []
    for n, (a, b, split) in enumerate(lays):
        for x0, x1 in regions(split):
            c0, c1 = round(x0 * sx), round(x1 * sx)
            M = G[a:b + 1, :, c0:c1] > MARK
            ink = M.mean(axis=(1, 2))
            diff = np.array([(M[i] != M[i - 1]).mean() for i in range(1, len(M))])

            start = None
            for i in range(len(M)):
                step = i > 0 and diff[i - 1] > FLIP
                if ink[i] > INK_PRESENT and start is None:
                    start = i
                elif start is not None and (step or ink[i] <= INK_PRESENT):
                    if (i - start) / SCAN_FPS >= MIN_PAGE:
                        out.append(((a + start) / SCAN_FPS, (a + i - 1) / SCAN_FPS,
                                    x0, x1, n))
                    start = i if ink[i] > INK_PRESENT else None
            if start is not None and (len(M) - start) / SCAN_FPS >= MIN_PAGE:
                out.append(((a + start) / SCAN_FPS, (a + len(M) - 1) / SCAN_FPS,
                            x0, x1, n))
    out = [p for p in out if p[0] < MUSIC_END]
    out.sort(key=lambda p: (p[4], p[2], p[0]))
    return out


# ---------------------------------------------------------------- stage 3: one page


def render(t0, t1, x0, x1, idx):
    span = t1 - t0
    a, b = t0 + span * TRIM, t1 - span * TRIM
    fps = max(SAMPLES / max(b - a, 0.2), 1.0)

    d = f"{WORK}/fr/{idx:03d}"
    os.makedirs(d, exist_ok=True)
    for old in glob.glob(f"{d}/*.png"):
        os.remove(old)
    run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", VIDEO,
         "-vf", f"fps={fps:.4f},crop={x1 - x0}:{BAND_H}:{x0}:{BAND_Y}",
         f"{d}/p%03d.png", "-y"])

    top = None
    for f in sorted(glob.glob(f"{d}/*.png")):
        g = np.asarray(Image.open(f).convert("L"))
        top = g if top is None else np.maximum(top, g)
    for f in glob.glob(f"{d}/*.png"):
        os.remove(f)

    return np.clip((top.astype(np.float32) - LO) / (HI - LO), 0.0, 1.0)


if __name__ == "__main__":
    ps = pages()
    print(f"{len(ps)} tab windows")
    os.makedirs(f"{WORK}/pages", exist_ok=True)
    os.makedirs(f"{WORK}/png", exist_ok=True)
    with open(f"{WORK}/pages.json", "w") as fh:
        json.dump(ps, fh)
    which = [int(x) for x in sys.argv[1:]] or range(len(ps))
    for i in which:
        t0, t1, x0, x1, n = ps[i]
        a = render(t0, t1, x0, x1, i)
        np.save(f"{WORK}/pages/{i:03d}.npy", a)
        Image.fromarray(((1 - a) * 255).astype(np.uint8)).save(f"{WORK}/png/{i:03d}.png")
        print(f"  {i:03d}  layout {n}  x{x0}-{x1}  {t0:6.1f}-{t1:6.1f}s"
              f"  ink={(a >= 0.5).mean():.4f}", flush=True)
