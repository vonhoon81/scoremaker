#!/usr/bin/env python3
"""Extract the tab systems from the AironA 怪獣の花唄 cover into .npy colour pages.

This video is a scrolling window, not a slide deck. The tab sits on an opaque white
panel across the bottom of the frame; the panel holds however many bars fit its width
(5 in the dense verses, 12 in the rest-heavy bridge) and a black rectangle boxes the bar
being played. The window advances only when the box has walked to the *second to last*
bar shown -- the new window then opens on the bar that used to be last. So every pair of
consecutive windows overlaps by exactly one bar, which stitch.py has to remove.

  1. scan a full-res band containing the box's top and bottom edges. Those edges are
     1px black horizontal runs a few hundred px long, and nothing else in the band comes
     close, so the longest black run in a frame *is* the box. Downscaling the band first
     would be cheaper but blends the 1px line into the white panel and loses it.
  2. cut the timeline where the box jumps backwards: that is a window flip.
  3. per window, take the per-pixel temporal *median* of full-res panels. The box and the
     red "currently sounding" digits are both transient -- each covers a given pixel for
     at most one bar out of the four the window is on screen -- so the median keeps the
     black-on-white engraving and erases both. A temporal max would erase the box but
     turn every played digit red; a min would keep every box outline.
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
VIDEO = os.path.join(ROOT, '【TAB譜】怪獣の花唄 Vaundy ギター 弾いてみた.mp4')
WORK = common.work("sm6")

W = 1920
PANEL_Y, PANEL_H = 806, 274  # the white panel, constant for the whole video
BAND_Y, BAND_H = 845, 145  # contains the box's top (847) and bottom (984) edges

SCAN_FPS = 5  # a bar lasts 1.6s at 150bpm, so this is 8 samples per box step
BLACK = 100  # the engraving and the box are black; the staff lines are grey (234)
MIN_RUN = 90  # shortest box seen is the bridge's narrow rest bars, at ~130px
BACK_JUMP = 40  # px the box must retreat to count as a flip, not detection jitter
MIN_PAGE = 2.0  # seconds

SAMPLES = 15  # full-res frames medianed per window
TRIM_LEAD, TRIM_TAIL = 0.10, 0.08  # fraction of the window dropped at either end


def run(cmd):
    subprocess.run(cmd, check=True)


def longest_black_run(row):
    """(length, start) of the longest run of black pixels in a boolean row."""
    d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if len(s) == 0:
        return 0, 0
    i = (e - s).argmax()
    return int(e[i] - s[i]), int(s[i])


def scan():
    """Box left edge per scan frame, or None where no box is on screen."""
    d = f"{WORK}/band"
    if not os.path.isdir(d) or not os.listdir(d):
        os.makedirs(d, exist_ok=True)
        run(["ffmpeg", "-v", "error", "-i", VIDEO, "-vf",
             f"fps={SCAN_FPS},crop={W}:{BAND_H}:0:{BAND_Y}", f"{d}/f%05d.png", "-y"])
    out = []
    for f in sorted(glob.glob(f"{d}/*.png")):
        b = np.asarray(Image.open(f).convert("RGB")).max(axis=2) < BLACK
        best = (0, 0)
        for y in range(b.shape[0]):
            best = max(best, longest_black_run(b[y]))
        out.append(best[1] if best[0] >= MIN_RUN else None)
    return out


def pages(xs):
    """[(t0, t1)] for each window, from the frames where the box moves backwards."""
    cuts, prev = [], None
    for i, x in enumerate(xs):
        if x is None:
            continue
        if prev is None or x < prev - BACK_JUMP:
            cuts.append(i)
        prev = x
    bounds = cuts + [len(xs)]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        if (b - a) / SCAN_FPS >= MIN_PAGE:
            out.append((a / SCAN_FPS, (b - 1) / SCAN_FPS))
    return out


def render(t0, t1, idx):
    span = t1 - t0
    a, b = t0 + span * TRIM_LEAD, t1 - span * TRIM_TAIL
    fps = max(SAMPLES / max(b - a, 0.2), 1.0)

    d = f"{WORK}/fr/{idx:03d}"
    os.makedirs(d, exist_ok=True)
    for old in glob.glob(f"{d}/*.png"):
        os.remove(old)
    run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", VIDEO,
         "-vf", f"fps={fps:.4f},crop={W}:{PANEL_H}:0:{PANEL_Y}", f"{d}/p%03d.png", "-y"])

    fs = sorted(glob.glob(f"{d}/*.png"))
    stack = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in fs])
    med = np.median(stack, axis=0).astype(np.uint8)
    for f in fs:
        os.remove(f)
    return med, len(fs)


if __name__ == "__main__":
    xs = scan()
    ps = pages(xs)
    print(f"{len(xs)} scan frames, {len(ps)} windows")
    os.makedirs(f"{WORK}/pages", exist_ok=True)
    os.makedirs(f"{WORK}/png", exist_ok=True)
    with open(f"{WORK}/pages.json", "w") as fh:
        json.dump(ps, fh)
    which = [int(x) for x in sys.argv[1:]] or range(len(ps))
    for i in which:
        t0, t1 = ps[i]
        g, n = render(t0, t1, i)
        np.save(f"{WORK}/pages/{i:03d}.npy", g)
        Image.fromarray(g).save(f"{WORK}/png/{i:03d}.png")
        print(f"  page {i:03d}  {t0:6.1f}-{t1:6.1f}s  n={n:2d}"
              f"  ink={(g.max(axis=2) < 200).mean():.4f}", flush=True)
