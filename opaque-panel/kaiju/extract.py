#!/usr/bin/env python3
"""Extract the tab systems from the Kaiju no Hanauta cover into .npy grayscale pages.

This video is laid out the other way round from the 짱구 one: the tab sits on an opaque
white panel across the bottom of the frame, so nothing bleeds through and there is no
contrast to recover. What does move is a red playhead and a highlight on the current
bar, and both are dealt with for free -- the panel is white and the marks are dark, so
the per-pixel temporal *maximum* over a page keeps the static engraving and erases
anything that swept past.

  1. scan at 10fps and mark the flips. Reduce each frame by its brightest channel first:
     that turns the saturated highlight white, so only the engraving moving counts as a
     page change (within a page the mask diff is 0.00025, at a flip it is 0.05+).
  2. per page, take the temporal max of full-res panels. Colour is kept: the bar
     numbers are red, and in a 136-bar score they are the thing you navigate by.

Unlike the 짱구 video the pages do not overlap -- each flip shows a fresh set of bars --
so they are concatenated rather than stitched. See make_pdf.py.
"""

import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)  # the repo root, for common.py and the source videos

import common  # noqa: E402
VIDEO = os.path.join(ROOT, 'Vaundy - Kaiju no Hanauta (guitar cover with tabs & chords).mp4')
WORK = common.work("sm3")

PANEL_Y, PANEL_H = 535, 545  # the white panel, constant for the whole video

SCAN_FPS = 10
SCAN_W, SCAN_H = 960, 274
DARK = 200  # a pixel darker than this, in the brightest channel, is engraving
FLIP = 0.01  # mask-diff within a page is 0.00025; at a flip it is 0.045 and up
MIN_PAGE = 1.5

SAMPLES = 14  # full-res frames max'd per page
TRIM_LEAD, TRIM_TAIL = 0.15, 0.10  # fraction of the page cut off, to clear transitions


def run(cmd):
    subprocess.run(cmd, check=True)


def scan():
    d = f"{WORK}/sB"
    if not os.path.isdir(d) or not os.listdir(d):
        os.makedirs(d, exist_ok=True)
        run(["ffmpeg", "-v", "error", "-i", VIDEO, "-vf",
             f"fps={SCAN_FPS},crop=1920:{PANEL_H}:0:{PANEL_Y},scale={SCAN_W}:{SCAN_H}",
             f"{d}/f%05d.png", "-y"])
    masks = []
    for f in sorted(glob.glob(f"{d}/*.png")):
        a = np.asarray(Image.open(f).convert("RGB")).astype(np.int16)
        masks.append(a.max(axis=2) < DARK)
    M = np.stack(masks)
    return np.array([(M[i] != M[i - 1]).mean() for i in range(1, len(M))]), len(M)


def pages():
    diff, n = scan()
    cuts = [0] + [i for i in range(1, n) if diff[i - 1] > FLIP] + [n]
    # a flip can straddle two scan frames; keep only the first of each cluster
    keep = [cuts[0]]
    for c in cuts[1:]:
        if c - keep[-1] > 3:
            keep.append(c)
    out = []
    for a, b in zip(keep, keep[1:]):
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
         "-vf", f"fps={fps:.4f},crop=1920:{PANEL_H}:0:{PANEL_Y}", f"{d}/p%03d.png", "-y"])

    top = None
    for f in sorted(glob.glob(f"{d}/*.png")):
        a_ = np.asarray(Image.open(f).convert("RGB"))
        top = a_ if top is None else np.maximum(top, a_)
    for f in glob.glob(f"{d}/*.png"):
        os.remove(f)
    return top


if __name__ == "__main__":
    ps = pages()
    print(f"{len(ps)} pages")
    os.makedirs(f"{WORK}/pages", exist_ok=True)
    os.makedirs(f"{WORK}/png", exist_ok=True)
    which = [int(x) for x in sys.argv[1:]] or range(len(ps))
    for i in which:
        t0, t1 = ps[i]
        g = render(t0, t1, i)
        np.save(f"{WORK}/pages/{i:03d}.npy", g)
        Image.fromarray(g).save(f"{WORK}/png/{i:03d}.png")
        print(f"  page {i:02d}  {t0:6.1f}-{t1:6.1f}s"
              f"  ink={(g.max(axis=2) < 200).mean():.4f}",
              flush=True)
