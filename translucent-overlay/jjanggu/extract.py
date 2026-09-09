#!/usr/bin/env python3
"""Extract the guitar-tab systems from the 짱구 브금 lesson video into .npy alpha maps.

The video shows one static tab system at a time in a translucent dark strip pinned to
the bottom of the frame; it flips to the next system every few seconds while a playhead
and a highlight sweep across it. So:

  1. scan the strip at 10fps and mark the flips. A raw pixel diff can't tell a flip from
     the highlight moving, so diff a *top-hat* mask instead -- that ignores the large,
     slow brightness changes the highlight makes and only sees the marks move.
  2. per page, take the per-pixel temporal minimum of full-res strips. The tab is static
     and the player behind it is not, so this pushes the bleed-through toward its darkest.
  3. top-hat to drop what bleed-through remains, then divide by a per-column staff-line
     reference: where the guitar body glows through, the overlay's contrast (and hence
     every mark's top-hat response) is scaled down, and the six string lines run the full
     width of every system, so they measure that attenuation exactly.
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

# the only source that is not kept beside the scripts
VIDEO = os.path.expanduser(
    "~/다운로드/기타 1일차 vs 기타 10년차 ｜ 짱구 브금 [악보있음].mp4")
WORK = common.work("sm2")

BAND_Y, BAND_H = 858, 210  # the tab strip; below it is letterbox, above it is video
N_STRINGS = 6
SPACING = (21.0, 24.0)  # string-line pitch to search over, in pixels
MIN_REF = 50.0  # a column with a weaker staff line than this carries no tab
# (real systems sit at 61 and up; video leaking in beside the panel reaches 39)

SCAN_FPS = 10
SCAN_W, SCAN_H = 960, 105
MASK_SE = 21  # horizontal top-hat width for the scan mask
MASK_LEVEL = 28
FLIP = 0.05  # mask-diff at a page flip is ~0.13; within a page it is ~0.004
INK_PRESENT = 0.03  # mask ink below this means no tab on screen (title card, intro)
MIN_PAGE = 1.2  # seconds; shorter "pages" are just flip transitions

SAMPLES = 20  # full-res frames min'd per page
TRIM = 0.12  # fraction of the page cut off each end, to clear the transitions
SE = 41  # top-hat structuring element; must exceed the thickest mark
NORM_LO, NORM_HI = 0.42, 1.15  # mark strength relative to the staff line, mapped to 0..1
MIN_COVER = 0.35  # a page needs staff lines across at least this much of its width


def run(cmd):
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- morphology


def _slide(a, k, op, axis):
    r = k // 2
    p = np.pad(a, ((r, r), (0, 0)) if axis == 0 else ((0, 0), (r, r)), mode="edge")
    if axis == 1:
        out = p[:, 0:p.shape[1] - k + 1].copy()
        for i in range(1, k):
            out = op(out, p[:, i:p.shape[1] - k + 1 + i])
    else:
        out = p[0:p.shape[0] - k + 1].copy()
        for i in range(1, k):
            out = op(out, p[i:p.shape[0] - k + 1 + i])
    return out


def opening(a, k, axes=(0, 1)):
    for ax in axes:
        a = _slide(a, k, np.minimum, ax)
    for ax in axes:
        a = _slide(a, k, np.maximum, ax)
    return a


def tophat(a, k, axes=(0, 1)):
    return a - opening(a, k, axes)


# ---------------------------------------------------------------- stage 1: pages


def scan():
    d = f"{WORK}/scan2"
    if not os.path.isdir(d) or not os.listdir(d):
        os.makedirs(d, exist_ok=True)
        run(["ffmpeg", "-v", "error", "-i", VIDEO, "-vf",
             f"fps={SCAN_FPS},crop=1920:{BAND_H}:0:{BAND_Y},scale={SCAN_W}:{SCAN_H}",
             f"{d}/f%05d.png", "-y"])

    masks = []
    for f in sorted(glob.glob(f"{d}/*.png")):
        g = np.asarray(Image.open(f).convert("L")).astype(np.int16)
        masks.append(tophat(g, MASK_SE, axes=(1,)) > MASK_LEVEL)
    M = np.stack(masks)
    ink = M.mean(axis=(1, 2))
    diff = np.array([(M[i] != M[i - 1]).mean() for i in range(1, len(M))])
    return ink, diff


def pages():
    """Page time ranges, as (t0, t1) in seconds."""
    ink, diff = scan()
    n = len(ink)
    present = ink > INK_PRESENT

    out, start = [], None
    for i in range(n):
        flip = i > 0 and diff[i - 1] > FLIP
        if present[i] and (start is None):
            start, prev_flip = i, True
        elif start is not None and (flip or not present[i]):
            if (i - start) / SCAN_FPS >= MIN_PAGE:
                out.append((start / SCAN_FPS, (i - 1) / SCAN_FPS))
            start = i if present[i] else None
    if start is not None and (n - start) / SCAN_FPS >= MIN_PAGE:
        out.append((start / SCAN_FPS, (n - 1) / SCAN_FPS))
    return out


# ---------------------------------------------------------------- stage 2+3: a page


def find_staff(th):
    """Locate the six string lines by sliding a six-tooth comb down the row profile.
    They are not always at the same height -- the closing system is drawn in a narrow
    panel that sits about nine pixels lower than the full-width strip."""
    rp = th.mean(axis=1)
    best = (-1.0, None)
    for pitch in np.arange(SPACING[0], SPACING[1], 0.1):
        span = pitch * (N_STRINGS - 1)
        for top in np.arange(10, BAND_H - span - 10, 0.5):
            ys = np.round(top + pitch * np.arange(N_STRINGS)).astype(int)
            score = rp[ys].sum()
            if score > best[0]:
                best = (score, ys)
    return best[1]


def longest_run(v):
    """Keep only the longest contiguous True run -- drops the video that leaks in
    beside the narrow closing panel."""
    d = np.diff(np.concatenate(([0], v.view(np.int8), [0])))
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = np.zeros_like(v)
    if len(starts):
        i = np.argmax(stops - starts)
        out[starts[i]:stops[i]] = True
    return out


def staff_ref(th, staff, k=81):
    """Per-column top-hat response of the string lines: the yardstick every other mark
    on that column is measured against. Columns where the lines themselves are missing
    hold no tab at all -- outside the closing panel, or on a frame with no overlay."""
    rows = np.stack([th[y - 1:y + 2].max(axis=0) for y in staff])
    # second-weakest line, not the median: a stray bright edge in the video beside the
    # closing panel can fake three or four of the six, but not five. A full six-string
    # chord does hide them all, and the running median below bridges that.
    ref = np.sort(rows, axis=0)[1]
    p = np.pad(ref, k // 2, mode="edge")
    ref = np.median(np.lib.stride_tricks.sliding_window_view(p, k), axis=1)
    return ref


def render(t0, t1, idx):
    span = t1 - t0
    a, b = t0 + span * TRIM, t1 - span * TRIM
    fps = max(SAMPLES / max(b - a, 0.2), 1.0)

    d = f"{WORK}/fr/{idx:03d}"
    os.makedirs(d, exist_ok=True)
    for old in glob.glob(f"{d}/*.png"):
        os.remove(old)
    run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", VIDEO,
         "-vf", f"fps={fps:.4f},crop=1920:{BAND_H}:0:{BAND_Y}", f"{d}/p%03d.png", "-y"])

    tmin = None
    for f in sorted(glob.glob(f"{d}/*.png")):
        g = np.asarray(Image.open(f).convert("L"))
        tmin = g if tmin is None else np.minimum(tmin, g)
    for f in glob.glob(f"{d}/*.png"):
        os.remove(f)

    th = tophat(tmin.astype(np.int16), SE).astype(np.float32)
    staff = find_staff(th)
    ref = staff_ref(th, staff)
    valid = longest_run(ref >= MIN_REF)  # a system is one unbroken strip
    alpha = th / np.maximum(ref, MIN_REF)[None, :]
    alpha = np.clip((alpha - NORM_LO) / (NORM_HI - NORM_LO), 0.0, 1.0)
    alpha[:, ~valid] = 0.0
    return alpha, staff, valid


if __name__ == "__main__":
    ps = pages()
    print(f"{len(ps)} candidate pages")
    os.makedirs(f"{WORK}/pages", exist_ok=True)
    os.makedirs(f"{WORK}/png", exist_ok=True)
    which = [int(x) for x in sys.argv[1:]] or range(len(ps))
    for i in which:
        t0, t1 = ps[i]
        a, staff, valid = render(t0, t1, i)
        cover = valid.mean()
        tag = "" if cover >= MIN_COVER else "   <- no tab, dropped"
        if cover >= MIN_COVER:
            np.save(f"{WORK}/pages/{i:03d}.npy", a)
            Image.fromarray(((1 - a) * 255).astype(np.uint8)).save(
                f"{WORK}/png/{i:03d}.png")
        print(f"  page {i:02d}  {t0:6.1f}-{t1:6.1f}s  staff@{staff[0]:3d}"
              f"  cover={cover:.2f}  ink={(a >= 0.15).mean():.4f}{tag}", flush=True)
