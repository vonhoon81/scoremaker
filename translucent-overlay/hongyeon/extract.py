#!/usr/bin/env python3
"""Extract guitar-tab systems from a lesson video into a printable PDF.

Pipeline:
  1. scan the video at low res to find the frame ranges where the tab overlay is static
  2. for each such "page", pull full-res bottom-half frames and take the per-pixel
     temporal minimum (kills the moving performance behind the translucent overlay)
  3. white top-hat on that minimum to estimate and subtract the remaining background,
     recovering the ~50%-opacity staff lines and chord-diagram fret grids
  4. globally dedupe the pages, trim, and lay the survivors out one-per-line in a PDF
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
VIDEO = os.path.join(ROOT, '648.안예은 - 홍연 (Guitar Tab).mp4')
WORK = common.work("sm")

SCAN_FPS = 4  # page changes are >=4s apart, so 4fps never misses one
SCAN_W = 640
CHANGE_THRESHOLD = 0.004  # mask-diff fraction; noise floor is ~0.0006
SAMPLES_PER_PAGE = 12  # full-res frames averaged (min'd) per page

SE = 61  # structuring element for the opening; must exceed the thickest overlay stroke
TOPHAT_LO, TOPHAT_HI = 18, 95  # top-hat response mapped to alpha 0..1

# Genuine overlay marks are flat white: 44-80% of their pixels sit above 0.9. Video
# that survived the top-hat (a hand held still) is soft gradient -- it may spike above
# 0.8 in one spot but has almost nothing above 0.9. So filter blobs on both peak
# brightness and how solidly white they are, and drop specks outright.
KEEP_ALPHA = 0.80
KEEP_SOLID = 0.15  # min fraction above 0.9, applied to blobs big enough to judge
SOLID_MIN_AREA = 300
KEEP_AREA = 12
FLOOR = 0.15  # below this, treat as paper


def run(cmd):
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- separable morphology


def _slide(a, k, op):
    """Separable min/max filter with a k x k square structuring element."""
    r = k // 2
    pad = np.pad(a, r, mode="edge")
    out = pad
    # horizontal pass
    acc = out[:, 0 : out.shape[1] - k + 1].copy()
    for i in range(1, k):
        acc = op(acc, out[:, i : out.shape[1] - k + 1 + i])
    # vertical pass
    res = acc[0 : acc.shape[0] - k + 1].copy()
    for i in range(1, k):
        res = op(res, acc[i : acc.shape[0] - k + 1 + i])
    return res


def opening(a, k):
    return _slide(_slide(a, k, np.minimum), k, np.maximum)


# ---------------------------------------------------------------- blob filtering


def _runs(row):
    """Start/stop indices of each True run in a boolean row."""
    d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def drop_video_leftovers(alpha):
    """Zero out 8-connected blobs that never reach KEEP_ALPHA, or are tiny."""
    mask = alpha >= FLOOR
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        x, y = find(x), find(y)
        if x != y:
            parent[y] = x

    labels = np.zeros(mask.shape, dtype=np.int32)
    p_starts, p_stops, p_labs = [], [], []
    for y in range(mask.shape[0]):
        starts, stops = _runs(mask[y])
        c_starts, c_stops, c_labs = [], [], []
        j = 0  # runs are sorted, so this pointer only ever moves forward
        for s, e in zip(starts.tolist(), stops.tolist()):
            parent.append(len(parent))
            lab = len(parent) - 1
            while j < len(p_stops) and p_stops[j] < s:
                j += 1
            k = j
            while k < len(p_starts) and p_starts[k] <= e:  # 8-connectivity
                union(p_labs[k], lab)
                lab = find(lab)
                k += 1
            labels[y, s:e] = lab
            c_starts.append(s)
            c_stops.append(e)
            c_labs.append(lab)
        p_starts, p_stops, p_labs = c_starts, c_stops, c_labs

    if len(parent) == 1:
        return alpha
    roots = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    flat_lab = roots[labels.ravel()]
    n = len(parent)
    flat_alpha = alpha.ravel()
    area = np.bincount(flat_lab, minlength=n)
    peak = np.zeros(n, dtype=np.float32)
    np.maximum.at(peak, flat_lab, flat_alpha)
    solid = np.bincount(flat_lab[flat_alpha > 0.9], minlength=n) / np.maximum(area, 1)

    keep = (peak >= KEEP_ALPHA) & (area >= KEEP_AREA)
    keep &= (area < SOLID_MIN_AREA) | (solid >= KEEP_SOLID)
    keep[0] = False  # label 0 is background
    out = np.where(keep[flat_lab].reshape(alpha.shape), alpha, 0.0)
    return np.where(out >= FLOOR, out, 0.0)


# ---------------------------------------------------------------- stage 1: page ranges


def scan_pages():
    d = f"{WORK}/scan"
    if not os.path.isdir(d) or not os.listdir(d):
        os.makedirs(d, exist_ok=True)
        run(["ffmpeg", "-v", "error", "-i", VIDEO, "-vf",
             f"fps={SCAN_FPS},crop=iw:ih/2:0:ih/2,scale={SCAN_W}:-1",
             "-q:v", "3", f"{d}/f%05d.jpg", "-y"])

    files = sorted(glob.glob(f"{d}/*.jpg"))
    masks = []
    for f in files:
        a = np.asarray(Image.open(f).convert("RGB")).astype(np.int16)
        masks.append(a.min(axis=2) >= 185)

    segs, start = [], 0
    for i in range(1, len(masks)):
        if (masks[i] != masks[i - 1]).mean() > CHANGE_THRESHOLD:
            segs.append((start, i - 1))
            start = i
    segs.append((start, len(masks) - 1))
    return [(a / SCAN_FPS, (b + 1) / SCAN_FPS) for a, b in segs]


# ---------------------------------------------------------------- stage 2+3: one page


def render_page(t0, t1, idx, crop_top=1080):
    """crop_top: first row of the frame to keep. Defaults to the halfway cut, but a
    few systems put their section label above it and need extra headroom."""
    # stay clear of the transitions at either end
    span = t1 - t0
    a, b = t0 + span * 0.25, t1 - span * 0.15
    fps = max(SAMPLES_PER_PAGE / (b - a), 1.0)

    d = f"{WORK}/frames/{idx:03d}"
    os.makedirs(d, exist_ok=True)
    for old in glob.glob(f"{d}/*.png"):
        os.remove(old)
    run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", VIDEO,
         "-vf", f"fps={fps:.4f},crop=iw:ih-{crop_top}:0:{crop_top}",
         f"{d}/p%03d.png", "-y"])

    tmin = None
    for f in sorted(glob.glob(f"{d}/*.png")):
        mn = np.asarray(Image.open(f).convert("RGB")).min(axis=2)
        tmin = mn if tmin is None else np.minimum(tmin, mn)
    for f in glob.glob(f"{d}/*.png"):
        os.remove(f)

    tmin = tmin.astype(np.int16)
    tophat = tmin - opening(tmin, SE)
    alpha = (tophat.astype(np.float32) - TOPHAT_LO) / (TOPHAT_HI - TOPHAT_LO)
    return drop_video_leftovers(np.clip(alpha, 0.0, 1.0))


if __name__ == "__main__":
    pages = scan_pages()
    print(f"{len(pages)} static tab pages", flush=True)
    os.makedirs(f"{WORK}/pages", exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith("--top=")]
    top = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--top=")), 1080)
    which = [int(x) for x in args] or range(len(pages))
    for i in which:
        t0, t1 = pages[i]
        alpha = render_page(t0, t1, i, crop_top=top)
        np.save(f"{WORK}/pages/{i:03d}.npy", alpha)
        print(f"  page {i:02d}  {t0:6.2f}-{t1:6.2f}s  ink={alpha.mean():.4f}", flush=True)
