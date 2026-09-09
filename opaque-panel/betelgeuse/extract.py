#!/usr/bin/env python3
"""Extract the tab systems from the BETELGEUSE cover, one set per guitar part.

The video plays the song through three times -- once for each guitar part -- with a
"Gt.1 2 3" badge in the corner lit on whichever is current. The tab sits on an opaque
white panel across the bottom and flips a system at a time; within a page literally
nothing moves (consecutive frames differ by 0.00001 of the mask), so a flip is
unmistakable and a page needs only a handful of frames to capture.

Because all three passes are the same performance at the same tempo, their flips land
at the same offsets within each pass, which is what lets make_pdf.py stack them into
one score. Part 1 carries one extra short page at the very top -- an intro fragment
before the music starts -- and dropping it by duration lines all three up at 21 pages.
"""

import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # where the source videos live
VIDEO = os.path.join(ROOT, 'Yuuri - BETELGEUSE ｜ 일렉 기타 커버 타브 악보 노래방.mp4')
WORK = "/tmp/sm4"

PANEL_Y, PANEL_H = 650, 430  # the white tab panel; above it is the performance video
BADGE = (120, 8, 140, 60)  # x, y, w, h of the three digits in the "Gt.123" badge

SCAN_FPS = 10
SCAN_W, SCAN_H = 640, 143
DARK = 170
FLIP = 0.02  # within a page the mask diff is 1e-5; at a flip it is 0.13 and up
MIN_PAGE = 5.0  # drops part 1's 2.4s intro fragment, leaving 21 pages in every part

SAMPLES = 6
TRIM_LEAD, TRIM_TAIL = 0.20, 0.12


def run(cmd):
    subprocess.run(cmd, check=True)


def _frames(vf, out, fps):
    if not os.path.isdir(out) or not os.listdir(out):
        os.makedirs(out, exist_ok=True)
        run(["ffmpeg", "-v", "error", "-i", VIDEO, "-vf", f"fps={fps},{vf}",
             f"{out}/f%05d.png", "-y"])
    return sorted(glob.glob(f"{out}/*.png"))


def parts():
    """(start, end) of each guitar part, read off the lit digit in the badge."""
    x, y, w, h = BADGE
    fs = _frames(f"crop={w}:{h}:{x}:{y}", f"{WORK}/gt", 2)
    A = np.stack([np.asarray(Image.open(f).convert("L")) for f in fs]).astype(np.float32)
    third = w // 3
    lit = np.argmax(np.stack([
        np.percentile(A[:, :, i * third:(i + 1) * third], 97, axis=(1, 2))
        for i in range(3)]), axis=0)
    out, cur, st = [], lit[0], 0
    for i in range(1, len(lit)):
        if lit[i] != cur:
            out.append((st / 2, i / 2))
            cur, st = lit[i], i
    out.append((st / 2, len(lit) / 2))
    return [(a, b) for a, b in out if b - a > 10]


def pages():
    """(start, end, part index) for every system shown, in order."""
    fs = _frames(f"crop=1920:{PANEL_H}:0:{PANEL_Y},scale={SCAN_W}:{SCAN_H}",
                 f"{WORK}/scan", SCAN_FPS)
    M = np.stack([np.asarray(Image.open(f).convert("L")) < DARK for f in fs])
    n = len(M)
    diff = np.array([(M[i] != M[i - 1]).mean() for i in range(1, n)])

    ev = [i for i in range(1, n) if diff[i - 1] > FLIP]
    grp = []
    for i in ev:
        if grp and i - grp[-1][-1] <= 4:  # a flip can straddle a few scan frames
            grp[-1].append(i)
        else:
            grp.append([i])
    starts = [0] + [g[-1] + 1 for g in grp]
    stops = [g[0] - 1 for g in grp] + [n - 1]

    ps = parts()
    out = []
    for a, b in zip(starts, stops):
        t0, t1 = a / SCAN_FPS, b / SCAN_FPS
        if t1 - t0 < MIN_PAGE:
            continue
        mid = (t0 + t1) / 2
        p = next((i for i, (x, y) in enumerate(ps) if x <= mid < y), None)
        if p is not None:
            out.append((t0, t1, p))
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
    per = [sum(1 for _, _, p in ps if p == k) for k in range(3)]
    print(f"{len(ps)} pages  ->  Gt.1 {per[0]}, Gt.2 {per[1]}, Gt.3 {per[2]}")
    os.makedirs(f"{WORK}/pages", exist_ok=True)
    os.makedirs(f"{WORK}/png", exist_ok=True)
    which = [int(x) for x in sys.argv[1:]] or range(len(ps))
    for i in which:
        t0, t1, p = ps[i]
        g = render(t0, t1, i)
        np.save(f"{WORK}/pages/{i:03d}.npy", g)
        Image.fromarray(g).save(f"{WORK}/png/{i:03d}.png")
        print(f"  {i:03d}  Gt.{p + 1}  {t0:6.1f}-{t1:6.1f}s"
              f"  ink={(g.max(axis=2) < DARK).mean():.4f}", flush=True)
