#!/usr/bin/env python3
"""Locate each skill-level section in the stitched strip.

The video is one continuous score played six times over at rising skill levels, and the
label in the corner is the only thing that says where one level hands off to the next.
Each handover happens at a known time, so to place it in the score we need the bar being
played at that moment -- which the teal highlight marks directly.
"""

import glob
import os
import subprocess

import numpy as np
from PIL import Image

import extract

# When each level takes over: the cut back from a title card, or (for 반년차) the hard
# cut between takes. The corner label fades in a beat or two later.
SECTIONS = [
    ("1일차", 7.4),
    ("반년차", 19.0),
    ("1년차", 42.4),
    ("5년차", 75.3),
    ("10년차", 95.2),
    ("100년차 이상", 139.4),
]

TEAL_LEVEL = 6.0  # mean cyan-minus-red across the strip height
SETTLE = 0.3  # wait this long after a cut or a flip before trusting the highlight
PROBE = 0.6   # and then read it over this long, taking the median


def teal_left(t, span=PROBE, n=4):
    """Left edge of the highlighted bar at time t, in page coordinates. Read over a
    short window and take the median: right after a cut the overlay is still
    cross-fading and a single frame can land anywhere."""
    d = f"{extract.WORK}/teal"
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{d}/*.png"):
        os.remove(f)
    subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-t", f"{span:.3f}",
                    "-i", extract.VIDEO, "-vf",
                    f"fps={n / span:.3f},crop=1920:{extract.BAND_H}:0:{extract.BAND_Y}",
                    f"{d}/p%03d.png", "-y"], check=True)
    seen = []
    for f in sorted(glob.glob(f"{d}/*.png")):
        a = np.asarray(Image.open(f).convert("RGB")).astype(np.float32)
        prof = ((a[..., 1] + a[..., 2]) / 2 - a[..., 0]).mean(axis=0)
        on = np.flatnonzero(prof > TEAL_LEVEL)
        if len(on):
            seen.append(int(on[0]))
    return int(np.median(seen)) if seen else None


def _probe(pages, origin, t):
    """First moment at or after t that sits safely inside a page we actually kept.
    A level can begin during a title card or a page flip, when there is nothing to
    read; in that case the answer is the start of the page that comes out of it."""
    for i, (a, b) in enumerate(pages):
        if i not in origin or b < t + SETTLE:
            continue
        start = max(t, a) + SETTLE
        if start + PROBE <= b:
            return i, start
    return None, None


def locate(pages, origin, bars, snap=260):
    """(label, strip index, x in that strip) for each section, snapped to a bar line."""
    out = []
    for k, (label, t) in enumerate(SECTIONS):
        page, at = _probe(pages, origin, t)
        if page is None:
            print(f"  {label}: no page at {t}s -- skipped")
            continue
        strip, x0 = origin[page]
        left = 0 if k == 0 else (teal_left(at) or 0)  # the first level opens the score
        x = x0 + left
        near = [b for b in bars.get(strip, []) if abs(b - x) <= snap]
        if near:
            x = min(near, key=lambda b: abs(b - x))
        out.append((label, strip, x))
        print(f"  {label:12s} t={t:5.1f}s  read@{at:6.2f}s  page {page:02d}"
              f"  strip {strip}  x={x}")
    return out
