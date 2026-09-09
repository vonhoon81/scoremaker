#!/usr/bin/env python3
"""Cut the extracted windows into bars and put the score back together.

Each window is left-aligned to a barline and then filled with as many bars as the
1920px panel holds, so the bar at the right edge is always chopped off mid-way -- and
that chopped bar is exactly the one the next window opens with. So the score is simply
every *complete* bar (a bar with a detected barline on both sides) of every window,
concatenated in order: the partial bar at each window's right edge is the duplicate and
drops out for free.

The one exception is the final window, which the renderer right-aligns onto the last
bar instead of left-aligning onto a barline, so its leading bar is chopped instead. That
bar is complete on the window before, so the rule still holds.

Bars are located by barline, not by reading the red bar numbers, but the numbers give a
free check on the result: the count of complete bars has to come out at BARS.
"""

import glob
import os

import numpy as np
from PIL import Image

import extract

WORK = extract.WORK
BARS = 137  # the last bar number shown in the video

SKIP_TOP = 3  # the dark frame edge just above the panel
STAFF_Y0, STAFF_Y1 = 60, 161  # the six tab lines, at rows 60/80/100/119/139/158
DARK = 170  # barlines and engraving are black; the staff lines themselves are grey 234
BARLINE_COVER = 95  # a barline is dark over nearly the whole staff height
MERGE = 4  # px; the opening time signature reads as one thick "barline"

INK = 215  # anything below this counts as content when finding the rows to keep
PAD_TOP, PAD_BOT = 6, 6
END_BARLINE_PAD = 18  # room for the thick closing barline on the very last bar


def barlines(a):
    """Left x of every barline in a window."""
    dark = a.max(axis=2) < DARK
    cols = np.flatnonzero(dark[STAFF_Y0:STAFF_Y1].sum(axis=0) >= BARLINE_COVER)
    out, prev = [int(cols[0])], int(cols[0])
    for c in cols[1:]:
        if c - prev > MERGE:  # gap from the previous column, not from the run's start:
            out.append(int(c))  # the opening time signature is a 13px-wide dark band
        prev = int(c)
    return out


def load_pages():
    fs = sorted(glob.glob(f"{WORK}/pages/*.npy"))
    if not fs:
        raise SystemExit("no extracted pages; run extract.py first")
    return [np.load(f) for f in fs]


def rows_to_keep(pages):
    """One row window shared by every bar, so the staves line up when concatenated."""
    lo, hi = 10**9, 0
    for a in pages:
        rows = np.flatnonzero((a[SKIP_TOP:].max(axis=2) < INK).any(axis=1)) + SKIP_TOP
        lo, hi = min(lo, int(rows[0])), max(hi, int(rows[-1]))
    return max(SKIP_TOP, lo - PAD_TOP), min(pages[0].shape[0], hi + 1 + PAD_BOT)


def bars():
    """The whole score as a list of (page index, x0, x1), one entry per bar."""
    pages = load_pages()
    r0, r1 = rows_to_keep(pages)
    out = []
    for i, a in enumerate(pages):
        bl = barlines(a)
        out += [(i, x0, x1) for x0, x1 in zip(bl, bl[1:])]
    # the score's last barline is the thick closing one; keep it with the final bar
    i, x0, x1 = out[-1]
    out[-1] = (i, x0, min(pages[i].shape[1], x1 + END_BARLINE_PAD))
    return pages, (r0, r1), out


def strips():
    """Bar images, all the same height, in score order."""
    pages, (r0, r1), bs = bars()
    out = [pages[i][r0:r1, x0:x1] for i, x0, x1 in bs]
    return out


if __name__ == "__main__":
    pages, (r0, r1), bs = bars()
    print(f"{len(pages)} windows, rows {r0}-{r1} ({r1 - r0}px)")
    widths = [x1 - x0 for _, x0, x1 in bs]
    print(f"{len(bs)} bars (expected {BARS})"
          f"  widths {min(widths)}-{max(widths)}px, median {int(np.median(widths))}")
    if len(bs) != BARS:
        print("  !! bar count does not match the video's last bar number")
    n = 0
    for i, a in enumerate(pages):
        k = sum(1 for p, _, _ in bs if p == i)
        print(f"  window {i:02d}: bars {n + 1}-{n + k}")
        n += k
