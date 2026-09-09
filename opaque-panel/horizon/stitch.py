#!/usr/bin/env python3
"""Cut the extracted pages into bars and put the written score back together once.

Two things stop this from being a plain concatenation:

  * the page does not advance by a fixed number of bars. Bar widths here run from 149px
    to 559px depending on how much is in them, so a page holds three or four complete
    bars and the same bar is sometimes complete on two consecutive pages. The overlap
    has to be measured, not assumed.
  * the video plays a repeat, so bars 26-56 are shown twice: once by pages 0-19 and
    again by pages 20 onward, which then carry on to the end of the song.

Both come out of the same measurement. Consecutive pages are overlaid: the only shifts
worth trying are the ones that line one page's barlines up with the other's, which is at
most a few dozen candidates, and the right one is whichever minimises the pixel
difference over the overlapping strip. A normal page flip leaves a few hundred px of
overlap that matches almost exactly; a jump back to a repeat sign leaves nothing that
matches at any shift, and that is what splits the passes. Bars of page k+1 whose shifted
position lands on a complete bar of page k are the ones already in hand.

The passes are merged by aligning the second against the first and keeping the first
pass's copies of the shared bars -- the panel sits 20px higher after the repeat, high
enough that the frame clips the tops of the bar numbers there.
"""

import glob
import json
import os

import numpy as np

import extract

WORK = extract.WORK
W = extract.W

DARK = 170
BARLINE_RUN = 300  # a barline spans notation staff to tab staff, ~360px; nothing else does
MERGE = 20  # px; double barlines and repeat signs are several lines and dots together
REPEAT_W = 12  # px. A repeat sign is a thick line, a thin one and two dots; a plain
               # section double bar comes to 8-9px and a single barline to 1-2px.
NUM_HALF = 8  # half a bar number's width; it is centred on its barline
NUM_LEFT = 10  # a bar number straddles its own barline, so bars are cut this far left
              # of it -- otherwise every bar carries the *next* bar's number, half of it

NUM_BAND = extract.ABOVE - 3  # rows above the staff: bar numbers, clipped on pass 2
MIN_OVERLAP = 120  # px of shared strip needed before a shift can be scored
SNAP = 6  # px tolerance when deciding that two barlines are the same one
PASS_BREAK = 8.0  # mean abs grey difference above which no shift explains the junction


def vertical_runs(a, x0=0, x1=None):
    """Longest run of dark pixels down each column."""
    dark = a.min(axis=2)[:, x0:x1] < DARK
    pad = np.zeros((1, dark.shape[1]), np.int8)
    d = np.diff(np.concatenate([pad, dark.view(np.int8), pad]), axis=0)
    runs = np.zeros(dark.shape[1], np.int32)
    for x in range(dark.shape[1]):
        s, e = np.flatnonzero(d[:, x] == 1), np.flatnonzero(d[:, x] == -1)
        runs[x] = (e - s).max() if len(s) else 0
    return runs


def barlines(a):
    """Left x of every barline on a page, double barlines merged into one."""
    cols = np.flatnonzero(vertical_runs(a) > BARLINE_RUN)
    if not len(cols):
        return []
    out, prev = [int(cols[0])], int(cols[0])
    for c in cols[1:]:
        if c - prev > MERGE:
            out.append(int(c))
        prev = int(c)
    return out


def page_tops():
    """Each page's detected staff row. Pages below extract.ABOVE were clipped by the
    frame, losing the upper half of their bar numbers."""
    with open(f"{WORK}/pages.json") as fh:
        return [t for _, _, t in json.load(fh)]


def load_pages():
    fs = sorted(glob.glob(f"{WORK}/pages/*.npy"))
    if not fs:
        raise SystemExit("no extracted pages; run extract.py first")
    return [np.load(f) for f in fs]


def grey(a):
    """Greyscale, minus the bar-number band the second pass has clipped."""
    return a[NUM_BAND:].astype(np.float32).mean(axis=2)


def overlay(ga, gb, s):
    """Mean abs difference where page b, shifted right by s, covers page a."""
    n = ga.shape[1] - s
    if n < MIN_OVERLAP:
        return None
    return float(np.abs(ga[:, s:] - gb[:, :n]).mean())


def junction(ga, ba, gb, bb):
    """(shift, overlap in bars, score) between two consecutive pages."""
    cands = {a - b for a in ba for b in bb if a - b > 0}
    best = (None, 1e9)
    for s in sorted(cands):
        d = overlay(ga, gb, s)
        if d is not None and d < best[1]:
            best = (s, d)
    s, score = best
    if s is None:
        return None, None, 1e9
    # page b's bars that land on a complete bar of page a are already in hand
    starts = set(ba[:-1])
    o = sum(any(abs(a - (s + b)) <= SNAP for a in starts) for b in bb[:-1])
    return s, o, score


def assemble():
    """(pages, bars, per-pass bar lists, junction diagnostics)."""
    pages = load_pages()
    bls = [barlines(a) for a in pages]
    greys = [grey(a) for a in pages]

    js = [junction(greys[k], bls[k], greys[k + 1], bls[k + 1])
          for k in range(len(pages) - 1)]

    groups, cur = [], [0]
    for k, (s, o, sc) in enumerate(js):
        if sc > PASS_BREAK:
            groups.append(cur)
            cur = [k + 1]
        else:
            cur.append(k + 1)
    groups.append(cur)

    seqs = []
    for g in groups:
        seq = []
        for n, k in enumerate(g):
            bars = [(k, max(0, x0 - NUM_LEFT), x1 - NUM_LEFT)
                    for x0, x1 in zip(bls[k], bls[k][1:])]
            seq += bars[0 if n == 0 else js[g[n - 1]][1]:]
        seqs.append(seq)

    bars = list(seqs[0])
    for seq in seqs[1:]:
        d = align(pages, bars, seq)
        bars += seq[len(bars) - d:]
    return pages, bars, seqs, js


def align(pages, have, seq):
    """Index in `have` where `seq` starts, matched on a run of three bar images."""
    def img(b):
        k, x0, x1 = b
        return grey(pages[k][:, x0:x1])

    def cmp(u, v):
        n = min(u.shape[1], v.shape[1])
        if abs(u.shape[1] - v.shape[1]) > 8:
            return 1e9
        return float(np.abs(u[:, :n] - v[:, :n]).mean())

    first = [img(b) for b in seq[:3]]
    best = (0, 1e9)
    for d in range(len(have) - len(first) + 1):
        m = float(np.mean([cmp(f, img(have[d + i])) for i, f in enumerate(first)]))
        if m < best[1]:
            best = (d, m)
    return best[0]


def close_extra(a, x1):
    """Px past a bar's right edge that hold its closing barline, double bar or end bar."""
    x = x1 + NUM_LEFT  # where the barline itself sits
    w = min(a.shape[1], x + 26)
    runs = vertical_runs(a, x, w)
    hit = np.flatnonzero(runs > BARLINE_RUN)
    end = x + int(hit[-1]) + 2 if len(hit) else x + 2
    return min(end, a.shape[1]) - x1


def repeat_bars(pages, bars):
    """Indices of bars opened by a repeat sign rather than an ordinary barline.

    make_pdf forces a system break at each of these, so that a repeat opens a system
    and the bar before its closing sign ends one, instead of the signs being buried
    mid-line where they are easy to read straight past.
    """
    out = []
    for i, (k, x0, x1) in enumerate(bars):
        a = pages[k]
        x = x0 + NUM_LEFT
        runs = vertical_runs(a, max(0, x - 12), min(a.shape[1], x + 14))
        hit = np.flatnonzero(runs > BARLINE_RUN)
        if len(hit) and hit[-1] - hit[0] + 1 >= REPEAT_W:
            out.append(i)
    return out


def bar_image(pages, bar, close=False):
    """The bar, optionally extended to carry the barline that closes its system."""
    k, x0, x1 = bar
    a = pages[k]
    if not close:
        return a[:, x0:x1]
    out = a[:, x0:x1 + close_extra(a, x1)].copy()
    # that barline is straddled by the *next* bar's number; drop it, keep the line
    n = x1 + NUM_LEFT - NUM_HALF - x0
    out[:extract.ABOVE - 5, n:] = out[0, 0]
    return out


def strips(closers=()):
    """Bar images in score order; `closers` are indices to give a closing barline."""
    pages, bars, _, _ = assemble()
    return [bar_image(pages, b, i in closers) for i, b in enumerate(bars)]


if __name__ == "__main__":
    pages, bars, seqs, js = assemble()
    print("junction   shift  overlap   score")
    for k, (s, o, sc) in enumerate(js):
        flag = "  <== new pass (repeat jump)" if sc > PASS_BREAK else ""
        print(f"  {k:3d}->{k + 1:<3d} {str(s):>6} {str(o):>7}  {sc:8.2f}{flag}")
    print()
    for i, s in enumerate(seqs):
        print(f"pass {i}: {len(s):3d} bars from pages {s[0][0]}-{s[-1][0]}")
    w = [x1 - x0 for _, x0, x1 in bars]
    print(f"\n{len(bars)} bars in the written score, widths {min(w)}-{max(w)}px")
    json.dump(bars, open(f"{WORK}/bars.json", "w"))
