#!/usr/bin/env python3
"""Join the extracted pages into one score strip, then cut it into bars.

Nothing here is per-song -- every constant is a property of the 하루베이스 template. See
extract.py for what changes between videos from the channel.

This chart needs the *least* machinery of the nine, not the most, and it took looking at
the pages to see that. probe.py and the Nanmonee pipeline it was copied from both assume
consecutive pages overlap and have to be registered; here they do not overlap at all.

Each page is a self-contained system: N complete bars justified to fill the staff's own
span, x = 94..1823. Nothing is shared with the page before or after it. Three things say
so, and the first two are what a registration pass gets wrong:

  * every consecutive pair registers at an offset of a full page width, with a
    disagreement of 0.25-0.5 -- there is no overlap for the matcher to lock onto, so it
    is fitting noise.
  * the ties line up. Page 007 ends on a tied note and page 008 opens with that note
    bracketed as a continuation, at the page's first bar rather than part way in.
  * a page's width per bar is ~432px whether it shows two bars or five, because it
    justifies whatever it has across the same staff.

So the join is a concatenation -- the cheapest of the four in the README's list -- and
the score strip is just the pages' staff spans laid end to end, in order.

The barline detector had to change too. The engraving is thin and the background
subtraction nicks a barline in two or three places, which a run-length test cannot
bridge; asking instead that a column be dark over 90% of the staff's rows finds every
one. That admits a note stem passing through the staff, so a detected line must also be
at least 2px wide -- real barlines measure 2-9px, a stem exactly 1.
"""

import glob
import json
import os

import numpy as np

import extract

WORK = extract.WORK
X0, X1 = extract.STAFF_X0, extract.STAFF_X1
PAGE_W = X1 - X0

DARK = 170
STAFF_TOP = int(round(extract.STAFF_Y0))
STAFF_BOT = int(round(extract.STAFF_Y0 + (extract.STAFF_N - 1) * extract.STAFF_SPACING))
SPAN = 0.90  # fraction of the staff's rows a barline's column must darken
MIN_W = 2  # px; a barline is 2-9px wide, a note stem crossing the staff exactly 1
MERGE = 20  # px; a double bar is several lines and the gap between them
MIN_BAR = 150  # px. Bars run 334-865px, so anything shorter is the tail of a double
               # bar and is merged into the bar that follows it.
BARLINE_W = 3  # px of the barline drawn into a page junction the source left open
NEAR = 10  # px either side of a junction searched for a barline before one is drawn

LABEL_SPLIT = 55  # rows above this are the chord band, printed over live video
NUM_LEFT = 0  # this chart prints no bar numbers, so nothing straddles a barline and
NUM_HALF = 0  # bars are cut on the line itself
END_PAD = 26  # room for the closing double bar on the very last bar


def load_pages():
    fs = sorted(glob.glob(f"{WORK}/pages/*.npy"))
    if not fs:
        raise SystemExit("no extracted pages; run extract.py first")
    return [np.load(f) for f in fs]


def score():
    """Every page's staff span, end to end: the whole chart as one strip.

    The source never draws a barline at a page's left edge and only sometimes at its
    right, so about half the junctions would come out with no bar boundary at all. One
    is drawn in wherever the source left the junction open -- and *only* there, because
    where the source did close the page the two lines would otherwise sit 7px apart and
    read as a double bar the music does not have.
    """
    pages = load_pages()
    strip = np.hstack([a[:, X0:X1] for a in pages])
    top = int(extract.staff_rows()[0])
    bot = int(extract.staff_rows()[-1]) + 2
    drawn = 0
    for j in range(0, strip.shape[1], PAGE_W):
        lo, hi = max(0, j - NEAR), min(strip.shape[1], j + NEAR)
        if not barline_cols(strip[:, lo:hi]):
            strip[top:bot, j:j + BARLINE_W] = 0
            drawn += 1
    return strip, len(pages), drawn


def load_score():
    path = f"{WORK}/score.npy"
    if not os.path.exists(path):
        s, n, _ = score()
        np.save(path, s)
        json.dump({"pages": n, "page_w": PAGE_W}, open(f"{WORK}/score.json", "w"))
    return np.load(path)


def barline_cols(a):
    """Every column dark over SPAN of the staff's rows -- ungrouped, unfiltered."""
    dark = a[STAFF_TOP:STAFF_BOT + 1] < DARK
    return list(np.flatnonzero(dark.mean(axis=0) >= SPAN))


def barlines(a):
    """Left x of every barline, multi-line signs merged, note stems rejected."""
    hit = barline_cols(a)
    if not hit:
        return []
    groups, prev = [[hit[0]]], hit[0]
    for x in hit[1:]:
        (groups[-1] if x - prev <= MERGE else groups.append([]) or groups[-1]).append(x)
        prev = x
    return [int(g[0]) for g in groups if g[-1] - g[0] + 1 >= MIN_W]


def bars():
    """(x0, x1) of every bar of the score, in score-strip coordinates."""
    s = load_score()
    bl = barlines(s)
    out = []
    for x0, x1 in zip(bl, bl[1:]):
        if out and out[-1][1] - out[-1][0] < MIN_BAR:
            out[-1] = (out[-1][0], x1)  # merge a double bar's tail into this bar
        else:
            out.append((int(x0), int(x1)))
    return s, out


def label_spans(pano, band=(4, 50), gap=16, least=8):
    """Extent of every mark drawn above the staff, so make_pdf can avoid breaking one.

    Columns are grouped with a tolerance because a chord symbol has gaps between its
    letters, and "F/A" would otherwise read as three separate marks.
    """
    cols = np.flatnonzero((pano[band[0]:band[1]] < DARK + 45).any(axis=0))
    runs = []
    for c in cols:
        if runs and c - runs[-1][1] <= gap:
            runs[-1][1] = int(c)
        else:
            runs.append([int(c), int(c)])
    return [(a, b) for a, b in runs if b - a >= least]


def score_bars():
    """Bars, the last one widened to carry the closing double bar."""
    s, bs = bars()
    out = list(bs)
    out[0] = (0, out[0][1])  # the first bar keeps whatever opens the score
    x0, x1 = out[-1]
    out[-1] = (x0, min(s.shape[1], x1 + END_PAD))
    return s, out


def close_extra(pano, x1):
    """Px past a bar's right edge that hold the barline closing its system."""
    hit = barline_cols(pano[:, x1:min(pano.shape[1], x1 + 26)])
    return hit[-1] + 3 if hit else 2


def bar_image(pano, bar, close=False):
    """The bar, optionally extended to carry the barline that closes its system."""
    x0, x1 = bar
    if not close:
        return pano[:, x0:x1]
    end = min(pano.shape[1], x1 + close_extra(pano, x1))
    out = pano[:, x0:end].copy()
    out[:STAFF_TOP - 6, x1 - x0:] = 255  # drop a chord symbol leaning into the padding
    return out


def strips():
    pano, bs = score_bars()
    return [bar_image(pano, b) for b in bs]


if __name__ == "__main__":
    s, n, drawn = score()
    np.save(f"{WORK}/score.npy", s)
    json.dump({"pages": n, "page_w": PAGE_W}, open(f"{WORK}/score.json", "w"))
    print(f"{n} pages concatenated -> score strip {s.shape[1]}px "
          f"({drawn} junctions the source left open, barline drawn in)\n")

    _, bs = bars()
    w = [b - a for a, b in bs]
    print(f"{len(bs)} bars, widths {min(w)}-{max(w)}px\n")

    # No bar numbers are printed on this chart, so the README's read-the-numbers check
    # is not available. This is the cross-check that replaces it: the display advances
    # when the bars it is showing have been played, so a page's time on screen should be
    # proportional to how many bars it shows. A page whose seconds-per-bar is out of
    # line with the rest has either had a flip missed or been split spuriously.
    ps = json.load(open(f"{WORK}/pages.json"))
    print("page   on screen   bars   s/bar")
    per = []
    for i, (t0, t1) in enumerate(ps):
        lo, hi = i * PAGE_W, (i + 1) * PAGE_W
        # by midpoint: a page's closing barline can sit a few px short of its right
        # edge, and counting on a bar's left edge would lend that bar to the wrong page
        k = len([1 for a, b in bs if lo <= (a + b) // 2 < hi])
        per.append((t1 - t0) / k if k else 0.0)
        flag = ""
        if i and k and abs(per[-1] - np.median(per)) > 0.35 * np.median(per):
            flag = "   <-- out of line"
        print(f" {i:03d}   {t0:6.1f}-{t1:6.1f}   {k:2d}   {per[-1]:5.2f}{flag}")
    med = float(np.median([p for p in per if p]))
    total = sum(t1 - t0 for t0, t1 in ps)
    print(f"\nmedian {med:.2f}s per bar -> {60 / med * 4:.0f} bpm in 4/4")
    print(f"{total:.0f}s of score on screen / {med:.2f} = {total / med:.0f} bars played "
          f"against {len(bs)} written")
