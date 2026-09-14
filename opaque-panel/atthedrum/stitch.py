#!/usr/bin/env python3
"""Join the extracted pages into one score strip, cut it into bars, and number them.

Pages do not overlap: every one is four self-contained bars filling the staff, opened by
its own thick barline and closed by one at x=1893. The test is the same one that settled
`harrubass`, and it gave the same answer -- registering consecutive pairs scores 0.35-0.48
at scattered offsets (904, 840, 892, 820, 1400, 828) with no clean population of good
matches, which is what "nothing to match on" looks like. So the join is np.hstack.

Two things this chart has that no earlier one did:

  * **a boxed section letter** sits to the *left* of the staff, above it. Cutting at the
    staff's own start would drop the song's structure, so pages are cut from MARK_X0 and
    the five staff lines are bridged across the resulting lead-in. A page without a
    letter just gets ~74px of empty staff before its first bar.
  * **a multi-bar rest.** Page 013 is a 3-bar rest plus one played bar -- two bar *boxes*
    holding four bars of music. The README's rule is to print no bar numbers at all when
    that happens, because boxes and numbers disagree. They can be reconciled instead:
    seconds-per-bar is a constant of the video, so a page's *music* bars are its time on
    screen over that constant, and a page with fewer boxes than music bars has the
    difference sitting in its multi-rest. That is what bar_numbers() does, and the table
    __main__ prints is the evidence for it.
"""

import glob
import json
import os

import numpy as np

import extract

WORK = extract.WORK
X0, X1 = extract.MARK_X0, extract.STAFF_X1
PAGE_W = X1 - X0
LEAD = extract.STAFF_X0 - X0  # blank staff before a page's first barline

DARK = 170
STAFF_TOP = int(round(extract.STAFF_Y0))
STAFF_BOT = int(round(extract.STAFF_Y0 + (extract.STAFF_N - 1) * extract.STAFF_SPACING))
STAFF_GREY = 90  # the source draws the staff near-black; bridged rows match it
SPAN = 0.90  # fraction of the staff's rows a barline's column must darken
MIN_W = 2  # px; a barline is 2-3px, a stem crossing the staff exactly 1
MERGE = 20
MIN_BAR = 150

LABEL_SPLIT = STAFF_TOP - 6  # rows above the staff: beams, and the section letter
END_PAD = 26
MULTIREST = 0.55  # fraction of a bar's width a horizontal dark run must span for the
                  # bar to be a multi-bar rest rather than a played bar


def load_pages():
    fs = sorted(glob.glob(f"{WORK}/pages/*.npy"))
    if not fs:
        raise SystemExit("no extracted pages; run extract.py first")
    return [np.load(f) for f in fs]


def score():
    """Every page's marker-and-staff span, end to end, staff bridged over the lead-in."""
    pages = load_pages()
    strip = np.hstack([a[:, X0:X1] for a in pages])
    for i in range(len(pages)):  # bridge the blank lead-in so the staff runs unbroken
        j = i * PAGE_W
        for y in extract.staff_rows():
            r = int(round(y))
            strip[r:r + 2, j:j + LEAD] = np.minimum(strip[r:r + 2, j:j + LEAD],
                                                    STAFF_GREY)
    return strip, len(pages)


def load_score():
    path = f"{WORK}/score.npy"
    if not os.path.exists(path):
        s, n = score()
        np.save(path, s)
        json.dump({"pages": n, "page_w": PAGE_W}, open(f"{WORK}/score.json", "w"))
    return np.load(path)


def barline_cols(a):
    dark = a[STAFF_TOP:STAFF_BOT + 1] < DARK
    return list(np.flatnonzero(dark.mean(axis=0) >= SPAN))


def barlines(a):
    hit = barline_cols(a)
    if not hit:
        return []
    groups, prev = [[hit[0]]], hit[0]
    for x in hit[1:]:
        (groups[-1] if x - prev <= MERGE else groups.append([]) or groups[-1]).append(x)
        prev = x
    return [int(g[0]) for g in groups if g[-1] - g[0] + 1 >= MIN_W]


def bars():
    s = load_score()
    bl = barlines(s)
    # The first page has no predecessor, so its opening bar has no barline to its left
    # and would be dropped. Every later page gets one from the page before it closing.
    if bl and bl[0] - LEAD >= MIN_BAR:
        bl = [LEAD] + bl
    out = []
    for x0, x1 in zip(bl, bl[1:]):
        if out and out[-1][1] - out[-1][0] < MIN_BAR:
            out[-1] = (out[-1][0], x1)
        else:
            out.append((int(x0), int(x1)))
    return s, out


def _between_staff_rows():
    """Rows inside the staff that are not one of the five drawn lines.

    The multi-rest test looks for a long horizontal run, and a staff line is exactly
    that -- testing every row marks all 55 bars as multi-rests.
    """
    keep = np.ones(STAFF_BOT - STAFF_TOP + 1, bool)
    for y in extract.staff_rows():
        r = int(round(y)) - STAFF_TOP
        keep[max(0, r - 1):r + 3] = False
    return keep


def is_multirest(s, bar):
    """A multi-bar rest is a thick horizontal bar across most of its box."""
    x0, x1 = bar
    band = (s[STAFF_TOP:STAFF_BOT + 1, x0:x1] < DARK)[_between_staff_rows()]
    best = 0
    for row in band:
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if len(a):
            best = max(best, int((b - a).max()))
    return best >= MULTIREST * (x1 - x0)


def bar_numbers():
    """Music-bar number each bar box starts on, multi-bar rests counted in full."""
    s, bs = bars()
    ps = json.load(open(f"{WORK}/pages.json"))
    per_page = [[i for i, (a, b) in enumerate(bs)
                 if k * PAGE_W <= (a + b) // 2 < (k + 1) * PAGE_W]
                for k in range(len(ps))]
    # seconds per bar, from the pages whose boxes are all played bars (the majority)
    spb = float(np.median([(t1 - t0) / len(ix)
                           for (t0, t1), ix in zip(ps, per_page) if ix]))
    span = [1] * len(bs)
    rows = []
    for (t0, t1), ix in zip(ps, per_page):
        if not ix:
            continue
        music = int(round((t1 - t0) / spb))
        extra = music - len(ix)
        rest = [i for i in ix if is_multirest(s, bs[i])]
        took = extra > 0 and bool(rest)
        if took:
            span[rest[0]] += extra  # the page's multi-rest absorbs the difference
        rows.append((t0, t1, len(ix), music, extra, len(rest), took))
    start, n = [], 1
    for sp in span:
        start.append(n)
        n += sp
    return s, bs, start, span, spb, rows, n - 1


def label_spans(pano, band=(4, LABEL_SPLIT), gap=16, least=8):
    """Extent of every mark above the staff, so make_pdf can avoid breaking one.

    Here that is mostly the beamed hi-hat groups, which reach across a whole bar, plus
    the boxed section letter -- both things a system break must not slice.
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
    """Bars, the last widened to carry the closing double bar."""
    s, bs = bars()
    out = list(bs)
    out[0] = (max(0, out[0][0] - LEAD), out[0][1])  # keep the opening clef
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
    return pano[:, x0:end]


if __name__ == "__main__":
    s, n = score()
    np.save(f"{WORK}/score.npy", s)
    json.dump({"pages": n, "page_w": PAGE_W}, open(f"{WORK}/score.json", "w"))
    print(f"{n} pages concatenated -> score strip {s.shape[1]}px\n")

    _, bs, start, span, spb, rows, total = bar_numbers()
    w = [b - a for a, b in bs]
    print(f"{len(bs)} bar boxes, widths {min(w)}-{max(w)}px -> {total} bars of music\n")
    print(f"{spb:.2f}s per bar -> {60 / spb * 4:.0f} bpm in 4/4\n")
    print("page   on screen    boxes  music  extra  multirests")
    for i, (t0, t1, nb, music, extra, nr, took) in enumerate(rows):
        flag = ("   <-- multi-bar rest absorbs the extra" if took
                else "   <-- boxes and timing disagree, no multi-rest to explain it"
                if extra else "")
        print(f" {i:03d}   {t0:6.1f}-{t1:6.1f}   {nb:3d}   {music:4d}   {extra:3d}"
              f"      {nr}{flag}")
    tot_t = sum(t1 - t0 for t0, t1, *_ in rows)
    print(f"\n{tot_t:.0f}s on screen / {spb:.2f} = {tot_t / spb:.0f} bars played "
          f"against {total} counted")
