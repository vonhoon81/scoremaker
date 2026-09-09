#!/usr/bin/env python3
"""Register the extracted pages into one panorama of the score, then cut it into bars.

The video flips through the score once, straight through, roughly every 5.4s, advancing
about four bars of the seven a page shows. So consecutive pages overlap heavily and the
panorama comes together from consecutive-pair shifts alone -- there is no segno, coda or
practice loop here, unlike the Pretender chart this shares its machinery with.

Pages are still registered against the panorama rather than only against their
predecessor, so offsets cannot drift, and the panorama takes the per-pixel median of
every page covering a column. That last part matters: each stretch of score is seen by
two or three pages, and the guitar-neck residue that survives the extraction sits
somewhere different on each of them, so the median clears it.
"""

import glob
import json
import os

import numpy as np

import extract

WORK = extract.WORK
W = extract.W

DARK = 170
STAFF_TOP = int(round(extract.STAFF_Y0))
STAFF_BOT = int(round(extract.STAFF_Y0 + (extract.STAFF_N - 1) * extract.STAFF_SPACING))
SPAN_SLACK = 3  # how far a barline's run may fall short of the staff at either end
BRIDGE = 2  # px of vertical gap closed before measuring a run: subtracting the
            # background nicks a barline wherever it crosses a staff line
MERGE = 20  # px; repeat signs and double bars are several lines and dots together
MIN_BAR = 120  # px. A section's closing double bar and the next system's opening
               # barline are ~57px apart, with the new system's TAB clef between them.
               # That reads as a bar of its own, so anything this short is merged into
               # the bar that follows it -- dropping it would lose the clef.
INSET = 12  # px of each page's own edges left out of the composite: a glyph the page
            # edge cut in half would otherwise outvote the pages that show it whole
# Below the staff the panel is dark behind the ink on every page, so the median is the
# right combiner. Above it the chord symbols are printed straight over the video, and a
# given symbol lands on a different part of the guitar in each page that shows it -- so
# there the darkest observations are the ones that had contrast to begin with, and a low
# percentile recovers labels the median leaves grey. It costs a little more neck residue.
LABEL_SPLIT = 150
PCTL_LABELS = 20
NUM_LEFT = 10  # a bar number straddles its own barline (-7..+9), so bars are cut this
NUM_HALF = 10  # far left of it, or every bar would carry half of the *next* bar's number
END_PAD = 26  # room for the closing double bar on the very last bar

MATCH_Y0 = 150  # rows above this are the chord band, printed over live video and far too
                # noisy to register on; keep them out
MIN_OVERLAP = 200  # px of shared strip needed before an offset can be scored
JUMP = 0.08  # dilated disagreement above which two pages do not overlap at all;
             # ordinary flips come in under 0.05, the jumps at 0.14 and up
COARSE = 6  # candidate offsets kept from the 1D profile pass


def load_pages():
    fs = sorted(glob.glob(f"{WORK}/pages/*.npy"))
    if not fs:
        raise SystemExit("no extracted pages; run extract.py first")
    return [np.load(f) for f in fs]


def match_rows(h):
    """Rows used for registration: below the neck residue, minus the redrawn staff."""
    keep = np.ones(h, bool)
    keep[:MATCH_Y0] = False
    for y in extract.staff_rows():
        keep[int(round(y)):int(round(y)) + 2] = False
    return keep


def dilate(m):
    d = m.copy()
    d[:, 1:] |= m[:, :-1]
    d[:, :-1] |= m[:, 1:]
    d[1:] |= m[:-1]
    d[:-1] |= m[1:]
    return d


def disagree(a, b, da, db):
    """Ink in either mask that the other does not cover even after dilation."""
    n = a.sum() + b.sum()
    if not n:
        return 1.0
    return float((a & ~db).sum() + (b & ~da).sum()) / n


def slide(a, b, da, db, offsets):
    """(offset, score) minimising disagreement with b placed `offset` right of a."""
    best = (None, 1.0)
    for o in offsets:
        lo, hi = max(0, o), min(a.shape[1], o + b.shape[1])
        if hi - lo < MIN_OVERLAP:
            continue
        u, v = slice(lo, hi), slice(lo - o, hi - o)
        d = disagree(a[:, u], b[:, v], da[:, u], db[:, v])
        if d < best[1]:
            best = (o, d)
    return best


def pair(ma, mb, da, db):
    return slide(ma, mb, da, db, range(ma.shape[1] - MIN_OVERLAP))


def runs(pairs):
    """Page indices grouped into stretches the video flipped through without jumping."""
    out = [[0]]
    for k, (_, sc) in enumerate(pairs):
        (out[-1] if sc <= JUMP else out.append([]) or out[-1]).append(k + 1)
    return out


def coarse(pa, pb):
    """Best COARSE offsets by normalised correlation of column ink profiles."""
    scores = []
    for o in range(-len(pb) + MIN_OVERLAP, len(pa) - MIN_OVERLAP):
        lo, hi = max(0, o), min(len(pa), o + len(pb))
        if hi - lo < MIN_OVERLAP:
            continue
        u, v = pa[lo:hi], pb[lo - o:hi - o]
        n = np.linalg.norm(u) * np.linalg.norm(v)
        if n > 0:
            scores.append((-float(u @ v) / n, o))
    scores.sort()
    return [o for _, o in scores[:COARSE]]


def offsets(pages):
    """Absolute panorama offset of every page, plus registration diagnostics."""
    keep = match_rows(pages[0].shape[0])
    masks = [(a < DARK)[keep] for a in pages]
    dils = [dilate(m) for m in masks]

    pairs = [pair(masks[k], masks[k + 1], dils[k], dils[k + 1])
             for k in range(len(pages) - 1)]
    groups = runs(pairs)

    offs = [None] * len(pages)
    pano = cover = None
    diag = []
    for g in groups:
        local = [0]
        for k in g[1:]:
            local.append(local[-1] + pairs[k - 1][0])
        h, span = masks[0].shape[0], local[-1] + W
        run_mask = np.zeros((h, span), bool)
        for k, o in zip(g, local):
            fresh = run_mask[:, o:o + W] | masks[k]
            run_mask[:, o:o + W] = fresh

        if pano is None:
            base, score = 0, 0.0
            pano = np.zeros((h, span), bool)
            cover = np.zeros(span, bool)
        else:
            rd = dilate(run_mask)
            cands = coarse(pano.sum(axis=0).astype(np.float64),
                           run_mask.sum(axis=0).astype(np.float64))
            fine = [o for c in cands for o in range(c - 3, c + 4)]
            base, score = slide(pano, run_mask, dilate(pano), rd, fine)
            if base is None:
                base, score = pano.shape[1], 1.0
            need = max(pano.shape[1], base + span)
            if need > pano.shape[1]:
                pano = np.pad(pano, ((0, 0), (0, need - pano.shape[1])))
                cover = np.pad(cover, (0, need - len(cover)))
        for k, o in zip(g, local):
            offs[k] = base + o
        seg = slice(max(0, base), base + span)
        fresh = ~cover[seg]
        pano[:, seg] = np.where(fresh, run_mask, pano[:, seg])
        cover[seg] = True
        diag.append((g[0], g[-1], base, round(score, 4), span))

    shift = -min(offs)
    return [o + shift for o in offs], diag


def panorama():
    """The whole score as one paper-white greyscale strip."""
    pages = load_pages()
    offs, diag = offsets(pages)
    h = pages[0].shape[0]
    width = max(offs) + W

    # Composite over intervals on which the set of contributing pages is constant --
    # a block that some page only half covers must not have the uncovered half voting.
    edges = sorted({0, width} | {o + d for o in offs for d in (INSET, W - INSET)}
                   | {o + d for o in offs for d in (0, W)})
    edges = [e for e in edges if 0 <= e <= width]
    out = np.full((h, width), 255, np.uint8)
    for x0, x1 in zip(edges, edges[1:]):
        if x1 <= x0:
            continue
        inner = [(a, o) for a, o in zip(pages, offs)
                 if o + INSET <= x0 and x1 <= o + W - INSET]
        pool = inner or [(a, o) for a, o in zip(pages, offs) if o <= x0 and x1 <= o + W]
        if not pool:
            continue
        stack = np.stack([a[:, x0 - o:x1 - o] for a, o in pool])
        top = np.percentile(stack[:, :LABEL_SPLIT], PCTL_LABELS, axis=0)
        bot = np.median(stack[:, LABEL_SPLIT:], axis=0)
        out[:LABEL_SPLIT, x0:x1] = top.astype(np.uint8)
        out[LABEL_SPLIT:, x0:x1] = bot.astype(np.uint8)
    return out, offs, diag


def barline_cols(a):
    """Every column whose ink spans the tab staff, ungrouped."""
    dark = a < DARK
    for _ in range(BRIDGE):  # close 1-2px nicks so a barline reads as one run
        dark[1:-1] |= dark[:-2] & dark[2:]
    pad = np.zeros((1, a.shape[1]), np.int8)
    d = np.diff(np.concatenate([pad, dark.view(np.int8), pad]), axis=0)
    hit = []
    for x in range(a.shape[1]):
        s, e = np.flatnonzero(d[:, x] == 1), np.flatnonzero(d[:, x] == -1)
        if any(u <= STAFF_TOP + SPAN_SLACK and v >= STAFF_BOT - SPAN_SLACK
               for u, v in zip(s, e)):
            hit.append(x)
    return hit


def barlines(a):
    """Left x of every barline, multi-line signs merged into one."""
    hit = barline_cols(a)
    if not hit:
        return []
    out, prev = [hit[0]], hit[0]
    for x in hit[1:]:
        if x - prev > MERGE:
            out.append(x)
        prev = x
    return out


def load_panorama():
    path = f"{WORK}/pano.npy"
    if not os.path.exists(path):
        pano, offs, _ = panorama()
        np.save(path, pano)
        json.dump({"offsets": offs}, open(f"{WORK}/pano.json", "w"))
    return np.load(path)


def bars():
    """(x0, x1) of every bar of the score, in panorama coordinates."""
    pano = load_panorama()
    bl = barlines(pano)
    out = []
    for x0, x1 in zip(bl, bl[1:]):
        if out and out[-1][1] - out[-1][0] < MIN_BAR:
            out[-1] = (out[-1][0], x1)  # merge the inter-system gap into this bar
        else:
            out.append((x0, x1))
    return pano, out


def label_spans(pano, band=(4, 96), gap=16, least=8):
    """Extent of every mark drawn above the staff: text, brackets, repeat counts.

    Re-flowing cuts the score at bar boundaries, and anything horizontal that reaches
    across one gets sliced -- "Da Coda" came out as "Da Cod" with a stray "a" opening
    the next system. make_pdf uses these to keep line breaks out of a mark. Columns are
    grouped with a tolerance because a text label has gaps between its letters.
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
    """Bars cut so each carries its own number, the last one its closing double bar."""
    pano, bs = bars()
    out = [(max(0, x0 - NUM_LEFT), x1 - NUM_LEFT) for x0, x1 in bs]
    out[0] = (0, out[0][1])  # the first bar keeps the "s.guit." label, clef and 4/4
    x0, x1 = out[-1]
    out[-1] = (x0, min(pano.shape[1], x1 + NUM_LEFT + END_PAD))
    return pano, out


def close_extra(pano, x1):
    """Px past a bar's right edge that hold the barline closing its system."""
    x = x1 + NUM_LEFT
    hit = barline_cols(pano[:, x:min(pano.shape[1], x + 26)])
    return (hit[-1] + 3 if hit else 2) + NUM_LEFT


def bar_image(pano, bar, close=False):
    """The bar, optionally extended to carry the barline that closes its system."""
    x0, x1 = bar
    if not close:
        return pano[:, x0:x1]
    end = min(pano.shape[1], x1 + close_extra(pano, x1))
    out = pano[:, x0:end].copy()
    # that barline is straddled by the *next* bar's number; drop it, keep the line
    n = x1 + NUM_LEFT - NUM_HALF - x0
    out[:STAFF_TOP - 6, n:] = 255
    return out


def strips():
    pano, bs = score_bars()
    return [bar_image(pano, b) for b in bs]


if __name__ == "__main__":
    pano, offs, diag = panorama()
    np.save(f"{WORK}/pano.npy", pano)
    json.dump({"offsets": offs}, open(f"{WORK}/pano.json", "w"))
    print("run (pages)   placed at   score   width")
    for a, b, base, score, span in diag:
        print(f"  {a:2d}-{b:<2d}      {base:8d} {score:8.4f} {span:7d}")
    print(f"\npanorama {pano.shape[1]}px")
    print("offsets", offs)
    bl = barlines(pano)
    bs = [(x0, x1) for x0, x1 in zip(bl, bl[1:]) if x1 - x0 >= MIN_BAR]
    w = [x1 - x0 for x0, x1 in bs]
    print(f"{len(bs)} bars, widths {min(w)}-{max(w)}px")
