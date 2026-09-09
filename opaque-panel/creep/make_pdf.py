#!/usr/bin/env python3
"""Cut the stitched strips into systems at bar lines and lay them out as an A4 score.

Each strip is one passage of the lesson, headed with the chapter (or chapters) it was
taught under. Where a chapter changes part-way along a strip the boundary is tagged
inline; the video has no playhead, so the tag is placed at the left edge of the window
that was on screen when the chapter started -- right to within the two bars a step moves.
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import stitch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)  # the repo root, for common.py and the source videos

import common  # noqa: E402

OUT = common.out_pdf("Creep 기타 악보.pdf")
TITLE = "Creep 기타 악보"
SUB1 = "Radiohead · ♩= 76 · Standard Tuning (E A D G B E)"
SUB2 = ("레슨 영상은 곡 전체를 연주하지 않고 반복되는 주요 구간만 다룹니다 — "
        "아래는 화면에 나온 구간을 그대로 이어붙인 것입니다.")
SOURCE = "출처: Nikola Gugoski — Radiohead - Creep (Guitar lesson with TAB)"

# the video's own chapter marks
CHAPTERS = [(0.0, 30.0, "Intro"), (30.0, 84.0, "Verse"),
            (84.0, 120.0, "Chorus 1"), (120.0, 206.0, "Chorus 2 & Bridge")]
PARTS = {2: "Gt. 1", 972: "Gt. 2"}  # panel x origin -> part, in the split-screen chapter

DPI = 200
PAGE_W, PAGE_H = round(210 / 25.4 * DPI), round(297 / 25.4 * DPI)
MARGIN_X = round(12 / 25.4 * DPI)
MARGIN_TOP = round(12 / 25.4 * DPI)
MARGIN_BOT = round(14 / 25.4 * DPI)
GAP = round(5 / 25.4 * DPI)  # between systems
HEAD_GAP = round(9 / 25.4 * DPI)  # extra space above a strip heading

INK = 0.15
N_STRINGS = 6
PITCH = (34.0, 40.0)  # string-line spacing to search over
BAR_LEVEL = 0.95  # fraction of the staff height a bar line must span
BAR_MERGE = 30  # px; closer than this is one double bar line, not two
TARGET = 1950  # px of source per system -- about three bars
STRETCH = 1.12  # horizontal-only stretch allowed to justify a line to the right margin
NUM_PAD = 32  # bar numbers straddle their bar line; keep them with the bar they open
BOOST = 1.35  # the staff lines are drawn lighter than the marks; even them out for print

font = common.font  # resolves a CJK face per platform


# ---------------------------------------------------------------- geometry


def staff_rows(a):
    """The six string lines, found by sliding a six-tooth comb down the row profile."""
    rp = (a >= 0.3).mean(axis=1)
    best = (-1.0, None)
    for pitch in np.arange(PITCH[0], PITCH[1], 0.1):
        span = pitch * (N_STRINGS - 1)
        for top in np.arange(0, a.shape[0] - span - 1, 0.5):
            ys = np.round(top + pitch * np.arange(N_STRINGS)).astype(int)
            score = rp[ys].sum()
            if score > best[0]:
                best = (score, ys)
    return best[1]


def barlines(a, staff):
    """x of every bar line. Note stems reach into the staff too, but only a bar line
    runs the whole way from the top string to the bottom one."""
    band = a[staff[0]:staff[-1] + 1]
    col = (band >= 0.5).mean(axis=0) >= BAR_LEVEL
    d = np.diff(np.concatenate(([0], col.view(np.int8), [0])))
    xs = ((np.flatnonzero(d == 1) + np.flatnonzero(d == -1)) // 2).tolist()
    out = []
    for x in xs:
        if out and x - out[-1] < BAR_MERGE:
            continue
        out.append(x)
    return out


def cut(a, bars):
    """Split into systems at bar lines. Bar lines only come every ~650px, so a greedy
    walk leaves lines a whole bar shorter than their neighbours -- and since the score
    prints at one scale, those stop well short of the margin. Fix the line count first,
    then pick the whole set of breaks at once by shortest path over the bar lines, which
    spreads the leftover evenly instead of dumping it on the last line."""
    cols = np.flatnonzero((a >= INK).any(axis=0))
    lo, hi = int(cols[0]), int(cols[-1]) + 1
    # the last window stopped wherever the display happened to sit, so the strip trails
    # off in the middle of a bar; close the passage on its last bar line instead
    if len(bars) > 2 and bars[-1] > lo + 300:
        hi = bars[-1] + 4
    pts = [lo] + [b for b in bars if lo + 300 < b < hi - 300] + [hi]
    n = min(max(1, round((hi - lo) / TARGET)), len(pts) - 1)
    want = (hi - lo) / n

    INF = float("inf")
    best = [[INF] * len(pts) for _ in range(n + 1)]
    prev = [[0] * len(pts) for _ in range(n + 1)]
    best[0][0] = 0.0
    for k in range(1, n + 1):
        for j in range(k, len(pts)):
            for i in range(k - 1, j):
                if best[k - 1][i] == INF:
                    continue
                c = best[k - 1][i] + (pts[j] - pts[i] - want) ** 2
                if c < best[k][j]:
                    best[k][j], prev[k][j] = c, i
    cuts, j = [len(pts) - 1], len(pts) - 1
    for k in range(n, 0, -1):
        j = prev[k][j]
        cuts.append(j)
    cuts = [pts[i] for i in reversed(cuts)]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def trim_rows(a):
    rows = np.flatnonzero((a >= INK).any(axis=1))
    return max(0, rows[0] - 6), rows[-1] + 7


# ---------------------------------------------------------------- labelling


def chapters_of(t0, t1):
    return [name for a, b, name in CHAPTERS if a < t1 and b > t0]


def heading(strip):
    name = " · ".join(chapters_of(strip["t0"], strip["t1"]))
    part = PARTS.get(strip["x0"]) if strip["layout"] == 1 else None
    return f"{name} — {part}" if part else name


def inline_marks(strip):
    """(chapter name, x in the strip) for every chapter that starts inside this strip."""
    out = []
    for a, _, name in CHAPTERS:
        if not strip["t0"] < a <= strip["t1"]:
            continue
        at = max((w for w in strip["windows"] if w[0] <= a), key=lambda w: w[0], default=None)
        if at is None:
            at = min(strip["windows"], key=lambda w: w[0])
        out.append((name, at[2]))
    return out


# ---------------------------------------------------------------- layout


def main():
    strips = stitch.build()

    systems = []  # (image, heading or None, [(chapter, fraction across the line)])
    for s in strips:
        a = s["img"]
        staff = staff_rows(a)
        bars = barlines(a, staff)
        spans = cut(a, bars)
        marks = inline_marks(s)
        print(f"strip {heading(s):26s} {len(bars)} bar lines -> {len(spans)} systems")
        for k, (x0, x1) in enumerate(spans):
            # the bar number sits astride the bar line, so open the line early enough to
            # carry it whole, then clear the half-number dangling off its right edge --
            # and, on an opening line, off the bar the strip itself cuts into
            sub = a[:, max(0, x0 - NUM_PAD if k else x0):x1 + 3].copy()
            sub[:staff[0] - 4, -(NUM_PAD - 6):] = 0.0
            if not k:  # the opening line has no line before it to inherit a number from
                sub[:staff[0] - 4, :NUM_PAD - 6] = 0.0
            r0, r1 = trim_rows(sub)
            here = [(n, (x - x0) / (x1 - x0)) for n, x in marks if x0 <= x < x1]
            systems.append((sub[r0:r1], heading(s) if k == 0 else None, here))

    # one scale for the whole score, set by the widest system, so a short closing line
    # prints at the same size as everything else instead of being blown up
    content_w = PAGE_W - 2 * MARGIN_X
    scale = content_w / max(img.shape[1] for img, _, _ in systems)
    scaled = []
    for img, head, here in systems:
        h = max(1, round(img.shape[0] * scale))
        # then stretch horizontally, up to STRETCH, to justify the line to the margin
        w = min(content_w, round(img.shape[1] * scale * STRETCH))
        g = 255 - np.clip(img * BOOST, 0, 1) * 255
        im = Image.fromarray(g.astype(np.uint8), "L").resize((w, h), Image.LANCZOS)
        scaled.append((im, head, here))

    title_f, s1_f, s2_f = font(56, "Bold"), font(28), font(21)
    head_f, num_f, tag_f, foot_f = font(30, "Bold"), font(20), font(23, "Bold"), font(21)
    lead = 32  # room above each system for its number and any chapter tag

    top0 = MARGIN_TOP + 172  # title block on page 1
    pages, cur, y = [], [], top0
    for im, head, here in scaled:
        need = (HEAD_GAP + 40 if head else 0) + lead + im.height
        if cur and y + need > PAGE_H - MARGIN_BOT:
            pages.append(cur)
            cur, y = [], MARGIN_TOP
            need = (40 if head else 0) + lead + im.height  # no gap at the top of a page
        hy = y + (HEAD_GAP if (cur and head) else 0) if head else None
        iy = y + need - im.height
        cur.append((im, head, here, hy, iy))
        y += need + GAP
    pages.append(cur)

    out, idx = [], 0
    for p, items in enumerate(pages):
        canvas = Image.new("L", (PAGE_W, PAGE_H), 255)
        d = ImageDraw.Draw(canvas)
        if p == 0:
            d.text((MARGIN_X, MARGIN_TOP), TITLE, font=title_f, fill=0)
            d.text((MARGIN_X, MARGIN_TOP + 68), SUB1, font=s1_f, fill=55)
            d.text((MARGIN_X, MARGIN_TOP + 108), SUB2, font=s2_f, fill=95)
            d.line((MARGIN_X, MARGIN_TOP + 150, PAGE_W - MARGIN_X, MARGIN_TOP + 150),
                   fill=0, width=2)
        for im, head, here, hy, iy in items:
            idx += 1
            if head:
                d.text((MARGIN_X, hy), head, font=head_f, fill=0)
                d.line((MARGIN_X, hy + 40, PAGE_W - MARGIN_X, hy + 40), fill=150, width=1)
            d.text((MARGIN_X, iy - 26), str(idx), font=num_f, fill=125)
            canvas.paste(im, (MARGIN_X, iy))
            for name, frac in here:
                x = MARGIN_X + int(frac * im.width)
                w = d.textlength(name, font=tag_f)
                x = min(max(x, MARGIN_X), PAGE_W - MARGIN_X - w)
                d.rectangle((x - 6, iy - 30, x + w + 6, iy - 2), fill=228)
                d.text((x, iy - 28), name, font=tag_f, fill=0)
                d.line((x - 6, iy - 2, x - 6, iy + im.height), fill=0, width=3)
        foot = f"{p + 1} / {len(pages)}"
        d.text(((PAGE_W - d.textlength(foot, font=foot_f)) / 2,
                PAGE_H - MARGIN_BOT + 20), foot, font=foot_f, fill=90)
        if p == len(pages) - 1:
            d.text((MARGIN_X, PAGE_H - MARGIN_BOT + 20), SOURCE, font=foot_f, fill=90)
        out.append(canvas)

    out[0].save(OUT, "PDF", resolution=DPI, save_all=True, append_images=out[1:])
    print(f"wrote {OUT}  ({len(out)} pages, {idx} systems)")
    os.makedirs(f"{stitch.WORK}/preview", exist_ok=True)
    for i, c in enumerate(out):
        c.resize((PAGE_W // 2, PAGE_H // 2)).save(f"{stitch.WORK}/preview/p{i + 1}.png")


if __name__ == "__main__":
    main()
