#!/usr/bin/env python3
"""Re-flow the panorama's bars into a justified A4 score.

Same layout engine as the other songs: a DP picks the line breaks that minimise the
squared deviation from a target width, then each line is stretched horizontally to the
exact content width, so the breaks stay balanced and the stretch stays within a few
percent of the fixed vertical scale. The scale itself is derived from LINES_PER_PAGE --
pick how many systems a page should hold and the target line width follows.

A system break is forced at every repeat sign, so a repeat opens a system and the bar
before its closing sign ends one, rather than the signs being buried mid-line. That
matters more here than anywhere else: this arrangement is navigated by its marks, with a
segno, a Da Coda, a D.S. al Coda and four separate repeats.
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

OUT = common.out_pdf("Pretender 기타 악보.pdf")
TITLE = "Pretender 기타 악보"
SUBTITLE = "Official髭男dism · ♩=92 · 62마디 전체 채보"
CREDIT = "출처: 「Official Hige Dandism - Pretender | 일렉기타 + TAB악보」"
# U+1D10B (segno) is not in Noto Sans CJK, so the mark is named rather than drawn
ROUTE = "진행 표시: Segno 9마디 · Da Coda 18마디 뒤 · D.S. al Coda 29마디 뒤 · Coda 30마디"

DPI = 200
PAGE_W, PAGE_H = round(210 / 25.4 * DPI), round(297 / 25.4 * DPI)
MARGIN_X = round(11 / 25.4 * DPI)
MARGIN_TOP = round(10 / 25.4 * DPI)
MARGIN_BOT = round(12 / 25.4 * DPI)
GAP = round(4 / 25.4 * DPI)  # vertical space between systems
HEAD_H = 176  # title block on page 1

ROW0, ROW1 = 4, 260  # the panorama's own rows that carry ink
LINES_PER_PAGE = 8
MAX_STRETCH = 1.18
REPEAT_W = 12  # px; a repeat sign is a thick line, a thin one and dots. A plain double
               # bar comes to 8-9px and a single barline to 1-2px.
FORCED_GAP = 3  # bars. This chart has eight repeat signs, several of them a bar apart
                # around the first/second endings; opening a system at every one of them
                # would leave single-bar lines, so they are thinned to this spacing.
SPLIT_COST = 4.0  # multiple of the target width, squared, charged for breaking a line
                  # inside a mark drawn above the staff. Heavy enough never to be
                  # chosen unless nothing else fits.

font = common.font  # resolves a CJK face per platform


def repeat_bars(pano, bars):
    """Indices of bars opened by a repeat sign rather than an ordinary barline."""
    out = []
    for i, (x0, _) in enumerate(bars):
        x = x0 + stitch.NUM_LEFT
        hit = stitch.barline_cols(pano[:, max(0, x - 12):x + 14])
        if len(hit) and hit[-1] - hit[0] + 1 >= REPEAT_W:
            out.append(i)
    return out


def break_lines(widths, target, forced=(), costly=()):
    """Bar indices where each line starts; no line may span a forced break."""
    n = len(widths)
    cap = target * MAX_STRETCH
    forced, costly = set(forced), set(costly)
    penalty = (SPLIT_COST * target) ** 2
    best = [0.0] + [float("inf")] * n
    back = [0] * (n + 1)
    for j in range(1, n + 1):
        w = 0
        for i in range(j - 1, -1, -1):
            w += widths[i]
            if w > cap and i < j - 1:
                break
            if any(i < b < j for b in forced):
                break
            c = best[i] + (w - target) ** 2 + (penalty if i in costly else 0)
            if c < best[j]:
                best[j], back[j] = c, i
    starts, j = [], n
    while j > 0:
        starts.append(back[j])
        j = back[j]
    return starts[::-1] + [n]


def build_lines():
    pano, bars = stitch.score_bars()
    widths = [x1 - x0 for x0, x1 in bars]
    h = ROW1 - ROW0

    content_w = PAGE_W - 2 * MARGIN_X
    avail = PAGE_H - MARGIN_TOP - MARGIN_BOT
    line_h = (avail - (LINES_PER_PAGE - 1) * GAP) // LINES_PER_PAGE
    v = line_h / h
    target = round(content_w / v)

    forced, last = [], -FORCED_GAP
    for i in repeat_bars(pano, bars):
        if i >= FORCED_GAP and i - last >= FORCED_GAP:
            forced.append(i)
            last = i
    spans = stitch.label_spans(pano)
    costly = [i for i, (x0, _) in enumerate(bars) if any(a < x0 < b for a, b in spans)]
    # a repeat sign that a mark reaches across loses: the mark has to stay in one piece,
    # and the repeat is still legible mid-line
    forced = [i for i in forced if i not in set(costly)]
    print(f"forcing a system break at bars {[i + 1 for i in forced]} (repeat signs)")
    print(f"avoiding a break at bars {[i + 1 for i in costly]} (would split a mark)")
    cuts = break_lines(widths, target, forced, costly)

    out = []
    for a, b in zip(cuts, cuts[1:]):
        imgs = [stitch.bar_image(pano, bars[i], close=i == b - 1)[ROW0:ROW1]
                for i in range(a, b)]
        w = sum(g.shape[1] for g in imgs)
        strip = np.full((h, w), 255, np.uint8)
        x = 0
        for g in imgs:
            strip[:, x:x + g.shape[1]] = g
            x += g.shape[1]
        im = Image.fromarray(strip, "L")
        dst_w = content_w if w / content_w > 1 / MAX_STRETCH else round(w * v)
        out.append((im.resize((dst_w, line_h), Image.LANCZOS), a + 1, b))
    return out, target, content_w


def paginate(lines):
    lh = lines[0][0].height + GAP
    avail = PAGE_H - MARGIN_TOP - MARGIN_BOT + GAP
    cap, cap0 = int(avail // lh), int((avail - HEAD_H) // lh)
    n = len(lines)
    k = 0 if n <= cap0 else -(-(n - cap0) // cap)
    counts = [cap0] + [cap] * k
    while sum(counts) > n:
        m = max(counts)
        counts[len(counts) - 1 - counts[::-1].index(m)] -= 1

    pages, i = [], 0
    for p, c in enumerate(counts):
        y = MARGIN_TOP + (HEAD_H if p == 0 else 0)
        cur = []
        for im, a, b in lines[i:i + c]:
            cur.append((im, a, b, y))
            y += im.height + GAP
        pages.append(cur)
        i += c
    return pages


def main():
    lines, target, content_w = build_lines()
    print(f"{lines[-1][2]} bars -> {len(lines)} lines  (target {target}px, "
          f"content {content_w}px, line height {lines[0][0].height}px)")
    for im, a, b in lines:
        print(f"  bars {a:3d}-{b:3d}  {b - a + 1} bars  {im.width}x{im.height}")

    pages = paginate(lines)
    title_f, sub_f = font(50, "Bold"), font(26)
    foot_f = font(21)

    out = []
    for p, items in enumerate(pages):
        canvas = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        d = ImageDraw.Draw(canvas)
        if p == 0:
            d.text((MARGIN_X, MARGIN_TOP), TITLE, font=title_f, fill=(0, 0, 0))
            d.text((MARGIN_X, MARGIN_TOP + 58), SUBTITLE, font=sub_f, fill=(60, 60, 60))
            d.text((MARGIN_X, MARGIN_TOP + 92), ROUTE, font=foot_f, fill=(60, 60, 60))
            d.text((MARGIN_X, MARGIN_TOP + 118), CREDIT, font=foot_f,
                   fill=(120, 120, 120))
            d.line((MARGIN_X, MARGIN_TOP + 148, PAGE_W - MARGIN_X, MARGIN_TOP + 148),
                   fill=(0, 0, 0), width=2)
        for im, a, b, y in items:
            # no per-line bar range here: these count bar *boxes*, and the five-bar
            # multi-rest makes those disagree with the numbers printed on the staff
            canvas.paste(im, (MARGIN_X, y))
        foot = f"{p + 1} / {len(pages)}"
        d.text(((PAGE_W - d.textlength(foot, font=foot_f)) / 2,
                PAGE_H - MARGIN_BOT + 22), foot, font=foot_f, fill=(90, 90, 90))
        out.append(canvas)

    out[0].save(OUT, "PDF", resolution=DPI, save_all=True, append_images=out[1:])
    print(f"wrote {OUT}  ({len(out)} pages, {len(lines)} lines)")

    os.makedirs(f"{stitch.WORK}/preview", exist_ok=True)
    for i, c in enumerate(out):
        c.resize((PAGE_W // 2, PAGE_H // 2)).save(f"{stitch.WORK}/preview/p{i + 1}.png")


if __name__ == "__main__":
    main()
