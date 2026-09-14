#!/usr/bin/env python3
"""Re-flow the score strip's bars into a justified A4 drum score.

Same layout engine as the other songs: a DP picks the line breaks that minimise the
squared deviation from a target width, each line is stretched to the exact content
width, and the scale follows from LINES_PER_PAGE.

Specific to this chart: it prints no bar numbers of its own, so one is drawn into the
left margin at the start of every system, and it has a multi-bar rest, so those numbers
come from stitch.bar_numbers() rather than from counting boxes -- the rest is three bars
wide but one box, and numbering boxes would run three short for the rest of the score.
There are no repeat signs (every barline group measures 2-3px), so nothing forces a
break; the machinery is kept because it is what reports that.
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import stitch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)  # the repo root, for common.py and the source videos

import common  # noqa: E402

# ---------------------------------------------------------------- PER-SONG -----------
OUT = common.out_pdf("만찬가 드럼 악보.pdf")
TITLE = "만찬가(晩餐歌) — 드럼 악보"
SUBTITLE = "tuki. · ♩=102 (4/4) · 77마디 전체 채보"
CREDIT = "출처: atthedrum 「tuki. - 만찬가(晩餐歌) 드럼커버 | 드럼악보」"
NOTE = "영상의 가사 줄은 제외하고 드럼 표기만 옮김 · 마디번호는 다중쉼표를 포함해 센 것"
# ------------------------------------------------------------- CHANNEL TEMPLATE -------

DPI = 200
PAGE_W, PAGE_H = round(210 / 25.4 * DPI), round(297 / 25.4 * DPI)
MARGIN_X = round(13 / 25.4 * DPI)  # a little wider than the guitar charts: the bar
                                   # number in the margin needs room
MARGIN_TOP = round(10 / 25.4 * DPI)
MARGIN_BOT = round(12 / 25.4 * DPI)
GAP = round(5 / 25.4 * DPI)
HEAD_H = 168

ROW0, ROW1 = 4, 204  # the strip's rows that carry ink
LINES_PER_PAGE = 9
MAX_STRETCH = 1.18
REPEAT_W = 6  # px; nothing in this chart reaches it
FORCED_GAP = 3
SPLIT_COST = 4.0

font = common.font


def repeat_bars(pano, bars):
    out = []
    for i, (x0, _) in enumerate(bars):
        hit = stitch.barline_cols(pano[:, max(0, x0 - 12):x0 + 14])
        if len(hit) and hit[-1] - hit[0] + 1 >= REPEAT_W:
            out.append(i)
    return out


def break_lines(widths, target, forced=(), costly=()):
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
    _, _, start, span, _, _, total = stitch.bar_numbers()
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
    forced = [i for i in forced if i not in set(costly)]
    print(f"forcing a system break at bars {[start[i] for i in forced]} (repeat signs)")
    print(f"avoiding a break at {len(costly)} bars (would split a beam or a section box)")
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
        im = Image.fromarray(strip)
        dst_w = content_w if w / content_w > 1 / MAX_STRETCH else round(w * v)
        out.append((im.resize((dst_w, line_h), Image.LANCZOS), start[a],
                    start[b - 1] + span[b - 1] - 1))
    return out, target, content_w, v, total


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
    lines, target, content_w, v, total = build_lines()
    print(f"{total} bars -> {len(lines)} lines  (target {target}px, content {content_w}px,"
          f" line height {lines[0][0].height}px, scale {v:.3f})")
    for im, a, b in lines:
        print(f"  bars {a:3d}-{b:3d}  {im.width}x{im.height}")

    pages = paginate(lines)
    title_f, sub_f = font(50, "Bold"), font(26)
    foot_f, num_f = font(21), font(22)
    staff_y = round((stitch.STAFF_TOP - ROW0) * v)

    out = []
    for p, items in enumerate(pages):
        canvas = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        d = ImageDraw.Draw(canvas)
        if p == 0:
            d.text((MARGIN_X, MARGIN_TOP), TITLE, font=title_f, fill=(0, 0, 0))
            d.text((MARGIN_X, MARGIN_TOP + 58), SUBTITLE, font=sub_f, fill=(60, 60, 60))
            d.text((MARGIN_X, MARGIN_TOP + 92), NOTE, font=foot_f, fill=(60, 60, 60))
            d.text((MARGIN_X, MARGIN_TOP + 118), CREDIT, font=foot_f, fill=(120, 120, 120))
            d.line((MARGIN_X, MARGIN_TOP + 148, PAGE_W - MARGIN_X, MARGIN_TOP + 148),
                   fill=(0, 0, 0), width=2)
        for im, a, b, y in items:
            canvas.paste(im, (MARGIN_X, y))
            lab = str(a)
            d.text((MARGIN_X - 8 - d.textlength(lab, font=num_f), y + staff_y - 8),
                   lab, font=num_f, fill=(110, 110, 110))
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
