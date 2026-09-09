#!/usr/bin/env python3
"""Re-flow the stitched bars into a justified A4 score.

Same idea as the other scripts: the video's own line breaks are unusable (its pages
overlap and are cut mid-bar at both edges), so lines are re-broken from scratch with a
DP that minimises the squared deviation from a target width, and each line is stretched
horizontally to the exact content width.

What is different is that a system here is 438px tall -- notation staff, Korean lyrics,
chord symbols and tab staff -- so vertical space, not width, is what decides how large
the score can be printed. The scale is therefore derived from LINES_PER_PAGE: pick how
many systems a page should hold, and the target line width follows from it.
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

OUT = common.out_pdf("사건의 지평선 기타 악보.pdf")
TITLE = "사건의 지평선 기타 악보"
SUBTITLE = "윤하 · A major · ♩=98 · 89마디 전체 채보"
CREDIT = "출처: 「사건의 지평선 - 윤하 | 일렉기타 입문곡 시리즈 #022」"

DPI = 200
PAGE_W, PAGE_H = round(210 / 25.4 * DPI), round(297 / 25.4 * DPI)
MARGIN_X = round(11 / 25.4 * DPI)
MARGIN_TOP = round(10 / 25.4 * DPI)
MARGIN_BOT = round(12 / 25.4 * DPI)
GAP = round(3.5 / 25.4 * DPI)  # vertical space between systems
HEAD_H = 156  # title block on page 1

LINES_PER_PAGE = 5  # sets the scale: fewer lines, bigger score, more pages
MAX_STRETCH = 1.18  # cap on how far a short line may be stretched to fill the width
BG = 244  # the panel's off-white, mapped to paper white
FLOOR = 8

# Bars 60-89 exist only in the video's second pass, where the panel sits high enough for
# the frame to cut its bar numbers in half. Those are redrawn to match the ones the
# source still shows: same 12px cap height, same grey, same place above the staff.
NUM_GREY = (158, 158, 157)
NUM_ROWS = (16, 27)  # rows the source draws its numbers on, 6 clear of the staff
NUM_MID = 10  # numbers are centred on the barline, which sits this far into the crop

font = common.font  # resolves a CJK face per platform


def whiten(a):
    """Map the panel's off-white background to paper white, keeping the ink."""
    v = (a.astype(np.float32) - FLOOR) * (255.0 / (BG - FLOOR))
    return np.clip(v, 0, 255).astype(np.uint8)


def break_lines(widths, target, forced=()):
    """Bar indices where each line starts, minimising sum (line width - target)^2.

    `forced` bars must start a line, so no line may span one.
    """
    n = len(widths)
    cap = target * MAX_STRETCH
    forced = set(forced)
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
            c = best[i] + (w - target) ** 2
            if c < best[j]:
                best[j], back[j] = c, i
    starts, j = [], n
    while j > 0:
        starts.append(back[j])
        j = back[j]
    return starts[::-1] + [n]


def number_font():
    """The size whose digits match the 12px cap height the source draws."""
    want = NUM_ROWS[1] - NUM_ROWS[0] + 1
    return min((font(s) for s in range(11, 24)),
               key=lambda f: abs((lambda b: b[3] - b[1])(f.getbbox("88")) - want))


def stamp_number(img, n, f):
    """Replace a bar number the frame clipped with one drawn where the source drew it."""
    y0, y1 = NUM_ROWS
    img[y0 - 3:y1 + 3, :NUM_MID + 14] = 255
    im = Image.fromarray(img, "RGB")
    b = f.getbbox(str(n))
    ImageDraw.Draw(im).text((NUM_MID - (b[2] - b[0]) / 2 - b[0], y0 - b[1]),
                            str(n), font=f, fill=NUM_GREY)
    return np.asarray(im)


def build_lines():
    pages, bars, _, _ = stitch.assemble()
    tops = stitch.page_tops()
    clipped = [tops[k] < stitch.extract.ABOVE for k, _, _ in bars]
    num_f = number_font()
    widths = [x1 - x0 for _, x0, x1 in bars]
    h = pages[0].shape[0]

    content_w = PAGE_W - 2 * MARGIN_X
    avail = PAGE_H - MARGIN_TOP - MARGIN_BOT
    # floor, not round: a line one px too tall costs a whole page of capacity
    line_h = (avail - (LINES_PER_PAGE - 1) * GAP) // LINES_PER_PAGE
    v = line_h / h
    target = round(content_w / v)
    forced = stitch.repeat_bars(pages, bars)
    print(f"forcing a system break at bars {[b + 1 for b in forced]} (repeat signs)")
    cuts = break_lines(widths, target, forced)

    out = []
    for a, b in zip(cuts, cuts[1:]):
        # the last bar of a line carries the barline that closes the system
        imgs = [whiten(stitch.bar_image(pages, bars[i], close=i == b - 1))
                for i in range(a, b)]
        imgs = [stamp_number(g, a + j + 1, num_f) if clipped[a + j] else g
                for j, g in enumerate(imgs)]
        w = sum(g.shape[1] for g in imgs)
        strip = np.full((h, w, 3), 255, np.uint8)
        x = 0
        for g in imgs:
            strip[:, x:x + g.shape[1]] = g
            x += g.shape[1]
        im = Image.fromarray(strip, "RGB")
        dst_w = content_w if w / content_w > 1 / MAX_STRETCH else round(w * v)
        out.append((im.resize((dst_w, line_h), Image.LANCZOS), a + 1, b))
    return out, target, content_w


def paginate(lines):
    """Line counts per page, filled to capacity then evened out."""
    lh = lines[0][0].height + GAP
    avail = PAGE_H - MARGIN_TOP - MARGIN_BOT + GAP
    cap, cap0 = int(avail // lh), int((avail - HEAD_H) // lh)
    n = len(lines)
    k = 0 if n <= cap0 else -(-(n - cap0) // cap)
    counts = [cap0] + [cap] * k
    while sum(counts) > n:  # trim from the fullest page, latest first
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
    foot_f, num_f = font(21), font(19)

    out = []
    for p, items in enumerate(pages):
        canvas = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        d = ImageDraw.Draw(canvas)
        if p == 0:
            d.text((MARGIN_X, MARGIN_TOP), TITLE, font=title_f, fill=(0, 0, 0))
            d.text((MARGIN_X, MARGIN_TOP + 58), SUBTITLE, font=sub_f, fill=(60, 60, 60))
            d.text((MARGIN_X, MARGIN_TOP + 90), CREDIT, font=foot_f, fill=(120, 120, 120))
            d.line((MARGIN_X, MARGIN_TOP + 122, PAGE_W - MARGIN_X, MARGIN_TOP + 122),
                   fill=(0, 0, 0), width=2)
        for im, a, b, y in items:
            d.text((MARGIN_X, y - 24), f"{a}–{b}", font=num_f, fill=(150, 150, 150))
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
