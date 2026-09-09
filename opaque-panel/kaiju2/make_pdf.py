#!/usr/bin/env python3
"""Re-flow the stitched bars into a justified A4 score.

The video's own line breaks are useless here -- its windows overlap by a bar and are cut
mid-bar at the right edge -- so the score is re-engraved from scratch: bars are packed
into lines the way a typesetter packs words into a paragraph.

Bar widths are content-driven in the source (390px for a strummed verse bar, 136px for a
bar of rest), so greedy packing would leave a quarter of some lines empty. Instead a DP
picks the breaks that minimise the squared deviation from a target line width of
BARS_PER_LINE full-width bars, and each line is then stretched horizontally to the exact
content width. Because the breaks are balanced, that stretch stays within a few percent
of the fixed vertical scale, so the note heads never visibly distort.
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import stitch

OUT = "/home/vonhoon/projects/scoremaker/괴수의 하나우타 기타 악보 v2.pdf"
TITLE = "괴수의 하나우타 기타 악보"
SUBTITLE = "Vaundy · 怪獣の花唄 · 137마디 전체 채보 · ♩=150"
CREDIT = "출처: AironA 「【TAB譜】怪獣の花唄 Vaundy ギター 弾いてみた」"

DPI = 200
PAGE_W, PAGE_H = round(210 / 25.4 * DPI), round(297 / 25.4 * DPI)
MARGIN_X = round(12 / 25.4 * DPI)
MARGIN_TOP = round(12 / 25.4 * DPI)
MARGIN_BOT = round(14 / 25.4 * DPI)
GAP = round(4.5 / 25.4 * DPI)  # vertical space between systems
HEAD_H = 150  # title block on page 1

BARS_PER_LINE = 4  # of a full-width (390px) bar; narrow rest bars pack tighter
MAX_STRETCH = 1.18  # cap on how far a short line may be stretched to fill the width
BOOST = 2.2  # the source draws staff lines at ~237 grey; pull them toward black

CJK_VF = "/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc"
NANUM = "/usr/share/fonts/naver-nanum-gothic-fonts/NanumGothic%s.ttf"


def font(size, weight="Regular"):
    if os.path.exists(CJK_VF):
        f = ImageFont.truetype(CJK_VF, size)
        f.set_variation_by_name(weight)
        return f
    path = NANUM % ("" if weight == "Regular" else weight)
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    raise SystemExit("no CJK font found")


def darken(a):
    return (255 - np.clip((255 - a.astype(np.float32)) * BOOST, 0, 255)).astype(np.uint8)


def break_lines(widths, target):
    """Bar indices where each line starts, minimising sum (line width - target)^2."""
    n = len(widths)
    cap = target * MAX_STRETCH  # a line wider than this would have to shrink too far
    best = [0.0] + [float("inf")] * n
    back = [0] * (n + 1)
    for j in range(1, n + 1):
        w = 0
        for i in range(j - 1, -1, -1):
            w += widths[i]
            if w > cap and i < j - 1:
                break
            c = best[i] + (w - target) ** 2
            if c < best[j]:
                best[j], back[j] = c, i
    starts, j = [], n
    while j > 0:
        starts.append(back[j])
        j = back[j]
    return starts[::-1] + [n]


def build_lines():
    """Each line as a full-content-width image, plus its bar range."""
    bars = [darken(s) for s in stitch.strips()]
    widths = [b.shape[1] for b in bars]
    h = bars[0].shape[0]

    full = int(np.median([w for w in widths if w > 300]))
    target = BARS_PER_LINE * full
    cuts = break_lines(widths, target)
    content_w = PAGE_W - 2 * MARGIN_X
    v = content_w / target  # one vertical scale for every line, so staves match

    out = []
    for a, b in zip(cuts, cuts[1:]):
        w = sum(widths[a:b])
        strip = np.full((h, w, 3), 255, np.uint8)
        x = 0
        for bar in bars[a:b]:
            strip[:, x:x + bar.shape[1]] = bar
            x += bar.shape[1]
        im = Image.fromarray(strip, "RGB")
        dst_w = content_w if w / content_w > 1 / MAX_STRETCH else round(w * v)
        out.append((im.resize((dst_w, max(1, round(h * v))), Image.LANCZOS), a + 1, b))
    return out, target, content_w


def paginate(lines):
    """Line counts per page, evened out so the last page is not left nearly empty."""
    def fits(counts):
        pages, i = [], 0
        for p, n in enumerate(counts):
            y = MARGIN_TOP + HEAD_H if p == 0 else MARGIN_TOP
            cur = []
            for im, a, b in lines[i:i + n]:
                if cur and y + im.height > PAGE_H - MARGIN_BOT:
                    return None
                cur.append((im, a, b, y))
                y += im.height + GAP
                i += 1
            pages.append(cur)
        return pages

    greedy, y = [0], MARGIN_TOP + HEAD_H
    for im, _, _ in lines:
        if greedy[-1] and y + im.height > PAGE_H - MARGIN_BOT:
            greedy.append(0)
            y = MARGIN_TOP
        greedy[-1] += 1
        y += im.height + GAP

    n_pages, n = len(greedy), len(lines)
    even = [n // n_pages + (1 if i < n % n_pages else 0) for i in range(n_pages)]
    return fits(even) or fits(greedy)


def main():
    lines, target, content_w = build_lines()
    print(f"{sum(b - a + 1 for _, a, b in lines)} bars -> {len(lines)} lines"
          f"  (target {target}px, content {content_w}px)")
    for im, a, b in lines:
        print(f"  bars {a:3d}-{b:3d}  {b - a + 1} bars  {im.width}x{im.height}")

    pages = paginate(lines)
    title_f, sub_f = font(54, "Bold"), font(28)
    foot_f, num_f = font(22), font(20)

    out = []
    for p, items in enumerate(pages):
        canvas = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        d = ImageDraw.Draw(canvas)
        if p == 0:
            d.text((MARGIN_X, MARGIN_TOP), TITLE, font=title_f, fill=(0, 0, 0))
            d.text((MARGIN_X, MARGIN_TOP + 68), SUBTITLE, font=sub_f, fill=(60, 60, 60))
            d.text((MARGIN_X, MARGIN_TOP + 104), CREDIT, font=foot_f, fill=(120, 120, 120))
            d.line((MARGIN_X, MARGIN_TOP + HEAD_H - 16,
                    PAGE_W - MARGIN_X, MARGIN_TOP + HEAD_H - 16), fill=(0, 0, 0), width=2)
        for im, a, b, y in items:
            d.text((MARGIN_X, y - 24), f"{a}–{b}", font=num_f, fill=(150, 150, 150))
            canvas.paste(im, (MARGIN_X, y))
        foot = f"{p + 1} / {len(pages)}"
        d.text(((PAGE_W - d.textlength(foot, font=foot_f)) / 2,
                PAGE_H - MARGIN_BOT + 20), foot, font=foot_f, fill=(90, 90, 90))
        out.append(canvas)

    out[0].save(OUT, "PDF", resolution=DPI, save_all=True, append_images=out[1:])
    print(f"wrote {OUT}  ({len(out)} pages, {len(lines)} lines)")

    os.makedirs(f"{stitch.WORK}/preview", exist_ok=True)
    for i, c in enumerate(out):
        c.resize((PAGE_W // 2, PAGE_H // 2)).save(f"{stitch.WORK}/preview/p{i + 1}.png")


if __name__ == "__main__":
    main()
