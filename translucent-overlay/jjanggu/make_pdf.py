#!/usr/bin/env python3
"""Cut the stitched strip into systems at bar lines and lay them out as an A4 score."""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import extract
import sections
import stitch

OUT = "/home/vonhoon/projects/scoremaker/짱구 브금 기타 악보.pdf"
TITLE = "짱구 브금 기타 악보"
SUBTITLE = "기타 1일차 vs 기타 10년차 · 채보 전체 악보"

DPI = 200
PAGE_W, PAGE_H = round(210 / 25.4 * DPI), round(297 / 25.4 * DPI)
MARGIN_X = round(12 / 25.4 * DPI)
MARGIN_TOP = round(12 / 25.4 * DPI)
MARGIN_BOT = round(14 / 25.4 * DPI)
GAP = round(6 / 25.4 * DPI)

INK = 0.15
BAR_LEVEL = 0.95  # fraction of the staff height a bar line must span
BAR_MERGE = 24  # px; closer than this is one double bar line, not two
TARGET = 1750  # px of source per system -- about four bars, ~13mm of staff on A4

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


def staff_rows(pan):
    th = pan * 100.0
    return extract.find_staff(th)


def barlines(pan, staff):
    """x of every bar line. Note stems reach the staff too, but only a bar line runs
    the whole way from the top string to the bottom one."""
    band = pan[staff[0]:staff[-1] + 1]
    col = (band >= 0.5).mean(axis=0) >= BAR_LEVEL
    d = np.diff(np.concatenate(([0], col.view(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    xs = ((s + e) // 2).tolist()
    merged = []
    for x in xs:
        if merged and x - merged[-1] < BAR_MERGE:
            continue
        merged.append(x)
    return merged


def cut(pan, bars):
    """Split into systems at bar lines. Pick the system count first, then aim for equal
    widths, so the lines all reach the right margin instead of trailing off raggedly."""
    cols = np.flatnonzero((pan >= INK).any(axis=0))
    lo, hi = int(cols[0]), int(cols[-1]) + 1
    n = max(1, round((hi - lo) / TARGET))
    inner = [b for b in bars if lo + 300 < b < hi - 300]
    cuts = [lo]
    for i in range(1, n):
        want = lo + (hi - lo) * i / n
        nxt = min((b for b in inner if b > cuts[-1] + 300),
                  key=lambda b: abs(b - want), default=None)
        if nxt is not None:
            cuts.append(nxt)
    cuts.append(hi)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def trim_rows(a):
    rows = np.flatnonzero((a >= INK).any(axis=1))
    return max(0, rows[0] - 5), rows[-1] + 6


def main():
    strips, origin = stitch.build()
    pages = extract.pages()
    bars = {n: barlines(pan, staff_rows(pan)) for n, (pan, _, _) in enumerate(strips)}
    print("sections:")
    secs = sections.locate(pages, origin, bars)

    systems = []  # (image, list of section labels starting on this line)
    for n, (pan, ids, xs) in enumerate(strips):
        spans = cut(pan, bars[n])
        print(f"strip {n}: {len(bars[n])} bar lines -> {len(spans)} systems")
        for a, b in spans:
            sub = pan[:, a:b]
            r0, r1 = trim_rows(sub)
            here = [(lab, (x - a) / (b - a)) for lab, s, x in secs
                    if s == n and a <= x < b]
            systems.append((sub[r0:r1], here))

    # One scale for the whole score, set by the widest system, so the closing line --
    # which is only a few bars long -- prints at the same size as everything else
    # instead of being blown up to fill the width.
    content_w = PAGE_W - 2 * MARGIN_X
    scale = content_w / max(img.shape[1] for img, _ in systems)
    scaled = []
    for img, here in systems:
        w = max(1, round(img.shape[1] * scale))
        h = max(1, round(img.shape[0] * scale))
        im = Image.fromarray((255 - np.clip(img, 0, 1) * 255).astype(np.uint8), "L")
        scaled.append((im.resize((w, h), Image.LANCZOS), here))

    title_f, sub_f = font(56, "Bold"), font(30)
    foot_f, num_f, sec_f = font(22), font(20), font(24, "Bold")
    head_h = MARGIN_TOP + 150
    lead = 34  # room above each system for its number and any section tag

    pages_out, cur, y = [], [], head_h + lead
    for item in scaled:
        h = item[0].height
        if cur and y + h > PAGE_H - MARGIN_BOT:
            pages_out.append(cur)
            cur, y = [], MARGIN_TOP + lead
        cur.append((item, y))
        y += h + GAP + lead
    pages_out.append(cur)

    out, idx = [], 0
    for p, items in enumerate(pages_out):
        canvas = Image.new("L", (PAGE_W, PAGE_H), 255)
        d = ImageDraw.Draw(canvas)
        if p == 0:
            d.text((MARGIN_X, MARGIN_TOP), TITLE, font=title_f, fill=0)
            d.text((MARGIN_X, MARGIN_TOP + 70), SUBTITLE, font=sub_f, fill=60)
            d.line((MARGIN_X, MARGIN_TOP + 118, PAGE_W - MARGIN_X, MARGIN_TOP + 118),
                   fill=0, width=2)
        for (im, here), y in items:
            idx += 1
            d.text((MARGIN_X, y - 26), str(idx), font=num_f, fill=120)
            canvas.paste(im, (MARGIN_X, y))
            for lab, frac in here:
                x = MARGIN_X + int(frac * im.width)
                w = d.textlength(lab, font=sec_f)
                x = min(max(x, MARGIN_X), PAGE_W - MARGIN_X - w)
                d.rectangle((x - 6, y - 30, x + w + 6, y - 2), fill=225)
                d.text((x, y - 28), lab, font=sec_f, fill=0)
                d.line((x - 6, y - 2, x - 6, y + im.height), fill=0, width=3)
        foot = f"{p + 1} / {len(pages_out)}"
        d.text(((PAGE_W - d.textlength(foot, font=foot_f)) / 2,
                PAGE_H - MARGIN_BOT + 20), foot, font=foot_f, fill=90)
        out.append(canvas)

    out[0].save(OUT, "PDF", resolution=DPI, save_all=True, append_images=out[1:])
    print(f"wrote {OUT}  ({len(out)} pages, {idx} systems)")
    os.makedirs(f"{extract.WORK}/preview", exist_ok=True)
    for i, c in enumerate(out):
        c.resize((PAGE_W // 2, PAGE_H // 2)).save(f"{extract.WORK}/preview/p{i + 1}.png")


if __name__ == "__main__":
    main()
