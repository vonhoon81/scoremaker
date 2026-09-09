#!/usr/bin/env python3
"""Lay the extracted Kaiju no Hanauta systems out as an A4 score.

There is no stitching to do here. The video's pages do not overlap, and each one is
already a complete system, so they only need trimming and stacking in order. They are
also not vertically interchangeable: the renderer moves the two staves apart to make
room for ledger lines, so the gap between them differs from page to page and every
system has to be trimmed on its own rather than to one shared box.
"""

import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import extract

OUT = "/home/vonhoon/projects/scoremaker/괴수의 하나우타 기타 악보.pdf"
TITLE = "괴수의 하나우타 기타 악보"
SUBTITLE = "Vaundy · Kaiju no Hanauta · 채보 전체 악보"

DPI = 200
PAGE_W, PAGE_H = round(210 / 25.4 * DPI), round(297 / 25.4 * DPI)
MARGIN_X = round(12 / 25.4 * DPI)
MARGIN_TOP = round(12 / 25.4 * DPI)
MARGIN_BOT = round(14 / 25.4 * DPI)
GAP = round(5 / 25.4 * DPI)

SKIP_TOP = 3  # the panel's own top edge row, which is faintly grey
INK = 200  # darker than this in the brightest channel is engraving
PAD = 6
BOOST = 2.4  # the source draws staff lines at ~234 grey; darken toward the paper white


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


def load():
    out = []
    for f in sorted(glob.glob(f"{extract.WORK}/pages/*.npy")):
        a = np.load(f)
        rows = np.flatnonzero((a[SKIP_TOP:].max(axis=2) < INK).any(axis=1)) + SKIP_TOP
        a = a[max(SKIP_TOP, rows[0] - PAD):rows[-1] + PAD]
        out.append(255 - np.clip((255 - a.astype(np.float32)) * BOOST, 0, 255))
    return [x.astype(np.uint8) for x in out]


def main():
    systems = load()
    print(f"{len(systems)} systems, heights {min(s.shape[0] for s in systems)}"
          f"-{max(s.shape[0] for s in systems)}px")

    content_w = PAGE_W - 2 * MARGIN_X
    scale = content_w / max(s.shape[1] for s in systems)
    scaled = []
    for s in systems:
        w, h = round(s.shape[1] * scale), max(1, round(s.shape[0] * scale))
        scaled.append(Image.fromarray(s, "RGB").resize((w, h), Image.LANCZOS))

    title_f, sub_f = font(56, "Bold"), font(30)
    foot_f, num_f = font(22), font(20)
    lead = 22

    pages, cur, y = [], [], MARGIN_TOP + 150 + lead
    for im in scaled:
        if cur and y + im.height > PAGE_H - MARGIN_BOT:
            pages.append(cur)
            cur, y = [], MARGIN_TOP + lead
        cur.append((im, y))
        y += im.height + GAP + lead
    pages.append(cur)

    out, idx = [], 0
    for p, items in enumerate(pages):
        canvas = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        d = ImageDraw.Draw(canvas)
        if p == 0:
            d.text((MARGIN_X, MARGIN_TOP), TITLE, font=title_f, fill=(0, 0, 0))
            d.text((MARGIN_X, MARGIN_TOP + 70), SUBTITLE, font=sub_f, fill=(70, 70, 70))
            d.line((MARGIN_X, MARGIN_TOP + 118, PAGE_W - MARGIN_X, MARGIN_TOP + 118),
                   fill=(0, 0, 0), width=2)
        for im, y in items:
            idx += 1
            d.text((MARGIN_X, y - 24), str(idx), font=num_f, fill=(140, 140, 140))
            canvas.paste(im, (MARGIN_X, y))
        foot = f"{p + 1} / {len(pages)}"
        d.text(((PAGE_W - d.textlength(foot, font=foot_f)) / 2,
                PAGE_H - MARGIN_BOT + 20), foot, font=foot_f, fill=(90, 90, 90))
        out.append(canvas)

    out[0].save(OUT, "PDF", resolution=DPI, save_all=True, append_images=out[1:])
    print(f"wrote {OUT}  ({len(out)} pages, {idx} systems)")
    os.makedirs(f"{extract.WORK}/preview", exist_ok=True)
    for i, c in enumerate(out):
        c.resize((PAGE_W // 2, PAGE_H // 2)).save(f"{extract.WORK}/preview/p{i + 1}.png")


if __name__ == "__main__":
    main()
