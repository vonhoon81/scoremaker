#!/usr/bin/env python3
"""Stack the three guitar parts into one score, a system at a time.

The video plays the song through three times, once per part, at the same tempo, so the
pages line up one for one: page N of Gt.1, Gt.2 and Gt.3 all cover the same bars, and
because the engraving is the same the bar lines land at the same x. Each slot therefore
becomes one three-part system, Gt.1 over Gt.2 over Gt.3.

Where a part rests for a whole system the arranger hid its tab staff and left only a
line of rests, so those rows come out shorter. That is the score, not a defect.
"""

import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import crop
import extract

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)  # the repo root, for common.py and the source videos

import common  # noqa: E402

OUT = common.out_pdf("BETELGEUSE 기타 3파트 악보.pdf")
TITLE = "BETELGEUSE (ベテルギウス) 기타 3파트 악보"
SUB1 = "Yuuri · 일렉 기타 커버 · ♩= 90"
SUB2 = ("Gt.1 — Capo On 3 Fret   |   Gt.2 · Gt.3 — Standard Tuning (E A D G B E)"
        "   |   편곡 출처: mymusicsheet.com/coverskills")

DPI = 200
PAGE_W, PAGE_H = round(210 / 25.4 * DPI), round(297 / 25.4 * DPI)
MARGIN_X = round(11 / 25.4 * DPI)
MARGIN_TOP = round(11 / 25.4 * DPI)
MARGIN_BOT = round(13 / 25.4 * DPI)
GUTTER = 86  # left column holding the Gt.1 / Gt.2 / Gt.3 labels
STAVE_GAP = 6  # between the three parts of one system
SLOT_GAP = round(9 / 25.4 * DPI)  # between systems

BOOST = 1.25

font = common.font  # resolves a CJK face per platform


def slots():
    """[[Gt.1 image, Gt.2 image, Gt.3 image], ...] -- one entry per system."""
    ps = extract.pages()
    files = sorted(glob.glob(f"{extract.WORK}/pages/*.npy"))
    by_part = {0: [], 1: [], 2: []}
    for f, (_, _, p) in zip(files, ps):
        by_part[p].append(f)
    n = min(len(v) for v in by_part.values())
    if len({len(v) for v in by_part.values()}) != 1:
        print("  parts differ in length: " + str({k: len(v) for k, v in by_part.items()})
              + f" -- using the first {n}")
    out = []
    for i in range(n):
        row = []
        for p in range(3):
            a = np.load(by_part[p][i])
            box = crop.system_box(a)
            sub = a[box[0]:box[1]] if box else a
            row.append(255 - np.clip((255 - sub.astype(np.float32)) * BOOST, 0, 255))
        out.append([x.astype(np.uint8) for x in row])
    return out


def main():
    rows = slots()
    print(f"{len(rows)} systems x 3 parts")

    music_w = PAGE_W - 2 * MARGIN_X - GUTTER
    scale = music_w / max(im.shape[1] for r in rows for im in r)
    lead = 30

    groups = []
    for r in rows:
        ims = [Image.fromarray(im, "RGB").resize(
            (round(im.shape[1] * scale), max(1, round(im.shape[0] * scale))),
            Image.LANCZOS) for im in r]
        h = sum(i.height for i in ims) + STAVE_GAP * 2
        groups.append((ims, h))

    pages, cur, y = [], [], MARGIN_TOP + 190 + lead
    for g in groups:
        if cur and y + g[1] > PAGE_H - MARGIN_BOT:
            pages.append(cur)
            cur, y = [], MARGIN_TOP + lead
        cur.append((g, y))
        y += g[1] + SLOT_GAP + lead
    pages.append(cur)

    lab_f, num_f = font(26, "Bold"), font(21)
    title_f, s1_f, s2_f, foot_f = font(52, "Bold"), font(29), font(21), font(21)

    out, idx = [], 0
    for p, items in enumerate(pages):
        canvas = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        d = ImageDraw.Draw(canvas)
        if p == 0:
            d.text((MARGIN_X, MARGIN_TOP), TITLE, font=title_f, fill=(0, 0, 0))
            d.text((MARGIN_X, MARGIN_TOP + 66), SUB1, font=s1_f, fill=(60, 60, 60))
            d.text((MARGIN_X, MARGIN_TOP + 106), SUB2, font=s2_f, fill=(95, 95, 95))
            d.line((MARGIN_X, MARGIN_TOP + 150, PAGE_W - MARGIN_X, MARGIN_TOP + 150),
                   fill=(0, 0, 0), width=2)
        for (ims, h), y in items:
            idx += 1
            d.text((MARGIN_X, y - 27), f"{idx}", font=num_f, fill=(150, 150, 150))
            d.line((MARGIN_X + GUTTER - 14, y, MARGIN_X + GUTTER - 14, y + h),
                   fill=(170, 170, 170), width=2)
            yy = y
            for k, im in enumerate(ims):
                d.text((MARGIN_X + 4, yy + im.height // 2 - 16), f"Gt.{k + 1}",
                       font=lab_f, fill=(40, 40, 40))
                canvas.paste(im, (MARGIN_X + GUTTER, yy))
                yy += im.height + STAVE_GAP
        foot = f"{p + 1} / {len(pages)}"
        d.text(((PAGE_W - d.textlength(foot, font=foot_f)) / 2,
                PAGE_H - MARGIN_BOT + 18), foot, font=foot_f, fill=(90, 90, 90))
        out.append(canvas)

    out[0].save(OUT, "PDF", resolution=DPI, save_all=True, append_images=out[1:])
    print(f"wrote {OUT}  ({len(out)} pages, {idx} systems)")
    os.makedirs(f"{extract.WORK}/preview", exist_ok=True)
    for i, c in enumerate(out):
        c.resize((PAGE_W // 2, PAGE_H // 2)).save(f"{extract.WORK}/preview/p{i + 1}.png")


if __name__ == "__main__":
    main()
