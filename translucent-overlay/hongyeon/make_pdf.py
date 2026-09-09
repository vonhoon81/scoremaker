#!/usr/bin/env python3
"""Dedupe the extracted tab systems and lay them out one-per-line in a PDF."""

import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)  # the repo root, for common.py and the source videos

import common  # noqa: E402

WORK = common.work("sm")
OUT = common.out_pdf("홍연 어쿠스틱 기타 악보.pdf")

TITLE = "홍연 어쿠스틱 기타 악보"
SUBTITLE = "안예은 - 홍연"

DPI = 200
PAGE_W = round(210 / 25.4 * DPI)  # A4 portrait
PAGE_H = round(297 / 25.4 * DPI)
MARGIN_X = round(13 / 25.4 * DPI)
MARGIN_TOP = round(13 / 25.4 * DPI)
MARGIN_BOT = round(13 / 25.4 * DPI)
GAP = round(4 / 25.4 * DPI)  # vertical space between systems

INK = 0.15  # alpha at/above which a pixel counts as ink
# Duplicate detection runs on a *confident* mask (faint staff lines flicker between
# takes) and on content-aligned crops (the overlay sits at different heights in the
# frame depending on whether the system carries a section label). Real duplicates
# land at 0.001-0.004; the closest genuinely-different pair is 0.046.
SOLID = 0.5
DUP_TOLERANCE = 0.02

font = common.font  # resolves a CJK face per platform


def load_systems():
    out = []
    for f in sorted(glob.glob(f"{WORK}/pages/*.npy")):
        out.append((os.path.basename(f)[:3], np.load(f)))
    return out


def _aligned(a):
    """Crop to the confident-ink bounding box so systems can be compared by content."""
    m = a >= SOLID
    r = np.flatnonzero(m.any(axis=1))
    c = np.flatnonzero(m.any(axis=0))
    return m[r[0] : r[-1] + 1, c[0] : c[-1] + 1]


def same_system(a, b):
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pa = np.zeros((h, w), bool)
    pa[: a.shape[0], : a.shape[1]] = a
    pb = np.zeros((h, w), bool)
    pb[: b.shape[0], : b.shape[1]] = b
    return (pa != pb).mean() < DUP_TOLERANCE


def dedupe(systems, keep_repeats):
    """Find recurring systems; drop them only if keep_repeats is False."""
    seen, repeats, kept = [], [], []
    for name, a in systems:
        m = _aligned(a)
        match = next((kn for kn, km in seen if same_system(m, km)), None)
        if match is None:
            seen.append((name, m))
        else:
            repeats.append((name, match))
            if not keep_repeats:
                continue
        kept.append((name, a, match))
    return kept, repeats


def common_columns(systems):
    """One horizontal crop shared by every system, so barlines stay aligned."""
    lo, hi = 10**9, 0
    for _, a, _ in systems:
        cols = np.flatnonzero((a >= INK).any(axis=0))
        lo, hi = min(lo, cols[0]), max(hi, cols[-1])
    pad = 6
    return max(0, lo - pad), hi + 1 + pad


def trim_rows(a):
    rows = np.flatnonzero((a >= INK).any(axis=1))
    pad = 4
    return max(0, rows[0] - pad), rows[-1] + 1 + pad


def to_image(a):
    return Image.fromarray((255 - np.clip(a, 0, 1) * 255).astype(np.uint8), "L")


def main(keep_repeats=True):
    systems = load_systems()
    kept, repeats = dedupe(systems, keep_repeats)
    verb = "kept in place" if keep_repeats else "removed"
    print(f"{len(systems)} systems -> {len(kept)} lines "
          f"({len(repeats)} recurring systems {verb})")
    for name, same_as in repeats:
        print(f"  system {name} repeats system {same_as}")

    c0, c1 = common_columns(kept)
    strips = []
    for _, a, _ in kept:
        r0, r1 = trim_rows(a)
        strips.append(to_image(a[r0:r1, c0:c1]))

    content_w = PAGE_W - 2 * MARGIN_X
    scaled = [s.resize((content_w, max(1, round(s.height * content_w / s.width))),
                       Image.LANCZOS) for s in strips]

    title_f = font(58, "Bold")
    sub_f, foot_f, num_f = font(34), font(24), font(22)
    head_h = MARGIN_TOP + 168  # leaves room for the title block and the rule under it

    # ---- paginate: greedy to find the page count, then even out so the last page
    # isn't left holding a single system
    def lay_out(counts):
        pages, i = [], 0
        for p, n in enumerate(counts):
            y = head_h if p == 0 else MARGIN_TOP
            cur = []
            for s in scaled[i : i + n]:
                if cur and y + s.height > PAGE_H - MARGIN_BOT:
                    return None  # does not fit
                cur.append((i, s, y))
                y += s.height + GAP
                i += 1
            pages.append(cur)
        return pages

    greedy, y = [0], head_h
    for s in scaled:
        if greedy[-1] and y + s.height > PAGE_H - MARGIN_BOT:
            greedy.append(0)
            y = MARGIN_TOP
        greedy[-1] += 1
        y += s.height + GAP

    n_pages, n = len(greedy), len(scaled)
    even = [n // n_pages + (1 if i < n % n_pages else 0) for i in range(n_pages)]
    pages = lay_out(even) or lay_out(greedy)

    # ---- draw
    line_of = {name: i + 1 for i, (name, _, _) in enumerate(kept)}
    out = []
    for p, items in enumerate(pages):
        canvas = Image.new("L", (PAGE_W, PAGE_H), 255)
        d = ImageDraw.Draw(canvas)
        if p == 0:
            d.text((MARGIN_X, MARGIN_TOP), TITLE, font=title_f, fill=0)
            d.text((MARGIN_X, MARGIN_TOP + 74), SUBTITLE, font=sub_f, fill=0)
            d.line((MARGIN_X, MARGIN_TOP + 120, PAGE_W - MARGIN_X, MARGIN_TOP + 120),
                   fill=0, width=2)
        for i, s, y in items:
            d.text((MARGIN_X, y - 26), str(i + 1), font=num_f, fill=110)
            src = line_of.get(kept[i][2])
            if src:
                d.text((MARGIN_X + 34, y - 26), f"(= {src}행 반복)", font=num_f, fill=140)
            canvas.paste(s, (MARGIN_X, y))
        foot = f"{p + 1} / {len(pages)}"
        w = d.textlength(foot, font=foot_f)
        d.text(((PAGE_W - w) / 2, PAGE_H - MARGIN_BOT + 16), foot, font=foot_f, fill=90)
        if p == len(pages) - 1:
            uniq = len(set(n for n, _, m in kept if m is None) | {m for _, _, m in kept if m})
            note = f"총 {len(scaled)}행 (고유 {uniq}행) · {SUBTITLE} · 2026-08-16"
            d.text((MARGIN_X, PAGE_H - MARGIN_BOT + 16), note, font=foot_f, fill=90)
        out.append(canvas)

    out[0].save(OUT, "PDF", resolution=DPI, save_all=True, append_images=out[1:])
    print(f"wrote {OUT}  ({len(out)} pages, {len(scaled)} lines)")

    os.makedirs(f"{WORK}/preview", exist_ok=True)
    for i, c in enumerate(out):
        c.resize((PAGE_W // 2, PAGE_H // 2)).save(f"{WORK}/preview/page{i + 1}.png")


if __name__ == "__main__":
    main(keep_repeats="--unique-only" not in sys.argv)
