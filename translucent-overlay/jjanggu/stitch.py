#!/usr/bin/env python3
"""Chain the extracted systems back into the single continuous strip they were cut from.

Consecutive systems share their outer ~646 px: the display advances 1274 px per flip,
so every page repeats about a third of the one before it. Line those repeats up and the
whole score becomes one image again. Where two pages cover the same column, average
them -- real marks agree, and whatever bleed-through survived extraction does not.
"""

import glob

import numpy as np

from despeckle import despeckle

WORK = "/tmp/sm2"
SOLID = 0.35
DUP_IOU = 0.75  # above this at zero shift, two pages are the same system
MIN_OV, MAX_OV = 120, 1900


def load():
    out = {}
    for f in sorted(glob.glob(f"{WORK}/pages/*.npy")):
        out[int(f[-7:-4])] = np.load(f)
    return out


def _iou(A, B, k):
    x, y = A[:, -k:] >= SOLID, B[:, :k] >= SOLID
    return (x & y).sum() / max((x | y).sum(), 1)


def overlap(A, B):
    """How much of B's left edge repeats A's right edge, and how well it matches."""
    ca = (A >= SOLID).sum(axis=0).astype(np.float32)
    cb = (B >= SOLID).sum(axis=0).astype(np.float32)
    cand = []
    for k in range(MIN_OV, min(MAX_OV, A.shape[1], B.shape[1])):
        x, y = ca[-k:], cb[:k]
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        cand.append((np.corrcoef(x, y)[0, 1], k))
    if not cand:
        return None, 0.0
    cand.sort(reverse=True)
    best = max((_iou(A, B, kk), kk)
               for _, k in cand[:6] for kk in range(max(k - 6, 1), k + 7))
    return best[1], best[0]


def chain(pages):
    """Order the pages, dropping the ones that merely repeat their predecessor.
    After each title card the video resumes on the system it was already showing.
    `alias` maps a dropped page onto the twin that stands in for it."""
    ids = sorted(pages)
    keep, links, alias = [ids[0]], [], {}
    for i in ids[1:]:
        prev = keep[-1]
        A, B = pages[prev], pages[i]
        if A.shape == B.shape and _iou(A, B, A.shape[1]) >= DUP_IOU:
            print(f"  page {i:02d} repeats page {prev:02d} (title card) -- dropped")
            alias[i] = prev
            continue
        k, q = overlap(A, B)
        if k is None or q < 0.5:
            print(f"  page {prev:02d} -> {i:02d}: no overlap (q={q:.2f}) -- new strip")
            keep.append(i)
            links.append(None)
            continue
        keep.append(i)
        links.append(k)
    return keep, links, alias


def panorama(pages, order, links):
    """Composite one strip. Returns the image and each page's x origin in it."""
    xs, x = [0], 0
    for k in links:
        x += pages[order[len(xs) - 1]].shape[1] - k
        xs.append(x)
    h = max(pages[i].shape[0] for i in order)
    w = max(x0 + pages[i].shape[1] for x0, i in zip(xs, order))
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros(w, np.float32)
    for x0, i in zip(xs, order):
        a = pages[i]
        acc[:a.shape[0], x0:x0 + a.shape[1]] += a
        cnt[x0:x0 + a.shape[1]] += 1
    return despeckle(acc / np.maximum(cnt, 1)[None, :]), xs


def build():
    pages = load()
    print(f"{len(pages)} extracted systems")
    order, links, alias = chain(pages)
    # a None link means the strip is discontinuous there; split into runs
    runs, cur, curlinks = [], [order[0]], []
    for i, k in zip(order[1:], links):
        if k is None:
            runs.append((cur, curlinks))
            cur, curlinks = [i], []
        else:
            cur.append(i)
            curlinks.append(k)
    runs.append((cur, curlinks))
    out, origin = [], {}
    for n, (ids, ks) in enumerate(runs):
        pan, xs = panorama(pages, ids, ks)
        out.append((pan, ids, xs))
        for i, x0 in zip(ids, xs):
            origin[i] = (n, x0)
        print(f"  strip: pages {ids[0]}-{ids[-1]} ({len(ids)}) -> {pan.shape[1]}px"
              + (f"  overlaps {min(ks)}-{max(ks)}" if ks else ""))
    for dup, src in alias.items():
        origin[dup] = origin[src]
    return out, origin


if __name__ == "__main__":
    strips, origin = build()
    for n, (pan, ids, xs) in enumerate(strips):
        np.save(f"{WORK}/strip{n}.npy", pan)
        from PIL import Image
        im = Image.fromarray(((1 - pan) * 255).astype(np.uint8))
        im.resize((pan.shape[1] // 6, pan.shape[0] // 6)).save(f"{WORK}/strip{n}_thumb.png")
