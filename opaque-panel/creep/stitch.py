#!/usr/bin/env python3
"""Chain the extracted windows back into the strips of score they were cut from.

Each step advances the display about two bars while the window holds three, so
consecutive windows repeat roughly a third of each other; line those repeats up and the
run becomes one long image again. Where two windows cover the same column, average them.

The lesson does not play the song through, though -- it jumps to whichever passage it is
teaching next -- so a run can also break outright. When no overlap explains the step, the
chain starts a new strip there rather than inventing a join.
"""

import json
import glob
import os

import numpy as np

WORK = "/tmp/sm5"
SOLID = 0.45
DUP_IOU = 0.80  # above this at zero shift, two windows show the same passage
MIN_OV, MAX_OV = 200, 1800
MIN_Q = 0.85  # IoU on the overlap below which the join is not believed
K_TOL = 12  # px a join may stray from the run's own step before it is disbelieved


def load():
    with open(f"{WORK}/pages.json") as fh:
        meta = json.load(fh)
    pages = {}
    for f in sorted(glob.glob(f"{WORK}/pages/*.npy")):
        pages[int(os.path.basename(f)[:3])] = np.load(f)
    return pages, meta


def _iou(A, B, k):
    x, y = A[:, -k:] >= SOLID, B[:, :k] >= SOLID
    return (x & y).sum() / max((x | y).sum(), 1)


def overlap(A, B):
    """How much of B's left edge repeats A's right edge, and how well it matches.

    Scored on the full two-dimensional mask, not a column-ink profile: through the bars
    of rests every column looks like every other one, and a profile match there is happy
    to slide the join a whole (narrow) bar sideways -- which double-strikes every bar
    number downstream. The digits and note heads pin it down, so keep the rows.
    """
    a = (A >= SOLID).astype(np.float32)
    b = (B >= SOLID).astype(np.float32)
    wa, wb = a.shape[1], b.shape[1]
    n = 1 << (wa + wb - 1).bit_length()
    # inter(k) = sum over the k overlapping columns of a & b, for every k at once
    C = np.fft.irfft(np.fft.rfft(a, n, axis=1) * np.conj(np.fft.rfft(b, n, axis=1)),
                     n, axis=1).sum(axis=0)
    ca = np.concatenate(([0.0], np.cumsum(a.sum(axis=0)[::-1])))  # a's last k columns
    cb = np.concatenate(([0.0], np.cumsum(b.sum(axis=0))))        # b's first k columns

    best = (0.0, None)
    for k in range(MIN_OV, min(MAX_OV, wa, wb)):
        inter = C[wa - k]
        union = ca[k] + cb[k] - inter
        q = inter / union if union > 0 else 0.0
        if q > best[0]:
            best = (q, k)
    return best[1], best[0]


def runs(pages, meta):
    """Split the windows into strips: one per unbroken stretch of overlapping windows.
    Windows are grouped by layout and panel first -- the split-screen halves are two
    separate passages that happen to be on screen at the same time.

    A break is not always obvious from the overlap score alone. Where the lesson cuts
    from one chapter to the next it re-renders the passage with its own bar numbering,
    and if the cut lands in a run of empty bars the two windows still line up over most
    of their marks -- only the bar numbers disagree. What gives it away is the step: the
    display advances a fixed number of pixels every time, so a join that needs a
    different advance from the rest of its run is not the same score continuing."""
    order = {}
    for i, (t0, t1, x0, x1, n) in enumerate(meta):
        order.setdefault((n, x0), []).append(i)

    out = []
    for key in sorted(order):
        ids = sorted(order[key], key=lambda i: meta[i][0])
        joins = {}
        for a, b in zip(ids, ids[1:]):
            joins[b] = overlap(pages[a], pages[b])
        good = [k for k, q in joins.values() if k is not None and q >= MIN_Q]
        step = float(np.median(good)) if good else None

        cur, links = [ids[0]], []
        for a, i in zip(ids, ids[1:]):
            A, B = pages[a], pages[i]
            if A.shape == B.shape and _iou(A, B, A.shape[1]) >= DUP_IOU:
                print(f"  window {i:02d} repeats {a:02d} -- dropped")
                continue
            k, q = joins[i]
            why = None
            if k is None or q < MIN_Q:
                why = f"no overlap (q={q:.2f})"
            elif step is not None and abs(k - step) > K_TOL:
                why = f"step {k} not the run's {step:.0f}"
            if why:
                print(f"  {a:02d} -> {i:02d}: {why} -- new strip")
                out.append((key, cur, links))
                cur, links = [i], []
                continue
            cur.append(i)
            links.append(k)
        out.append((key, cur, links))
    return out


def panorama(pages, ids, links):
    xs, x = [0], 0
    for k in links:
        x += pages[ids[len(xs) - 1]].shape[1] - k
        xs.append(x)
    h = max(pages[i].shape[0] for i in ids)
    w = max(x0 + pages[i].shape[1] for x0, i in zip(xs, ids))
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros(w, np.float32)
    for x0, i in zip(xs, ids):
        a = pages[i]
        acc[:a.shape[0], x0:x0 + a.shape[1]] += a
        cnt[x0:x0 + a.shape[1]] += 1
    return acc / np.maximum(cnt, 1)[None, :], xs


def build():
    pages, meta = load()
    print(f"{len(pages)} windows")
    strips = []
    for key, ids, links in runs(pages, meta):
        pan, xs = panorama(pages, ids, links)
        t0 = meta[ids[0]][0]
        t1 = meta[ids[-1]][1]
        strips.append({"layout": key[0], "x0": key[1], "ids": ids,
                       "t0": t0, "t1": t1, "img": pan,
                       "windows": [(meta[i][0], meta[i][1], x) for i, x in zip(ids, xs)]})
        print(f"  strip layout {key[0]} x{key[1]:4d}  windows {ids[0]:02d}-{ids[-1]:02d}"
              f" ({len(ids)})  {t0:6.1f}-{t1:6.1f}s  -> {pan.shape[1]}px"
              + (f"  overlaps {min(links)}-{max(links)}" if links else ""))
    return strips


if __name__ == "__main__":
    from PIL import Image
    for n, s in enumerate(build()):
        a = s["img"]
        np.save(f"{WORK}/strip{n}.npy", a)
        im = Image.fromarray(((1 - a) * 255).astype(np.uint8))
        im.resize((max(1, a.shape[1] // 6), a.shape[0] // 6)).save(
            f"{WORK}/strip{n}_thumb.png")
