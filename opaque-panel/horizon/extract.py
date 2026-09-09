#!/usr/bin/env python3
"""Extract the score systems from the 사건의 지평선 lesson video into .npy colour pages.

This one puts the score in the *top* half of the frame -- notation staff, Korean lyrics,
chord symbols and tab staff, about 430px of it -- with the performance below. A thin
translucent blue playhead sweeps across the page and, when it reaches the last bar, the
page jumps forward. Every page therefore shows a partial bar at *both* edges: the tail of
the bar before it and the head of the bar after, with three complete bars in between.
stitch.py keeps only the complete ones, which is what makes the apparent repeat of the
last bar at each scroll drop out.

Two things here that the other videos did not have:

  * the panel's vertical position is not fixed. It sits 20px higher from t=140 on, so
    each page is cropped relative to its own detected notation staff, not to the frame.
  * the playhead jumps backwards twice for different reasons. A normal page flip moves it
    back to the left edge; but at t=140 the score's repeat sign sends it from bar 56 all
    the way back to bar 26, so the video covers bars 26-56 twice. Both look identical
    here -- telling them apart is stitch.py's job, from the bar images.

Scanning streams raw frames through a pipe rather than writing PNGs: the playhead is a
single translucent column, so the scan cannot be done at reduced resolution, and 1500
full-width frames are not worth putting on disk.
"""

import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)  # the repo root, for common.py and the source videos

import common  # noqa: E402
VIDEO = os.path.join(ROOT, '사건의 지평선 - 윤하 [ 일렉기타 입문곡 시리즈 #022 ] [imfG3LKUxQ0].mp4')
WORK = common.work("sm7")

W, PANEL_H = 1920, 444  # the score panel, pinned to the top of the frame

SCAN_FPS = 5
BLUE = 18  # playhead columns run this much bluer than they are red or green
BLUE_ROWS = 80  # ...over at least this many rows of the panel
STAFF_TH, STAFF_COVER = 210, 0.9  # a staff line is grey across the whole width
BACK_JUMP = 200  # px the playhead must retreat for a page flip
CLUSTER = 6  # scan frames; a flip can look like two retreats in a row
MIN_PAGE = 1.0  # seconds
MUSIC_END = 296.9  # the panel cuts to a black end card here

SAMPLES = 15  # full-res frames medianed per page
TRIM_LEAD, TRIM_TAIL = 0.15, 0.12

ABOVE, BELOW = 33, 405  # rows kept above / below the notation staff's top line. 33 is all the
# frame has above the staff on the first pass; the second pass sits 20px higher than that
# and the frame clips it, so those pages come out padded with white instead.


def frames(fps, t0=None, dur=None, n_expected=None):
    """Stream panel crops as (h, w, 3) uint8 arrays."""
    cmd = ["ffmpeg", "-v", "error"]
    if t0 is not None:
        cmd += ["-ss", f"{t0:.3f}", "-t", f"{dur:.3f}"]
    cmd += ["-i", VIDEO, "-vf", f"fps={fps:.4f},crop={W}:{PANEL_H}:0:0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = W * PANEL_H * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n * 4)
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(PANEL_H, W, 3)
    p.stdout.close()
    p.wait()


def playhead(a):
    """(x, coverage) of the bluest column, or (-1, 0) if the playhead is absent."""
    b = a[..., 2].astype(np.int16) - np.maximum(a[..., 0], a[..., 1])
    cnt = (b > BLUE).sum(axis=0)
    x = int(cnt.argmax())
    return (x, int(cnt[x])) if cnt[x] >= BLUE_ROWS else (-1, 0)


def staff_top(a):
    """Row of the notation staff's top line, the reference for vertical alignment."""
    rows = np.flatnonzero((a.min(axis=2) < STAFF_TH).mean(axis=1) > STAFF_COVER)
    return int(rows[0]) if len(rows) else -1


def scan():
    return [(playhead(a)[0], staff_top(a)) for a in frames(SCAN_FPS)]


def pages(s):
    """[(t0, t1, staff top row)] for each page the video holds still on."""
    cuts, prev = [], None
    for i, (x, _) in enumerate(s):
        if x < 0:
            continue
        if prev is None or x < prev - BACK_JUMP:
            cuts.append(i)
        prev = x
    keep = [cuts[0]]
    for c in cuts[1:]:  # a flip can register as two retreats; the last one is the page
        if c - keep[-1] <= CLUSTER:
            keep[-1] = c
        else:
            keep.append(c)

    out = []
    bounds = keep + [len(s)]
    for a, b in zip(bounds, bounds[1:]):
        t0, t1 = a / SCAN_FPS, min(b / SCAN_FPS, MUSIC_END)
        if t1 - t0 < MIN_PAGE:
            continue
        tops = [t for x, t in s[a:b] if x >= 0 and t >= 0]
        out.append((t0, t1, int(np.median(tops))))
    return out


def render(t0, t1, top):
    """Median of the page's frames, cropped to a fixed window around the staff."""
    a, b = t0 + (t1 - t0) * TRIM_LEAD, t1 - (t1 - t0) * TRIM_TAIL
    fps = SAMPLES / max(b - a, 0.2)
    stack = np.stack(list(frames(fps, a, b - a)))
    med = np.median(stack, axis=0).astype(np.uint8)

    out = np.full((ABOVE + BELOW, W, 3), 255, np.uint8)
    src0, src1 = max(0, top - ABOVE), min(PANEL_H, top + BELOW)
    out[src0 - (top - ABOVE):src1 - (top - ABOVE)] = med[src0:src1]
    return out, len(stack)


if __name__ == "__main__":
    import json
    cache = f"{WORK}/scan.npy"
    if os.path.exists(cache):
        s = [tuple(r) for r in np.load(cache)]
    else:
        s = scan()
        os.makedirs(WORK, exist_ok=True)
        np.save(cache, np.array(s))
    ps = pages(s)
    print(f"{len(s)} scan frames, {len(ps)} pages")
    os.makedirs(f"{WORK}/pages", exist_ok=True)
    os.makedirs(f"{WORK}/png", exist_ok=True)
    with open(f"{WORK}/pages.json", "w") as fh:
        json.dump(ps, fh)
    which = [int(x) for x in sys.argv[1:]] or range(len(ps))
    for i in which:
        t0, t1, top = ps[i]
        g, n = render(t0, t1, top)
        np.save(f"{WORK}/pages/{i:03d}.npy", g)
        Image.fromarray(g).save(f"{WORK}/png/{i:03d}.png")
        print(f"  page {i:03d}  {t0:6.1f}-{t1:6.1f}s  top={top:3d}  n={n:2d}", flush=True)
