#!/usr/bin/env python3
"""Extract the score systems from the Nanmonee lesson video into .npy grey pages.

The tab is a translucent dark panel across the bottom of the frame -- the guitar shows
through it, dimmed to about a fifth of its brightness -- and the chord symbols and
section labels sit *above* the panel directly over the live video, where they are barely
readable against the guitar's white body. So the band worth keeping runs from well above
the panel's top edge down to the frame, and none of it has a clean plate behind it.

The separation is the same one that worked for Pretender:

  * the room barely moves, so a temporal median across the video is a good estimate of
    everything that is not score -- room, guitar, panel wash, and the six static tab
    staff lines. Subtracting it isolates the ink.
  * the staff lines go with the background, since they never move, and are redrawn from
    their measured rows.
  * what does move -- the fretting hand and the red playhead sweeping the bar -- is not
    in every frame of a page, while the engraving is in all of them. So a page is
    composited at a low temporal percentile rather than a median, which drops the hand
    and the playhead and leaves the engraving untouched.

Pages are static between flips and the video runs straight through the score once, so
flips come from the jump in the ink column profile with no jump handling needed.
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
VIDEO = os.path.join(ROOT, '【TAB】Nanmonee (なんもねえ) ⧸ Wasureranneyo (忘れらんねえよ) ｜ Yani Neko  [iUpm4Xa2gM4].mp4')
WORK = common.work("sm9")

W = 1920
PANEL_Y, PANEL_H = 700, 380  # from above the section labels down to the frame's edge

STAFF_Y0, STAFF_SPACING, STAFF_N = 160.0, 18.6, 6  # tab lines, in band rows
STAFF_GREY = 96  # what they are redrawn at; the source draws them lighter than the ink

MUSIC = (3.2, 171.6)  # the score is on screen between these; a stage card book-ends it
BG_FPS = 1
INK_T = 20

SCAN_FPS = 5
SCAN_ROWS = (150, 330)  # the staff and the rhythm slashes under it. The chord band above
                        # sits over live video and is far too noisy to segment pages on.
INK_SCAN = 40  # a mark, once the background is gone, is much brighter than the residue
FLIP = 0.02  # fraction of the panel's ink mask that changes at a page flip. Inside a
             # page it is 0.004 (the playhead moving), at a flip 0.06 and up.
CLUSTER = 6  # scan frames; a flip cross-fades over three of them
MIN_PAGE = 1.2

SAMPLES = 30
PCTL = 5
TRIM_LEAD, TRIM_TAIL = 0.10, 0.08
# The marks are white, so how much ink a pixel can possibly show is 255 minus whatever
# is behind it -- over the dimmed panel that is nearly the full range, but a chord symbol
# printed over the guitar's white body has barely 45 levels to work with and comes out a
# pale grey at any fixed gain. Ink is therefore divided by the headroom the background
# left it, which recovers those labels without touching the ones over the dark panel.
HEADROOM_MIN = 45
BOOST = 1.3


def frames(fps, t0=None, dur=None):
    """Stream the band as (h, w) uint8 arrays of the brightest channel."""
    cmd = ["ffmpeg", "-v", "error"]
    if t0 is not None:
        cmd += ["-ss", f"{t0:.3f}", "-t", f"{dur:.3f}"]
    cmd += ["-i", VIDEO, "-vf", f"fps={fps:.4f},crop={W}:{PANEL_H}:0:{PANEL_Y}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = W * PANEL_H * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n * 4)
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(PANEL_H, W, 3).max(axis=2)
    p.stdout.close()
    p.wait()


def background():
    """Everything that never moves: room, guitar, panel wash and the staff lines."""
    path = f"{WORK}/bg.npy"
    if os.path.exists(path):
        return np.load(path)
    t0, t1 = MUSIC
    stack = np.stack(list(frames(BG_FPS, t0, t1 - t0))).astype(np.float32)
    bg = np.median(stack, axis=0)
    os.makedirs(WORK, exist_ok=True)
    np.save(path, bg)
    return bg


def ink(a, bg):
    return np.clip(a.astype(np.float32) - bg - INK_T, 0, None)


def scan(bg):
    """Per-frame-pair change in the panel's ink mask."""
    path = f"{WORK}/scan.npy"
    if os.path.exists(path):
        return np.load(path)
    r0, r1 = SCAN_ROWS
    t0, t1 = MUSIC
    out, prev = [], None
    for a in frames(SCAN_FPS, t0, t1 - t0):
        m = ink(a, bg)[r0:r1] > INK_SCAN
        if prev is not None:
            out.append(float((m != prev).mean()))
        prev = m
    d = np.array(out)
    np.save(path, d)
    return d


def pages(d):
    """[(t0, t1)] for each page the video holds still on, in video time."""
    flips = [i + 1 for i, v in enumerate(d) if v > FLIP]
    groups = [[flips[0]]]
    for f in flips[1:]:
        if f - groups[-1][-1] <= CLUSTER:
            groups[-1].append(f)
        else:
            groups.append([f])
    bounds = [0] + [g[-1] for g in groups] + [len(d) + 1]
    base = MUSIC[0]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        t0, t1 = base + a / SCAN_FPS, min(base + b / SCAN_FPS, MUSIC[1])
        if t1 - t0 >= MIN_PAGE:
            out.append((t0, t1))
    return out


def staff_rows():
    return [STAFF_Y0 + i * STAFF_SPACING for i in range(STAFF_N)]


def render(t0, t1, bg):
    """One page as paper-white greyscale, staff lines redrawn."""
    a = t0 + (t1 - t0) * TRIM_LEAD
    b = t1 - (t1 - t0) * TRIM_TAIL
    stack = np.stack([ink(f, bg) for f in frames(SAMPLES / (b - a), a, b - a)])
    v = np.percentile(stack, PCTL, axis=0)

    headroom = np.clip(255.0 - bg, HEADROOM_MIN, 255.0)
    out = 255 - np.clip(v * (255.0 / headroom) * BOOST, 0, 255)
    for y in staff_rows():
        lo = int(y)
        out[lo:lo + 2] = np.minimum(out[lo:lo + 2], STAFF_GREY)
    return out.astype(np.uint8), len(stack)


if __name__ == "__main__":
    import json
    bg = background()
    ps = pages(scan(bg))
    print(f"{len(ps)} pages")
    os.makedirs(f"{WORK}/pages", exist_ok=True)
    os.makedirs(f"{WORK}/png", exist_ok=True)
    with open(f"{WORK}/pages.json", "w") as fh:
        json.dump(ps, fh)
    which = [int(x) for x in sys.argv[1:]] or range(len(ps))
    for i in which:
        t0, t1 = ps[i]
        g, n = render(t0, t1, bg)
        np.save(f"{WORK}/pages/{i:03d}.npy", g)
        Image.fromarray(g).save(f"{WORK}/png/{i:03d}.png")
        print(f"  page {i:03d}  {t0:6.1f}-{t1:6.1f}s  n={n:2d}"
              f"  ink={(g < 170).mean():.4f}", flush=True)
