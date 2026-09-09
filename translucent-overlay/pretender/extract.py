#!/usr/bin/env python3
"""Extract the score systems from the Pretender lesson video into .npy grey pages.

The hard part here is that the tab is a *translucent* dark panel across the bottom of the
frame -- the guitar and the room show straight through it -- so unlike the opaque panels
of the other videos there is no clean plate to cut out. Three observations make it
separable anyway:

  * the room barely moves (frames a second apart differ over 1.5% of pixels), so a
    temporal median across the *whole* video is an excellent estimate of everything that
    is not score: panel wash, room, guitar, and the six static tab staff lines. What is
    left after subtracting it is the score's ink, near-perfectly isolated.
  * the staff lines go with the background, since they never move. They are redrawn from
    their measured rows rather than recovered, which also gives a crisper printed line.
  * whatever *does* move -- the fretting hand, the neck drifting under vibrato, and the
    playhead sweeping the bar -- is not in every frame of a page, while the engraving is
    in all of them. So the page composite is a low temporal percentile, not a median:
    at PCTL over SAMPLES frames the neck smear and the playhead both vanish and the
    engraving survives untouched. A median leaves a grey swoosh across the bar numbers.

Pages are static between flips, as in the other videos, so they are found from the jump
in the ink column profile. The practice repeats this arrangement is full of ("12x", "8x")
loop the playhead *within* a page and never send it back to an earlier one, so unlike the
사건의 지평선 video the page sequence still runs straight through the score.
"""

import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # where the source videos live
VIDEO = os.path.join(ROOT, 'Official Hige Dandism - Pretender ｜ 일렉기타 + TAB악보 [unj2QL-yi7Y].mp4')
WORK = "/tmp/sm8"

W = 1920
PANEL_Y, PANEL_H = 812, 268  # the translucent panel, from its top edge to the frame's

STAFF_Y0, STAFF_SPACING, STAFF_N = 119.0, 18.6, 6  # tab lines, in panel rows
STAFF_GREY = 96  # what they are redrawn at; the source draws them lighter than the ink

BG_FPS = 1  # samples per second for the whole-video background median
INK_T = 18  # how far above the background a pixel must sit to count as ink

SCAN_FPS = 5
FLIP = 0.25  # relative change in the ink profile at a page flip; within a page it is <0.2
CLUSTER = 6  # scan frames; a flip cross-fades over two or three of them
MIN_PAGE = 1.2  # seconds
MUSIC_END = 296.4  # the overlay fades out from here, and a faded page would
                   # dominate the low percentile and come out grey

SAMPLES = 30
PCTL = 10  # temporal percentile per page
TRIM_LEAD, TRIM_TAIL = 0.08, 0.06
GAIN = 2.4  # ink scaled by this before being written as paper-white greyscale
INK_ON = 35  # ink at least this strong; matches the 170 grey the stitcher calls dark
BARLINE_SPAN = (85, 105)  # a barline's vertical run covers the staff and nothing more


def frames(fps, t0=None, dur=None):
    """Stream the panel as (h, w) uint8 arrays of the brightest channel."""
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
    """Everything that never moves: panel wash, room, guitar and the staff lines."""
    path = f"{WORK}/bg.npy"
    if os.path.exists(path):
        return np.load(path)
    bg = np.median(np.stack(list(frames(BG_FPS))).astype(np.float32), axis=0)
    os.makedirs(WORK, exist_ok=True)
    np.save(path, bg)
    return bg


def ink(a, bg):
    return np.clip(a.astype(np.float32) - bg - INK_T, 0, None)


def scan(bg):
    path = f"{WORK}/profs.npy"
    if os.path.exists(path):
        return np.load(path)
    P = np.stack([(ink(a, bg) > 25).sum(axis=0).astype(np.float32)
                  for a in frames(SCAN_FPS)])
    np.save(path, P)
    return P


def pages(P):
    """[(t0, t1)] for each page the video holds still on."""
    d = [np.abs(P[i] - P[i - 1]).sum() / max(P[i].sum(), 1) for i in range(1, len(P))]
    flips = [i + 1 for i, v in enumerate(d) if v > FLIP]
    groups = [[flips[0]]]
    for f in flips[1:]:
        if f - groups[-1][-1] <= CLUSTER:
            groups[-1].append(f)
        else:
            groups.append([f])
    bounds = [0] + [g[-1] for g in groups] + [len(P)]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        t0, t1 = a / SCAN_FPS, min(b / SCAN_FPS, MUSIC_END)
        if t1 - t0 >= MIN_PAGE:
            out.append((t0, t1))
    return out


def staff_rows():
    return [STAFF_Y0 + i * STAFF_SPACING for i in range(STAFF_N)]


def staff_top(v):
    """The row this page's staff starts on, from where its barlines start.

    The overlay is not always at the same height -- it sits 5px lower for the opening
    page -- and the staff lines cannot be measured directly because they belong to the
    background that was subtracted away. Barlines can: they span the staff exactly,
    which the rhythm stems below it do not, so the most common start row among runs of
    staff height is the top line.
    """
    mask = v > INK_ON
    pad = np.zeros((1, mask.shape[1]), np.int8)
    d = np.diff(np.concatenate([pad, mask.view(np.int8), pad]), axis=0)
    counts = {}
    for x in range(mask.shape[1]):
        s, e = np.flatnonzero(d[:, x] == 1), np.flatnonzero(d[:, x] == -1)
        for u, w in zip(s, e):
            if BARLINE_SPAN[0] <= w - u <= BARLINE_SPAN[1]:
                counts[int(u)] = counts.get(int(u), 0) + 1
    if not counts:
        return int(round(STAFF_Y0))
    top = max(counts, key=counts.get)
    return top if abs(top - STAFF_Y0) <= 25 else int(round(STAFF_Y0))


def render(t0, t1, bg):
    """One page as paper-white greyscale, staff lines redrawn."""
    a = t0 + (t1 - t0) * TRIM_LEAD
    b = t1 - (t1 - t0) * TRIM_TAIL
    stack = np.stack([ink(f, bg) for f in frames(SAMPLES / (b - a), a, b - a)])
    v = np.percentile(stack, PCTL, axis=0)
    v = np.roll(v, int(round(STAFF_Y0)) - staff_top(v), axis=0)

    out = 255 - np.clip(v * GAIN, 0, 255)
    # Floor, not round: the redrawn line has to sit on the rows the source drew it on,
    # because subtracting the background cancelled the ink of everything crossing it --
    # every barline comes out of the subtraction with a 1px gap at each staff line.
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
