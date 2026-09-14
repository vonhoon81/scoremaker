#!/usr/bin/env python3
"""Extract the score systems from an atthedrum drum-cover video into .npy grey pages.

The first **drum** chart here, and like `harrubass/` it is named for a channel rather
than a song: atthedrum (@atthedrum) lays every cover out the same way, so a new video
should need only the PER-SONG block below.

Five-line drum notation on an opaque white panel across the bottom, black ink -- so the
family is `kaiju2`'s, not the translucent one, and a plain threshold separates the ink.
Four things are specific to it:

  * **an opening title card, 26.6s of it.** The video opens on full-screen cover art and
    the panel simply does not exist yet; the fades in and out of that card move more of
    the frame than a page flip does, so probe.py reports flips at 1.8, 6.0, 12.0, 24.0
    and 25.4s that are not flips. MUSIC clamps the scan to where the panel is actually
    up, and is found by asking when the band's median brightness sits at the panel's
    white rather than by eye.
  * **a blue box highlights the beat being played**, sweeping each page. Reducing the
    colour frame with `.max(axis=2)` rather than `.min(axis=2)` disposes of it almost
    for free: the box is (0, 80, 200), so its brightest channel is 200 and it reads as
    light, while the engraving is black in every channel and stays black.
  * **grey noteheads are real.** They are ghost notes, drawn grey in the source, and both
    channel reductions agree on them -- do not chase them as a compositing artifact. The
    only saturated pixels anywhere in the band are the 487 belonging to the blue box.
  * **the panel carries a line of lyrics under the staff.** PANEL_H stops above it: a
    drum chart is read off the notation, and the lyric row is a sixth of the band's
    height, which is a sixth of the printed size for the same page count. Raise PANEL_H
    to ~240 to keep it.

What moves within a page is the blue box and nothing else, and it covers a given pixel
for one beat out of the sixteen a page is on screen, so a per-pixel temporal **median**
erases it and leaves the engraving untouched -- the same argument as kaiju2.
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

# ---------------------------------------------------------------- PER-SONG -----------
# MUSIC is when the white panel is up, not when the video starts -- see the title-card
# note above. `python -c` over the band's median brightness finds it in one pass.
VIDEO = common.video('tuki. - 만찬가(晩餐歌) 드럼커버 ｜ 드럼악보 [A7lL4razoUQ].mp4')
WORK = common.work("sm11")
MUSIC = (26.6, 205.6)
# ------------------------------------------------------------- CHANNEL TEMPLATE -------

W = 1920
PANEL_Y, PANEL_H = 818, 204  # the notation only; the lyric row starts at frame row 1025

STAFF_Y0, STAFF_SPACING, STAFF_N = 95.0, 15.38, 5  # drum staff, in band rows

INK = 150  # the panel is 244 and the engraving is black; nothing sits in between
PANEL_WHITE = 244  # ...and 244 is what the panel prints as: a grey wash behind every
                   # system. Stretched to 255 so the paper is paper.
SCAN_FPS = 5
FLIP = 0.010  # fraction of the ink mask that changes at a page flip. Within a page the
              # blue box alone moves and it is invisible to `.max(axis=2)`, so quiet
              # frames sit at 0.0008 and a flip at 0.157 -- a 200:1 margin.
CLUSTER = 4  # scan frames; a flip cross-fades over two or three of them
MIN_PAGE = 2.0

STAFF_X0, STAFF_X1 = 108, 1897  # the staff's own ends. X1 is 3px past the staff so the
                                # closing barline (frame x 1893-1895) is kept whole: cut
                                # at 1894 it survives as a single column and the
                                # minimum-width guard in stitch.py then discards it,
                                # silently losing every page's closing bar boundary.
MARK_X0 = 34  # the boxed section letter (A/B/C, or a roman numeral) sits left of the
              # staff, above it. Kept, and the staff bridged across to it in stitch.py,
              # because on a drum chart those letters are the song's structure.

# The lyric line under the staff is excluded by PANEL_H on a normal page -- notation
# stops at frame row 1016 and the lyrics start at 1025. But on a page with *no* notation
# below the staff (a multi-bar rest) the engraver raises the lyrics to row 999, well
# inside the band. So the fixed cut is backed up by an adaptive one: below the staff,
# find a clear horizontal gap with ink under it, and drop everything past the gap.
LYRIC_BLANK = 0.002  # row ink density counting as blank
LYRIC_GAP = 6  # rows of gap that separate the lyric line from the notation above it

SAMPLES = 30
PCTL = 50  # the median. A max would keep the box's white interior over any note it
           # covered; a min would keep the box outline on every page.
TRIM_LEAD, TRIM_TAIL = 0.10, 0.08


def frames(fps, t0=None, dur=None):
    """Stream the band as (h, w) uint8 arrays of the *brightest* channel.

    Brightest, not darkest: that is what makes the blue highlight box read as light.
    """
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


def scan():
    """Per-frame-pair change in the panel's ink mask, over MUSIC only."""
    path = f"{WORK}/scan.npy"
    if os.path.exists(path):
        return np.load(path)
    t0, t1 = MUSIC
    out, prev = [], None
    for a in frames(SCAN_FPS, t0, t1 - t0):
        m = a < INK
        if prev is not None:
            out.append(float((m != prev).mean()))
        prev = m
    d = np.array(out)
    os.makedirs(WORK, exist_ok=True)
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


def trim_lyrics(v):
    """Blank the lyric line where it has been raised into the band. See LYRIC_GAP.

    Leaves a page alone unless there is a real gap with ink beneath it, so a page whose
    notation simply reaches the bottom of the band is untouched.
    """
    d = (v < INK).mean(axis=1)
    floor = int(round(STAFF_Y0 + (STAFF_N - 1) * STAFF_SPACING)) + 4
    runs, s = [], None
    for i in range(floor, len(d)):
        if d[i] < LYRIC_BLANK:
            s = i if s is None else s
        elif s is not None:
            runs.append((s, i - 1))
            s = None
    for a, b in reversed(runs):  # the lowest qualifying gap wins
        if b - a + 1 >= LYRIC_GAP and (d[b + 1:] >= LYRIC_BLANK).any():
            v = v.copy()
            v[b + 1:] = 255
            return v
    return v


def render(t0, t1):
    """One page as paper-white greyscale."""
    a = t0 + (t1 - t0) * TRIM_LEAD
    b = t1 - (t1 - t0) * TRIM_TAIL
    stack = np.stack(list(frames(SAMPLES / (b - a), a, b - a))).astype(np.float32)
    v = np.percentile(stack, PCTL, axis=0) * (255.0 / PANEL_WHITE)
    return trim_lyrics(np.clip(v, 0, 255).astype(np.uint8)), len(stack)


if __name__ == "__main__":
    import json
    ps = pages(scan())
    print(f"{len(ps)} pages")
    os.makedirs(f"{WORK}/pages", exist_ok=True)
    os.makedirs(f"{WORK}/png", exist_ok=True)
    with open(f"{WORK}/pages.json", "w") as fh:
        json.dump(ps, fh)
    which = [int(x) for x in sys.argv[1:]] or range(len(ps))
    for i in which:
        t0, t1 = ps[i]
        g, n = render(t0, t1)
        np.save(f"{WORK}/pages/{i:03d}.npy", g)
        Image.fromarray(g).save(f"{WORK}/png/{i:03d}.png")
        print(f"  page {i:03d}  {t0:6.1f}-{t1:6.1f}s  n={n:2d}"
              f"  ink={(g < 170).mean():.4f}", flush=True)
