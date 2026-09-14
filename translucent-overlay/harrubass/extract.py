#!/usr/bin/env python3
"""Extract the score systems from a 하루베이스 bass lesson into .npy pages.

Unlike every other directory here this one is named for a **channel**, not a song:
하루베이스 (@harrubass, youtube.com/channel/UCzGjV-2jL1iLZLojqubPygw) numbers its uploads
("405. ...") and uses one layout across all of them, so a new video from it should need
nothing but the PER-SONG block below changed. Everything under it was measured on
`TFnx91Cfj04` and is a property of the channel's template. Re-measure only if a video
turns out to be framed differently -- the bar-timing table stitch.py prints will say so.

A four-line **bass** tab, so the staff is 4 lines at 26.6px rather than the 6 at 18-19px
every guitar chart here has. It is drawn straight over the live video with no panel
behind it at all -- but the studio is black, so the background the white ink sits on is
already near zero and the ink separates almost perfectly. That is what makes this the
easiest of the translucent family: probe.py's ink sweep reaches 208:1 between a page flip
and a quiet frame, where Nanmonee managed 4.3:1 and needed the background gone first.

The background is still subtracted, for the two things it buys that a plain threshold
does not:

  * the four staff lines never move, so they go with the background and are redrawn from
    their measured rows -- crisper in print, and it is what the barline-span test in
    stitch.py expects to find.
  * the player is lit in a cream hoodie against a white bass, and when either drifts into
    the band it is far brighter than the ink. The median over the whole song puts it in
    the background where it belongs.

What moves within a page -- the fretting hand, and the salmon-coloured playhead bar
sweeping the beat -- is not in every frame of a page while the engraving is, so a page is
composited at a low temporal percentile and both drop out without ever being detected.

Chord symbols sit above the staff over live video, as in Nanmonee; the band keeps them.
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
# The only three things a new video from this channel should need. MUSIC is read off the
# tab's first and last appearance (the ink sweep in probe.py, or ffplay and a stopwatch);
# WORK just has to be a scratch name nothing else uses.
#
# NOTE: yt-dlp wrote this name in NFD -- the Korean syllables are stored decomposed, so a
# hand-typed (NFC) literal will not open it. Keep the bytes as they are, or glob the id.
VIDEO = common.video('405. Orangestar - 내일의 밤하늘 초계반 (Hebi.cover ) 【★★★★☆】 (Bass Cover) ｜ 베이스 악보[TAB] [TFnx91Cfj04].mp4')
WORK = common.work("sm10")
MUSIC = (6.2, 179.0)  # the tab is on screen between these; a title card and an outro
                      # book-end it, and both spike the page-flip detector
# ------------------------------------------------------------- CHANNEL TEMPLATE -------

W = 1920
PANEL_Y, PANEL_H = 786, 220  # chord symbols at the top down to the ends of the beams

STAFF_Y0, STAFF_SPACING, STAFF_N = 66.4, 26.59, 4  # four bass strings, in band rows
STAFF_GREY = 118  # what they are redrawn at; the source draws them well below ink white
STAFF_X0, STAFF_X1 = 94, 1823  # the staff's own ends, measured off the background

# Every page is a self-contained system: N complete bars justified to fill STAFF_X0..X1,
# with no overlap onto its neighbour (the ties across page edges line up exactly). The
# source draws the barlines *between* those bars but never one at the left edge, and only
# sometimes one at the right, so a page junction can land with no barline at all. That is
# filled in by stitch.py rather than here -- it needs to see both sides of the junction
# to tell a missing line from one the source already drew a few px short of the edge.

BG_FPS = 1
INK_T = 20

SCAN_FPS = 5
SCAN_ROWS = (60, 215)  # the staff and the rhythm stems under it. The chord band above
                       # sits over live video and is too noisy to segment pages on.
INK_SCAN = 40
FLIP = 0.02  # fraction of the ink mask that changes at a page flip. Within a page it is
             # the playhead only, ~0.004; a flip clears 0.09 and up.
CLUSTER = 6  # scan frames; a flip cross-fades over three of them
MIN_PAGE = 1.2

SAMPLES = 30
PCTL = 5
TRIM_LEAD, TRIM_TAIL = 0.10, 0.08
# White ink again, so a pixel can only show 255 minus whatever is behind it. Over the
# black studio that is the full range and the division is a no-op; it earns its keep on
# the few pages where the bass body drifts under a chord symbol.
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
    """Everything that never moves: the room, the player, and the four staff lines."""
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
    """Per-frame-pair change in the band's ink mask."""
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
    out[:, :STAFF_X0] = 255  # the redrawn staff must not run past the real one
    out[:, STAFF_X1:] = 255
    rows = staff_rows()
    for y in rows:
        lo = int(y)
        out[lo:lo + 2, STAFF_X0:STAFF_X1] = np.minimum(
            out[lo:lo + 2, STAFF_X0:STAFF_X1], STAFF_GREY)
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
