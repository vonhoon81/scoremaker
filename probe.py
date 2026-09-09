#!/usr/bin/env python3
"""Work out how a lesson video presents its tab, so a pipeline can be adapted to it.

    python probe.py "【TAB】Some Song [abc123].mp4"

Answers the four questions that decide which pipeline to copy, and prints the constants
the copy will need:

  1. where the tab is -- the band's rows, and its six staff lines with their spacing
  2. light panel or dark, so the ink is dark-on-light or light-on-dark
  3. opaque panel or translucent overlay: does the band change *between* page flips? An
     opaque panel only changes where the playhead is; a translucent one shows the
     performance moving behind the ink and changes everywhere
  4. page flips or a smooth scroll, and if it flips, at what times

It also writes sample frames and the band's temporal median into the scratch directory
and prints their paths -- look at those images before trusting any of the numbers.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image

import common

MEDIAN_SAMPLES = 40  # frames spread over the video, for the static-background median
UNIFORM = 0.90  # fraction of a row's columns that must sit near its median to be a line
CONTRAST = 22  # how far a line's median must sit from the rows around it
SCAN_FPS = 5
FLIP_RATIO = 6.0  # a flip must change the ink mask this many times more than a quiet frame
INK_MARGINS = (50, 78, 105, 135)  # how far past the panel's own level counts as ink;
                                  # swept, because the right value varies per video
TRANSLUCENT = 1.0  # mean raw change per frame *inside the panel* above which the
                   # performance is showing through the overlay


def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True).stdout.split()
    num, den = (out[2].split("/") + ["1"])[:2]
    return int(out[0]), int(out[1]), float(num) / float(den), float(out[3])


def band_frames(path, fps, y, h, w, t0=None, dur=None):
    """Stream one channel of a horizontal band as (h, w) uint8 arrays."""
    cmd = ["ffmpeg", "-v", "error"]
    if t0 is not None:
        cmd += ["-ss", f"{t0:.3f}", "-t", f"{dur:.3f}"]
    cmd += ["-i", path, "-vf", f"fps={fps:.4f},crop={w}:{h}:0:{y}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = w * h * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n * 4)
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).max(axis=2)
    p.stdout.close()
    p.wait()


def static_median(path, w, h, dur):
    """The temporal median of the whole frame: everything that never moves.

    The score is the only thing that changes every page, so it averages away and what is
    left is the room, the panel wash and the staff lines -- which is both how the staff
    is found and, for a translucent overlay, the background an extraction subtracts.
    """
    fps = MEDIAN_SAMPLES / max(dur - 2.0, 1.0)
    stack = np.stack(list(band_frames(path, fps, 0, h, w, 1.0, dur - 2.0)))
    return np.median(stack.astype(np.float32), axis=0)


def staff_lines(med):
    """Rows that are a uniform full-width line, grouped, plus the evenly-spaced run."""
    rows = []
    for y in range(1, med.shape[0] - 1):
        row = med[y]
        mid = float(np.median(row))
        if (np.abs(row - mid) < 25).mean() < UNIFORM:
            continue
        near = np.median(med[max(0, y - 9):y + 10])
        if abs(mid - float(near)) >= CONTRAST:
            rows.append((y, mid))
    lines, prev = [], None
    for y, mid in rows:  # a drawn line is one or two rows thick
        if prev is not None and y - prev <= 2:
            lines[-1].append((y, mid))
        else:
            lines.append([(y, mid)])
        prev = y
    centres = [sum(y for y, _ in g) / len(g) for g in lines]

    best = []
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            step = centres[j] - centres[i]
            if step < 6:
                continue
            run = [centres[i], centres[j]]
            while True:
                nxt = [c for c in centres if abs(c - (run[-1] + step)) <= 2.5]
                if not nxt:
                    break
                run.append(nxt[0])
            if len(run) > len(best):
                best = run
    return centres, best


def scan(path, y, h, w, dur, panel, dark_panel, rows):
    """Per-frame change in the band, raw and as an ink mask.

    The raw series answers whether the overlay is translucent: an opaque panel only
    changes where the playhead is, so it sits near zero between flips, while a
    translucent one moves everywhere because the performance shows through.

    Flips have to be read off an ink *mask* instead. On a translucent overlay the raw
    difference is swamped -- the performer moving changes the band three times as much as
    a page flip does -- so a raw diff finds no flips at all in a video that plainly has
    thirty of them. Thresholding at INK_MARGIN past the panel's own level fixes that,
    because the marks are drawn at full strength while the performance behind a
    translucent panel only ever bleeds through dimmed.

    `rows` narrows the mask to the staff and the rhythm marks under it. The headroom
    above, where chord symbols and section labels go, is often printed straight over the
    video and is far too noisy to segment pages on -- including it hides the flips just
    as thoroughly as using the raw signal does.
    """
    r0, r1 = rows
    raw, masks = [], {m: [] for m in INK_MARGINS}
    prev_raw, prev = None, {}
    for f in band_frames(path, SCAN_FPS, y, h, w, 0.0, dur):
        band = f.astype(np.int16)[r0:r1]
        if prev_raw is not None:
            # raw change is measured inside the panel only: the automatic band reaches
            # above the panel's top edge for the headroom that holds chord symbols, and
            # that part is live video even when the panel itself is opaque
            raw.append(float(np.abs(band - prev_raw).mean()))
        for t in INK_MARGINS:
            m = band > panel + t if dark_panel else band < panel - t
            if t in prev:
                masks[t].append(float((m != prev[t]).mean()))
            prev[t] = m
        prev_raw = band
    return np.array(raw), {t: np.array(v) for t, v in masks.items()}


def flip_times(d):
    """When the ink mask jumps, grouped so one cross-fade is one flip."""
    quiet = float(np.median(d)) or 1e-6
    spikes = [i for i, v in enumerate(d) if v > quiet * FLIP_RATIO]
    groups = []
    for i in spikes:  # a flip cross-fades over two or three scan frames
        if groups and i - groups[-1][-1] <= 4:
            groups[-1].append(i)
        else:
            groups.append([i])
    return quiet, [(g[-1] + 1) / SCAN_FPS for g in groups]


def scroll_shift(path, y, h, w, at):
    """Horizontal shift between two frames 0.2s apart, by 1D profile correlation."""
    fs = list(band_frames(path, 10, y, h, w, at, 2.0))
    if len(fs) < 3:
        return None
    a, b = (f.astype(np.float32).sum(axis=0) for f in (fs[0], fs[2]))
    a, b = a - a.mean(), b - b.mean()
    best = (0, -1e18)
    for s in range(-240, 241):
        u = a[max(0, s):w + min(0, s)]
        v = b[max(0, -s):w - max(0, s)]
        if len(u) < w // 2:
            continue
        score = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
        if score > best[1]:
            best = (s, score)
    return best


def main():
    common.utf8_console()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", help="filename in the repo root, or any path")
    ap.add_argument("--name", default="probe", help="scratch directory name")
    args = ap.parse_args()

    path = args.video if os.path.exists(args.video) else common.video(args.video)
    if not os.path.exists(path):
        raise SystemExit(f"no such video: {args.video}")
    work = common.work(args.name)

    w, h, fps, dur = ffprobe(path)
    print(f"{os.path.basename(path)}\n  {w}x{h}  {fps:.3g} fps  {dur:.1f}s\n")

    print("sample frames (look at these):")
    for t in [dur * f for f in (0.02, 0.15, 0.35, 0.55, 0.75, 0.95)]:
        out = os.path.join(work, f"t{int(t):04d}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", path,
                        "-frames:v", "1", out, "-y"], check=False)
        print(f"  {out}")

    med = static_median(path, w, h, dur)
    Image.fromarray(med.astype(np.uint8)).save(os.path.join(work, "static.png"))
    print(f"  {os.path.join(work, 'static.png')}   (temporal median: everything static)\n")

    centres, staff = staff_lines(med)
    print(f"full-width lines at rows: {[round(c, 1) for c in centres]}")
    if len(staff) < 4:
        raise SystemExit("could not find an evenly-spaced staff -- read static.png and "
                         "pass the band by hand")
    step = (staff[-1] - staff[0]) / (len(staff) - 1)
    print(f"staff: {len(staff)} lines from row {staff[0]:.1f}, spacing {step:.2f}"
          f"  ({'top' if staff[0] < h / 2 else 'bottom'} of the frame)")

    y0 = max(0, int(staff[0] - 6 * step))  # headroom for numbers, chords, labels
    y1 = min(h, int(staff[-1] + 4 * step))
    band_h, band_y = y1 - y0, y0
    print(f"band: rows {band_y}..{y1}  (height {band_h})")

    panel = float(np.median(med[int(staff[0]):int(staff[-1])]))
    ink = "dark ink on a light panel" if panel > 128 else "light ink on a dark panel"
    print(f"panel median brightness {panel:.0f} -> {ink}\n")

    mask_rows = (max(0, int(staff[0] - band_y - step * 0.5)),
                 min(band_h, int(staff[-1] - band_y + step * 2.5)))
    print(f"segmenting pages on band rows {mask_rows[0]}..{mask_rows[1]} "
          f"(the staff and the rhythm marks under it)\n")
    raw, sweep = scan(path, band_y, band_h, w, dur, panel, panel <= 128, mask_rows)

    # Answer the scroll question first, and from the shift -- it is the only decisive
    # test. Every video tried so far flips between static pages, including ones that
    # look like a continuous scroll, so this is worth checking rather than assuming.
    scored = sorted(((float(d.max()) / max(float(np.median(d)), 1e-9), t, d)
                     for t, d in sweep.items()), reverse=True)
    ratio, margin, d = scored[0]
    quiet, times = flip_times(d)
    gaps = np.diff([0.0] + times) if len(times) >= 2 else np.array([0.0])
    inside = dur * 0.5
    if len(times) >= 3:
        longest = int(np.argmax(gaps))
        inside = (times[longest - 1] if longest else 0.0) + gaps[longest] * 0.5

    shift, score = scroll_shift(path, band_y, band_h, w, inside) or (None, 0.0)
    print(f"horizontal shift over 0.2s at t={inside:.1f}s: {shift}px "
          f"(correlation {score:.3f})")
    if shift is not None and abs(shift) > 3:
        print("  -> SMOOTH SCROLL: the tab moves between flips. None of the existing\n"
              "     pipelines handle this -- they all assume static pages. You would\n"
              "     register every frame into a panorama instead of every page.\n")
    else:
        print("  -> PAGE FLIPS: the tab is still between flips, as every pipeline "
              "assumes.\n")

    print("ink margin sweep -- a page flip should dwarf a quiet frame:")
    for r, t, dd in sorted(scored, key=lambda x: x[1]):
        q, tm = flip_times(dd)
        print(f"  +/-{t:4d}: quiet {q:.4f}  peak/quiet {r:7.1f}  flips {len(tm)}")
    if ratio >= FLIP_RATIO and len(times) >= 3:
        print(f"\nbest margin {margin}: {len(times)} flips, every "
              f"{np.median(gaps):.1f}s (shortest {gaps.min():.1f}s, longest "
              f"{gaps.max():.1f}s)")
        print(f"  first few at {[round(t, 1) for t in times[:8]]}")
        print("  (spikes at the very start and end are usually a title or outro card,\n"
              "   not a flip -- clamp the scanned range to where the score is on screen)")
    else:
        print(f"\nbest margin {margin} only reaches peak/quiet {ratio:.1f} -- NOT a "
              "usable page split.\n"
              "  Something moves within a page as much as a flip does: a bar highlight,\n"
              "  or the performance behind a translucent panel. Both worked examples\n"
              "  had to segment on something cleaner than a raw threshold:\n"
              "    opaque-panel/kaiju2   diffs a mask of the *engraving* only\n"
              "    translucent-overlay/nanmonee  subtracts the median background first,\n"
              "        then thresholds, reaching 0.004 quiet against 0.06 at a flip")

    rawq = float(np.median(raw))
    print(f"\nraw change inside the panel, per frame: median {rawq:.2f}")
    if rawq < 0.5:
        print("  -> almost nothing moves inside the panel, so it is OPAQUE and a plain\n"
              "     threshold will separate the ink.\n"
              "     Start from opaque-panel/kaiju2 (light panel), creep (dark panel),\n"
              "     or horizon/ if the tab sits on top of the frame.")
    else:
        print("  -> something moves inside the panel. That is EITHER a translucent\n"
              "     overlay (the performance shows through) OR an opaque panel with a\n"
              "     big moving highlight -- this number cannot tell them apart, so look\n"
              "     at the sample frames above and decide:\n"
              "       can you see the guitar through the tab?\n"
              "         yes -> translucent-overlay/nanmonee (or pretender, if the video\n"
              "                jumps around the score for repeats)\n"
              "         no  -> opaque-panel/kaiju2 (light) or creep (dark)")

    print(f"\nconstants for the new extract.py:\n"
          f"  PANEL_Y, PANEL_H = {band_y}, {band_h}\n"
          f"  STAFF_Y0, STAFF_SPACING, STAFF_N = {staff[0] - band_y:.1f}, "
          f"{step:.2f}, {len(staff)}\n"
          f"(measured from the temporal median; check them against static.png, and note\n"
          f" the band is padded {6 * step:.0f}px above the staff for bar numbers, chord\n"
          f" symbols and section labels -- trim it once you can see what is there)")



if __name__ == "__main__":
    main()
