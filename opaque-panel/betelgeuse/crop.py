#!/usr/bin/env python3
"""Cut each captured panel down to just its system.

The panel is a sliding window over an engraved sheet, so besides the system it can also
catch the sheet's title header, its page footer, or the tail of the system above. What
identifies the system itself is its staves: five notation lines about 12px apart,
followed by six tab lines about 20px apart. Find that pair and the rest falls away.
"""

import numpy as np

LINE_LEVEL, LINE_COVER = 200, 0.60  # a staff line is dark across most of the width
NOTATION_GAP = (9.5, 14.5)
TAB_GAP = (17.5, 22.5)
CHORD_SPACE = 92  # chord names and the bar number sit this far above the top line
TAIL = 42  # stems and fingering hang below the bottom tab line


def staff_lines(page):
    cov = (page.max(axis=2) < LINE_LEVEL).mean(axis=1)
    hits = [r for r, v in enumerate(cov) if v > LINE_COVER]
    grp = []
    for r in hits:
        if grp and r - grp[-1][-1] <= 2:
            grp[-1].append(r)
        else:
            grp.append([r])
    return [int(np.mean(g)) for g in grp]


def _runs(lines, count, gap):
    """Start indices of every run of `count` lines evenly spaced within `gap`."""
    out = []
    for i in range(len(lines) - count + 1):
        d = np.diff(lines[i:i + count])
        if len(d) and gap[0] <= d.min() and d.max() <= gap[1]:
            out.append(i)
    return out


def system_box(page):
    """(top, bottom) rows of the system on this page, or None if there isn't one."""
    lines = staff_lines(page)
    h = page.shape[0]
    for i in _runs(lines, 5, NOTATION_GAP):
        top = lines[i]
        for j in _runs(lines, 6, TAB_GAP):
            if j >= i + 5 and lines[j] - lines[i + 4] < 180:
                return max(0, top - CHORD_SPACE), min(h, lines[j + 5] + TAIL)
    # the closing system can be notation-only (a bar of rests and the final bar line)
    runs = _runs(lines, 5, NOTATION_GAP)
    if runs:
        i = runs[-1]
        return max(0, lines[i] - CHORD_SPACE), min(h, lines[i + 4] + TAIL)
    return None
