#!/usr/bin/env python3
"""Drop what the stitch could not average away: faint ghosts of the video.

Every real mark is normalised against the string lines it sits on, so it reaches full
strength somewhere. A ghost never does. Staff lines survive because each one is a single
blob running the length of the strip, and somewhere along it the alpha hits 1.
"""

import numpy as np

FAINT = 0.70  # a blob that never gets brighter than this is not part of the tab
SPECK = 8     # px
FLOOR = 0.15


def _runs(row):
    d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def label_blobs(mask):
    """8-connected labels, one pass over run-lengths with union-find."""
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        x, y = find(x), find(y)
        if x != y:
            parent[y] = x

    labels = np.zeros(mask.shape, np.int32)
    ps, pe, pl = [], [], []
    for y in range(mask.shape[0]):
        st, en = _runs(mask[y])
        cs, ce, cl = [], [], []
        j = 0  # runs are sorted, so this only ever moves forward
        for s, e in zip(st.tolist(), en.tolist()):
            parent.append(len(parent))
            lab = len(parent) - 1
            while j < len(pe) and pe[j] < s:
                j += 1
            k = j
            while k < len(ps) and ps[k] <= e:
                union(pl[k], lab)
                lab = find(lab)
                k += 1
            labels[y, s:e] = lab
            cs.append(s)
            ce.append(e)
            cl.append(lab)
        ps, pe, pl = cs, ce, cl
    roots = np.array([find(i) for i in range(len(parent))], np.int32)
    return roots[labels.ravel()].reshape(mask.shape), len(parent)


def despeckle(alpha):
    lab, n = label_blobs(alpha >= FLOOR)
    flat, fa = lab.ravel(), alpha.ravel()
    area = np.bincount(flat, minlength=n)
    peak = np.zeros(n, np.float32)
    np.maximum.at(peak, flat, fa)
    keep = (peak >= FAINT) & (area >= SPECK)
    keep[0] = False
    return np.where(keep[flat].reshape(alpha.shape), alpha, 0.0)
