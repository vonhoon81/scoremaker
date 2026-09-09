#!/usr/bin/env python3
"""Re-run the blob filter over already-extracted pages (skips video decoding)."""

import glob

import numpy as np

from extract import FLOOR, drop_video_leftovers

for f in sorted(glob.glob("/tmp/sm/pages/*.npy")):
    a = np.load(f)
    before = (a >= FLOOR).sum()
    a = drop_video_leftovers(a)
    after = (a >= FLOOR).sum()
    np.save(f, a)
    cols = np.flatnonzero((a >= FLOOR).any(axis=0))
    print(f"{f[-7:-4]}  removed {before - after:6d} px  cols {cols[0]}-{cols[-1]}")
