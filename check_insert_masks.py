#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick sanity check for insertion target/socket LangSAM masks."""

import os
import sys

import numpy as np


def check_mask(path, name, max_ratio):
    if not os.path.exists(path):
        print("%s missing: %s" % (name, path))
        return False
    mask = np.load(path).astype(bool)
    pixels = int(np.count_nonzero(mask))
    ratio = float(pixels) / float(mask.size)
    ys, xs = np.where(mask)
    print("%s: shape=%s pixels=%d ratio=%.4f" % (
        name, tuple(mask.shape), pixels, ratio))
    if pixels == 0:
        print("%s ERROR: empty mask" % name)
        return False
    if ratio > max_ratio:
        print("%s ERROR: mask is too large for this insertion object" % name)
        return False
    if len(xs) > 0:
        print("%s bbox: x=[%d,%d] y=[%d,%d]" % (
            name, int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))
    return True


def main():
    target_path = "/mnt/hgfs2/tmp_vision/current_mask.npy"
    anchor_path = "/mnt/hgfs2/tmp_vision/current_anchor_mask.npy"
    ok = True
    ok = check_mask(target_path, "target/cylinder", 0.035) and ok
    ok = check_mask(anchor_path, "anchor/socket", 0.030) and ok
    if not ok:
        return 1
    print("insert masks look plausible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
