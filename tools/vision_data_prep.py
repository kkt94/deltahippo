#!/usr/bin/env python3
"""Decode ImageNet-R once into a uint8 cache (Resize 256 + CenterCrop 224) with a fixed 80/20 split.

Input: data_vision/imagenet-r/<wnid>/<image> (200 class folders).

Split: per class, a fixed-seed (RandomState(0)) permutation; the first round(0.8 n) images train, the rest test.
Output: data_vision/cache/imr_{train,test}_{x,y}.npy (x: N x 224 x 224 x 3 uint8; y: class index 0..199 in the sorted
wnid order of the folder).
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vision")
SRC = os.path.join(ROOT, "imagenet-r")
OUT = os.path.join(ROOT, "cache")


def load(p):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    s = 256.0 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    wnids = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
    assert len(wnids) == 200, len(wnids)
    rs = np.random.RandomState(0)
    tr, te = [], []
    for c, w in enumerate(wnids):
        fs = sorted(os.listdir(os.path.join(SRC, w)))
        perm = rs.permutation(len(fs))
        k = int(round(0.8 * len(fs)))
        tr += [(os.path.join(SRC, w, fs[i]), c) for i in perm[:k]]
        te += [(os.path.join(SRC, w, fs[i]), c) for i in perm[k:]]
    for nm, lst in (("train", tr), ("test", te)):
        with Pool(48) as pool:
            X = np.stack(pool.map(load, [p for p, _ in lst], chunksize=64))
        y = np.array([c for _, c in lst], dtype=np.int64)
        np.save(os.path.join(OUT, f"imr_{nm}_x.npy"), X)
        np.save(os.path.join(OUT, f"imr_{nm}_y.npy"), y)
        print(nm, X.shape, y.shape, np.bincount(y).min(), np.bincount(y).max(), flush=True)
    with open(os.path.join(OUT, "imr_wnids.txt"), "w") as fh:
        fh.write("\n".join(wnids))


if __name__ == "__main__":
    sys.exit(main())
