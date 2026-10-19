#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bit-for-bit comparison of two GEMM builds' outputs on the same seeded inputs (tools/embedgemma_gemm_probe.py run
--save-outputs, plus each build directory's epilogue_<M>.npz).

Used to check that an owned matmul (kernels/gemm_bf16/mm_bf16.cc) reproduces upstream mm.cc exactly: every key
present in both files must have identical bytes. Prints per key: equal, or the count of differing words and the
largest difference in float32 ulps. Exit 0 only when every compared key is bit-equal.

    python tools/gemm_bitdiff.py REF.npz NEW.npz [--epilogue REF_DIR NEW_DIR --modes 0 6]
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def diff(name, x, y):
    if x.shape != y.shape:
        print(f"{name}: SHAPE {x.shape} vs {y.shape}")
        return False
    a, b = x.view(np.uint32), y.view(np.uint32)
    bad = a != b
    if not bad.any():
        print(f"{name}: bit-equal ({a.size} words)")
        return True
    ulp = np.abs(a.astype(np.int64)[bad] - b.astype(np.int64)[bad]).max()
    print(f"{name}: {int(bad.sum())} of {a.size} words differ, max {ulp} ulp")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--epilogue", type=Path, nargs=2, metavar=("REF_DIR", "NEW_DIR"))
    ap.add_argument("--modes", type=int, nargs="+", default=[0, 6])
    ap.add_argument("--ms", type=int, nargs="+", default=[256, 512, 2048])
    a = ap.parse_args()
    ok, n = True, 0
    r, w = np.load(a.ref), np.load(a.new)
    for k in sorted(set(r.files) & set(w.files)):
        ok &= diff(k, r[k], w[k])
        n += 1
    if a.epilogue:
        for M in a.ms:
            fr, fw = a.epilogue[0] / f"epilogue_{M}.npz", a.epilogue[1] / f"epilogue_{M}.npz"
            er, ew = np.load(fr), np.load(fw)
            for md in a.modes:
                ok &= diff(f"epilogue {M} mode {md}", er[f"c{md}"], ew[f"c{md}"])
                n += 1
    print(f"{'ALL BIT-EQUAL' if ok else 'DIFFERENT'} over {n} outputs")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
