#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The host comparison point for the fused GeGLU epilogue: EmbeddingGemma 2's MLP GeGLU (gelu_tanh(gate) * up,
F = 2048) per document, timed three ways on the CPU over the same [rows, 2F] fp32 input:
  numpy   model.py's own expression (single-threaded NumPy, what the hybrid's host path runs)
  native  kernels/embedgemma/host_geglu.c (OpenMP + AVX2, ``native.py build --host-geglu``)
and checked against fp64 (worst |error| / max |reference|). Rows are the document length; the per-document figure
is one call times the 24 layers. Check host contention first; this is a CPU timing.

    python tools/embedgemma_host_geglu_bench.py --dll DIR/egemm_host_geglu.dll [--lengths 128 512 2048] [--reps 20]
"""
import argparse
import ctypes
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

LAYERS, F = 24, 2048


def numpy_geglu(gu):
    x, u = gu[:, :F], gu[:, F:]
    return (0.5 * x * (1.0 + np.tanh(np.float32(math.sqrt(2.0 / math.pi)) * (x + np.float32(0.044715) * x * x * x)))
            ).astype(np.float32) * u


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dll", type=Path, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    lib = ctypes.CDLL(str(a.dll.resolve()))
    lib.host_geglu.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.host_geglu.restype = None
    rng = np.random.default_rng(0)
    for L in a.lengths:
        gu = (rng.standard_normal((L, 2 * F)) * 1.5).astype(np.float32)
        x64, u64 = gu[:, :F].astype(np.float64), gu[:, F:].astype(np.float64)
        ref = 0.5 * x64 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x64 + 0.044715 * x64 ** 3))) * u64
        ff = np.empty((L, F), np.float32)

        def native():
            lib.host_geglu(gu.ctypes.data, L, F, ff.ctypes.data)
            return ff
        for name, fn in (("numpy", lambda: numpy_geglu(gu)), ("native", native)):
            out = fn()   # warm-up, and the accuracy check
            err = float(np.abs(out - ref).max() / np.abs(ref).max())
            ts = []
            for _ in range(a.reps):
                t = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t) * 1000)
            med = statistics.median(ts)
            print(f"L={L:5d} {name:6s}: {med:8.3f} ms/call (min {min(ts):.3f}) -> {med * LAYERS:8.1f} ms per document "
                  f"(24 layers); worst |err| / max {err:.1e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
