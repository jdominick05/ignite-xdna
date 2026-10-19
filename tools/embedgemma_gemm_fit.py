#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where a generic-GEMM forward pass's time goes, from tools/embedgemma_gemm_probe.py logs (m = 64 tiles, M = 2048).

Per (K, N) shape, fits   real call = d + alpha * core estimate,   core estimate = calls per core x cycles per call
from the L1-resident bench (results/aie/l1_tile_mm_direct_64x64x64_*: mm.cc 64x64x64 bf16 5427.4 cycles; W8 adds
--dequant-cycles, the dequant's static loop estimate), and   hollow call = d_h + beta * DERIVED bytes per call
(A re-read per 256-column block, B per 256-row block, C in f32). alpha ~ 1 with a small residual says the run is
core-bound; d is the fixed cost per dispatch (host sync, call, pipeline fill and drain).

    python tools/embedgemma_gemm_fit.py REAL_BF16 HOLLOW_BF16 REAL_W8 HOLLOW_W8 [--dequant-cycles 1504]
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results" / "aie"
MPC, CORES, HZ, M = 64 * 64 * 64, 16, 1.8e9, 2048
sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedgemma_gemm_probe as p  # noqa: E402


def shapes(log):
    txt = (R / log).read_text(encoding="utf-8", errors="replace")
    return {(int(a), int(b)): float(c) for a, b, c in re.findall(rf"    {M}x(\d+)x(\d+): median ([0-9.]+) ms/call", txt)}


def fit(x, y):
    X = np.stack([np.ones_like(x), x], 1)
    (d, a), *_ = np.linalg.lstsq(X, y, rcond=None)
    return d, a, float(np.abs(X @ [d, a] - y).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for n in ("real_bf16", "hollow_bf16", "real_w8", "hollow_w8"):
        ap.add_argument(n, help="log name under results/aie")
    ap.add_argument("--dequant-cycles", type=float, default=1504.0)
    a = ap.parse_args()
    cnt = Counter((K, N) for _, K, N in p.forward(p.CONFIG))
    for name, real_log, hol_log, cyc, bpe in [("bf16", a.real_bf16, a.hollow_bf16, 5427.4, 2),
                                              ("w8", a.real_w8, a.hollow_w8, 5427.4 + a.dequant_cycles, 4352 / 4096)]:
        real, hol = shapes(real_log), shapes(hol_log)
        ks = sorted(real)
        Np = {k: -(-k[1] // 256) * 256 for k in ks}
        comp = np.array([M * K * Np[(K, N)] / MPC / CORES * cyc / HZ * 1000 for K, N in ks])
        byts = np.array([(M * K * 2 * (Np[(K, N)] // 256) + K * Np[(K, N)] * bpe * (M // 256) + M * Np[(K, N)] * 4)
                         / 2**20 for K, N in ks])
        y, yh = np.array([real[k] for k in ks]), np.array([hol[k] for k in ks])
        d, al, e = fit(comp, y)
        dh, b, eh = fit(byts, yh)
        w = np.array([cnt[k] for k in ks])
        smax = float((w * np.maximum(yh, comp)).sum())
        print(f"{name}: real call = {d:.3f} ms + {al:.3f} x L1 core estimate (worst residual {e:.3f} ms); "
              f"hollow call = {dh:.3f} ms + {b * 1000:.2f} us/MiB (worst residual {eh:.3f} ms)")
        print(f"   per forward pass: fixed {d * w.sum():.1f} ms over {w.sum()} calls, core {al * (w * comp).sum():.1f}, "
              f"transport excl. fixed {b * (w * byts).sum():.1f}; real {(w * y).sum():.1f}, "
              f"sum of per-shape max(hollow, core estimate) {smax:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
