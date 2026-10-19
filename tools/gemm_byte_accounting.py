#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DDR byte accounting for EmbeddingGemma 2's forward-pass weight GEMMs on kernels/gemm_generic/whole_array_generic.py.

The design (upstream's whole-array GEMM, 4 x 4 cores) computes C in blocks of (rows*m) x (cols*n). For each block it
streams A's rows*m rows over all of K (broadcast along an array row) and B's K x cols*n strip (broadcast down an
array column), so for one M x K x N product:
    A bytes = M K a_bytes * N / (cols n)      B bytes = K N b_bytes * M / (rows m)      C bytes = M N c_bytes
The minimum is each operand once. This tool tabulates both per shape and per pass, and, given a probe log of the
hollow build (no MACs), the effective DDR rate each shape reached and its fixed share (--fixed-ms per call, the
fit's 0.34 ms). Nothing here is measured except what the log carries; the byte model is DERIVED from the design.

    python tools/gemm_byte_accounting.py --log results/aie/<hollow probe>.log [--m 64] [--M 2048] [--w8]
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedgemma_gemm_probe as probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, help="probe run log with per-shape 'MxKxN: median X ms/call' lines")
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--w8", action="store_true", help="B as W8 tiles: (64*64 + 4*64) bytes per 64x64 tile")
    ap.add_argument("--c-bytes", type=int, default=4)
    ap.add_argument("--fixed-ms", type=float, default=0.34)
    a = ap.parse_args()
    cnt = Counter((K, N) for _, K, N in probe.forward(probe.CONFIG))
    ms = {}
    if a.log:
        for M_, K, N, t in re.findall(r"(\d+)x(\d+)x(\d+): median ([\d.]+) ms/call", a.log.read_text(encoding="utf-8")):
            if int(M_) == a.M:
                ms[(int(K), int(N))] = float(t)
    M = a.M
    bb = (64 * 64 + 4 * 64) / (64 * 64) if a.w8 else 2.0
    rowsm, colsn = a.rows * a.m, a.cols * a.n
    print(f"M={M}, tile m={a.m} n={a.n}, {a.rows}x{a.cols} cores: block {rowsm} x {colsn}; B {bb:.3f} bytes/weight")
    print("| K x N | calls | A MB (x reads) | B MB (x reads) | C MB | total MB | min MB | x min | hollow ms | GB/s | GB/s ex fixed |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot = dict(a=0.0, b=0.0, c=0.0, mn=0.0, t=0.0, tx=0.0)
    for (K, N), c in sorted(cnt.items()):
        Np = probe.padded(N)
        ra, rb = Np / colsn, M / rowsm
        A, B, C = M * K * 2 * ra, K * Np * bb * rb, M * Np * a.c_bytes
        mn = M * K * 2 + K * Np * bb + M * Np * a.c_bytes
        t = ms.get((K, N))
        rate = (A + B + C) / (t * 1e-3) / 1e9 if t else None
        ratex = (A + B + C) / ((t - a.fixed_ms) * 1e-3) / 1e9 if t and t > a.fixed_ms else None
        print(f"| {K} x {N} | {c} | {A / 1e6:.1f} ({ra:g}) | {B / 1e6:.1f} ({rb:g}) | {C / 1e6:.1f} | {(A + B + C) / 1e6:.1f} | "
              f"{mn / 1e6:.1f} | {(A + B + C) / mn:.2f} | {t if t else '-'} | {f'{rate:.1f}' if rate else '-'} | "
              f"{f'{ratex:.1f}' if ratex else '-'} |")
        for k, v in (("a", A), ("b", B), ("c", C), ("mn", mn)):
            tot[k] += c * v
        if t:
            tot["t"] += c * t
            tot["tx"] += c * (t - a.fixed_ms)
    total = tot["a"] + tot["b"] + tot["c"]
    print(f"\nper pass: A {tot['a'] / 1e6:.0f} MB, B {tot['b'] / 1e6:.0f} MB, C {tot['c'] / 1e6:.0f} MB = {total / 1e6:.0f} MB; "
          f"minimum {tot['mn'] / 1e6:.0f} MB ({total / tot['mn']:.2f}x); calls {sum(cnt.values())}")
    if tot["t"]:
        print(f"hollow time per pass {tot['t']:.1f} ms -> {total / tot['t'] / 1e6:.1f} GB/s; without {a.fixed_ms} ms x "
              f"{sum(cnt.values())} calls fixed: {tot['tx']:.1f} ms -> {total / tot['tx'] / 1e6:.1f} GB/s")
        print(f"at that rate, the minimum bytes would take {tot['mn'] * tot['tx'] / total:.1f} ms "
              f"+ fixed {a.fixed_ms * sum(cnt.values()):.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
