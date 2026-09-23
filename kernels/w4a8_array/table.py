# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-tile table of a W4A8 array sweep's raw JSONL. No hardware.

Per (tile, arm) over every passing run: the NPU-bracket averages of each run, their mean,
GOPS from that mean, time relative to the same tile's upstream arm, and cycles per kernel
call per core at the measured 1.80 GHz (DERIVED: mean us x 1,800 / calls per core, where a
core makes (M/m)(N/n)/16 x K/k calls). "resolved" says whether the arm's range of run
averages is disjoint from upstream's at that tile -- with 3 runs a side, a difference
inside the overlap is not a difference this data can report.

Usage:
    python kernels/w4a8_array/table.py results/aie/w4a8_array_raw.jsonl [--tags main,tile128]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

CLOCK_MHZ = 1800  # results/aie/clock_probe_npu.log, power mode Default


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl")
    ap.add_argument("--tags", help="comma list of tag prefixes to include (default: all)")
    args = ap.parse_args(argv)
    prefixes = [t for t in (args.tags or "").split(",") if t]

    rows = []
    with open(args.jsonl, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                if r.get("kind") == "run" and r.get("verdict") == "PASS" and r.get("npu_us"):
                    if not prefixes or any(r["tag"].startswith(p) for p in prefixes):
                        rows.append(r)
    if not rows:
        print("no passing runs")
        return 1

    groups: dict = {}
    for r in rows:
        tile = (r["M"], r["K"], r["N"], r["m"], r["k"], r["n"])
        groups.setdefault(tile, {}).setdefault((r["arm"], r["mode"], r["c_single_buffer"]), []).append(r)

    hashes = {r["c_sha256"] for r in rows}
    print(f"{len(rows)} passing runs; C sha256 values: {sorted(hashes)}")
    for tile, arms in groups.items():
        M, K, N, m, k, n = tile
        calls = (M // m) * (N // n) // 16 * (K // k)
        ops = 2.0 * M * K * N
        up = arms.get(("upstream", "default", 1))
        up_avgs = [r["npu_us"][0] for r in up] if up else None
        up_mean = statistics.mean(up_avgs) if up else None
        print(f"\n{M}x{K}x{N}  tile {m}/{k}/{n}  ({calls:,} kernel calls per core)")
        print(f"  {'arm':<20} {'B':<4} {'runs':>4}  {'run averages (us)':<24} {'mean':>7} "
              f"{'GOPS':>8} {'vs up':>6} {'cyc/call':>8}  resolved")
        order = ["upstream", "i8", "unpack", "native"]
        for key in sorted(arms, key=lambda a: (order.index(a[0]), a[1], -a[2])):
            rs = arms[key]
            avgs = [r["npu_us"][0] for r in rs]
            mean = statistics.mean(avgs)
            if up and key != ("upstream", "default", 1):
                disjoint = max(avgs) < min(up_avgs) or min(avgs) > max(up_avgs)
                res = "yes" if disjoint else "no (overlaps upstream)"
            else:
                res = "-"
            name = f"{key[0]}:{key[1]}" + ("" if key[2] else " C x2")
            print(f"  {name:<20} {'int4' if rs[0]['b_packed'] else 'int8':<4} {len(rs):>4}  "
                  f"{' '.join(f'{a:.1f}' for a in avgs):<24} {mean:>7.1f} {ops / (1000 * mean):>8.2f} "
                  f"{(up_mean / mean) if up_mean else float('nan'):>5.3f}x {mean * CLOCK_MHZ / calls:>8,.0f}  {res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
