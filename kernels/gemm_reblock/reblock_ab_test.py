"""Alternating A/B: does the reblocked int8 GEMM kernel actually run faster?

The static side is unambiguous (`results/aie/gemm_reblock_h11.log`): switching the int8
path from `matmul_vectorized_4x2_mmul` (8 live accumulators) to `matmul_vectorized_2x2_mmul`
(4) removes every accumulator spill and tightens the inner loop.

    stock 4x2    144 bundles   frame 416 B   33 stack refs   9-bundle loop, 8 vmac (0.889/cyc)
    reblocked    88 bundles    frame  32 B    5 stack refs   8-bundle loop, 8 vmac (1.000/cyc)

Whether that reaches the wall clock is the open question, and this repo's own cost model
predicts it will not: the design is bound by a per-buffer delivery floor of roughly 3,200
cycles that the core already finishes inside (`results/aie/gemm_cost_model_nest.log`).

Each arm gets its OWN `NPU_CACHE_HOME`. That is not tidiness: `@iron.jit` keys its cache on
the design and its compile-time arguments, not on the kernel source, so two arms sharing a
cache silently reuse one xclbin and the second arm never compiles at all. The only symptom is
that no new object appears on disk.

Usage (ironenv, XRT SDK on PATH and pyxrt on PYTHONPATH):
    python kernels/gemm_reblock/reblock_ab_test.py --repeats 4 --iters 40
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build_reblocked.py"

RE_NPU = re.compile(
    r"NPU time\s+\(avg/min/max us\):\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)")
RE_G = re.compile(r"NPU GFLOPS\s*:\s*([\d.]+)")


def one(arm: str, cache: Path, args) -> dict | None:
    cmd = [sys.executable, str(BUILD), "--cache-home", str(cache)]
    if arm == "stock":
        cmd.append("--stock")
    cmd += ["--rest", "--dev", "npu",
            "-M", str(args.M), "-K", str(args.K), "-N", str(args.N),
            "-m", "64", "-k", "64", "-n", "64", "--n-aie-cols", "4",
            "--dtype_in", "i8", "--dtype_out", "i32",
            "--warmup", str(args.warmup), "--iters", str(args.iters)]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE.parents[1])
    out = p.stdout + p.stderr
    m, g = RE_NPU.search(out), RE_G.search(out)
    if not m:
        print(out[-1200:])
        return None
    return {"avg": float(m.group(1)), "min": float(m.group(2)),
            "max": float(m.group(3)),
            "gflops": float(g.group(1)) if g else None, "pass": "PASS!" in out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("-M", type=int, default=4096)
    ap.add_argument("-K", type=int, default=2048)
    ap.add_argument("-N", type=int, default=2048)
    ap.add_argument("--cache-root", default=None,
                    help="where the two private caches live (default: beside this file)")
    args = ap.parse_args(argv)

    root = Path(args.cache_root).resolve() if args.cache_root else HERE / "_abcache"
    caches = {"stock": root / "stock", "reblock": root / "reblock"}
    for c in caches.values():
        c.mkdir(parents=True, exist_ok=True)

    print(f"shape {args.M}x{args.K}x{args.N}  tile 64/64/64  4 columns  int8->int32")
    print(f"{args.repeats} alternating repeats per arm, "
          f"{args.warmup} warmup + {args.iters} timed iterations each")
    print(f"caches: {root}")
    print()
    print(f"{'rep':>4} {'arm':<9} {'avg us':>10} {'min us':>10} {'max us':>10} "
          f"{'GFLOPS':>9}  verify")
    print("-" * 60)

    got: dict[str, list[dict]] = {a: [] for a in caches}
    for rep in range(1, args.repeats + 1):
        order = list(caches) if rep % 2 else list(reversed(list(caches)))
        for arm in order:
            r = one(arm, caches[arm], args)
            if r is None:
                print(f"{rep:>4} {arm:<9}  RUN FAILED")
                return 1
            got[arm].append(r)
            print(f"{rep:>4} {arm:<9} {r['avg']:>10.1f} {r['min']:>10.1f} "
                  f"{r['max']:>10.1f} {r['gflops']:>9.1f}  "
                  f"{'PASS' if r['pass'] else 'FAIL'}")

    print()
    print(f"{'arm':<9} {'best us':>10} {'median avg':>12} {'best GFLOPS':>12}  n")
    print("-" * 49)
    summ = {}
    for arm in caches:
        best = min(r["min"] for r in got[arm])
        med = statistics.median(r["avg"] for r in got[arm])
        bg = max(r["gflops"] for r in got[arm])
        summ[arm] = (best, med)
        print(f"{arm:<9} {best:>10.1f} {med:>12.1f} {bg:>12.1f}  {len(got[arm])}")

    bs, ms = summ["stock"]
    br, mr = summ["reblock"]
    print()
    print(f"  best-to-best   reblock is {100 * (bs - br) / bs:+.1f}% vs stock")
    print(f"  median-to-med  reblock is {100 * (ms - mr) / ms:+.1f}% vs stock")
    print()
    print("  Positive means the reblocked kernel is FASTER. Its inner loop issues")
    print("  1.000 vmac/cycle against stock's 0.889 and it spills nothing, so an")
    print("  issue-bound design should show roughly +12%. The per-buffer floor")
    print("  reading predicts approximately 0%.")
    if not all(r["pass"] for a in caches for r in got[a]):
        print("  WARNING: a run did not verify; these numbers are void.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
