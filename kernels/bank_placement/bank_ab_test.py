"""Does separating the int8 GEMM's operands into different memory banks speed it up?

This is the controlled test H12 asks for (`docs/BENCHMARKS.md`). The production
`whole_array` int8 GEMM lands both of its input tiles in one 16 KB bank while a fourth
bank sits empty, and its inner loop contains a bundle that issues two loads at once --
the situation measured to cost one extra cycle per iteration
(`results/aie/bank_conflict_survey.log`, `results/aie/bank_check_validation.log`).

THE LEVER IS THE STACK, WHICH IS WHY THIS IS CLEAN. The allocator lays a core's local
buffers out immediately above its stack, so raising `stack_size` shifts every buffer up
by the same amount and changes nothing else: same kernel source, same tile shapes, same
ObjectFifo depths, same DMA, same schedule, and -- verifiable from the cache -- the same
compiled kernel object, since its filename carries a content hash. At the default 0xD00
the operands share bank 2; at 0x2000 the A tiles move wholly into the empty bank 3 and
the B tiles stay in bank 2.

WHAT EACH OUTCOME MEANS. `results/aie/gemm_cost_model_nest.log` says the core issues for
about 75% of this dispatch and that a per-buffer floor of roughly 3,200 cycles, set by
data delivery, is what caps it. If that is right, removing the collision changes the
core's issuing time by a known ~4% and the dispatch does not get faster, because the core
is not the critical path. If the dispatch does speed up by about that much, the design is
issue-bound after all and the floor is not what limits it.

Both arms are run ALTERNATELY rather than in two blocks, because this machine is shared
and drifts. The statistic reported is the MINIMUM over repeats: a minimum is the right
estimator for a floor, and this measurement's spread is dominated by upward excursions
from host contention, not by anything the NPU is doing.

Usage (ironenv, with the XRT SDK on PATH and pyxrt importable):
    python kernels/bank_placement/bank_ab_test.py --repeats 5
    python kernels/bank_placement/bank_ab_test.py --repeats 5 --iters 10 --warmup 3
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN = os.path.join(HERE, "whole_array_bankpad.py")

RE_NPU = re.compile(
    r"NPU time\s+\(avg/min/max us\):\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)")
RE_GFLOPS = re.compile(r"NPU GFLOPS\s*:\s*([\d.]+)")

ARMS = {
    "collide": "0xD00",     # stock: A and B share bank 2
    "separate": "0x2000",   # A moves wholly into the empty bank 3
}


def one_run(stack: str, args) -> dict | None:
    cmd = [
        sys.executable, DESIGN, "--dev", "npu",
        "-M", str(args.M), "-K", str(args.K), "-N", str(args.N),
        "-m", str(args.m), "-k", str(args.k), "-n", str(args.n),
        "--n-aie-cols", str(args.cols),
        "--dtype_in", "i8", "--dtype_out", "i32",
        "--stack-size", stack,
        "--warmup", str(args.warmup), "--iters", str(args.iters),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    m, g = RE_NPU.search(out), RE_GFLOPS.search(out)
    if not m:
        print(out[-1200:])
        return None
    return {
        "avg": float(m.group(1)), "min": float(m.group(2)), "max": float(m.group(3)),
        "gflops": float(g.group(1)) if g else None,
        "pass": "PASS!" in out,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("-M", type=int, default=4096)
    ap.add_argument("-K", type=int, default=2048)
    ap.add_argument("-N", type=int, default=2048)
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args(argv)

    print(f"shape {args.M}x{args.K}x{args.N}  tile m={args.m} k={args.k} n={args.n}  "
          f"{args.cols} columns  int8->int32")
    print(f"{args.repeats} alternating repeats per arm, "
          f"{args.warmup} warmup + {args.iters} timed iterations each")
    print()
    print(f"{'rep':>4} {'arm':<10} {'avg us':>10} {'min us':>10} {'max us':>10} "
          f"{'GFLOPS':>9}  verify")
    print("-" * 62)

    got: dict[str, list[dict]] = {a: [] for a in ARMS}
    for rep in range(1, args.repeats + 1):
        # Alternate the ORDER too, so neither arm always follows a cold cache or a
        # neighbour's burst.
        order = list(ARMS) if rep % 2 else list(reversed(list(ARMS)))
        for arm in order:
            r = one_run(ARMS[arm], args)
            if r is None:
                print(f"{rep:>4} {arm:<10}  RUN FAILED")
                return 1
            got[arm].append(r)
            print(f"{rep:>4} {arm:<10} {r['avg']:>10.1f} {r['min']:>10.1f} "
                  f"{r['max']:>10.1f} {r['gflops']:>9.1f}  "
                  f"{'PASS' if r['pass'] else 'FAIL'}")
            time.sleep(1)

    print()
    print(f"{'arm':<10} {'best us':>10} {'median avg':>12} {'best GFLOPS':>12}  n")
    print("-" * 50)
    summary = {}
    for arm in ARMS:
        best = min(r["min"] for r in got[arm])
        med = statistics.median(r["avg"] for r in got[arm])
        bg = max(r["gflops"] for r in got[arm])
        summary[arm] = (best, med, bg)
        print(f"{arm:<10} {best:>10.1f} {med:>12.1f} {bg:>12.1f}  {len(got[arm])}")

    b_col, m_col, _ = summary["collide"]
    b_sep, m_sep, _ = summary["separate"]
    print()
    print(f"  best-to-best   separate is {100 * (b_col - b_sep) / b_col:+.1f}% vs collide")
    print(f"  median-to-med  separate is {100 * (m_col - m_sep) / m_col:+.1f}% vs collide")
    print()
    print("  A positive number means separating the banks made it FASTER.")
    print("  The cost model predicts about +4% if the core is the critical path,")
    print("  and about 0% if the per-buffer delivery floor is.")
    if not all(r["pass"] for arm in ARMS for r in got[arm]):
        print("  WARNING: at least one run did not verify; the numbers above are void.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
