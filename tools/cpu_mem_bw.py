#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU DRAM read bandwidth: threaded numpy reductions over one large buffer.

    python tools/cpu_mem_bw.py [--gib 4] [--threads 1 2 4 8 16] [--reps 5]

The buffer (float32, --gib GiB, 256x the 8700G's 16 MiB L3 at the default) is written once so
every page is resident, then each thread sums a disjoint contiguous slice with np.add.reduce, which
releases the GIL. One pass reads the whole buffer once; bandwidth = buffer bytes / pass wall time.
A reduction also spends ALU time per element, so a result is a lower bound on what the memory
system can deliver to this process: MEASURED for "numpy float32 sum", not a STREAM figure.

Each thread count prints one ROW_JSON line; tools/llm_decode_verdict.py reads them as context.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time

import numpy as np


def one_pass(views, nthreads):
    out = [0.0] * nthreads
    start = threading.Barrier(nthreads + 1)

    def work(i):
        start.wait()
        out[i] = float(np.add.reduce(views[i]))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(nthreads)]
    for t in threads:
        t.start()
    start.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, sum(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gib", type=float, default=4.0)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    n = int(args.gib * (1 << 30)) // 4
    n -= n % (64 * max(args.threads))
    print("HEADER " + json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                                  "env": os.environ.get("CONDA_DEFAULT_ENV", "?"), "host": platform.node(),
                                  "cpu": platform.processor(), "logical_cpus": os.cpu_count()}, sort_keys=True))
    t0 = time.perf_counter()
    buf = np.ones(n, dtype=np.float32)
    print(f"buffer {buf.nbytes / 2**30:.2f} GiB float32, written in {time.perf_counter() - t0:.2f} s")
    expect = float(n)
    for t in args.threads:
        views = np.array_split(buf, t)
        one_pass(views, t)                          # warm the threads and the TLB once
        ts = []
        for _ in range(args.reps):
            dt, total = one_pass(views, t)
            ts.append(dt)
            if abs(total - expect) > 1e-3 * expect:
                raise SystemExit(f"threads {t}: sum {total} != {expect}; a slice was skipped")
        med = statistics.median(ts)
        row = {"kind": "cpu_read", "threads": t, "bytes": buf.nbytes, "reps": args.reps,
               "median_ms": med * 1e3, "min_ms": min(ts) * 1e3, "max_ms": max(ts) * 1e3,
               "gbps": buf.nbytes / med / 1e9, "gbps_best": buf.nbytes / min(ts) / 1e9}
        print(f"threads {t:2d}: median {med * 1e3:8.2f} ms -> {row['gbps']:6.2f} GB/s (best {row['gbps_best']:.2f})")
        print("ROW_JSON " + json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
