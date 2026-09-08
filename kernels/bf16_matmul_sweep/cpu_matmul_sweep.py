#!/usr/bin/env python3
"""CPU GEMM throughput at the shapes the NPU whole_array bf16 matmul sweep uses.

WHY THIS EXISTS
----------------
results/aie/mlir_aie_bf16_matmul_npu.log measured 895 GFLOPS for a 4-column bf16
whole_array 512x512x512 matmul on this hardware -- the one op in kernels/ that reads
as a genuine NPU strength rather than a loss. attention_bf16/README.md's own math
(quoting the dispatch floor, results/aie/dispatch_floor_npu.log) says a *perfect*
aie::mmul-based op needs roughly 20x more compute than MobileViT-scale attention
before that throughput outruns the ~617us dispatch floor and starts mattering. But
no CPU GEMM baseline at these shapes existed anywhere in this repo -- every other CPU
number here is int8 QDQ (conv), not bf16/fp32 matmul. This is that baseline.

USAGE (resnet_env -- has torch; NOT ironenv)
    conda activate resnet_env
    python kernels/bf16_matmul_sweep/cpu_matmul_sweep.py --iters 20 --warmup 5

Run in the same sitting as the NPU sweep (kernels/bf16_matmul_sweep/*.log from
whole_array.py) per this repo's drift invariant.
"""

import argparse
import os
import statistics
import time

import torch

torch.manual_seed(0)


def flops(M, K, N):
    return 2 * M * K * N


def time_matmul(M, K, N, dtype, iters, warmup):
    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)
    if dtype == torch.bfloat16:
        a = a.to(torch.bfloat16)
        b = b.to(torch.bfloat16)

    for _ in range(warmup):
        torch.matmul(a, b)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        torch.matmul(a, b)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--shapes",
        type=str,
        default="512x512x512,512x512x2048,512x2048x2048,2048x2048x2048,4096x4096x4096",
        help="comma-separated MxKxN list",
    )
    args = p.parse_args()

    shapes = []
    for tok in args.shapes.split(","):
        m, k, n = tok.lower().split("x")
        shapes.append((int(m), int(k), int(n)))

    print("CPU GEMM sweep -- torch, single process, this machine's Zen4 cores")
    # Thread count is part of the number: scripts/lib.sh now exports OMP_NUM_THREADS=8
    # (2026-09-07), and the same FP32 model moved 30% on that alone (docs/BENCHMARKS.md,
    # "Known limitations"). This script does not source lib.sh; record what torch got.
    print(f"torch {torch.__version__}, torch threads {torch.get_num_threads()} "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')}), "
          f"iters={args.iters} warmup={args.warmup}\n")

    for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        print(f"--- dtype={label} ---")
        head = f"{'MxKxN':>20} {'MFLOP':>10} {'mean ms':>10} {'min ms':>10} {'GFLOPS':>10}"
        print(head)
        print("-" * len(head))
        for m, k, n in shapes:
            f = flops(m, k, n)
            times = time_matmul(m, k, n, dtype, args.iters, args.warmup)
            mean_t, min_t = statistics.mean(times), min(times)
            print(
                f"{f'{m}x{k}x{n}':>20} {f/1e6:>10.2f} {mean_t:>10.4f} {min_t:>10.4f} "
                f"{f/(mean_t*1e6):>10.1f}"
            )
        print()


if __name__ == "__main__":
    main()
