"""Fused BF16 Self-Attention on XDNA1 (npu1) via mlir-aie/IRON.

Computes single-head and multi-head attention:
    Scores = (Q @ K^T) * (1 / sqrt(D))
    Attn   = Softmax(Scores, dim=-1)
    Out    = Attn @ V

Runs in the mlir-aie ironenv (PowerShell, `. C:\\Users\\<user>\\mlir-aie\\iron_env.ps1`):
    python kernels/attention_bf16/attention.py -d npu --golden-dir data/golden/attn_s3_l0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import (
    Buffer,
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils import config
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

_KERNEL_SRC = Path(__file__).resolve().parent / "attention_kernels.cc"
_KERNEL_OBJ = "attention_kernels.o"


def _make_kernel(N: int, D: int, D_pad: int, scale: float):
    qkv_ty = np.ndarray[(3 * N * D_pad,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(N * D_pad,), np.dtype[bfloat16]]
    scores_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    common = dict(
        source_file=str(_KERNEL_SRC),
        object_file_name=_KERNEL_OBJ,
        include_dirs=[config.cxx_header_path()],
        compile_flags=[
            f"-DN_TOKENS={N}",
            f"-DHEAD_DIM={D}",
            f"-DHEAD_DIM_PAD={D_pad}",
            f"-DSCALE_VAL={scale}f",
        ],
    )
    return ExternalFunction(
        "attention_single_head",
        arg_types=[qkv_ty, out_ty, scores_ty],
        **common,
    )


@iron.jit
def attention_single_core(
    qkv_in: In,
    out: Out,
    *,
    N: CompileTime[int] = 64,
    D: CompileTime[int] = 20,
    D_pad: CompileTime[int] = 32,
    scale: CompileTime[float] = 0.2236068,
):
    qkv_ty = np.ndarray[(3 * N * D_pad,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(N * D_pad,), np.dtype[bfloat16]]
    scores_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    attn_fn = _make_kernel(N, D, D_pad, scale)

    of_qkv = ObjectFifo(qkv_ty, depth=1, name="of_qkv")
    of_out = ObjectFifo(out_ty, depth=1, name="of_out")
    scores_buf = Buffer(scores_ty, name="scores")

    def core_fn(qkv_cons, out_prod, scores, fn):
        qkv = qkv_cons.acquire(1)
        o = out_prod.acquire(1)

        fn(qkv, o, scores)

        qkv_cons.release(1)
        out_prod.release(1)

    worker = Worker(
        core_fn,
        fn_args=[of_qkv.cons(), of_out.prod(), scores_buf, attn_fn],
        tile=Tile(0, 2),
        stack_size=2048,
    )


    def sequence(qkv, o, of_qkv_p, of_out_c):
        tg = TaskGroup()
        of_qkv_p.fill(qkv, group=tg)
        of_out_c.drain(o, wait=True, group=tg)
        tg.finish()

    in_qkv_p = of_qkv.prod(tile=Tile(0, 0))
    out_c = of_out.cons(tile=Tile(0, 0))

    rt = Runtime(sequence, [qkv_ty, out_ty, in_qkv_p, out_c])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@iron.jit
def attention_multi_core(
    qkv_in: In,
    out: Out,
    *,
    N: CompileTime[int] = 64,
    D: CompileTime[int] = 20,
    D_pad: CompileTime[int] = 32,
    scale: CompileTime[float] = 0.2236068,
    num_cores: CompileTime[int] = 4,
):
    span_in = 3 * N * D_pad
    span_out = N * D_pad
    qkv_chunk_ty = np.ndarray[(span_in,), np.dtype[bfloat16]]
    out_chunk_ty = np.ndarray[(span_out,), np.dtype[bfloat16]]
    qkv_total_ty = np.ndarray[(num_cores * span_in,), np.dtype[bfloat16]]
    out_total_ty = np.ndarray[(num_cores * span_out,), np.dtype[bfloat16]]
    scores_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    attn_fn = _make_kernel(N, D, D_pad, scale)

    def core_fn(qkv_cons, out_prod, scores, fn):
        qkv = qkv_cons.acquire(1)
        o = out_prod.acquire(1)
        fn(qkv, o, scores)
        qkv_cons.release(1)
        out_prod.release(1)

    workers = []
    in_prods = []
    out_conses = []

    for k in range(num_cores):
        col = k % 4
        row = 2 + (k // 4)
        of_qkv = ObjectFifo(qkv_chunk_ty, depth=1, name=f"of_qkv_{k}")
        of_out = ObjectFifo(out_chunk_ty, depth=1, name=f"of_out_{k}")
        scores_buf = Buffer(scores_ty, name=f"scores_{k}")

        worker = Worker(
            core_fn,
            fn_args=[of_qkv.cons(), of_out.prod(), scores_buf, attn_fn],
            tile=Tile(col, row),
            stack_size=2048,
        )
        workers.append(worker)
        in_prods.append(of_qkv.prod(tile=Tile(col, 0)))
        out_conses.append(of_out.cons(tile=Tile(col, 0)))

    taps_in = [
        TensorAccessPattern((num_cores * span_in,), k * span_in, [1, 1, 1, span_in], [0, 0, 0, 1])
        for k in range(num_cores)
    ]
    taps_out = [
        TensorAccessPattern((num_cores * span_out,), k * span_out, [1, 1, 1, span_out], [0, 0, 0, 1])
        for k in range(num_cores)
    ]

    def sequence(qkv, o, in_p, out_c):
        tg = TaskGroup()
        for k in range(num_cores):
            in_p[k].fill(qkv, taps_in[k], group=tg)
        for k in range(num_cores):
            out_c[k].drain(o, taps_out[k], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [qkv_total_ty, out_total_ty, in_prods, out_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(description="Test fused BF16 attention on AIE")
    add_compile_args(p)
    add_benchmark_args(p)
    p.add_argument("--golden-dir", default="data/golden/attn_s3_l0", help="golden tensor directory")
    p.add_argument("-N", type=int, default=None, help="override tokens count")
    p.add_argument("-D", type=int, default=None, help="override head dimension")
    p.add_argument("--num-cores", type=int, default=1, choices=[1, 4, 8, 16], help="number of AIE cores (1, 4, 8, 16)")
    args = p.parse_args()

    num_cores = args.num_cores
    meta_file = os.path.join(args.golden_dir, "meta.json")
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)
        N = args.N or meta["q_shape"][2]
        D = args.D or meta["head_dim"]
        scale = float(meta["scale"])
        print(f"Loaded config from {meta_file}: N={N}, D={D}, scale={scale}, num_cores={num_cores}")

        q_all = np.load(os.path.join(args.golden_dir, "q.npy")).reshape(-1, N, D)
        k_all = np.load(os.path.join(args.golden_dir, "k.npy")).reshape(-1, N, D)
        v_all = np.load(os.path.join(args.golden_dir, "v.npy")).reshape(-1, N, D)
        ctx_all = np.load(os.path.join(args.golden_dir, "context.npy")).reshape(-1, N, D)

        q_sub = q_all[:num_cores]
        k_sub = k_all[:num_cores]
        v_sub = v_all[:num_cores]
        ctx_ref = ctx_all[:num_cores].astype(np.float32)
    else:
        N = args.N or 64
        D = args.D or 20
        scale = 1.0 / np.sqrt(D)
        print(f"Using synthetic inputs: N={N}, D={D}, scale={scale}, num_cores={num_cores}")
        rng = np.random.default_rng(42)
        q_sub = rng.standard_normal((num_cores, N, D)).astype(bfloat16)
        k_sub = rng.standard_normal((num_cores, N, D)).astype(bfloat16)
        v_sub = rng.standard_normal((num_cores, N, D)).astype(bfloat16)
        ctx_ref = np.zeros((num_cores, N, D), dtype=np.float32)
        for k in range(num_cores):
            q_mat = q_sub[k].astype(np.float32)
            k_mat = k_sub[k].astype(np.float32)
            v_mat = v_sub[k].astype(np.float32)
            scores = (q_mat @ k_mat.T) * scale
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_s = np.exp(scores - scores_max)
            attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
            ctx_ref[k] = attn @ v_mat

    D_pad = 16 if D <= 16 else 32
    print(f"Using D={D}, D_pad={D_pad}, num_cores={num_cores}")

    qkv_chunks = []
    for k in range(num_cores):
        qp = np.zeros((N, D_pad), dtype=bfloat16)
        qp[:, :D] = q_sub[k]
        kp = np.zeros((N, D_pad), dtype=bfloat16)
        kp[:, :D] = k_sub[k]
        vp = np.zeros((N, D_pad), dtype=bfloat16)
        vp[:, :D] = v_sub[k]
        qkv_chunks.append(np.concatenate([qp.flatten(), kp.flatten(), vp.flatten()]))
    qkv_np = np.concatenate(qkv_chunks).astype(bfloat16)

    dev = args.dev or "npu"
    print(f"Allocating device tensors on {dev} for {num_cores} cores...")
    qkv_dev = iron.tensor(qkv_np, dtype=bfloat16, device=dev)
    out_dev = iron.tensor(np.zeros((num_cores * N * D_pad,), dtype=bfloat16), dtype=bfloat16, device=dev)

    def run_fn():
        if num_cores == 1:
            attention_single_core(qkv_dev, out_dev, N=N, D=D, D_pad=D_pad, scale=scale)
        else:
            attention_multi_core(qkv_dev, out_dev, N=N, D=D, D_pad=D_pad, scale=scale, num_cores=num_cores)

    print(f"Running JIT compilation and execution on {dev}...")
    t0 = time.perf_counter()
    run_fn()
    compile_and_run_time = (time.perf_counter() - t0) * 1000
    print(f"First run (including JIT compile): {compile_and_run_time:.2f} ms")

    # Read output and verify
    out_np = out_dev.numpy().astype(np.float32).reshape(num_cores, N, D_pad)[:, :, :D]

    nan_locs = np.where(np.isnan(out_np))
    nan_cores = np.unique(nan_locs[0])
    print(f"NaN count in out_np: {len(nan_locs[0])} of {out_np.size}, across cores: {nan_cores.tolist()}")

    abs_err = np.abs(out_np - ctx_ref)
    max_err = np.max(abs_err)
    rel_l2 = np.linalg.norm(out_np - ctx_ref) / np.linalg.norm(ctx_ref)

    print(f"\n--- Numerical Verification across all {num_cores} cores ---")
    print(f"Max absolute error: {max_err:.6f}")
    print(f"Relative L2 error:  {rel_l2 * 100:.3f}%")
    for k in range(num_cores):
        k_err = np.linalg.norm(out_np[k] - ctx_ref[k]) / np.linalg.norm(ctx_ref[k])
        print(f"  Core/Head {k:2d}: rel L2 = {k_err * 100:.3f}%")

    if rel_l2 < 0.05:
        print("VERIFICATION: PASS (within bfloat16 precision bounds)")
    else:
        print("VERIFICATION: FAIL (error exceeds bfloat16 tolerance)")

    # Benchmark iterations
    if args.iters > 0:
        print(f"\n--- Benchmarking ({args.warmup} warmup, {args.iters} iterations) ---")
        for _ in range(args.warmup):
            run_fn()

        times = []
        for _ in range(args.iters):
            t_start = time.perf_counter()
            run_fn()
            times.append((time.perf_counter() - t_start) * 1e6)  # microseconds

        mean_us = np.mean(times)
        min_us = np.min(times)
        max_us = np.max(times)

        # Flops: GEMM1 (2*N*N*D) + Softmax (~5*N*N) + GEMM2 (2*N*N*D) per head
        flops_per_head = 4 * N * N * D + 5 * N * N
        total_flops = flops_per_head * num_cores
        gflops = (total_flops / (mean_us * 1e-6)) / 1e9

        print(f"Execution time (us): min={min_us:.1f}, mean={mean_us:.1f}, max={max_us:.1f}")
        print(f"Arithmetic throughput: {gflops:.2f} GFLOPS ({total_flops / 1e6:.2f} MFLOP in {mean_us/1000:.2f} ms)")


if __name__ == "__main__":
    main()
