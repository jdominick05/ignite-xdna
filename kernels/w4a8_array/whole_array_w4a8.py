#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# Adapted from mlir-aie v1.4.2 programming_examples/basic/matrix_multiplication/whole_array/whole_array.py, by way of kernels/bank_placement/whole_array_bankpad.py; the docstring says what changed.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""W4A8 on the whole array: upstream's whole_array int8 GEMM with B stored as int4.

A local copy of mlir-aie v1.4.2's
programming_examples/basic/matrix_multiplication/whole_array/whole_array.py, by way of
kernels/bank_placement/whole_array_bankpad.py (which adds --c-single-buffer and
--stack-size), cut down to the one case this measures: int8 A, int32 C, row-major B and
C, 4 rows x n_aie_cols columns of cores. The fifo topology, the core loop, the tensor
tilers and the runtime sequence are upstream's, line for line. Two things vary:

  --arm       which kernel every core links, and whether B is int8 or packed int4
                upstream  kernels.mm(): upstream mm.cc's matmul_i8_i32 + zero_i32, B int8
                          (the design as whole_array.py builds it -- the anchor)
                i8        mm_i8i8_local: upstream's kernel re-typed, B int8
                unpack    mm_i8i4_unpack: B packed int4, widened to int8 on load, int8 mmul
                native    mm_i8i4_native: B packed int4, aie::mmul<4,16,8,int8,int4>
              The last three come from kernels/w4a8_array/w4a8_array_kernels.cc, which
              includes the one-core probe's kernels/w4a8_probe/w4a8_kernels.cc unchanged.
  --mode      the probe's k-loop modes (default, no-unroll, unroll2); upstream takes only
              default.

Packed B is K x N/2 bytes in host memory: row-major, two int4 per byte along N, column
2j in the low nibble of byte j, two's complement -- the layout the one-core probe
verified. Every B dimension below is therefore in bytes: the shim->memtile tiler walks a
(K, N/2) byte array, and the memtile->core transform emits s x t int4 blocks as s rows of
t/2 bytes, which makes the innermost DMA dimension 4 bytes (one 32-bit word) where the
int8 design's is 8.

Every arm draws the same A (int8, full range) and the same B values (in [-8, 7], int4's
range, for the int8 arms too), so every arm must return the same C, bit-exact; the run
prints C's sha256 so a sweep can check that across arms as well as against numpy.

Usage (ironenv):
    python kernels/w4a8_array/whole_array_w4a8.py --dev npu -M 2048 -K 2048 -N 2048 \\
        -m 64 -k 128 -n 64 --c-single-buffer 1 --arm native --mode unroll2 \\
        --warmup 3 --iters 10
    ... --xclbin-path out.xclbin --insts-path out.bin      # compile only, no device
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    StreamDims,
    TaskGroup,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, from_name
from aie.iron.kernel import ExternalFunction, Kernel
from aie.helpers.taplib import TensorTiler2D
from aie.utils import config
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "w4a8_array_kernels.cc"
_PROBE_DIR = _HERE.parent / "w4a8_probe"
_PROBE_SRC = _PROBE_DIR / "w4a8_kernels.cc"

# arm -> (symbol, (r, s, t), B packed int4?)
ARMS = {
    "upstream": ("matmul_i8_i32", (4, 8, 8), False),
    "i8": ("mm_i8i8_local", (4, 8, 8), False),
    "unpack": ("mm_i8i4_unpack", (4, 8, 8), True),
    "native": ("mm_i8i4_native", (4, 16, 8), True),
}
# The one-core probe's k-loop modes (kernels/w4a8_probe/static_probe.py MODES).
MODES = {
    "default": [],
    "no-unroll": ["-DINNER_NO_UNROLL"],
    "unroll2": ["-DINNER_UNROLL2"],
}
SEED = 1726250518  # whole_array.py's own


def source_rev() -> str:
    """@iron.jit keys its cache on the generator and its CompileTime args only; the .cc
    files (this design's, and the probe's it includes) go in through this hash."""
    h = hashlib.sha256()
    for p in (Path(__file__).resolve(), _SRC, _PROBE_SRC):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def kernel_obj_name(arm, mode, m, k, n) -> str:
    return f"w4a8a_{arm}_{mode}_{m}x{k}x{n}.o"


def l1_estimate(m, k, n, arm, c_single_buffer, stack_size=0xD00):
    """2A + 2B + (1|2)C + stack: the allocator arithmetic the tile sweep matched exactly,
    with B at half its bytes when it is packed."""
    b_bytes = k * n // 2 if ARMS[arm][2] else k * n
    return 2 * m * k + 2 * b_bytes + (1 if c_single_buffer else 2) * m * n * 4 + stack_size


def _device_for(dev_str, n_aie_cols):
    return from_name(dev_str, n_cols=n_aie_cols if dev_str == "npu" else None)


def _build_design(dev, M, K, N, m, k, n, n_aie_cols, arm, mode, c_single_buffer,
                  stack_size):
    if isinstance(dev, NPU2):
        raise AssertionError("this design targets NPU1 (Phoenix/Hawk Point, AIE2) only")
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if arm == "upstream" and mode != "default":
        raise ValueError("the upstream arm is kernels.mm() as whole_array builds it: mode default only")

    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols
    dtype_in, dtype_out = np.int8, np.int32
    symbol, (r, s, t), packed = ARMS[arm]

    # B's width in BYTES: half of n when two int4 share a byte.
    nb, tb, Nb = (n // 2, t // 2, N // 2) if packed else (n, t, N)

    a_ty = np.ndarray[(m * k,), np.dtype[dtype_in]]
    b_ty = np.ndarray[(k * nb,), np.dtype[dtype_in]]
    c_ty = np.ndarray[(m * n,), np.dtype[dtype_out]]

    if arm == "upstream":
        matmul_kernel = kernels.mm(
            dim_m=m, dim_k=k, dim_n=n, input_dtype=dtype_in, output_dtype=dtype_out,
            vectorized=True,
        )
        zero_kernel = matmul_kernel.zero
        assert tuple(matmul_kernel.mac_dims) == (r, s, t), matmul_kernel.mac_dims
    else:
        obj = kernel_obj_name(arm, mode, m, k, n)
        matmul_kernel = ExternalFunction(
            symbol,
            source_file=str(_SRC),
            object_file_name=obj,
            arg_types=[a_ty, b_ty, c_ty],
            include_dirs=[config.cxx_header_path(), str(_PROBE_DIR)],
            compile_flags=[f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}", *MODES[mode]],
        )
        zero_kernel = Kernel("zero_i32", obj, [c_ty])

    assert M % (m * n_aie_rows) == 0, "A must be tileable into (m * n_aie_rows, k)-sized blocks"
    assert K % k == 0
    assert N % (n * n_aie_cols) == 0, "B must be tileable into (k, n * n_aie_cols)-sized blocks"
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    fifo_depth = 2
    c_l1l2_depth = 1 if c_single_buffer else fifo_depth
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    n_shim_mem_A = n_aie_rows if n_aie_cols > n_aie_rows else n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * Nb,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * nb,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, nb), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    A_l3l2_fifos: list[ObjectFifo] = []
    A_l2l1_fifos: list[ObjectFifo] = []
    B_l3l2_fifos: list[ObjectFifo] = []
    B_l2l1_fifos: list[ObjectFifo] = []
    C_l1l2_fifos: list[list[ObjectFifo]] = [[] for _ in range(n_aie_rows)]
    C_l2l3_fifos: list[ObjectFifo] = []

    for i in range(n_shim_mem_A):
        a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        a_dims: list[StreamDims] = [
            [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
        ] * (stop_row - start_row)
        a_tmp_fifos = a_l3l2.cons().split(
            of_offsets,
            obj_types=[A_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=a_dims,
        )
        A_l2l1_fifos.extend(a_tmp_fifos)

    for col in range(n_aie_cols):
        b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        # upstream's row-major B transform, in bytes: s x t blocks as s rows of tb bytes
        b_dims: StreamDims = [(k // s, s * nb), (nb // tb, tb), (s, nb), (tb, 1)]
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims)
        )

        c_dims: StreamDims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        c_l2l3 = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, dims_to_stream=c_dims)
        C_l2l3_fifos.append(c_l2l3)
        of_offsets = [m * n * i for i in range(n_aie_rows)]
        c_tmp_fifos = c_l2l3.prod().join(
            of_offsets,
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[c_l1l2_depth] * n_aie_rows,
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j].append(c_tmp_fifos[j])

    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range(1)  # Workaround for issue #1547
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=stack_size,
        ),
    )

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m * n_A_tiles_per_shim, k), (1, K // k),
        pattern_repeat=N // n // n_aie_cols, prune_step=False,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K, Nb), (k, nb),
        tile_group_repeats=(K // k, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        tile_group_col_major=True, prune_step=False,
    )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N), (m * n_aie_rows, n),
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols), prune_step=False,
    )
    flat_workers = [w for row in workers for w in row]

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        c_index = 0
        tg = TaskGroup()
        for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min([tb_max_n_rows // 2, M // m // n_aie_rows - row_base])
                for col in range(n_aie_cols):
                    C_hs[col].drain(C, tap=C_tiles[c_index], wait=True, group=tg)
                    c_index += 1
                    for tile_row in range(current_tb_n_rows):
                        tile_offset = ((row_base + tile_row) * n_shim_mem_A + col) % len(A_tiles)
                        if col < n_aie_rows:
                            A_hs[col].fill(A, tap=A_tiles[tile_offset], group=tg)
                        B_hs[col].fill(B, tap=B_tiles[col], group=tg)
                if tb > 0 or (tb == 0 and pingpong > 0):
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses])
    return Program(dev, rt, workers=flat_workers).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def whole_array_w4a8(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    arm: CompileTime[str],
    mode: CompileTime[str],
    rev: CompileTime[str],
    c_single_buffer: CompileTime[bool] = False,
    stack_size: CompileTime[int] = 0xD00,
):
    return _build_design(iron.get_current_device(), M, K, N, m, k, n, n_aie_cols, arm,
                         mode, c_single_buffer, stack_size)


def pack_int4(b: np.ndarray) -> np.ndarray:
    """(K, N) values in [-8, 7] -> (K, N/2) bytes, column 2j in the low nibble."""
    assert b.min() >= -8 and b.max() <= 7 and b.shape[1] % 2 == 0
    u = b.astype(np.int16) & 0xF
    return ((u[:, 1::2] << 4) | u[:, 0::2]).astype(np.uint8).view(np.int8)


def unpack_int4(p: np.ndarray) -> np.ndarray:
    """Inverse of pack_int4, for the host-side self-check."""
    u = p.view(np.uint8).astype(np.int16)
    lo, hi = u & 0xF, u >> 4
    out = np.empty((p.shape[0], p.shape[1] * 2), dtype=np.int16)
    out[:, 0::2], out[:, 1::2] = lo, hi
    return np.where(out >= 8, out - 16, out).astype(np.int8)


def _make_argparser():
    p = argparse.ArgumentParser(prog="W4A8 whole-array GEMM")
    add_compile_args(p, short_dev=None)
    p.add_argument("-M", type=int, default=2048)
    p.add_argument("-K", type=int, default=2048)
    p.add_argument("-N", type=int, default=2048)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=128)
    p.add_argument("-n", type=int, default=64)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4], default=4)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--mode", choices=sorted(MODES), default="default")
    p.add_argument("--c-single-buffer", type=int, choices=[0, 1], default=0)
    p.add_argument("--stack-size", default="0xD00")
    p.add_argument("--json-out", help="append one JSON row with the timings and C's hash")
    add_benchmark_args(p)
    return p


def _validate_shape_args(opts):
    n_aie_rows = 4
    if opts.M % (opts.m * n_aie_rows) or opts.K % opts.k or opts.N % (opts.n * opts.n_aie_cols):
        sys.exit("shape not tileable: need M % 4m == 0, K % k == 0, N % (n * cols) == 0")
    if (opts.M // opts.m // n_aie_rows) % 2:
        sys.exit("M/m/4 must be even (the design's transfer-block row count)")
    if opts.arm == "upstream" and opts.mode != "default":
        sys.exit("--arm upstream takes --mode default only")


def _compile_kwargs(opts):
    return dict(
        M=opts.M, K=opts.K, N=opts.N, m=opts.m, k=opts.k, n=opts.n,
        n_aie_cols=opts.n_aie_cols, arm=opts.arm, mode=opts.mode, rev=source_rev(),
        c_single_buffer=bool(opts.c_single_buffer), stack_size=int(opts.stack_size, 0),
    )


def _run_and_verify(opts):
    kw = _compile_kwargs(opts)
    packed = ARMS[opts.arm][2]
    rng = np.random.default_rng(SEED)
    A_np = rng.integers(-128, 128, size=(opts.M, opts.K), dtype=np.int8)
    B_log = rng.integers(-8, 8, size=(opts.K, opts.N), dtype=np.int8)
    if packed:
        B_np = pack_int4(B_log)
        assert np.array_equal(unpack_int4(B_np), B_log)
    else:
        B_np = B_log
    A_t = iron.tensor(A_np.reshape(-1), dtype=np.int8, device="npu")
    B_t = iron.tensor(B_np.reshape(-1), dtype=np.int8, device="npu")
    C_t = iron.zeros((opts.M * opts.N,), dtype=np.int32, device="npu")

    print(f"w4a8 array: arm={opts.arm} mode={opts.mode} M/K/N={opts.M}/{opts.K}/{opts.N} "
          f"m/k/n={opts.m}/{opts.k}/{opts.n} cols={opts.n_aie_cols} "
          f"c_single_buffer={opts.c_single_buffer} B={'int4 packed' if packed else 'int8'} "
          f"L1 est {l1_estimate(opts.m, opts.k, opts.n, opts.arm, opts.c_single_buffer):,} B "
          f"rev={kw['rev']}", flush=True)

    bench = run_iters(whole_array_w4a8, A_t, B_t, C_t, **kw, warmup=opts.warmup, iters=opts.iters)

    # K * 128 * 8 <= 2**53 at any K this design takes, so float64 BLAS is exact here.
    expected = (A_np.astype(np.float64) @ B_log.astype(np.float64)).astype(np.int32)
    actual = C_t.numpy().reshape(opts.M, opts.N)
    mismatches = int(np.count_nonzero(actual != expected))
    c_hash = hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest()[:16]
    ops = 2.0 * opts.M * opts.K * opts.N
    row = {
        "arm": opts.arm, "mode": opts.mode, "M": opts.M, "K": opts.K, "N": opts.N,
        "m": opts.m, "k": opts.k, "n": opts.n, "cols": opts.n_aie_cols,
        "c_single_buffer": opts.c_single_buffer, "b_packed": packed, "rev": kw["rev"],
        "l1_est": l1_estimate(opts.m, opts.k, opts.n, opts.arm, opts.c_single_buffer),
        "iters": opts.iters, "warmup": opts.warmup,
        "npu_us": [bench.npu.avg_us, bench.npu.min_us, bench.npu.max_us] if bench.npu else None,
        "e2e_us": [bench.e2e.avg_us, bench.e2e.min_us, bench.e2e.max_us],
        "gops": ops / (1000.0 * bench.npu.avg_us) if bench.npu else None,
        "mismatches": mismatches, "c_sha256": c_hash,
    }
    print("W4A8_ROW " + json.dumps(row), flush=True)
    if opts.json_out:
        with open(opts.json_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    assert_close_with_benchmark(actual, expected, bench=bench, ops=ops,
                                fail_msg="output does not match A @ B", mismatch_indices=True)


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        whole_array_w4a8,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: _device_for(o.dev, o.n_aie_cols),
        validate=_validate_shape_args,
    )


if __name__ == "__main__":
    main()
