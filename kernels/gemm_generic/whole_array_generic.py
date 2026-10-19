#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# Adapted from mlir-aie v1.4.2 programming_examples/basic/matrix_multiplication/whole_array/whole_array.py; local changes are marked inline.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Whole-array matrix multiply — IRON API design with ``@iron.jit`` compilation.

A 4xN_cols AIE array computes ``C = A @ B`` (optionally with ``B`` column-major
or ``C`` column-major).  Each compute tile owns one (m, n) output sub-tile and
streams (m, k) x (k, n) inputs through three layers of ObjectFifos.

The script has two modes:

* ``--xclbin-path=... --insts-path=...`` — compile the design ahead-of-time
  and write artifacts to the given paths (bypasses the JIT cache).  Used by
  ``makefile-common`` so ``test.cpp`` + ``sweep.sh`` can drive the design
  via ``make``.
* default — JIT-compile + run on the attached NPU + verify against numpy.

Local change (ignite-xdna): ``--rtp-generic``. The core reads its output-tile count and its K-tile count from a
run-time parameter (RTP) buffer that the runtime sequence writes, gated by a ``WorkerRuntimeBarrier``, instead of
baking ``(M//m)*(N//n)//cores`` and ``K//k`` into the core program. With m, k, n, the dtypes and the buffering
fixed, every (M, K, N) then compiles to the same core program and differs only in its runtime sequence, so one
loaded design can serve several GEMM shapes through their instruction streams alone. Whether the device images
really are identical across shapes is checked by tools/gemm_generic_probe.py, not assumed here.

Local change (ignite-xdna): ``--epilogue-src FILE --epilogue-mode N`` (with --rtp-generic). Links FILE's
``egemm_epilogue(float *c_tile, int32_t mode)`` into every core, which calls it on each output tile after its last
K step, with a third RTP word that this shape's runtime sequence sets to N. With the epilogue linked, every
shape's core program is again identical, so one xclbin serves all modes; mode 0 must leave the tile alone.
kernels/embedgemma/geglu_epilogue.cc is the one in use (mode 1 = GeGLU).
"""

import argparse
import sys

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence, TensorTiler2D  # local change: W8 B
from aie.extras import types as T  # local change: RTP loop bounds
from aie.extras.dialects import arith  # local change: RTP loop bounds
from aie.iron import (
    Buffer,  # local change
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    StreamDims,
    TaskGroup,
    Worker,
    WorkerRuntimeBarrier,  # local change
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction  # local change: epilogue
from aie.utils import config  # local change: epilogue
from aie.iron.device import NPU2, from_name
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark


def _device_for(dev_str, n_aie_cols):
    # On NPU1 pick the matching ColN variant (or NPU1 itself when
    # n_aie_cols == max = 4).  On NPU2 use the unrestricted device
    # regardless of n_aie_cols so the placer has the full 8-column array.
    return from_name(dev_str, n_cols=n_aie_cols if dev_str == "npu" else None)


def _build_design(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    c_col_maj,
    emulate_bf16_mmul_with_bfp16,
    use_chess,
    scalar,
    c_single_buffer=False,
    rtp_generic=False,  # local change
    epilogue_src="",  # local change
    epilogue_mode=0,  # local change
    hollow_src="",  # local change: timing probe, a no-op in matmul's place
    b_w8=0,  # local change: B as host pre-swizzled W8 tiles (kernels/embedgemma/w8_dequant.cc)
    w8_dequant_src="",  # local change: the W8 dequant kernel (absent: no dequant, a transport probe)
    mm_src="",  # local change: an owned bf16 matmul in mm.cc's place (kernels/gemm_bf16/mm_bf16.cc)
    b_resident=0,  # local change: each column's K x n B strip leaves DDR once; its MemTile re-sends it per row block
    *,
    generate_taps=False,
):
    """Build the whole-array matmul IRON design and resolve to MLIR."""
    dev_str = "npu2" if isinstance(dev, NPU2) else "npu"

    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(b_col_maj),
        c_col_maj=bool(c_col_maj),
        use_chess=use_chess,
        emulate_bf16_mmul_with_bfp16=emulate_bf16_mmul_with_bfp16,
        vectorized=not scalar,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims
    epilogue_kernel = None  # local change: the tile epilogue
    if epilogue_src:
        assert rtp_generic, "--epilogue-src needs --rtp-generic (its mode is an RTP word)"
        epilogue_kernel = ExternalFunction(
            "egemm_epilogue",
            source_file=str(epilogue_src),
            compile_flags=[f"-DEPI_M={m}"],  # local change: the epilogue's row blocks follow the tile
            arg_types=[np.ndarray[(m, n), np.dtype[str_to_dtype(dtype_out_str)]], np.int32],
            include_dirs=[config.cxx_header_path()],
        )

    if dev_str == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    if dev_str == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    assert (
        M % (m * n_aie_rows) == 0
    ), "A must be tileable into (m * n_aie_rows, k)-sized blocks"
    assert K % k == 0
    assert (
        N % (n * n_aie_cols) == 0
    ), "B must be tileable into (k, n * n_aie_cols)-sized blocks"
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    fifo_depth = 2
    # The C_L1L2 fifo is each core's own output tile buffer (not the DMA
    # re-stream path A/B use) -- core_fn zeroes it, accumulates K//k matmuls
    # into it, then releases once per output tile, so there is no benefit to
    # double-buffering it unless the core can start its next tile before this
    # one's L1->L2 DMA drains. Single-buffering trades that (unmeasured) overlap
    # for m*n*dtype_out_bytes of freed L1 -- e.g. 16 KiB at m=n=64, f32 out.
    c_l1l2_depth = 1 if c_single_buffer else fifo_depth
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    else:
        n_shim_mem_A = n_aie_cols

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_taps = []
    B_taps = []
    C_taps = []

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
    if b_w8:  # local change: each B tile is k*n sign-magnitude bytes in matmul's L1 order + 4n bytes of scales
        assert (k, n) == (64, 64) and (s, t) == (8, 4), "w8_dequant.cc is written for k = n = 64, bf16 4x8x4"
        b_tile_bytes = k * n + 4 * n
        B_ty = np.ndarray[((K // k) * (N // n) * b_tile_bytes,), np.dtype[np.uint8]]
        B_l2_ty = np.ndarray[(b_tile_bytes,), np.dtype[np.uint8]]
        B8_l1_ty = B_l2_ty
        Bdq_l1_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]  # the matmul kernel takes its operands flat

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
            [
                (m // r, r * k),
                (k // s, s),
                (r, k),
                (s, 1),
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = a_l3l2.cons().split(
            of_offsets,
            obj_types=[A_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=a_dims,
        )
        A_l2l1_fifos.extend(a_tmp_fifos)

    b_repeat = M // (m * n_aie_rows)  # local change: row blocks per B strip under --b-resident
    if b_resident:
        assert not (b_w8 or b_col_maj or c_col_maj), "--b-resident: bf16 row-major B and C only"
        assert 1 <= b_repeat <= 255, "MemTile repeat_count is 1..255"
        B_strip_ty = np.ndarray[(K * n,), np.dtype[dtype_in]]
        strip_depth = 2 if 2 * K * n * np.dtype(dtype_in).itemsize <= 256 * 1024 else 1
    for col in range(n_aie_cols):
        if b_resident:  # local change: L2 holds the whole K x n strip; the cores still take k x n tiles
            b_l3l2 = ObjectFifo(B_strip_ty, name=f"B_L3L2_{col}", depth=strip_depth)
            B_l3l2_fifos.append(b_l3l2)
            fwd = b_l3l2.cons().forward(
                obj_type=B_strip_ty,
                depth=strip_depth,
                name=f"B_L2L1_{col}",
                # today's per-tile pattern over all of K: the strip's k x n tiles are consecutive rows
                dims_to_stream=[(K // s, s * n), (n // t, t), (s, n), (t, 1)],
                repeat_count=b_repeat,
            )
            fwd._consumer_obj_type = B_l1_ty  # asymmetric granularity (ObjectFifo consumer_obj_type); forward() lacks it
            B_l2l1_fifos.append(fwd)
        else:
            b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
            B_l3l2_fifos.append(b_l3l2)
            b_dims: StreamDims = (
                [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
                if b_col_maj
                else [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
            )
            B_l2l1_fifos.append(
                b_l3l2.cons().forward(obj_type=B8_l1_ty, name=f"B_L2L1_{col}")  # local change: pre-swizzled, linear
                if b_w8 else
                b_l3l2.cons().forward(
                    obj_type=B_l1_ty,
                    name=f"B_L2L1_{col}",
                    dims_to_stream=b_dims,
                )
            )

        c_dims: StreamDims = (
            [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
            if not c_col_maj
            else [(n // t, t * m), (t, r), (m // r, r * t), (r, 1)]
        )
        c_l2l3 = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=c_dims,
        )
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

    # Local change: the shape-generic core. rtp[0] = output tiles per core, rtp[1] = K // k.
    def core_fn_generic(in_a, in_b, out_c, zero, matmul, rtp, barrier, *extra):
        dequant, bscratch = (extra[0], extra[1]) if w8_dequant_src else (None, None)  # local change: W8
        epilogue = extra[2:] if w8_dequant_src else extra
        barrier.wait_for_value(1)
        n_tiles = arith.index_cast(rtp[0], to=T.index())
        k_tiles = arith.index_cast(rtp[1], to=T.index())
        mode = rtp[2] if epilogue else None  # local change: the epilogue's mode, read once per call
        for _ in range_(n_tiles):
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(k_tiles):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                if dequant is not None:  # local change: W8 tile -> bf16 in L1, then the bf16 matmul
                    dequant(elem_in_b, bscratch)
                    matmul(elem_in_a, bscratch, elem_out)
                else:
                    matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            if epilogue:
                epilogue[0](elem_out, mode)
            out_c.release(1)
        # Hand the barrier back: acquire-equal does not consume it, so without this the core never waits
        # again and runs the next call on this call's parameters (results/laya/laya_gemm_onectx_desktop2_20261013T012132Z.log). Upstream's
        # ml/mm_activation_epilogue ends each epoch the same way.
        barrier.release_with_value(1)

    n_rtp = 3 if epilogue_kernel else 2  # local change: [tiles, K tiles, epilogue mode]
    rtps = [[Buffer(np.ndarray[(n_rtp,), np.dtype[np.int32]], initial_value=np.zeros(n_rtp, np.int32),
                    name=f"rtp_{row}_{col}", use_write_rtp=True)
             for col in range(n_aie_cols)] for row in range(n_aie_rows)] if rtp_generic else None
    barriers = [[WorkerRuntimeBarrier() for _ in range(n_aie_cols)] for _ in range(n_aie_rows)] if rtp_generic else None

    if mm_src:  # local change: the owned matmul (same layouts and K order as mm.cc); mm.cc keeps zero_f32
        assert (dtype_in_str, dtype_out_str) == ("bf16", "f32") and (r, s, t) == (4, 8, 4), "--mm-src is bf16 -> f32"
        matmul_kernel = ExternalFunction("egemm_mm_bf16", source_file=str(mm_src),
                                         arg_types=[A_l1_ty, Bdq_l1_ty if b_w8 else B_l1_ty, C_l1_ty],
                                         compile_flags=[f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}",
                                                        "-DMM_SYM=egemm_mm_bf16", "-DMM_NO_ZERO"],
                                         include_dirs=[config.cxx_header_path()])
    if hollow_src:  # local change: every DMA/lock step kept, the MACs dropped (output wrong by construction)
        matmul_kernel = ExternalFunction("egemm_nomatmul", source_file=str(hollow_src),
                                         arg_types=[A_l1_ty, (B8_l1_ty if not w8_dequant_src else Bdq_l1_ty) if b_w8 else B_l1_ty,
                                                    C_l1_ty], include_dirs=[config.cxx_header_path()])
    if b_w8 and w8_dequant_src:  # local change: one dequant kernel and one bf16 B scratch tile per core
        assert rtp_generic, "--w8-dequant-src needs --rtp-generic"
        dequant_kernel = ExternalFunction("egemm_w8_dequant", source_file=str(w8_dequant_src),
                                          arg_types=[B8_l1_ty, Bdq_l1_ty], include_dirs=[config.cxx_header_path()])
        bscratch = [[Buffer(Bdq_l1_ty, name=f"bdq_{row}_{col}") for col in range(n_aie_cols)]
                    for row in range(n_aie_rows)]
    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn_generic if rtp_generic else core_fn,  # local change
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ]
            + ([rtps[row][col], barriers[row][col]] if rtp_generic else [])  # local change
            + ([dequant_kernel, bscratch[row][col]] if b_w8 and w8_dequant_src else [])  # local change
            + ([epilogue_kernel] if epilogue_kernel else []),  # local change
            stack_size=0xD00,
        ),
    )

    tb_max_n_rows = 4 if not c_col_maj else 2
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (m * n_A_tiles_per_shim, k),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
        prune_step=False,
    )
    if b_w8:  # local change: column col's tiles are contiguous in the order its cores consume them
        per_col = (K // k) * (N // n // n_aie_cols) * b_tile_bytes
        B_tiles = [TensorAccessPattern((per_col * n_aie_cols,), col * per_col,
                                       [1, 1, per_col // (b_tile_bytes // 4), b_tile_bytes // 4],
                                       [0, 0, b_tile_bytes // 4, 1]) for col in range(n_aie_cols)]
    elif b_col_maj:
        B_tiles = TensorTiler2D.step_tiler(
            (N, K),
            (n, k),
            tile_group_repeats=(N // n // n_aie_cols, K // k),
            tile_group_steps=(n_aie_cols, 1),
            prune_step=False,
        )
    else:
        B_tiles = TensorTiler2D.step_tiler(
            (K, N),
            (k, n),
            tile_group_repeats=(K // k, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            tile_group_col_major=True,
            prune_step=False,
        )
    if c_col_maj:
        # Splitting n_aie_rows out of the tile dim is what lets TensorTiler emit
        # the (col-fast, row_block-slow) DMA pattern; iter_col_major matches it.
        C_tiles = TensorTiler2D.step_tiler(
            (N, M),
            (n, m),
            tile_group_repeats=(N // n // n_aie_cols, n_aie_rows),
            tile_group_steps=(n_aie_cols, 1),
            iter_col_major=True,
            prune_step=False,
        )
    else:
        C_tiles = TensorTiler2D.step_tiler(
            (M, N),
            (m * n_aie_rows, n),
            tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            prune_step=False,
        )
    flat_workers = [w for row in workers for w in row]

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        if rtp_generic:  # local change: this shape's loop counts, then release every core
            for row in range(n_aie_rows):
                for col in range(n_aie_cols):
                    rtps[row][col][0] = n_tiles_per_core
                    rtps[row][col][1] = K // k
                    if epilogue_kernel:
                        rtps[row][col][2] = epilogue_mode
            for row in range(n_aie_rows):
                for col in range(n_aie_cols):
                    barriers[row][col].set(1)
        if b_resident:  # local change: per column n-tile j, its B strip once; A and C over every row block
            J, rb = N // n // n_aie_cols, M // (m * n_aie_rows)
            tg = TaskGroup()
            for jj in range(J):
                for col in range(n_aie_cols):
                    nb = col + jj * n_aie_cols  # this column's jj-th n-tile
                    C_hs[col].drain(C, tap=TensorAccessPattern((M, N), nb * n, [1, rb, m * n_aie_rows, n],
                                                               [0, m * n_aie_rows * N, N, 1]), wait=True, group=tg)
                    if col < n_aie_rows:  # A shim col feeds core row col: its m rows of every row block, K tiles inner
                        A_hs[col].fill(A, tap=TensorAccessPattern((M, K), col * m * K, [rb, K // k, m, k],
                                                                  [m * n_aie_rows * K, k, K, 1]), group=tg)
                    B_hs[col].fill(B, tap=TensorAccessPattern((K, N), nb * n, [1, K // k, k, n], [0, k * N, N, 1]),
                                   group=tg)
                if jj > 0:
                    tg.finish()
                    tg = TaskGroup()
            tg.finish()
            return
        c_index = 0
        tg = TaskGroup()
        for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break

                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                )

                for col in range(n_aie_cols):
                    C_taps.append(C_tiles[c_index])
                    C_hs[col].drain(
                        C,
                        tap=C_tiles[c_index],
                        wait=True,
                        group=tg,
                    )
                    c_index += 1

                    for tile_row in range(current_tb_n_rows):
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)
                        if col < n_aie_rows:
                            A_hs[col].fill(
                                A,
                                tap=A_tiles[tile_offset],
                                group=tg,
                            )
                        B_hs[col].fill(
                            B,
                            tap=B_tiles[col],
                            group=tg,
                        )
                        A_taps.append(A_tiles[tile_offset])
                        B_taps.append(B_tiles[col])

                if tb > 0 or (tb == 0 and pingpong > 0):
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    rt = Runtime(
        sequence,
        [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses],
    )

    # resolve_program() runs the sequence body, which populates the closure
    # A_taps/B_taps/C_taps lists as a side effect. The generate_taps path needs
    # that side effect, so resolve first and discard the module for that mode.
    program = Program(dev, rt, workers=flat_workers)
    module = program.resolve_program()

    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )

    return module


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def whole_array(
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
    dtype_in_str: CompileTime[str],
    dtype_out_str: CompileTime[str],
    b_col_maj: CompileTime[int] = 0,
    c_col_maj: CompileTime[int] = 0,
    emulate_bf16_mmul_with_bfp16: CompileTime[bool] = False,
    use_chess: CompileTime[bool] = False,
    scalar: CompileTime[bool] = False,
    c_single_buffer: CompileTime[bool] = False,
    rtp_generic: CompileTime[bool] = False,  # local change
    epilogue_src: CompileTime[str] = "",  # local change
    epilogue_mode: CompileTime[int] = 0,  # local change
    hollow_src: CompileTime[str] = "",  # local change
    b_w8: CompileTime[int] = 0,  # local change
    w8_dequant_src: CompileTime[str] = "",  # local change
    mm_src: CompileTime[str] = "",  # local change
    b_resident: CompileTime[int] = 0,  # local change
):
    return _build_design(
        iron.get_current_device(),
        M,
        K,
        N,
        m,
        k,
        n,
        n_aie_cols,
        dtype_in_str,
        dtype_out_str,
        b_col_maj,
        c_col_maj,
        emulate_bf16_mmul_with_bfp16,
        use_chess,
        scalar,
        c_single_buffer,
        rtp_generic,  # local change
        epilogue_src,  # local change
        epilogue_mode,  # local change
        hollow_src,  # local change
        b_w8,  # local change
        w8_dequant_src,  # local change
        mm_src,  # local change
        b_resident,  # local change
    )


def generate_taps(
    M,
    K,
    N,
    m,
    k,
    n,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    b_col_maj=0,
    c_col_maj=0,
    emulate_bf16_mmul_with_bfp16=False,
    dev="npu",
    c_single_buffer=False,
):
    """Return ``(A_taps, B_taps, C_taps)`` for the visualization notebook."""
    dev_obj = _device_for(dev, n_aie_cols)
    iron.set_current_device(dev_obj)
    return _build_design(
        dev_obj,
        M,
        K,
        N,
        m,
        k,
        n,
        n_aie_cols,
        dtype_in_str,
        dtype_out_str,
        b_col_maj,
        c_col_maj,
        emulate_bf16_mmul_with_bfp16,
        use_chess=False,
        scalar=False,
        c_single_buffer=c_single_buffer,
        generate_taps=True,
    )


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Matrix Multiplication (Whole Array)")
    add_compile_args(p, short_dev=None)
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--c-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument(  # local change
        "--rtp-generic",
        action="store_true",
        help="shape-generic core: loop counts come from a runtime parameter, not the compiled program",
    )
    p.add_argument("--epilogue-src", default="", help="local change: link egemm_epilogue(c, mode) from this file")
    p.add_argument("--epilogue-mode", type=int, default=0, help="local change: the mode this shape's call runs")
    p.add_argument("--hollow-matmul", default="", help="local change: timing probe, link this no-op in matmul's place")
    p.add_argument("--b-w8", type=int, default=0, help="local change: B as pre-swizzled W8 tiles (uint8)")
    p.add_argument("--w8-dequant-src", default="", help="local change: link this egemm_w8_dequant (else no dequant)")
    p.add_argument("--b-resident", type=int, default=0, help="local change: weight-resident B - each column's K x n "
                   "strip leaves DDR once and its MemTile re-sends it to the column's cores for every row block "
                   "(ObjectFifo repeat_count = M / (rows m), so M and K are compile-time)")
    p.add_argument("--mm-src", default="", help="local change: an owned bf16 x bf16 -> f32 matmul in mm.cc's place, same "
                   "layouts (kernels/gemm_bf16/mm_bf16.cc); mm.cc still supplies zero_f32")
    p.add_argument(
        "--c-single-buffer",
        type=int,
        choices=[0, 1],
        default=0,
        help="single-buffer the per-core C_L1L2 output tile (frees "
        "m*n*dtype_out_bytes of L1 vs the default double buffer, at the cost "
        "of any next-tile/DMA-out overlap)",
    )
    p.add_argument(
        "--emulate-bf16-mmul-with-bfp16", type=int, choices=[0, 1], default=0
    )
    p.add_argument("--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16")
    p.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i16",
    )
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--scalar",
        type=int,
        choices=[0, 1],
        default=0,
        help="use scalar (non-vector) matmul/zero kernels for debugging smaller sizes",
    )
    add_benchmark_args(p)
    return p


def _validate_shape_args(opts):
    n_aie_rows = 4
    if opts.M % (opts.m * n_aie_rows) != 0:
        sys.exit(
            f"-M {opts.M} must be a multiple of -m * n_aie_rows ({opts.m} * {n_aie_rows} = {opts.m * n_aie_rows})"
        )
    if opts.K % opts.k != 0:
        sys.exit(f"-K {opts.K} must be a multiple of -k {opts.k}")
    if opts.N % (opts.n * opts.n_aie_cols) != 0:
        sys.exit(
            f"-N {opts.N} must be a multiple of -n * --n-aie-cols ({opts.n} * {opts.n_aie_cols} = {opts.n * opts.n_aie_cols})"
        )
    tb_n_rows = 2
    n_row_blocks = opts.M // opts.m // n_aie_rows
    if n_row_blocks % tb_n_rows != 0:
        sys.exit(
            f"M/m/n_aie_rows = {n_row_blocks} must be a multiple of "
            f"{tb_n_rows} (transfer-block row count). Try a larger -M or smaller -m."
        )
    if opts.dev == "npu" and opts.n_aie_cols > 4:
        sys.exit(
            f"--n-aie-cols {opts.n_aie_cols} > 4 not supported on NPU1 (Phoenix/Hawk)"
        )
    if opts.dev == "npu2" and opts.n_aie_cols > 8:
        sys.exit(
            f"--n-aie-cols {opts.n_aie_cols} > 8 not supported on NPU2 (Strix/Strix Halo/Krackan)"
        )


def _numpy_reference(A_np, B_np, b_col_maj, dtype_out):
    B_logical = B_np.T if b_col_maj else B_np
    if np.issubdtype(A_np.dtype, np.integer):
        # numpy has no BLAS path for integer matmul, so an int64 product is a scalar
        # loop: minutes at the 4096-class shapes against milliseconds of NPU time. A
        # float64 product is bit-exact as long as every partial sum fits the 53-bit
        # mantissa, K * max|a| * max|b| <= 2**53 (K <= 2**39 for i8 x i8), and it runs
        # through BLAS dgemm in about a second. int64 stays the fallback beyond that.
        amax = lambda dt: max(abs(int(np.iinfo(dt).min)), int(np.iinfo(dt).max))
        if A_np.shape[1] * amax(A_np.dtype) * amax(B_logical.dtype) <= 2**53:
            return (A_np.astype(np.float64) @ B_logical.astype(np.float64)).astype(dtype_out)
        return (A_np.astype(np.int64) @ B_logical.astype(np.int64)).astype(dtype_out)
    return (A_np.astype(np.float32) @ B_logical.astype(np.float32)).astype(dtype_out)


def _compile_kwargs(opts):
    return dict(
        M=opts.M,
        K=opts.K,
        N=opts.N,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
        b_col_maj=opts.b_col_maj,
        c_col_maj=opts.c_col_maj,
        emulate_bf16_mmul_with_bfp16=bool(opts.emulate_bf16_mmul_with_bfp16),
        use_chess=bool(opts.use_chess),
        scalar=bool(opts.scalar),
        c_single_buffer=bool(opts.c_single_buffer),
        rtp_generic=bool(opts.rtp_generic),  # local change
        epilogue_src=opts.epilogue_src,  # local change
        epilogue_mode=opts.epilogue_mode,  # local change
        hollow_src=opts.hollow_matmul,  # local change
        b_w8=opts.b_w8,  # local change
        w8_dequant_src=opts.w8_dequant_src,  # local change
        mm_src=opts.mm_src,  # local change
        b_resident=opts.b_resident,  # local change
    )


def _run_and_verify(opts):
    dtype_in = str_to_dtype(opts.dtype_in)
    dtype_out = str_to_dtype(opts.dtype_out)

    rng = np.random.default_rng(1726250518)
    if np.issubdtype(dtype_in, np.integer):
        info = np.iinfo(dtype_in)
        # The K-reduction accumulates in a dtype_out-typed buffer, so the exact product
        # K * max|a| * max|b| has to fit dtype_out or BOTH the hardware and the numpy
        # reference overflow -- the reference visibly, as an "invalid value encountered
        # in cast" RuntimeWarning followed by a FAIL. The upstream +/-(max/4) bound is
        # safe for i8 -> i32 (good to K = 2**21) but overflows i32 for i16 inputs from
        # K >= 32, i.e. at every useful GEMM shape. Narrow the input RANGE rather than
        # the K range, so a real shape stays measurable; throughput is data-independent
        # at fixed shape and dtype, so the timing this produces is still the real one.
        bound = info.max // 4
        if np.issubdtype(dtype_out, np.integer):
            q = int(np.iinfo(dtype_out).max) // max(1, opts.K)
            b = int(q**0.5)
            if b * b > q:
                b -= 1
            bound = max(1, min(bound, b))
        A_np = rng.integers(-bound, bound, size=(opts.M, opts.K), dtype=dtype_in)
        B_shape = (opts.N, opts.K) if opts.b_col_maj else (opts.K, opts.N)
        B_np = rng.integers(-bound, bound, size=B_shape, dtype=dtype_in)
    else:
        A_np = (rng.random((opts.M, opts.K)) * 4.0).astype(dtype_in)
        B_shape = (opts.N, opts.K) if opts.b_col_maj else (opts.K, opts.N)
        B_np = (rng.random(B_shape) * 4.0).astype(dtype_in)
    A_t = iron.tensor(A_np, dtype=dtype_in, device="npu")
    B_t = iron.tensor(B_np, dtype=dtype_in, device="npu")
    C_t = iron.zeros((opts.M, opts.N), dtype=dtype_out, device="npu")

    bench = run_iters(
        whole_array,
        A_t,
        B_t,
        C_t,
        M=opts.M,
        K=opts.K,
        N=opts.N,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
        b_col_maj=opts.b_col_maj,
        c_col_maj=opts.c_col_maj,
        emulate_bf16_mmul_with_bfp16=bool(opts.emulate_bf16_mmul_with_bfp16),
        use_chess=bool(opts.use_chess),
        scalar=bool(opts.scalar),
        c_single_buffer=bool(opts.c_single_buffer),
        rtp_generic=bool(opts.rtp_generic),  # local change
        epilogue_src=opts.epilogue_src,  # local change
        epilogue_mode=opts.epilogue_mode,  # local change
        hollow_src=opts.hollow_matmul,  # local change
        b_w8=opts.b_w8,  # local change
        w8_dequant_src=opts.w8_dequant_src,  # local change
        mm_src=opts.mm_src,  # local change
        b_resident=opts.b_resident,  # local change
        warmup=opts.warmup,
        iters=opts.iters,
    )

    expected_logical = _numpy_reference(A_np, B_np, opts.b_col_maj, dtype_out)
    if opts.c_col_maj:
        actual = C_t.numpy().reshape(opts.N, opts.M)
        expected = expected_logical.T
    else:
        actual = C_t.numpy().reshape(opts.M, opts.N)
        expected = expected_logical

    assert_close_with_benchmark(
        actual,
        expected,
        bench=bench,
        ops=2.0 * opts.M * opts.K * opts.N,
        fail_msg="output does not match A @ B",
        mismatch_indices=True,
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        whole_array,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: _device_for(o.dev, o.n_aie_cols),
        validate=_validate_shape_args,
    )


if __name__ == "__main__":
    main()
