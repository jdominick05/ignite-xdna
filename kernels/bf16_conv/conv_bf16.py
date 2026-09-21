"""Run the bf16 convolution kernel on the NPU and check it against a CPU reference.

Milestone 1 of a half-precision path: prove that a real convolution - a sliding KxK window with
per-output-channel weights and bias, accumulating in fp32 - executes on AIE2's native bfloat16
matrix unit. It is deliberately one core and one tile; scaling out is only useful once the
arithmetic is known to be right.

The correctness bar is NOT bit-exactness. The int8 engine asserts exactness three ways because
integer arithmetic has one answer; bf16 does not, and this repo's standing bar for bf16 kernels is a
tolerance against a reference computed at the same width (see the bf16 GEMM and GroupNorm logs under
results/aie/). The reference here accumulates in fp32 from bf16-rounded inputs, which is what the
hardware does, so agreement should be close to exact and any real gap is a bug rather than noise.

    bash scripts/research-lowlevel.sh --log results/aie/bf16_conv_npu_<date>.log --npu \\
        -- bash scripts/research-iron.sh kernels/bf16_conv/conv_bf16.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16

SOURCE = Path(__file__).with_name("conv_bf16.cc")


def whole(n: int) -> TensorAccessPattern:
    """Move a buffer end to end, with no reshaping."""
    return TensorAccessPattern((n,), 0, [1, 1, 1, n], [0, 0, 0, 1])


@iron.jit
def conv_bf16(
    act_in: In,
    wts_in: In,
    out: Out,
    *,
    kdim: CompileTime[int] = 3,
    rows_out: CompileTime[int] = 4,
    cols_out: CompileTime[int] = 16,
    ncin: CompileTime[int] = 1,
    ncout: CompileTime[int] = 1,
):
    if cols_out % 4:
        raise ValueError("cols_out must be a multiple of 4: mmul<4,8,4> does four pixels at a time")
    rows_in, cols_in = rows_out + kdim - 1, cols_out + kdim - 1

    n_act = ncin * rows_in * cols_in * 8
    # Weights and bias share one buffer: a core tile has two input DMA channels, and activations
    # take the other. See the kernel's header comment.
    n_wts = ncout * kdim * kdim * ncin * 32 + ncout * 16
    n_out = ncout * rows_out * cols_out * 4

    act_ty = np.ndarray[(n_act,), np.dtype[bfloat16]]
    wts_ty = np.ndarray[(n_wts,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(n_out,), np.dtype[bfloat16]]

    kernel = ExternalFunction(
        "conv_bf16",
        arg_types=[act_ty, wts_ty, out_ty],
        source_file=str(SOURCE),
        object_file_name="conv_bf16.o",
        include_dirs=[config.cxx_header_path()],
        compile_flags=[
            f"-DKDIM={kdim}", f"-DROWS_OUT={rows_out}", f"-DCOLS_OUT={cols_out}",
            f"-DROWS_IN={rows_in}", f"-DCOLS_IN={cols_in}",
            f"-DNCIN={ncin}", f"-DNCOUT={ncout}",
        ],
    )

    f_act, f_wts = ObjectFifo(act_ty, name="act"), ObjectFifo(wts_ty, name="wts")
    f_out = ObjectFifo(out_ty, name="out")

    def core_fn(c_act, c_wts, p_out, compute):
        a = c_act.acquire(1)
        w = c_wts.acquire(1)
        o = p_out.acquire(1)
        compute(a, w, o)
        c_act.release(1)
        c_wts.release(1)
        p_out.release(1)

    worker = Worker(
        core_fn,
        fn_args=[f_act.cons(), f_wts.cons(), f_out.prod(), kernel],
        tile=Tile(0, 2),
    )

    def sequence(a, w, o, in_h, out_h):
        in_h[0].fill(a, whole(n_act))
        in_h[1].fill(w, whole(n_wts))
        out_h[0].drain(o, whole(n_out), wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(sequence, [act_ty, wts_ty, out_ty,
                           [f_act.prod(tile=Tile(0, 0)), f_wts.prod(tile=Tile(0, 0))],
                           [f_out.cons(tile=Tile(0, 0))]]),
        workers=[worker],
    ).resolve_program()


def reference(act, wts, bias, kdim, rows_out, cols_out, ncin, ncout):
    """The same convolution in fp32 from bf16-rounded inputs, which is what the core does.

    Layouts, and they are the contract the kernel reads:
      act  [cin_block][row][col][8]        channel = cb * 8 + k
      wts  [cout_block][ky][kx][cin_block][k * 4 + n]   mmul B is K by N, row major
      out  [cout_block][row][col][4]       channel = ob * 4 + n
    """
    rows_in, cols_in = rows_out + kdim - 1, cols_out + kdim - 1
    a = act.astype(np.float32).reshape(ncin, rows_in, cols_in, 8)
    w = wts.astype(np.float32).reshape(ncout, kdim, kdim, ncin, 8, 4)
    out = np.zeros((ncout, rows_out, cols_out, 4), np.float32)
    # The bias rides in the weight buffer as bf16, so the reference rounds it the same way.
    out += bias.astype(bfloat16).astype(np.float32).reshape(ncout, 1, 1, 4)
    for ky in range(kdim):
        for kx in range(kdim):
            # [cb][rows_out][cols_out][k] x [cb][k][n] -> [rows_out][cols_out][n], summed over cb,k
            window = a[:, ky:ky + rows_out, kx:kx + cols_out, :]
            for ob in range(ncout):
                out[ob] += np.einsum("brxk,bkn->rxn", window, w[ob, ky, kx])
    return out.reshape(-1).astype(bfloat16)


# Shapes worth proving rather than one: a 3x3 like a CNN backbone's, a 1x1 like the pointwise
# convolutions that dominate modern detectors, and a deeper-channel 3x3.
SWEEP = [
    dict(kdim=3, rows_out=4, cols_out=16, ncin=1, ncout=1),
    dict(kdim=3, rows_out=8, cols_out=32, ncin=2, ncout=4),
    dict(kdim=1, rows_out=8, cols_out=32, ncin=4, ncout=4),
    dict(kdim=3, rows_out=8, cols_out=32, ncin=4, ncout=2),
]


def run_one(k, ro, co, ncin, ncout, seed, rtol) -> int:
    ri, ci = ro + k - 1, co + k - 1

    rng = np.random.default_rng(seed)
    act = rng.normal(scale=1.0, size=ncin * ri * ci * 8).astype(np.float32).astype(bfloat16)
    wts = rng.normal(scale=0.25, size=ncout * k * k * ncin * 32).astype(np.float32).astype(bfloat16)
    bias = rng.normal(scale=0.1, size=ncout * 4).astype(np.float32)

    want = reference(act, wts, bias, k, ro, co, ncin, ncout)

    # One buffer: weights, then the bias replicated to mmul's 4x4 accumulator shape (element m*4+n
    # is bias[n]). Both because the core tile has only two input DMA channels, and because bf16 has
    # no 4-element load.
    packed = np.concatenate([wts, np.tile(bias.reshape(ncout, 4), (1, 4)).reshape(-1).astype(bfloat16)])

    act_t = iron.tensor(act, dtype=bfloat16, device="npu")
    wts_t = iron.tensor(packed, dtype=bfloat16, device="npu")
    out_t = iron.zeros(ncout * ro * co * 4, dtype=bfloat16, device="npu")

    conv_bf16(act_t, wts_t, out_t,
              kdim=k, rows_out=ro, cols_out=co, ncin=ncin, ncout=ncout)

    got = np.asarray(out_t.numpy())
    g, e = got.astype(np.float32), want.astype(np.float32)
    exact = int(np.sum(g == e))
    denom = np.maximum(np.abs(e), 1e-6)
    rel = np.abs(g - e) / denom
    worst = float(rel.max()) if rel.size else 0.0
    rel_l2 = float(np.linalg.norm(g - e) / max(np.linalg.norm(e), 1e-12))

    print("CONV_BF16", json.dumps({
        "kdim": k, "rows_out": ro, "cols_out": co, "ncin": ncin, "ncout": ncout,
        "in_channels": ncin * 8, "out_channels": ncout * 4,
        "elements": int(got.size), "bit_exact": exact, "bit_exact_frac": exact / max(got.size, 1),
        "max_rel": worst, "rel_l2": rel_l2, "nan": int(np.isnan(g).sum()),
        "macs": int(ncout * 4 * ro * co * ncin * 8 * k * k),
    }, sort_keys=True), flush=True)

    if np.isnan(g).any() or rel_l2 > rtol:
        print("FAIL: bf16 convolution does not match the fp32-from-bf16 reference")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kdim", type=int, default=3)
    ap.add_argument("--rows-out", type=int, default=4)
    ap.add_argument("--cols-out", type=int, default=16)
    ap.add_argument("--ncin", type=int, default=1, help="input channel blocks of 8")
    ap.add_argument("--ncout", type=int, default=1, help="output channel blocks of 4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--sweep", action="store_true", help="run the whole shape family in one process")
    args = ap.parse_args()

    shapes = SWEEP if args.sweep else [dict(kdim=args.kdim, rows_out=args.rows_out,
                                            cols_out=args.cols_out, ncin=args.ncin,
                                            ncout=args.ncout)]
    bad = 0
    for s in shapes:
        bad += run_one(s["kdim"], s["rows_out"], s["cols_out"], s["ncin"], s["ncout"],
                       args.seed, args.rtol)
    if bad:
        print(f"FAIL: {bad} of {len(shapes)} shapes did not match")
        return 1
    print(f"PASS! {len(shapes)} of {len(shapes)} shapes bit-exact against the reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
