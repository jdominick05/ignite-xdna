"""BF16 GroupNorm(32) over a [1, 32, L] tensor on XDNA1 (npu1) via mlir-aie/IRON.

This is the ONNX InstanceNormalization that resnetv2_50x3_xint8.onnx runs on the
CPU (VitisAI EP has no kernel for it), re-expressed as a hand-written AIE kernel:

    y[g, :] = scale[g] * (x[g, :] - mean[g]) / sqrt(var[g] + eps) + bias[g]

Layout on the array: 8 workers, two per column on rows 2 and 3 of all four
columns, each owning 4 consecutive groups (a contiguous 4*L block of the input).
Every worker has its own input and output ObjectFifo pinned to its column's shim
tile, so each of the 8 shim MM2S channels and 8 S2MM channels on the device
carries exactly one stream -- the design is meant to be DMA-bound, and this is
the layout that lets every shim DMA channel run at once.

Per worker the host queues three transfers on the input fifo and one on the
output fifo:

    fill  params chunk   (scale/bias for its 4 groups, raw fp32 bits in a bf16 chunk)
    fill  its 4*L block  (pass 1: statistics)
    fill  its 4*L block  (pass 2: normalise)
    drain its 4*L block

so the two-pass reduction never touches the host, and the compute is the four
C++ functions in groupnorm_kernels.cc (init / stats / finalize / affine).

Run in the mlir-aie ironenv (PowerShell, `. C:\\Users\\<user>\\mlir-aie\\iron_env.ps1`):

    python kernels/groupnorm_bf16/groupnorm.py -d npu --L 301056                # random data
    python kernels/groupnorm_bf16/groupnorm.py -d npu --golden-dir data/golden/instancenorm_L301056

The golden directory is produced by extract_golden.py (resnet_env17), and holds
the real node input / params / ORT output for one image.
"""

import argparse
import json
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import (
    Buffer,
    CompileTime,
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
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

GROUPS = 32
N_COLS = 4  # npu1
CORES_PER_COL = 2  # one core per shim DMA channel pair
N_CORES = N_COLS * CORES_PER_COL
GROUPS_PER_CORE = GROUPS // N_CORES
N_PARAMS = 2 * GROUPS_PER_CORE  # scale[4] then bias[4]
ACC_LEN = GROUPS_PER_CORE * 32  # 16 lanes sum + 16 lanes sumsq per group
EPS = 1e-5  # matches the fp32 epsilon on every InstanceNormalization node

_KERNEL_SRC = Path(__file__).resolve().parent / "groupnorm_kernels.cc"
_KERNEL_OBJ = "groupnorm_kernels.o"  # one compile shared by the four symbols


def _make_kernels(chunk: int, L: int):
    chunk_ty = np.ndarray[(chunk,), np.dtype[bfloat16]]
    params_ty = np.ndarray[(N_PARAMS,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(ACC_LEN,), np.dtype[np.float32]]
    coef_ty = np.ndarray[(2 * GROUPS_PER_CORE,), np.dtype[np.float32]]
    common = dict(
        source_file=str(_KERNEL_SRC),
        object_file_name=_KERNEL_OBJ,
        include_dirs=[config.cxx_header_path()],
        compile_flags=[
            f"-DCHUNK={chunk}",
            f"-DGROUP_LEN={L}",
            f"-DGROUPS_PER_CORE={GROUPS_PER_CORE}",
        ],
    )
    k_init = ExternalFunction("gn_init", arg_types=[chunk_ty, params_ty, acc_ty], **common)
    k_stats = ExternalFunction("gn_stats", arg_types=[chunk_ty, acc_ty, np.int32], **common)
    k_finalize = ExternalFunction("gn_finalize", arg_types=[acc_ty, params_ty, coef_ty], **common)
    k_affine = ExternalFunction(
        "gn_affine", arg_types=[chunk_ty, chunk_ty, coef_ty, np.int32], **common
    )
    return k_init, k_stats, k_finalize, k_affine


@iron.jit
def groupnorm32(
    x_in: In,
    prm_in: In,
    y_out: Out,
    *,
    L: CompileTime[int],
    chunk: CompileTime[int] = 3072,
):
    if L % chunk != 0:
        raise ValueError(f"L ({L}) must be a multiple of chunk ({chunk})")
    n_chunks = L // chunk
    span = GROUPS_PER_CORE * L  # contiguous elements owned by one core

    tensor_ty = np.ndarray[(GROUPS * L,), np.dtype[bfloat16]]
    prm_ty = np.ndarray[(N_CORES * chunk,), np.dtype[bfloat16]]
    chunk_ty = np.ndarray[(chunk,), np.dtype[bfloat16]]

    k_init, k_stats, k_finalize, k_affine = _make_kernels(chunk, L)

    def core_fn(of_in, of_out, params, acc, coef, init, stats, finalize, affine):
        # Object 0: this core's scale/bias, plus accumulator reset.
        e = of_in.acquire(1)
        init(e, params, acc)
        of_in.release(1)
        # Pass 1: statistics, one group after another (g is a Python int, so
        # the group loop unrolls and only the chunk loop is an scf.for).
        for g in range(GROUPS_PER_CORE):
            for _ in range_(n_chunks):
                e = of_in.acquire(1)
                stats(e, acc, g)
                of_in.release(1)
        finalize(acc, params, coef)
        # Pass 2: normalise, same order, now also producing output.
        for g in range(GROUPS_PER_CORE):
            for _ in range_(n_chunks):
                e = of_in.acquire(1)
                o = of_out.acquire(1)
                affine(e, o, coef, g)
                of_in.release(1)
                of_out.release(1)

    workers, in_prods, out_conses = [], [], []
    for k in range(N_CORES):
        col, row = k // CORES_PER_COL, 2 + (k % CORES_PER_COL)
        of_in = ObjectFifo(chunk_ty, name=f"in{k}")
        of_out = ObjectFifo(chunk_ty, name=f"out{k}")
        params = Buffer(np.ndarray[(N_PARAMS,), np.dtype[np.float32]], name=f"params{k}")
        acc = Buffer(np.ndarray[(ACC_LEN,), np.dtype[np.float32]], name=f"acc{k}")
        coef = Buffer(np.ndarray[(2 * GROUPS_PER_CORE,), np.dtype[np.float32]], name=f"coef{k}")
        workers.append(
            Worker(
                core_fn,
                fn_args=[
                    of_in.cons(),
                    of_out.prod(),
                    params,
                    acc,
                    coef,
                    k_init,
                    k_stats,
                    k_finalize,
                    k_affine,
                ],
                tile=Tile(col, row),
            )
        )
        # Pin both host-side endpoints to this column's shim tile so the two
        # cores of a column split that shim's two MM2S / two S2MM channels.
        in_prods.append(of_in.prod(tile=Tile(col, 0)))
        out_conses.append(of_out.cons(tile=Tile(col, 0)))

    prm_taps = [
        TensorAccessPattern((N_CORES * chunk,), k * chunk, [1, 1, 1, chunk], [0, 0, 0, 1])
        for k in range(N_CORES)
    ]
    data_taps = [
        TensorAccessPattern((GROUPS * L,), k * span, [1, 1, 1, span], [0, 0, 0, 1])
        for k in range(N_CORES)
    ]

    def sequence(x, prm, y, in_h, out_h):
        tg = TaskGroup()
        for k in range(N_CORES):
            in_h[k].fill(prm, prm_taps[k], group=tg)
            in_h[k].fill(x, data_taps[k], group=tg)  # pass 1
            in_h[k].fill(x, data_taps[k], group=tg)  # pass 2
        for k in range(N_CORES):
            out_h[k].drain(y, data_taps[k], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [tensor_ty, prm_ty, tensor_ty, in_prods, out_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# ----------------------------------------------------------------------------
# Host side: packing, references, verification, timing
# ----------------------------------------------------------------------------


def pack_params(scale: np.ndarray, bias: np.ndarray, chunk: int) -> np.ndarray:
    """One bf16 chunk per core whose leading bytes are that core's fp32 params."""
    prm = np.zeros((N_CORES, chunk), dtype=bfloat16)
    raw = prm.view(np.uint16)
    for k in range(N_CORES):
        sl = slice(k * GROUPS_PER_CORE, (k + 1) * GROUPS_PER_CORE)
        vals = np.concatenate([scale[sl], bias[sl]]).astype(np.float32)
        raw[k, : 2 * N_PARAMS] = vals.view(np.uint16)
    return prm.reshape(-1)


def reference(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    """float64 GroupNorm with a centred two-pass variance; x is (32, L)."""
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=1, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=1, keepdims=True)
    return (x64 - mean) / np.sqrt(var + eps) * scale[:, None].astype(np.float64) + bias[
        :, None
    ].astype(np.float64)


def _report(label: str, out: np.ndarray, ref: np.ndarray) -> float:
    err = np.abs(out - ref)
    rel_l2 = np.sqrt((err**2).sum(axis=1) / (ref**2).sum(axis=1))
    i = np.unravel_index(err.argmax(), err.shape)
    print(
        f"{label}: max|err| {err.max():.5f} at group {i[0]} (ref {ref[i]:.4f}, out {out[i]:.4f}),"
        f" mean|err| {err.mean():.6f}, per-group rel-L2 mean {rel_l2.mean():.5f} max {rel_l2.max():.5f}"
    )
    return err.max()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE bf16 GroupNorm(32)", description=__doc__.split("\n")[0])
    add_compile_args(p, with_elf=True)
    add_benchmark_args(p, default_warmup=2, default_iters=10)
    p.add_argument("--L", type=int, default=301056, help="group length (spatial extent per group)")
    p.add_argument("--chunk", type=int, default=3072, help="bf16 elements per ObjectFifo object")
    p.add_argument(
        "--golden-dir",
        type=str,
        default=None,
        help="directory from extract_golden.py (x.npy, scale.npy, bias.npy, y_ort.npy, meta.json); "
        "overrides --L and the random input",
    )
    p.add_argument("--seed", type=int, default=0)
    return p


def _load_inputs(opts):
    if opts.golden_dir:
        d = Path(opts.golden_dir)
        with open(d / "meta.json") as f:
            meta = json.load(f)
        x = np.load(d / "x.npy").reshape(GROUPS, -1).astype(np.float32)
        scale = np.load(d / "scale.npy").astype(np.float32)
        bias = np.load(d / "bias.npy").astype(np.float32)
        y_ort = np.load(d / "y_ort.npy").reshape(GROUPS, -1).astype(np.float32)
        eps = float(meta["epsilon"])
        opts.L = x.shape[1]
        print(f"golden: {meta['node']} from {Path(meta['model']).name}, L={opts.L}, eps={eps}")
        return x, scale, bias, y_ort, eps
    rng = np.random.default_rng(opts.seed)
    # Per-group offset and spread so the mean/var path is exercised, not just identity.
    mean = rng.uniform(-1.0, 1.0, size=(GROUPS, 1))
    std = rng.uniform(0.5, 2.0, size=(GROUPS, 1))
    x = (rng.standard_normal(size=(GROUPS, opts.L)) * std + mean).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, size=(GROUPS,)).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, size=(GROUPS,)).astype(np.float32)
    print(f"random input: L={opts.L}, seed={opts.seed}")
    return x, scale, bias, None, EPS


def _compile_kwargs(opts):
    if opts.golden_dir:
        # --L is taken from the golden tensor; make sure the design sees it.
        opts.L = np.load(Path(opts.golden_dir) / "x.npy", mmap_mode="r").shape[-1]
    return dict(L=opts.L, chunk=opts.chunk)


def _run_and_verify(opts):
    x32, scale, bias, y_ort, eps = _load_inputs(opts)
    L = opts.L
    if L % opts.chunk != 0:
        sys.exit(f"L={L} is not a multiple of --chunk {opts.chunk}")

    x_bf = x32.astype(bfloat16)  # what the kernel actually sees
    prm = pack_params(scale, bias, opts.chunk)

    x_t = iron.tensor(x_bf.reshape(-1), dtype=bfloat16, device="npu")
    prm_t = iron.tensor(prm, dtype=bfloat16, device="npu")
    y_t = iron.zeros(GROUPS * L, dtype=bfloat16, device="npu")

    kw = dict(L=L, chunk=opts.chunk)
    groupnorm32(x_t, prm_t, y_t, **kw)  # compiles (or hits the cache) and runs once
    out = y_t.numpy().astype(np.float32).reshape(GROUPS, L)

    # --- numerical checks -------------------------------------------------
    ref_bf = reference(x_bf, scale, bias, eps)  # exact op on the bf16-rounded input
    ref_bf_q = ref_bf.astype(bfloat16).astype(np.float64)  # ...then rounded like the kernel output
    print(f"output |y| max {np.abs(ref_bf).max():.4f}")
    e_exact = _report("kernel vs f64 reference on bf16 input   ", out.astype(np.float64), ref_bf)
    e_quant = _report("kernel vs same reference rounded to bf16", out.astype(np.float64), ref_bf_q)
    # How much of the error is the kernel's, versus what bf16 output rounding costs anyway?
    _report("bf16-rounded reference vs f64 reference ", ref_bf_q, ref_bf)
    if y_ort is not None:
        _report("kernel vs ORT CPU fp32 InstanceNorm       ", out.astype(np.float64), y_ort.astype(np.float64))
        _report("f64 ref on fp32 input vs ORT CPU fp32    ", reference(x32, scale, bias, eps), y_ort.astype(np.float64))

    # Variance cancellation exposure of the kernel's E[x^2]-mean^2 (in fp32):
    x64 = x_bf.astype(np.float64)
    mean = x64.mean(axis=1)
    var_c = ((x64 - mean[:, None]) ** 2).mean(axis=1)
    var_u = (x64**2).mean(axis=1) - mean**2
    print(
        f"per-group |mean|/std max {(np.abs(mean) / np.sqrt(var_c)).max():.3f}; "
        f"E[x^2]-mean^2 vs centred variance: max rel diff {np.abs(var_u - var_c).max() / var_c.min():.2e}"
    )

    # Pass criterion: within bf16 output quantisation of the exact result.
    # |y| <= ~10 here, so 2^-8 * |y| is ~0.04 at the extremes; allow that plus
    # a small absolute floor for the near-zero outputs.
    tol = 0.02 + np.abs(ref_bf) * 2**-8
    bad = np.abs(out.astype(np.float64) - ref_bf) > tol
    n_bad = int(bad.sum())
    print(f"elements outside 0.02 + |y|/256 tolerance: {n_bad} / {bad.size}")

    # --- timing --------------------------------------------------------------
    bench = run_iters(
        groupnorm32, x_t, prm_t, y_t, warmup=opts.warmup, iters=opts.iters, **kw
    )
    print_benchmark(bench)
    nbytes = GROUPS * L * 2
    if bench.npu is not None:
        moved = 3 * nbytes  # 2 reads + 1 write of the tensor
        print(
            f"tensor {nbytes / 1e6:.2f} MB bf16; DMA traffic {moved / 1e6:.2f} MB -> "
            f"{moved / bench.npu.avg_us / 1e3:.2f} GB/s at NPU avg, "
            f"{moved / bench.npu.min_us / 1e3:.2f} GB/s at NPU min"
        )
    if n_bad:
        print("FAIL: numerical check")
        sys.exit(1)
    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        groupnorm32,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),  # all 4 columns
    )


if __name__ == "__main__":
    main()
