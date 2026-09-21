"""Run the header-parameterized bf16 engine core on one AIE tile and check it against the emulator.

Milestone 2 of the half-precision path. Milestone 1 (conv_bf16.py) proved a bf16 convolution runs;
its every dimension was a compile-time constant, so it could serve exactly one layer. This runs the
core program that a MODEL needs: one compiled program whose shape - kernel size, stride, input
channel count, activation geometry, epilogue - arrives in the weight packet's header and is read on
the core, exactly as the int8 engine does it.

The bar is BYTE EQUALITY with src/ignite_xdna/compiler/engine_bf16_emulator.py, not a tolerance.
There is no bf16 convolution engine anywhere to compare against, so the emulator is the reference,
and it is itself checked against a naive octuple loop before it is trusted. Anything short of byte
equality on a fixed-point-free datapath means a layout is wrong, not that floating point is fuzzy.

    bash scripts/research-lowlevel.sh --log results/aie/engine_bf16_npu_<date>.log --checks-only --npu \\
        -- bash scripts/research-iron.sh kernels/bf16_conv/engine_bf16.py --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ignite_xdna.compiler.engine_bf16_emulator import (  # noqa: E402
    F_EMIT, F_RELU, F_RELU6, HOLD_OFFSET_ELEMS, NCO, OUT_BLOCK_ELEMS, PSUM_BLOCK_ELEMS,
    TILE_COLS, TILE_ROWS, make_header, run_packet, to_bf16,
)

SOURCE = Path(__file__).with_name("engine_bf16.cc")

# Packet sizes, derived rather than guessed. The weight packet is the int8 engine's 9,472 B exactly:
# bf16's mmul<4,8,4> halves the output channels per block and doubles the bytes each, so they
# cancel. The ACTIVATION packet is what genuinely doubles - int8's widest chunk already fills
# 6,400 B at one input channel block, and there is no channel count left to trade.
W_BYTES = 9472
A_BYTES = 12800
O_BYTES = NCO * OUT_BLOCK_ELEMS * 2                       # 3,200
PSUM_FLOATS = HOLD_OFFSET_ELEMS + NCO * OUT_BLOCK_ELEMS // 2   # sums, then the held bf16 tile


def whole(n: int) -> TensorAccessPattern:
    return TensorAccessPattern((n,), 0, [1, 1, 1, n], [0, 0, 0, 1])


@iron.jit
def engine_bf16(wpkt: In, apkt: In, out: Out):
    w_ty = np.ndarray[(W_BYTES // 4,), np.dtype[np.int32]]
    a_ty = np.ndarray[(A_BYTES // 2,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(O_BYTES // 2,), np.dtype[bfloat16]]
    psum_ty = np.ndarray[(PSUM_FLOATS,), np.dtype[np.float32]]

    kernel = ExternalFunction(
        "engine_bf16",
        arg_types=[w_ty, a_ty, o_ty, psum_ty, np.int32],
        source_file=str(SOURCE),
        object_file_name="engine_bf16.o",
        include_dirs=[config.cxx_header_path()],
    )

    f_w, f_a = ObjectFifo(w_ty, name="w"), ObjectFifo(a_ty, name="a")
    f_o = ObjectFifo(o_ty, name="o")

    def core_fn(w_in, a_in, o_out, engine, psum, row):
        w = w_in.acquire(1)
        a = a_in.acquire(1)
        o = o_out.acquire(1)
        engine(w, a, o, psum, row)
        a_in.release(1)
        o_out.release(1)
        w_in.release(1)

    psum = Buffer(psum_ty, name="psum_0_2")
    worker = Worker(
        core_fn,
        fn_args=[f_w.cons(), f_a.cons(), f_o.prod(), kernel, psum, 0],
        tile=Tile(0, 2),
        # engine_bf16() frames are small, but Peano's default 1 KB stack grows UPWARD into the tile
        # buffers and corrupts them silently. 2 KB is what the int8 engine uses.
        stack_size=0x800,
    )

    def sequence(w, a, o, in_h, out_h):
        in_h[0].fill(w, whole(W_BYTES // 4))
        in_h[1].fill(a, whole(A_BYTES // 2))
        out_h[0].drain(o, whole(O_BYTES // 2), wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(sequence, [w_ty, a_ty, o_ty,
                           [f_w.prod(tile=Tile(0, 0)), f_a.prod(tile=Tile(0, 0))],
                           [f_o.cons(tile=Tile(0, 0))]]),
        workers=[worker],
    ).resolve_program()


def build_packets(k, stride, ncin, flags, seed):
    """Host-side packet construction, in the exact layouts the core walks.

    Returns the device buffers and, separately, the float32 views the emulator scores.
    """
    rows_in = (TILE_ROWS - 1) * stride + k
    cols_in = (TILE_COLS - 1) * stride + k
    plane = rows_in * cols_in * 8
    # The stride-2 path loads EIGHT pixels and keeps the even four, so at the last column group it
    # reads up to one pixel (8 elements, 16 B) past the end of the final plane. Every lane it KEEPS
    # is in bounds - only discarded lanes fall off - but the packet must still own those bytes, so
    # require a pixel of slack rather than leaving it to whatever follows in L1.
    slack = 16 if stride == 2 else 0
    if plane * ncin * 2 + slack > A_BYTES:
        raise ValueError(f"activation {plane * ncin * 2} B (+{slack} B stride-2 over-read) "
                         f"exceeds the {A_BYTES} B packet")

    rng = np.random.default_rng(seed)
    act_f = to_bf16(rng.normal(size=ncin * plane).astype(np.float32))
    wts_f = to_bf16(rng.normal(scale=0.25, size=k * k * ncin * NCO * 32).astype(np.float32))
    # Replicated on the host to mmul's 4x4 C shape: element m*4+n is output channel n's bias.
    # bf16 has no 4-element load, so the core cannot broadcast it itself.
    per_ch = to_bf16(rng.normal(scale=0.1, size=NCO * 4).astype(np.float32)).reshape(NCO, 4)
    bias_f = np.repeat(per_ch[:, None, :], 4, axis=1).reshape(-1)

    header = make_header(k=k, stride=stride, ncin=ncin, flags=flags,
                         rows_in=rows_in, cols_in=cols_in, plane_elems=plane)

    w_bytes = np.zeros(W_BYTES, np.uint8)
    w_bytes[:128] = header.view(np.uint8)
    w_bytes[128:256] = bias_f.astype(bfloat16).view(np.uint8)
    wb = wts_f.astype(bfloat16).view(np.uint8)
    if wb.size > W_BYTES - 256:
        raise ValueError(f"weights {wb.size} B exceed the {W_BYTES - 256} B payload")
    w_bytes[256:256 + wb.size] = wb

    a_bytes = np.zeros(A_BYTES // 2, bfloat16)
    a_bytes[:act_f.size] = act_f.astype(bfloat16)

    return (w_bytes.view(np.int32), a_bytes, header, act_f, wts_f, bias_f, rows_in, cols_in, plane)


def run_one(k, stride, ncin, flags, seed) -> int:
    w_i32, a_bf, header, act_f, wts_f, bias_f, rows_in, cols_in, plane = \
        build_packets(k, stride, ncin, flags, seed)

    psum = np.zeros(PSUM_FLOATS, np.float32)
    want = np.zeros(NCO * OUT_BLOCK_ELEMS, np.float32)
    run_packet(header, act_f, wts_f, bias_f, psum, want)

    w_t = iron.tensor(w_i32, dtype=np.int32, device="npu")
    a_t = iron.tensor(a_bf, dtype=bfloat16, device="npu")
    o_t = iron.zeros(O_BYTES // 2, dtype=bfloat16, device="npu")

    engine_bf16(w_t, a_t, o_t)

    got = np.asarray(o_t.numpy()).astype(np.float32)
    exact = int(np.sum(got == want))
    worst = float(np.abs(got - want).max()) if got.size else 0.0

    print("ENGINE_BF16 " + json.dumps({
        "kdim": k, "stride": stride, "ncin": ncin, "in_channels": ncin * 8,
        "out_channels": NCO * 4, "relu": bool(flags & F_RELU), "relu6": bool(flags & F_RELU6),
        "rows_in": rows_in, "cols_in": cols_in, "plane_elems": plane,
        "elements": int(got.size), "bit_exact": exact,
        "bit_exact_frac": exact / max(got.size, 1), "max_abs": worst,
        "nan": int(np.isnan(got).sum()),
        "macs": int(NCO * 4 * TILE_ROWS * TILE_COLS * ncin * 8 * k * k),
    }, sort_keys=True), flush=True)

    if exact != got.size:
        print("FAIL: the core does not match the emulator byte for byte")
        return 1
    return 0


# A 3x3 like a backbone's, a 1x1 like the pointwise convolutions that dominate modern networks, a
# stride-2 downsample, and the ReLU6 that MODNet-Cut's 35 Clip nodes are.
SWEEP = [
    dict(k=3, stride=1, ncin=1, flags=F_EMIT),
    dict(k=1, stride=1, ncin=4, flags=F_EMIT),
    dict(k=3, stride=1, ncin=2, flags=F_EMIT | F_RELU),
    dict(k=3, stride=2, ncin=1, flags=F_EMIT),
    # ReLU6 is BOTH flags: F_RELU is the floor and F_RELU6 the ceiling, kept separate so a plain
    # ReLU is the same opcode with the ceiling left off. A Clip(0, 6) lowers to both.
    dict(k=3, stride=1, ncin=2, flags=F_EMIT | F_RELU | F_RELU6),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kdim", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ncin", type=int, default=1, help="input channel blocks of 8")
    ap.add_argument("--relu", action="store_true")
    ap.add_argument("--relu6", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="run the whole shape family in one process")
    args = ap.parse_args()

    if args.sweep:
        return max(run_one(s["k"], s["stride"], s["ncin"], s["flags"], args.seed) for s in SWEEP)
    flags = F_EMIT | (F_RELU if args.relu else 0) | (F_RELU6 if args.relu6 else 0)
    return run_one(args.kdim, args.stride, args.ncin, flags, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
