"""Run the header-parameterized bf16 engine core on one AIE tile and check it against the emulator.

Milestone 2 of the half-precision path. Milestone 1 (conv_bf16.py) proved a bf16 convolution runs;
its every dimension was a compile-time constant, so it could serve exactly one layer. This runs the
core program that a MODEL needs: one compiled program whose shape - kernel size, stride, input
channel count, activation geometry, epilogue - arrives in the weight packet's header and is read on
the core, exactly as the int8 engine does it.

The bar is BYTE EQUALITY with src/ignite_xdna/compiler/engine_bf16_emulator.py, not a tolerance,
and it is taken on the 16-bit patterns: comparing float values calls -0.0 and +0.0 equal. There is
no bf16 convolution engine anywhere to compare against, so the emulator is the reference, and
tests/test_engine_bf16_emulator_offline.py checks it against a naive loop.

Three modes, and only the last may carry a timing claim:

  --sweep   the shape family, one packet per dispatch, plus a two-packet F_LOAD_PSUM chain run
            INSIDE ONE DISPATCH (the partial sums cross packets as unrounded fp32, which is the
            one place the accumulation order is directly observable).
  --probe   packets built so that different models of the one-instruction multiply-accumulate
            give different answers (0.0 against 1.0, not a last-bit difference). A measurement,
            not a gate: it prints what the silicon returned beside each model's prediction.
  --bench   this core against milestone 1's fixed-shape kernel on identical work, alternating in
            one process, the kernel call repeated in-core so the dispatch is amortised.

--probe-log LOG opens no device: it scores every model the emulator now has against the silicon a
--probe sitting recorded, so a model added later is scored on silicon recorded before it existed.
That is a test only if the model was not chosen by the same rows.

    bash scripts/research-lowlevel.sh --log results/aie/engine_bf16_npu_<date>.log --checks-only --npu \\
        -- bash scripts/research-iron.sh kernels/bf16_conv/engine_bf16.py --sweep --probe
    bash scripts/research-lowlevel.sh --log results/aie/engine_bf16_bench_npu_<date>.log --npu \\
        -- bash scripts/research-iron.sh kernels/bf16_conv/engine_bf16.py --bench --repeat 128

Packets per dispatch, the in-core repeat AND the kernel source are compile-time parameters of the
DESIGN (each triple is its own xclbin). The source has to be one: the jit keys its cache before the
generator body runs, on the generator's bytecode and its CompileTime values, so a source chosen
inside the body is invisible to the key and a second source silently runs the first source's
xclbin. Every design this file compiles prints an ENGINE_BF16_DESIGN line naming the object it
linked and that object's .text size, so a log shows which kernel ran. H_COUNT_OUT / H_COUNT_ACC are
not read here: a weight packet serves exactly one activation packet in this harness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ignite_xdna.compiler import engine_bf16_emulator as em  # noqa: E402
from ignite_xdna.compiler.engine_bf16_emulator import (  # noqa: E402
    F_EMIT, F_LOAD_PSUM, F_RELU, F_RELU6, NCO, OUT_BLOCK_ELEMS, PSUM_FLOATS, TILE_COLS, TILE_ROWS,
    bf16_bits, make_header, run_packet, to_bf16,
)

SOURCE = Path(__file__).with_name("engine_bf16.cc")
# --source swaps the kernel source the design compiles (a variant copy from
# tools/engine_bf16_loop_variants.py); it must define the same `engine_bf16` entry point.
SOURCE_OVERRIDE: Path | None = None


def source_key(path: Path) -> str:
    """The kernel source as a CompileTime value: its path and a digest of its text.

    The path alone would let an edited file reuse a stale xclbin; the digest alone would not say
    which file. Relative to the repository where it can be, so the key is the same in every
    checkout.
    """
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return f"{rel.as_posix()}#{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"


def source_path(key: str) -> Path:
    p = Path(key.split("#", 1)[0])
    return p if p.is_absolute() else ROOT / p

# Packet sizes, derived rather than guessed. The weight packet is the int8 engine's 9,472 B exactly:
# bf16's mmul<4,8,4> halves the output channels per block and doubles the bytes each, so they
# cancel. The ACTIVATION packet is what genuinely doubles - int8's widest chunk already fills
# 6,400 B at one input channel block, and there is no channel count left to trade.
W_BYTES = 9472
A_BYTES = 12800
O_ELEMS = NCO * OUT_BLOCK_ELEMS                           # 1,600 bf16 = 3,200 B


def whole(n: int) -> TensorAccessPattern:
    return TensorAccessPattern((n,), 0, [1, 1, 1, n], [0, 0, 0, 1])


@iron.jit
def engine_bf16(wpkt: In, apkt: In, out: Out, *, source: CompileTime[str],
                packets: CompileTime[int] = 1, repeat: CompileTime[int] = 1):
    w_ty = np.ndarray[(W_BYTES // 4,), np.dtype[np.int32]]
    a_ty = np.ndarray[(A_BYTES // 2,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(O_ELEMS,), np.dtype[bfloat16]]
    psum_ty = np.ndarray[(PSUM_FLOATS,), np.dtype[np.float32]]
    # The host buffers hold `packets` objects end to end; one shim transfer feeds them to the core
    # one object at a time, which is how the 16-core int8 design moves 64 packets per descriptor.
    w_host = np.ndarray[(packets * W_BYTES // 4,), np.dtype[np.int32]]
    a_host = np.ndarray[(packets * A_BYTES // 2,), np.dtype[bfloat16]]
    o_host = np.ndarray[(packets * O_ELEMS,), np.dtype[bfloat16]]

    kernel = ExternalFunction(
        "engine_bf16",
        arg_types=[w_ty, a_ty, o_ty, psum_ty, o_ty, np.int32],
        source_file=str(source_path(source)),
        object_file_name="engine_bf16.o",
        include_dirs=[config.cxx_header_path()],
    )

    f_w, f_a = ObjectFifo(w_ty, name="w"), ObjectFifo(a_ty, name="a")
    f_o = ObjectFifo(o_ty, name="o")

    def core_fn(w_in, a_in, o_out, engine, psum, scratch, row):
        for _ in range_(packets):
            w = w_in.acquire(1)
            a = a_in.acquire(1)
            o = o_out.acquire(1)
            # `repeat` calls on one packet amortise the dispatch for timing. An emitting packet
            # that does not load psum is idempotent, so repeating it does not change the result.
            for _ in range_(repeat):
                engine(w, a, o, psum, scratch, row)
            a_in.release(1)
            o_out.release(1)
            w_in.release(1)

    psum = Buffer(psum_ty, name="psum_0_2")
    # The hold buffer. It is the output tile's type because a held tile IS an emitted tile, and it
    # is a separate buffer rather than a tail of psum so that a 16-core design fits 65,536 B.
    scratch = Buffer(o_ty, name="scratch_0_2")
    worker = Worker(
        core_fn,
        fn_args=[f_w.cons(), f_a.cons(), f_o.prod(), kernel, psum, scratch, 0],
        tile=Tile(0, 2),
        # engine_bf16() frames are small, but Peano's default 1 KB stack grows UPWARD into the tile
        # buffers and corrupts them silently. 2 KB is what the int8 engine uses.
        stack_size=0x800,
    )

    def sequence(w, a, o, in_h, out_h):
        in_h[0].fill(w, whole(packets * W_BYTES // 4))
        in_h[1].fill(a, whole(packets * A_BYTES // 2))
        out_h[0].drain(o, whole(packets * O_ELEMS), wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(sequence, [w_host, a_host, o_host,
                           [f_w.prod(tile=Tile(0, 0)), f_a.prod(tile=Tile(0, 0))],
                           [f_o.cons(tile=Tile(0, 0))]]),
        workers=[worker],
    ).resolve_program()


def geometry(k, stride):
    rows_in = (TILE_ROWS - 1) * stride + k
    cols_in = (TILE_COLS - 1) * stride + k
    return rows_in, cols_in, rows_in * cols_in * 8


def pack(header, act_f, wts_f, bias_f):
    """One W packet and one A packet, in the exact layouts the core walks."""
    w_bytes = np.zeros(W_BYTES, np.uint8)
    w_bytes[:128] = header.view(np.uint8)
    w_bytes[128:256] = bias_f.astype(bfloat16).view(np.uint8)
    wb = wts_f.astype(bfloat16).view(np.uint8)
    if wb.size > W_BYTES - 256:
        raise ValueError(f"weights {wb.size} B exceed the {W_BYTES - 256} B payload")
    w_bytes[256:256 + wb.size] = wb
    a = np.zeros(A_BYTES // 2, bfloat16)
    a[:act_f.size] = act_f.astype(bfloat16)
    return w_bytes.view(np.int32), a


def random_packet(k, stride, ncin, flags, seed):
    rows_in, cols_in, plane = geometry(k, stride)
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
    return header, act_f, wts_f, bias_f


def residual_packet(flags, seed):
    """An OP_RESIDUAL packet: its A is an output-shaped tile and its weights are never read.

    The core takes the other addend from `scratch`, where an earlier F_HOLD packet left it, so a
    residual only makes sense as the second half of a pair. Its weights and bias still have to be
    present because every packet carries a W object, but OP_RESIDUAL never loads them - they are
    zeros so that a core which wrongly convolved this packet would emit the bias, not noise.
    """
    rng = np.random.default_rng(seed)
    act_f = to_bf16(rng.normal(scale=1.5, size=O_ELEMS).astype(np.float32))
    wts_f = np.zeros(NCO * 32, np.float32)
    bias_f = np.zeros(NCO * 16, np.float32)
    header = make_header(op=em.OP_RESIDUAL, k=1, stride=1, ncin=1, flags=flags)
    return header, act_f, wts_f, bias_f


_designs: dict[tuple[str, int, int], tuple[object, str]] = {}


def design(source: Path | None, packets: int, repeat: int):
    """The compiled design for one (source, packets, repeat), built once per process.

    Compiled eagerly so the log carries, before any result, the jit cache entry the design came
    from and the .text size of the kernel object linked into it: the one line that tells a
    variant's run from a cache hit on another source. Returns (design, cache entry).
    """
    src = source or SOURCE_OVERRIDE or SOURCE
    key = (source_key(src), packets, repeat)
    if key not in _designs:
        from tools.engine_linked_size import text_sections  # noqa: PLC0415 - needs the ironenv's llvm-size

        d = engine_bf16.specialize(source=key[0], packets=packets, repeat=repeat)
        xclbin, _ = d.compile()
        obj = xclbin.parent / "engine_bf16.o"
        entry = xclbin.parent.name
        print("ENGINE_BF16_DESIGN " + json.dumps({
            "source": key[0].split("#", 1)[0], "packets": packets, "repeat": repeat,
            "cache_entry": entry[:8],
            "object_text_bytes": sum(size for _, size in text_sections(obj)),
        }, sort_keys=True), flush=True)
        _designs[key] = (d, entry)
    return _designs[key]


def dispatch(pkts, repeat=1):
    """Send a list of (header, act, wts, bias) through ONE dispatch; return the output tiles' bits."""
    packed = [pack(*p) for p in pkts]
    w_t = iron.tensor(np.concatenate([w for w, _ in packed]), dtype=np.int32, device="npu")
    a_t = iron.tensor(np.concatenate([a for _, a in packed]), dtype=bfloat16, device="npu")
    o_t = iron.zeros(len(pkts) * O_ELEMS, dtype=bfloat16, device="npu")
    design(None, len(pkts), repeat)[0](w_t, a_t, o_t)
    return np.asarray(o_t.numpy()).view(np.uint16).reshape(len(pkts), O_ELEMS)


def expect(pkts, model):
    """The emulator's bits for the LAST packet of a chain sharing one psum and one hold buffer."""
    psum, want = np.zeros(PSUM_FLOATS, np.float32), np.zeros(O_ELEMS, np.float32)
    scratch = np.zeros(em.SCRATCH_ELEMS, np.float32)   # carries a held tile between packets
    for header, act_f, wts_f, bias_f in pkts:
        run_packet(header, act_f, wts_f, bias_f, psum, want, scratch, mac_model=model)
    return bf16_bits(want)


def check(label, pkts) -> int:
    got = dispatch(pkts)[-1]
    header = pkts[-1][0]
    per_model = {m: int(np.sum(got == expect(pkts, m))) for m in em.MAC_MODELS}
    want = expect(pkts, em.MAC_MODEL)
    exact = per_model[em.MAC_MODEL]
    values = int(np.sum(em.from_bf16_bits(got) == em.from_bf16_bits(want)))
    k, ncin, flags = int(header[em.H_K]), int(header[em.H_NCIN]), int(header[em.H_FLAGS])
    print("ENGINE_BF16 " + json.dumps({
        "case": label, "packets_in_dispatch": len(pkts), "kdim": k, "stride": int(header[em.H_STRIDE]),
        "ncin": ncin, "in_channels": ncin * 8, "out_channels": NCO * 4,
        "relu": bool(flags & F_RELU), "relu6": bool(flags & F_RELU6), "load_psum": bool(flags & F_LOAD_PSUM),
        "elements": int(got.size), "mac_model": em.MAC_MODEL, "bytes_equal": exact,
        "bytes_equal_frac": exact / got.size, "values_equal": values,
        "bytes_equal_by_model": per_model,
        "nan": int(np.isnan(em.from_bf16_bits(got)).sum()),
        "macs": int(NCO * 4 * TILE_ROWS * TILE_COLS * ncin * 8 * k * k),
    }, sort_keys=True), flush=True)
    if exact != got.size:
        print(f"FAIL {label}: the core does not match the emulator byte for byte under {em.MAC_MODEL!r}")
        return 1
    return 0


# A 3x3 like a backbone's, a 1x1 like the pointwise convolutions that dominate modern networks, a
# stride-2 downsample, the ReLU6 that MODNet-Cut's 35 Clip nodes are - and the shapes the first
# sweep left out: the 5x5 both target models open with, and a 1x1 that fills the activation packet.
SWEEP = [
    dict(k=3, stride=1, ncin=1, flags=F_EMIT),
    dict(k=1, stride=1, ncin=4, flags=F_EMIT),
    dict(k=3, stride=1, ncin=2, flags=F_EMIT | F_RELU),
    dict(k=3, stride=2, ncin=1, flags=F_EMIT),
    # ReLU6 is BOTH flags: F_RELU is the floor and F_RELU6 the ceiling, kept separate so a plain
    # ReLU is the same opcode with the ceiling left off. A Clip(0, 6) lowers to both.
    dict(k=3, stride=1, ncin=2, flags=F_EMIT | F_RELU | F_RELU6),
    dict(k=5, stride=1, ncin=1, flags=F_EMIT | F_RELU),
    dict(k=1, stride=1, ncin=8, flags=F_EMIT),
    dict(k=3, stride=1, ncin=4, flags=F_EMIT | F_RELU6 | F_RELU),
]


def sweep(seed) -> int:
    rc = 0
    for s in SWEEP:
        rc |= check(f"k{s['k']}s{s['stride']}c{s['ncin']}", [random_packet(s["k"], s["stride"], s["ncin"], s["flags"], seed)])
    # A layer whose input channels do not fit one packet: the first packet leaves fp32 partial sums
    # in psum, the second continues them and emits. One dispatch, so psum is the core's own.
    first = random_packet(3, 1, 2, 0, seed + 1)
    second = random_packet(3, 1, 2, F_LOAD_PSUM | F_EMIT | F_RELU, seed + 2)
    rc |= check("chain_k3s1_c2+c2", [first, second])
    first = random_packet(5, 1, 1, 0, seed + 3)
    second = random_packet(5, 1, 1, F_LOAD_PSUM | F_EMIT, seed + 4)
    rc |= check("chain_k5s1_c1+c1", [first, second])
    # A residual: the first packet activates its tile and HOLDS it in scratch, the second adds its
    # own A to what it finds there and emits. This pair is the only path that reads the hold buffer
    # back, so before it existed nothing could tell a working F_HOLD from one writing where no one
    # looks - and OP_RESIDUAL, declared since milestone 2, had never executed at all.
    held = random_packet(3, 1, 1, em.F_HOLD | F_RELU, seed + 5)
    rc |= check("residual_hold_add", [held, residual_packet(F_EMIT, seed + 6)])
    # The residual activates at its OWN packet's flags, not the held packet's.
    held = random_packet(3, 1, 2, em.F_HOLD, seed + 7)
    rc |= check("residual_relu6", [held, residual_packet(F_EMIT | F_RELU | F_RELU6, seed + 8)])
    # A residual may hold its own result for a second one: scratch is read and written at the same
    # offset, which is only safe because each iteration loads before it stores.
    held = random_packet(3, 1, 1, em.F_HOLD, seed + 9)
    rc |= check("residual_chain", [held, residual_packet(em.F_HOLD, seed + 10),
                                   residual_packet(F_EMIT | F_RELU, seed + 11)])
    return rc


# ---------------------------------------------------------------------------------------------------
# The probe. With k = 1 and a weight of +1 on every input channel of one output lane, the eight
# activations of a pixel ARE the eight products of that lane's multiply-accumulate, so each of the
# 100 pixels is an independent experiment on the instruction's nine-operand sum.
# ---------------------------------------------------------------------------------------------------
BIG = 30   # 2**30 swamps a +1 in fp32 (24-bit significand) and is far from overflow


def probe_vectors():
    """(label, [products of input block 0], [block 1], [block 2]) per pixel; unused blocks are None."""
    P = []
    for e in range(16, 41):                                # how wide is the sum INSIDE one instruction?
        P.append((f"intra 2^{e},+1,-2^{e}", [2.0 ** e, 1.0, -(2.0 ** e)], None, None))
    big = 2.0 ** BIG
    for i, j, l in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0),   # every order
                    (0, 1, 7), (0, 7, 1), (6, 7, 0), (0, 2, 1), (0, 4, 1), (0, 7, 3), (3, 4, 0)]:
        v = [0.0] * 8
        v[i], v[j], v[l] = big, 1.0, -big
        P.append((f"order +B@{i} +1@{j} -B@{l}", v, None, None))
    for small in (1.0, 3.0, 5.0):                          # how does the fp32 sum round: even, up, truncate?
        P.append((f"round 2^24,+{small:g},-2^24", [2.0 ** 24, small, -(2.0 ** 24)], None, None))
    P.append(("round 2^25,+2,-2^25", [2.0 ** 25, 2.0, -(2.0 ** 25)], None, None))
    P.append(("round 2^25,+6,-2^25", [2.0 ** 25, 6.0, -(2.0 ** 25)], None, None))
    P.append(("smalls 1,1,1,1,-2^25 (lane bias 2^25 tells sequential from dot-first)",
              [1.0, 1.0, 1.0, 1.0, -(2.0 ** 25)], None, None))
    P.append(("tie 2^-8 (lane bias 1: bf16 tie, even is 1.0)", [2.0 ** -8], None, None))
    P.append(("tie 3*2^-8 (lane bias 1: bf16 tie, even is 1+2^-6)", [3 * 2.0 ** -8], None, None))
    P.append(("above tie 2^-8+2^-20 (lane bias 1)", [2.0 ** -8, 2.0 ** -20], None, None))
    P.append(("product exactness (1+2^-7)^2 - (1+2^-6) on lane 2", [1 + 2.0 ** -7, 1 + 2.0 ** -6], None, None))
    for e in range(16, 41):                                # is the accumulator BETWEEN instructions fp32?
        P.append((f"inter 2^{e} | +1 | -2^{e}", [2.0 ** e], [1.0], [-(2.0 ** e)]))
    P.append(("inter eight +1 | -2^25 (lane bias 2^25)", [1.0] * 8, [-(2.0 ** 25)], None))
    # A sum that CARRIES past 24 bits has to be renormalised: how does that last step round?
    # Ties-to-even gives 0 and 4 below, truncation 0 and 2, round-half-up 2 and 4.
    for small in (1.0, 3.0):
        P.append((f"carry 2^23,2^23,+{small:g} | -2^24", [2.0 ** 23, 2.0 ** 23, small], [-(2.0 ** 24)], None))
    P.append(("carry 2^23,2^23,+1,+1 | -2^24 (exact: 2)", [2.0 ** 23, 2.0 ** 23, 1.0, 1.0], [-(2.0 ** 24)], None))
    assert len(P) <= TILE_ROWS * TILE_COLS, len(P)
    return P


def probe_packet(vectors, ncin):
    rows_in, cols_in, plane = geometry(1, 1)
    act = np.zeros(ncin * plane, np.float32)
    for p, (_, *blocks) in enumerate(vectors):
        for c in range(ncin):
            if blocks[c] is not None:
                act[c * plane + p * 8: c * plane + p * 8 + len(blocks[c])] = blocks[c]
    wts = np.zeros((ncin, NCO, 8, 4), np.float32)
    wts[:, 0, :, 0] = 1.0                                   # block 0 lane 0: the products themselves
    wts[:, 0, :, 1] = -1.0                                  # lane 1: every product negated
    wts[:, 0, 0, 2], wts[:, 0, 1, 2] = 1 + 2.0 ** -7, -1.0  # lane 2: is a bf16 x bf16 product exact?
    wts[:, 0, :, 3] = 2.0 ** -3                             # lane 3: scaled, the same cancellations
    wts[:, 1, :, :] = 1.0                                   # block 1: products against a loaded accumulator
    per_ch = np.zeros((NCO, 4), np.float32)
    per_ch[1] = [2.0 ** 25, 1.0, 2.0 ** 24, -(2.0 ** 25)]
    bias = np.repeat(per_ch[:, None, :], 4, axis=1).reshape(-1)
    assert np.array_equal(to_bf16(act), act) and np.array_equal(to_bf16(wts), wts), "probe values must be exact in bf16"
    header = make_header(k=1, stride=1, ncin=ncin, flags=F_EMIT, rows_in=rows_in, cols_in=cols_in, plane_elems=plane)
    return header, act, wts.reshape(-1), bias


def accumulator_blocks(flat: np.ndarray) -> np.ndarray:
    """An emitted tile as [accumulator block][pixel][lane], the order a --probe row's b<b>n<n> names.

    Since 2026-09-22 the core emits the 8-channel activation layout (``em.interleave_out``). The
    2026-09-21 probe ran before that, when the emitted order already was this one. Reshaping today's
    output as if it were would pair each row's label with another pixel's values. The round trip is
    checked, so a later layout change fails here instead of mislabelling rows.
    """
    blocks = (np.asarray(flat)[:em.OUT_ELEMS]
              .reshape(em.OUT_BLOCKS_8, TILE_ROWS, TILE_COLS, 2, 4)
              .transpose(0, 3, 1, 2, 4)
              .reshape(NCO, TILE_ROWS, TILE_COLS, 4))
    if not np.array_equal(em.interleave_out(blocks), np.asarray(flat)[:em.OUT_ELEMS]):
        raise AssertionError("accumulator_blocks is not the inverse of the emulator's interleave_out")
    return blocks.reshape(NCO, TILE_ROWS * TILE_COLS, 4)


def probe() -> int:
    vectors = probe_vectors()
    intra = [v for v in vectors if v[2] is None]
    inter = [v for v in vectors if v[2] is not None]
    for name, vecs, ncin in (("intra-instruction", intra, 1), ("inter-instruction", inter, 3)):
        pkt = probe_packet(vecs, ncin)
        got = em.from_bf16_bits(accumulator_blocks(dispatch([pkt])[0]))
        bits = bf16_bits(got)
        pred = {m: em.from_bf16_bits(accumulator_blocks(expect([pkt], m))) for m in em.MAC_MODELS}
        score = {m: int(np.sum(bf16_bits(pred[m]) == bits)) for m in em.MAC_MODELS}
        print("ENGINE_BF16_PROBE " + json.dumps({"set": name, "ncin": ncin, "elements": int(got.size),
                                               "bytes_equal_by_model": score}, sort_keys=True), flush=True)
        for p, (label, *_) in enumerate(vecs):
            row = {"probe": label,
                   "silicon": {f"b{b}n{n}": float(got[b, p, n]) for b in (0, 1) for n in range(4)},
                   "disagree": {m: {f"b{b}n{n}": float(pred[m][b, p, n]) for b in (0, 1) for n in range(4)
                                    if bf16_bits(pred[m][b, p, n:n + 1])[0] != bits[b, p, n]}
                                for m in em.MAC_MODELS}}
            row["disagree"] = {m: d for m, d in row["disagree"].items() if d}
            print("ENGINE_BF16_PROBE_ROW " + json.dumps(row, sort_keys=True), flush=True)
    return 0


def probe_rescore(log: Path) -> int:
    """Every emulator model, offline, against the silicon a --probe log recorded. No device is opened.

    A --probe log keeps eight elements per probe vector (blocks 0 and 1, lanes 0-3), so a model
    added after the sitting can be scored only there. Every other element has no nonzero product
    and returns its bias. Those elements are checked here: if every model agrees on them, and some
    model matched the whole set in the log, then that model's prediction there IS the silicon, and a
    whole-set count follows for every model. It is DERIVED, not measured. The derived counts of the
    models the log did score must equal the log's own counts, or the rescoring is wrong and says so.
    """
    silicon, logged = {}, {}
    for line in log.read_text(encoding="utf-8").splitlines():
        tag, _, body = line.partition(" ")
        if tag == "ENGINE_BF16_PROBE_ROW":
            r = json.loads(body)
            silicon[r["probe"]] = r["silicon"]
        elif tag == "ENGINE_BF16_PROBE":
            r = json.loads(body)
            logged[r["set"]] = r
    vectors = probe_vectors()
    intra = [v for v in vectors if v[2] is None]
    inter = [v for v in vectors if v[2] is not None]
    rc = 0
    for name, vecs, ncin in (("intra-instruction", intra, 1), ("inter-instruction", inter, 3)):
        missing = [label for label, *_ in vecs if label not in silicon]
        if missing or name not in logged:
            raise SystemExit(f"{log}: no silicon for set {name!r} or for {missing[:3]}; not a --probe log of this file")
        pkt = probe_packet(vecs, ncin)
        shape = (NCO, TILE_ROWS * TILE_COLS, 4)
        pred = {m: accumulator_blocks(expect([pkt], m)) for m in em.MAC_MODELS}
        sil = np.zeros(shape, np.uint16)
        recorded = np.zeros(shape, bool)
        for p, (label, *_) in enumerate(vecs):
            for b in (0, 1):
                for n in range(4):
                    sil[b, p, n] = bf16_bits(np.array([silicon[label][f"b{b}n{n}"]], np.float32))[0]
                    recorded[b, p, n] = True
        score = {m: int(np.sum(pred[m][recorded] == sil[recorded])) for m in em.MAC_MODELS}
        misses = {m: sorted({vecs[p][0] for b, p, n in zip(*np.nonzero(recorded & (pred[m] != sil)))})
                  for m in em.MAC_MODELS}
        rest = ~recorded
        rest_split = int(np.sum(np.any([pred[m][rest] != pred[em.MAC_MODEL][rest] for m in em.MAC_MODELS], axis=0)))
        whole = [m for m, c in logged[name]["bytes_equal_by_model"].items() if c == logged[name]["elements"]]
        derived = None
        if rest_split == 0 and whole:
            truth = pred[whole[0]][rest]
            derived = {m: score[m] + int(np.sum(pred[m][rest] == truth)) for m in em.MAC_MODELS}
        agree = derived is not None and all(derived[m] == c for m, c in logged[name]["bytes_equal_by_model"].items())
        rc |= int(not agree)
        print("ENGINE_BF16_PROBE_RESCORE " + json.dumps({
            "set": name, "log": log.name, "recorded_elements": int(recorded.sum()),
            "recorded_equal_by_model": score, "recorded_misses_by_model": {m: v for m, v in misses.items() if v},
            "unrecorded_elements": int(rest.sum()), "unrecorded_where_models_differ": rest_split,
            "whole_set_equal_by_model_DERIVED": derived, "logged_equal_by_model": logged[name]["bytes_equal_by_model"],
            "derived_reproduces_logged": agree}, sort_keys=True), flush=True)
    return rc


def bench(args) -> int:
    """This core against milestone 1's fixed-shape kernel: identical MACs, alternating, one process."""
    from conv_bf16 import conv_bf16  # noqa: PLC0415 - milestone 1's harness, imported, never edited

    k, ncin, rep = args.kdim, args.ncin, args.repeat
    flags = F_EMIT
    pkt = random_packet(k, 1, ncin, flags, args.seed)
    macs = NCO * 4 * TILE_ROWS * TILE_COLS * ncin * 8 * k * k
    w_i32, a_bf = pack(*pkt)
    w_t = iron.tensor(w_i32, dtype=np.int32, device="npu")
    a_t = iron.tensor(a_bf, dtype=bfloat16, device="npu")
    o_t = iron.zeros(O_ELEMS, dtype=bfloat16, device="npu")

    rows_in, cols_in, plane = geometry(k, 1)
    rng = np.random.default_rng(args.seed)
    m1_act = iron.tensor(rng.normal(size=ncin * plane).astype(np.float32).astype(bfloat16), dtype=bfloat16, device="npu")
    m1_wts = iron.tensor(rng.normal(scale=0.25, size=NCO * k * k * ncin * 32 + NCO * 16).astype(np.float32).astype(bfloat16),
                         dtype=bfloat16, device="npu")
    m1_out = iron.zeros(NCO * TILE_ROWS * TILE_COLS * 4, dtype=bfloat16, device="npu")

    def engine_arm(source):
        # Each source is its own design - the source is a CompileTime value of the jit, so it is in
        # the cache key - and the DESIGN is what is timed; the labels are the variant directory
        # names so the log reads without a key. Designs are resolved here, outside the clock: the
        # key digests the source text, and that read must not be timed.
        designs = {r: design(source, 1, r)[0] for r in (1, rep)}

        def run(r):
            designs[r](w_t, a_t, o_t)
        return run

    def run_m1(r):
        conv_bf16(m1_act, m1_wts, m1_out, kdim=k, rows_out=TILE_ROWS, cols_out=TILE_COLS, ncin=ncin, ncout=NCO, repeat=r)

    sources = {"engine_bf16": SOURCE_OVERRIDE or SOURCE}
    for extra in args.also:
        p = Path(extra).resolve()
        sources[f"engine_bf16[{p.parent.name}]"] = p
    arms = {name: engine_arm(src) for name, src in sources.items()}
    arms["conv_bf16_milestone1"] = run_m1
    for fn in arms.values():                       # compile and warm every design before any clock starts
        for r in (1, rep):
            for _ in range(3):
                fn(r)
    # Two arms on one cache entry would time one kernel twice under two names. Refuse to.
    for r in (1, rep):
        entries = {name: design(src, 1, r)[1] for name, src in sources.items()}
        if len(set(entries.values())) != len(entries):
            raise SystemExit(f"arms share a compiled design at repeat={r}: {entries}")
    rounds = []
    for rnd in range(args.rounds):
        for name, fn in arms.items():              # A, B, A, B ... so drift shows as spread, not as a result
            for label, r in (("single_dispatch", 1), ("amortised", rep)):
                times = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    fn(r)
                    times.append(time.perf_counter() - t0)
                best, med = min(times), float(np.median(times))
                rounds.append({"round": rnd, "arm": name, "mode": label, "passes": r,
                               "best_us_per_pass": best / r * 1e6, "median_us_per_pass": med / r * 1e6,
                               "best_gflops": 2.0 * macs * r / best / 1e9})
    # The dispatch-free cost of a pass: the slope between one pass and `repeat` passes.
    summary = {}
    for name in arms:
        one = min(x["best_us_per_pass"] for x in rounds if x["arm"] == name and x["mode"] == "single_dispatch")
        many = [x["best_us_per_pass"] for x in rounds if x["arm"] == name and x["mode"] == "amortised"]
        slope = [(m * rep - one) / (rep - 1) for m in many]
        summary[name] = {"amortised_us_per_pass_by_round": many,
                         "slope_us_per_pass_by_round": slope,
                         "slope_gflops_by_round": [2.0 * macs / (s * 1e-6) / 1e9 for s in slope],
                         "single_dispatch_us": one}
    print("ENGINE_BF16_BENCH " + json.dumps({
        "kdim": k, "stride": 1, "ncin": ncin, "in_channels": ncin * 8, "out_channels": NCO * 4,
        "tile": [TILE_ROWS, TILE_COLS], "macs_per_pass": int(macs), "repeat": rep, "iters": args.iters,
        "rounds": args.rounds, "npu_core_ceiling_gflops": 2.0 * 128 * 1.80e9 / 1e9,
        # Every engine design is a live hardware context for the whole sitting; milestone 1's two
        # are not counted here (its harness owns them).
        "engine_designs_alive": len(_designs),
        "summary": summary, "per_round": rounds,
    }, sort_keys=True), flush=True)
    return 0


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
    ap.add_argument("--probe", action="store_true", help="measure the multiply-accumulate's summation model")
    ap.add_argument("--probe-log", type=Path, default=None,
                    help="offline, no device: score every emulator model against the silicon a --probe log recorded")
    ap.add_argument("--bench", action="store_true", help="time this core against milestone 1 on identical work")
    ap.add_argument("--repeat", type=int, default=128, help="--bench: kernel calls per dispatch")
    ap.add_argument("--iters", type=int, default=20, help="--bench: timed dispatches per arm per round")
    ap.add_argument("--rounds", type=int, default=3, help="--bench: alternations of the two arms")
    ap.add_argument("--mac-model", choices=em.MAC_MODELS, default=None,
                    help="the emulator's multiply-accumulate model to gate on (default: the emulator's own)")
    ap.add_argument("--source", default=None, help="a variant copy of engine_bf16.cc to build instead of the repository's")
    ap.add_argument("--also", nargs="*", default=[], metavar="VARIANT_CC",
                    help="--bench: further kernel sources timed as extra arms, alternating with the others")
    args = ap.parse_args()
    if args.mac_model:
        em.MAC_MODEL = args.mac_model
    if args.source:
        global SOURCE_OVERRIDE
        SOURCE_OVERRIDE = Path(args.source).resolve()
        print(f"ENGINE_BF16_SOURCE {SOURCE_OVERRIDE.relative_to(ROOT) if SOURCE_OVERRIDE.is_relative_to(ROOT) else SOURCE_OVERRIDE.name}", flush=True)

    if args.probe_log:
        return probe_rescore(args.probe_log)
    if args.bench:
        return bench(args)
    rc = 0
    if args.sweep:
        rc |= sweep(args.seed)
    if args.probe:
        rc |= probe()
    if not (args.sweep or args.probe):
        flags = F_EMIT | (F_RELU if args.relu else 0) | (F_RELU6 if args.relu6 else 0)
        rc |= check("single", [random_packet(args.kdim, args.stride, args.ncin, flags, args.seed)])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
