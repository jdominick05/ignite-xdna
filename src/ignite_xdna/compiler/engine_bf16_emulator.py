"""A NumPy model of one bf16 engine packet, exact to the bit against the core program.

The int8 engine's emulator earns its keep by asserting that the silicon result of a packet equals
`run_packet` byte for byte, and that contract is what makes a wrong layer findable at all. It is
worth more here than it was there, because a bf16 convolution engine has no reference
implementation anywhere to compare against - not in this repo, not in mlir-aie's aie_kernels, not in
its programming_examples.

This mirrors kernels/bf16_conv/engine_bf16.cc line for line, including the things that look like
details and are not:

  * the accumulation is fp32 from bf16 operands, ONE multiply-accumulate per (ky, kx, input channel
    block) in the core's walk order, because float addition does not reassociate. An earlier
    version contracted a whole tap in one einsum, whose summation order NumPy does not define;
  * every value that the core STORES is rounded to bf16 round-to-nearest-even, and every value it
    keeps in psum stays fp32;
  * the held tile lives in the core's `scratch` buffer, not in a tail of psum, and the emitted tile
    is written in the next layer's ACTIVATION layout - eight channels a block, interleaved from the
    pairs of 4-channel accumulator blocks that mmul<4, 8, 4> produces;
  * the bias is read pre-replicated to the accumulator's 4x4 shape, so element m*4+n is the bias of
    output channel n - replication happens on the host because bf16 has no 4-element load.

WHAT IS NOT YET KNOWN. One multiply-accumulate is a single hardware instruction
(aie_api's `::mac_4x8_8x4_conf`): it forms eight products per output lane and adds them to the
accumulator inside the vector unit. The order and the width of that nine-operand sum cannot be read
from any source, so `mac_model` names the candidates and `MAC_MODEL` is the one in force. Output is
compared only after rounding to bf16's 8-bit mantissa, which hides almost every difference between
the candidates on ordinary data: a passing random sweep does not select a model, a probe built to
make them disagree does (kernels/bf16_conv/engine_bf16.py --probe).

The geometry constants are the kernel's, and a mismatch between the two files is a bug in whichever
was edited second.
"""
from __future__ import annotations

import numpy as np

TILE_ROWS = 5
TILE_COLS = 20
NCO = 4                      # output blocks of 4 channels
GROUPS = TILE_COLS // 4      # 5 groups of 4 pixels

HDR_BYTES = 128
HDR_WORDS = HDR_BYTES // 4
BIAS_ELEMS = NCO * 16                            # 64 bf16, 128 B
W_OFFSET_ELEMS = HDR_BYTES // 2 + BIAS_ELEMS     # counted in bf16 elements
OUT_BLOCK_ELEMS = TILE_ROWS * TILE_COLS * 4      # 400 bf16, one 4-channel accumulator block
PSUM_BLOCK_ELEMS = TILE_ROWS * TILE_COLS * 4     # 400 float
PSUM_FLOATS = NCO * PSUM_BLOCK_ELEMS             # 1,600 float32 = 6,400 B, the core's psum buffer

# The emitted tile is written in the ACTIVATION layout, [cin_block][row][col][8], because that is
# what the next layer reads and what Placement lays out in DDR (engine_schedule.Placement.pitch is
# `(width + 2*halo) * 8 * itemsize` - eight channels a block, whatever the dtype). The core
# accumulates in 4-channel blocks because mmul<4,8,4> blocks the output by 4, so a pair of
# accumulator blocks interleaves into one 8-channel block on the way out.
OUT_BLOCKS_8 = NCO // 2                          # 2 blocks of 8 channels
OUT_BLOCK8_ELEMS = TILE_ROWS * TILE_COLS * 8     # 800 bf16
OUT_ELEMS = NCO * OUT_BLOCK_ELEMS                # 1,600 bf16 = 3,200 B, either way it is counted

# The held tile lives in `scratch`, NOT in a tail of psum. The int8 core aliases its hold into
# psum's tail and this file used to mirror that, but bf16 cannot afford it: at int8 depths the
# 16-core design needs 65,792 B of a 65,536 B tile, and the 3,200 B hold is what puts it over.
# `scratch` is already allocated per core (design.py's `scratch_{c}_{r}`, o_ty, 3,200 B) and is
# never written - the accumulate-only loop passes it as `out` and a non-emitting packet writes
# psum, not `out`. Holding there costs nothing, and it removes the reinterpret_cast aliasing that
# the milestone-2 commit recorded as needing to close before OP_RESIDUAL existed.
SCRATCH_ELEMS = OUT_ELEMS                        # 1,600 bf16 = 3,200 B

H_OP, H_K, H_STRIDE, H_NCIN, H_FLAGS = 0, 1, 2, 3, 4
H_COUNT_OUT, H_COUNT_ACC = 5, 6
H_ROWS_IN, H_COLS_IN, H_PLANE_ELEMS = 7, 8, 9
H_PHASE0 = 12

OP_NOP, OP_CONV, OP_RESIDUAL = 0, 1, 2
F_LOAD_PSUM, F_EMIT, F_RELU, F_RELU6, F_HOLD = 1, 2, 4, 8, 16

# THE OPCODE NUMBERS COLLIDE WITH THE INT8 ENGINE'S AND THE NAMES DO NOT, which is exactly why a
# name table has to travel with its emulator instead of being reached for globally. Opcode 2 is
# RESIDUAL here and MAXPOOL there; flag 4 is F_RELU here and F_HSWISH there. Reading a bf16
# container's packets through the int8 table names the ops wrongly, and engine_compile's
# check_kernel_covers_packets then refuses a container for reaching a case it never reaches.
OP_NAMES = {OP_NOP: "NOP", OP_CONV: "CONV", OP_RESIDUAL: "RESIDUAL"}

# Byte budgets, for the container manifest and for walking a packet blob. Derived from the element
# counts above rather than restated, so a geometry change cannot leave them behind.
# kernels/bf16_conv/design.py owns the same three numbers and must agree.
A_ELEMS = 6400                                   # one activation object
A_BYTES = A_ELEMS * 2                            # 12,800 B, twice the int8 object
O_BYTES = OUT_ELEMS * 2                          # 3,200 B, the same object as int8's, half the
                                                 # channels: 16 at two bytes against 32 at one
W_BYTES = 9472                                   # the weight packet object, identical to int8's

# Candidate models of the one-instruction multiply-accumulate, acc <- acc + sum_k a[k] * w[k]:
#   "aligned"     the accumulator and the eight products are aligned to the largest exponent among
#                 the nine, each rounded SEPARATELY to a 24-bit grid at that exponent (ties to even),
#                 and the rounded operands added exactly
#   "wide"        the nine operands summed exactly, one rounding to fp32
#   "sequential"  acc <- fl(acc + a[k] * w[k]) for k = 0..7, a rounding per product
#   "dot_first"   d <- the eight products summed in order with fp32 rounding, then acc <- fl(acc + d)
#   "exp_sum"     as "aligned", except where the grid sits: a product is placed by the SUM of its two
#                 operands' exponents (ea + ew) rather than by its own leading bit, and the accumulator
#                 by its own exponent. Two significands multiply to [1, 4), so a product in [2, 4)
#                 leads one bit above its place and the grid keeps one more bit below it. Where every
#                 product's significands multiply to under 2 this IS "aligned", which is why the
#                 probe's vectors, all powers of two and small integers against weights of 1, could
#                 not tell them apart. A CANDIDATE, not the model in force: it was fitted on 2026-09-23
#                 to the nine SESR-M7 positions "aligned" misses on silicon (BENCHMARKS, "The AdaRound
#                 mismatch starts in body.1"), and is unconfirmed on the core.
# Every bf16 x bf16 product is exact in fp32 (two 8-bit significands), so the models differ only in
# how the sum rounds. On ordinary data all of them agree after rounding to bf16; they are told apart by
# kernels/bf16_conv/engine_bf16.py --probe.
MAC_MODELS = ("aligned", "wide", "sequential", "dot_first", "exp_sum")
MAC_MODEL = "aligned"


def to_bf16(a: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16, round-to-nearest-even, kept in a float32 container.

    On the bits rather than via ml_dtypes, so the tie-break is visible and the module needs no
    dependency the compiler does not already have. NaN is preserved: adding the rounding bias to a
    NaN payload can otherwise carry it into infinity.
    """
    f = np.ascontiguousarray(a, dtype=np.float32)
    u = f.view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)
    return np.where(np.isnan(f), f, r).astype(np.float32)


def bf16_bits(a: np.ndarray) -> np.ndarray:
    """The 16-bit patterns of bf16-representable float32 values: bf16 is fp32's top half.

    This is what a byte comparison against device memory must use. Comparing the float values
    instead calls -0.0 and +0.0 equal, which they are not in memory.
    """
    f = np.ascontiguousarray(a, dtype=np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16)


def from_bf16_bits(u: np.ndarray) -> np.ndarray:
    """Inverse of `bf16_bits`: widen 16-bit patterns to the float32 values they denote."""
    return (np.ascontiguousarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def make_header(op=OP_CONV, k=3, stride=1, ncin=1, flags=F_EMIT,
                count_out=1, count_acc=0, rows_in=7, cols_in=22, plane_elems=None,
                phases=(0, 0, 0, 0)) -> np.ndarray:
    """Build the 32 int32 words the core reads. plane_elems defaults to a tight rows_in x cols_in."""
    h = np.zeros(HDR_WORDS, np.int32)
    h[H_OP], h[H_K], h[H_STRIDE], h[H_NCIN], h[H_FLAGS] = op, k, stride, ncin, flags
    h[H_COUNT_OUT], h[H_COUNT_ACC] = count_out, count_acc
    h[H_ROWS_IN], h[H_COLS_IN] = rows_in, cols_in
    h[H_PLANE_ELEMS] = rows_in * cols_in * 8 if plane_elems is None else plane_elems
    for i, p in enumerate(phases):
        h[H_PHASE0 + i] = p
    return h


def mac(acc: np.ndarray, a: np.ndarray, w: np.ndarray, model: str) -> np.ndarray:
    """One multiply-accumulate over a whole tile: acc[b][r][x][n] += sum_k a[r][x][k] * w[b][k][n].

    acc float32 [NCO][R][C][4], a float32 [R][C][8], w float32 [NCO][8][4]. Vectorised over every
    lane the hardware treats independently; the sum over k is the only place order can matter.
    """
    if model == "aligned":
        prod = a.astype(np.float64)[None, :, :, :, None] * w.astype(np.float64)[:, None, None, :, :]
        ops = np.concatenate([acc.astype(np.float64)[:, :, :, None, :], prod], axis=3)   # [NCO][R][C][9][4]
        _, ex = np.frexp(np.abs(ops).max(axis=3))          # peak = m * 2**ex with m in [0.5, 1)
        quantum = np.ldexp(1.0, ex - 24)                   # 24 bits below the largest operand's leading bit
        units = np.rint(ops / quantum[:, :, :, None, :]).sum(axis=3)   # rint is ties-to-even; exact in fp64
        # An exactly zero sum is written +0.0; what sign the core gives one is unmeasured.
        return (units * quantum + 0.0).astype(np.float32)
    if model == "wide":
        prod = a.astype(np.float64)[None, :, :, :, None] * w.astype(np.float64)[:, None, None, :, :]
        return (acc.astype(np.float64) + prod.sum(axis=3)).astype(np.float32)
    if model == "sequential":
        out = acc
        for k in range(8):
            out = (out + a[None, :, :, k, None] * w[:, None, None, k, :]).astype(np.float32)
        return out
    if model == "dot_first":
        dot = np.zeros_like(acc)
        for k in range(8):
            dot = (dot + a[None, :, :, k, None] * w[:, None, None, k, :]).astype(np.float32)
        return (acc + dot).astype(np.float32)
    if model == "exp_sum":
        a64, w64, c64 = a.astype(np.float64), w.astype(np.float64), acc.astype(np.float64)
        prod = a64[None, :, :, :, None] * w64[:, None, None, :, :]            # [NCO][R][C][8][4]
        # floor(log2|x|) is frexp's exponent less one; a zero operand places nothing.
        place = (np.frexp(a64)[1] - 1)[None, :, :, :, None] + (np.frexp(w64)[1] - 1)[:, None, None, :, :]
        place = np.where(prod != 0, place, np.iinfo(np.int32).min)
        top = np.maximum(place.max(axis=3), np.where(c64 != 0, np.frexp(c64)[1] - 1, np.iinfo(np.int32).min))
        top = np.where(top == np.iinfo(np.int32).min, 0, top)             # all nine zero: any grid gives 0
        quantum = np.ldexp(1.0, top - 23)                                   # 24 bits down from the top place
        ops = np.concatenate([c64[:, :, :, None, :], prod], axis=3)
        units = np.rint(ops / quantum[:, :, :, None, :]).sum(axis=3)       # ties-to-even; exact in fp64
        return (units * quantum + 0.0).astype(np.float32)
    raise ValueError(f"mac model {model!r} is not one of {MAC_MODELS}")


def epilogue(acc: np.ndarray, flags: int) -> np.ndarray:
    """The core's whole epilogue: round to bf16, then the floor and the ceiling.

    0 and 6 are exact in bf16 and rounding is monotone, so the order is not observable - it is
    kept anyway. F_RELU is the floor and F_RELU6 the ceiling, INDEPENDENTLY, so a plain ReLU is
    the same opcode with the ceiling left off and a Clip(0, 6) sets both. What the core's max
    makes of -0.0 and of NaN is unmeasured; here -0.0 becomes +0.0 and NaN passes through.
    """
    y = to_bf16(acc)
    if flags & F_RELU:
        y = np.where(y > 0, y, np.where(np.isnan(y), y, np.float32(0.0)))
    if flags & F_RELU6:
        y = np.where(y < 6, y, np.where(np.isnan(y), y, np.float32(6.0)))
    return y.astype(np.float32)


def interleave_out(y: np.ndarray) -> np.ndarray:
    """[NCO][rows][cols][4] accumulator blocks -> the flat activation layout, eight a block.

    Accumulator block b, channel n is channel ``(b % 2) * 4 + n`` of 8-channel block ``b // 2``.
    On the core this is one ``interleave_zip`` per block pair - the exact inverse of the
    ``interleave_unzip`` the stride-2 load path already uses, so the idiom is not new here.

    This exists because mmul<4, 8, 4> blocks the OUTPUT by 4 while every activation the engine
    reads is blocked by 8 (``Placement.pitch``). int8 never needed it: mmul<4, 8, 8> emits the
    same width it consumes.
    """
    return (y.reshape(OUT_BLOCKS_8, 2, TILE_ROWS, TILE_COLS, 4)
             .transpose(0, 2, 3, 1, 4)
             .reshape(-1))


def _residual(header: np.ndarray, act: np.ndarray, scratch: np.ndarray, out: np.ndarray) -> None:
    """OP_RESIDUAL: add this packet's A to the held tile, activate, retire.

    Both addends are already in the emitted 8-channel layout - the held tile because an F_HOLD
    packet wrote it that way, this packet's A because the schedule hands the residual branch over
    as an output-shaped tile. The int8 core's ``residual_tile`` does the same add and then spends
    fifteen lines requantizing it; in bf16 the requantization is deleted rather than widened, so
    what is left is the add, the flag epilogue and one rounding on the store.
    """
    flags = int(header[H_FLAGS])
    held = np.ascontiguousarray(scratch, dtype=np.float32)[:OUT_ELEMS]
    res = np.ascontiguousarray(act, dtype=np.float32)[:OUT_ELEMS]
    y = epilogue(held + res, flags)
    if flags & F_HOLD:
        scratch[:OUT_ELEMS] = y
    else:
        out[:OUT_ELEMS] = y


def run_packet(header: np.ndarray, act: np.ndarray, wts: np.ndarray, bias: np.ndarray,
               psum: np.ndarray, out: np.ndarray, scratch: np.ndarray,
               core_row: int = 0, mac_model: str | None = None) -> None:
    """Execute one packet in place, exactly as the core would.

    act     float32 holding bf16-representable values, [ncin * plane_elems]; for OP_RESIDUAL it is
            instead the residual tile, [OUT_ELEMS] in the emitted 8-channel layout
    wts     float32 holding bf16-representable values, [k][k][ncin][NCO][32], 32 = kk*4 + n
    bias    float32 holding bf16-representable values, [NCO][16], element m*4+n is channel n's bias
    psum    float32 [PSUM_FLOATS], the fp32 partial sums
    out     float32 [OUT_ELEMS], modified in place
    scratch float32 [SCRATCH_ELEMS], the hold buffer, modified in place

    An unknown opcode RAISES. It used to fall through to the convolution on both sides at once -
    the core had no switch and this had no else - so a mis-typed packet quietly convolved and
    byte-exactness could not catch it, because the emulator reproduced the bug faithfully. The
    int8 core fails closed on a `default: break;` and so does this.
    """
    model = MAC_MODEL if mac_model is None else mac_model
    op = int(header[H_OP])
    if op == OP_NOP:
        return
    if op == OP_RESIDUAL:
        _residual(header, act, scratch, out)
        return
    if op != OP_CONV:
        raise ValueError(f"unknown opcode {op}; the core's switch has no case for it")
    k, stride, ncin = int(header[H_K]), int(header[H_STRIDE]), int(header[H_NCIN])
    flags, cols_in, plane = int(header[H_FLAGS]), int(header[H_COLS_IN]), int(header[H_PLANE_ELEMS])

    planes = np.ascontiguousarray(act, dtype=np.float32)[:ncin * plane].reshape(ncin, -1, 8)
    w = np.ascontiguousarray(wts, dtype=np.float32).reshape(k, k, ncin, NCO, 8, 4)
    rows = np.arange(TILE_ROWS)
    cols = np.arange(TILE_COLS)

    if flags & F_LOAD_PSUM:
        acc = psum[:NCO * PSUM_BLOCK_ELEMS].reshape(NCO, TILE_ROWS, TILE_COLS, 4).astype(np.float32)
    else:
        # Element m*4+n of a replicated bias block is channel n, so any row of the 4x4 will do.
        acc = np.broadcast_to(bias.reshape(NCO, 4, 4)[:, 0, :].reshape(NCO, 1, 1, 4),
                              (NCO, TILE_ROWS, TILE_COLS, 4)).astype(np.float32)

    # The core's walk: ky, kx, input channel block - one multiply-accumulate each, in this order.
    for ky in range(k):
        for kx in range(k):
            idx = ((rows * stride + ky)[:, None] * cols_in + (cols * stride + kx)[None, :]).reshape(-1)
            for c in range(ncin):
                win = planes[c, idx, :].reshape(TILE_ROWS, TILE_COLS, 8)
                acc = mac(acc, win, w[ky, kx, c], model)

    if flags & (F_EMIT | F_HOLD):
        y = interleave_out(epilogue(acc, flags))
        # F_EMIT retires the tile to the output object; F_HOLD keeps it for a residual packet.
        # Both destinations carry the same layout, so a held tile and an emitted one are the same
        # bytes and OP_RESIDUAL can add them without knowing which produced which.
        if flags & F_EMIT:
            out[:OUT_ELEMS] = y
        else:
            scratch[:OUT_ELEMS] = y
    else:
        psum[:NCO * PSUM_BLOCK_ELEMS] = acc.reshape(-1)


def held_tile(scratch: np.ndarray) -> np.ndarray:
    """The tile an F_HOLD packet left in `scratch`, in the emitted 8-channel layout.

    Every value in it is bf16-representable because it went through `epilogue`, so this is the
    same convention `out` and `act` use: a float32 container holding bf16 values. Use `bf16_bits`
    to compare it against device memory - as floats, -0.0 and +0.0 compare equal and they are not
    the same bytes.
    """
    return np.ascontiguousarray(scratch, dtype=np.float32)[:OUT_ELEMS].copy()
