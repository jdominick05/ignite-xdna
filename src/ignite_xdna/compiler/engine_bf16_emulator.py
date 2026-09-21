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
  * the held tile is bf16 aliased into the tail of psum, so it occupies half as many float32 slots
    as it has elements - stored here as the same bit patterns at the same byte offsets;
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
OUT_BLOCK_ELEMS = TILE_ROWS * TILE_COLS * 4      # 400 bf16
PSUM_BLOCK_ELEMS = TILE_ROWS * TILE_COLS * 4     # 400 float
HOLD_OFFSET_ELEMS = NCO * PSUM_BLOCK_ELEMS       # in float32 slots: the held tile starts past the sums
HOLD_SLOTS = NCO * OUT_BLOCK_ELEMS // 2          # 1,600 bf16 occupy 800 float32 slots
PSUM_FLOATS = HOLD_OFFSET_ELEMS + HOLD_SLOTS     # 2,400 float32 = 9,600 B, the core's psum buffer

H_OP, H_K, H_STRIDE, H_NCIN, H_FLAGS = 0, 1, 2, 3, 4
H_COUNT_OUT, H_COUNT_ACC = 5, 6
H_ROWS_IN, H_COLS_IN, H_PLANE_ELEMS = 7, 8, 9
H_PHASE0 = 12

OP_NOP, OP_CONV, OP_RESIDUAL = 0, 1, 2
F_LOAD_PSUM, F_EMIT, F_RELU, F_RELU6, F_HOLD = 1, 2, 4, 8, 16

# Candidate models of the one-instruction multiply-accumulate, acc <- acc + sum_k a[k] * w[k]:
#   "wide"        the nine operands summed exactly, one rounding to fp32
#   "sequential"  acc <- fl(acc + a[k] * w[k]) for k = 0..7, a rounding per product
#   "dot_first"   d <- the eight products summed in order with fp32 rounding, then acc <- fl(acc + d)
# Every bf16 x bf16 product is exact in fp32 (two 8-bit significands), so the models differ only in
# how the sum rounds.
MAC_MODELS = ("wide", "sequential", "dot_first")
MAC_MODEL = "wide"   # UNMEASURED: set from the silicon probe, not from taste


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
    raise ValueError(f"mac model {model!r} is not one of {MAC_MODELS}")


def run_packet(header: np.ndarray, act: np.ndarray, wts: np.ndarray, bias: np.ndarray,
               psum: np.ndarray, out: np.ndarray, core_row: int = 0, mac_model: str | None = None) -> None:
    """Execute one packet in place, exactly as the core would.

    act   float32 holding bf16-representable values, [ncin * plane_elems]
    wts   float32 holding bf16-representable values, [k][k][ncin][NCO][32], 32 = kk*4 + n
    bias  float32 holding bf16-representable values, [NCO][16], element m*4+n is channel n's bias
    psum  float32 [PSUM_FLOATS]: NCO * PSUM_BLOCK_ELEMS partial sums, then the held tile's bf16 bits
    out   float32 [NCO * OUT_BLOCK_ELEMS], modified in place
    """
    model = MAC_MODEL if mac_model is None else mac_model
    op = int(header[H_OP])
    if op == OP_NOP:
        return
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
        # The core's order: round to bf16, then the floor and the ceiling. 0 and 6 are exact in
        # bf16 and rounding is monotone, so the order is not observable - it is kept anyway.
        y = to_bf16(acc)
        # F_RELU is the floor and F_RELU6 the ceiling, independently, so a plain ReLU is the same
        # opcode with the ceiling left off and a Clip(0, 6) sets both. What the core's max makes of
        # -0.0 and of NaN is unmeasured; here -0.0 becomes +0.0 and NaN passes through.
        if flags & F_RELU:
            y = np.where(y > 0, y, np.where(np.isnan(y), y, np.float32(0.0)))
        if flags & F_RELU6:
            y = np.where(y < 6, y, np.where(np.isnan(y), y, np.float32(6.0)))
        y = y.astype(np.float32).reshape(-1)
        if flags & F_EMIT:
            out[:NCO * OUT_BLOCK_ELEMS] = y
        else:
            # The held tile is bf16 aliased into the tail of psum: the same bits, the same offsets.
            psum[HOLD_OFFSET_ELEMS:HOLD_OFFSET_ELEMS + HOLD_SLOTS].view(np.uint16)[:] = bf16_bits(y)
    else:
        psum[:NCO * PSUM_BLOCK_ELEMS] = acc.reshape(-1)


def held_tile(psum: np.ndarray) -> np.ndarray:
    """The tile a F_HOLD packet left in psum's tail, widened back to float32 values."""
    return from_bf16_bits(psum[HOLD_OFFSET_ELEMS:HOLD_OFFSET_ELEMS + HOLD_SLOTS].view(np.uint16))
