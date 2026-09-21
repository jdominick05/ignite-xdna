"""A NumPy model of one bf16 engine packet, exact to the bit against the core program.

The int8 engine's emulator earns its keep by asserting that the silicon result of a packet equals
`run_packet` byte for byte, and that contract is what makes a wrong layer findable at all. It is
worth more here than it was there, because a bf16 convolution engine has no reference
implementation anywhere to compare against - not in this repo, not in mlir-aie's aie_kernels, not in
its programming_examples.

This mirrors kernels/bf16_conv/engine_bf16.cc line for line, including the things that look like
details and are not:

  * the accumulation is fp32 from bf16 operands, which is what mmul<4,8,4> does;
  * every value that the core STORES is rounded to bf16 round-to-nearest-even, and every value it
    keeps in psum stays fp32;
  * the bias is read pre-replicated to the accumulator's 4x4 shape, so element m*4+n is the bias of
    output channel n - replication happens on the host because bf16 has no 4-element load.

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
HOLD_OFFSET_ELEMS = NCO * PSUM_BLOCK_ELEMS

H_OP, H_K, H_STRIDE, H_NCIN, H_FLAGS = 0, 1, 2, 3, 4
H_COUNT_OUT, H_COUNT_ACC = 5, 6
H_ROWS_IN, H_COLS_IN, H_PLANE_ELEMS = 7, 8, 9
H_PHASE0 = 12

OP_NOP, OP_CONV, OP_RESIDUAL = 0, 1, 2
F_LOAD_PSUM, F_EMIT, F_RELU, F_RELU6, F_HOLD = 1, 2, 4, 8, 16


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


def run_packet(header: np.ndarray, act: np.ndarray, wts: np.ndarray, bias: np.ndarray,
               psum: np.ndarray, out: np.ndarray, core_row: int = 0) -> None:
    """Execute one packet in place, exactly as the core would.

    act   float32 holding bf16-representable values, [ncin * plane_elems]
    wts   float32 holding bf16-representable values, [k][k][ncin][NCO][32], 32 = kk*4 + n
    bias  float32 holding bf16-representable values, [NCO][16], element m*4+n is channel n's bias
    psum  float32 [NCO * PSUM_BLOCK_ELEMS + held tile], modified in place
    out   float32 [NCO * OUT_BLOCK_ELEMS], modified in place
    """
    op = int(header[H_OP])
    if op == OP_NOP:
        return
    k, stride, ncin = int(header[H_K]), int(header[H_STRIDE]), int(header[H_NCIN])
    flags, cols_in, plane = int(header[H_FLAGS]), int(header[H_COLS_IN]), int(header[H_PLANE_ELEMS])

    a = act.reshape(ncin, plane)
    w = wts.reshape(k, k, ncin, NCO, 8, 4)

    # Gather every (row, column) the tile produces, as index arrays, so the convolution itself is a
    # single einsum per tap rather than a Python loop over pixels.
    rows = np.arange(TILE_ROWS)
    cols = np.arange(TILE_COLS)

    acc = np.zeros((NCO, TILE_ROWS, TILE_COLS, 4), np.float32)
    if flags & F_LOAD_PSUM:
        acc[:] = psum[:NCO * PSUM_BLOCK_ELEMS].reshape(NCO, TILE_ROWS, TILE_COLS, 4)
    else:
        # Element m*4+n of a replicated bias block is channel n, so any row of the 4x4 will do.
        acc[:] = bias.reshape(NCO, 4, 4)[:, 0, :].reshape(NCO, 1, 1, 4)

    for ky in range(k):
        for kx in range(k):
            r_in = rows * stride + ky                      # [TILE_ROWS]
            c_in = cols * stride + kx                      # [TILE_COLS]
            # act plane laid out [row][col][8]; index it as such and take the window.
            planes = a.reshape(ncin, -1, 8)
            idx = (r_in[:, None] * cols_in + c_in[None, :])           # [R][C]
            win = planes[:, idx.reshape(-1), :].reshape(ncin, TILE_ROWS, TILE_COLS, 8)
            # [cin][R][C][kk] x [cin][NCO][kk][n] -> [NCO][R][C][n]
            acc += np.einsum("crxk,cbkn->brxn", win, w[ky, kx], optimize=True).astype(np.float32)

    if flags & (F_EMIT | F_HOLD):
        y = acc
        # F_RELU is the floor and F_RELU6 the ceiling, independently, so a plain ReLU is the same
        # opcode with the ceiling left off and a Clip(0, 6) sets both.
        if flags & F_RELU:
            y = np.maximum(y, 0.0)
        if flags & F_RELU6:
            y = np.minimum(y, 6.0)
        y = to_bf16(y).reshape(-1)
        if flags & F_EMIT:
            out[:NCO * OUT_BLOCK_ELEMS] = y
        else:
            # KNOWN DIVERGENCE, and it must be closed before OP_RESIDUAL exists on either side.
            # The core stores the held tile as bf16 aliased into the tail of psum, so it occupies
            # half as many float32 slots as this does. Nothing reads it back yet, so byte equality
            # on an emitted tile is unaffected; a residual packet would be the first to care.
            psum[HOLD_OFFSET_ELEMS:HOLD_OFFSET_ELEMS + NCO * OUT_BLOCK_ELEMS] = y
    else:
        psum[:NCO * PSUM_BLOCK_ELEMS] = acc.reshape(-1)
