"""Bit-exact NumPy emulation of the Phoenix convolution engine core program.

Every rule here mirrors ``kernels/aie2/conv_engine/engine.cc``: the packet
formats, the accumulator arithmetic, the round-half-to-even shifts, the
int16/uint8 saturations and the scratch state a core keeps between packets.
The silicon result of a packet must equal ``run_packet`` byte for byte; the
compiler uses the same functions to check a layer's HardSwish constants and
to produce the reference activations of a whole graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

W_BYTES = 9472
A_BYTES = 6400
O_BYTES = 3200
PSUM_BYTES = 16000
HDR_BYTES = 128
BIAS_OFFSET = 128
W_OFFSET = 256
W_MAX_BYTES = W_BYTES - W_OFFSET  # 9,216

OP_NOP, OP_CONV, OP_MAXPOOL, OP_RESIDUAL = 0, 1, 2, 3
F_LOAD_PSUM, F_EMIT, F_HSWISH, F_UP2, F_HOLD, F_RES_SHIFTS = 1, 2, 4, 8, 16, 32

TILE_ROWS = 5
TILE_COLS = 20
OUT_BLOCKS = 4
OUT_BLOCK_BYTES = TILE_ROWS * TILE_COLS * 8  # 800

(H_OP, H_K, H_STRIDE, H_NCIN, H_NCO, H_FLAGS, H_SHIFT_OUT, H_A1, H_B1, H_S1,
 H_QMAX, H_K2, H_S2, H_YSH, H_RSH, H_COUNT_OUT, H_COUNT_ACC, H_PHASE0,
 H_ROWS_IN, H_COLS_IN, H_PLANE_BYTES, H_RLSH_M, H_RLSH_R) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                             13, 14, 15, 16, 17, 21, 22, 23, 24, 25)

# Packet geometries the header advertises: (rows_in, cols_in, plane_bytes, ncin).
# The core always computes four output blocks; ``ncin`` is the number of input
# channel blocks a conv packet reduces over (a maxpool packet's block count).
GEOM_K1 = (5, 20, 800, 8)
GEOM_K1_UP2 = (5, 20, 800, 8)      # header geometry after the in-core expansion
GEOM_K3S1 = (8, 25, 1600, 4)
GEOM_K3S2 = (16, 50, 6400, 1)      # 11 rows x 41 pixels needed, contiguous rows
GEOM_POOL = (16, 25, 3200, 2)
GEOM_RESIDUAL = (5, 20, 800, 4)
# Raw layout of an up2 source packet before expansion: 10 blocks of [4][20][8].
UP2_SRC_ROWS, UP2_SRC_COLS, UP2_SRC_BLOCK_BYTES = 4, 20, 640


@dataclass
class HardSwishParams:
    """Integer constants of the HardSwish epilogue (see engine.cc)."""
    a1: int
    b1: int
    s1: int
    qmax: int
    k2: int
    s2: int
    ysh: int


@dataclass
class PacketHeader:
    op: int = OP_CONV
    k: int = 1
    stride: int = 1
    ncin: int = 1
    nco: int = 1
    flags: int = F_EMIT
    shift_out: int = 0
    hs: Optional[HardSwishParams] = None
    rsh: int = 0
    count_out: int = 1
    count_acc: int = 0
    phases: tuple = (0, 0, 0, 0)
    rows_in: int = 5
    cols_in: int = 20
    plane_bytes: int = 800
    rlsh_m: int = 0      # residual: left shift of the held tile (read with F_RES_SHIFTS)
    rlsh_r: int = 0      # residual: left shift of the residual tile (read with F_RES_SHIFTS)

    def words(self) -> np.ndarray:
        h = np.zeros(HDR_BYTES // 4, dtype=np.int32)
        h[H_OP], h[H_K], h[H_STRIDE], h[H_NCIN], h[H_NCO] = self.op, self.k, self.stride, self.ncin, self.nco
        h[H_FLAGS], h[H_SHIFT_OUT] = self.flags, self.shift_out
        if self.hs is not None:
            h[H_A1], h[H_B1], h[H_S1], h[H_QMAX] = self.hs.a1, self.hs.b1, self.hs.s1, self.hs.qmax
            h[H_K2], h[H_S2], h[H_YSH] = self.hs.k2, self.hs.s2, self.hs.ysh
        h[H_RSH], h[H_COUNT_OUT], h[H_COUNT_ACC] = self.rsh, self.count_out, self.count_acc
        for i in range(4):
            h[H_PHASE0 + i] = self.phases[i]
        h[H_ROWS_IN], h[H_COLS_IN], h[H_PLANE_BYTES] = self.rows_in, self.cols_in, self.plane_bytes
        h[H_RLSH_M], h[H_RLSH_R] = self.rlsh_m, self.rlsh_r
        return h

    @classmethod
    def from_words(cls, h: np.ndarray) -> "PacketHeader":
        h = np.asarray(h, dtype=np.int32)
        hs = HardSwishParams(int(h[H_A1]), int(h[H_B1]), int(h[H_S1]), int(h[H_QMAX]),
                             int(h[H_K2]), int(h[H_S2]), int(h[H_YSH]))
        return cls(op=int(h[H_OP]), k=int(h[H_K]), stride=int(h[H_STRIDE]), ncin=int(h[H_NCIN]),
                   nco=int(h[H_NCO]), flags=int(h[H_FLAGS]), shift_out=int(h[H_SHIFT_OUT]), hs=hs,
                   rsh=int(h[H_RSH]), count_out=int(h[H_COUNT_OUT]), count_acc=int(h[H_COUNT_ACC]),
                   phases=tuple(int(h[H_PHASE0 + i]) for i in range(4)), rows_in=int(h[H_ROWS_IN]),
                   cols_in=int(h[H_COLS_IN]), plane_bytes=int(h[H_PLANE_BYTES]),
                   rlsh_m=int(h[H_RLSH_M]), rlsh_r=int(h[H_RLSH_R]))


def rne_shift(x: np.ndarray, s: int) -> np.ndarray:
    """Arithmetic right shift by ``s`` rounding half to even (AIE conv_even)."""
    x = np.asarray(x, dtype=np.int64)
    if s <= 0:
        return x << (-s)
    q = x >> s
    rem = x - (q << s)
    half = 1 << (s - 1)
    q = q + ((rem > half) | ((rem == half) & ((q & 1) == 1))).astype(np.int64)
    return q


def sat_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.int64), 0, 255).astype(np.uint8)


def sat_i16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.int64), -32768, 32767)


def hswish_epilogue(q1: np.ndarray, hs: HardSwishParams) -> np.ndarray:
    """uint8 conv output -> uint8 activated output, exactly as the core computes it."""
    t = q1.astype(np.int64) - 128
    h = sat_i16(rne_shift(t * hs.a1 + hs.b1, hs.s1))
    h = np.clip(h, 0, hs.qmax)
    qh = np.minimum(sat_i16(rne_shift(h * hs.k2, hs.s2)), 127)
    y = sat_i16(rne_shift(t * qh, hs.ysh))
    return sat_u8(y + 128)


def residual_combine(qm: np.ndarray, qr: np.ndarray, rsh: int, lsh_m: int = 0,
                     lsh_r: Optional[int] = None) -> np.ndarray:
    """uint8 held tile ``qm`` + uint8 residual tile ``qr``: rne((tm << lsh_m) + (tr << lsh_r), rsh).

    ``lsh_r=None`` is the original rule (residual shifted by ``rsh``, held tile unshifted)."""
    lsh_r = rsh if lsh_r is None else lsh_r
    tm = qm.astype(np.int64) - 128
    tr = qr.astype(np.int64) - 128
    y = sat_i16(rne_shift((tm << lsh_m) + (tr << lsh_r), rsh))
    return sat_u8(y + 128)


def pack_w_packet(hdr: PacketHeader, bias: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    """Serialize a W packet. ``weights`` is int8 [taps][ncin][4 blocks][8 ci][8 co]."""
    pkt = np.zeros(W_BYTES, dtype=np.uint8)
    pkt[:HDR_BYTES] = hdr.words().view(np.uint8)
    b = np.zeros(32, dtype=np.int32)
    bias = np.asarray(bias, dtype=np.int32).ravel()
    b[:bias.size] = bias
    pkt[BIAS_OFFSET:BIAS_OFFSET + 128] = b.view(np.uint8)
    if weights is not None:
        w = np.asarray(weights, dtype=np.int8)
        expected = (hdr.k * hdr.k, hdr.ncin, OUT_BLOCKS, 8, 8)
        if w.shape != expected:
            raise ValueError(f"weights shape {w.shape} != {expected}")
        raw = w.ravel().view(np.uint8)
        if raw.size > W_MAX_BYTES:
            raise ValueError(f"{raw.size} weight bytes exceed the {W_MAX_BYTES}-byte packet budget")
        pkt[W_OFFSET:W_OFFSET + raw.size] = raw
    return pkt


def unpack_w_packet(pkt: np.ndarray):
    pkt = np.asarray(pkt, dtype=np.uint8)
    hdr = PacketHeader.from_words(pkt[:HDR_BYTES].view(np.int32))
    bias = pkt[BIAS_OFFSET:BIAS_OFFSET + 128].view(np.int32).copy()
    n = hdr.k * hdr.k * hdr.ncin * OUT_BLOCKS * 64
    weights = pkt[W_OFFSET:W_OFFSET + n].view(np.int8).reshape(hdr.k * hdr.k, hdr.ncin, OUT_BLOCKS, 8, 8).copy() \
        if hdr.op == OP_CONV else None
    return hdr, bias, weights


@dataclass
class CoreState:
    """Scratch a core keeps across the packets of one tile."""
    psum: np.ndarray = field(default_factory=lambda: np.zeros((OUT_BLOCKS, TILE_ROWS, TILE_COLS, 8), dtype=np.int64))
    hold: np.ndarray = field(default_factory=lambda: np.zeros((OUT_BLOCKS, TILE_ROWS, TILE_COLS, 8), dtype=np.uint8))


def up2_expand(a: np.ndarray, phase: int) -> np.ndarray:
    """Expand a [10][4][20][8] source packet into the [8][5][20][8] tile the core builds in place."""
    src = np.asarray(a, dtype=np.uint8)[:10 * UP2_SRC_BLOCK_BYTES].reshape(10, UP2_SRC_ROWS, UP2_SRC_COLS, 8)
    rows = [(r + phase) >> 1 for r in range(TILE_ROWS)]
    cols = [x >> 1 for x in range(TILE_COLS)]
    tile = src[:8][:, rows][:, :, cols]
    out = np.zeros(A_BYTES, dtype=np.uint8)
    out[:8 * OUT_BLOCK_BYTES] = tile.reshape(-1)
    return out


def _gather_conv_input(hdr: PacketHeader, a: np.ndarray, core_row: int) -> np.ndarray:
    """Return the A operands as [taps][ncin][5][20][8] uint8 exactly as the core reads them."""
    a = np.asarray(a, dtype=np.uint8)
    phase = hdr.phases[core_row & 3]
    if hdr.flags & F_UP2:
        a = up2_expand(a, phase)
    taps = hdr.k * hdr.k
    out = np.zeros((taps, hdr.ncin, TILE_ROWS, TILE_COLS, 8), dtype=np.uint8)
    rows_in, cols_in, pb = hdr.rows_in, hdr.cols_in, hdr.plane_bytes
    for c in range(hdr.ncin):
        base = c * pb
        for tap in range(taps):
            ky, kx = divmod(tap, hdr.k)
            for r in range(TILE_ROWS):
                for x in range(TILE_COLS):
                    if hdr.stride == 2:
                        row = 2 * r + ky
                        col = 2 * x + kx
                        off = base + (row * cols_in + col) * 8
                    else:
                        row = r + ky
                        col = x + kx
                        off = base + (row * cols_in + col) * 8
                    out[tap, c, r, x, :] = a[off:off + 8]
    return out


def run_packet(wpkt: np.ndarray, apkt: np.ndarray, state: CoreState, core_row: int) -> Optional[np.ndarray]:
    """Emulate one packet; return the 3,200-byte output object when the header emits one."""
    hdr, bias, weights = unpack_w_packet(wpkt)
    a = np.asarray(apkt, dtype=np.uint8)
    if a.size != A_BYTES:
        raise ValueError(f"A packet must be {A_BYTES} bytes, got {a.size}")
    out = np.zeros((OUT_BLOCKS, TILE_ROWS, TILE_COLS, 8), dtype=np.uint8)
    emit = bool(hdr.flags & F_EMIT)
    if hdr.op == OP_CONV:
        gathered = _gather_conv_input(hdr, a, core_row)  # [taps][ncin][5][20][8]
        # acc[b, r, x, co] = init + sum_{tap,c,ci} A[tap,c,r,x,ci] * W[tap,c,b,ci,co]
        acc = np.einsum("tcrxi,tcbio->brxo", gathered.astype(np.int64), weights.astype(np.int64))
        if hdr.flags & F_LOAD_PSUM:
            init = state.psum
        else:
            init = bias[:OUT_BLOCKS * 8].astype(np.int64).reshape(OUT_BLOCKS, 1, 1, 8)
        acc = acc + init
        if hdr.flags & (F_EMIT | F_HOLD):
            q = sat_u8(rne_shift(acc, hdr.shift_out))
            if hdr.flags & F_HSWISH:
                q = hswish_epilogue(q, hdr.hs)
            if emit:
                out[:] = q
            else:
                state.hold[:] = q
        else:
            state.psum[:] = acc
    elif hdr.op == OP_MAXPOOL:
        # Two packets per tile: the first holds blocks 0-1, the second emits all four.
        pooled = np.zeros((2, TILE_ROWS, TILE_COLS, 8), dtype=np.uint8)
        for b in range(2):
            plane = a[b * hdr.plane_bytes:(b + 1) * hdr.plane_bytes].reshape(hdr.rows_in, hdr.cols_in, 8)
            m = np.zeros((TILE_ROWS, TILE_COLS, 8), dtype=np.uint8)
            for dy in range(5):
                for dx in range(5):
                    m = np.maximum(m, plane[dy:dy + TILE_ROWS, dx:dx + TILE_COLS, :])
            pooled[b] = m
        if emit:
            out[0:2] = state.hold[0:2]
            out[2:4] = pooled
        else:
            state.hold[0:2] = pooled
    elif hdr.op == OP_RESIDUAL:
        res = a[:OUT_BLOCKS * OUT_BLOCK_BYTES].reshape(OUT_BLOCKS, TILE_ROWS, TILE_COLS, 8)
        if hdr.flags & F_RES_SHIFTS:
            out[:] = residual_combine(state.hold, res, hdr.rsh, hdr.rlsh_m, hdr.rlsh_r)
        else:
            out[:] = residual_combine(state.hold, res, hdr.rsh)
    elif hdr.op == OP_NOP:
        pass
    else:
        raise ValueError(f"unknown op {hdr.op}")
    return out.reshape(-1) if emit else None


def run_sequence(wpkts, apkts, core_row: int, state: Optional[CoreState] = None):
    """Emulate a core's packet stream: a list of (W packet, [A packets...]) pairs."""
    state = state or CoreState()
    outputs = []
    for wpkt, a_list in zip(wpkts, apkts):
        for apkt in a_list:
            o = run_packet(wpkt, apkt, state, core_row)
            if o is not None:
                outputs.append(o)
    return outputs, state
