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

OP_NOP, OP_CONV, OP_MAXPOOL, OP_RESIDUAL, OP_FUSED_CONV, OP_MUL, OP_SCALE, OP_POOL = 0, 1, 2, 3, 4, 5, 6, 7
F_LOAD_PSUM, F_EMIT, F_HSWISH, F_UP2, F_HOLD, F_RES_SHIFTS, F_SIGMOID, F_RESIDUAL = 1, 2, 4, 8, 16, 32, 64, 128
SIGMOID_LINES = 4   # line 1 in H_A1/H_B1, lines 2-4 in H_A2..H_B4 (the header's last six words)

TILE_ROWS = 5
TILE_COLS = 20
OUT_BLOCKS = 4
OUT_BLOCK_BYTES = TILE_ROWS * TILE_COLS * 8  # 800

(H_OP, H_K, H_STRIDE, H_NCIN, H_NCO, H_FLAGS, H_SHIFT_OUT, H_A1, H_B1, H_S1,
 H_QMAX, H_K2, H_S2, H_YSH, H_RSH, H_COUNT_OUT, H_COUNT_ACC, H_PHASE0,
 H_ROWS_IN, H_COLS_IN, H_PLANE_BYTES, H_RLSH_M, H_RLSH_R) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                             13, 14, 15, 16, 17, 21, 22, 23, 24, 25)
H_A2, H_B2, H_A3, H_B3, H_A4, H_B4 = 26, 27, 28, 29, 30, 31

# Packet geometries the header advertises: (rows_in, cols_in, plane_bytes, ncin).
# The core always computes four output blocks; ``ncin`` is the number of input
# channel blocks a conv packet reduces over (a maxpool packet's block count).
GEOM_K1 = (5, 20, 800, 8)
GEOM_K1_UP2 = (5, 20, 800, 8)      # header geometry after the in-core expansion
GEOM_K3S1 = (8, 25, 1600, 4)
GEOM_K3S2 = (16, 50, 6400, 1)      # 11 rows x 41 pixels needed, contiguous rows
GEOM_POOL = (16, 25, 3200, 2)
GEOM_RESIDUAL = (5, 20, 800, 4)
# OP_SCALE coefficients: 4 blocks x 32 int16 lanes (8 channel values replicated 4x per block)
# at W_OFFSET in the W packet — 512 bytes, the same region a conv packet's weights occupy.
SCALE_COEF_BYTES = OUT_BLOCKS * 32 * 2
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
class SigmoidParams:
    """Integer constants of the piecewise-linear sigmoid SiLU epilogue (see engine.cc).

    ``lines`` holds SIGMOID_LINES (A, B) pairs; ``s`` is the lines' shift and ``ysh`` the output shift."""
    lines: tuple
    s: int
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
    sig: Optional[SigmoidParams] = None   # with F_SIGMOID; shares H_A1, H_B1, H_S1 and H_YSH with ``hs``
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
        if self.hs is not None and self.sig is not None:
            raise ValueError("a packet carries HardSwish or sigmoid constants, not both")
        if self.hs is not None:
            h[H_A1], h[H_B1], h[H_S1], h[H_QMAX] = self.hs.a1, self.hs.b1, self.hs.s1, self.hs.qmax
            h[H_K2], h[H_S2], h[H_YSH] = self.hs.k2, self.hs.s2, self.hs.ysh
        if self.sig is not None:
            if len(self.sig.lines) != SIGMOID_LINES:
                raise ValueError(f"sigmoid epilogue takes {SIGMOID_LINES} lines, got {len(self.sig.lines)}")
            for (a_idx, b_idx), (a, b) in zip(((H_A1, H_B1), (H_A2, H_B2), (H_A3, H_B3), (H_A4, H_B4)),
                                              self.sig.lines):
                if not 0 <= a < 1 << 15 or not -(1 << 31) <= b < 1 << 31:
                    raise ValueError(f"sigmoid line ({a}, {b}) does not fit an int16 slope and an int32 intercept")
                h[a_idx], h[b_idx] = a, b
            h[H_S1], h[H_YSH] = self.sig.s, self.sig.ysh
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
        sig = None
        if int(h[H_FLAGS]) & F_SIGMOID:
            sig = SigmoidParams(tuple((int(h[a]), int(h[b])) for a, b in
                                      ((H_A1, H_B1), (H_A2, H_B2), (H_A3, H_B3), (H_A4, H_B4))),
                                int(h[H_S1]), int(h[H_YSH]))
            hs = None
        return cls(op=int(h[H_OP]), k=int(h[H_K]), stride=int(h[H_STRIDE]), ncin=int(h[H_NCIN]),
                   nco=int(h[H_NCO]), flags=int(h[H_FLAGS]), shift_out=int(h[H_SHIFT_OUT]), hs=hs, sig=sig,
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


def sigmoid_epilogue(q1: np.ndarray, sig: SigmoidParams) -> np.ndarray:
    """uint8 linear conv output -> uint8 SiLU output through the piecewise-linear sigmoid, as the core computes it.

    u = |t|;  g = clip(min_i sat16(rne((u * A_i + B_i) >> S)), 0, 64);  y = sat16(rne(((t << 6) + u * g) >> YSH)),
    which is t * (64 + sign(t) * g): a sigmoid in 1/128 steps that reaches 1.0."""
    t = q1.astype(np.int64) - 128
    u = np.abs(t)
    g = None
    for a, b in sig.lines:
        line = sat_i16(rne_shift(u * a + b, sig.s))
        g = line if g is None else np.minimum(g, line)
    g = np.minimum(np.maximum(g, 0), 64)
    y = sat_i16(rne_shift((t << 6) + u * g, sig.ysh))
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


def mul_combine(qm: np.ndarray, qr: np.ndarray, ysh: int) -> np.ndarray:
    """Elementwise product of the uint8 held tile ``qm`` and the uint8 operand tile ``qr``:
    ``sat_u8(sat16(rne((tm * tr) >> ysh)) + 128)`` with centered operands (see engine.cc OP_MUL)."""
    tm = qm.astype(np.int64) - 128
    tr = qr.astype(np.int64) - 128
    y = sat_i16(rne_shift(tm * tr, ysh))
    return sat_u8(y + 128)


def scale_apply(t_tile: np.ndarray, coef: np.ndarray, ysh: int) -> np.ndarray:
    """Per-channel scalar gain on the centered int64 tile ``t_tile`` [blocks][pixels]:
    ``sat_u8(sat16(rne((t * co) >> ysh)) + 128)`` with int16 coefficients (see engine.cc OP_SCALE)."""
    y = sat_i16(rne_shift(t_tile * coef, ysh))
    return sat_u8(y + 128)


def pack_w_packet(hdr: PacketHeader, bias: np.ndarray, weights: Optional[np.ndarray],
                  extra: Optional[np.ndarray] = None) -> np.ndarray:
    """Serialize a W packet. ``weights`` is int8 [taps][ncin][4 blocks][8 ci][8 co]; ``extra`` is a raw
    byte payload for the same W_OFFSET region (an OP_SCALE packet's int16 coefficients)."""
    pkt = np.zeros(W_BYTES, dtype=np.uint8)
    pkt[:HDR_BYTES] = hdr.words().view(np.uint8)
    b = np.zeros(32, dtype=np.int32)
    bias = np.asarray(bias, dtype=np.int32).ravel()
    b[:bias.size] = bias
    pkt[BIAS_OFFSET:BIAS_OFFSET + 128] = b.view(np.uint8)
    if weights is not None and extra is not None:
        raise ValueError("a packet carries weights or a raw payload, not both")
    if weights is not None:
        w = np.asarray(weights, dtype=np.int8)
        expected = (hdr.k * hdr.k, hdr.ncin, OUT_BLOCKS, 8, 8)
        if w.shape != expected:
            raise ValueError(f"weights shape {w.shape} != {expected}")
        raw = w.ravel().view(np.uint8)
    elif extra is not None:
        raw = np.asarray(extra, dtype=np.uint8).ravel()
    else:
        raw = None
    if raw is not None:
        if raw.size > W_MAX_BYTES:
            raise ValueError(f"{raw.size} payload bytes exceed the {W_MAX_BYTES}-byte packet budget")
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
            if hdr.flags & F_SIGMOID:   # the core's separate loop over the finished tile, after the passes
                q = sigmoid_epilogue(q, hdr.sig)
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
            q = residual_combine(state.hold, res, hdr.rsh, hdr.rlsh_m, hdr.rlsh_r)
        else:
            q = residual_combine(state.hold, res, hdr.rsh)
        if hdr.flags & F_HSWISH:  # activation after the add
            q = hswish_epilogue(q, hdr.hs)
        out[:] = q
    elif hdr.op == OP_MUL:
        hs = hdr.hs or HardSwishParams(0, 0, 0, 0, 0, 0, 0)
        res = a[:OUT_BLOCKS * OUT_BLOCK_BYTES].reshape(OUT_BLOCKS, TILE_ROWS, TILE_COLS, 8)
        out[:] = mul_combine(state.hold, res, hs.ysh)
    elif hdr.op == OP_SCALE:
        hs = hdr.hs or HardSwishParams(0, 0, 0, 0, 0, 0, 0)
        coef = np.asarray(wpkt)[W_OFFSET:W_OFFSET + SCALE_COEF_BYTES].view(np.int16).reshape(OUT_BLOCKS, 32)
        co = np.tile(coef, (1, OUT_BLOCK_BYTES // 32)).reshape(OUT_BLOCKS, TILE_ROWS, TILE_COLS, 8)
        t = a[:OUT_BLOCKS * OUT_BLOCK_BYTES].reshape(OUT_BLOCKS, TILE_ROWS, TILE_COLS, 8).astype(np.int64) - 128
        q = scale_apply(t, co, hs.ysh)
        if hdr.flags & F_EMIT:
            out[:] = q
        else:  # the core's ternary: without F_EMIT the tile lands in the hold buffer
            state.hold[:] = q
    elif hdr.op == OP_POOL:
        # k x k stride-1 average pool on the max pool's two-packet geometry; the reciprocal
        # K2/S2 divides the window sum. Mirrors avgpool_tile's int16 accumulation (k <= 5).
        hs = hdr.hs or HardSwishParams(0, 0, 0, 0, 0, 0, 0)
        if hdr.k > 5:
            raise ValueError(f"average pool window {hdr.k}x{hdr.k} exceeds the int16 sum bound (k <= 5)")
        pooled = np.zeros((2, TILE_ROWS, TILE_COLS, 8), dtype=np.uint8)
        for b in range(2):
            plane = a[b * hdr.plane_bytes:(b + 1) * hdr.plane_bytes].reshape(hdr.rows_in, hdr.cols_in, 8)
            sums = np.zeros((TILE_ROWS, TILE_COLS, 8), dtype=np.int64)
            for dy in range(hdr.k):
                for dx in range(hdr.k):
                    sums += plane[dy:dy + TILE_ROWS, dx:dx + TILE_COLS, :].astype(np.int64)
            pooled[b] = sat_u8(rne_shift(sums * hs.k2, hs.s2))
        if emit:
            out[0:2] = state.hold[0:2]
            out[2:4] = pooled
        else:
            state.hold[0:2] = pooled
    elif hdr.op == OP_FUSED_CONV:
        # Fused Stage 1 (Conv3x3 + HardSwish) -> Stage 2 (Conv3x3 + HardSwish + Residual)
        y_quad, x0, H, W = hdr.phases
        y0 = y_quad + TILE_ROWS * core_row
        r_start = 1 if y0 == 0 else 0
        r_end = 6 if y0 + TILE_ROWS >= H else 7
        c_start = 1 if x0 == 0 else 0
        c_end = 21 if x0 + TILE_COLS >= W else 22

        w_words = wpkt.view(np.int32)
        w1_off = int(w_words[H_A3])
        s1_bytes = wpkt[w1_off:w1_off + 2400]
        s1_params = s1_bytes[:32].view(np.int32)
        hs1 = HardSwishParams(
            a1=int(s1_params[0]), b1=int(s1_params[1]), s1=int(s1_params[2]),
            qmax=int(s1_params[3]), k2=int(s1_params[4]), s2=int(s1_params[5]),
            ysh=int(s1_params[6]),
        )
        shift_out1 = int(s1_params[7])
        b1 = s1_bytes[32:96].view(np.int32)
        w1 = s1_bytes[96:96 + 2304].view(np.int8).reshape(9, 2, 2, 8, 8)

        mid = np.full((2, 7, 24, 8), 128, dtype=np.uint8)
        cols_in = hdr.cols_in
        plane_bytes = hdr.plane_bytes

        for rm in range(7):
            if rm < r_start or rm >= r_end:
                mid[:, rm, :, :] = 128
                continue
            for g in range(6):
                cm0 = 18 if g == 5 else g * 4
                acc0 = np.full((4, 8), b1[:8], dtype=np.int64)
                acc1 = np.full((4, 8), b1[8:16], dtype=np.int64)
                for ky in range(3):
                    for kx in range(3):
                        tap = ky * 3 + kx
                        for c in range(2):
                            aoff = ((rm + ky) * cols_in + cm0 + kx) * 8
                            plane = a[c * plane_bytes:(c + 1) * plane_bytes]
                            av = plane[aoff:aoff + 32].reshape(4, 8)
                            acc0 += av.astype(np.int64) @ w1[tap, c, 0].astype(np.int64)
                            acc1 += av.astype(np.int64) @ w1[tap, c, 1].astype(np.int64)
                q0 = sat_u8(rne_shift(acc0, shift_out1))
                q0 = hswish_epilogue(q0, hs1)
                q1 = sat_u8(rne_shift(acc1, shift_out1))
                q1 = hswish_epilogue(q1, hs1)
                mid[0, rm, cm0:cm0 + 4, :] = q0
                mid[1, rm, cm0:cm0 + 4, :] = q1
            if c_start == 1:
                mid[:, rm, 0, :] = 128
            if c_end == 21:
                mid[:, rm, 21, :] = 128

        w2_off = int(w_words[H_A4])
        w2 = wpkt[w2_off:w2_off + 4608].view(np.int8).reshape(9, 2, 4, 8, 8)
        bias2 = bias

        for r in range(TILE_ROWS):
            for g in range(5):
                g0 = g * 4
                acc = np.zeros((4, 4, 8), dtype=np.int64)
                for b in range(4):
                    acc[b, :, :] = bias2[b * 8:(b + 1) * 8]
                for ky in range(3):
                    for kx in range(3):
                        tap = ky * 3 + kx
                        mid_r = r + ky
                        mid_c = g0 + kx
                        for c in range(2):
                            av = mid[c, mid_r, mid_c:mid_c + 4, :].reshape(4, 8)
                            for b in range(4):
                                acc[b, :, :] += av.astype(np.int64) @ w2[tap, c, b].astype(np.int64)
                for b in range(4):
                    q = sat_u8(rne_shift(acc[b], hdr.shift_out))
                    if hdr.flags & F_HSWISH:
                        q = hswish_epilogue(q, hdr.hs)
                    if (hdr.flags & F_RESIDUAL) and b < 2:
                        res_off = ((2 + r) * cols_in + 2 + g0) * 8
                        qres = a[b * plane_bytes + res_off:b * plane_bytes + res_off + 32].reshape(4, 8)
                        if hdr.flags & F_RES_SHIFTS:
                            q = residual_combine(q, qres, hdr.rsh, hdr.rlsh_m, hdr.rlsh_r)
                        else:
                            q = residual_combine(q, qres, hdr.rsh)
                    out[b, r, g0:g0 + 4, :] = q
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
