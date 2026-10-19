"""DDR layout of the whole-array M=1 GEMV: weights packed once at load, x objects written per call.

Geometry. Sixteen cores, four per column; each column's two shim MM2S channels each feed two cores through a MemTile
split. Core j of column c makes output rows ``c*N/4 + g*48 + j*12 + [0, 12)`` for its groups g, so the MemTile join
of a column's four cores writes the column's quarter of y in natural order and y needs no host permutation.

Streams. A call reads, on each of the 8 channels, one x object pair and then that channel's weight stream: for each
group g and K chunk q, the chunk of core 2h then the chunk of core 2h+1, each ``OBJ_WORDS`` long in mmul<4,8,4> order
``[kstep 24][acc 3][kk 8][n 4]``. A GEMV's weight block is the 8 streams back to back, (c, h) in order, so it is
exactly N*K bf16 with no padding, and ``block_words`` is N*K/2.

Every array here holds bf16 as its uint16 bit pattern; a word is two of them, little-endian (element 2i is the low
half of word i), which is how the core's ``reinterpret_cast<const bfloat16 *>`` reads it.
"""
from __future__ import annotations

import numpy as np

COLS, ROWS = 4, 4
CORES = COLS * ROWS
CHANNELS = COLS * 2
OUT_GROUP = 12                       # outputs one core makes per group: three mmul accumulators of 4
KCHUNK = 192                         # inputs per weight object; gcd(576, 1536), so no SmolLM2 shape pads
OBJ_ELEMS = OUT_GROUP * KCHUNK       # 2304 bf16
OBJ_WORDS = OBJ_ELEMS // 2           # 1152
HDR_WORDS = 16                       # the header is the LAST HDR_WORDS of a call's first object; x is at its front
MAX_K = (OBJ_WORDS - HDR_WORDS) * 2 // KCHUNK * KCHUNK
X_SLOT_WORDS = 2 * OBJ_WORDS         # one x object per core of a channel's pair
N_QUANTUM = CORES * OUT_GROUP        # 192


def check_shape(n: int, k: int) -> None:
    if n <= 0 or n % N_QUANTUM:
        raise ValueError(f"N={n} is not a positive multiple of {N_QUANTUM} (16 cores x {OUT_GROUP} outputs)")
    if k <= 0 or k % KCHUNK or k > MAX_K:
        raise ValueError(f"K={k} is not a positive multiple of {KCHUNK} up to {MAX_K} (one x object)")


def groups(n: int) -> int:
    """Groups of 12 outputs each core makes."""
    return n // N_QUANTUM


def chunks(k: int) -> int:
    return k // KCHUNK


def block_words(n: int, k: int) -> int:
    check_shape(n, k)
    return n * k // 2


def stream_elems(n: int, k: int) -> int:
    """bf16 elements one channel reads for the weights of one call."""
    return groups(n) * chunks(k) * 2 * OBJ_ELEMS


def core_rows(n: int, c: int, j: int, g: int) -> slice:
    base = c * (n // COLS) + g * (ROWS * OUT_GROUP) + j * OUT_GROUP
    return slice(base, base + OUT_GROUP)


def chunk(w: np.ndarray, rows: slice, q: int) -> np.ndarray:
    """One weight object: W[rows, q*192:(q+1)*192] as [kstep][acc][kk][n], flattened."""
    t = w[rows, q * KCHUNK:(q + 1) * KCHUNK]                      # [12][192] = [acc 3][n 4][kstep 24][kk 8]
    return t.reshape(3, 4, KCHUNK // 8, 8).transpose(2, 0, 3, 1).reshape(-1)


def pack(w: np.ndarray) -> np.ndarray:
    """uint16 [N][K] bf16 bits -> uint32 [N*K/2]: the GEMV's weight block, 8 channel streams back to back."""
    w = np.ascontiguousarray(w)
    if w.dtype != np.uint16 or w.ndim != 2:
        raise TypeError("pack takes bf16 bits as a 2-D uint16 array [N][K]")
    n, k = w.shape
    check_shape(n, k)
    out = np.empty(n * k, np.uint16)
    pos = 0
    for c in range(COLS):
        for h in range(2):
            for g in range(groups(n)):
                for q in range(chunks(k)):
                    for e in range(2):
                        piece = chunk(w, core_rows(n, c, 2 * h + e, g), q)
                        out[pos:pos + OBJ_ELEMS] = piece
                        pos += OBJ_ELEMS
    assert pos == n * k
    return out.view(np.uint32)


# Header words, the last HDR_WORDS of a call's first object (gemv.cc reads the same indices)
H_G, H_Q, H_K, H_N, H_FLAGS, H_INIT_OFF, H_COPY = range(7)
F_INIT = 1        # the residual: accumulators start at h, which follows x in the object
F_NORM = 2        # RMSNorm folded in: gamma is in W; the epilogue scales by RSQ[bf16(sum x*x)]


def lead_elems(n: int, k: int, flags: int = 0) -> int:
    """bf16 elements at the front of a call's first object: x, then h when the call adds a residual."""
    return k + (n if flags & F_INIT else 0)


def check_call(n: int, k: int, flags: int = 0) -> None:
    check_shape(n, k)
    if flags & ~(F_INIT | F_NORM):
        raise ValueError(f"unknown flags {flags:#x}")
    lead = lead_elems(n, k, flags)
    if lead > MAX_K or lead % 32:
        raise ValueError(f"x and h are {lead} elements: more than the core's {MAX_K}-element buffer, or not a "
                         "multiple of 32 (the core copies 16 words at a time)")


def tail(n: int, k: int, flags: int = 0) -> np.ndarray:
    """uint32 [OBJ_WORDS - lead/2]: what follows x (and h) in a call's first object, zero pad then the header. The
    only source of headers: G and Q come from the same (n, k) the design derives its object counts from, so the
    core consumes exactly what the runtime sequence sends. Constant per (shape, flags), so a chained call reads it
    from a fixed region."""
    check_call(n, k, flags)
    lead = lead_elems(n, k, flags)
    t = np.zeros(OBJ_WORDS - lead // 2, np.uint32)
    hdr = t[-HDR_WORDS:]
    hdr[H_G], hdr[H_Q], hdr[H_K], hdr[H_N], hdr[H_FLAGS] = groups(n), chunks(k), k, n, flags
    hdr[H_INIT_OFF], hdr[H_COPY] = k, lead
    return t


def x_slot(x: np.ndarray, n: int, h: np.ndarray | None = None, flags: int = 0) -> np.ndarray:
    """uint16 [K] bf16 bits (and h [N] with F_INIT) -> uint32 [X_SLOT_WORDS]: the [x][h][pad][header] object, twice
    (one per core of a pair)."""
    x = np.ascontiguousarray(x)
    if x.dtype != np.uint16 or x.ndim != 1:
        raise TypeError("x_slot takes bf16 bits as a 1-D uint16 array")
    if (h is not None) != bool(flags & F_INIT):
        raise ValueError("h is given exactly when F_INIT is set")
    k = x.shape[0]
    lead = [x] if h is None else [x, np.ascontiguousarray(h, dtype=np.uint16)]
    obj = np.concatenate([np.concatenate(lead).view(np.uint32), tail(n, k, flags)])
    assert obj.size == OBJ_WORDS
    return np.concatenate([obj, obj])


def unpack_y(words: np.ndarray, n: int) -> np.ndarray:
    """The output region's first N/2 words -> uint16 [N] bf16 bits, already in natural order."""
    return np.ascontiguousarray(words[:n // 2], dtype=np.uint32).view(np.uint16).copy()
