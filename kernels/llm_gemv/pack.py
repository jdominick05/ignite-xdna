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
# A chained call's lead (x, h, tail: pieces of the out and x arguments) goes to both cores of a channel's pair. With
# LEAD_COPIES = 1 the pieces are one chained BD task pushed with repeat 1 (design.lead_task), so the shim sends the
# lead twice from one set of BDs; with 2, each core's copy has its own BDs, as the lowering of dma_memcpy_nd emits.
LEAD_COPIES = 1


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


def stream_elems(n: int, k: int, flags: int = 0) -> int:
    """bf16 elements one channel reads for the weights of one call (a SiLU call reads gate and up: twice)."""
    return groups(n) * objects_per_group(k, flags) * 2 * OBJ_ELEMS


def objects_per_group(k: int, flags: int = 0) -> int:
    return chunks(k) * (2 if flags & F_SILU else 1)


def weight_words(n: int, k: int, flags: int = 0) -> int:
    """Words of a call's weight block: N*K/2, or twice that for a SiLU call (gate and up rows)."""
    return block_words(n, k) * (2 if flags & F_SILU else 1)


def core_rows(n: int, c: int, j: int, g: int) -> slice:
    base = c * (n // COLS) + g * (ROWS * OUT_GROUP) + j * OUT_GROUP
    return slice(base, base + OUT_GROUP)


def chunk(w: np.ndarray, rows: slice, q: int) -> np.ndarray:
    """One weight object: W[rows, q*192:(q+1)*192] as [kstep][acc][kk][n], flattened."""
    t = w[rows, q * KCHUNK:(q + 1) * KCHUNK]                      # [12][192] = [acc 3][n 4][kstep 24][kk 8]
    return t.reshape(3, 4, KCHUNK // 8, 8).transpose(2, 0, 3, 1).reshape(-1)


def rope_perm(n: int, rotated: int, head_dim: int = 64) -> np.ndarray:
    """Row order of a projection whose first ``rotated`` rows are RoPE heads: within each head, output position
    2i holds dim i and 2i+1 holds dim i + head_dim/2, so each rotated pair sits in one core's group. The rest keep
    their order. A dot product of two vectors in this order equals the dot in the natural one."""
    if rotated % head_dim or rotated > n:
        raise ValueError("rotated rows are whole heads")
    perm = np.arange(n)
    half = head_dim // 2
    for h in range(rotated // head_dim):
        base = h * head_dim
        perm[base:base + head_dim:2] = base + np.arange(half)
        perm[base + 1:base + head_dim:2] = base + half + np.arange(half)
    return perm


def pack_silu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """uint16 [N][K] gate and up bf16 bits -> the weight block of an F_SILU call: for every group and core, the
    group's gate chunks, then the same rows' up chunks."""
    if gate.shape != up.shape or gate.dtype != np.uint16 or up.dtype != np.uint16:
        raise TypeError("pack_silu takes two uint16 [N][K] arrays of one shape")
    n, k = gate.shape
    check_shape(n, k)
    out = np.empty(2 * n * k, np.uint16)
    pos = 0
    for c in range(COLS):
        for h in range(2):
            for g in range(groups(n)):
                for w in (gate, up):
                    for q in range(chunks(k)):
                        for e in range(2):
                            out[pos:pos + OBJ_ELEMS] = chunk(w, core_rows(n, c, 2 * h + e, g), q)
                            pos += OBJ_ELEMS
    assert pos == out.size
    return out.view(np.uint32)


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
H_G, H_Q, H_K, H_N, H_FLAGS, H_AUX_OFF, H_COPY, H_E, H_ROPE_END = range(9)
F_INIT = 1        # the residual: accumulators start at h, which follows x in the object
F_NORM = 2        # RMSNorm folded in: gamma is in W; the epilogue scales by RSQ[bf16(sum x*x)]
F_ROPE = 4        # RoPE in the epilogue on output positions below ROPE_END (rows in rope_perm order); the
                  # position's cos/sin row (32 + 32 bf16) opens the tail
F_SILU = 8        # each group reads gate chunks then up chunks into two accumulator sets; emits silu(gate) * up
FLAGS_ALL = F_INIT | F_NORM | F_ROPE | F_SILU
CS_ELEMS = 64


def lead_elems(n: int, k: int, flags: int = 0) -> int:
    """bf16 elements at the front of a call's first object that come from activations: x, then h with F_INIT."""
    return k + (n if flags & F_INIT else 0)


def copy_elems(n: int, k: int, flags: int = 0) -> int:
    """What the core copies into its x buffer: the lead, then cos/sin with F_ROPE."""
    return lead_elems(n, k, flags) + (CS_ELEMS if flags & F_ROPE else 0)


def check_call(n: int, k: int, flags: int = 0, rope_end: int = 0) -> None:
    check_shape(n, k)
    if flags & ~FLAGS_ALL:
        raise ValueError(f"unknown flags {flags:#x}")
    if flags & F_INIT and flags & (F_SILU | F_NORM | F_ROPE):
        raise ValueError("a residual call takes no other flag")
    if flags & F_ROPE and flags & F_SILU:
        raise ValueError("RoPE and SiLU do not combine")
    if bool(flags & F_ROPE) != (rope_end > 0) or rope_end % (OUT_GROUP * 2) or rope_end > n:
        raise ValueError("ROPE_END is a positive multiple of 24 up to N exactly when F_ROPE is set")
    copy = copy_elems(n, k, flags)
    if copy > MAX_K or copy % 32:
        raise ValueError(f"the core copies {copy} elements: more than its {MAX_K}-element buffer, or not a "
                         "multiple of 32 (it copies 16 words at a time)")


def tail(n: int, k: int, flags: int = 0, rope_end: int = 0, cs: np.ndarray | None = None) -> np.ndarray:
    """uint32 [OBJ_WORDS - lead/2]: what follows x (and h) in a call's first object: cos/sin with F_ROPE, zero pad,
    then the header. The only source of headers: G and Q come from the same (n, k, flags) the design derives its
    object counts from, so the core consumes exactly what the runtime sequence sends. Constant per call (per
    position with F_ROPE), so a chained call reads it from a fixed region."""
    check_call(n, k, flags, rope_end)
    if (cs is not None) != bool(flags & F_ROPE):
        raise ValueError("cos/sin are given exactly when F_ROPE is set")
    lead = lead_elems(n, k, flags)
    t = np.zeros(OBJ_WORDS - lead // 2, np.uint32)
    if cs is not None:
        cs = np.ascontiguousarray(cs, dtype=np.uint16)
        if cs.shape != (CS_ELEMS,):
            raise ValueError("cs is cos[0:32] then sin[0:32], bf16 bits")
        t[:CS_ELEMS // 2] = cs.view(np.uint32)
    hdr = t[-HDR_WORDS:]
    hdr[H_G], hdr[H_Q], hdr[H_K], hdr[H_N], hdr[H_FLAGS] = groups(n), objects_per_group(k, flags), k, n, flags
    hdr[H_AUX_OFF], hdr[H_COPY], hdr[H_E], hdr[H_ROPE_END] = k, copy_elems(n, k, flags), 1, rope_end
    return t


def x_slot(x: np.ndarray, n: int, h: np.ndarray | None = None, flags: int = 0, rope_end: int = 0,
           cs: np.ndarray | None = None) -> np.ndarray:
    """uint16 [K] bf16 bits (and h [N] with F_INIT) -> uint32 [X_SLOT_WORDS]: the [x][h][pad][header] object, twice
    (one per core of a pair)."""
    x = np.ascontiguousarray(x)
    if x.dtype != np.uint16 or x.ndim != 1:
        raise TypeError("x_slot takes bf16 bits as a 1-D uint16 array")
    if (h is not None) != bool(flags & F_INIT):
        raise ValueError("h is given exactly when F_INIT is set")
    k = x.shape[0]
    lead = [x] if h is None else [x, np.ascontiguousarray(h, dtype=np.uint16)]
    obj = np.concatenate([np.concatenate(lead).view(np.uint32), tail(n, k, flags, rope_end, cs)])
    assert obj.size == OBJ_WORDS
    return np.concatenate([obj, obj])


def unpack_y(words: np.ndarray, n: int) -> np.ndarray:
    """The output region's first N/2 words -> uint16 [N] bf16 bits, already in natural order."""
    return np.ascontiguousarray(words[:n // 2], dtype=np.uint32).view(np.uint16).copy()


# Attention: two more modes of the same core program. Columns 0..KV_COLS-1 each own one KV head and the GQA q heads
# that share it; the last column takes no part. The KV cache is [position][KV_COLS * HEAD_DIM] bf16 (the KV append's
# rows), K in rope_perm order within a head, V natural. CAP positions are streamed whatever p is, so every stream
# length is fixed and p rides in the header.
H_POS = 9
F_SCORE = 16      # K chunks of SCORE_ROWS positions in, bf16(K q) out per (4 positions x GQA heads) block
F_PV = 32         # the scores in, softmax over positions <= p, then e V over this core's 16 dims
KV_COLS, GQA, HEAD_DIM = 3, 3, 64
CACHE_WORDS = KV_COLS * HEAD_DIM // 2      # one cache row
SCORE_ROWS = OBJ_ELEMS // HEAD_DIM         # 36 positions of one head's K per object
PV_ROWS = OBJ_ELEMS // (HEAD_DIM // ROWS)  # 144 positions of a core's 16 V dims per object
SCORE_EMITS = SCORE_ROWS // 4              # 9 blocks of 4 positions per K object


def check_cap(cap: int) -> None:
    if cap <= 0 or cap % (2 * PV_ROWS) or GQA * cap > MAX_K:
        raise ValueError(f"CAP={cap} must be a positive multiple of {2 * PV_ROWS} with {GQA}*CAP <= {MAX_K}")


def attn_tail(flags: int, cap: int, p: int = 0) -> np.ndarray:
    """The tail of an attention call's first object (after the q heads or the scores): zero pad and the header."""
    check_cap(cap)
    if flags not in (F_SCORE, F_PV) or not 0 <= p < cap:
        raise ValueError("one of F_SCORE and F_PV, and 0 <= p < CAP")
    lead = attn_lead_words(flags, cap)
    t = np.zeros(OBJ_WORDS - lead, np.uint32)
    hdr = t[-HDR_WORDS:]
    if flags == F_SCORE:      # one group per K object a core takes, one chunk each, 9 emits
        hdr[H_G], hdr[H_Q], hdr[H_E] = cap // (SCORE_ROWS * ROWS), 1, SCORE_EMITS
    else:                     # one group of 48 outputs: CAP/144 V objects, 4 emits
        hdr[H_G], hdr[H_Q], hdr[H_E] = 1, cap // PV_ROWS, GQA * (HEAD_DIM // ROWS) // OUT_GROUP
    hdr[H_K], hdr[H_N], hdr[H_FLAGS], hdr[H_COPY], hdr[H_POS] = cap, HEAD_DIM, flags, 2 * lead, p
    return t


def attn_lead_words(flags: int, cap: int) -> int:
    """Words at the front of a column's first object: its GQA q heads, or its GQA x CAP scores."""
    return GQA * HEAD_DIM // 2 if flags == F_SCORE else GQA * cap // 2


def score_index(pos: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Where the score of (position, head) sits in its column's F_SCORE output: core j of the column takes K objects
    4g+j, and each round of the join is 12 values from each of the four cores, one (4 positions x 3 heads) block."""
    pos, head = np.asarray(pos), np.asarray(head)
    chunk, within = pos // SCORE_ROWS, pos % SCORE_ROWS
    g, core, e, pib = chunk // ROWS, chunk % ROWS, within // 4, within % 4
    return ((g * SCORE_EMITS + e) * ROWS + core) * OUT_GROUP + pib * GQA + head


def attn_out_perm() -> np.ndarray:
    """Natural index (head*64 + dim) of each element of the attention output as the device writes it: column c's
    96 words hold heads 3c..3c+2; core j makes dims 16j..16j+15 of each, value t = head*16 + d, 12 per emit."""
    out = np.empty(KV_COLS * GQA * HEAD_DIM, np.int64)
    for c in range(KV_COLS):
        for e in range(GQA * HEAD_DIM // ROWS // OUT_GROUP):
            for j in range(ROWS):
                for i in range(OUT_GROUP):
                    t = e * OUT_GROUP + i
                    out[c * GQA * HEAD_DIM + (e * ROWS + j) * OUT_GROUP + i] = \
                        (c * GQA + t // 16) * HEAD_DIM + 16 * j + t % 16
    return out


# W8A16: int8 weights with one bf16 scale per row (reference.Q8, w8_matvec). An object is still OBJ_WORDS: 24 outputs
# x 192 inputs of int8 in mmul<4,8,4> order [kstep 24][acc 6][kk 8][n 4], so a core's group is 24 outputs and it
# emits them as two runs of 12, rows c*N/4 + g*96 + e*48 + j*12 + [0, 12) for emit e: the column's join still writes
# natural order. N pads up to a multiple of W8_QUANTUM with zero rows (the caller routes them to a junk region).
# Each core's scales ride in its own tail, after x (and h, and cos/sin): the lead is per core, not per pair.
F_W8 = 64
W8_GROUP = 2 * OUT_GROUP                 # 24 outputs per core group
W8_QUANTUM = CORES * W8_GROUP            # 384
H_SC = 10                                # element offset of the core's scales in its x buffer
X_BUF = OBJ_WORDS - HDR_WORDS            # words of the core's x buffer under F_W8 (lead, cos/sin and scales)


def w8_pad(n: int) -> int:
    return -(-n // W8_QUANTUM) * W8_QUANTUM


def core_rows8(n: int, c: int, j: int, g: int) -> np.ndarray:
    """Rows of core j of column c in group g, in accumulator order: emit 0's 12, then emit 1's."""
    base = c * (n // COLS) + g * (ROWS * W8_GROUP) + j * OUT_GROUP
    return np.concatenate([base + e * ROWS * OUT_GROUP + np.arange(OUT_GROUP) for e in range(2)])


def chunk8(codes: np.ndarray, rows: np.ndarray, q: int) -> np.ndarray:
    """int8 [N][K] -> one W8 object, codes[rows, q*192:(q+1)*192] as [kstep][acc][kk][n], flattened (4,608 B)."""
    t = codes[rows, q * KCHUNK:(q + 1) * KCHUNK]                  # [24][192] = [acc 6][n 4][kstep 24][kk 8]
    return t.reshape(6, 4, KCHUNK // 8, 8).transpose(2, 0, 3, 1).reshape(-1)


def check_w8(n: int, k: int) -> None:
    if n <= 0 or n % W8_QUANTUM:
        raise ValueError(f"W8: N={n} is not a positive multiple of {W8_QUANTUM} (pad with w8_pad)")
    if k <= 0 or k % KCHUNK:
        raise ValueError(f"W8: K={k} is not a positive multiple of {KCHUNK}")


def pack8(*mats: np.ndarray) -> np.ndarray:
    """int8 [N][K] codes (two of one shape for F_SILU: gate, up) -> uint32 words of the W8 weight block: for each
    channel, group, (set,) chunk and core of the pair, one object."""
    mats = [np.ascontiguousarray(m) for m in mats]
    if not 1 <= len(mats) <= 2 or any(m.dtype != np.int8 or m.shape != mats[0].shape for m in mats):
        raise TypeError("pack8 takes one or two int8 [N][K] arrays of one shape")
    n, k = mats[0].shape
    check_w8(n, k)
    out = np.empty(len(mats) * n * k, np.int8)
    pos, size = 0, W8_GROUP * KCHUNK
    for c in range(COLS):
        for h in range(2):
            for g in range(n // W8_QUANTUM):
                for m in mats:
                    for q in range(chunks(k)):
                        for e in range(2):
                            out[pos:pos + size] = chunk8(m, core_rows8(n, c, 2 * h + e, g), q)
                            pos += size
    assert pos == out.size
    return out.view(np.uint32)


def scales8(n: int, *s: np.ndarray) -> np.ndarray:
    """bf16 scale bits [N] (two for F_SILU) -> uint16 [CORES][rows per core]: each core's scales in the order its
    accumulators make them (set, group, emit, row)."""
    out = []
    for c in range(COLS):
        for j in range(ROWS):
            out.append(np.concatenate([v[core_rows8(n, c, j, g)] for v in s for g in range(n // W8_QUANTUM)]))
    return np.array(out, np.uint16)


def lead_elems8(n: int, k: int, flags: int, h_rows: int | None = None) -> int:
    """A W8 call's lead: x, then with F_INIT only the h_rows real rows of h (the padded rows read past it and land
    in the junk region)."""
    return k + ((n if h_rows is None else h_rows) if flags & F_INIT else 0)


def copy_elems8(n: int, k: int, flags: int, h_rows: int | None = None) -> int:
    """What a W8 core copies into its x buffer: lead, cos/sin with F_ROPE, its scales; rounded up to 32."""
    rows = n // CORES * (2 if flags & F_SILU else 1)
    cs = CS_ELEMS if flags & F_ROPE else 0
    return -(-(lead_elems8(n, k, flags, h_rows) + cs + rows) // 32) * 32


def tail8(n: int, k: int, flags: int, scales: np.ndarray, rope_end: int = 0,
          cs: np.ndarray | None = None, h_rows: int | None = None) -> np.ndarray:
    """One core's tail under F_W8 (flags without F_W8): [cos/sin][its scales][pad][header], to follow its lead."""
    check_w8(n, k)
    if flags & ~FLAGS_ALL:
        raise ValueError(f"unknown flags {flags:#x}")
    if (cs is not None) != bool(flags & F_ROPE):
        raise ValueError("cos/sin are given exactly when F_ROPE is set")
    copy = copy_elems8(n, k, flags, h_rows)
    if copy > 2 * X_BUF:
        raise ValueError(f"W8: the core would copy {copy} elements, past its {2 * X_BUF}")
    lead = lead_elems8(n, k, flags, h_rows)
    if lead % 2:
        raise ValueError("the lead is whole words")
    t = np.zeros(OBJ_WORDS - lead // 2, np.uint16).repeat(2)
    at = 0
    if cs is not None:
        t[:CS_ELEMS] = np.ascontiguousarray(cs, dtype=np.uint16)
        at = CS_ELEMS
    scales = np.ascontiguousarray(scales, dtype=np.uint16)
    if scales.shape != (n // CORES * (2 if flags & F_SILU else 1),):
        raise ValueError("one core's scales: N/16 rows, twice with F_SILU")
    t[at:at + scales.size] = scales
    t = t.view(np.uint32)
    hdr = t[-HDR_WORDS:]
    hdr[H_G], hdr[H_Q], hdr[H_K], hdr[H_N] = n // W8_QUANTUM, objects_per_group(k, flags), k, n
    hdr[H_FLAGS], hdr[H_AUX_OFF], hdr[H_COPY], hdr[H_E] = flags | F_W8, k if flags & F_INIT else lead, copy, 2
    hdr[H_ROPE_END], hdr[H_SC] = rope_end, lead + at
    return t
