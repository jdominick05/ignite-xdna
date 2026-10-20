"""SmolLM2-135M decode with the rounding points the NPU decoder will have: the reference and, later, the oracle.

Every value that crosses from one op to the next is bf16, and each op is built only from primitives whose bits are
known on this core: the mmul MAC (``engine_bf16_emulator.mac``, exp_sum), bf16 x bf16 products (exact in fp32),
round-to-nearest-even to bf16, and tables indexed by bf16 bits, computed on the host with correct rounding. So the
semantics below are what each kernel must reproduce bit for bit.

Rounding points, per layer at position p (``fused=True``, the default and the device's plan; ``fused=False`` is v0,
which ran the norms as their own op and added rounded projection outputs to the residual):
  norm+qkv  r = RSQ[bf16(sum(h*h))] (the sum through the MAC, clamped to [1, 2^32); RSQ a correctly rounded
            table of 1/sqrt(ss/576 + eps) by ss's bits); gamma folded into the weights at pack
            time, W_g = bf16(W * gamma); qkv = bf16(bf16(W_g h) * r)
  (v0)      RMSNorm y = bf16(gamma * bf16(h * r)) as its own op, then qkv = projection of y
  RoPE      bf16(q*cos + rotate_half(q)*sin), the two products exact and summed once; cos/sin are HF's bf16 tables
  KV        k (after RoPE) and v appended at row p
  scores    s_j = bf16(MAC(q, k_j)) / 8 (a power of two, exact); d_j = bf16(MAC(s_j - max s)); e_j = exp_bits(d_j);
            z = bf16(sum e_j); out = bf16(bf16(sum e_j v_j) * recip_bits(z))
  o+res     h = bf16(h + W_o attn): h is the accumulator's starting value, one rounding (v0: bf16(h + bf16(W_o attn)))
  MLP       gate+up as norm+qkv with the second norm's gamma; m = bf16(silu_bits(gate) * up); down+res as o+res
  Tables    exp_bits, silu_bits, recip_bits and RSQ are range-reduced device tables (see silu_bits and below):
            correctly rounded inside their ranges, fixed rules outside, never a subnormal
Then the final norm folded into the tied head the same way (its own bf16(embed * gamma) copy; the embedding lookup
keeps the plain table), and argmax (lowest index on a tie).

``exact=True`` runs every MAC through the exp_sum emulator. The default computes each MAC-accumulated sum in float64
and rounds once, which differs from exp_sum only in the last bit of rare sums: right for judging quality, wrong as a
device oracle.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from ignite_xdna.compiler import engine_bf16_emulator as em
from kernels.llm_gemv.pack import attn_out_perm

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
LAYERS, HIDDEN, HEADS, KV_HEADS, HEAD_DIM, MLP, VOCAB = 30, 576, 9, 3, 64, 1536, 49152
EPS, THETA = 1e-5, 100000.0


def model_file() -> Path:
    """The cached model.safetensors (fetch it once with ``hf download HuggingFaceTB/SmolLM2-135M-Instruct``)."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    found = sorted((home / "hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots").glob("*/model.safetensors"))
    if len(found) != 1:
        raise SystemExit(f"expected one cached {MODEL} snapshot under {home}, found {len(found)}")
    return found[0]


def safetensors(path) -> Dict[str, np.ndarray]:
    """name -> uint16 array of bf16 bits, read with numpy alone (the iron env has no safetensors package)."""
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    head = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, meta in head.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] != "BF16":
            raise ValueError(f"{name}: {meta['dtype']}, expected BF16")
        a, b = meta["data_offsets"]
        out[name] = np.frombuffer(raw, np.uint16, (b - a) // 2, base + a).reshape(meta["shape"])
    return out


def bf16(a) -> np.ndarray:
    """Round to bf16 (RNE), kept in float32."""
    return em.to_bf16(np.asarray(a, dtype=np.float32))


def bf16_from64(a) -> np.ndarray:
    """Round a float64 value to bf16 once (no float32 step in between): correct rounding of an exact sum."""
    a = np.asarray(a, dtype=np.float64)
    f = a.astype(np.float32)
    # float64 -> float32 rounds; if that rounding lands on a bf16 tie it could double-round. Resolve ties by the
    # residual's sign: nudge the float32 one ulp toward the true value before the bf16 rounding.
    resid = a - f.astype(np.float64)
    u = f.view(np.uint32)
    tie = (u & 0xFFFF) == 0x8000
    nudge = np.where(tie & (resid > 0), u + np.where(f >= 0, 1, -1), np.where(tie & (resid < 0), u - np.where(f >= 0, 1, -1), u))
    return bf16(nudge.astype(np.uint32).view(np.float32))


def _table(fn) -> np.ndarray:
    """fn over every bf16 bit pattern (float64), correctly rounded back to bf16: a 65536-entry table by bits.
    NaN and infinite inputs map to themselves; overflow to infinity and underflow are kept as they round."""
    with np.errstate(all="ignore"):
        vals = em.from_bf16_bits(np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)).astype(np.float64)
        out = np.where(np.isfinite(vals), fn(vals), vals)
        return bf16_from64(out)


EXP = _table(np.exp)
SILU = _table(lambda x: x / (1.0 + np.exp(-x)))
RECIP = _table(lambda x: 1.0 / x)


# The device's tables are range-reduced: a 65536-entry table is 128 KB, twice a core's data memory. Each function
# below is the device semantics over bf16 bits, pinned here; tables_header() emits the same arrays for the core.
# Inside a table's range the value is the correctly rounded one (subnormal results flushed to +0); outside it a
# rule decides, chosen so no subnormal ever reaches a MAC. Measured over every finite bf16 against the full tables.
SILU_POS = (0x3B80, 0x4100)      # x in [2^-8, 8)
SILU_NEG = (0xBB80, 0xC200)      # x in (-32, -2^-8]
EXP_RANGE = (0xBB00, 0xC300)     # d in (-128, -2^-9]


def _flush(bits: np.ndarray) -> np.ndarray:
    """Subnormal bf16 bits -> +0."""
    return np.where((bits & 0x7F80) == 0, 0, bits).astype(np.uint16)


def silu_table() -> np.ndarray:
    """uint16 [3072]: SILU at bits SILU_POS then SILU_NEG, correctly rounded."""
    full = em.bf16_bits(SILU)
    return _flush(np.concatenate([full[SILU_POS[0]:SILU_POS[1]], full[SILU_NEG[0]:SILU_NEG[1]]]))


def exp_table() -> np.ndarray:
    """uint16 [2048]: exp at bits EXP_RANGE, correctly rounded, underflow to +0."""
    return _flush(em.bf16_bits(EXP)[EXP_RANGE[0]:EXP_RANGE[1]])


def recip_table() -> np.ndarray:
    """uint16 [128]: bf16(1 / 1.m) for each mantissa m; 1/z for z = 2^e * 1.m is this shifted down by e, exact."""
    return em.bf16_bits(RECIP)[0x3F80:0x4000].copy()


SILU_T, EXP_T, RECIP_T = silu_table(), exp_table(), recip_table()


def silu_bits(u: np.ndarray) -> np.ndarray:
    """The device's SiLU on bf16 bits: x for x >= 8, +0 for x <= -32, x/2 below 2^-8 (signed 0 if subnormal), else
    the table."""
    u = np.asarray(u, np.uint16).astype(np.int64)
    mag = u & 0x7FFF
    neg = (u & 0x8000) != 0
    half = np.where((u & 0x7F80) > 0x80, u - 0x80, u & 0x8000)
    pos_i = np.clip(u - SILU_POS[0], 0, SILU_POS[1] - SILU_POS[0] - 1)
    neg_i = (SILU_POS[1] - SILU_POS[0]) + np.clip(u - SILU_NEG[0], 0, SILU_NEG[1] - SILU_NEG[0] - 1)
    out = np.where(neg, SILU_T[neg_i], SILU_T[pos_i])
    out = np.where(mag < 0x3B80, half, out)
    out = np.where(~neg & (mag >= 0x4100), u, out)
    out = np.where(neg & (mag >= 0x4200), 0, out)
    return out.astype(np.uint16)


def exp_bits(u: np.ndarray) -> np.ndarray:
    """The device's exp on bf16 bits of d <= 0: 1 for |d| < 2^-9, +0 for d <= -128, else the table."""
    u = np.asarray(u, np.uint16).astype(np.int64)
    if np.any(((u & 0x8000) == 0) & ((u & 0x7FFF) != 0)):
        raise ValueError("exp is defined here for d <= 0 only")
    mag = u & 0x7FFF
    i = np.clip(u - EXP_RANGE[0], 0, EXP_RANGE[1] - EXP_RANGE[0] - 1)
    out = np.where(mag < 0x3B00, 0x3F80, np.where(mag >= 0x4300, 0, EXP_T[i]))
    return out.astype(np.uint16)


def recip_bits(u: np.ndarray) -> np.ndarray:
    """The device's 1/z on bf16 bits of z >= 1 (a softmax denominator): RECIP_T[mantissa] shifted down by e."""
    u = np.asarray(u, np.uint16).astype(np.int64)
    if np.any(u < 0x3F80) or np.any(u >= 0x5F80):
        raise ValueError("recip is defined here for 1 <= z < 2^64 only (1/z stays normal)")
    e = (u >> 7) - 127
    return (RECIP_T[u & 0x7F].astype(np.int64) - (e << 7)).astype(np.uint16)


def mac_terms(pairs) -> np.ndarray:
    """sum_t a_t * w_t elementwise through one MAC from zero, fp32: a_t arrays [n], w_t scalars shared by every
    element (lane t of the core's mmul, B's column 0 holding the w_t)."""
    if not 0 < len(pairs) <= 8:
        raise ValueError("one MAC takes 1 to 8 terms")
    n = np.asarray(pairs[0][0]).size
    a = np.zeros((1, n, 8), np.float32)
    w = np.zeros((1, 8, 4), np.float32)
    for t, (x, y) in enumerate(pairs):
        a[0, :, t] = np.asarray(x, np.float32).reshape(-1)
        w[0, t, 0] = np.float32(y)
    return em.mac(np.zeros((1, 1, n, 4), np.float32), a, w, em.MAC_MODEL)[0, 0, :, 0]


def dev(fn, x: np.ndarray) -> np.ndarray:
    """Apply a bits function to bf16 values, returning float32."""
    return em.from_bf16_bits(fn(em.bf16_bits(bf16(x))))


def lut(table: np.ndarray, x: np.ndarray) -> np.ndarray:
    return table[em.bf16_bits(bf16(x)).astype(np.int64)]


def dot(a: np.ndarray, b: np.ndarray, exact: bool) -> np.ndarray:
    """Sums of bf16 products over the last axis: [..., K] x [K] or [..., K] x [..., K] -> fp32."""
    if not exact:
        return (a.astype(np.float64) * b.astype(np.float64)).sum(-1)
    a, b = np.broadcast_arrays(np.asarray(a, np.float32), np.asarray(b, np.float32))
    lead, k = a.shape[:-1], a.shape[-1]
    pad = (-k) % 8
    a = np.pad(a.reshape(-1, k), ((0, 0), (0, pad)))
    b = np.pad(b.reshape(-1, k), ((0, 0), (0, pad)))
    acc = np.zeros((a.shape[0], 1, 1, 1), np.float32)
    for s in range(0, a.shape[1], 8):
        # one lane per row: a row's 8 inputs times its own 8 weights, as an mmul with a single output column
        acc = np.stack([em.mac(acc[i:i + 1], a[i, s:s + 8].reshape(1, 1, 8), b[i, s:s + 8].reshape(1, 8, 1),
                               em.MAC_MODEL)[0] for i in range(a.shape[0])])
    return acc.reshape(lead).astype(np.float64)


def matvec(w: np.ndarray, x: np.ndarray, exact: bool, init: np.ndarray | None = None) -> np.ndarray:
    """bf16 W [N][K] times bf16 x -> bf16 y, one rounding: the GEMV. ``init`` (bf16 [N]) is the accumulator's
    starting value, which is how a residual add rides in the projection: y = bf16(init + W x), rounded once."""
    if isinstance(w, Q8):
        return w8_matvec(w, x, exact, init)
    if exact:
        from kernels.llm_gemv import oracle
        start = None if init is None else em.bf16_bits(init)
        return em.from_bf16_bits(oracle.gemv(em.bf16_bits(w), em.bf16_bits(x), init=start))
    x = x.astype(np.float64)
    if ACT_BITS:            # fake per-token symmetric activation quantization (quantize_weights' W8A8 study)
        qmax = 2 ** (ACT_BITS - 1) - 1
        s = np.abs(x).max() / qmax
        if s > 0:
            x = np.clip(np.rint(x / s), -qmax, qmax) * s
    acc = w @ x
    return bf16_from64(acc if init is None else acc + init.astype(np.float64))


ACT_BITS = 0                # 0: activations stay bf16; 8: W8A8's per-token int8 (fast mode only)


@dataclass
class Q8:
    """A W8A16 projection as the device holds it: int8 codes [N][K] (as float64) and one bf16 scale per row [N]."""
    codes: np.ndarray
    s: np.ndarray

    @classmethod
    def of(cls, w: np.ndarray) -> "Q8":
        """Symmetric round-to-nearest against the STORED scale: s = bf16(max|row| / 127), codes = rint(w / s) in
        [-127, 127] (bf16 rounding can leave s just under max/127; the clip takes the one code it would exceed)."""
        s = bf16_from64(np.abs(w).max(axis=1) / 127.0).astype(np.float64)
        safe = np.where(s == 0, 1.0, s)
        return cls(np.clip(np.rint(w / safe[:, None]), -127, 127), s)

    def __getitem__(self, idx):         # column selection only (the exact o-projection's input order)
        rows, cols = idx
        if rows != slice(None):
            raise IndexError("Q8 selects columns only")
        return Q8(self.codes[:, cols], self.s)

    @property
    def shape(self):
        return self.codes.shape


def w8_matvec(w: Q8, x: np.ndarray, exact: bool, init: np.ndarray | None = None) -> np.ndarray:
    """The W8A16 GEMV: acc = codes x in fp32 (the bf16 GEMV's MAC walk: int8 codes are exact in bf16), then one MAC
    y = bf16(hi*s + lo*s + init) with hi = bf16(acc), lo = bf16(acc - hi): the scale applied at fp32-grade with one
    rounding, and a residual riding in the same MAC."""
    if not exact:
        acc = w.codes @ x.astype(np.float64)
        return bf16_from64(w.s * acc + (0.0 if init is None else init.astype(np.float64)))
    from kernels.llm_gemv import oracle
    acc = oracle.gemv_acc(em.bf16_bits(w.codes.astype(np.float32)), em.bf16_bits(x)).astype(np.float32)
    hi = em.to_bf16(acc).astype(np.float32)
    lo = em.to_bf16((acc - hi).astype(np.float32)).astype(np.float32)
    a = np.zeros((acc.size, 8), np.float32)
    b = np.zeros((acc.size, 8), np.float32)
    a[:, 0], a[:, 1] = hi, lo
    b[:, 0] = b[:, 1] = w.s
    if init is not None:
        a[:, 2], b[:, 2] = np.asarray(init, np.float32), 1.0
    return bf16(mac_rows(a, b))


def mac_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One exp_sum MAC from zero per row, a[r] . b[r] (8 terms each): engine_bf16_emulator.mac's model with a
    different B per row, which its shared-operand form cannot express. float32 [rows]."""
    if em.MAC_MODEL != "exp_sum":
        raise ValueError(f"mac_rows implements exp_sum, not {em.MAC_MODEL}")
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    prod = a64 * b64
    place = (np.frexp(a64)[1] - 1) + (np.frexp(b64)[1] - 1)
    place = np.where(prod != 0, place, np.iinfo(np.int32).min)
    top = place.max(axis=1)
    top = np.where(top == np.iinfo(np.int32).min, 0, top)
    quantum = np.ldexp(1.0, top - 23)
    units = np.rint(prod / quantum[:, None]).sum(axis=1)
    return (units * quantum + 0.0).astype(np.float32)


def quantize(w: np.ndarray, bits: int, group: int = 0) -> np.ndarray:
    """Symmetric round-to-nearest fake quantization, returned dequantized (float64): one scale per output row, or
    per ``group`` inputs of a row; codes in [-(2^(bits-1)-1), 2^(bits-1)-1]."""
    qmax = 2 ** (bits - 1) - 1
    n, k = w.shape
    g = w.reshape(n, k // group, group) if group else w.reshape(n, 1, k)
    s = np.abs(g).max(axis=2, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    return (np.clip(np.rint(g / s), -qmax, qmax) * s).reshape(n, k)


SCHEMES = {"w8a16": (8, 0, 0), "w8a8": (8, 0, 8), "w4a16": (4, 32, 0), "w8a16-dev": (8, 0, 0)}
# w8a16-dev: the device's W8A16 (Q8 and w8_matvec), exact mode included; the other three are fast-mode studies


def quantize_weights(w: "Weights", scheme: str) -> "Weights":
    """The fused decoder's weights under ``scheme`` (SCHEMES: weight bits, group, activation bits), fast mode: every
    projection with its gamma folded in, and the head; the embedding lookup stays bf16. Sets ACT_BITS."""
    global ACT_BITS
    bits, group, act = SCHEMES[scheme]
    ACT_BITS = act
    q = Q8.of if scheme == "w8a16-dev" else (lambda m: quantize(m, bits, group))  # noqa: E731
    layers = []
    for L in w.layers:
        L = dict(L)
        for name in ("qkv_g", "o", "gu_g", "down"):
            L[name] = q(L[name])
        layers.append(L)
    out = Weights(w.embed, layers, w.norm)
    out.head_g = q(w.head_g)
    return out


RSQ_BASE, RSQ_SIZE = 0x3F80, 0x1000     # bf16 bits of 1.0 up to the largest bf16 below 2^32


def rsq_table() -> np.ndarray:
    """uint16 [RSQ_SIZE]: r = bf16(1/sqrt(ss/576 + eps)) for every bf16 ss in [1, 2^32), correctly rounded. The core
    looks r up by ss's bits, clamped to the table (on 13,481 real norm evaluations ss ran from 4.0 to 6.8e8)."""
    ss = em.from_bf16_bits((RSQ_BASE + np.arange(RSQ_SIZE)).astype(np.uint16)).astype(np.float64)
    return em.bf16_bits(bf16_from64(1.0 / np.sqrt(ss / HIDDEN + EPS)))


RSQ = rsq_table()
# The core stores only ss's first RSQ_STORED octaves: from there on eps no longer shows in the rounding, and octave e
# is octave e-2 halved, exactly (tests check every entry). rsq_stored_bits is the core's lookup.
RSQ_STORED = 8


def rsq_stored_bits(u: int) -> int:
    e, m = (u >> 7) - 127, u & 0x7F
    if e < RSQ_STORED:
        return int(RSQ[e * 128 + m])
    base = RSQ_STORED - 2 + ((e - RSQ_STORED) & 1)
    return int(RSQ[base * 128 + m]) - (((e - base) >> 1) << 7)


def rscale(h, exact):
    """RMSNorm's scale: ss = bf16(sum(h*h)), the sum walked by the core's MAC from zero; r = RSQ[ss's bits], clamped
    to the table's range."""
    if exact:
        from kernels.llm_gemv import oracle
        hb = em.bf16_bits(h)
        u = int(oracle.gemv(np.tile(hb, (4, 1)), hb)[0])
    else:
        u = int(em.bf16_bits(bf16_from64(np.dot(h.astype(np.float64), h.astype(np.float64)))).reshape(-1)[0])
    u = min(max(u, RSQ_BASE), RSQ_BASE + RSQ_SIZE - 1)
    return em.from_bf16_bits(np.uint16(RSQ[u - RSQ_BASE])).astype(np.float32)


def rmsnorm(h, gamma, exact):
    """v0, unfused: y = bf16(gamma * bf16(h * r)), its own op before the projection."""
    return bf16(gamma * bf16(h * rscale(h, exact)))


def normed_matvec(w_gamma, h, exact):
    """Fused: gamma folded into the weights at pack time (w_gamma = bf16(W * gamma)), r applied in the epilogue:
    y = bf16(bf16(w_gamma h) * r). Every core recomputes r from the h its header carries."""
    return bf16(matvec(w_gamma, h, exact) * rscale(h, exact))


def rope_tables(positions: int):
    """HF's cos/sin for the non-interleaved rotation, computed in float32 and cast to bf16, [positions][64]."""
    inv = (1.0 / (THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.int64).astype(np.float32) / HEAD_DIM))).astype(np.float32)
    f = np.arange(positions, dtype=np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([f, f], axis=1).astype(np.float32)
    return bf16(np.cos(emb)), bf16(np.sin(emb))


def rope(x, cos, sin, exact=False):
    """x [heads][64] bf16 -> bf16(x*cos + rotate_half(x)*sin): for each pair (a, b) = (x[i], x[i+32]),
    a' = a*cos_i - b*sin_i and b' = a*sin_i + b*cos_i, each the two products through one MAC from zero (the core's
    mmul, A = [a, b, ...], B's columns [cos, -sin], [sin, cos]), rounded once."""
    if not exact:
        rot = np.concatenate([-x[:, HEAD_DIM // 2:], x[:, :HEAD_DIM // 2]], axis=1)
        return bf16_from64(x.astype(np.float64) * cos + rot.astype(np.float64) * sin)
    half = HEAD_DIM // 2
    out = np.empty_like(x, dtype=np.float32)
    for i in range(half):
        a = np.zeros((1, x.shape[0], 8), np.float32)
        a[0, :, 0], a[0, :, 1] = x[:, i], x[:, i + half]
        w = np.zeros((1, 8, 4), np.float32)
        w[0, 0, 0], w[0, 1, 0], w[0, 0, 1], w[0, 1, 1] = cos[i], -sin[i], sin[i], cos[i]
        acc = em.mac(np.zeros((1, 1, x.shape[0], 4), np.float32), a, w, em.MAC_MODEL)[0, 0]
        out[:, i], out[:, i + half] = bf16(acc[:, 0]), bf16(acc[:, 1])
    return out


# The device's dim order within a head: q and K leave the RoPE projection with each (i, i+32) pair adjacent.
DEV_DIMS = np.column_stack([np.arange(HEAD_DIM // 2), np.arange(HEAD_DIM // 2) + HEAD_DIM // 2]).reshape(-1)


def div8_bits(u: np.ndarray) -> np.ndarray:
    """x/8 on bf16 bits: the exponent less 3, and a result below the normal range flushed to a signed zero."""
    u = np.asarray(u, np.uint16)
    return np.where((u >> 7) & 0xFF > 3, u - np.uint16(3 << 7), u & np.uint16(0x8000)).astype(np.uint16)


def _mac(acc, a, w):
    return em.mac(acc, np.asarray(a, np.float32), np.asarray(w, np.float32), em.MAC_MODEL)


def attention_scores(q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """bf16 [positions][g]: the F_SCORE output before div8, K rows by q heads over DEV_DIMS, 8 steps from zero."""
    kd, qd = np.asarray(K, np.float32)[:, DEV_DIMS], np.asarray(q, np.float32)[:, DEV_DIMS]
    acc = np.zeros((1, 1, kd.shape[0], qd.shape[0]), np.float32)
    for s in range(0, HEAD_DIM, 8):
        acc = _mac(acc, kd[None, :, s:s + 8], qd[:, s:s + 8].T[None])
    return bf16(acc[0, 0])


def attention_group(q: np.ndarray, K: np.ndarray, V: np.ndarray, p: int) -> np.ndarray:
    """The device's attention for one KV head and the q heads that share it, exact: q [g][64], K and V [>p][64],
    all bf16 values in natural dim order; returns bf16 [g][64]. One core's mmul walk, lane for lane:
    - scores: A = cache rows x 8 dims, B = 8 dims x the q heads, dims in DEV_DIMS order, 8 steps from zero, then
      bf16 and div8;
    - d = bf16(s - max) as one MAC of [s, max] x [1, -1]; e = EXP table at d;
    - per block of 8 positions from 0 (positions past p zero in both operands): A = e [heads][8 positions],
      B = V [8 positions][dims] into acc, and B = ones into z; out = bf16(bf16(acc) * RECIP[bf16(z)])."""
    g, n = q.shape[0], p + 1
    s = dev(div8_bits, attention_scores(q, np.asarray(K)[:n]))             # [n][g]
    m = s.max(axis=0)
    d = np.stack([bf16(mac_terms([(s[:, h], 1.0), (np.full(n, m[h], np.float32), -1.0)])) for h in range(g)], 1)
    e = dev(exp_bits, d)
    blocks = -(-n // 8)
    ez = np.zeros((blocks * 8, g), np.float32)
    ez[:n] = e
    vz = np.zeros((blocks * 8, HEAD_DIM), np.float32)
    vz[:n] = np.asarray(V, np.float32)[:n]
    out = np.zeros((1, 1, g, HEAD_DIM), np.float32)
    z = np.zeros((1, 1, g, 1), np.float32)
    for b in range(blocks):
        a = ez[b * 8:(b + 1) * 8].T[None]                                  # [1][g][8]
        out = _mac(out, a, vz[b * 8:(b + 1) * 8][None])
        z = _mac(z, a, np.ones((1, 8, 1), np.float32))
    r = dev(recip_bits, z[0, 0, :, 0])
    return bf16(bf16(out[0, 0]) * r[:, None])


@dataclass
class Weights:
    embed: np.ndarray                       # [V][576] bf16 as float32; also the head
    layers: List[Dict[str, np.ndarray]]
    norm: np.ndarray

    @classmethod
    def load(cls, tensors: Dict[str, np.ndarray]):
        # float64 holds bf16 exactly, and keeps every fast-mode matvec from re-widening 269 MB of weights per token
        f = lambda n: em.from_bf16_bits(tensors[n]).astype(np.float64)  # noqa: E731
        layers = []
        for i in range(LAYERS):
            p = f"model.layers.{i}."
            layers.append(dict(
                n1=f(p + "input_layernorm.weight"), n2=f(p + "post_attention_layernorm.weight"),
                qkv=np.concatenate([f(p + "self_attn.q_proj.weight"), f(p + "self_attn.k_proj.weight"),
                                    f(p + "self_attn.v_proj.weight")]),
                o=f(p + "self_attn.o_proj.weight"),
                gu=np.concatenate([f(p + "mlp.gate_proj.weight"), f(p + "mlp.up_proj.weight")]),
                down=f(p + "mlp.down_proj.weight")))
            L = layers[-1]
            # gamma folded into the following projection's input columns, rounded to bf16 once at pack time
            L["qkv_g"] = bf16(L["qkv"] * L["n1"][None, :]).astype(np.float64)
            L["gu_g"] = bf16(L["gu"] * L["n2"][None, :]).astype(np.float64)
        w = cls(f("model.embed_tokens.weight"), layers, f("model.norm.weight"))
        w.head_g = bf16(w.embed * w.norm[None, :]).astype(np.float64)   # the tied head with the final norm's gamma
        return w


@dataclass
class Decoder:
    w: Weights
    max_positions: int = 2048
    exact: bool = False
    fused: bool = True        # False: v0 (norms as their own op, residual adds of rounded outputs)
    k: List[list] = field(default_factory=lambda: [[] for _ in range(LAYERS)])
    v: List[list] = field(default_factory=lambda: [[] for _ in range(LAYERS)])

    def __post_init__(self):
        self.cos, self.sin = rope_tables(self.max_positions)

    def attention(self, i, q, p):
        K = np.stack(self.k[i])                         # [p+1][3][64]
        V = np.stack(self.v[i])
        out = np.zeros((HEADS, HEAD_DIM), np.float32)
        if self.exact:
            g = HEADS // KV_HEADS
            for kv in range(KV_HEADS):
                out[kv * g:(kv + 1) * g] = attention_group(q.reshape(HEADS, HEAD_DIM)[kv * g:(kv + 1) * g],
                                                           K[:, kv], V[:, kv], p)
            return out.reshape(-1)
        for hd in range(HEADS):
            kv = hd // (HEADS // KV_HEADS)
            s = bf16(dot(K[:, kv], q[hd], self.exact)) / 8.0      # exact: /8 is a power of two
            d = (bf16(mac_terms([(s, 1.0), (np.full_like(s, s.max()), -1.0)]))
                 if self.exact else bf16(s - s.max()))
            e = dev(exp_bits, d)
            z = bf16_from64(dot(e, np.ones_like(e), self.exact) if self.exact else e.astype(np.float64).sum())
            acc = bf16_from64(dot(V[:, kv].T, e, self.exact))
            out[hd] = bf16(acc * dev(recip_bits, z))
        return out.reshape(-1)

    def step(self, token: int, p: int) -> np.ndarray:
        """One decode step at position p: returns the bf16 logits [V]."""
        h = self.w.embed[token].copy()
        for i in range(len(self.w.layers)):
            h = self.layer(i, h, p)
        if self.fused:
            return normed_matvec(self.w.head_g, h, self.exact)
        return matvec(self.w.embed, rmsnorm(h, self.w.norm, self.exact), self.exact)

    def layer(self, i: int, h: np.ndarray, p: int, trace: dict | None = None) -> np.ndarray:
        """Decoder layer i at position p: appends its k and v, returns the new h. ``trace`` collects the
        intermediates (qkv, attn, h1, gu, m) when given."""
        L, x = self.w.layers[i], self.exact
        cos, sin = self.cos[p], self.sin[p]
        if self.fused:
            qkv = normed_matvec(L["qkv_g"], h, x)
        else:
            qkv = matvec(L["qkv"], rmsnorm(h, L["n1"], x), x)
        q = rope(qkv[:HIDDEN].reshape(HEADS, HEAD_DIM), cos, sin, x)
        k = rope(qkv[HIDDEN:HIDDEN + KV_HEADS * HEAD_DIM].reshape(KV_HEADS, HEAD_DIM), cos, sin, x)
        v = qkv[HIDDEN + KV_HEADS * HEAD_DIM:].reshape(KV_HEADS, HEAD_DIM)
        self.k[i].append(k)
        self.v[i].append(v)
        attn = self.attention(i, q, p)
        if self.fused:      # the residual is the accumulator's start: one rounding of h + W x
            if x:           # the device's o-projection takes attention's output in its own order
                perm = attn_out_perm()
                h1 = matvec(L["o"][:, perm], attn[perm], x, init=h)
            else:
                h1 = matvec(L["o"], attn, x, init=h)
            gu = normed_matvec(L["gu_g"], h1, x)
        else:
            h1 = bf16_from64(h.astype(np.float64) + matvec(L["o"], attn, x))
            gu = matvec(L["gu"], rmsnorm(h1, L["n2"], x), x)
        m = bf16(dev(silu_bits, gu[:MLP]) * gu[MLP:])
        if self.fused:
            out = matvec(L["down"], m, x, init=h1)
        else:
            out = bf16_from64(h1.astype(np.float64) + matvec(L["down"], m, x))
        if trace is not None:
            trace.update(q=q, k=k, v=v, attn=attn, h1=h1, m=m, h2=out)
        return out

    def generate(self, prompt: List[int], new_tokens: int, eos: int | None = None) -> List[int]:
        out = []
        logits = None
        for p, t in enumerate(prompt):
            logits = self.step(t, p)
        for _ in range(new_tokens):
            t = int(np.argmax(logits))
            out.append(t)
            if eos is not None and t == eos:
                break
            logits = self.step(t, len(prompt) + len(out) - 1)
        return out
