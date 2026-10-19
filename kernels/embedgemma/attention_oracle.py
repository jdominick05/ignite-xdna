# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Blockwise online-softmax attention for EmbeddingGemma 2, written the way a fused AIE kernel would compute it.

One document's queries are processed in blocks of ``q_block`` rows. Each block sweeps the key blocks of
``k_block`` keys it can see, carrying a running row maximum m, row sum l and an fp32 output accumulator O
(flash-attention order):
    S = Q_blk @ K_blk^T                     (scaling 1.0; fp32 accumulate of BF16 operands)
    m' = max(m, rowmax(S));  alpha = exp(m - m');  P = exp(S - m')
    l = alpha * l + rowsum(P);  O = alpha * O + P @ V_blk
    out = O / l
Local layers see only keys with |i - j| <= window; key blocks entirely outside the band are skipped, which is
what a fused kernel saves over full scores plus a host mask. ``p_bf16`` rounds P to BF16 before the value product
(the AIE MAC takes BF16 operands) while l still sums fp32 P; ``exp_bf16`` rounds each exp to BF16 as a
lookup-table exp returns it. Q, K and V are rounded to BF16 on entry when ``qkv_bf16``. ``attention`` returns
[L, H*hd] fp32 like ``Encoder._attention``.

``exp="lut"`` models the kernel's BF16 table exp exactly (bit-equal to the device on a 4096-point probe): x clamped to
[-80, tau], index = round-half-even(x * 256) as int16, split by a floor shift, result = bf16(e^(index >> 8)) * bf16(e^((index & 255) / 256)) in fp32 (the product of two BF16 values is
exact in fp32). ``exp="fixed"`` is the v5 kernel: scores and the running max in fixed point (S x 256 rounded to
int32), masked lanes exactly zero, the index split by an arithmetic shift, lazy rescale at 8 (see fixed_block).
``tau > 0`` is the kernel's lazy rescale: a row's reference max moves only when its block max exceeds
it by more than tau, so P can reach e^tau.
"""
import numpy as np

from model import bf16_round

_ILUT = bf16_round(np.exp(np.minimum(np.arange(256, dtype=np.int64).astype(np.int8), 88).astype(np.float64)).astype(np.float32))   # n > 88 never indexed (clamp at tau)
_FLUT = bf16_round(np.exp(np.arange(256, dtype=np.float64) / 256).astype(np.float32))


def lut_exp(x, tau=0.0):
    """The kernel's BF16 table exp: [-80, tau] clamp, int16 index at 8 fraction bits, two BF16 tables, fp32 product."""
    x = np.asarray(x, np.float32)
    tiny = x < -80
    xc = np.clip(x, -80, tau)
    idx = np.round(xc.astype(np.float64) * 256).astype(np.int64).astype(np.int16).view(np.uint16).astype(np.int64)
    out = (_ILUT[idx >> 8] * _FLUT[idx & 255]).astype(np.float32)
    return np.where(tiny, np.float32(0), out).astype(np.float32)


TAUF, LOF, MASKV, MINIT = 8 * 256, -80 * 256, -(1 << 29), -(1 << 28)


def to_index(S, mode="round"):
    """fp32 scores -> int32 fixed point at 8 fraction bits, as the kernel's to_fixed<int32>(S, 8)."""
    x = np.asarray(S, np.float32).astype(np.float64) * 256
    return (np.floor(x) if mode == "floor" else np.round(x)).astype(np.int64)


def expfix(d):
    """The v5 kernel's exp of a fixed-point difference: clamp to [-80, 8] x 256, e^(d >> 8) * e^((d & 255) / 256)
    from the two BF16 tables, product exact in fp32."""
    d = np.clip(np.asarray(d, np.int64), LOF, TAUF)
    return (_ILUT[(d >> 8) & 255] * _FLUT[d & 255]).astype(np.float32)


def fixed_block(S, vis, m, l, O, V, mode="round"):
    """One 32-key block of the v5 kernel's integer softmax. S fp32 [rows, keys]; m int64 [rows]; l fp32; O fp32
    [rows, hd]; V the block's values. Returns the new (m, l, O)."""
    idx = np.where(vis, to_index(S, mode), MASKV)
    rowmax = idx.max(axis=1)
    up = rowmax > m + TAUF
    mn = np.where(up, rowmax, m)
    alpha = np.where(up, expfix(m - mn), np.float32(1)).astype(np.float32)
    alpha = np.where(l == 0, np.float32(0), alpha).astype(np.float32)
    P = bf16_round(np.where(idx == MASKV, np.float32(0), expfix(idx - mn[:, None])))
    l = (alpha * l + P.sum(axis=1, dtype=np.float32)).astype(np.float32)
    O = (alpha[:, None] * O + (P.astype(np.float64) @ V.astype(np.float64))).astype(np.float32)
    return mn, l, O


def attention(q, k, v, kind, window, *, q_block=16, k_block=32, qkv_bf16=True, p_bf16=True, exp_bf16=True,
              exp="exact", tau=0.0):
    """q [L, H, hd], k/v [L, kv, hd] fp32 (normed, rotated) for ONE document."""
    L, H, hd = q.shape
    kv = k.shape[1]
    rep = H // kv
    if qkv_bf16:
        q, k, v = bf16_round(q), bf16_round(k), bf16_round(v)
    out = np.empty((L, H, hd), np.float32)
    if exp == "fixed":   # the v5 kernel's integer softmax (fixed-point S and m, lazy rescale at 8)
        for h in range(H):
            K, V = k[:, h // rep, :], v[:, h // rep, :]
            for a in range(0, L, q_block):
                rows = np.arange(a, min(a + q_block, L))
                m = np.full(len(rows), MINIT, np.int64)
                l = np.zeros(len(rows), np.float32)
                O = np.zeros((len(rows), hd), np.float32)
                for b in range(0, L, k_block):
                    cols = np.arange(b, min(b + k_block, L))
                    vis = np.abs(rows[:, None] - cols[None, :]) <= window if kind == "sliding_attention" else                         np.ones((len(rows), len(cols)), bool)
                    if not vis.any():
                        continue
                    S = (q[rows, h, :] @ K[cols].T).astype(np.float32)
                    m, l, O = fixed_block(S, vis, m, l, O, V[cols])
                out[rows, h, :] = O / l[:, None]
        return out.reshape(L, H * hd)
    for h in range(H):
        K, V = k[:, h // rep, :], v[:, h // rep, :]
        for a in range(0, L, q_block):
            Q = q[a:a + q_block, h, :]
            rows = np.arange(a, min(a + q_block, L))
            m = np.full(len(rows), -np.inf, np.float32)
            l = np.zeros(len(rows), np.float32)
            O = np.zeros((len(rows), hd), np.float32)
            for b in range(0, L, k_block):
                cols = np.arange(b, min(b + k_block, L))
                if kind == "sliding_attention":
                    band = np.abs(rows[:, None] - cols[None, :]) <= window
                    if not band.any():
                        continue
                else:
                    band = None
                S = (Q @ K[b:b + k_block].T).astype(np.float32)
                if band is not None:
                    S = np.where(band, S, -np.inf).astype(np.float32)
                bm = S.max(axis=1)
                mn = np.where(bm > m + tau, bm, m).astype(np.float32) if tau > 0 else np.maximum(m, bm)
                # A row that has seen no visible key yet keeps m = -inf; give it a finite reference so its
                # alpha and P are 0 instead of exp(-inf - -inf) (a kernel selects on l == 0 the same way).
                ref = np.where(np.isfinite(mn), mn, 0).astype(np.float32)
                ex = (lambda z: lut_exp(z, tau)) if exp == "lut" else np.exp
                with np.errstate(invalid="ignore", over="ignore"):
                    alpha = np.where(np.isfinite(m), ex(np.where(np.isfinite(m), m - ref, 0)), 0).astype(np.float32)
                    P = np.where(np.isfinite(S), ex(np.where(np.isfinite(S), S - ref[:, None], 0)), 0).astype(np.float32)
                if exp_bf16:
                    alpha, P = bf16_round(alpha), bf16_round(P)
                l = alpha * l + P.sum(axis=1)
                Pm = bf16_round(P) if p_bf16 else P
                O = alpha[:, None] * O + Pm @ V[b:b + k_block]
                m = mn
            out[a:a + q_block, h, :] = O / l[:, None]
    return out.reshape(L, H * hd)
