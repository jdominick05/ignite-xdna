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
"""
import numpy as np

from model import bf16_round


def attention(q, k, v, kind, window, *, q_block=16, k_block=32, qkv_bf16=True, p_bf16=True, exp_bf16=True):
    """q [L, H, hd], k/v [L, kv, hd] fp32 (normed, rotated) for ONE document."""
    L, H, hd = q.shape
    kv = k.shape[1]
    rep = H // kv
    if qkv_bf16:
        q, k, v = bf16_round(q), bf16_round(k), bf16_round(v)
    out = np.empty((L, H, hd), np.float32)
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
                mn = np.maximum(m, S.max(axis=1))
                # A row that has seen no visible key yet keeps m = -inf; give it a finite reference so its
                # alpha and P are 0 instead of exp(-inf - -inf) (a kernel selects on l == 0 the same way).
                ref = np.where(np.isfinite(mn), mn, 0).astype(np.float32)
                alpha = np.where(np.isfinite(m), np.exp(m - ref), 0).astype(np.float32)
                P = np.exp(S - ref[:, None]).astype(np.float32)
                if exp_bf16:
                    alpha, P = bf16_round(alpha), bf16_round(P)
                l = alpha * l + P.sum(axis=1)
                Pm = bf16_round(P) if p_bf16 else P
                O = alpha[:, None] * O + Pm @ V[b:b + k_block]
                m = mn
            out[a:a + q_block, h, :] = O / l[:, None]
    return out.reshape(L, H * hd)
