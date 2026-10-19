# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Low-bit weights for EmbeddingGemma 2's weight products (W4 / W8), dequantized to bf16 for the bf16 GEMM.

Symmetric, one scale per group of G consecutive K rows of each output column, s = bf16(alpha * max |w| / qmax)
(an all-zero group gets s = 1), the GEMM sees bf16(q * s). qmax = 7 for 4 bits (q in [-8, 7]) and 127 for 8 bits
(q in [-127, 127], the repo's INT8 producer clipping). alpha = 1 is plain round-to-nearest; ``clip_points`` > 0
picks alpha per group from linspace(clip_lo, 1, clip_points) by least squared weight error, using the weights only
(no activations, nothing from the gate documents). All 146 weight products are quantized; embeddings stay.

Configurations run (each fixed before its run, none swept on the gate documents):
    W4 RTN G=128 (2026-10-19): gate FAIL.
    W8 RTN G=128 and W8 G=64 with a 20-point clip search over alpha in [0.85, 1] (the user's proposal, 2026-10-19).

    q, s = quantize(w, group, bits=4, clip_points=0, clip_lo=0.85)
    dequant(q, s)                 -> fp32 [K, N] holding exact bf16 values
    pack(q)                       -> uint8 [K, N / 2], two 4-bit codes per byte (low nibble = even column)
"""
import numpy as np


def bf16(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def _q(g, s, lo, hi):
    return np.clip(np.round(g / s[:, None, :]), lo, hi)


def quantize(w, group=128, bits=4, clip_points=0, clip_lo=0.85):
    w = np.asarray(w, np.float32)
    K, N = w.shape
    if K % group:
        raise ValueError(f"K = {K} is not a multiple of the group {group}")
    qmax = {4: 7, 8: 127}[bits]
    lo = -8 if bits == 4 else -127
    g = w.reshape(K // group, group, N)
    amax = np.abs(g).max(axis=1)

    def scale(alpha):
        s = bf16(np.float32(alpha) * amax / np.float32(qmax))
        return np.where(s == 0, np.float32(1), s).astype(np.float32)
    s = scale(1.0)
    if clip_points:
        best = None
        for alpha in np.linspace(clip_lo, 1.0, clip_points):
            sa = scale(alpha)
            err = ((bf16(_q(g, sa, lo, qmax) * sa[:, None, :]) - g) ** 2).sum(axis=1)
            if best is None:
                best, s = err, sa
            else:
                take = err < best
                best = np.where(take, err, best)
                s = np.where(take, sa, s)
    return _q(g, s, lo, qmax).astype(np.int8).reshape(K, N), s


def quantize_rows(x):
    """W8A8 activations, per token over all of K: s = bf16(max |x| / 127) per row (an all-zero row gets 1),
    q = clip(rne(x / s), -127, 127) -> (q int8 [R, K], s fp32 [R] holding bf16 values)."""
    x = np.asarray(x, np.float32)
    s = bf16(np.abs(x).max(axis=1) / np.float32(127))
    s = np.where(s == 0, np.float32(1), s).astype(np.float32)
    return np.clip(np.rint(x / s[:, None]), -127, 127).astype(np.int8), s


def dequant(q, s):
    K, N = q.shape
    group = K // s.shape[0]
    return bf16(q.reshape(-1, group, N).astype(np.float32) * s[:, None, :]).reshape(K, N)


def pack(q):
    u = (np.asarray(q, np.int8).astype(np.uint8) & 0xF)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)
