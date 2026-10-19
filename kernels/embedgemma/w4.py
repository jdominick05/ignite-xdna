# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""W4 weights for EmbeddingGemma 2's weight products: int4 in memory, dequantized to bf16 for the bf16 GEMM.

Configuration, fixed before the first accuracy run (2026-10-19) and not swept against the gate documents:
    symmetric round-to-nearest, q in [-8, 7]; one scale per group of G = 128 consecutive K rows of each output
    column, s = bf16(max |w| / 7) (an all-zero group gets s = 1); the value the GEMM sees is bf16(q * s).
All 146 weight products are quantized (embeddings are not weight products and stay as they are).

    q, s = quantize(w, group)     w [K, N] fp32 -> q int8 [K, N] (values in [-8, 7]), s fp32 [K / group, N] (bf16)
    dequant(q, s)                 -> fp32 [K, N] holding exact bf16 values
    pack(q)                       -> uint8 [K, N / 2], two 4-bit codes per byte (low nibble = even column)
"""
import numpy as np


def bf16(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def quantize(w, group=128):
    w = np.asarray(w, np.float32)
    K, N = w.shape
    if K % group:
        raise ValueError(f"K = {K} is not a multiple of the group {group}")
    g = w.reshape(K // group, group, N)
    s = bf16(np.abs(g).max(axis=1) / np.float32(7))
    s = np.where(s == 0, np.float32(1), s).astype(np.float32)
    q = np.clip(np.round(g / s[:, None, :]), -8, 7).astype(np.int8)
    return q.reshape(K, N), s


def dequant(q, s):
    K, N = q.shape
    group = K // s.shape[0]
    return bf16(q.reshape(-1, group, N).astype(np.float32) * s[:, None, :]).reshape(K, N)


def pack(q):
    u = (np.asarray(q, np.int8).astype(np.uint8) & 0xF)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)
