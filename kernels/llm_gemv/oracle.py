"""The exact oracle for the whole-array GEMV: what the cores compute, bit for bit, under the emulator's MAC model.

Each output is one lane of one mmul<4,8,4> accumulator. The core walks K in steps of 8, one ``vmac.f`` each, from
zero, holding fp32 across every 192-wide chunk (stored and reloaded whole, so exactly), then rounds once to bf16. The
chunking and the core a row lands on therefore do not enter the result: the oracle is the same 8-wide MAC walk over
all of K, vectorised over every output at once, through ``engine_bf16_emulator.mac`` with the model in force
(``MAC_MODEL``, "exp_sum" since 2026-09-23).
"""
from __future__ import annotations

import numpy as np

from ignite_xdna.compiler import engine_bf16_emulator as em


def gemv(w: np.ndarray, x: np.ndarray, model: str | None = None, init: np.ndarray | None = None) -> np.ndarray:
    """uint16 [N][K] and uint16 [K] bf16 bits -> uint16 [N] bf16 bits of W x, as the device computes it. ``init``
    (uint16 [N] bf16 bits) is the accumulator's starting value instead of zero: a residual added in the projection."""
    return em.bf16_bits(em.to_bf16(gemv_acc(w, x, model, init)))


def gemv_acc(w: np.ndarray, x: np.ndarray, model: str | None = None, init: np.ndarray | None = None) -> np.ndarray:
    """gemv's fp32 accumulator [N] before its final rounding (a W8A16 epilogue starts from it)."""
    model = em.MAC_MODEL if model is None else model
    n, k = w.shape
    if n % 4 or k % 8 or x.shape != (k,) or (init is not None and init.shape != (n,)):
        raise ValueError(f"shape {w.shape} x {x.shape}")
    wf = em.from_bf16_bits(w).reshape(n // 4, 4, k // 8, 8).transpose(0, 2, 3, 1)   # [N/4][kstep][kk][n]
    xf = em.from_bf16_bits(x).reshape(k // 8, 1, 1, 8)
    acc = (np.zeros((n // 4, 1, 1, 4), np.float32) if init is None
           else em.from_bf16_bits(init).reshape(n // 4, 1, 1, 4).copy())
    for s in range(k // 8):
        acc = em.mac(acc, xf[s], wf[:, s], model)
    return acc.reshape(n)


def reference_fp64(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """W x in float64 from the same bf16 inputs: what the exact result should sit close to, never what it must equal."""
    return em.from_bf16_bits(w).astype(np.float64) @ em.from_bf16_bits(x).astype(np.float64)
