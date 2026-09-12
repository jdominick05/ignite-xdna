#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/runtime/parity.py
Numerical parity metrics and hardware SRS mathematical emulator for AMD Phoenix AIE2.
"""

from typing import Dict, Any
import numpy as np


def srs_s8_s32(accum: np.ndarray, shift: int = 10) -> np.ndarray:
    """
    Bit-exact emulator for hardware vst.srs.s8.s32 instruction.
    Computes: saturate_s8((accum + (1 << (shift - 1))) >> shift)
    """
    if shift <= 0:
        rounded = accum.astype(np.int64)
    else:
        bias = 1 << (shift - 1)
        # In AIE2 architecture, rounding adds 1 << (shift - 1)
        rounded = np.right_shift(accum.astype(np.int64) + bias, shift)
    return np.clip(rounded, -128, 127).astype(np.int8)


class Im2ColGoldenReference:
    """
    NumPy bit-exact emulator for 3x3 im2col spatial convolution on AIE2 cores.
    Each core computes M=2 spatial patches (288 bytes each) against stationary weights (2304 bytes),
    accumulates into 4 32-lane INT32 accumulator registers, and saturates via SRS into 256 INT8 outputs.
    """
    def __init__(self, shift_bias: int = 10, seed: int = 42):
        self.shift_bias = shift_bias
        self.rng = np.random.RandomState(seed)

    def generate_inputs_and_weights(self, num_cores: int = 4):
        """
        Generate deterministic INT8 input feature map and stationary weights.
        For num_cores=4 (1 column):
          - Ingress: 2048 bytes (8x8x32 feature map)
          - Weights: 2304 bytes (9 taps x 4 blocks x 64 bytes)
          - Expected Egress: 1024 bytes (4 cores x 256 bytes)
        For num_cores=20 (5 columns):
          - Ingress: 10240 bytes (5 x 2048 bytes)
          - Weights: 2304 bytes stationary
          - Expected Egress: 5120 bytes (20 cores x 256 bytes)
        """
        in_bytes = num_cores * 512
        if num_cores == 4:
            in_bytes = 2048
        elif num_cores == 20:
            in_bytes = 10240

        inputs = self.rng.randint(-16, 16, size=in_bytes, dtype=np.int8)
        weights = self.rng.randint(-8, 8, size=2304, dtype=np.int8)
        return inputs, weights

    def compute_core_output(self, patch_a: np.ndarray, patch_b: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Simulate conv_im2col_kernel_m2_srs on two 288-byte patches with stationary weights.
        Returns 256 INT8 elements (128 from patch_a, 128 from patch_b).
        """
        w_taps = weights.reshape(9, 4, 64)
        p_a = patch_a[:288].reshape(9, 32)
        p_b = patch_b[:288].reshape(9, 32)

        acc_a = np.zeros((4, 32), dtype=np.int32)
        acc_b = np.zeros((4, 32), dtype=np.int32)

        for k in range(9):
            va = p_a[k].astype(np.int32)
            vb = p_b[k].astype(np.int32)
            for blk in range(4):
                w_blk = w_taps[k, blk].reshape(2, 32).astype(np.int32)
                acc_a[blk] += np.sum(va * w_blk[0]) + np.sum(va * w_blk[1])
                acc_b[blk] += np.sum(vb * w_blk[0]) + np.sum(vb * w_blk[1])

        out_a = srs_s8_s32(acc_a.flatten(), self.shift_bias)
        out_b = srs_s8_s32(acc_b.flatten(), self.shift_bias)
        return np.concatenate([out_a, out_b])

    def compute_full_reference(self, inputs: np.ndarray, weights: np.ndarray, num_cores: int = 4) -> np.ndarray:
        """
        Compute golden reference for all cores in the workload.
        """
        core_outputs = []
        for c in range(num_cores):
            offset = (c * 288) % (len(inputs) - 576 + 1)
            p_a = inputs[offset : offset + 288]
            p_b = inputs[offset + 288 : offset + 576]
            if len(p_a) < 288:
                p_a = np.pad(p_a, (0, 288 - len(p_a)))
            if len(p_b) < 288:
                p_b = np.pad(p_b, (0, 288 - len(p_b)))
            core_out = self.compute_core_output(p_a, p_b, weights)
            core_outputs.append(core_out)
        return np.concatenate(core_outputs)


def calculate_numerical_parity(golden: np.ndarray, actual: np.ndarray) -> Dict[str, Any]:
    """
    Compute RMSE, Max Absolute Error, and bit-for-bit agreement percentage.
    """
    g = golden.flatten().astype(np.float64)
    a = actual.flatten().astype(np.float64)
    min_len = min(len(g), len(a))
    g = g[:min_len]
    a = a[:min_len]

    diff = np.abs(g - a)
    mae = float(np.mean(diff))
    max_ae = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bit_matches = int(np.sum(g == a))
    bit_agreement_pct = float(bit_matches / min_len * 100.0)

    return {
        "count": min_len,
        "mae": mae,
        "max_ae": max_ae,
        "rmse": rmse,
        "bit_matches": bit_matches,
        "bit_agreement_pct": bit_agreement_pct,
    }
