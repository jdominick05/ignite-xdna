#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignite_xdna/quantization/__init__.py

Standalone Python Post-Training Quantization (PTQ) Package for AMD Phoenix XDNA1.
Implements Cross-Layer Equalization (CLE), High-Bias Absorption, and Adaptive
Rounding (AdaRound) optimization to eliminate external Vitis-AI Docker dependencies.
"""

from ignite_xdna.quantization.cle import (
    cross_layer_equalize,
    compute_channel_ranges,
    compute_equalization_scales,
    equalize_conv_weights,
    perform_high_bias_absorption,
    verify_cle_mathematical_equivalence,
    ClePair,
    CleReport,
)

from ignite_xdna.quantization.adaround import (
    adaround_optimize,
    AdaRoundOptimizer,
    FastFinetuneConfig,
    AdaRoundReport,
    rectified_sigmoid,
    rectified_sigmoid_grad,
    compute_symmetric_scale,
)

from ignite_xdna.quantization.calib import (
    CalibrationDataset,
    collect_calibration_activations,
    letterbox,
    preprocess_image,
)

from ignite_xdna.quantization.engine import (
    PTQEngine,
    QuantizationConfig,
    quantize_model,
    HEAD_OUTS,
)

__all__ = [
    # CLE & Bias Absorption
    "cross_layer_equalize",
    "compute_channel_ranges",
    "compute_equalization_scales",
    "equalize_conv_weights",
    "perform_high_bias_absorption",
    "verify_cle_mathematical_equivalence",
    "ClePair",
    "CleReport",
    # AdaRound Optimization
    "adaround_optimize",
    "AdaRoundOptimizer",
    "FastFinetuneConfig",
    "AdaRoundReport",
    "rectified_sigmoid",
    "rectified_sigmoid_grad",
    "compute_symmetric_scale",
    # Calibration
    "CalibrationDataset",
    "collect_calibration_activations",
    "letterbox",
    "preprocess_image",
    # Engine & Configuration
    "PTQEngine",
    "QuantizationConfig",
    "quantize_model",
    "HEAD_OUTS",
]
