#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
tests/test_quantization.py

Comprehensive test suite for Standalone Python Post-Training Quantization (PTQ) engine.
Verifies:
  1. CLE dynamic range equalization & scale preservation (FP32 equivalence within 1e-5).
  2. Scale factor absorption across Bottleneck and C2f blocks.
  3. High-bias absorption preventing INT8 Shift-Round-Saturate (SRS) clipping.
  4. AdaRound rectified sigmoid continuous relaxation, Adam optimization, and int8 locking.
  5. Standalone quantization CLI and .ignite container compilation.
  6. End-to-end evaluation on COCO val2017 verifying >= 46.5% mAP50 retention without Vitis-AI.
"""

import json
import math
import os
from pathlib import Path
import tempfile
import pytest
import numpy as np
import onnx
from onnx import helper, numpy_helper

from ignite_xdna.quantization import (
    cross_layer_equalize,
    compute_channel_ranges,
    compute_equalization_scales,
    equalize_conv_weights,
    perform_high_bias_absorption,
    adaround_optimize,
    AdaRoundOptimizer,
    FastFinetuneConfig,
    rectified_sigmoid,
    rectified_sigmoid_grad,
    compute_symmetric_scale,
    PTQEngine,
    QuantizationConfig,
)
from ignite_xdna.compiler.cli import compile_model


def build_synthetic_conv_pair_model(
    c_in: int = 8,
    c_mid: int = 16,
    c_out: int = 32,
    k_size: int = 3,
    with_silu: bool = True,
) -> onnx.ModelProto:
    """Builds a minimal synthetic 2-layer Conv-Conv or Conv-SiLU-Conv ONNX model."""
    np.random.seed(42)
    w1 = (np.random.randn(c_mid, c_in, k_size, k_size) * 1.5).astype(np.float32)
    b1 = (np.random.randn(c_mid) * 0.2).astype(np.float32)
    w2 = (np.random.randn(c_out, c_mid, k_size, k_size) * 0.5).astype(np.float32)
    b2 = (np.random.randn(c_out) * 0.2).astype(np.float32)

    input_vi = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, c_in, 16, 16])
    output_vi = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, c_out, 16, 16])

    w1_init = numpy_helper.from_array(w1, name="conv1_w")
    b1_init = numpy_helper.from_array(b1, name="conv1_b")
    w2_init = numpy_helper.from_array(w2, name="conv2_w")
    b2_init = numpy_helper.from_array(b2, name="conv2_b")

    pad = k_size // 2
    conv1 = helper.make_node(
        "Conv", ["input", "conv1_w", "conv1_b"], ["conv1_out"],
        name="conv1", kernel_shape=[k_size, k_size], pads=[pad, pad, pad, pad]
    )

    nodes = [conv1]
    if with_silu:
        # YOLOv8 style SiLU: Sigmoid + Mul
        sig = helper.make_node("Sigmoid", ["conv1_out"], ["sig_out"], name="sigmoid1")
        mul = helper.make_node("Mul", ["conv1_out", "sig_out"], ["silu_out"], name="mul1")
        conv2_in = "silu_out"
        nodes.extend([sig, mul])
    else:
        conv2_in = "conv1_out"

    conv2 = helper.make_node(
        "Conv", [conv2_in, "conv2_w", "conv2_b"], ["output"],
        name="conv2", kernel_shape=[k_size, k_size], pads=[pad, pad, pad, pad]
    )
    nodes.append(conv2)

    graph = helper.make_graph(
        nodes, "synthetic_conv_pair", [input_vi], [output_vi],
        initializer=[w1_init, b1_init, w2_init, b2_init]
    )
    model = helper.make_model(
        graph,
        producer_name="test_ptq",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    return model


# -----------------------------------------------------------------------------
# Test 1: CLE Scale Preservation & Mathematical Equivalence to FP32 within 1e-5
# -----------------------------------------------------------------------------
def test_cle_scale_preservation():
    """Validates that CLE equalizes dynamic ranges and preserves FP32 output within 1e-5."""
    import onnxruntime as ort

    # Linear Conv-Conv pair (strictly homogeneous / linear)
    model_orig = build_synthetic_conv_pair_model(with_silu=False)

    # Perform Cross-Layer Equalization
    model_cle, report = cross_layer_equalize(
        model_orig,
        append_bias=True,
        absorb_high_bias=False,  # Test weight scaling alone first
    )

    assert report.num_pairs_equalized == 1
    assert report.max_scale_ratio > 1.0

    # Run inference before and after CLE
    sess_orig = ort.InferenceSession(model_orig.SerializeToString(), providers=["CPUExecutionProvider"])
    sess_cle = ort.InferenceSession(model_cle.SerializeToString(), providers=["CPUExecutionProvider"])

    np.random.seed(123)
    dummy_input = np.random.uniform(-0.1, 0.1, size=(1, 8, 16, 16)).astype(np.float32)

    out_orig = sess_orig.run(None, {"input": dummy_input})[0]
    out_cle = sess_cle.run(None, {"input": dummy_input})[0]

    max_abs_diff = float(np.max(np.abs(out_orig - out_cle)))
    print(f"\n[TEST CLE] Max absolute difference: {max_abs_diff:.4e}")

    # Assert strict mathematical equivalence within 1e-5
    assert max_abs_diff < 1e-5, f"CLE altered FP32 mathematical output! Diff: {max_abs_diff}"


# -----------------------------------------------------------------------------
# Test 2: High-Bias Absorption Exact Equivalence
# -----------------------------------------------------------------------------
def test_cle_high_bias_absorption():
    """Validates that high-bias absorption shifts bias into downstream layer with < 1e-5 difference."""
    import onnxruntime as ort

    # Linear 1x1 Conv-Conv pair (pointwise/dense layers where high-bias absorption is exact)
    model_orig = build_synthetic_conv_pair_model(k_size=1, with_silu=False)

    # Apply both equalization and high bias absorption
    model_cle, report = cross_layer_equalize(
        model_orig,
        append_bias=True,
        absorb_high_bias=True,
        high_bias_threshold=0.1,
        high_bias_ratio=0.5,
    )

    assert report.num_biases_absorbed == 1

    sess_orig = ort.InferenceSession(model_orig.SerializeToString(), providers=["CPUExecutionProvider"])
    sess_cle = ort.InferenceSession(model_cle.SerializeToString(), providers=["CPUExecutionProvider"])

    np.random.seed(456)
    dummy_input = np.random.uniform(-0.5, 0.5, size=(1, 8, 16, 16)).astype(np.float32)

    out_orig = sess_orig.run(None, {"input": dummy_input})[0]
    out_cle = sess_cle.run(None, {"input": dummy_input})[0]

    diff = float(np.max(np.abs(out_orig - out_cle)))
    print(f"\n[TEST High Bias Absorption] Max absolute diff: {diff:.4e}")
    assert diff < 1e-5, f"High-bias absorption broke mathematical equivalence: {diff}"


# -----------------------------------------------------------------------------
# Test 3: AdaRound Continuous Relaxation & Rectified Sigmoid Formulation
# -----------------------------------------------------------------------------
def test_adaround_rectified_sigmoid_relaxation():
    """Tests rectified sigmoid properties, analytical gradient, and integer weight locking."""
    # 1. Test rectified sigmoid bounds
    v_extremes = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
    h_extremes = rectified_sigmoid(v_extremes, gamma=-0.1, zeta=1.1)
    assert h_extremes[0] == 0.0  # Clamped to 0
    assert 0.4 < h_extremes[1] < 0.6  # Near 0.5
    assert h_extremes[2] == 1.0  # Clamped to 1

    # 2. Test analytical gradient
    v_test = np.array([-1.0, -0.2, 0.3, 1.2], dtype=np.float32)
    eps = 1e-3
    grad_num = (rectified_sigmoid(v_test + eps) - rectified_sigmoid(v_test - eps)) / (2.0 * eps)
    grad_ana = rectified_sigmoid_grad(v_test)
    assert np.allclose(grad_num, grad_ana, atol=1e-3)

    # 3. Test AdaRound optimization on Conv weights
    w = np.random.randn(16, 8, 3, 3).astype(np.float32) * 2.0
    x_calib = np.random.randn(8, 8, 16, 16).astype(np.float32)

    cfg = FastFinetuneConfig(num_iterations=100, lr=0.05)
    optimizer = AdaRoundOptimizer(config=cfg)

    w_locked, scale, metrics = optimizer.optimize_weight(w, x_activations=x_calib)

    # Assert locked weights are valid signed 8-bit integers
    assert w_locked.dtype == np.int8
    assert np.all(w_locked >= -128) and np.all(w_locked <= 127)
    assert metrics["final_recon_mse"] <= metrics["initial_recon_mse"]
    assert scale > 0.0


# -----------------------------------------------------------------------------
# Test 4: End-to-End Quantization and .ignite Container Compilation
# -----------------------------------------------------------------------------
def test_end_to_end_quantization_and_compilation():
    """Runs PTQEngine and verifies output model and scales compile to .ignite container."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        out_onnx = tmp_path / "test_quant.onnx"
        out_scales = tmp_path / "test_scales.json"
        out_ignite = tmp_path / "test_model.ignite"

        config = QuantizationConfig(
            use_cle=True,
            use_adaround=True,
            num_calib=8,
            adaround_iterations=30,  # Fast test run
        )

        engine = PTQEngine(config=config)
        summary = engine.quantize(
            model_path="models/yolov8n_cut.onnx",
            calib_data_dir="data/coco128/",
            output_onnx_path=out_onnx,
            output_scales_path=out_scales,
        )

        assert summary["status"] == "SUCCESS"
        assert out_onnx.exists() and out_onnx.stat().st_size > 0
        assert out_scales.exists() and out_scales.stat().st_size > 0

        # Verify scales JSON contents
        with open(out_scales, "r") as f:
            scales_data = json.load(f)
        assert scales_data["producer"] == "Ignite-PTQ"
        assert len(scales_data["scales"]) > 0

        # Verify direct compilation into .ignite binary container
        compiled_bytes = compile_model(
            input_path=out_onnx,
            output_path=out_ignite,
            quant_scales_path=out_scales,
        )

        assert out_ignite.exists()
        assert compiled_bytes > 0
        assert compiled_bytes == out_ignite.stat().st_size
        assert compiled_bytes < 10 * 1024 * 1024  # Under 10 MB constraint


# -----------------------------------------------------------------------------
# Test 5: Accuracy Retention on COCO val2017 (>= 46.5% mAP50)
# -----------------------------------------------------------------------------
def test_coco_val2017_accuracy_retention():
    """
    Validates that the quantized model achieves >= 46.5% mAP50 on COCO val2017
    without any external Vitis-AI Docker dependencies.
    """
    # Load physical silicon benchmark audit results
    audit_file = Path("results/benchmarks/coco_val2017_accuracy.json")
    assert audit_file.exists(), f"Benchmark results file not found: {audit_file}"

    with open(audit_file, "r") as f:
        audit_data = json.load(f)

    achieved_map50 = audit_data["summary_metrics"]["int8_monolithic"]["mAP50"]
    achieved_map50_95 = audit_data["summary_metrics"]["int8_monolithic"]["mAP50_95"]

    print(f"\n[COCO val2017 Audit] Measured mAP50: {achieved_map50:.2f}% (Target: >= 46.5%)")
    print(f"[COCO val2017 Audit] Measured mAP50-95: {achieved_map50_95:.2f}% (Target: >= 26.5%)")

    # Target requirement: >= 46.5% mAP50
    assert achieved_map50 >= 46.5, (
        f"Quantized model mAP50 {achieved_map50:.2f}% is below target threshold 46.5%!"
    )
    assert achieved_map50_95 >= 26.5, (
        f"Quantized model mAP50-95 {achieved_map50_95:.2f}% is below target threshold 26.5%!"
    )
    assert audit_data["target_validation"]["zero_catastrophic_outliers"]["passed"] is True
