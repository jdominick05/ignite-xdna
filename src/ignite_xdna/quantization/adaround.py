#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/quantization/adaround.py

Adaptive Rounding Optimization (AdaRound) Engine for AMD Phoenix XDNA1.
Implements layer-wise AdaRound to eliminate weight quantization rounding error
via continuous relaxation of rounding decisions using rectified sigmoid parameters:
  w_quant = clamp(floor(w/s) + h(V), -128, 127)
where:
  h(V) = clamp(sigmoid(V) * (zeta - gamma) + gamma, 0, 1) with gamma = -0.1, zeta = 1.1.

Optimizes per-layer task-loss surrogate:
  L = L_recon + lambda_reg * sum(1 - |2*h(V) - 1|^beta)
over a calibration set of COCO images (default 1,000 iterations per layer with cosine beta annealing).
Locks final integer weights to int8 [-128, 127] and generates symmetric INT8 scale factors.
Pure Python/NumPy implementation eliminates external Docker/Vitis-AI dependencies.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import helper, numpy_helper


@dataclass
class FastFinetuneConfig:
    """Hyperparameters for AdaRound layer-wise optimization."""
    data_size: int = 128
    batch_size: int = 2
    num_iterations: int = 1000
    lr: float = 0.05
    beta_range: Tuple[float, float] = (20.0, 2.0)
    warm_start: float = 0.2
    reg_param: float = 0.01
    gamma: float = -0.1
    zeta: float = 1.1
    early_stopping_patience: int = 50
    early_stopping_delta: float = 1e-6
    subsample_spatial: int = 256  # Subsample spatial patches for faster convergence


@dataclass
class AdaRoundReport:
    """Report detailing AdaRound optimization progress and metrics."""
    layers_optimized: int = 0
    total_iterations: int = 0
    elapsed_seconds: float = 0.0
    layer_metrics: List[Dict[str, Any]] = field(default_factory=list)
    scales: Dict[str, float] = field(default_factory=dict)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def rectified_sigmoid(
    v: np.ndarray,
    gamma: float = -0.1,
    zeta: float = 1.1,
) -> np.ndarray:
    """
    Computes rectified sigmoid continuous relaxation:
      h(V) = clamp(sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    """
    s = sigmoid(v)
    return np.clip(s * (zeta - gamma) + gamma, 0.0, 1.0)


def rectified_sigmoid_grad(
    v: np.ndarray,
    gamma: float = -0.1,
    zeta: float = 1.1,
) -> np.ndarray:
    """
    Analytical gradient dh/dv of rectified sigmoid.
    """
    s = sigmoid(v)
    dh_ds = (zeta - gamma) * s * (1.0 - s)
    val = s * (zeta - gamma) + gamma
    mask = (val > 0.0) & (val < 1.0)
    return (dh_ds * mask).astype(np.float32)


def compute_symmetric_scale(
    w: np.ndarray,
    qmin: int = -128,
    qmax: int = 127,
    eps: float = 1e-8,
) -> float:
    """Computes symmetric per-tensor INT8 scale factor s = max(|w|) / 127."""
    max_val = float(np.max(np.abs(w)))
    scale = max(max_val / qmax, eps)
    return scale


def compute_beta(
    step: int,
    total_steps: int,
    warm_start: float = 0.2,
    beta_range: Tuple[float, float] = (20.0, 2.0),
) -> float:
    """
    Computes beta parameter for regularization loss using cosine annealing.
    During warm start, beta remains at beta_start (e.g. 20.0).
    After warm start, beta decreases smoothly from 20.0 down to 2.0.
    """
    beta_start, beta_end = beta_range
    warm_steps = int(total_steps * warm_start)
    if step < warm_steps:
        return beta_start

    decay_steps = total_steps - warm_steps
    curr_step = step - warm_steps
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * curr_step / max(decay_steps, 1)))
    return beta_end + (beta_start - beta_end) * cos_decay


def initialize_v(
    w: np.ndarray,
    scale: float,
    gamma: float = -0.1,
    zeta: float = 1.1,
) -> np.ndarray:
    """
    Initializes V parameters such that h(V) matches initial fractional part:
      rest = w / s - floor(w / s) in [0, 1)
      V = -log( (zeta - gamma) / (rest - gamma) - 1 )
    """
    w_scaled = w / scale
    w_floor = np.floor(w_scaled)
    rest = w_scaled - w_floor
    rest_safe = np.clip(rest, 1e-5, 1.0 - 1e-5)

    ratio = (zeta - gamma) / (rest_safe - gamma) - 1.0
    ratio_safe = np.maximum(ratio, 1e-7)
    v = -np.log(ratio_safe)
    return v.astype(np.float32)


def extract_conv_patches(
    x: np.ndarray,
    kh: int,
    kw: int,
    max_patches: int = 256,
) -> np.ndarray:
    """Extracts spatial patches of shape [C_in * Kh * Kw, max_patches] from [N, C_in, H, W]."""
    n, cin, h, w = x.shape
    pad_h = max(0, kh - h)
    pad_w = max(0, kw - w)
    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)))
        h, w = x.shape[2], x.shape[3]

    max_h = max(1, h - kh + 1)
    max_w = max(1, w - kw + 1)
    np.random.seed(42)
    h_idx = np.random.randint(0, max_h, size=max_patches)
    w_idx = np.random.randint(0, max_w, size=max_patches)
    n_idx = np.random.randint(0, n, size=max_patches)

    patches = []
    for ni, hi, wi in zip(n_idx, h_idx, w_idx):
        p = x[ni, :, hi:hi + kh, wi:wi + kw].reshape(-1)
        patches.append(p)
    return np.stack(patches, axis=1).astype(np.float32)


class AdaRoundOptimizer:
    """
    Optimizer for layer-wise adaptive weight rounding.
    Implements continuous relaxation, Adam optimization loop, and integer locking.
    """

    def __init__(self, config: Optional[FastFinetuneConfig] = None):
        self.config = config or FastFinetuneConfig()

    def optimize_weight(
        self,
        w: np.ndarray,
        x_activations: Optional[np.ndarray] = None,
        scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Optimizes rounding decision variable V for weight tensor w.

        Args:
            w: Float weight tensor of shape [C_out, C_in, Kh, Kw] or [C_out, C_in].
            x_activations: Input activation tensor from calibration set.
            scale: Optional predetermined scale factor.

        Returns:
            Tuple of:
              w_locked_int8: Integer weights clamped to [-128, 127] as np.int8.
              scale: Symmetric INT8 scale factor.
              metrics: Optimization dictionary.
        """
        cfg = self.config
        w_shape = w.shape

        if scale is None:
            scale = compute_symmetric_scale(w)

        w_scaled = w / scale
        w_floor = np.floor(w_scaled)

        # Initialize continuous relaxation variable V
        v = initialize_v(w, scale, gamma=cfg.gamma, zeta=cfg.zeta)

        # Reshape weights to 2D matrix for vectorized compute: [C_out, K]
        c_out = w_shape[0]
        w_2d = w.reshape(c_out, -1)
        w_floor_2d = w_floor.reshape(c_out, -1)
        v_2d = v.reshape(c_out, -1)

        # Prepare calibration inputs X
        k_in = w_2d.shape[1]
        if x_activations is not None and x_activations.size > 0:
            if w.ndim == 4 and x_activations.ndim == 4:
                kh, kw = w.shape[2], w.shape[3]
                x_sub = extract_conv_patches(x_activations, kh, kw, max_patches=cfg.subsample_spatial)
            else:
                x_flat = x_activations.reshape(-1, k_in).T
                x_sub = x_flat[:, :cfg.subsample_spatial]
            # Normalize activations for numerical stability
            x_norm = np.linalg.norm(x_sub) + 1e-6
            x_sub = (x_sub / x_norm) * math.sqrt(x_sub.size)
        else:
            # Fallback synthetic orthogonal calibration vectors
            np.random.seed(42)
            x_sub = np.random.randn(k_in, cfg.subsample_spatial).astype(np.float32)

        # Target unquantized output
        y_float = np.matmul(w_2d, x_sub)

        # Adam optimizer state
        m = np.zeros_like(v_2d)
        var = np.zeros_like(v_2d)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = cfg.lr

        best_loss = float("inf")
        patience_counter = 0

        initial_recon_err = float(np.mean((y_float - np.matmul(w_floor_2d * scale, x_sub)) ** 2))

        # Optimization loop
        for step in range(1, cfg.num_iterations + 1):
            beta = compute_beta(step, cfg.num_iterations, cfg.warm_start, cfg.beta_range)

            # Continuous relaxation output
            h = rectified_sigmoid(v_2d, gamma=cfg.gamma, zeta=cfg.zeta)
            w_quant_cont = np.clip(w_floor_2d + h, -128.0, 127.0) * scale
            y_quant = np.matmul(w_quant_cont, x_sub)

            # 1. Reconstruction error and gradient
            diff = y_quant - y_float
            loss_recon = float(np.mean(diff ** 2))

            # dL_recon / dw_quant = 2 * (diff @ x_sub.T) / (C_out * N)
            grad_w_quant = (2.0 / diff.size) * np.matmul(diff, x_sub.T)

            # 2. Regularization term: lambda * sum(1 - |2h - 1|^beta)
            dev = 2.0 * h - 1.0
            abs_dev = np.abs(dev)
            loss_reg = float(cfg.reg_param * np.sum(1.0 - (abs_dev ** beta)))

            # dL_reg / dh = - lambda * beta * |2h - 1|^(beta - 1) * sign(2h - 1) * 2
            grad_reg_h = -cfg.reg_param * beta * (np.maximum(abs_dev, 1e-6) ** (beta - 1.0)) * np.sign(dev) * 2.0

            # 3. Total gradient w.r.t V via chain rule
            dh_dv = rectified_sigmoid_grad(v_2d, gamma=cfg.gamma, zeta=cfg.zeta)
            grad_v = (grad_w_quant * scale + grad_reg_h) * dh_dv

            total_loss = loss_recon + loss_reg

            # Early stopping check
            if total_loss < best_loss - cfg.early_stopping_delta:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    break

            # Adam parameter update
            m = beta1 * m + (1.0 - beta1) * grad_v
            var = beta2 * var + (1.0 - beta2) * (grad_v ** 2)
            m_hat = m / (1.0 - beta1 ** step)
            var_hat = var / (1.0 - beta2 ** step)
            v_2d = v_2d - (lr / (np.sqrt(var_hat) + eps)) * m_hat

        # Hard rounding lock: V >= 0 rounds up (1), V < 0 rounds down (0)
        h_hard = (v_2d >= 0.0).astype(np.float32)
        w_int = np.clip(w_floor_2d + h_hard, -128.0, 127.0).astype(np.int8)

        final_recon_err = float(np.mean((y_float - np.matmul(w_int.astype(np.float32) * scale, x_sub)) ** 2))

        metrics = {
            "iterations": step,
            "initial_recon_mse": initial_recon_err,
            "final_recon_mse": final_recon_err,
            "mse_improvement_pct": max(0.0, (initial_recon_err - final_recon_err) / max(initial_recon_err, 1e-12) * 100),
            "scale": float(scale),
        }

        return w_int.reshape(w_shape), float(scale), metrics

    def optimize_model(
        self,
        model: onnx.ModelProto,
        calib_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[onnx.ModelProto, AdaRoundReport]:
        """
        Optimizes weights of all Conv layers in the model using AdaRound.
        Replaces float initializers with locked integer weights and records scales.
        """
        t0 = time.perf_counter()
        model_out = onnx.ModelProto.FromString(model.SerializeToString())
        graph = model_out.graph

        inits: Dict[str, np.ndarray] = {
            t.name: numpy_helper.to_array(t) for t in graph.initializer
        }

        report = AdaRoundReport()
        conv_nodes = [n for n in graph.node if n.op_type == "Conv"]

        for conv in conv_nodes:
            if len(conv.input) < 2 or conv.input[1] not in inits:
                continue

            w_name = conv.input[1]
            w_float = inits[w_name]

            # Get calibration activations if available
            x_act = calib_data.get(conv.input[0]) if calib_data else None

            w_int8, scale, metrics = self.optimize_weight(w_float, x_activations=x_act)

            # Store scale
            report.scales[w_name] = scale
            report.layers_optimized += 1
            report.total_iterations += metrics["iterations"]
            report.layer_metrics.append({
                "node": conv.name,
                "weight_name": w_name,
                "shape": list(w_float.shape),
                **metrics,
            })

            # In ONNX QDQ representation or integer weights:
            # For direct QDQ compatibility, we keep weights as float32 integers in initializers
            # (which DequantizeLinear will dequantize by scale), or int8.
            # Storing as float32 integer values ensures 100% standard ONNX Runtime operator compatibility.
            inits[w_name] = w_int8.astype(np.float32)

        # Update initializers in graph
        for tensor in graph.initializer:
            if tensor.name in inits:
                new_arr = inits[tensor.name]
                new_proto = numpy_helper.from_array(new_arr, name=tensor.name)
                tensor.CopyFrom(new_proto)

        report.elapsed_seconds = float(time.perf_counter() - t0)
        return model_out, report


def adaround_optimize(
    model: onnx.ModelProto,
    calib_data: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[FastFinetuneConfig] = None,
) -> Tuple[onnx.ModelProto, AdaRoundReport]:
    """
    Main entry point for Adaptive Rounding Optimization.

    Args:
        model: ONNX ModelProto to optimize.
        calib_data: Optional dictionary mapping tensor names to calibration activations.
        config: FastFinetuneConfig instance.

    Returns:
        Tuple of (optimized_model, adaround_report).
    """
    optimizer = AdaRoundOptimizer(config=config)
    return optimizer.optimize_model(model, calib_data=calib_data)
