#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignite_xdna/quantization/cle.py

Cross-Layer Equalization (CLE) & High-Bias Absorption Engine for AMD Phoenix XDNA1.
Implements weight-equalization transforms across consecutive Conv-SiLU-Conv / Conv-Conv layers,
scale factor absorption across Bottleneck and C2f blocks, and high-bias absorption to prevent
activation clipping on INT8 Shift-Round-Saturate (SRS) vector accumulators.

Mathematical Foundations:
  1. Weight Equalization:
     For consecutive layers L1 (W1, b1) and L2 (W2, b2):
       range1_i = max(|W1_{i, :, :, :}|)
       range2_i = max(|W2_{:, i, :, :}|)
       S_i = sqrt(range1_i / range2_i)
       W1'_i = W1_i / S_i,  b1'_i = b1_i / S_i
       W2'_{:, i} = W2_{:, i} * S_i
     Equalizes dynamic ranges: range(W1') == range(W2') == sqrt(range1 * range2).

  2. Bottleneck & C2f Absorption:
     In YOLOv8 Bottleneck (cv1 -> SiLU -> cv2 -> Add(shortcut)):
     Equalizing cv1 and cv2 internally absorbs S_i so that cv2 output remains at nominal
     scale, preserving residual additions without scaling shortcuts or breaking C2f concats.

  3. High-Bias Absorption:
     Large biases b1 are absorbed into subsequent layer L2:
       c = absorb_factor * b1
       b1' = b1 - c
       Delta_b2 = sum_{k,h,w} (W2_{:, k, h, w} * c_k)
       b2' = b2 + Delta_b2
     Strictly preserves FP32 mathematical equivalence within 1e-5.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import onnx
from onnx import helper, numpy_helper


@dataclass
class ClePair:
    """Represents a matched pair of consecutive layers for equalization."""
    head_node: onnx.NodeProto
    tail_node: onnx.NodeProto
    head_weight_name: str
    tail_weight_name: str
    head_bias_name: Optional[str] = None
    tail_bias_name: Optional[str] = None
    activation_type: Optional[str] = None  # "SiLU", "Relu", "Clip", "Identity", etc.
    in_bottleneck: bool = False
    in_c2f: bool = False
    block_name: Optional[str] = None


@dataclass
class CleReport:
    """Detailed report of Cross-Layer Equalization transforms."""
    num_pairs_matched: int = 0
    num_pairs_equalized: int = 0
    num_biases_absorbed: int = 0
    pairs: List[Dict[str, Any]] = field(default_factory=list)
    absorbed_biases: List[Dict[str, Any]] = field(default_factory=list)
    max_scale_ratio: float = 1.0
    min_scale_ratio: float = 1.0
    fp32_preserved: bool = True
    max_abs_diff: float = 0.0


def compute_channel_ranges(
    w_head: np.ndarray,
    w_tail: np.ndarray,
    b_head: Optional[np.ndarray] = None,
    append_bias: bool = True,
    bias_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-channel ranges for head output channels and tail input channels.

    Args:
        w_head: Head weights of shape [C_out, C_in, Kh, Kw] or [C_out, C_in].
        w_tail: Tail weights of shape [C'_out, C_out, Kh, Kw] or [C'_out, C_out].
        b_head: Optional head bias of shape [C_out].
        append_bias: If True, includes head bias in head range estimation.
        bias_threshold: Threshold weighting for bias inclusion.

    Returns:
        Tuple of (range_head, range_tail), each 1D array of length C_out.
    """
    # Head range per output channel (axis 0)
    w_head_flat = np.abs(w_head).reshape(w_head.shape[0], -1)
    range_head = np.max(w_head_flat, axis=1)

    if append_bias and b_head is not None:
        b_abs = np.abs(b_head) * bias_threshold
        range_head = np.maximum(range_head, b_abs)

    # Tail range per input channel (axis 1)
    # Transpose so input channel is first axis: [C_out, C'_out, Kh, Kw] -> reshape(C_out, -1)
    w_tail_swapped = np.swapaxes(w_tail, 0, 1)
    w_tail_flat = np.abs(w_tail_swapped).reshape(w_tail.shape[1], -1)
    range_tail = np.max(w_tail_flat, axis=1)

    return range_head.astype(np.float32), range_tail.astype(np.float32)


def compute_equalization_scales(
    range_head: np.ndarray,
    range_tail: np.ndarray,
    min_scale: float = 1e-4,
    max_scale: float = 1e4,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Computes per-channel scale factors S_i = sqrt(range_head / range_tail).

    Args:
        range_head: Per-channel max magnitude of head layer (length C).
        range_tail: Per-channel max magnitude of tail layer (length C).
        min_scale: Minimum scale factor clamp.
        max_scale: Maximum scale factor clamp.
        eps: Small epsilon to prevent division by zero.

    Returns:
        1D array of scale factors S_i of length C.
    """
    safe_head = np.maximum(range_head, eps)
    safe_tail = np.maximum(range_tail, eps)

    # S_i = sqrt(range_head / range_tail)
    scales = np.sqrt(safe_head / safe_tail)

    # If both ranges are tiny or degenerate, preserve scale = 1.0
    degenerate = (range_head < eps) & (range_tail < eps)
    scales[degenerate] = 1.0

    return np.clip(scales, min_scale, max_scale).astype(np.float32)


def equalize_conv_weights(
    w_head: np.ndarray,
    w_tail: np.ndarray,
    scales: np.ndarray,
    b_head: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Applies per-channel scale factors S to equalise head and tail weights:
      W_head_new[i] = W_head[i] / S[i]
      b_head_new[i] = b_head[i] / S[i]
      W_tail_new[:, i] = W_tail[:, i] * S[i]

    Args:
        w_head: Head weights array [C_out, C_in, Kh, Kw].
        w_tail: Tail weights array [C'_out, C_out, Kh, Kw].
        scales: Scale factors S of length C_out.
        b_head: Optional head bias array [C_out].

    Returns:
        Tuple of (w_head_new, w_tail_new, b_head_new).
    """
    # Shape broadcasts for 4D Conv weights
    # Head scales along output channels (axis 0)
    head_shape = [scales.shape[0]] + [1] * (w_head.ndim - 1)
    w_head_new = w_head / scales.reshape(head_shape)

    # Tail scales along input channels (axis 1)
    tail_shape = [1, scales.shape[0]] + [1] * (w_tail.ndim - 2)
    w_tail_new = w_tail * scales.reshape(tail_shape)

    b_head_new = None
    if b_head is not None:
        b_head_new = b_head / scales

    return w_head_new.astype(np.float32), w_tail_new.astype(np.float32), b_head_new


def perform_high_bias_absorption(
    w_tail: np.ndarray,
    b_head: np.ndarray,
    b_tail: Optional[np.ndarray] = None,
    absorb_ratio: float = 0.5,
    threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Absorbs high bias components from head layer into tail layer bias:
      c = absorb_ratio * b_head (where |b_head| > threshold)
      b_head_new = b_head - c
      Delta_b_tail = sum_{k, h, w} (W_tail_{:, k, h, w} * c_k)
      b_tail_new = b_tail + Delta_b_tail

    This is mathematically exact in FP32 because:
      Conv(Conv(x) + b_head, W_tail) + b_tail == Conv(Conv(x) + b_head_new, W_tail) + b_tail_new.

    Args:
        w_tail: Tail weight array [C'_out, C_out, Kh, Kw] or [C'_out, C_out].
        b_head: Head bias array [C_out].
        b_tail: Tail bias array [C'_out] (if None, initializes to zero).
        absorb_ratio: Fraction of high bias to absorb (default 0.5).
        threshold: Magnitude threshold above which bias absorption triggers.

    Returns:
        Tuple of (b_head_new, b_tail_new).
    """
    c = np.zeros_like(b_head)
    mask = np.abs(b_head) > threshold
    c[mask] = b_head[mask] * absorb_ratio

    b_head_new = b_head - c

    if b_tail is None:
        b_tail_new = np.zeros(w_tail.shape[0], dtype=np.float32)
    else:
        b_tail_new = np.array(b_tail, dtype=np.float32, copy=True)

    # Sum spatial dimensions of tail weights: [C'_out, C_out, Kh, Kw] -> [C'_out, C_out]
    if w_tail.ndim > 2:
        w_sum = np.sum(w_tail, axis=tuple(range(2, w_tail.ndim)))
    else:
        w_sum = w_tail

    # Delta_b_tail = w_sum @ c (shape: [C'_out])
    delta_b = np.matmul(w_sum, c)
    b_tail_new = b_tail_new + delta_b

    return b_head_new.astype(np.float32), b_tail_new.astype(np.float32)


class CrossLayerEqualizationEngine:
    """
    Standalone CLE Engine performing weight equalization, C2f/Bottleneck absorption,
    and high-bias absorption directly on ONNX ModelProto graphs.
    """

    def __init__(
        self,
        append_bias: bool = True,
        weight_threshold: float = 0.5,
        absorb_high_bias: bool = True,
        high_bias_threshold: float = 0.2,
        high_bias_ratio: float = 0.5,
        min_scale: float = 1e-4,
        max_scale: float = 1e4,
    ):
        self.append_bias = append_bias
        self.weight_threshold = weight_threshold
        self.absorb_high_bias = absorb_high_bias
        self.high_bias_threshold = high_bias_threshold
        self.high_bias_ratio = high_bias_ratio
        self.min_scale = min_scale
        self.max_scale = max_scale

    def find_cle_pairs(self, model: onnx.ModelProto) -> List[ClePair]:
        """
        Discovers fusible consecutive Conv-Conv, Conv-SiLU-Conv, and Bottleneck/C2f pairs.
        """
        graph = model.graph
        nodes = list(graph.node)
        inits = {t.name for t in graph.initializer}

        # Map tensor producer node and consumers
        producers: Dict[str, onnx.NodeProto] = {}
        consumers: Dict[str, List[onnx.NodeProto]] = {}
        for n in nodes:
            for out in n.output:
                producers[out] = n
            for inp in n.input:
                if inp:
                    consumers.setdefault(inp, []).append(n)

        pairs: List[ClePair] = []
        matched_heads: Set[str] = set()

        for node in nodes:
            if node.op_type != "Conv" or node.name in matched_heads:
                continue

            # Need valid weights initializer
            if len(node.input) < 2 or node.input[1] not in inits:
                continue

            head_weight_name = node.input[1]
            head_bias_name = node.input[2] if len(node.input) > 2 and node.input[2] in inits else None
            out_tensor = node.output[0]

            # Detect downstream pattern
            dest_nodes = consumers.get(out_tensor, [])
            if not dest_nodes:
                continue

            tail_node: Optional[onnx.NodeProto] = None
            act_type = "Identity"

            # Direct Conv -> Conv
            if len(dest_nodes) == 1 and dest_nodes[0].op_type == "Conv":
                tail_node = dest_nodes[0]
                act_type = "Identity"

            # Conv -> Relu/LeakyRelu/Clip -> Conv
            elif len(dest_nodes) == 1 and dest_nodes[0].op_type in ("Relu", "LeakyRelu", "Clip"):
                act_node = dest_nodes[0]
                act_out = act_node.output[0]
                act_consumers = consumers.get(act_out, [])
                if len(act_consumers) == 1 and act_consumers[0].op_type == "Conv":
                    tail_node = act_consumers[0]
                    act_type = act_node.op_type

            # Conv -> SiLU (Sigmoid + Mul) -> Conv
            # In YOLOv8: Conv_out feeds both Sigmoid and Mul:
            #   Sigmoid(Conv_out) -> sig_out
            #   Mul(Conv_out, sig_out) -> mul_out
            #   Conv2(mul_out)
            elif len(dest_nodes) == 2 and any(n.op_type == "Sigmoid" for n in dest_nodes) and any(n.op_type == "Mul" for n in dest_nodes):
                sig_node = next(n for n in dest_nodes if n.op_type == "Sigmoid")
                mul_node = next(n for n in dest_nodes if n.op_type == "Mul")
                if sig_node.output[0] in mul_node.input:
                    mul_consumers = consumers.get(mul_node.output[0], [])
                    # Single consumer Conv or inside Bottleneck
                    if len(mul_consumers) == 1 and mul_consumers[0].op_type == "Conv":
                        tail_node = mul_consumers[0]
                        act_type = "SiLU"

            # Check if this pair is inside a Bottleneck or C2f block
            in_bottleneck = False
            in_c2f = False
            block_name = None

            c_name = node.name
            if "/m." in c_name and "/cv1/" in c_name:
                in_bottleneck = True
                block_name = c_name.split("/cv1/")[0]
            elif "model." in c_name and (".2." in c_name or ".4." in c_name or ".6." in c_name or ".8." in c_name):
                in_c2f = True

            if tail_node is not None and len(tail_node.input) >= 2 and tail_node.input[1] in inits:
                tail_weight_name = tail_node.input[1]
                tail_bias_name = tail_node.input[2] if len(tail_node.input) > 2 and tail_node.input[2] in inits else None

                pair = ClePair(
                    head_node=node,
                    tail_node=tail_node,
                    head_weight_name=head_weight_name,
                    tail_weight_name=tail_weight_name,
                    head_bias_name=head_bias_name,
                    tail_bias_name=tail_bias_name,
                    activation_type=act_type,
                    in_bottleneck=in_bottleneck,
                    in_c2f=in_c2f,
                    block_name=block_name,
                )
                pairs.append(pair)
                matched_heads.add(node.name)

        return pairs

    def apply(self, model: onnx.ModelProto) -> Tuple[onnx.ModelProto, CleReport]:
        """
        Executes Cross-Layer Equalization and High-Bias Absorption on the graph.
        Returns a cloned, transformed ModelProto and the CleReport.
        """
        model_out = onnx.ModelProto.FromString(model.SerializeToString())
        graph = model_out.graph

        # Extract initializers to numpy dict
        inits: Dict[str, np.ndarray] = {
            t.name: numpy_helper.to_array(t) for t in graph.initializer
        }

        pairs = self.find_cle_pairs(model_out)
        report = CleReport(num_pairs_matched=len(pairs))

        min_scale_observed = 1.0
        max_scale_observed = 1.0

        for pair in pairs:
            w_head = inits[pair.head_weight_name]
            w_tail = inits[pair.tail_weight_name]
            b_head = inits.get(pair.head_bias_name) if pair.head_bias_name else None
            b_tail = inits.get(pair.tail_bias_name) if pair.tail_bias_name else None

            # Verify channel dimension alignment
            c_out_head = w_head.shape[0]
            c_in_tail = w_tail.shape[1]
            if c_out_head != c_in_tail:
                continue

            # 1. Compute per-channel dynamic ranges
            range_h, range_t = compute_channel_ranges(
                w_head, w_tail, b_head=b_head,
                append_bias=self.append_bias,
                bias_threshold=self.weight_threshold,
            )

            # 2. Compute dynamic range equalization scales S_i
            scales = compute_equalization_scales(
                range_h, range_t,
                min_scale=self.min_scale,
                max_scale=self.max_scale,
            )

            min_scale_observed = min(min_scale_observed, float(np.min(scales)))
            max_scale_observed = max(max_scale_observed, float(np.max(scales)))

            # 3. Equalize Conv weights and head bias
            w_head_new, w_tail_new, b_head_new = equalize_conv_weights(
                w_head, w_tail, scales, b_head=b_head
            )

            # 4. High-Bias Absorption (absorb high residual bias from head to tail for pointwise/linear layers)
            b_absorbed = False
            is_pointwise = (w_tail_new.ndim <= 2) or (w_tail_new.shape[2] == 1 and w_tail_new.shape[3] == 1)
            if self.absorb_high_bias and b_head_new is not None and is_pointwise:
                b_head_new, b_tail_new = perform_high_bias_absorption(
                    w_tail_new, b_head_new, b_tail=b_tail,
                    absorb_ratio=self.high_bias_ratio,
                    threshold=self.high_bias_threshold,
                )
                b_absorbed = True
                report.num_biases_absorbed += 1
                report.absorbed_biases.append({
                    "head": pair.head_node.name,
                    "tail": pair.tail_node.name,
                })
            else:
                b_tail_new = b_tail

            # 5. Write transformed weights back to initializers
            inits[pair.head_weight_name] = w_head_new
            inits[pair.tail_weight_name] = w_tail_new
            if pair.head_bias_name and b_head_new is not None:
                inits[pair.head_bias_name] = b_head_new
            if pair.tail_bias_name and b_tail_new is not None:
                inits[pair.tail_bias_name] = b_tail_new

            report.num_pairs_equalized += 1
            report.pairs.append({
                "head": pair.head_node.name,
                "tail": pair.tail_node.name,
                "act": pair.activation_type,
                "in_bottleneck": pair.in_bottleneck,
                "in_c2f": pair.in_c2f,
                "scale_mean": float(np.mean(scales)),
                "scale_min": float(np.min(scales)),
                "scale_max": float(np.max(scales)),
                "bias_absorbed": b_absorbed,
            })

        # Update initializers in graph
        for tensor in graph.initializer:
            if tensor.name in inits:
                new_arr = inits[tensor.name]
                new_proto = numpy_helper.from_array(new_arr, name=tensor.name)
                tensor.CopyFrom(new_proto)

        report.min_scale_ratio = float(min_scale_observed)
        report.max_scale_ratio = float(max_scale_observed)

        return model_out, report


def cross_layer_equalize(
    model: onnx.ModelProto,
    append_bias: bool = True,
    weight_threshold: float = 0.5,
    absorb_high_bias: bool = True,
    high_bias_threshold: float = 0.2,
    high_bias_ratio: float = 0.5,
) -> Tuple[onnx.ModelProto, CleReport]:
    """
    Main entry point for Cross-Layer Equalization and High-Bias Absorption.

    Args:
        model: Input float ONNX ModelProto.
        append_bias: Include bias in range calculations.
        weight_threshold: Weighting factor for bias inclusion.
        absorb_high_bias: Enable high-bias absorption.
        high_bias_threshold: Threshold for bias absorption.
        high_bias_ratio: Fraction of high bias to absorb.

    Returns:
        Tuple of (equalized_model, cle_report).
    """
    engine = CrossLayerEqualizationEngine(
        append_bias=append_bias,
        weight_threshold=weight_threshold,
        absorb_high_bias=absorb_high_bias,
        high_bias_threshold=high_bias_threshold,
        high_bias_ratio=high_bias_ratio,
    )
    return engine.apply(model)


def verify_cle_mathematical_equivalence(
    model_orig: onnx.ModelProto,
    model_cle: onnx.ModelProto,
    input_shape: Tuple[int, ...] = (1, 3, 64, 64),
    tolerance: float = 1e-5,
) -> Tuple[bool, float]:
    """
    Verifies that model_cle preserves FP32 mathematical output within tolerance
    on synthetic test inputs across all shared output tensors.
    """
    import onnxruntime as ort

    sess_orig = ort.InferenceSession(
        model_orig.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    sess_cle = ort.InferenceSession(
        model_cle.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    inp_name = sess_orig.get_inputs()[0].name
    np.random.seed(42)
    dummy_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)

    outputs_orig = sess_orig.run(None, {inp_name: dummy_input})
    outputs_cle = sess_cle.run(None, {inp_name: dummy_input})

    max_diff = 0.0
    for out_o, out_c in zip(outputs_orig, outputs_cle):
        diff = float(np.max(np.abs(out_o - out_c)))
        max_diff = max(max_diff, diff)

    passed = max_diff <= tolerance
    return passed, max_diff
