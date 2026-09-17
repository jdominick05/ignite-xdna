"""Graph transformation passes and pattern matchers for ignite-xdna compiler.

Implements native classification head lowering:
Matches terminal ONNX GlobalAveragePool -> [Mul] -> [QuantizeLinear -> DequantizeLinear]
-> [Flatten] -> [QuantizeLinear -> DequantizeLinear] -> Gemm / MatMul -> QuantizeLinear -> DequantizeLinear
chains and lowers them into persistent core engine ConvLayers (OP_CONV, k=1, stride=1, pad=0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

ZP = 128


@dataclass
class ClassificationHead:
    gemm_node: onnx.NodeProto
    pool_node: Optional[onnx.NodeProto] = None
    flatten_node: Optional[onnx.NodeProto] = None
    mul_node: Optional[onnx.NodeProto] = None
    # Immediate input to Gemm (e.g. from Flatten)
    input_q: str = ""
    in_scale: float = 1.0
    in_zp: int = ZP
    # Pooled input (e.g. from GlobalAveragePool, if different)
    pool_q: str = ""
    pool_scale: float = 1.0
    pool_zp: int = ZP
    cin: int = 0
    cout: int = 0
    weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int8))  # int8 [Cout, Cin]
    weight_scale: float = 1.0
    bias: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))         # int32 [Cout]
    bias_scale: float = 1.0
    output_q: str = ""                   # uint8 output tensor name
    out_scale: float = 1.0
    out_zp: int = ZP
    output_f: str = ""                   # float output name
    consumed_nodes: Set[str] = field(default_factory=set)


def _attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def match_classification_head(G: Any) -> Optional[ClassificationHead]:
    """Inspect graph G and extract a terminal classification head if present.

    Recognizes:
      [GlobalAveragePool] -> [Mul] -> [Q/DQ] -> [Flatten/Reshape] -> [Q/DQ] -> Gemm/MatMul -> Q/DQ -> Graph Output
    or direct Flatten -> Gemm, or Gemm alone.
    """
    # 1. Find terminal Gemm or MatMul node
    gemm_candidates = [n for n in G.g.node if n.op_type in ("Gemm", "MatMul")]
    if not gemm_candidates:
        return None

    # Pick the last one in topological order
    gemm_node = gemm_candidates[-1]
    consumed_nodes: Set[str] = {gemm_node.name}

    # Verify Gemm output is quantized to graph output (or near graph output)
    gemm_out = gemm_node.output[0]
    out_q_node = None
    for c in G.consumers.get(gemm_out, []):
        if c.op_type == "QuantizeLinear":
            out_q_node = c
            break

    if out_q_node is None:
        return None

    consumed_nodes.add(out_q_node.name)
    out_q = out_q_node.output[0]
    out_scale, out_zp = G.scale_zp(out_q_node)

    # DequantizeLinear exposing graph output
    dq_outs = [c for c in G.consumers.get(out_q, []) if c.op_type == "DequantizeLinear"]
    out_f = dq_outs[0].output[0] if dq_outs else gemm_out
    if dq_outs:
        consumed_nodes.add(dq_outs[0].name)

    # Ensure this is actually a terminal output head
    graph_outputs = {o.name for o in G.g.output}
    if not ({out_f, gemm_out, out_q} & graph_outputs):
        return None

    # 2. Extract weights and bias
    if gemm_node.op_type == "Gemm":
        w_input = gemm_node.input[1]
        b_input = gemm_node.input[2] if len(gemm_node.input) > 2 and gemm_node.input[2] else None
        trans_b = int(_attr(gemm_node, "transB", 0))
    else:  # MatMul
        w_input = gemm_node.input[1]
        b_input = None
        trans_b = 0

    # Resolve weights
    w_const = G.const(w_input)
    w_scale = 1.0
    if w_const is None:
        w_src = G.by_output.get(w_input)
        if w_src is not None and w_src.op_type == "DequantizeLinear":
            consumed_nodes.add(w_src.name)
            w_const = G.const(w_src.input[0])
            w_scale = float(G.const(w_src.input[1]).flatten()[0])

    if w_const is None:
        return None

    weights = np.asarray(w_const)
    if weights.ndim != 2:
        return None

    if gemm_node.op_type == "Gemm":
        if trans_b == 0:
            weights = weights.T
    else:  # MatMul
        weights = weights.T

    cout, cin = int(weights.shape[0]), int(weights.shape[1])
    if weights.dtype != np.int8:
        weights = np.clip(np.round(weights.astype(np.float64) / w_scale).astype(np.int64), -128, 127).astype(np.int8)

    # Resolve bias
    bias = np.zeros(cout, dtype=np.int32)
    b_scale = w_scale
    if b_input:
        b_const = G.const(b_input)
        if b_const is None:
            b_src = G.by_output.get(b_input)
            if b_src is not None and b_src.op_type == "DequantizeLinear":
                consumed_nodes.add(b_src.name)
                b_const = G.const(b_src.input[0])
                b_scale = float(G.const(b_src.input[1]).flatten()[0])
        if b_const is not None:
            b_arr = np.asarray(b_const).flatten()
            if b_arr.size == cout:
                bias = b_arr.astype(np.int32)

    # 3. Trace backwards from input 0 of Gemm
    a_input = gemm_node.input[0]
    pool_node = None
    flatten_node = None
    mul_node = None

    # Resolve immediate Gemm input
    try:
        gemm_in_q, gemm_in_scale, gemm_in_zp = G.q_source(a_input)
        if a_input in G.by_output:
            consumed_nodes.add(G.by_output[a_input].name)
    except Exception:
        gemm_in_q, gemm_in_scale, gemm_in_zp = "", 1.0, ZP

    pool_in_q = gemm_in_q
    pool_in_scale = gemm_in_scale
    pool_in_zp = gemm_in_zp

    curr = a_input
    # Trace further backwards through Flatten / Mul / GlobalAveragePool
    while curr:
        prod = G.by_output.get(curr)
        if prod is None:
            break
        consumed_nodes.add(prod.name)
        if prod.op_type in ("Flatten", "Reshape"):
            flatten_node = prod
            curr = prod.input[0]
            try:
                q, s, z = G.q_source(curr)
                if curr in G.by_output:
                    consumed_nodes.add(G.by_output[curr].name)
                pool_in_q, pool_in_scale, pool_in_zp = q, s, z
                curr = q
            except Exception:
                pass
        elif prod.op_type == "Mul":
            mul_node = prod
            curr = prod.input[0]
        elif prod.op_type == "GlobalAveragePool":
            pool_node = prod
            curr = prod.input[0]
            try:
                q, s, z = G.q_source(curr)
                if curr in G.by_output:
                    consumed_nodes.add(G.by_output[curr].name)
                # Input to GAP
                curr = q
            except Exception:
                pass
            break
        elif prod.op_type == "QuantizeLinear":
            curr = prod.input[0]
        elif prod.op_type == "DequantizeLinear":
            curr = prod.input[0]
        else:
            break

    if not gemm_in_q and not pool_in_q:
        return None

    return ClassificationHead(
        gemm_node=gemm_node,
        pool_node=pool_node,
        flatten_node=flatten_node,
        mul_node=mul_node,
        input_q=gemm_in_q,
        in_scale=gemm_in_scale,
        in_zp=gemm_in_zp,
        pool_q=pool_in_q,
        pool_scale=pool_in_scale,
        pool_zp=pool_in_zp,
        cin=cin,
        cout=cout,
        weights=weights,
        weight_scale=w_scale,
        bias=bias,
        bias_scale=b_scale,
        output_q=out_q,
        out_scale=out_scale,
        out_zp=out_zp,
        output_f=out_f,
        consumed_nodes=consumed_nodes,
    )
