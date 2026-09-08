"""Controlled mutations of a known-good ResNet XINT8 graph for EP experiments.

These are diagnostic artifacts, not supported quantizer presets. A placement
result alone never establishes correct numerical execution. No Quark is used.
"""
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper

from .graph import Graph


NOTES = {
    "baseline": "Unmodified owned no-CLE graph.",
    "strip_metadata": "Clear model producer/version/domain/docstring and metadata properties only.",
    "producer_ignition": "Change only producer_name to Ignition, the quantizer's chosen project name.",
    "domain_msft": "Move all Q/DQ nodes to com.microsoft opset 1; parameters unchanged.",
    "act_int8_zp0": "Replace UINT8 zp128 activation zero points with INT8 zp0, same scales; equivalent real grid.",
    "float_act_scales": "Multiply activation scales except final graph-output scale by float32 1.01; weights unchanged.",
    "float_weight_scales": "Multiply Conv/Gemm weight scales by float32 1.01; activations/biases unchanged.",
    "bias_int32_dtype": "Cast stored INT8 biases and their zero points to INT32; same scales and real bias values.",
    "bias_int32_product": "Store INT32 biases at input-scale times weight-scale, preserving original decoded bias values by rounding.",
    "weights_per_channel": "Repeat each scalar weight scale/zp across output channels, same integer weights; equivalent real grid.",
    "drop_gap_mul": "Remove the sourced GAP correction Constant/Mul only.",
}


@dataclass
class MutationReport:
    name: str
    note: str
    changed_initializers: list[dict]
    changed_nodes: list[str]
    nodes_before: int
    nodes_after: int


def mutate(base: Graph, name: str) -> tuple[Graph, MutationReport]:
    if name not in NOTES:
        raise ValueError(f"Unknown mutation {name}")
    model = onnx.ModelProto()
    model.CopyFrom(base.model)
    g = Graph(model)
    before = {t.name: base.initializer(t.name) for t in base.model.graph.initializer}
    nodes_before = {n.name: n.SerializeToString() for n in base.nodes()}
    qnodes = [n for n in g.nodes() if n.op_type == "QuantizeLinear"]
    compute = [n for n in g.nodes() if n.op_type in ("Conv", "Gemm")]

    def parent(node, slot):
        dq = g.producer(node.input[slot])
        if dq is None or dq.op_type != "DequantizeLinear":
            raise ValueError(f"Missing DQ at {node.name} input {slot}")
        return dq

    if name == "strip_metadata":
        for field in ("producer_name", "producer_version", "domain", "doc_string"):
            g.model.ClearField(field)
        del g.model.metadata_props[:]
    elif name == "producer_ignition":
        g.model.producer_name = "Ignition"
    elif name == "domain_msft":
        for node in g.nodes():
            if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
                node.domain = "com.microsoft"
        if not any(o.domain == "com.microsoft" for o in g.model.opset_import):
            g.model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    elif name == "act_int8_zp0":
        for key in {n.input[2] for n in qnodes}:
            zp = g.initializer(key)
            if zp.dtype != np.uint8 or zp.shape != () or zp != 128:
                raise ValueError("Activation probe requires scalar UINT8 zp128")
            g.set_initializer(key, np.array(0, dtype=np.int8))
        # Existing value_info carries UINT8 types from the unmodified graph.
        del g.model.graph.value_info[:]
    elif name in ("float_act_scales", "float_weight_scales"):
        if name == "float_act_scales":
            final_scales = {g.producer(v.name).input[1] for v in g.model.graph.output}
            keys = {n.input[1] for n in qnodes} - final_scales
        else:
            keys = {parent(n, 1).input[1] for n in compute}
        for key in keys:
            scale = g.initializer(key)
            g.set_initializer(key, scale * np.float32(1.01))
    elif name in ("bias_int32_dtype", "bias_int32_product"):
        for node in compute:
            if len(node.input) != 3:
                continue
            bias = parent(node, 2)
            value = g.initializer(bias.input[0]).astype(np.int32)
            old_scale = g.initializer(bias.input[1])
            if name == "bias_int32_product":
                scale = g.initializer(parent(node, 0).input[1]) * g.initializer(parent(node, 1).input[1])
                value64 = np.rint(value.astype(np.float64) * float(old_scale) / float(scale))
                if np.any(value64 < np.iinfo(np.int32).min) or np.any(value64 > np.iinfo(np.int32).max):
                    raise ValueError("INT32 bias overflow")
                value = value64.astype(np.int32)
                g.set_initializer(bias.input[1], scale)
            g.set_initializer(bias.input[0], value)
            g.set_initializer(bias.input[2], np.array(0, dtype=np.int32))
        del g.model.graph.value_info[:]
    elif name == "weights_per_channel":
        for node in compute:
            dq = parent(node, 1)
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            axis = 0 if node.op_type == "Conv" or attrs.get("transB", 0) else 1
            channels = g.initializer(dq.input[0]).shape[axis]
            for key in dq.input[1:]:
                scalar = g.initializer(key)
                if scalar.shape != ():
                    raise ValueError("Expected per-tensor baseline weight parameters")
                g.set_initializer(key, np.repeat(scalar, channels))
            dq.attribute.append(helper.make_attribute("axis", axis))
    elif name == "drop_gap_mul":
        removed = set()
        for pool in [n for n in g.nodes() if n.op_type == "GlobalAveragePool"]:
            following = g.consumers(pool.output[0])
            if len(following) != 1 or following[0].op_type != "Mul":
                raise ValueError("Expected GAP followed by its correction Mul")
            mul = following[0]
            constants = [g.producer(i) for i in mul.input if i != pool.output[0]]
            if len(constants) != 1 or constants[0] is None or constants[0].op_type != "Constant":
                raise ValueError("Expected a Constant correction")
            constant = constants[0]
            if len(g.consumers(constant.output[0])) != 1:
                raise ValueError("Correction Constant has multiple consumers")
            pool.output[0] = mul.output[0]
            removed.update((mul.name, constant.name))
        keep = [n for n in g.nodes() if n.name not in removed]
        del g.model.graph.node[:]
        g.model.graph.node.extend(keep)
        del g.model.graph.value_info[:]
    g.infer_shapes()
    g.check()
    after = {t.name: g.initializer(t.name) for t in g.model.graph.initializer}
    changes = []
    for key in sorted(before.keys() | after.keys()):
        a, b = before.get(key), after.get(key)
        if a is None or b is None or a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            changes.append({"name": key, "before_dtype": str(a.dtype) if a is not None else None,
                            "after_dtype": str(b.dtype) if b is not None else None,
                            "before_shape": list(a.shape) if a is not None else None,
                            "after_shape": list(b.shape) if b is not None else None})
    nodes_after = {n.name: n.SerializeToString() for n in g.nodes()}
    changed_nodes = [key for key in nodes_before.keys() | nodes_after.keys()
                     if nodes_before.get(key) != nodes_after.get(key)]
    return g, MutationReport(name, NOTES[name], changes, sorted(changed_nodes),
                             len(base.nodes()), len(g.nodes()))


def output_difference(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Output comparison requires matching shapes and finite values")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return {"images": len(a), "max_abs": float(np.max(np.abs(diff))),
            "mae": float(np.mean(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff**2))),
            "exact_elements_fraction": float(np.mean(a == b)),
            "argmax_agreement_fraction": float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))}
