"""Structural and numeric comparisons; no hardware session is created here."""
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
import onnx
from onnx import numpy_helper

from .graph import Graph


@dataclass
class GraphDiff:
    node_delta: list[str] = field(default_factory=list)
    init_exact_mismatch: list[str] = field(default_factory=list)
    weight_lsb: dict[str, int] = field(default_factory=dict)
    weight_changed_elements: dict[str, int] = field(default_factory=dict)
    compared_int8_initializers: int = 0

    def ok(self, weight_tol_lsb: int = 1) -> bool:
        return (not self.node_delta and not self.init_exact_mismatch and
                all(delta <= weight_tol_lsb for delta in self.weight_lsb.values()))


def _structure(g: Graph):
    """Hash ordered edges/ports and attributes; ignore node/internal tensor names.

    Initializer names are anchors, consistent with the exact-parameter gate.
    Numeric initializer values are compared separately. This catches a rewire even
    when the multiset of operator types, attributes and input dtypes is unchanged.
    """
    inferred = onnx.shape_inference.infer_shapes(g.model)
    typed = Graph(inferred)
    tensors = {}
    for i, value in enumerate(g.model.graph.input):
        tensors[value.name] = ("input", i, typed.value_dtype(value.name), typed.value_shape(value.name))
    for value in g.model.graph.initializer:
        tensors[value.name] = ("initializer", value.name, value.data_type, tuple(value.dims))
    counts = Counter()
    for node in g.nodes():
        attrs = []
        for attr in sorted(node.attribute, key=lambda a: a.name):
            copy = onnx.AttributeProto()
            copy.CopyFrom(attr)
            if copy.type == onnx.AttributeProto.TENSOR:
                array = numpy_helper.to_array(copy.t)
                copy.t.CopyFrom(numpy_helper.from_array(array, ""))
            attrs.append(copy.SerializeToString().hex())
        signature = (node.domain, node.op_type, attrs, [tensors[name] if name else None for name in node.input],
                     [typed.value_dtype(name) for name in node.output])
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        counts[(node.op_type, digest)] += 1
        for i, name in enumerate(node.output):
            tensors[name] = (digest, i)
    return counts, [tensors[v.name] for v in g.model.graph.output]


def graph_diff(a: Graph, b: Graph) -> GraphDiff:
    a.check()
    b.check()
    result = GraphDiff()
    ac, ao = _structure(a)
    bc, bo = _structure(b)
    for (op, digest), count in (ac - bc).items():
        result.node_delta.append(f"only a: {op} {digest} x{count}")
    for (op, digest), count in (bc - ac).items():
        result.node_delta.append(f"only b: {op} {digest} x{count}")
    if ao != bo:
        result.node_delta.append("graph output connections differ")
    ai = {t.name: numpy_helper.to_array(t) for t in a.model.graph.initializer}
    bi = {t.name: numpy_helper.to_array(t) for t in b.model.graph.initializer}
    for name in sorted(ai.keys() | bi.keys()):
        if name not in ai or name not in bi:
            result.init_exact_mismatch.append(name + ": missing")
            continue
        x, y = ai[name], bi[name]
        if x.dtype != y.dtype or x.shape != y.shape:
            result.init_exact_mismatch.append(name + ": dtype/shape")
        elif x.dtype == np.int8 and name.endswith("_quantized"):
            result.compared_int8_initializers += 1
            difference = np.abs(x.astype(np.int16) - y.astype(np.int16))
            maximum = int(difference.max(initial=0))
            if maximum:
                result.weight_lsb[name] = maximum
                result.weight_changed_elements[name] = int(np.count_nonzero(difference))
        elif x.tobytes() != y.tobytes():
            result.init_exact_mismatch.append(name + ": bytes")
    return result
