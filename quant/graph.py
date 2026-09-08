"""Read and fingerprint ONNX graphs before implementing the Phase 1 emitter.

Graph.load enforces the frozen export contract, not EP acceptance. Extra opset
imports are retained: inspected Quark models contain unused custom-domain imports.
Loading sorts nodes in memory before checking; the source file is never written.
Other graph mutations and emission are not implemented in this first slice.
"""
from collections import Counter
import heapq
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


class Graph:
    def __init__(self, model: onnx.ModelProto):
        self.model = model
        self.node_order_changed = False

    @classmethod
    def load(cls, path: Path) -> "Graph":
        graph = cls(onnx.load(path))
        graph._check_contract()
        graph.node_order_changed = graph.topo_sort()
        graph.check()
        return graph

    def _check_contract(self) -> None:
        if self.model.ir_version != 8:
            raise ValueError(f"Expected IR 8, got {self.model.ir_version}")
        default = [o.version for o in self.model.opset_import if o.domain in ("", "ai.onnx")]
        if default != [17]:
            raise ValueError(f"Expected exactly one standard opset import at 17, got {default}")
        initializers = {t.name for t in self.model.graph.initializer}
        inputs = [v for v in self.model.graph.input if v.name not in initializers]
        if not inputs:
            raise ValueError("Expected a static batch-1 tensor input")
        for value in inputs:
            shape = self._shape(value)
            if shape is None or not shape or shape[0] != 1 or any(d <= 0 for d in shape):
                raise ValueError(f"Input {value.name!r} must have a fully static positive shape with batch 1")

    def check(self) -> None:
        self._check_contract()
        onnx.checker.check_model(self.model)

    def topo_sort(self) -> bool:
        """Stable topological order for flat graphs; reject missing edges and cycles.

        Quark's simulation appends Mul/Constant nodes after their consumers in the
        inspected XINT8 files. Repair ordering in memory, retaining every node.
        Subgraph captures need scoped dependency analysis, outside this first slice.
        """
        nodes = list(self.model.graph.node)
        roots = {v.name for v in self.model.graph.input} | {t.name for t in self.model.graph.initializer}
        producers = {}
        for i, node in enumerate(nodes):
            if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for a in node.attribute):
                raise ValueError("Topological sorting of nested graphs is not implemented")
            for output in node.output:
                if not output:
                    continue
                if output in roots or output in producers:
                    raise ValueError(f"Multiple definitions for tensor {output!r}")
                producers[output] = i
        pending, successors = [], [[] for _ in nodes]
        for i, node in enumerate(nodes):
            missing = {name for name in node.input if name and name not in roots and name not in producers}
            if missing:
                raise ValueError(f"Node {node.name!r} has undefined inputs: {sorted(missing)}")
            deps = {producers[name] for name in node.input if name in producers}
            pending.append(len(deps))
            for dep in deps:
                successors[dep].append(i)
        ready = [i for i, count in enumerate(pending) if count == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            i = heapq.heappop(ready)
            order.append(i)
            for child in successors[i]:
                pending[child] -= 1
                if pending[child] == 0:
                    heapq.heappush(ready, child)
        if len(order) != len(nodes):
            raise ValueError("Graph contains a cycle")
        changed = order != list(range(len(nodes)))
        if changed:
            del self.model.graph.node[:]
            self.model.graph.node.extend(nodes[i] for i in order)
        return changed

    def nodes(self) -> list[onnx.NodeProto]:
        """File order, checked as topological by load/check."""
        return list(self.model.graph.node)

    def producer(self, tensor: str) -> onnx.NodeProto | None:
        return next((n for n in self.model.graph.node if tensor in n.output), None)

    def consumers(self, tensor: str) -> list[onnx.NodeProto]:
        return [n for n in self.model.graph.node if tensor in n.input]

    def initializer(self, name: str) -> np.ndarray | None:
        value = next((t for t in self.model.graph.initializer if t.name == name), None)
        return None if value is None else numpy_helper.to_array(value)

    @staticmethod
    def _shape(value: onnx.ValueInfoProto) -> tuple[int, ...] | None:
        if not value.type.HasField("tensor_type") or not value.type.tensor_type.HasField("shape"):
            return None
        dims = value.type.tensor_type.shape.dim
        return tuple(d.dim_value for d in dims) if all(d.HasField("dim_value") for d in dims) else None

    def _value(self, tensor: str) -> onnx.ValueInfoProto | None:
        for values in (self.model.graph.input, self.model.graph.output, self.model.graph.value_info):
            for value in values:
                if value.name == tensor:
                    return value
        return None

    def value_shape(self, tensor: str) -> tuple[int, ...] | None:
        """Stored shape information only; unknown/symbolic dimensions return None."""
        value = self._value(tensor)
        if value is not None:
            return self._shape(value)
        init = next((t for t in self.model.graph.initializer if t.name == tensor), None)
        return None if init is None else tuple(init.dims)

    def value_dtype(self, tensor: str) -> int | None:
        value = self._value(tensor)
        if value is not None and value.type.HasField("tensor_type"):
            return value.type.tensor_type.elem_type or None
        init = next((t for t in self.model.graph.initializer if t.name == tensor), None)
        return None if init is None else init.data_type

    def fingerprint(self) -> dict:
        """JSON-compatible observations, including Constant-fed DPU factors.

        Roles count DQ nodes, not unique parameter sets (some share scale/zp).
        A fingerprint describes a file; it is not a graph-equivalence gate.
        """
        model = self.model
        inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
        constants = dict(inits)
        producers = {o: n for n in model.graph.node for o in n.output}
        consumers = {}
        for node in model.graph.node:
            for slot, name in enumerate(node.input):
                consumers.setdefault(name, []).append((node, slot))
            if node.op_type == "Constant" and node.domain == "":
                for attr in node.attribute:
                    if attr.name == "value":
                        constants[node.output[0]] = numpy_helper.to_array(attr.t)
        roles = {}
        for node in model.graph.node:
            if node.op_type != "DequantizeLinear":
                continue
            if len(node.input) != 3 or any(name not in inits for name in node.input[1:]):
                raise ValueError(f"DQ {node.name!r} needs initializer scale and explicit zero point")
            value = inits.get(node.input[0])
            role = "activation"
            if value is not None:
                slots = {i for n, i in consumers.get(node.output[0], [])
                         if n.op_type in ("Conv", "ConvTranspose", "Gemm", "MatMul")}
                role = "bias" if 2 in slots else "weight" if 1 in slots else "other_initializer"
            scale, zp = (inits[name] for name in node.input[1:])
            if scale.size == 0 or zp.size == 0:
                raise ValueError(f"Empty QDQ parameters at {node.name!r}")
            record = {
                "dtype": str(zp.dtype), "zp_shape": tuple(zp.shape), "zp": np.unique(zp).tolist(),
                "scale_shape": tuple(scale.shape), "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "pow2": bool(np.all(np.isfinite(scale) & (scale > 0)) and np.all(np.frexp(scale)[0] == 0.5)),
                "q_precedes": node.input[0] in producers and producers[node.input[0]].op_type == "QuantizeLinear",
            }
            if value is not None:
                record.update(value_min=int(value.min()), value_max=int(value.max()))
            roles.setdefault(role, []).append(record)
        qdq = {}
        for role, records in sorted(roles.items()):
            qdq[role] = {
                "dq_count": len(records), "dtypes": sorted({r["dtype"] for r in records}),
                "zero_points": sorted({v for r in records for v in r["zp"]}),
                "scale_shapes": [list(s) for s in sorted({r["scale_shape"] for r in records})],
                "zp_shapes": [list(s) for s in sorted({r["zp_shape"] for r in records})],
                "power_of_two_scales": sum(r["pow2"] for r in records),
                "scale_min": min(r["scale_min"] for r in records),
                "scale_max": max(r["scale_max"] for r in records),
                "q_precedes": sum(r["q_precedes"] for r in records),
            }
            if role != "activation":
                qdq[role].update(value_min=min(r["value_min"] for r in records),
                                 value_max=max(r["value_max"] for r in records))
        factors = Counter()
        for node in model.graph.node:
            if node.op_type == "Mul":
                for name in node.input:
                    value = constants.get(name)
                    if value is not None and value.size == 1:
                        factors[str(float(value.item()))] += 1
        hard_sigmoids = Counter()
        for node in model.graph.node:
            if node.op_type == "HardSigmoid":
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                hard_sigmoids[json.dumps(attrs, sort_keys=True)] += 1
        return {
            "ir_version": model.ir_version,
            "opsets": {o.domain: o.version for o in model.opset_import},
            "domains": dict(sorted(Counter(n.domain for n in model.graph.node).items())),
            "producer": [model.producer_name, model.producer_version],
            "metadata_keys": sorted(p.key for p in model.metadata_props),
            "nodes": len(model.graph.node),
            "op_counts": dict(sorted(Counter(n.op_type for n in model.graph.node).items())),
            "inputs": {v.name: list(self._shape(v)) if self._shape(v) is not None else None
                       for v in model.graph.input},
            "qdq": qdq,
            "direct_conv_add_relu": sum(n.op_type == "Relu" and n.input[0] in producers and
                                         producers[n.input[0]].op_type in ("Conv", "Add") for n in model.graph.node),
            "scalar_mul_factors": dict(sorted(factors.items())),
            "hardsigmoid_attributes": dict(sorted(hard_sigmoids.items())),
        }
