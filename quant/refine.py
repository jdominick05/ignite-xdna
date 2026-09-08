"""Position refinement for the folded ResNet operator set.

The rules are sourced from Quark 0.11rc1 QuantPosManager (DESIGN.md §2.3).
All writes update actual initializer data and record changes. This implementation
does not reproduce the vendor's temporary-list raw_data write bug.
"""
from dataclasses import dataclass

import numpy as np

from .graph import Graph
from .pow2 import pos2scale, scale2pos


@dataclass
class RefineReport:
    loops: int
    changes_by_rule: dict[str, int]
    log: list[dict]
    converged: bool


def refine(g: Graph, max_loops: int = 5) -> RefineReport:
    if max_loops < 1:
        raise ValueError("max_loops must be positive")
    producers = {out: n for n in g.nodes() for out in n.output}
    consumers = {}
    for node in g.nodes():
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    allowed = {"Conv", "Gemm", "Add", "Relu", "MaxPool", "GlobalAveragePool",
               "Flatten", "Constant", "Mul", "QuantizeLinear", "DequantizeLinear"}
    if any(n.op_type not in allowed or n.domain for n in g.nodes()):
        raise ValueError("Only the Phase 1 ResNet refinement operators are implemented")
    changes, counts = [], {}

    def iparam(node, slot=0):
        parent = producers.get(node.input[slot])
        return parent.input[1] if parent is not None and parent.op_type == "DequantizeLinear" else None

    def oparam(node):
        followers = consumers.get(node.output[0], [])
        q = next((n for n in followers if n.op_type == "QuantizeLinear"), None)
        if q is not None:
            return q.input[1]
        bridge = next((n for n in followers if
                       (node.op_type in ("Conv", "Add") and n.op_type == "Relu") or
                       (node.op_type == "GlobalAveragePool" and n.op_type == "Mul")), None)
        if bridge is not None:
            q = next((n for n in consumers.get(bridge.output[0], []) if n.op_type == "QuantizeLinear"), None)
            if q is not None:
                return q.input[1]
        return None

    def get(name):
        if name is None:
            raise ValueError("Missing position at a required refinement boundary")
        scale = g.initializer(name)
        if scale is None or scale.shape != () or scale.dtype != np.float32:
            raise ValueError(f"Expected scalar float32 scale {name}")
        pos = scale2pos(scale)
        if pos2scale(pos) != scale:
            raise ValueError(f"Non-power-of-two scale {name}")
        return pos

    def setpos(name, pos, rule, node):
        old = get(name)
        if old != pos:
            g.set_initializer(name, np.asarray(pos2scale(pos)))
            changes.append({"scale": name, "old": old, "new": pos, "rule": rule, "node": node.name})
            counts[rule] = counts.get(rule, 0) + 1

    def clamp(x, lo, hi):
        return min(max(x, lo), hi)

    for loop in range(1, max_loops + 1):
        before = len(changes)
        for node in g.nodes():
            if node.op_type in ("MaxPool", "GlobalAveragePool"):
                i, o = iparam(node), oparam(node)
                pos = min(get(i), get(o))
                setpos(i, pos, "align_pool", node)
                setpos(o, pos, "align_pool", node)
        for node in g.nodes():
            if node.op_type == "Add":
                names = [iparam(node, i) for i in range(len(node.input))]
                values = [get(n) for n in names]
                maximum = int(np.argmax(values))
                if max(values) - min(values) > 7:
                    setpos(names[maximum], min(values) + 7, "shift_read", node)
        for node in g.nodes():
            if node.op_type == "Add":
                values = [get(iparam(node, i)) for i in range(len(node.input))]
                out = oparam(node)
                shift = min(values) - get(out)
                setpos(out, min(values) - clamp(shift, -7, 25), "shift_write", node)
            elif node.op_type == "Mul":
                # The only Mul in this supported graph is the GAP correction,
                # whose Constant operand has no quantization position.
                if not any(producers.get(name) is not None and producers[name].op_type == "GlobalAveragePool"
                           for name in node.input):
                    raise ValueError(f"Unsupported non-GAP Mul in refinement: {node.name}")
        for node in g.nodes():
            if node.op_type in ("Conv", "Gemm"):
                inp, weight, out = iparam(node), iparam(node, 1), oparam(node)
                ip, wp, op = get(inp), get(weight), get(out)
                setpos(weight, clamp(wp + ip - op, 0, 16) + op - ip, "shift_cut", node)
        for node in g.nodes():
            if node.op_type in ("Conv", "Gemm") and len(node.input) == 3:
                ip, wp, op = get(iparam(node)), get(iparam(node, 1)), get(oparam(node))
                bias = iparam(node, 2)
                lower = min(0, wp + ip - op - 16)
                setpos(bias, wp + ip - clamp(wp + ip - get(bias), lower, 15), "shift_bias", node)
        if len(changes) == before:
            return RefineReport(loop, counts, changes, True)
    return RefineReport(max_loops, counts, changes, False)
