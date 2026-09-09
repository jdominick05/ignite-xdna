"""Position refinement transcribed from Quark 0.11rc1 QuantPosManager (refine.py).

Rules run in the source order inside each loop: align_concat, align_pool, align_pad,
align_slice, shift_read, shift_write, shift_cut, shift_bias, hard_sigmoid, shift_swish
(DESIGN.md section 2.3). Positions are located the way the source locates them: an input
position is the DequantizeLinear feeding that input slot; an output position is the
QuantizeLinear whose first input is the node output, or the one after a bridging node
(AveragePool/HardSigmoid to Mul, or an annotate-family node to a retained activation).
Writes go to the shared scale initializer, which is what the source's Q/DQ pair update
amounts to; the source's temporary-list raw_data write bug is not reproduced.
The source's Mul shift_write rule does not flag a change and so cannot by itself trigger
another loop; that is reproduced. Operators without a measured instance raise.
"""
from dataclasses import dataclass

import numpy as np

from .graph import Graph
from .passes import check_hard_sigmoid
from .pow2 import pos2scale, scale2pos
from .qdq import needs_annotated

QDQ_OPS = ("QuantizeLinear", "DequantizeLinear")
ANNOTATE_OPS = ("Conv", "Add", "MaxPool", "AveragePool", "GlobalAveragePool", "MatMul", "Gemm", "ConvTranspose")
# Clip/Relu/LeakyRelu/PRelu are quantizers/interface.py:62-70's defaults; qdq.needs_annotated
# holds the rule, because Clip qualifies only at the sourced bounds.
AVG_POOL_OPS = ("AveragePool", "GlobalAveragePool")
ALLOWED = {"Conv", "Gemm", "Add", "Relu", "Clip", "MaxPool", "GlobalAveragePool", "Flatten", "Constant",
           "Mul", "HardSigmoid", "Concat", "Slice", "Resize", *QDQ_OPS}


@dataclass
class RefineReport:
    loops: int
    changes_by_rule: dict[str, int]
    log: list[dict]
    converged: bool


def refine(g: Graph, max_loops: int = 5) -> RefineReport:
    if max_loops < 1:
        raise ValueError("max_loops must be positive")
    nodes = g.nodes()
    if any(n.op_type not in ALLOWED or n.domain for n in nodes):
        raise ValueError("Only the folded ResNet, head-cut YOLO and MODNet refinement operators are implemented")
    producers = {out: n for n in nodes for out in n.output if out}
    changes, counts, flagged = [], {}, [False]

    def dq_scale(tensor):
        """find_node_name: the Q/DQ node whose first output is this tensor."""
        parent = producers.get(tensor)
        return parent.input[1] if parent is not None and parent.op_type in QDQ_OPS else None

    def ipos_by_id(node, slot):
        """get_ipos_name_by_id: no walk-back."""
        return dq_scale(node.input[slot]) if len(node.input) > slot else None

    def ipos(node):
        """get_ipos_name: first input, with the source's one-step walk-back for average pools."""
        if not node.input:
            return None
        name = dq_scale(node.input[0])
        if name is not None:
            return name
        if node.op_type in AVG_POOL_OPS:
            for n in nodes:
                if n.output and n.output[0] == node.input[0]:
                    name = dq_scale(n.input[0])
                    if name is not None:
                        return name
        return None

    def q_scale_after(tensor):
        """find_o_name: first Q/DQ node in graph order whose first input is this tensor."""
        for n in nodes:
            if n.input and n.input[0] == tensor and n.op_type in QDQ_OPS:
                return n.input[1]
        return None

    def connected(pre, n):
        return ((pre in AVG_POOL_OPS + ("HardSigmoid",) and n.op_type == "Mul") or
                (pre in ANNOTATE_OPS and needs_annotated(g, n)))

    def opos(node):
        """get_opos_name, including its running rename of the searched tensor."""
        out = node.output[0]
        name = q_scale_after(out)
        if name is not None:
            return name
        for n in nodes:
            if n.input and n.input[0] == out and connected(node.op_type, n):
                out = n.output[0]
                name = q_scale_after(out)
                if name is not None:
                    return name
        return None

    def wpos(node):
        return dq_scale(node.input[1]) if len(node.input) > 1 else None

    def bpos(node):
        return dq_scale(node.input[2]) if len(node.input) > 2 else None

    def get(name):
        scale = g.initializer(name)
        if scale is None or scale.shape != () or scale.dtype != np.float32:
            raise ValueError(f"Expected scalar float32 scale {name}")
        pos = scale2pos(scale)
        if pos2scale(pos) != scale:
            raise ValueError(f"Non-power-of-two scale {name}")
        return pos

    def setpos(name, pos, rule, node, flag=True):
        old = get(name)
        if old != pos:
            g.set_initializer(name, np.asarray(pos2scale(pos)))
            changes.append({"scale": name, "old": old, "new": pos, "rule": rule, "node": node.name})
            counts[rule] = counts.get(rule, 0) + 1
            if flag:
                flagged[0] = True

    def clamp(x, lo, hi):
        return min(max(x, lo), hi)

    def is_sigmoid_layer(tensor):
        return any(check_hard_sigmoid(n) and n.input and n.input[0] == tensor for n in nodes)

    def align_concat():
        for node in nodes:
            if node.op_type != "Concat":
                continue
            o = opos(node)
            if o is None:
                continue
            value = get(o)
            minimum = value
            names = [ipos_by_id(node, i) for i in range(len(node.input))]
            for name in names:
                if name is not None:
                    minimum = min(get(name), minimum)
            if value != minimum:
                setpos(o, minimum, "align_concat", node)
            for name in names:
                if name is not None and get(name) != minimum:
                    setpos(name, minimum, "align_concat", node)

    def align_inout(op_types, rule):
        for node in nodes:
            if node.op_type not in op_types:
                continue
            i, o = ipos(node), opos(node)
            if i is None or o is None:
                continue
            ip, op = get(i), get(o)
            if op > ip:
                setpos(o, ip, rule, node)
            elif op < ip:
                setpos(i, op, rule, node)

    def input_positions(node):
        names = [ipos_by_id(node, i) for i in range(len(node.input))]
        if any(name is None for name in names):
            return None, None
        return names, [get(name) for name in names]

    def shift_read():
        for node in nodes:
            if node.op_type not in ("Add", "Sub"):
                continue
            names, values = input_positions(node)
            if names is None:
                continue
            id_max, id_min = int(np.argmax(values)), int(np.argmin(values))
            if values[id_max] - values[id_min] > 7:
                setpos(names[id_max], values[id_min] + 7, "shift_read", node)

    def shift_write():
        for node in nodes:
            if node.op_type not in ("Add", "Mul"):
                continue
            names, values = input_positions(node)
            if names is None:
                continue
            o = opos(node)
            if o is None:
                continue
            op = get(o)
            if node.op_type == "Add":
                low = min(values)
                sw = low - op
                if sw > 25 or sw < -7:
                    setpos(o, low - clamp(sw, -7, 25), "shift_write", node)
            else:
                total = sum(values)
                sw = total - op
                if sw > 32 or sw < 0:
                    # The source does not set has_change here (adjust_shift_write, Mul branch).
                    setpos(o, total - clamp(sw, 0, 32), "shift_write_mul", node, flag=False)

    def shift_cut():
        for node in nodes:
            if node.op_type not in ("Conv", "Gemm"):
                continue
            i, o, w = ipos(node), opos(node), wpos(node)
            if i is None or o is None or w is None:
                continue
            ip, op, wp = get(i), get(o), get(w)
            sc = wp + ip - op
            if sc < 0 or sc > 16:
                setpos(w, clamp(sc, 0, 16) + op - ip, "shift_cut", node)

    def shift_bias():
        for node in nodes:
            if node.op_type not in ("Conv", "Gemm"):
                continue
            b = bpos(node)
            if b is None:
                continue
            i, o, w = ipos(node), opos(node), wpos(node)
            if i is None or o is None or w is None:
                continue
            ip, op, wp, bp = get(i), get(o), get(w), get(b)
            shift_cut_value = wp + ip - op
            lower = min(0, -(24 - (8 + shift_cut_value)))
            if any(n.op_type == "LeakyRelu" and n.input and n.input[0] == node.output[0] for n in nodes):
                lower = 0
            sb = wp + ip - bp
            if sb < lower or sb > 15:
                setpos(b, wp + ip - clamp(sb, lower, 15), "shift_bias", node)

    def hard_sigmoid():
        for node in nodes:
            if node.op_type != "HardSigmoid" or not check_hard_sigmoid(node):
                continue
            i, o = ipos(node), opos(node)
            if i is None or o is None:
                continue
            ip, op = get(i), get(o)
            new_ip = clamp(ip, 0, 15)
            new_op = op if op > 7 else 7
            shift_sigmoid = 14 + new_ip - new_op
            new_op = new_op if shift_sigmoid > 0 else 14 + new_ip
            if new_ip != ip:
                setpos(i, new_ip, "hard_sigmoid", node)
            if new_op != op:
                setpos(o, new_op, "hard_sigmoid", node)

    def shift_swish():
        for node in nodes:
            if node.op_type != "Mul" or len(node.input) != 2:
                continue
            if not (is_sigmoid_layer(node.input[0]) or is_sigmoid_layer(node.input[1])):
                continue
            o = opos(node)
            if o is None:
                continue
            op = get(o)
            i0, i1 = ipos_by_id(node, 0), ipos_by_id(node, 1)
            if i0 is None or i1 is None:
                continue
            total = get(i0) + get(i1)
            sh = total - op
            if sh < 0 or sh > 15:
                setpos(o, total - clamp(sh, 0, 15), "shift_swish", node)

    for loop in range(1, max_loops + 1):
        flagged[0] = False
        align_concat()
        align_inout(("MaxPool", "AveragePool", "GlobalAveragePool"), "align_pool")
        align_inout(("Pad",), "align_pad")
        align_inout(("Slice",), "align_slice")
        shift_read()
        shift_write()
        shift_cut()
        shift_bias()
        hard_sigmoid()
        shift_swish()
        if not flagged[0]:
            return RefineReport(loop, counts, changes, True)
    return RefineReport(max_loops, counts, changes, False)
