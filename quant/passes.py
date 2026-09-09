"""Graph rewrites transcribed from Quark 0.11rc1's preprocessing and DPU simulation.

simplify is the vendor's SimplifyModel step and, like the vendor, delegates to onnxslim
rather than reimplementing it; it runs first, before CLE sees the node list.
split_to_slice mirrors optimizations/optimize.py ConvertSplitToSlice and runs before
calibration, as the vendor's pre-process does. avgpool_dpu_scale, sigmoid_to_hardsigmoid
and hardsigmoid_dpu_scale mirror postprocess/simulation/simulate_dpu.py and run on the
emitted QDQ graph, as the vendor's post-process does. Names, attribute values and the
Constant payloads follow the source so the emitted graph diffs empty against an oracle.
Shapes and operators without a measured instance raise instead of guessing.
"""
import math

import onnx
from onnx import helper

from .graph import Graph

HARD_SIGMOID_SCALE = (2731.0 / 16384.0) / (1.0 / 6.0)  # quant_utils.py:101
HARD_SIGMOID_ALPHA = 1.0 / 6.0


def _approximately_equal(a: float, b: float, epsilon: float = 1e-6) -> bool:
    """quant_utils.is_approximately_equal."""
    return abs(a - b) < epsilon


def check_hard_sigmoid(node: onnx.NodeProto) -> bool:
    """quant_utils.check_hard_sigmoid_condition: attribute test only, any operator type."""
    has_beta = any(a.name == "beta" for a in node.attribute)
    beta_half = any(a.name == "beta" and _approximately_equal(a.f, 0.5) for a in node.attribute)
    alpha_ok = any(a.name == "alpha" and _approximately_equal(a.f, HARD_SIGMOID_ALPHA) for a in node.attribute)
    return (not has_beta or beta_half) and alpha_ok


def _constant_i64(name: str, value: int) -> onnx.NodeProto:
    return helper.make_node("Constant", [], [name],
                            value=helper.make_tensor(name, onnx.TensorProto.INT64, [1], [value]))


def split_to_slice(g: Graph) -> int:
    """ConvertSplitToSlice (optimize.py:254-361): one Slice per output plus four INT64 Constants.

    The Constant nodes are unnamed and the Slice is named `<output>_<i>`, as in the source.
    Split sizes come from the initializer input or the `split` attribute; a Split without
    sizes or with a third input is rejected rather than silently kept.
    """
    nodes = g.nodes()
    keep, new_nodes, remove_inits, count = [], [], [], 0
    for node in nodes:
        if node.op_type != "Split" or node.domain:
            keep.append(node)
            continue
        axis = next((helper.get_attribute_value(a) for a in node.attribute if a.name == "axis"), None)
        if axis is None:
            raise ValueError(f"Split {node.name!r} has no axis attribute")
        if len(node.input) == 2:
            sizes = g.initializer(node.input[1])
            if sizes is None:
                raise ValueError(f"Split {node.name!r} needs an initializer for its split sizes")
            splits = [int(v) for v in sizes.tolist()]
            remove_inits.append(node.input[1])
        elif len(node.input) == 1:
            splits = next((list(helper.get_attribute_value(a)) for a in node.attribute if a.name == "split"), None)
            if splits is None:
                raise ValueError(f"Split {node.name!r} has no split sizes")
        else:
            raise ValueError(f"Split {node.name!r} has an unsupported input count {len(node.input)}")
        if len(splits) != len(node.output):
            raise ValueError(f"Split {node.name!r}: {len(splits)} sizes for {len(node.output)} outputs")
        starts = [sum(splits[:i]) for i in range(len(splits))]
        ends = [sum(splits[:i + 1]) for i in range(len(splits))]
        for i, output in enumerate(node.output):
            names = [f"{output}_{kind}_{i}" for kind in ("starts", "ends", "axes", "steps")]
            new_nodes.append(helper.make_node("Slice", [node.input[0], *names], [output], f"{output}_{i}"))
            for name, value in zip(names, (starts[i], ends[i], axis, 1), strict=True):
                new_nodes.append(_constant_i64(name, value))
        count += 1
    if count:
        del g.model.graph.node[:]
        g.model.graph.node.extend(keep + new_nodes)
        for name in remove_inits:
            g.remove_initializer(name)
        g.clean_initializers()
        g.topo_sort()
    return count


def _insert_mul(g: Graph, node: onnx.NodeProto, scale: float) -> None:
    """simulate_dpu.py insert_mul: `<out>_Scale` Constant (tensor named "scale") and `<out>_Mul`."""
    output = node.output[0]
    constant = helper.make_node("Constant", [], [output + "_Scale"],
                                value=helper.make_tensor("scale", onnx.TensorProto.FLOAT, [], [scale]))
    mul = helper.make_node("Mul", [output + "_Mul", output + "_Scale"], [output], output + "_Mul")
    g.model.graph.node.extend([constant, mul])
    if not node.name:
        node.name = output
    node.output[0] = output + "_Mul"


def avgpool_scale(kh: int, kw: int) -> float:
    """simulate_dpu.py _get_avgpool_scale: the sourced table, then its dyadic search."""
    if kh > 255 or kw > 255:
        return 1.0
    table = {(3, 3): 9.0 * 7.0 / 64.0, (5, 5): 25.0 * 10.0 / 256.0, (6, 6): 36.0 * 7.0 / 256.0,
             (7, 7): 49.0 * 21.0 / 1024.0, (14, 14): 196.0 * 21.0 / 4096.0}
    if (kh, kw) in table:
        return table[(kh, kw)]
    rec = kw * kh
    powers = [2 ** n for n in range(7 + math.ceil(math.log2(rec)))]
    ks = [round(power / rec) for power in powers]
    diffs = [abs(k / power - 1 / rec) for k, power in zip(ks, powers)]
    n = diffs.index(min(diffs))
    return ks[n] / powers[n] * rec


def avgpool_dpu_scale(g: Graph) -> dict:
    """convert_avg_pool_to_dpu_version, for GlobalAveragePool.

    The source reads the pool's pre-Q/DQ input shape, requires it to be rank 4 and square,
    and otherwise logs a warning and inserts nothing; that skip is reproduced and counted.
    An unknown shape is a shape-inference failure here, not a vendor case, so it raises.
    AveragePool has no instance on any measured graph and raises rather than guess.
    """
    scaled, skipped = 0, []
    for node in g.nodes():
        if node.op_type == "AveragePool":
            raise NotImplementedError("AveragePool DPU simulation has no measured instance")
        if node.op_type != "GlobalAveragePool":
            continue
        shape = g.value_shape(node.input[0])
        if shape is None or len(shape) != 4:
            raise ValueError(f"GlobalAveragePool {node.name} has no rank-4 input shape: {shape}")
        if shape[2] != shape[3]:
            skipped.append(node.name)
            continue
        _insert_mul(g, node, avgpool_scale(int(shape[2]), int(shape[3])))
        scaled += 1
    if scaled:
        g.topo_sort()
    return {"scaled": scaled, "skipped_not_square": skipped}


def simplify(g: Graph) -> dict:
    """SimplifyModel (preproc.py apply_pre_optimization_before_algo), the vendor's first step.

    Quark delegates this to onnxslim and so does Ignition: constant lowering, dead-node
    elimination, common-subexpression elimination and weight tying are onnxslim's, and
    reimplementing them would be transcribing a third-party optimizer rather than the
    vendor's dialect. The import is deferred so the core still imports without onnxslim.
    """
    from onnxslim import slim

    before = (len(g.model.graph.node), len(g.model.graph.initializer))
    names = {n.name for n in g.nodes()}
    g.model.CopyFrom(slim(g.model))
    after = (len(g.model.graph.node), len(g.model.graph.initializer))
    return {"nodes": [before[0], after[0]], "initializers": [before[1], after[1]],
            "nodes_removed": sorted(names - {n.name for n in g.nodes()})}


def sigmoid_to_hardsigmoid(g: Graph) -> int:
    """convert_sigmoid_to_hard_sigmoid: same name, inputs and outputs; alpha = 1/6, no beta."""
    count = 0
    for node in g.nodes():
        if node.op_type != "Sigmoid" or node.domain:
            continue
        replacement = helper.make_node("HardSigmoid", list(node.input), list(node.output), node.name)
        replacement.attribute.append(helper.make_attribute("alpha", HARD_SIGMOID_ALPHA))
        node.CopyFrom(replacement)
        count += 1
    return count


def hardsigmoid_dpu_scale(g: Graph) -> int:
    """convert_hard_sigmoid_to_dpu_version: Mul by HARD_SIGMOID_SCALE after each qualifying node."""
    count = 0
    for node in g.nodes():
        if node.op_type == "HardSigmoid" and not node.domain and check_hard_sigmoid(node):
            _insert_mul(g, node, HARD_SIGMOID_SCALE)
            count += 1
    if count:
        g.topo_sort()
    return count
