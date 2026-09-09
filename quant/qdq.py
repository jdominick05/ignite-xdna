"""QDQ emission for folded ResNet classifiers, head-cut YOLOv8 and MODNet, from float weights and positions.

No topology or integer weights are copied from a quantized model. Unsupported
operator families fail explicitly until their preprocessing/emission gates run.
Tensor marking follows the vendor's operator quantizers: Conv/Gemm mark their first
input, output and initializers; MaxPool and Resize (QDQDirect8BitOp) mark their first
input and share its parameters with their output; every other supported operator marks
all float activation inputs and outputs (QDQOperatorBase). Constant-node outputs and
the empty Resize roi are never marked, as the vendor's type check skips them. Clip marks
its first input and output only: its min and max are parameters the vendor leaves float.
"""
from dataclasses import dataclass, replace
import numpy as np
from onnx import helper

from .graph import Graph
from .pow2 import TensorQ, pos2scale, quantize, scale2pos

RESNET_OPS = {"Conv", "Relu", "Add", "MaxPool", "GlobalAveragePool", "Flatten", "Gemm"}
YOLO_OPS = {"Conv", "Sigmoid", "Mul", "Add", "Concat", "MaxPool", "Resize", "Slice", "Constant"}
MODNET_OPS = {"Conv", "Relu", "Clip", "Add", "Mul", "Concat", "Resize", "GlobalAveragePool", "Sigmoid"}
SUPPORTED_OPS = RESNET_OPS | YOLO_OPS | MODNET_OPS
SHARING_OPS = {"MaxPool", "Resize"}  # QDQDirect8BitOp: output reuses input[0]'s parameters
PARAMETER_OPS = {"Clip": 1}  # inputs from this index on are the vendor's float parameters
# quant_utils.annotate_op_type: a single-consumer output of one of these loses the Q/DQ
# pair before an annotated activation.
ANNOTATE_OPS = ("Conv", "Add", "MaxPool", "AveragePool", "GlobalAveragePool", "MatMul", "Gemm", "ConvTranspose")


def _clip_bound(g: Graph, node, slot: int):
    """get_clip_min_max, restricted to the initializer form the measured graphs use."""
    if len(node.input) <= slot or not node.input[slot]:
        return None
    value = g.initializer(node.input[slot])
    return None if value is None or value.size != 1 else float(value.reshape(-1)[0])


def needs_annotated(g: Graph, node) -> bool:
    """quant_utils.is_node_needs_annotated with the default remove_qdq_op_type."""
    if node.op_type == "Clip":
        bounds = (_clip_bound(g, node, 1), _clip_bound(g, node, 2))
        return bounds in ((0.0, 6.0), (0.0, 1.0))
    return node.op_type in ("Relu", "LeakyRelu", "PRelu")


def quantizable_tensors(g: Graph) -> tuple[list[str], list[str], dict[str, str]]:
    """Activation names, float initializer names, shared-parameter root providers.

    Walks the nodes in the vendor's topological order (Graph.vendor_order, the order
    onnxruntime.quantization's quantize_model visits them) because QDQDirect8BitOp's
    sharing is order-dependent: MaxPool and Resize hand their input's parameters to their
    output only when that input is already marked by the time the node is reached, and
    otherwise mark nothing at all, leaving the output to be marked plainly by whichever
    consumer reaches it. On the ResNet and YOLO exports every such input is produced by an
    earlier quantized node, so the rule is invisible there. MODNet feeds two Resize nodes
    straight from the graph input, which no earlier node has marked, and those two do get
    their own calibrated parameters in the vendor's own artifact.
    """
    acts, weights, sharing = {}, {}, {}
    initializers = {t.name for t in g.model.graph.initializer}
    nodes = g.nodes()
    constants = {out for n in nodes if n.op_type == "Constant" for out in n.output}
    for index in g.vendor_order():
        node = nodes[index]
        if node.domain or node.op_type not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported float operator: {node.domain}:{node.op_type}")
        if node.op_type == "Constant":
            continue
        if node.op_type in ("Conv", "Gemm"):
            if len(node.input) not in (2, 3):
                raise ValueError(f"Invalid inputs at {node.name}")
            acts[node.input[0]] = None
            for name in node.input[1:]:
                value = g.initializer(name)
                if value is None or value.dtype != np.float32:
                    raise ValueError(f"Expected float32 initializer {name}")
                weights[name] = None
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if node.op_type == "Gemm" and attrs.get("beta", 1.0) != 1.0:
                raise ValueError("Non-unit Gemm beta is not yet supported")
        elif node.op_type in PARAMETER_OPS:
            # The vendor quantizes the activation and leaves min/max as float parameters.
            for name in node.input[PARAMETER_OPS[node.op_type]:]:
                if name and name not in initializers and name not in constants:
                    raise ValueError(f"{node.op_type} {node.name} expects initializer parameters")
            acts[node.input[0]] = None
        elif node.op_type in SHARING_OPS:
            if len(node.output) != 1:
                raise ValueError(f"{node.op_type} {node.name} must have one output")
            if node.input[0] in initializers or node.input[0] in constants:
                raise ValueError(f"{node.op_type} {node.name} needs an activation input")
            if node.input[0] not in acts:
                # is_tensor_quantized is false here, so QDQDirect8BitOp.quantize marks
                # neither the input nor the output and returns.
                continue
            sharing[node.output[0]] = node.input[0]
            acts[node.output[0]] = None
            continue
        else:
            for name in node.input:
                if not name or name in constants:
                    continue
                if name in initializers:
                    raise ValueError(f"Initializer {name} feeding {node.op_type} {node.name} has no emission rule")
                acts[name] = None
        for name in node.output:
            acts[name] = None
    for name in list(sharing):
        root = sharing[name]
        while root in sharing:
            root = sharing[root]
        sharing[name] = root
    return list(acts), list(weights), sharing


def read_pos_table(g: Graph) -> dict[str, TensorQ]:
    """Read exact scalar standard-domain XINT8 positions, never rounded guesses."""
    table = {}
    producers = {out: n for n in g.nodes() for out in n.output}
    outputs = {v.name for v in g.model.graph.output}
    for node in g.nodes():
        if node.op_type != "DequantizeLinear":
            continue
        if node.domain or len(node.input) != 3:
            raise ValueError("Expected standard-domain DQ with explicit scale/zp")
        scale, zp = (g.initializer(n) for n in node.input[1:])
        if scale is None or zp is None or scale.shape != () or zp.shape != () or scale.dtype != np.float32:
            raise ValueError(f"Expected scalar float32 scale/zp at {node.name}")
        pos = scale2pos(scale)
        if pos2scale(pos) != scale:
            raise ValueError(f"Not a power-of-two scale: {node.input[1]}")
        value = g.initializer(node.input[0])
        if value is not None:
            if not node.input[0].endswith("_quantized") or value.dtype != np.int8 or zp.dtype != np.int8 or zp != 0:
                raise ValueError(f"Expected INT8 zp0 initializer at {node.name}")
            name = node.input[0].removesuffix("_quantized")
        else:
            parent = producers.get(node.input[0])
            if parent is None or parent.op_type != "QuantizeLinear" or parent.domain:
                raise ValueError(f"Expected activation Q->DQ at {node.name}")
            if list(parent.input[1:]) != list(node.input[1:]) or zp.dtype != np.uint8 or zp != 128:
                raise ValueError(f"Expected matching UINT8 zp128 QDQ at {node.name}")
            name = node.output[0] if node.output[0] in outputs else parent.input[0]
        tq = TensorQ(name, str(zp.dtype), pos, int(zp), "reference")
        if name in table and table[name] != tq:
            raise ValueError(f"Conflicting positions for {name}")
        table[name] = tq
    if not table:
        raise ValueError("No XINT8 positions found")
    return table


def _params(g: Graph, name: str, tq: TensorQ) -> tuple[str, str]:
    scale_name, zp_name = name + "_scale", name + "_zero_point"
    for key, value in ((scale_name, np.asarray(pos2scale(tq.pos))), (zp_name, np.asarray(tq.zp, dtype=tq.dtype))):
        previous = g.initializer(key)
        if previous is not None and (previous.dtype != value.dtype or previous.shape != value.shape or not np.array_equal(previous, value)):
            raise ValueError(f"Conflicting shared parameter {key}")
        g.set_initializer(key, value)
    return scale_name, zp_name


def insert_qdq(g: Graph, tensor: str, tq: TensorQ, provider: str | None = None):
    scale, zp = _params(g, provider or tensor, tq)
    quant_output = tensor + "_QuantizeLinear_Output"
    graph_output = tensor in {v.name for v in g.model.graph.output}
    quant_input = tensor + "_QuantizeLinear_Input" if graph_output else tensor
    dequant_output = tensor if graph_output else tensor + "_DequantizeLinear_Output"
    for node in g.nodes():
        for i, name in enumerate(node.input):
            if name == tensor:
                node.input[i] = dequant_output
        if graph_output:
            for i, name in enumerate(node.output):
                if name == tensor:
                    node.output[i] = quant_input
    q = helper.make_node("QuantizeLinear", [quant_input, scale, zp], [quant_output], tensor + "_QuantizeLinear")
    dq = helper.make_node("DequantizeLinear", [quant_output, scale, zp], [dequant_output], tensor + "_DequantizeLinear")
    g.model.graph.node.extend([q, dq])


def quantize_initializer(g: Graph, name: str, tq: TensorQ) -> None:
    value = g.initializer(name)
    if value is None:
        raise ValueError(f"Missing float initializer {name}")
    scale, zp = _params(g, name, tq)
    g.set_initializer(name + "_quantized", quantize(value, tq.pos, tq.zp, tq.dtype))
    output = name + "_DequantizeLinear_Output"
    for node in g.nodes():
        for i, inp in enumerate(node.input):
            if inp == name:
                node.input[i] = output
    g.model.graph.node.append(helper.make_node("DequantizeLinear", [name + "_quantized", scale, zp], [output], name + "_DequantizeLinear"))
    g.remove_initializer(name)


def prunable_tensors(g: Graph) -> dict[str, str]:
    """get_annotate_tensors: a single-consumer annotate-op output feeding an annotated activation."""
    result = {}
    for node in g.nodes():
        if node.op_type in ANNOTATE_OPS:
            followers = g.consumers(node.output[0])
            if len(followers) == 1 and needs_annotated(g, followers[0]):
                result[node.output[0]] = followers[0].output[0]
    return result


def prune_conv_relu(g: Graph) -> int:
    removed = []
    for node in g.nodes():
        if node.op_type not in ANNOTATE_OPS:
            continue
        qs = g.consumers(node.output[0])
        if len(qs) != 1 or qs[0].op_type != "QuantizeLinear":
            continue
        dqs = g.consumers(qs[0].output[0])
        if len(dqs) != 1 or dqs[0].op_type != "DequantizeLinear":
            continue
        followers = g.consumers(dqs[0].output[0])
        if len(followers) == 1 and needs_annotated(g, followers[0]):
            followers[0].input[0] = node.output[0]
            removed.extend([qs[0].name, dqs[0].name])
    keep = [n for n in g.nodes() if n.name not in removed]
    del g.model.graph.node[:]
    g.model.graph.node.extend(keep)
    g.clean_initializers()
    return len(removed) // 2


@dataclass
class EmitReport:
    activations: int
    initializers: int
    pruned: int


def emit(g: Graph, positions: dict[str, TensorQ]) -> EmitReport:
    acts, weights, sharing = quantizable_tensors(g)
    q = dict(positions)
    for name, successor in prunable_tensors(g).items():
        if name not in q and successor in q:
            # This temporary QDQ is removed immediately; its scale cannot affect
            # the result. Phase 1's reference contains no pre-Relu parameters.
            q[name] = replace(q[successor], name=name, source="temporary_pruned")
    missing = set(acts + weights) - q.keys()
    if missing:
        raise ValueError(f"Missing positions: {sorted(missing)}")
    for names, dtype, zp in ((acts, "uint8", 128), (weights, "int8", 0)):
        for name in names:
            if q[name].dtype != dtype or q[name].zp != zp:
                raise ValueError(f"Invalid dialect for {name}")
    for name in weights:
        quantize_initializer(g, name, q[name])
    for name in acts:
        insert_qdq(g, name, q[name], sharing.get(name))
    pruned = prune_conv_relu(g)
    g.topo_sort()
    return EmitReport(len(acts) - pruned, len(weights), pruned)
