"""DPU simulation rewrites needed by the first ResNet re-emission gate."""
import onnx
from onnx import helper
from .graph import Graph


def avgpool_dpu_scale(g: Graph) -> int:
    """Insert the sourced 7x7 GAP factor. Other shapes await their own gate."""
    count = 0
    for node in g.nodes():
        if node.op_type != "GlobalAveragePool":
            continue
        shape = g.value_shape(node.input[0])
        if shape is None or len(shape) != 4 or shape[-2:] != (7, 7):
            raise ValueError(f"Only 7x7 GAP simulation is implemented, got {shape}")
        output = node.output[0]
        node.output[0] = output + "_Mul"
        constant = helper.make_node("Constant", [], [output + "_Scale"],
                                    value=helper.make_tensor("scale", onnx.TensorProto.FLOAT, [], [49.0 * 21.0 / 1024.0]))
        mul = helper.make_node("Mul", [output + "_Mul", output + "_Scale"], [output], output + "_Mul")
        g.model.graph.node.extend([constant, mul])
        count += 1
    g.topo_sort()
    return count
