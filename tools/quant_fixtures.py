"""Clean ONNX graphs built in memory, for checking quant/ passes without a real model.

Why this exists. quant/passes.py and quant/shift_cut.py are pure AST code: they rewrite or
read a graph and never touch the NPU. Everything else in this repo is gated on a logged
hardware run, and rightly so, but these two have no hardware content to gate on -- what
they need is a graph whose expected rewrite is known by construction. A 25 MB quantized
ResNet cannot serve that role, because when a pass misbehaves on it there is no way to say
what the answer should have been.

These fixtures are deliberately tiny and fully specified: every shape, scale and attribute
is written here, so a check can assert an exact node list rather than a count. They are NOT
a substitute for the measured runs -- nothing here says a pass produces the vendor's bytes,
only that it does what its docstring says on a graph whose answer is known.

No hardware, no onnxruntime session, no NPU. Import-only, safe under peer device load.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17


def _t(name, array, dtype=None):
    array = np.asarray(array) if dtype is None else np.asarray(array, dtype=dtype)
    return numpy_helper.from_array(array, name=name)


def _model(nodes, inputs, outputs, initializers, name="fixture"):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    return model


def value(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


# --------------------------------------------------------------------------- passes.py

def split_model(axis=1, splits=(2, 3), channels=5):
    """A single Split with its sizes as an initializer -- split_to_slice's main path."""
    outputs = [value(f"out{i}", [1, s, 4, 4]) for i, s in enumerate(splits)]
    node = helper.make_node("Split", ["x", "split_sizes"], [o.name for o in outputs], "the_split", axis=axis)
    return _model([node], [value("x", [1, channels, 4, 4])], outputs,
                  [_t("split_sizes", list(splits), np.int64)])


def split_model_attribute(axis=1, splits=(2, 3), channels=5):
    """The same Split with `split` as an attribute instead of an input (opset < 13 shape)."""
    outputs = [value(f"out{i}", [1, s, 4, 4]) for i, s in enumerate(splits)]
    node = helper.make_node("Split", ["x"], [o.name for o in outputs], "the_split", axis=axis)
    node.attribute.append(helper.make_attribute("split", list(splits)))
    return _model([node], [value("x", [1, channels, 4, 4])], outputs, [])


def sigmoid_model(count=2):
    """`count` chained Sigmoids -- sigmoid_to_hardsigmoid and the DPU rescale after it."""
    nodes, prev = [], "x"
    for i in range(count):
        nodes.append(helper.make_node("Sigmoid", [prev], [f"s{i}"], f"sig{i}"))
        prev = f"s{i}"
    return _model(nodes, [value("x", [1, 3, 4, 4])], [value(prev, [1, 3, 4, 4])], [])


def hardsigmoid_model(alpha=1.0 / 6.0, beta=None):
    """One HardSigmoid with controllable alpha/beta, for check_hard_sigmoid's attribute test."""
    node = helper.make_node("HardSigmoid", ["x"], ["y"], "hs", alpha=alpha)
    if beta is not None:
        node.attribute.append(helper.make_attribute("beta", beta))
    return _model([node], [value("x", [1, 3, 4, 4])], [value("y", [1, 3, 4, 4])], [])


def global_avgpool_model(h=7, w=7):
    """GlobalAveragePool with a declared rank-4 input shape, for avgpool_dpu_scale."""
    node = helper.make_node("GlobalAveragePool", ["x"], ["y"], "gap")
    return _model([node], [value("x", [1, 8, h, w])], [value("y", [1, 8, 1, 1])], [])


# ------------------------------------------------------------------------ shift_cut.py

def _qdq(prefix, source, scale, zero_point, dtype):
    """Q->DQ pair on `source`; returns (nodes, initializers, dequantized_tensor_name)."""
    inits = [_t(f"{prefix}_scale", scale, np.float32),
             _t(f"{prefix}_zp", zero_point, np.uint8 if dtype == TensorProto.UINT8 else np.int8)]
    nodes = [helper.make_node("QuantizeLinear", [source, f"{prefix}_scale", f"{prefix}_zp"],
                              [f"{prefix}_q"], f"{prefix}_Q"),
             helper.make_node("DequantizeLinear", [f"{prefix}_q", f"{prefix}_scale", f"{prefix}_zp"],
                              [f"{prefix}_dq"], f"{prefix}_DQ")]
    return nodes, inits, f"{prefix}_dq"


def _dq_initializer(prefix, array, scale, zero_point):
    """DQ of a stored integer initializer, the shape a quantized weight or bias takes."""
    inits = [_t(f"{prefix}_q", array, np.int8),
             _t(f"{prefix}_scale", scale, np.float32),
             _t(f"{prefix}_zp", zero_point, np.int8)]
    node = helper.make_node("DequantizeLinear", [f"{prefix}_q", f"{prefix}_scale", f"{prefix}_zp"],
                            [f"{prefix}_dq"], f"{prefix}_DQ")
    return [node], inits, f"{prefix}_dq"


def qdq_conv_model(pos_x=8, pos_w=8, pos_y=2, channels=2, ksize=1, with_relu=False):
    """A single QDQ Conv whose three positions -- and therefore whose sigma -- are chosen.

    sigma = pos_x + pos_w - pos_y + 14, so a caller can place an operation anywhere on the
    shift-cut axis and check the analyzer reads back the sigma it asked for. Scales are
    exact powers of two, as the XINT8 dialect's always are.
    """
    sx, sw, sy = 2.0 ** -pos_x, 2.0 ** -pos_w, 2.0 ** -pos_y
    nodes, inits, x = _qdq("x", "input", sx, 128, TensorProto.UINT8)
    w_nodes, w_inits, w = _dq_initializer("w", np.ones((channels, channels, ksize, ksize), np.int8), sw, 0)
    nodes += w_nodes
    inits += w_inits
    conv_out = "conv_out"
    nodes.append(helper.make_node("Conv", [x, w], [conv_out], "the_conv",
                                  kernel_shape=[ksize, ksize], pads=[0, 0, 0, 0], strides=[1, 1]))
    tail = conv_out
    if with_relu:
        nodes.append(helper.make_node("Relu", [conv_out], ["relu_out"], "the_relu"))
        tail = "relu_out"
    y_nodes, y_inits, y = _qdq("y", tail, sy, 128, TensorProto.UINT8)
    nodes += y_nodes
    inits += y_inits
    return _model(nodes, [value("input", [1, channels, 4, 4])], [value(y, [1, channels, 4, 4])], inits)


def unresolved_mul_model():
    """The DPU-simulation shape passes.py::_insert_mul writes: Constant * HardSigmoid.

    Neither Mul input comes through a DequantizeLinear, so no scale can be read. This is
    the graph that was scored on two invented 1.0 defaults and counted as a passing
    operation before 2026-09-09; a checker asserts it is UNRESOLVED instead.
    """
    nodes = [helper.make_node("HardSigmoid", ["input"], ["hs_out_Mul"], "hs", alpha=1.0 / 6.0),
             helper.make_node("Constant", [], ["hs_out_Scale"], "scale_const",
                              value=helper.make_tensor("scale", TensorProto.FLOAT, [], [0.1666])),
             helper.make_node("Mul", ["hs_out_Mul", "hs_out_Scale"], ["mul_out"], "hs_out_Mul_node")]
    q_nodes, q_inits, y = _qdq("y", "mul_out", 2.0 ** -7, 128, TensorProto.UINT8)
    return _model(nodes + q_nodes, [value("input", [1, 3, 4, 4])], [value(y, [1, 3, 4, 4])], q_inits)


def branch_add_model(pos_a=4, pos_b=9):
    """Two QDQ branches meeting at an Add, with a chosen position spread between them."""
    nodes_a, inits_a, a = _qdq("a", "input", 2.0 ** -pos_a, 128, TensorProto.UINT8)
    nodes_b, inits_b, b = _qdq("b", "input", 2.0 ** -pos_b, 128, TensorProto.UINT8)
    add = helper.make_node("Add", [a, b], ["add_out"], "the_add")
    y_nodes, y_inits, y = _qdq("y", "add_out", 2.0 ** -5, 128, TensorProto.UINT8)
    return _model(nodes_a + nodes_b + [add] + y_nodes, [value("input", [1, 3, 4, 4])],
                  [value(y, [1, 3, 4, 4])], inits_a + inits_b + y_inits)


# ------------------------------------------------------------------------- adaround.py

def adaround_pair(channels=2, size=4, pos_x=7, pos_w=8, pos_y=5, seed=0):
    """A matched (float, quantized) pair the AdaRound transcription accepts.

    layer_targets requires the quantized Conv to carry the float Conv's name, its input to
    arrive through Q->DQ, and its weight and bias through a DQ of an initializer. The float
    weights are deliberately NOT on the quantization grid, so AdaRound has a rounding
    decision to make and the emitted integer weights actually depend on the optimisation.
    """
    rng = np.random.default_rng(seed)
    sx, sw, sy = 2.0 ** -pos_x, 2.0 ** -pos_w, 2.0 ** -pos_y
    weight = rng.normal(0.0, sw * 40, (channels, channels, 3, 3)).astype(np.float32)
    bias = rng.normal(0.0, 0.05, (channels,)).astype(np.float32)
    sb = 2.0 ** -12

    float_nodes = [helper.make_node("Conv", ["input", "W", "B"], ["conv_out"], "the_conv",
                                    kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
                   helper.make_node("Relu", ["conv_out"], ["relu_out"], "the_relu")]
    float_model = _model(float_nodes, [value("input", [1, channels, size, size])],
                         [value("relu_out", [1, channels, size, size])],
                         [_t("W", weight), _t("B", bias)])

    w_q = np.clip(np.round(weight / sw), -127, 127).astype(np.int8)
    b_q = np.clip(np.round(bias / sb), -127, 127).astype(np.int8)
    nodes, inits, x = _qdq("x", "input", sx, 128, TensorProto.UINT8)
    inits += [_t("W_quantized", w_q), _t("W_scale", sw, np.float32), _t("W_zp", 0, np.int8),
              _t("B_quantized", b_q), _t("B_scale", sb, np.float32), _t("B_zp", 0, np.int8)]
    nodes += [helper.make_node("DequantizeLinear", ["W_quantized", "W_scale", "W_zp"], ["W_dq"], "W_DQ"),
              helper.make_node("DequantizeLinear", ["B_quantized", "B_scale", "B_zp"], ["B_dq"], "B_DQ"),
              helper.make_node("Conv", [x, "W_dq", "B_dq"], ["conv_out"], "the_conv",
                               kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
              helper.make_node("Relu", ["conv_out"], ["relu_out"], "the_relu")]
    y_nodes, y_inits, y = _qdq("y", "relu_out", sy, 128, TensorProto.UINT8)
    quant_model = _model(nodes + y_nodes, [value("input", [1, channels, size, size])],
                         [value(y, [1, channels, size, size])], inits + y_inits)
    return float_model, quant_model


def images(count, channels=2, size=4, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, 1.0, (1, channels, size, size)).astype(np.float32) for _ in range(count)]
