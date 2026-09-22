"""A ReLU6 Clip is the engine's ReLU epilogue exactly when its upper bound cannot bind.

ReLU6 networks spell their activation Clip(0, 6). The engine has one integer epilogue, max(q, ZP)
with the store saturating at 255, so a Clip is that same function only when 6.0 lands at or above
the top of the uint8 range - then saturation performs the clamp and nothing is lost. When the
bound really does bind the epilogue cannot express it (its gate is non-decreasing in t), so the
layer must be refused rather than silently losing the clamp. These tests pin both directions and
the ways a bound can fail to be readable.
"""
import numpy as np
import onnx
import pytest
from onnx import helper as h, numpy_helper as nh

from ignite_xdna.compiler import graph_reference as gr
from ignite_xdna.compiler.dense_regions import lower_dense
from ignite_xdna.compiler.graph_ir import ConvLayer, _Graph, clip_bounds, lower_yolov8n

S_IN = 0.125        # input quantum
S_W = 0.25          # weight quantum, so the conv reaches +-11.9 and 6.0 is a bound that fires
COUT, CIN, HW = 19, 3, 32


def clip_model(hi=6.0, lo=0.0, out_scale=0.03125, opset6=False, dynamic_hi=False):
    """Conv -> Clip(lo, hi) -> QuantizeLinear, the shape every ReLU6 export produces.

    ``out_scale`` decides the whole question: 6.0 sits at quantum ``128 + 6/out_scale``, so
    0.03125 puts it at 320 (cannot bind, as in MODNet-Cut) and 0.125 at 176 (binds, as in
    MobileNetV2 and MiDaS-small).
    """
    inits = [nh.from_array(np.array(S_IN, np.float32), 's'),
             nh.from_array(np.array(128, np.uint8), 'z'),
             nh.from_array(np.array(S_W, np.float32), 'sw'),
             nh.from_array(np.array(0, np.int8), 'wz'),
             nh.from_array(np.array(out_scale, np.float32), 'so'),
             nh.from_array(np.ones((COUT, CIN, 1, 1), np.int8), 'w')]
    nodes = [h.make_node('QuantizeLinear', ['x', 's', 'z'], ['xq'], name='ingress'),
             h.make_node('DequantizeLinear', ['xq', 's', 'z'], ['xd']),
             h.make_node('DequantizeLinear', ['w', 'sw', 'wz'], ['wd']),
             h.make_node('Conv', ['xd', 'wd'], ['c'], name='conv', kernel_shape=[1, 1])]
    graph_inputs = [h.make_tensor_value_info('x', 1, [1, CIN, HW, HW])]
    if opset6:
        nodes.append(h.make_node('Clip', ['c'], ['r'], name='act', min=lo, max=hi))
    elif dynamic_hi:
        # A bound that is present but computed: reading it as +inf would drop a real clamp.
        graph_inputs.append(h.make_tensor_value_info('hi_in', 1, []))
        nodes.append(h.make_node('Identity', ['hi_in'], ['hi_dyn'], name='hi_producer'))
        inits.append(nh.from_array(np.array(lo, np.float32), 'lo'))
        nodes.append(h.make_node('Clip', ['c', 'lo', 'hi_dyn'], ['r'], name='act'))
    else:
        inits.append(nh.from_array(np.array(lo, np.float32), 'lo'))
        inits.append(nh.from_array(np.array(hi, np.float32), 'hi'))
        nodes.append(h.make_node('Clip', ['c', 'lo', 'hi'], ['r'], name='act'))
    nodes += [h.make_node('QuantizeLinear', ['r', 'so', 'z'], ['rq']),
              h.make_node('DequantizeLinear', ['rq', 'so', 'z'], ['out'])]
    model = h.make_model(
        h.make_graph(nodes, 'relu6', graph_inputs,
                     [h.make_tensor_value_info('out', 1, [1, COUT, HW, HW])], inits),
        opset_imports=[h.make_opsetid('', 17)])
    model.ir_version = 9
    return model


def lowered_alone(model):
    """The single Conv unit on its own, where a refusal raises instead of becoming a host layer."""
    inferred = onnx.shape_inference.infer_shapes(model)
    return lower_yolov8n(onnx.utils.Extractor(inferred).extract_model(['xq'], ['out']))


def test_non_binding_clip_runs_on_the_engine_and_matches_onnxruntime():
    """The equivalence itself: 6.0 at quantum 320 cannot bind, so ReLU6 IS ReLU here."""
    model = clip_model(out_scale=0.03125)
    ir = lower_dense(model, 'bisenetv2')
    assert any(isinstance(layer, ConvLayer) for layer in ir.layers), \
        "a non-binding Clip must reach the engine, not be carved to the host"
    x = np.random.default_rng(136).normal(scale=6.0, size=(1, CIN, HW, HW)).astype(np.float32)
    # Without this the test would be vacuous: if no convolution output ever exceeded 6.0 the Clip
    # would never fire and the two paths would agree for a reason that has nothing to do with the
    # equivalence being asserted. Weights are int8 ones, so the conv is S_W * sum_c dequant(x).
    xq = np.clip(np.rint(x / S_IN) + 128, 0, 255)
    conv = S_W * ((xq - 128) * S_IN).sum(axis=1)
    assert conv.max() > 6.0, "fixture no longer exercises the ReLU6 bound"

    direct = gr.run_direct(ir, x[0])
    reference = gr.host_session(model.SerializeToString()).run(None, {'x': x})[0]
    np.testing.assert_array_equal(direct['out'][None], reference)
    # And the saturation that stands in for the clamp is actually reached.
    assert reference.max() == pytest.approx(127 * 0.03125)


def test_opset6_clip_attributes_are_read_the_same_way():
    ir = lower_dense(clip_model(out_scale=0.03125, opset6=True), 'bisenetv2')
    assert any(isinstance(layer, ConvLayer) for layer in ir.layers)


def test_binding_clip_is_refused():
    """6.0 at quantum 176 is a real clamp; the epilogue cannot express it, so refuse."""
    with pytest.raises(ValueError, match=r"binds at quantum 176\.0"):
        lowered_alone(clip_model(out_scale=0.125))


def test_clip_with_a_floor_above_zero_is_refused():
    with pytest.raises(ValueError, match="lower bound"):
        lowered_alone(clip_model(lo=1.0, out_scale=0.03125))


def test_a_bound_that_is_not_a_constant_is_refused():
    """The bug this guards: defaulting an unreadable bound to +inf would drop the clamp."""
    model = clip_model(out_scale=0.03125, dynamic_hi=True)
    clip = next(n for n in model.graph.node if n.op_type == 'Clip')
    with pytest.raises(ValueError, match="is not a constant"):
        clip_bounds(_Graph(model), clip)


def test_an_absent_bound_is_genuinely_unbounded():
    """An omitted optional input is +inf, which is a plain ReLU and must still be accepted."""
    model = clip_model(out_scale=0.03125)
    clip = next(n for n in model.graph.node if n.op_type == 'Clip')
    del clip.input[2:]
    lo, hi = clip_bounds(_Graph(model), clip)
    assert (lo, hi) == (0.0, float('inf'))
