"""Five-row maps and ReLU after a residual add, offline: lowering, workspace, schedule and packet emulation.

A quad is one column of four cores stacked five rows apart, so it covers twenty rows; a map one tile row tall
(5x20, a sensor window rather than an image) used to stop at ``tile_origins``. ``Placement.replicas`` gives such
a tensor four copies of its layout, one per core row, so every fill and drain keeps a positive core-row stride
and core row 0's rows are the tensor. These tests build a synthetic XINT8-style QDQ residual block on a 5x20
input and check that the schedule, replayed through the NumPy core emulator, equals ``graph_reference.run_direct``
at every layer, and that the reference equals ONNX Runtime's CPU provider on the same QDQ graph. Silicon is not
exercised here.
"""
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper as h, numpy_helper

from ignite_xdna.compiler import engine_schedule as es
from ignite_xdna.compiler import graph_ir, graph_reference as gr


def _qdq(nodes, inits, name, scale, zp_dtype=np.uint8, zp=128):
    s, z = f"{name}_s", f"{name}_z"
    inits += [numpy_helper.from_array(np.array(scale, np.float32), s),
              numpy_helper.from_array(np.array(zp, zp_dtype), z)]
    nodes += [h.make_node("QuantizeLinear", [name, s, z], [f"{name}_q"]),
              h.make_node("DequantizeLinear", [f"{name}_q", s, z], [f"{name}_dq"])]
    return f"{name}_dq"


def _conv(nodes, inits, rng, x, name, cin, cout, k, sx):
    sw, sb = 2.0 ** -8, 2.0 ** -6
    w = rng.integers(-40, 41, size=(cout, cin, k, k)).astype(np.int8)
    b = rng.integers(-60, 61, size=(cout,)).astype(np.int8)
    for tag, arr, s in (("w", w, sw), ("b", b, sb)):
        inits += [numpy_helper.from_array(arr, f"{name}_{tag}"),
                  numpy_helper.from_array(np.array(s, np.float32), f"{name}_{tag}_s"),
                  numpy_helper.from_array(np.array(0, np.int8), f"{name}_{tag}_z")]
        nodes.append(h.make_node("DequantizeLinear", [f"{name}_{tag}", f"{name}_{tag}_s", f"{name}_{tag}_z"],
                                 [f"{name}_{tag}_dq"]))
    nodes.append(h.make_node("Conv", [x, f"{name}_w_dq", f"{name}_b_dq"], [name], name=name,
                             kernel_shape=[k, k], pads=[k // 2] * 4, strides=[1, 1]))
    return name


def residual_block_model(height=5, k=3, relu_after_add=False, seed=0) -> onnx.ModelProto:
    """Q -> conv k1-in 3x3 ReLU -> [conv k ReLU -> conv k] + skip (-> ReLU) -> 1x1 -> DQ, 32/64 channels."""
    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    x = _qdq(nodes, inits, "input", 2.0 ** -7)
    c0 = _conv(nodes, inits, rng, x, "c0", 32, 64, 3, 2.0 ** -7)
    nodes.append(h.make_node("Relu", [c0], ["c0_r"]))
    a0 = _qdq(nodes, inits, "c0_r", 2.0 ** -4)
    c1 = _conv(nodes, inits, rng, a0, "c1", 64, 64, k, 2.0 ** -4)
    nodes.append(h.make_node("Relu", [c1], ["c1_r"]))
    a1 = _qdq(nodes, inits, "c1_r", 2.0 ** -4)
    c2 = _conv(nodes, inits, rng, a1, "c2", 64, 64, k, 2.0 ** -4)
    a2 = _qdq(nodes, inits, "c2", 2.0 ** -4)
    nodes.append(h.make_node("Add", [a2, a0], ["add"], name="add"))
    if relu_after_add:
        nodes.append(h.make_node("Relu", ["add"], ["add_r"]))
        s = _qdq(nodes, inits, "add_r", 2.0 ** -3)
    else:
        s = _qdq(nodes, inits, "add", 2.0 ** -3)
    c3 = _conv(nodes, inits, rng, s, "c3", 64, 32, 1, 2.0 ** -3)
    out = _qdq(nodes, inits, "c3", 2.0 ** -4)
    nodes.append(h.make_node("Identity", [out], ["output"]))
    graph = h.make_graph(nodes, "short_map_block",
                         [h.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32, height, 20])],
                         [h.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, height, 20])], inits)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def _without_identity(model: onnx.ModelProto) -> onnx.ModelProto:
    """The graph output is the last DequantizeLinear's output, as a quantizer writes it."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    ident = [n for n in m.graph.node if n.op_type == "Identity"][0]
    src = ident.input[0]
    m.graph.node.remove(ident)
    for n in m.graph.node:
        n.output[:] = ["output" if o == src else o for o in n.output]
    return onnx.shape_inference.infer_shapes(m)


def _emulate(ir, ws, q_in, per_layer_read):
    scheds, store = es.schedule_graph(ir, ws)
    ws_arr = ws.halo_fill()
    ws.write_tensor(ws_arr, ir.input, q_in)
    got = {}
    for s in scheds:
        es.emulate_layer(s, store, ws_arr)
        if per_layer_read:
            L = ir.layers[s.layer_index]
            got[L.output] = ws.read_tensor(ws_arr, L.output)[:ir.tensors[L.output].channels].copy()
    return got


class ShortMap(unittest.TestCase):
    def _check(self, k, relu_after_add):
        model = _without_identity(residual_block_model(k=k, relu_after_add=relu_after_add))
        ir = graph_ir.lower_yolov8n(model)
        ir.task = "segment"
        self.assertEqual(len(ir.layers), 4)
        res = [L for L in ir.layers if L.residual is not None]
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].post_hswish is not None, relu_after_add)
        rng = np.random.default_rng(1)
        x = rng.uniform(-1, 1, size=(32, 5, 20)).astype(np.float32)
        t_in = ir.tensors[ir.input]
        q_in = gr.quantize_input(x, t_in.scale, t_in.zero_point)
        direct = gr.run_direct(ir, q_in)
        for reuse in (False, True):
            ws = es.plan_workspace(ir, reuse=reuse)
            for name, p in ws.placements.items():
                self.assertEqual(p.replicas, es.ROWS, name)
                self.assertEqual(p.band_rows, 0, name)
            got = _emulate(ir, ws, q_in, per_layer_read=True)
            for L in ir.layers:
                np.testing.assert_array_equal(got[L.output], direct[L.output], err_msg=f"{L.name} reuse={reuse}")
        return model, ir, x, direct

    def test_3x3_block_is_exact(self):
        self._check(3, False)

    def test_5x5_block_is_exact(self):
        self._check(5, False)

    def test_relu_after_add_is_exact_and_matches_onnxruntime(self):
        model, ir, x, direct = self._check(3, True)
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime is not installed")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        s = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
        ref = s.run(None, {"input": x[None]})[0][0]
        out = ir.tensors[ir.outputs[0][1]]
        y = (direct[ir.outputs[0][1]].astype(np.float32) - out.zero_point) * np.float32(out.scale)
        np.testing.assert_array_equal(y, ref)

    def test_replicas_do_not_overlap_other_tensors(self):
        model = _without_identity(residual_block_model())
        ir = graph_ir.lower_yolov8n(model)
        ws = es.plan_workspace(ir, reuse=False)
        spans = sorted((p.base, p.base + p.replicas * p.nbytes) for p in ws.placements.values())
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(end, start)

    def test_ten_rows_still_refuse(self):
        model = _without_identity(residual_block_model(height=10))
        ir = graph_ir.lower_yolov8n(model)
        ws = es.plan_workspace(ir, reuse=False)
        with self.assertRaisesRegex(ValueError, "10-pixel map is smaller than one 20-pixel tile"):
            es.schedule_graph(ir, ws)


if __name__ == "__main__":
    unittest.main()
