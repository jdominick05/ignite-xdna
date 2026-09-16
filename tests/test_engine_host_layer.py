"""Graph-engine host layers and depthwise convolutions, offline (no NPU).

  python tests/test_engine_host_layer.py

Uses models/yolo11n_cut_xint8.onnx, models/yolo11n_no_c2psa_cut_xint8.onnx and models/yolov8n_cut_xint8.onnx
(gitignored: each test is skipped when its model is absent) and onnxruntime. On assets/bus.jpg resized to 640:

- the six depthwise head convolutions lower as dense convolutions with zero off-diagonal taps, and every
  tensor of the ablated model equals ONNX Runtime's uint8 intermediates (graph optimizations off);
- stock YOLO11n with host region /model.10/ (its C2PSA attention block) lowers to 83 engine layers and one
  HostLayer, and every tensor, the host layer's output and the six heads equal ONNX Runtime's;
- the host layer schedules no DMA items and ``plan_segments`` cuts the layers into NPU, host, NPU;
- with the boundary region /model.10/m/m.0/attn/qkv/conv/Conv=/model.10/m/m.0/attn/Reshape_1 only the attention core
  is on the host: C2PSA's seven convolutions lower (91 layers), the pe convolution reads v as qkv blocks 8-15 and
  24-31, and every tensor equals ONNX Runtime's;
- stock YOLO-World v2 with its four text cross-attention blocks (/model.{12,15,18,21}/attn/) on the host lowers to 70
  layers with four HostLayers, alternating NPU and host segments; the text guide those blocks share, built from
  Constant nodes inside /model.12/attn/, is a constant rather than a region boundary, and every tensor equals ONNX
  Runtime's;
- dilated and grouped (non-depthwise) convolutions, 1x1 stride-2 convolutions and a host region that names
  no node are refused;
- ``ignite-compile --host-region`` reaches the graph compile, and a region Git Bash rewrote into a path is refused
  (no model needed).
"""
import dataclasses
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import onnx  # noqa: E402

from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import graph_ir  # noqa: E402
from ignite_xdna.compiler import graph_reference as gr  # noqa: E402

MODELS = ROOT / "models"
STOCK = MODELS / "yolo11n_cut_xint8.onnx"
ABLATED = MODELS / "yolo11n_no_c2psa_cut_xint8.onnx"
YOLOV8N = MODELS / "yolov8n_cut_xint8.onnx"
YOLOW = MODELS / "yolov8s-worldv2_cut_xint8.onnx"
YOLOW_ATTN = ("/model.12/attn/", "/model.15/attn/", "/model.18/attn/", "/model.21/attn/")
BUS = ROOT / "assets" / "bus.jpg"
C2PSA = "/model.10/"
CORE = "/model.10/m/m.0/attn/qkv/conv/Conv=/model.10/m/m.0/attn/Reshape_1"
QKV_Q = "/model.10/m/m.0/attn/qkv/conv/Conv_output_0_QuantizeLinear_Output"


def quantized_bus(ir):
    import cv2
    t = ir.tensors[ir.input]
    img = cv2.resize(cv2.imread(str(BUS)), (t.width, t.height))
    chw = np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, (2, 0, 1))
    return chw, gr.quantize_input(chw, t.scale, t.zero_point)


def depthwise_node_names(model_path):
    names = []
    for n in onnx.load(str(model_path)).graph.node:
        group = [onnx.helper.get_attribute_value(a) for a in n.attribute if a.name == "group"]
        if n.op_type == "Conv" and group and group[0] != 1:
            names.append(n.name)
    return names


class _OrtCase(unittest.TestCase):
    def assert_every_tensor_matches_ort(self, model_path, ir):
        chw, q = quantized_bus(ir)
        direct = gr.run_direct(ir, q)
        ref = gr.ort_intermediates(model_path, chw[None], [L.output for L in ir.layers])
        for L in ir.layers:
            c = ir.tensors[L.output].channels
            self.assertTrue(np.array_equal(direct[L.output][:c], ref[L.output]), f"{L.name} differs from ONNX Runtime")


@unittest.skipUnless(ABLATED.exists(), f"{ABLATED.name} not present")
class DepthwiseLowering(_OrtCase):
    def test_depthwise_heads_are_dense_diagonal_and_match_onnx_runtime(self):
        ir = graph_ir.lower_yolov8n(ABLATED)
        by_name = {L.name: L for L in ir.layers}
        names = depthwise_node_names(ABLATED)
        self.assertEqual(len(names), 6)
        for name in names:
            w = by_name[name].weights
            self.assertEqual(w.shape[0], w.shape[1], name)
            off = w.copy()
            off[np.arange(w.shape[0]), np.arange(w.shape[0])] = 0
            self.assertFalse(off.any(), f"{name}: off-diagonal taps are not zero")
        self.assert_every_tensor_matches_ort(ABLATED, ir)


@unittest.skipUnless(STOCK.exists(), f"{STOCK.name} not present")
class HostLayerLowering(_OrtCase):
    @classmethod
    def setUpClass(cls):
        cls.ir = graph_ir.lower_yolov8n(STOCK, host_regions=(C2PSA,))
        cls.hosts = [L for L in cls.ir.layers if isinstance(L, graph_ir.HostLayer)]

    def test_c2psa_is_one_host_layer(self):
        self.assertEqual(len(self.hosts), 1)
        self.assertEqual(len(self.ir.layers), 84)
        H = self.hosts[0]
        self.assertEqual(H.name, C2PSA)
        self.assertEqual(H.input.tensor, "/model.9/cv2/act/Mul_output_0_QuantizeLinear_Output")
        self.assertEqual(H.output, "/model.10/cv2/act/Mul_output_0_QuantizeLinear_Output")
        self.assertEqual(H.op_types.get("MatMul"), 2)
        self.assertEqual(H.op_types.get("Softmax"), 1)

    def test_every_tensor_matches_onnx_runtime(self):
        self.assert_every_tensor_matches_ort(STOCK, self.ir)

    def test_host_layer_splits_the_schedule(self):
        from ignite_xdna.compiler.engine_compile import plan_segments
        from ignite_xdna.compiler.engine_sequence import program_task_count
        ws = es.plan_workspace(self.ir)
        scheds, _ = es.schedule_graph(self.ir, ws)
        H = self.hosts[0]
        self.assertTrue(all(not prog for prog in scheds[H.index].programs))
        segs = plan_segments(self.ir, scheds)
        self.assertEqual([s["kind"] for s in segs], ["npu", "host", "npu"])
        self.assertEqual(segs[0]["layers"], [0, H.index])
        self.assertEqual(segs[2]["layers"], [H.index + 1, len(self.ir.layers)])
        total = sum(program_task_count(p) for s in scheds for p in s.programs)
        self.assertEqual(segs[0]["tasks"] + segs[2]["tasks"], total)
        self.assertEqual([s["blob"] for s in segs], ["insts_0.bin", "host_0.onnx", "insts_1.bin"])

    def test_region_without_nodes_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no node names start with it"):
            graph_ir.lower_yolov8n(STOCK, host_regions=("/model.99/",))


@unittest.skipUnless(STOCK.exists(), f"{STOCK.name} not present")
class AttentionCoreLowering(_OrtCase):
    @classmethod
    def setUpClass(cls):
        cls.ir = graph_ir.lower_yolov8n(STOCK, host_regions=(CORE,))
        cls.hosts = [L for L in cls.ir.layers if isinstance(L, graph_ir.HostLayer)]
        cls.by_name = {L.name: L for L in cls.ir.layers}

    def test_only_the_attention_core_is_on_the_host(self):
        self.assertEqual(len(self.hosts), 1)
        self.assertEqual(len(self.ir.layers), 91)
        H = self.hosts[0]
        self.assertEqual(H.name, CORE)
        self.assertEqual(H.input.tensor, QKV_Q)
        self.assertEqual(H.output, "/model.10/m/m.0/attn/Reshape_1_output_0_QuantizeLinear_Output")
        self.assertEqual(H.op_types.get("MatMul"), 2)
        self.assertEqual(H.op_types.get("Softmax"), 1)
        self.assertNotIn("Conv", H.op_types)
        self.assertLess(self.by_name["/model.10/m/m.0/attn/qkv/conv/Conv"].index, H.index)
        self.assertGreater(self.by_name["/model.10/cv2/conv/Conv"].index, H.index)

    def test_pe_reads_v_as_two_block_ranges_of_qkv(self):
        pe = self.by_name["/model.10/m/m.0/attn/pe/conv/Conv"]
        self.assertEqual([(s.tensor, s.block_offset, s.blocks) for s in pe.inputs], [(QKV_Q, 8, 8), (QKV_Q, 24, 8)])
        self.assertEqual(pe.residual.tensor, self.hosts[0].output)

    def test_every_tensor_matches_onnx_runtime(self):
        self.assert_every_tensor_matches_ort(STOCK, self.ir)

    def test_segments_stay_npu_host_npu(self):
        from ignite_xdna.compiler.engine_compile import plan_segments
        ws = es.plan_workspace(self.ir)
        scheds, _ = es.schedule_graph(self.ir, ws)
        self.assertEqual([s["kind"] for s in plan_segments(self.ir, scheds)], ["npu", "host", "npu"])

    def test_unknown_boundary_is_refused(self):
        with self.assertRaisesRegex(ValueError, "neither a uint8 tensor nor a node name"):
            graph_ir.lower_yolov8n(STOCK, host_regions=("/model.10/m/m.0/attn/qkv/conv/Conv=/nope",))


@unittest.skipUnless(YOLOW.exists(), f"{YOLOW.name} not present")
class TextAttentionLowering(_OrtCase):
    """YOLO-World v2's four text cross-attention blocks on the host. Their text guide is built once from Constant
    nodes inside /model.12/attn/ and read by the other three blocks: a constant, not a region boundary."""

    @classmethod
    def setUpClass(cls):
        cls.ir = graph_ir.lower_yolov8n(YOLOW, host_regions=YOLOW_ATTN)
        cls.hosts = [L for L in cls.ir.layers if isinstance(L, graph_ir.HostLayer)]

    def test_each_attention_block_is_one_host_layer(self):
        self.assertEqual([H.name for H in self.hosts], list(YOLOW_ATTN))
        self.assertEqual(len(self.ir.layers), 70)
        for H in self.hosts:
            self.assertEqual(H.op_types.get("Einsum"), 1, H.name)

    def test_a_tensor_derived_only_from_constants_is_constant(self):
        G = graph_ir._Graph(onnx.load(str(YOLOW)))
        self.assertTrue(G.is_constant("/model.12/attn/Constant_2_output_0_DequantizeLinear_Output"))
        self.assertFalse(G.is_constant("/model.12/m.0/cv2/act/Mul_output_0_QuantizeLinear_Output"))

    def test_every_tensor_matches_onnx_runtime(self):
        self.assert_every_tensor_matches_ort(YOLOW, self.ir)

    def test_segments_alternate_npu_and_host(self):
        from ignite_xdna.compiler.engine_compile import plan_segments
        ws = es.plan_workspace(self.ir)
        scheds, _ = es.schedule_graph(self.ir, ws)
        kinds = [s["kind"] for s in plan_segments(self.ir, scheds)]
        self.assertEqual(kinds, ["npu", "host"] * 4 + ["npu"])


@unittest.skipUnless(YOLOV8N.exists(), f"{YOLOV8N.name} not present")
class ConvolutionGuard(unittest.TestCase):
    def lowered_with(self, attr_name, value):
        m = onnx.load(str(YOLOV8N))
        conv = next(n for n in m.graph.node if n.op_type == "Conv")
        for a in list(conv.attribute):
            if a.name == attr_name:
                conv.attribute.remove(a)
        conv.attribute.append(onnx.helper.make_attribute(attr_name, value))
        return graph_ir.lower_yolov8n(m)

    def test_dilated_convolution_is_refused(self):
        with self.assertRaisesRegex(ValueError, "dilations"):
            self.lowered_with("dilations", [2, 2])

    def test_grouped_convolution_is_refused(self):
        with self.assertRaisesRegex(ValueError, "only depthwise"):
            self.lowered_with("group", 3)

    def test_pointwise_stride_two_is_refused(self):
        ir = graph_ir.lower_yolov8n(YOLOV8N)
        k1 = next(L for L in ir.layers if isinstance(L, graph_ir.ConvLayer) and L.k == 1)
        bad = dataclasses.replace(k1, stride=2)
        with self.assertRaisesRegex(ValueError, "1x1 stride-2"):
            es.conv_chunk_kind(bad, bad.inputs[0])


class HostRegionArgument(unittest.TestCase):
    def test_command_line_passes_host_regions_to_the_graph_compile(self):
        from ignite_xdna.compiler import cli
        with mock.patch.object(cli, "compile_graph_engine") as compile_graph_engine:
            cli.main(["compile", "--model", "m.onnx", "--output", "m.ignite",
                      "--host-region", C2PSA, "--host-region", "/model.11/"])
        compile_graph_engine.assert_called_once()
        self.assertEqual(compile_graph_engine.call_args.kwargs["host_regions"], [C2PSA, "/model.11/"])

    def test_region_rewritten_into_a_path_is_refused(self):
        from ignite_xdna.compiler import cli
        with self.assertRaisesRegex(ValueError, "MSYS_NO_PATHCONV=1"):
            cli.compile_graph_engine("m.onnx", "m.ignite", host_regions=["C:/Program Files/Git/model.10/"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
