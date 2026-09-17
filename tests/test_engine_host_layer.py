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
- with its four C2fAttn output convolutions also on the host (models/yolov8s-worldv2_cut_xint8_fp32cv2.onnx, those
  convolutions quantized in FP32), each of those regions reads the C2fAttn Concat as a list of segments: every tensor
  still equals ONNX Runtime's, the emulated host steps read the same view, and the manifest segments run NhhNhhNhhNhhN;
- with those convolutions split in two exact halves instead (models/yolov8s-worldv2_cut_split_xint8.onnx), each
  Add -> HardSigmoid * Mul lowers onto the attention half's convolution as a residual whose packet applies HardSwish
  after the add: 74 layers, the attention cores the only host layers, every tensor equal to ONNX Runtime's and the
  packet emulation of those four layers equal to the direct reference;
- with those convolutions requantized by GPTQ with int32 biases instead (models/yolov8s-worldv2_cut_xint8_gptqcv2.onnx),
  the biases reach the layers as int32 (no int8 truncation) and every tensor equals ONNX Runtime's; an accumulator
  bias outside int32 is refused; a host step whose text guide is replaced at run time with another class count
  (``HostStep.replace_constants``, no device) gives the same output as the host layer of a rewritten model;
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
# pipelines/yolow/3b_quantize_cut.py --exclude /model.{12,15,18,21}/cv2/ (the four C2fAttn output convs in FP32)
YOLOW_CV2 = MODELS / "yolov8s-worldv2_cut_xint8_fp32cv2.onnx"
# pipelines/yolow/3a_split_attn_conv.py, then 3b_quantize_cut.py --in models/yolov8s-worldv2_cut_split.onnx
YOLOW_SPLIT = MODELS / "yolov8s-worldv2_cut_split_xint8.onnx"
# pipelines/yolow/3c_gptq_cv2.py: those four convolutions with GPTQ int8 weights and int32 biases
YOLOW_GPTQ = MODELS / "yolov8s-worldv2_cut_xint8_gptqcv2.onnx"
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


@unittest.skipUnless(YOLOW_CV2.exists(), f"{YOLOW_CV2.name} not present")
class ConcatViewHostInput(_OrtCase):
    """YOLO-World v2 quantized with its four C2fAttn output convolutions in FP32: those regions read the C2fAttn
    Concat, a view over several physical tensors, so a host layer's input is a list of segments."""

    @classmethod
    def setUpClass(cls):
        cls.regions = tuple(f"/model.{b}/{p}/" for b in (12, 15, 18, 21) for p in ("attn", "cv2"))
        cls.ir = graph_ir.lower_yolov8n(YOLOW_CV2, host_regions=cls.regions)
        cls.hosts = [L for L in cls.ir.layers if isinstance(L, graph_ir.HostLayer)]

    def test_cv2_regions_read_a_concat_view(self):
        self.assertEqual(len(self.ir.layers), 70)
        self.assertEqual([H.name for H in self.hosts], list(self.regions))
        for H in self.hosts:
            view = H.name.endswith("/cv2/")
            self.assertEqual(bool(H.inputs), view, H.name)
            if view:
                self.assertGreater(len(H.inputs), 1)
                self.assertEqual(H.in_channels, 8 * sum(s.blocks for s in H.inputs))

    def test_every_tensor_matches_onnx_runtime(self):
        self.assert_every_tensor_matches_ort(YOLOW_CV2, self.ir)

    def test_emulated_host_steps_read_the_view(self):
        chw, q = quantized_bus(self.ir)
        direct = gr.run_direct(self.ir, q)
        ws = es.plan_workspace(self.ir)
        arr = np.zeros(ws.nbytes, dtype=np.uint8)
        for name, v in direct.items():
            if name in ws.placements:
                ws.write_tensor(arr, name, v)
        for H in self.hosts:
            es.emulate_host_layer(self.ir, ws, H, arr)
            c = self.ir.tensors[H.output].channels
            self.assertTrue(np.array_equal(ws.read_tensor(arr, H.output)[:c], direct[H.output][:c]), H.name)

    def test_manifest_segments_carry_the_view_and_pair_host_steps(self):
        from ignite_xdna.compiler.engine_compile import plan_segments
        ws = es.plan_workspace(self.ir)
        scheds, _ = es.schedule_graph(self.ir, ws)
        segs = plan_segments(self.ir, scheds)
        self.assertEqual("".join("N" if s["kind"] == "npu" else "h" for s in segs), "NhhNhhNhhNhhN")
        for s in segs:
            if s["kind"] == "host":
                self.assertEqual("inputs" in s, s["name"].endswith("/cv2/"), s["name"])


@unittest.skipUnless(YOLOW_SPLIT.exists(), f"{YOLOW_SPLIT.name} not present")
class ResidualHardSwish(_OrtCase):
    """YOLO-World v2 with each C2fAttn output convolution split in two exact halves (pipelines/yolow/3a_split_attn_conv.py,
    then XINT8): Add -> HardSigmoid * Mul after two convolutions lowers as a residual packet that applies HardSwish
    after the add, so only the attention cores stay on the host."""

    @classmethod
    def setUpClass(cls):
        cls.ir = graph_ir.lower_yolov8n(YOLOW_SPLIT, host_regions=YOLOW_ATTN)
        cls.post = [L for L in cls.ir.layers if isinstance(L, graph_ir.ConvLayer) and L.post_hswish is not None]

    def test_each_split_output_is_one_residual_layer_with_hardswish_after_the_add(self):
        self.assertEqual(len(self.ir.layers), 74)
        self.assertEqual([H.name for H in self.ir.layers if isinstance(H, graph_ir.HostLayer)], list(YOLOW_ATTN))
        self.assertEqual([L.name for L in self.post], [f"/model.{b}/cv2/split/attn/Conv" for b in (12, 15, 18, 21)])
        for L in self.post:
            self.assertEqual(L.residual.tensor, L.name.replace("/attn/Conv", "/abc/Conv") + "_output_0_QuantizeLinear_Output")
            self.assertEqual(L.post_hswish.max_error, 0, L.name)
            self.assertTrue(L.output.endswith("/act/Mul_output_0_QuantizeLinear_Output"), L.output)

    def test_every_tensor_matches_onnx_runtime(self):
        self.assert_every_tensor_matches_ort(YOLOW_SPLIT, self.ir)

    def test_packet_emulation_of_the_residual_hardswish_layers_matches_direct(self):
        _, q = quantized_bus(self.ir)
        direct = gr.run_direct(self.ir, q)
        ws = es.plan_workspace(self.ir)
        scheds, store = es.schedule_graph(self.ir, ws)
        arr = ws.halo_fill()
        for name, v in direct.items():
            if name in ws.placements:
                ws.write_tensor(arr, name, v)
        for L in self.post:
            c = self.ir.tensors[L.output].channels
            ws.write_tensor(arr, L.output, np.zeros_like(direct[L.output]))
            es.emulate_layer(scheds[L.index], store, arr)
            self.assertTrue(np.array_equal(ws.read_tensor(arr, L.output)[:c], direct[L.output][:c]), L.name)


@unittest.skipUnless(YOLOW_GPTQ.exists(), f"{YOLOW_GPTQ.name} not present")
class Int32Bias(_OrtCase):
    """YOLO-World v2 with its four C2fAttn output convolutions requantized by pipelines/yolow/3c_gptq_cv2.py: GPTQ int8
    weights and an int32 bias at the product scale. The bias reaches the accumulator unchanged; nothing is truncated to
    int8."""

    @classmethod
    def setUpClass(cls):
        cls.ir = graph_ir.lower_yolov8n(YOLOW_GPTQ, host_regions=YOLOW_ATTN)

    def test_the_four_output_convolutions_carry_int32_biases(self):
        self.assertEqual(len(self.ir.layers), 70)
        by_name = {L.name: L for L in self.ir.layers}
        for b in (12, 15, 18, 21):
            L = by_name[f"/model.{b}/cv2/conv/Conv"]
            self.assertEqual(L.bias_q.dtype, np.int32)
            self.assertGreater(int(np.abs(L.bias_q).max()), 127, L.name)
            self.assertAlmostEqual(L.bias_scale, L.in_scale * L.weight_scale, delta=1e-12)

    def test_every_tensor_matches_onnx_runtime(self):
        self.assert_every_tensor_matches_ort(YOLOW_GPTQ, self.ir)

    def test_host_constants_replaced_at_run_time_match_a_rewritten_model(self):
        """A vocabulary chosen at run time: another class count in a text guide changes only the host step."""
        from onnx import numpy_helper
        from ignite_xdna.runtime.graph_session import HostStep
        H = next(L for L in self.ir.layers if isinstance(L, graph_ir.HostLayer))
        name = f"{H.name}Reshape_output_0"
        guide = np.random.default_rng(5).standard_normal((1, 5, 4, 32)).astype(np.float32)
        step = object.__new__(HostStep)   # the ONNX Runtime part only: no device
        step.name, step.blob = H.name, H.onnx_bytes
        self.assertIn(name, step.initializer_names())
        self.assertEqual(step.replace_constants({name: guide}), [name])
        m = onnx.load_from_string(H.onnx_bytes)
        for t in m.graph.initializer:
            if t.name == name:
                t.CopyFrom(numpy_helper.from_array(guide, name))
        rewritten = dataclasses.replace(H, onnx_bytes=m.SerializeToString())
        t = self.ir.tensors[H.input.tensor]
        x = np.random.default_rng(6).integers(0, 256, (t.channels, t.height, t.width), dtype=np.uint8)
        got = step.ort_session.run(None, {step.input_name: x[None]})[0][0]
        self.assertTrue(np.array_equal(got, gr.run_host_layer(rewritten, x)))
        with self.assertRaisesRegex(ValueError, "rank"):
            step.replace_constants({name: guide[0]})


@unittest.skipUnless(YOLOV8N.exists(), f"{YOLOV8N.name} not present")
class ConvolutionGuard(unittest.TestCase):
    def test_accumulator_bias_outside_int32_is_refused(self):
        ir = graph_ir.lower_yolov8n(YOLOV8N)
        L = next(L for L in ir.layers if isinstance(L, graph_ir.ConvLayer))
        bad = dataclasses.replace(L, bias_q=np.full_like(L.bias_q, np.iinfo(np.int32).max))
        with self.assertRaisesRegex(ValueError, "does not fit int32"):
            es.conv_packet(bad, 0, es.layer_chunks(ir, bad)[0], 1, 0)

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
