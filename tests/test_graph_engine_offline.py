"""Offline invariants of the graph engine compiler (no device, any environment).

Covers the lowering of models/yolov8n_cut_xint8.onnx, the workspace plan, the
tile schedule, the packet formats, the instruction-stream splitter arithmetic
and the head_layout contract the runtime resolves.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.compiler import engine_emulator as em  # noqa: E402
from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import graph_ir  # noqa: E402
from ignite_xdna.compiler.engine_compile import (  # noqa: E402
    build_manifest, check_kernel_covers_packets)
from ignite_xdna.compiler.engine_sequence import merge_quad, program_task_count  # noqa: E402
from ignite_xdna.runtime.heads import resolve_head_layout  # noqa: E402

MODEL = ROOT / "models" / "yolov8n_cut_xint8.onnx"


@unittest.skipUnless(MODEL.exists(), "quantized model not present")
class GraphEngineOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ir = graph_ir.lower_yolov8n(MODEL)
        cls.ws = es.plan_workspace(cls.ir)
        cls.scheds, cls.store = es.schedule_graph(cls.ir, cls.ws)

    def test_01_lowering_covers_the_whole_network(self):
        convs = [L for L in self.ir.layers if isinstance(L, graph_ir.ConvLayer)]
        pools = [L for L in self.ir.layers if isinstance(L, graph_ir.PoolLayer)]
        self.assertEqual((len(convs), len(pools)), (63, 3))
        self.assertEqual(len(self.ir.outputs), 6)
        self.assertEqual(sum(1 for L in convs if L.act == "hswish"), 57)
        self.assertEqual(sum(1 for L in convs if L.residual is not None), 6)
        self.assertEqual(sum(1 for L in convs if any(s.up2 for s in L.inputs)), 2)

    def test_02_hardswish_tables_exact_and_scales_power_of_two(self):
        for L in self.ir.layers:
            if isinstance(L, graph_ir.ConvLayer):
                if L.hswish is not None:
                    self.assertEqual(L.hswish.max_error, 0, L.name)
                self.assertGreaterEqual(L.shift_out, 0, L.name)
                bias = L.bias_acc()
                self.assertTrue(np.all(np.abs(bias) < 2 ** 31), L.name)

    def test_03_workspace_placements_disjoint_and_aligned(self):
        ps = list(self.ws.placements.values())
        first_use = {self.ir.input: 0}
        last_use = {self.ir.input: 0}
        for idx, L in enumerate(self.ir.layers):
            first_use[L.output] = min(first_use.get(L.output, idx), idx)
            last_use[L.output] = max(last_use.get(L.output, idx), idx)
            in_tensors = []
            if isinstance(L, graph_ir.ConvLayer):
                in_tensors.extend(s.tensor for s in L.inputs)
                if L.residual:
                    in_tensors.append(L.residual.tensor)
            elif isinstance(L, graph_ir.PoolLayer):
                in_tensors.append(L.input.tensor)
            elif isinstance(L, graph_ir.HostLayer):
                in_tensors.extend(s.tensor for s in L.input_segments())
            for t in in_tensors:
                last_use[t] = max(last_use.get(t, 0), idx)

        for i, a in enumerate(ps):
            for b in ps[i + 1:]:
                a_range = (a.base, a.base + a.nbytes)
                b_range = (b.base, b.base + b.nbytes)
                spatial_overlap = max(a_range[0], b_range[0]) < min(a_range[1], b_range[1])
                if spatial_overlap:
                    a_life = (first_use[a.name], last_use[a.name])
                    b_life = (first_use[b.name], last_use[b.name])
                    temporal_overlap = max(a_life[0], b_life[0]) <= min(a_life[1], b_life[1])
                    self.assertFalse(temporal_overlap, f"{a.name} and {b.name} overlap in both space and time!")
        for p in ps:
            self.assertEqual(p.base % 64, 0)
            self.assertIn(p.halo_value, (0, 128))
        self.assertLess(self.ws.nbytes, 16 * 1024 * 1024)

    def test_04_every_transfer_moves_whole_packets_within_the_repeat_limit(self):
        from ignite_xdna.compiler.engine_sequence import MAX_REPEAT
        for s in self.scheds:
            for prog in s.programs:
                for it in prog:
                    pats = it[1] if it[0] == "a" else [it[1]] if it[0] in ("A", "o", "W") else []
                    for p in pats:
                        self.assertEqual(p.offset % 4, 0)
                        self.assertLessEqual(len(p.sizes), 4)
                        if len(p.sizes) == 4:
                            self.assertLessEqual(p.sizes[0], MAX_REPEAT, s.name)
                    if it[0] in ("a", "A"):
                        for p in pats:
                            self.assertEqual(p.nbytes % em.A_BYTES, 0, s.name)
                    elif it[0] == "o":
                        self.assertEqual(it[1].nbytes % (4 * em.O_BYTES), 0, s.name)
                    elif it[0] == "w":
                        self.assertEqual(it[2] % em.W_BYTES, 0)
                        self.assertGreater(it[3], 0)
                    elif it[0] == "W":
                        self.assertEqual(it[1].nbytes % em.W_BYTES, 0)
                        self.assertGreater(it[2], 0)

    def test_05_weight_objects_match_activation_objects(self):
        """Every column program delivers exactly as many weight-object uses as its fills deliver objects,
        and every held weight task or drain is released by exactly the activation items that follow it."""
        blob = self.store.blob()
        for s in self.scheds:
            for prog in s.programs:
                a_items = sum(1 for it in prog if it[0] in ("a", "A"))
                w_served = sum(it[3] if it[0] == "w" else it[2] for it in prog if it[0] in ("w", "W"))
                self.assertEqual(w_served, a_items, s.name)
                o_held = sum(it[2] for it in prog if it[0] == "o" and len(it) > 2)
                if o_held:
                    self.assertEqual(o_held, a_items, s.name)
                a_objects = sum(sum(p.nbytes for p in (it[1] if it[0] == "a" else [it[1]]))
                                for it in prog if it[0] in ("a", "A")) // (4 * em.A_BYTES)
                counts = 0
                for it in prog:
                    if it[0] == "w":
                        data = blob[it[1]:it[1] + it[2]]
                    elif it[0] == "W":
                        data = it[1].read(blob)
                    else:
                        continue
                    for k in range(0, data.size, em.W_BYTES):
                        hdr = em.unpack_w_packet(data[k:k + em.W_BYTES])[0]
                        counts += hdr.count_out + hdr.count_acc
                self.assertEqual(counts, a_objects, s.name)

    def test_06_schedule_size(self):
        rounds = sum(s.rounds for s in self.scheds)
        tasks = sum(program_task_count(p) for s in self.scheds for p in s.programs)
        self.assertEqual(rounds, 1415)
        self.assertLess(tasks, 4000)
        self.assertLess(self.store.nbytes, 12 * 1024 * 1024)

    def test_07_merge_quad_requires_regular_spacing(self):
        from ignite_xdna.compiler.engine_sequence import DmaPattern
        pats = [DmaPattern("ws", 1000 + 400 * i, (8, 5, 160), (2000, 400, 1)) for i in range(4)]
        m = merge_quad(pats)
        # The block stride is five 400-byte rows, so blocks and rows fold into 40 rows.
        self.assertEqual((m.sizes, m.strides), ((4, 40, 160), (400, 400, 1)))
        self.assertTrue(np.array_equal(np.concatenate([p.indices() for p in pats]), m.indices()))
        pats[3] = DmaPattern("ws", 1000 + 400 * 5, (8, 5, 160), (2000, 400, 1))
        self.assertIsNone(merge_quad(pats))

    def test_07b_canonical_folding_keeps_every_byte_in_order(self):
        from ignite_xdna.compiler.engine_sequence import DmaPattern, canonical, merge_runs
        # A 20-column tensor without a halo: a 5-row tile at a 160-byte pitch is one 800-byte run.
        p = DmaPattern("ws", 4096, (8, 5, 160), (3200, 160, 1))
        c = canonical(p)
        self.assertEqual((c.sizes, c.strides), ((8, 800), (3200, 1)))
        self.assertTrue(np.array_equal(p.indices(), c.indices()))
        # A 42-column pitch does not fold: the tile reads 25 of 42 columns.
        q = DmaPattern("ws", 4096, (4, 8, 200), (2816, 336, 1))
        self.assertEqual(canonical(q), q)
        # Unit dimensions drop out.
        u = DmaPattern("ws", 64, (1, 1, 6400), (0, 0, 1))
        self.assertEqual((canonical(u).sizes, canonical(u).strides), ((6400,), (1,)))
        # Folded chunk quads merge across chunks into one repeat pattern that reads the same bytes.
        quads = [DmaPattern("ws", 8000 + k * 8 * 3200, (4, 8, 5, 160), (800, 3200, 160, 1)) for k in range(3)]
        merged = merge_runs(quads)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].sizes, (3, 4, 8, 800))
        self.assertTrue(np.array_equal(np.concatenate([x.indices() for x in quads]), merged[0].indices()))

    def test_08_manifest_head_layout_resolves(self):
        manifest = build_manifest(self.ir, self.ws, self.scheds, self.store, "yolov8n", 0, "x", "y", 0.0)
        status = resolve_head_layout(manifest, int(manifest["egress_bytes"]))
        self.assertTrue(status.present, status.reason)
        self.assertEqual(manifest["egress_bytes"], 1_209_600)
        self.assertEqual(set(manifest["head_layout"]), {"p3_box", "p4_box", "p5_box", "p3_cls", "p4_cls", "p5_cls"})
        self.assertEqual(manifest["output_shapes"]["p3_cls"], [1, 80, 80, 80])

    def test_09_first_layers_packet_emulation_matches_direct_reference(self):
        from ignite_xdna.compiler import graph_reference as gr
        rng = np.random.default_rng(7)
        q_in = rng.integers(128, 256, size=(3, 640, 640), dtype=np.uint8)
        direct = gr.run_direct(self.ir, q_in, stop_after=1)
        ws_arr = self.ws.halo_fill()
        self.ws.write_tensor(ws_arr, self.ir.input, q_in)
        for s in self.scheds[:2]:
            es.emulate_layer(s, self.store, ws_arr)
            L = self.ir.layers[s.layer_index]
            got = self.ws.read_tensor(ws_arr, L.output)[:self.ir.tensors[L.output].channels]
            self.assertTrue(np.array_equal(got, direct[L.output]), L.name)

    def test_10_direct_ingress_plane_matches_the_int8_staging_path(self):
        """The native plane ingress is byte-identical to quantizing the int8 preprocessor output."""
        import cv2
        from ignite_xdna.pipelines import preprocess as pp
        from ignite_xdna.runtime.graph_session import input_lut
        pre = pp.FusedPreprocessor(640)
        if not pre.has_plane_ingress:
            self.skipTest("native preprocessor without plane ingress")
        lut = input_lut(self.ir.tensors[self.ir.input].scale, 128)
        p = self.ws.placements[self.ir.input]
        h = p.halo
        rng = np.random.default_rng(3)
        frames = [rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8),
                  rng.integers(0, 256, (1080, 607, 3), dtype=np.uint8),
                  rng.integers(0, 256, (640, 640, 3), dtype=np.uint8),
                  rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)]  # identity resize fast path
        bus = ROOT / "assets" / "bus.jpg"
        if bus.exists():
            frames.append(cv2.imread(str(bus)))
        for img in frames:
            q, pad, scale = pre.preprocess(img)
            ref = np.full((p.height + 2 * h, p.width + 2 * h, 8), 128, dtype=np.uint8)
            ref[h:h + p.height, h:h + p.width, :3] = np.moveaxis(lut[q[0].view(np.uint8) ^ 0x80], 0, -1)
            plane = np.full_like(ref, 128)
            pad2, scale2 = pre.preprocess_to_plane(img, plane, h, lut)
            self.assertEqual(pad, pad2, img.shape)
            self.assertEqual(scale, scale2, img.shape)
            self.assertTrue(np.array_equal(plane, ref), img.shape)

    def test_11_native_head_transpose_matches_numpy(self):
        from ignite_xdna.pipelines import preprocess as pp
        rng = np.random.default_rng(5)
        for blocks, hh, ww, c in ((10, 80, 80, 80), (8, 40, 40, 64), (1, 20, 20, 3)):
            src = rng.integers(0, 256, blocks * hh * ww * 8, dtype=np.uint8)
            dst = np.zeros(c * hh * ww, dtype=np.int8)
            if not pp.blocks_to_nchw_int8(src, blocks, hh, ww, c, dst):
                self.skipTest("native preprocessor without the head transpose")
            ref = (np.transpose(src.reshape(blocks, hh, ww, 8), (0, 3, 1, 2)).reshape(blocks * 8, hh, ww)[:c]
                   ^ 0x80).view(np.int8).reshape(-1)
            self.assertTrue(np.array_equal(dst, ref), (blocks, hh, ww, c))

    def test_12_native_class_max_and_decode_parity(self):
        """Per-anchor class maxima match numpy, and the decoder gives identical detections with them."""
        from ignite_xdna.pipelines import preprocess as pp
        from ignite_xdna.pipelines.yolo_pipeline import YoloDecoder
        rng = np.random.default_rng(11)
        for blocks, hh, ww, c in ((10, 80, 80, 80), (10, 40, 40, 80), (10, 20, 20, 80), (1, 20, 20, 3)):
            src = rng.integers(0, 256, blocks * hh * ww * 8, dtype=np.uint8)
            out = np.zeros(hh * ww, dtype=np.int8)
            if not pp.blocks_class_max_int8(src, blocks, hh, ww, c, out):
                self.skipTest("native preprocessor without the class maximum")
            nchw = (np.transpose(src.reshape(blocks, hh, ww, 8), (0, 3, 1, 2)).reshape(blocks * 8, hh, ww)[:c]
                    ^ 0x80).view(np.int8)
            self.assertTrue(np.array_equal(out, nchw.reshape(c, -1).max(0)), (blocks, hh, ww, c))
        # Decode parity on synthetic int8 heads with a few confident anchors.
        manifest = build_manifest(self.ir, self.ws, self.scheds, self.store, "yolov8n", 0, "x", "y", 0.0)
        status = resolve_head_layout(manifest, int(manifest["egress_bytes"]))
        egress = rng.integers(-128, -60, int(manifest["egress_bytes"]), dtype=np.int8)
        heads = status.layout.unpack(egress)
        for name, hw in (("p3_cls", 80), ("p4_cls", 40), ("p5_cls", 20)):
            flat = heads[name].reshape(80, -1)
            for a in rng.choice(hw * hw, 6, replace=False):
                flat[rng.integers(0, 80), a] = 120
        for name in ("p3_box", "p4_box", "p5_box"):
            heads[name][...] = rng.integers(-40, 40, heads[name].shape, dtype=np.int8)
        heads = dict(heads)
        heads["scales"] = status.layout.scales()
        dec = YoloDecoder(imgsz=640, conf_thres=0.25, iou_thres=0.5)
        ref = dec.postprocess(heads, (0, 0), 1.0)
        with_max = dict(heads)
        with_max["cls_max"] = {n: heads[n].reshape(80, -1).max(0) for n in ("p3_cls", "p4_cls", "p5_cls")}
        got = dec.postprocess(with_max, (0, 0), 1.0)
        self.assertGreater(len(ref), 0)
        self.assertEqual(ref, got)


YOLOV8S = ROOT / "models" / "yolov8s_cut_xint8.onnx"
SESR = ROOT / "models" / "sesr_m7_xint8.onnx"
POSE = ROOT / "models" / "yolov8n-pose_cut_xint8.onnx"


def _emulate_layers(ir, ws, scheds, store, direct, indices):
    """Seed a workspace with reference inputs per layer, clear chosen layer's output, emulate it back."""
    mismatches = {}
    for i in indices:
        ws_arr = ws.halo_fill()
        L = ir.layers[i]
        if isinstance(L, graph_ir.ConvLayer):
            for s in L.inputs:
                ws.write_tensor(ws_arr, s.tensor, direct[s.tensor])
            if L.residual:
                ws.write_tensor(ws_arr, L.residual.tensor, direct[L.residual.tensor])
        elif isinstance(L, graph_ir.PoolLayer):
            ws.write_tensor(ws_arr, L.input.tensor, direct[L.input.tensor])
        elif isinstance(L, graph_ir.HostLayer):
            for s in L.input_segments():
                ws.write_tensor(ws_arr, s.tensor, direct[s.tensor])
        c = ir.tensors[L.output].channels
        es.emulate_layer(scheds[i], store, ws_arr)
        mismatches[L.name] = int(np.sum(ws.read_tensor(ws_arr, L.output)[:c] != direct[L.output][:c]))
    return mismatches


class EngineGeneralizationOffline(unittest.TestCase):
    """Rules added for the model zoo: shifted residual operands (yolov8s), ReLU as an epilogue, 5x5 packets,
    overlapping edge tiles and a DepthToSpace output (SESR M7)."""

    def test_20_relu_epilogue_is_exact(self):
        fit = graph_ir.relu_epilogue()
        q = np.arange(256).astype(np.uint8)
        self.assertEqual(fit.max_error, 0)
        self.assertTrue(np.array_equal(em.hswish_epilogue(q, fit.params), np.maximum(q, 128)))

    def test_21_residual_shifts_match_float_semantics(self):
        """rne((tm << a) + (tr << b), s) is the requantized float Add for every operand pair and scale relation."""
        tm, tr = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
        for s_main, s_res, s_out in ((0.03125, 0.0625, 0.0625), (0.125, 0.0625, 0.0625), (4.0, 2.0, 4.0),
                                     (0.0625, 0.0625, 0.0625)):
            s_min = min(s_main, s_res)
            a, b, sh = (int(round(np.log2(v / s_min))) for v in (s_main, s_res, s_out))
            got = em.residual_combine(tm.astype(np.uint8), tr.astype(np.uint8), sh, a, b)
            x = (tm - 128).astype(np.float64) * s_main + (tr - 128).astype(np.float64) * s_res
            ref = np.clip(np.round(x / s_out) + 128, 0, 255).astype(np.uint8)  # np.round rounds half to even
            self.assertTrue(np.array_equal(got, ref), (s_main, s_res, s_out))
        # The original rule (no F_RES_SHIFTS) is the a = 0, b = rsh case.
        self.assertTrue(np.array_equal(em.residual_combine(tm.astype(np.uint8), tr.astype(np.uint8), 1),
                                       em.residual_combine(tm.astype(np.uint8), tr.astype(np.uint8), 1, 0, 1)))

    def test_22_tile_origins(self):
        self.assertEqual(es.tile_origins(640, 20), list(range(0, 640, 20)))
        self.assertEqual(es.tile_origins(256, 20), list(range(0, 240, 20)) + [236])
        self.assertEqual(es.tile_origins(20, 20), [0])
        with self.assertRaises(ValueError):
            es.tile_origins(16, 20)

    @unittest.skipUnless(YOLOV8S.exists(), "yolov8s model not present")
    def test_23_yolov8s_shifted_main_residual_emulates_exactly(self):
        from ignite_xdna.compiler import graph_reference as gr
        ir = graph_ir.lower_yolov8n(YOLOV8S)
        self.assertEqual(len(ir.layers), 66)
        shifted = [L for L in ir.layers if isinstance(L, graph_ir.ConvLayer) and L.residual is not None
                   and L.residual_lsh_main]
        self.assertEqual([(L.index, L.residual_lsh_main, L.residual_lsh_res, L.residual_shift) for L in shifted],
                         [(23, 1, 0, 0)])
        ws = es.plan_workspace(ir)
        scheds, store = es.schedule_graph(ir, ws)
        manifest = build_manifest(ir, ws, scheds, store, "yolov8s_cut_xint8", 0, "x", "y", 0.0)
        self.assertEqual(manifest["task"], "detect")
        self.assertTrue(resolve_head_layout(manifest, int(manifest["egress_bytes"])).present)
        rng = np.random.default_rng(23)
        direct = gr.run_direct(ir, rng.integers(0, 256, size=(3, 640, 640), dtype=np.uint8))
        self.assertEqual(_emulate_layers(ir, ws, scheds, store, direct, [23]), {shifted[0].name: 0})

    @unittest.skipUnless(SESR.exists(), "sesr_m7 model not present")
    def test_24_sesr_lowering_matches_onnx_runtime_and_emulates_exactly(self):
        from ignite_xdna.compiler import graph_reference as gr
        ir = graph_ir.lower_yolov8n(SESR)
        self.assertEqual([L.k for L in ir.layers], [5] + [3] * 7 + [5])
        self.assertEqual([L.act for L in ir.layers], [None] + ["relu"] * 7 + [None])
        L7 = ir.layers[7]
        self.assertEqual((L7.residual_lsh_main, L7.residual_lsh_res, L7.residual_shift), (1, 0, 1))
        self.assertEqual(ir.output_transforms, {"output": {"op": "depth_to_space", "blocksize": 2, "mode": "CRD"}})
        ws = es.plan_workspace(ir)
        scheds, store = es.schedule_graph(ir, ws)
        manifest = build_manifest(ir, ws, scheds, store, "sesr_m7_xint8", 0, "x", "y", 0.0)
        self.assertEqual(manifest["task"], "super_resolution")
        self.assertEqual(manifest["output_shapes"]["image"], [1, 3, 512, 512])
        self.assertEqual(manifest["egress_bytes"], 12 * 256 * 256)
        rng = np.random.default_rng(24)
        x = rng.integers(0, 256, size=(3, 256, 256)).astype(np.float32) - 128.0
        t_in = ir.tensors[ir.input]
        direct = gr.run_direct(ir, gr.quantize_input(x, t_in.scale, t_in.zero_point))
        ort = gr.ort_intermediates(SESR, x[None], [L.output for L in ir.layers])
        for L in ir.layers:
            c = ir.tensors[L.output].channels
            self.assertTrue(np.array_equal(direct[L.output][:c], ort[L.output]), L.name)
        self.assertEqual(set(_emulate_layers(ir, ws, scheds, store, direct, [0, 7, 8]).values()), {0})

    @unittest.skipUnless(POSE.exists(), "yolov8n-pose model not present")
    def test_25_pose_heads_resolve_and_emulate_exactly(self):
        from ignite_xdna.compiler import graph_reference as gr
        from ignite_xdna.compiler.engine_compile import graph_task, head_name_for
        from ignite_xdna.runtime.heads import POSE_HEAD_NAMES
        ir = graph_ir.lower_yolov8n(POSE)
        convs = [L for L in ir.layers if isinstance(L, graph_ir.ConvLayer)]
        self.assertEqual((len(convs), len(ir.layers)), (72, 75))
        self.assertEqual(sorted(head_name_for(o) for o, _ in ir.outputs), sorted(POSE_HEAD_NAMES))
        self.assertEqual(graph_task(ir), "pose")
        ws = es.plan_workspace(ir)
        scheds, store = es.schedule_graph(ir, ws)
        manifest = build_manifest(ir, ws, scheds, store, "yolov8n-pose_cut_xint8", 0, "x", "y", 0.0)
        self.assertEqual((manifest["task"], manifest["input_dtype"], manifest["num_classes"], manifest["kpt_shape"]),
                         ("pose", "int8", 1, [17, 3]))
        self.assertEqual(manifest["egress_bytes"], (64 + 1 + 51) * (80 * 80 + 40 * 40 + 20 * 20))
        status = resolve_head_layout(manifest, int(manifest["egress_bytes"]))
        self.assertTrue(status.present, status.reason)
        self.assertEqual(len(status.layout.heads), 9)
        self.assertEqual(status.layout.spec("p4_kpt").shape, (1, 51, 40, 40))
        self.assertEqual(status.layout.spec("p5_cls").shape, (1, 1, 20, 20))
        # The 1- and 51-channel heads fill 1 and 7 of their 8-channel blocks; the packets compute them exactly.
        index = {head_name_for(o): L.index for o, t in ir.outputs for L in ir.layers if L.output == t}
        rng = np.random.default_rng(25)
        direct = gr.run_direct(ir, rng.integers(0, 256, size=(3, 640, 640), dtype=np.uint8))
        picked = [index["p3_cls"], index["p3_kpt"], index["p5_kpt"]]
        self.assertEqual(set(_emulate_layers(ir, ws, scheds, store, direct, picked).values()), {0})


@unittest.skipUnless(MODEL.exists(), "quantized model not present")
class SiluSigmoidOffline(unittest.TestCase):
    """--silu-sigmoid: every SiLU through the core's four-line sigmoid epilogue, exact against the reference model."""

    @classmethod
    def setUpClass(cls):
        import onnx
        from ignite_xdna.compiler import silu_sigmoid
        cls.ir = graph_ir.lower_yolov8n(MODEL, silu_sigmoid=True)
        cls.ref = silu_sigmoid.reference_model(onnx.load(str(MODEL)))

    def test_26_every_silu_takes_the_sigmoid_epilogue(self):
        convs = [L for L in self.ir.layers if isinstance(L, graph_ir.ConvLayer)]
        self.assertTrue(self.ir.silu_sigmoid)
        self.assertEqual(sum(L.sigmoid is not None for L in convs), 57)
        self.assertEqual(sum(L.hswish is not None for L in convs), 0)
        for L in convs:
            if L.sigmoid is not None:
                self.assertEqual(len(L.sigmoid.params.lines), em.SIGMOID_LINES)
                self.assertLessEqual(L.sigmoid.max_error, 2, L.name)
                self.assertTrue(np.array_equal(L.sigmoid.table, em.sigmoid_epilogue(np.arange(256), L.sigmoid.params)))

    def test_27_direct_reference_matches_onnx_runtime_on_the_reference_model(self):
        from ignite_xdna.compiler import graph_reference as gr
        rng = np.random.default_rng(27)
        x = rng.random((3, 640, 640), dtype=np.float32)
        t_in = self.ir.tensors[self.ir.input]
        direct = gr.run_direct(self.ir, gr.quantize_input(x, t_in.scale, t_in.zero_point))
        ort = gr.ort_intermediates(self.ref, x[None], [L.output for L in self.ir.layers])
        for L in self.ir.layers:
            c = self.ir.tensors[L.output].channels
            self.assertTrue(np.array_equal(direct[L.output][:c], ort[L.output]), L.name)

    def test_28_emitted_and_held_sigmoid_layers_emulate_exactly(self):
        from ignite_xdna.compiler import graph_reference as gr
        ws = es.plan_workspace(self.ir)
        scheds, store = es.schedule_graph(self.ir, ws)
        convs = [L for L in self.ir.layers if isinstance(L, graph_ir.ConvLayer) and L.sigmoid is not None]
        emitted = next(L.index for L in convs if L.residual is None)
        held = next(L.index for L in convs if L.residual is not None)   # the tile is held, then the residual adds
        rng = np.random.default_rng(28)
        direct = gr.run_direct(self.ir, rng.integers(0, 256, size=(3, 640, 640), dtype=np.uint8), stop_after=held)
        mism = _emulate_layers(self.ir, ws, scheds, store, direct, [emitted, held])
        self.assertEqual(set(mism.values()), {0}, mism)

    def test_29_host_regions_are_refused(self):
        with self.assertRaises(ValueError):
            graph_ir.lower_yolov8n(MODEL, host_regions=["/model.9/"], silu_sigmoid=True)


class NativeDecodeOffline(unittest.TestCase):
    """The native int8 decode (pipelines/decode_native.c) returns the numpy path's detections exactly."""

    def test_native_decode_matches_numpy(self):
        from ignite_xdna.pipelines import decode_native
        from ignite_xdna.pipelines.yolo_pipeline import YoloDecoder
        if not decode_native.available():
            self.skipTest("native decode library not built")
        rng = np.random.default_rng(21)
        numpy_dec, native_dec = YoloDecoder(native_decode=False), YoloDecoder()
        self.assertTrue(native_dec.uses_native_decode)
        total = 0
        for trial in range(150):
            heads, scales = {}, {}
            for name, g in (("p3", 80), ("p4", 40), ("p5", 20)):
                scales[name + "_box"] = (float(2.0 ** rng.integers(-5, 1)), int(rng.integers(-8, 9)))
                scales[name + "_cls"] = (float(2.0 ** rng.integers(-4, 0)), int(rng.integers(-8, 9)))
                heads[name + "_cls"] = rng.integers(-128, -111, (1, 80, g, g), dtype=np.int8)
                heads[name + "_box"] = rng.integers(-128, 128, (1, 64, g, g), dtype=np.int8)
                for _ in range(int(rng.integers(0, 12))):  # same-class clusters with shared logits
                    y, x, r = int(rng.integers(0, g)), int(rng.integers(0, g)), int(rng.integers(0, 3))
                    c, q = int(rng.integers(0, 4)), int(rng.integers(-10, 128))
                    ys, xs = slice(max(y - r, 0), y + r + 1), slice(max(x - r, 0), x + r + 1)
                    heads[name + "_cls"][0, c, ys, xs] = q
                    if rng.random() < 0.3:  # a lower-index class just below: saturated sigmoids tie in float32
                        heads[name + "_cls"][0, (c + 79) % 80, ys, xs] = max(q - int(rng.integers(1, 7)), -128)
                    heads[name + "_box"][0, :, ys, xs] = rng.integers(-128, 128, 64, dtype=np.int8)[:, None, None]
            heads["scales"] = scales
            if trial % 2:
                heads["cls_max"] = {n: heads[n].reshape(80, -1).max(0) for n in ("p3_cls", "p4_cls", "p5_cls")}
            pad = (int(rng.integers(0, 200)), int(rng.integers(0, 200)))
            scale = float(np.float32(rng.uniform(0.3, 2.5)))
            conf, iou = ((0.25, 0.5), (0.1, 0.45), (0.6, 0.7), (0.25, 0.0))[trial % 4]
            ref = numpy_dec.postprocess(heads, pad, scale, conf, iou)
            got = native_dec.postprocess(heads, pad, scale, conf, iou)
            self.assertEqual(ref, got, f"trial {trial}")
            total += len(ref)
        self.assertGreater(total, 0)

    def test_float_heads_keep_the_numpy_path(self):
        from ignite_xdna.pipelines import decode_native
        from ignite_xdna.pipelines.yolo_pipeline import YoloDecoder
        if not decode_native.available():
            self.skipTest("native decode library not built")
        rng = np.random.default_rng(22)
        heads = []
        for channels in (64, 80):
            for g in (80, 40, 20):
                heads.append(rng.normal(-6.0 if channels == 80 else 0.0, 2.0, (1, channels, g, g)).astype(np.float32))
        heads = heads[:3] + heads[3:]
        native_dec = YoloDecoder()
        self.assertIsNone(native_dec._native.decode(heads[:3], heads[3:], None, {}, (0, 0), 1.0, 0.25, 0.5))
        self.assertEqual(YoloDecoder(native_decode=False).postprocess(heads, (0, 0), 1.0),
                         native_dec.postprocess(heads, (0, 0), 1.0))


HEAD_MODEL = ROOT / "build" / "test_resnet50_head.onnx"


@unittest.skipUnless(HEAD_MODEL.exists(), "test_resnet50_head model not present")
class ClassificationHeadOffline(unittest.TestCase):
    """Classification head lowering: 1x1 ConvLayer, 0 host segments, exact emulate_layer and layout."""

    @classmethod
    def setUpClass(cls):
        cls.ir = graph_ir.lower_yolov8n(HEAD_MODEL)
        cls.ws = es.plan_workspace(cls.ir)
        cls.scheds, cls.store = es.schedule_graph(cls.ir, cls.ws)
        cls.manifest = build_manifest(cls.ir, cls.ws, cls.scheds, cls.store, "resnet50_head", 0, "x", "y", 0.0)

    def test_01_classification_manifest_and_zero_host_segments(self):
        from ignite_xdna.compiler.engine_compile import graph_task
        self.assertEqual(graph_task(self.ir), "classify")
        self.assertEqual(len(self.ir.layers), 1)
        self.assertIsInstance(self.ir.layers[0], graph_ir.ConvLayer)
        self.assertEqual(self.ir.layers[0].k, 1)
        self.assertEqual(self.ir.layers[0].stride, 1)
        self.assertEqual(self.ir.layers[0].pad, 0)
        self.assertEqual(self.manifest["task"], "classify")
        self.assertEqual(self.manifest["num_classes"], 1000)
        self.assertEqual(self.manifest["output_shapes"]["logits"], [1, 1000])
        segments = self.manifest.get("graph_engine", {}).get("segments", [])
        host_segs = [s for s in segments if s.get("kind") == "host"]
        self.assertEqual(len(host_segs), 0)

    def test_02_layout_resolution_and_unpacking(self):
        from ignite_xdna.runtime.heads import resolve_classification_layout
        layout = resolve_classification_layout(self.manifest)
        self.assertIsNotNone(layout)
        self.assertEqual(layout.num_classes, 1000)
        self.assertEqual(layout.scale, 0.125)
        self.assertEqual(layout.zero_point, 128)
        # Verify 4D unpacking at (0, 0)
        blocks = np.zeros((125, 20, 20, 8), dtype=np.uint8)
        test_vals = np.arange(1000, dtype=np.uint8)
        blocks[:, 0, 0, :] = test_vals.reshape(125, 8)
        unpacked = layout.unpack(blocks)
        self.assertTrue(np.array_equal(unpacked, test_vals))
        deq = layout.dequantize(unpacked)
        expected_deq = (test_vals.astype(np.float32) - 128.0) * 0.125
        self.assertTrue(np.array_equal(deq, expected_deq))

    def test_03_emulate_layer_exact(self):
        from ignite_xdna.compiler import graph_reference as gr
        rng = np.random.default_rng(42)
        t_in = self.ir.tensors[self.ir.input]
        x_u8 = np.full((t_in.channels, 20, 20), 128, dtype=np.uint8)
        x_u8[:, 0, 0] = rng.integers(0, 256, size=t_in.channels, dtype=np.uint8)
        direct = gr.run_direct(self.ir, x_u8)
        mism = _emulate_layers(self.ir, self.ws, self.scheds, self.store, direct, [0])
        self.assertEqual(list(mism.values()), [0])

    def test_04_direct_matches_onnx_runtime(self):
        import onnxruntime as ort
        from ignite_xdna.compiler import graph_reference as gr
        from ignite_xdna.runtime.heads import resolve_classification_layout
        layout = resolve_classification_layout(self.manifest)
        rng = np.random.default_rng(42)
        t_in = self.ir.tensors[self.ir.input]
        feat = rng.integers(115, 140, size=t_in.channels, dtype=np.uint8)
        sess = ort.InferenceSession(str(HEAD_MODEL), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        ort_in = feat.reshape(1, t_in.channels, 1, 1)
        ort_out = sess.run(None, {in_name: ort_in})[0].reshape(-1)

        x_u8 = np.full((t_in.channels, 20, 20), 128, dtype=np.uint8)
        x_u8[:, 0, 0] = feat
        direct = gr.run_direct(self.ir, x_u8)
        direct_u8 = direct[self.ir.layers[0].output][:, 0, 0]
        direct_deq = layout.dequantize(direct_u8)
        diff = np.abs(direct_deq - ort_out)
        self.assertEqual(float(np.max(diff)), 0.0)


CLS_MODEL = ROOT / "models" / "yolov8n-cls_640_cut_xint8.onnx"


@unittest.skipUnless(CLS_MODEL.exists(), "yolov8n-cls model not present")
class ClassificationHeadPooling(unittest.TestCase):
    """A whole classifier's pooling is carved to the host; its head is never wired to the image.

    match_classification_head swallows GlobalAveragePool into consumed_nodes and nothing lowers it, so
    the pooled tensor is never stored and the head's input falls back to the graph input -- the image.
    The Gemm then addresses more channel blocks than its input holds (1280 channels against 3), which
    is a container that loads, runs and is quietly not the model. lower_yolov8n now names the pool's
    span as a host region on its own, so the average is actually computed."""

    @classmethod
    def setUpClass(cls):
        cls.ir = graph_ir.lower_yolov8n(CLS_MODEL)

    def test_the_pooling_is_carved_into_one_host_layer(self):
        hosts = [L for L in self.ir.layers if isinstance(L, graph_ir.HostLayer)]
        self.assertEqual(len(hosts), 1)
        self.assertIn("GlobalAveragePool", hosts[0].op_types)
        self.assertIn("Flatten", hosts[0].op_types)

    def test_the_head_reads_pooled_channels_and_not_the_graph_input(self):
        head = self.ir.layers[-1]
        self.assertEqual(head.weights.shape[1], sum(s.blocks for s in head.inputs) * 8)
        self.assertNotEqual(head.inputs[0].tensor, self.ir.input)

    def test_an_explicit_region_over_the_pool_is_not_carved_a_second_time(self):
        ir = graph_ir.lower_yolov8n(CLS_MODEL, host_regions=("/model.9/pool/", "/model.9/Flatten"))
        self.assertEqual(sum(1 for L in ir.layers if isinstance(L, graph_ir.HostLayer)), 2)
        self.assertEqual(ir.layers[-1].weights.shape[1], sum(s.blocks for s in ir.layers[-1].inputs) * 8)

    def test_the_gate_still_refuses_when_the_pool_cannot_be_carved_out(self):
        """The backstop behind the carve: if the span cannot become a host region, refuse -- do not wire
        the head to the graph input, which is the miscompile this whole path exists to prevent."""
        real = graph_ir._pool_host_spec
        graph_ir._pool_host_spec = lambda G, head: None
        try:
            with self.assertRaises(ValueError) as cm:
                graph_ir.lower_yolov8n(CLS_MODEL)
        finally:
            graph_ir._pool_host_spec = real
        msg = str(cm.exception)
        self.assertIn("classification head", msg)
        self.assertIn("GlobalAveragePool", msg)
        self.assertIn("1280", msg)


class HostOutputPlacement(unittest.TestCase):
    """A host layer returning a vector lands at pixel (0, 0) of the plane its tensor declares."""

    def test_a_flat_vector_is_placed_at_zero_zero_over_the_zero_point(self):
        from ignite_xdna.compiler.graph_ir import place_host_output
        vec = np.arange(10, 20, dtype=np.uint8)
        plane = place_host_output(vec, channels=16, height=20, width=20)
        self.assertEqual(plane.shape, (16, 20, 20))
        self.assertTrue(np.array_equal(plane[:10, 0, 0], vec))
        self.assertEqual(int(plane[10:, 0, 0].min()), 128)      # unused channels hold the zero point
        self.assertEqual(int(plane[:, 1:, 1:].min()), 128)      # every other pixel holds the zero point
        self.assertEqual(int(plane[:, 0, 1:].max()), 128)

    def test_a_full_plane_is_passed_through(self):
        from ignite_xdna.compiler.graph_ir import place_host_output
        x = np.random.default_rng(3).integers(0, 256, size=(8, 4, 4), dtype=np.uint8)
        self.assertTrue(np.array_equal(place_host_output(x, 8, 4, 4), x))

    def test_an_output_that_is_not_a_vector_is_refused(self):
        from ignite_xdna.compiler.graph_ir import place_host_output
        with self.assertRaises(ValueError):
            place_host_output(np.zeros((8, 3, 2), np.uint8), 8, 20, 20)


@unittest.skipUnless(MODEL.exists(), "quantized model not present")
class WorkspaceReadbackCoverage(unittest.TestCase):
    """Which layers a post-dispatch readback can actually see, and why that is fewer than all of them.

    plan_workspace's liveness reuse aliases tensors of identical geometry onto one slot base, each written
    from the slot start, so after a single dispatch only the last writer survives. verify_engine_container
    read every tensor as though all were still there and reported 25/66 on a device that was correct; these
    pin the model that tells the two cases apart."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tools"))
        from verify_engine_container import slot_final_tenants
        cls.final_tenants = staticmethod(slot_final_tenants)
        cls.ir = graph_ir.lower_yolov8n(MODEL)

    def test_co_tenants_of_a_slot_share_geometry_so_their_bytes_truly_overlap(self):
        ws = es.plan_workspace(self.ir)
        by_base = {}
        for L in self.ir.layers:
            by_base.setdefault(ws.placements[L.output].base, []).append(L.output)
        for base, names in by_base.items():
            shapes = {(self.ir.tensors[n].height, self.ir.tensors[n].width) for n in names}
            self.assertEqual(len(shapes), 1, f"the slot at base {base} aliases {shapes}")

    def test_only_a_slots_final_tenant_is_readable(self):
        ws = es.plan_workspace(self.ir)
        final = self.final_tenants(ws, self.ir)
        readable = {name for _, name in final.values()}
        self.assertTrue(readable <= {L.output for L in self.ir.layers})
        for L in self.ir.layers:
            idx, owner = final[ws.placements[L.output].base]
            if owner != L.output:
                self.assertGreater(idx, L.index, "a slot may only be retaken by a later layer")

    def test_reuse_off_makes_every_layer_readable_and_costs_the_workspace_back(self):
        off = es.plan_workspace(self.ir)
        full = es.plan_workspace(self.ir, reuse=False)
        self.assertEqual(len(self.final_tenants(full, self.ir)), len(self.ir.layers),
                         "with reuse off every tensor owns a slot, so all are readable")
        self.assertLess(len(self.final_tenants(off, self.ir)), len(self.ir.layers))
        self.assertLess(off.nbytes, full.nbytes, "reuse should still be the one that shrinks the workspace")


@unittest.skipUnless(MODEL.exists(), "quantized model not present")
class TestSpatialStencilFusion(unittest.TestCase):
    """Inter-layer spatial stencil fusion in persistent core engine and compiler."""

    @classmethod
    def setUpClass(cls):
        from ignite_xdna.compiler import passes
        cls.raw_ir = graph_ir.lower_yolov8n(MODEL)
        cls.fused_ir = passes.match_stencil_fusion(cls.raw_ir)
        cls.ws = es.plan_workspace(cls.fused_ir)
        cls.scheds, cls.store = es.schedule_graph(cls.fused_ir, cls.ws)

    def test_01_matcher_fuses_eligible_bottleneck(self):
        self.assertEqual(len(self.raw_ir.layers), 66)
        self.assertEqual(len(self.fused_ir.layers), 65)

        fused_layers = [L for L in self.fused_ir.layers if isinstance(L, graph_ir.FusedConvLayer)]
        self.assertEqual(len(fused_layers), 1)
        fused = fused_layers[0]
        self.assertEqual(fused.index, 3)
        self.assertEqual(fused.cin, 16)
        self.assertEqual(fused.cout, 16)
        self.assertEqual(fused.pad, 2)
        self.assertEqual(fused.stage1.name, "/model.2/m.0/cv1/conv/Conv")
        self.assertEqual(fused.stage2.name, "/model.2/m.0/cv2/conv/Conv")
        # Intermediate activation tensor omitted from physical workspace tensors
        self.assertNotIn(fused.stage1.output, self.fused_ir.tensors)

    def test_02_workspace_zero_ddr_intermediate(self):
        fused = self.fused_ir.layers[3]
        # Ingress tensor halo must expand to 2 to support compound receptive field
        in_tensor = fused.inputs[0].tensor
        self.assertEqual(self.ws.placements[in_tensor].halo, 2)
        # Intermediate tensor has zero DDR allocation
        self.assertNotIn(fused.stage1.output, self.ws.placements)

    def test_03_fused_scheduling_invariants(self):
        fused_sched = self.scheds[3]
        self.assertEqual(fused_sched.rounds, 64)
        self.assertEqual(fused_sched.packets, 256)
        self.assertEqual(fused_sched.w_fills, 64)

        blob = self.store.blob()
        for prog in fused_sched.programs:
            a_items = sum(1 for it in prog if it[0] in ("a", "A"))
            w_served = sum(it[3] if it[0] == "w" else it[2] for it in prog if it[0] in ("w", "W"))
            self.assertEqual(w_served, a_items)
            o_held = sum(it[2] for it in prog if it[0] == "o" and len(it) > 2)
            self.assertEqual(o_held, a_items)

    def test_04_emulate_fused_layer_bit_exact(self):
        from ignite_xdna.compiler import graph_reference as gr
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, size=(3, 640, 640), dtype=np.uint8)
        direct = gr.run_direct(self.raw_ir, img)

        ws_arr = self.ws.halo_fill()
        L = self.fused_ir.layers[3]
        for s in L.inputs:
            self.ws.write_tensor(ws_arr, s.tensor, direct[s.tensor])
        if L.residual:
            self.ws.write_tensor(ws_arr, L.residual.tensor, direct[L.residual.tensor])

        es.emulate_layer(self.scheds[3], self.store, ws_arr)
        c = self.fused_ir.tensors[L.output].channels
        got = self.ws.read_tensor(ws_arr, L.output)[:c]
        ref = direct[L.output][:c]
        diff = np.abs(got.astype(int) - ref.astype(int))
        self.assertEqual(int(diff.max()), 0)
        self.assertEqual(int(np.count_nonzero(diff)), 0)


class TestProgramRamEpilogueOps(unittest.TestCase):
    """The elementwise Mul, per-channel scalar Scale and k x k average Pool opcodes added under
    the program-RAM budget: the emulator's integer arithmetic matches a direct NumPy reference,
    computed from the documented contracts, on random tiles."""

    def test_mul_packet_matches_direct_reference(self):
        rng = np.random.default_rng(20260917)
        hold = rng.integers(0, 256, size=(4, 5, 20, 8), dtype=np.uint8)
        for ysh in (0, 6, 11):
            hdr = em.PacketHeader(op=em.OP_MUL, ncin=4, nco=4, flags=em.F_EMIT,
                                  hs=em.HardSwishParams(0, 0, 0, 0, 0, 0, ysh), count_out=1)
            w = em.pack_w_packet(hdr, np.zeros(32, np.int32), None)
            a = rng.integers(0, 256, size=em.A_BYTES, dtype=np.uint8)
            st = em.CoreState()
            st.hold[:] = hold
            got = em.run_packet(w, a, st, 0)
            tm = hold.astype(np.int64) - 128
            tr = a[:4 * 800].astype(np.int64).reshape(4, 5, 20, 8) - 128
            # Products are |.| <= 16,384 and the shifted result is exact as a binary fraction,
            # so numpy's round-half-to-even is the core's conv_even.
            ref = np.clip(np.round(tm * tr / (1 << ysh)).astype(np.int64) + 128, 0, 255)
            self.assertTrue(np.array_equal(got.reshape(4, 5, 20, 8), ref.astype(np.uint8)), ysh)

    def test_scale_packet_matches_direct_reference(self):
        rng = np.random.default_rng(2026)
        coef_ch = rng.integers(-2048, 2048, size=(4, 8), dtype=np.int64)  # one int16-range gain per channel
        coef32 = np.concatenate([np.tile(coef_ch[b], (4, 1)).ravel()[None, :] for b in range(4)],
                                axis=0).astype(np.int16)  # [4 blocks][32 lanes], channel = lane % 8
        hdr = em.PacketHeader(op=em.OP_SCALE, ncin=4, nco=4, flags=em.F_EMIT,
                              hs=em.HardSwishParams(0, 0, 0, 0, 0, 0, 5), count_out=1)
        w = em.pack_w_packet(hdr, np.zeros(32, np.int32), None, extra=coef32.view(np.uint8).ravel())
        a = rng.integers(0, 256, size=em.A_BYTES, dtype=np.uint8)
        got = em.run_packet(w, a, em.CoreState(), 0)
        t = a[:4 * 800].astype(np.int64).reshape(4, 5, 20, 8) - 128
        co = np.tile(coef32, (1, 25)).reshape(4, 5, 20, 8)
        ref = np.clip(np.round(t * co / 32).astype(np.int64) + 128, 0, 255)  # ysh = 5
        self.assertTrue(np.array_equal(got.reshape(4, 5, 20, 8), ref.astype(np.uint8)))

    def test_pool_packet_pair_matches_direct_reference(self):
        rng = np.random.default_rng(4242)
        # (k, K2, S2): the true 1/25 and 1/9 reciprocals, and the 2x reciprocal the synthetic
        # silicon scenario uses — 5243/2^16 ≈ 1/12.5, whose bright-region results exceed 255 and
        # engage the uint8 saturation.
        for k, k2, s2 in ((5, 2621, 16), (3, 7282, 16), (5, 5243, 16)):
            hdr_h = em.PacketHeader(op=em.OP_POOL, k=k, ncin=2, nco=4, flags=em.F_HOLD,
                                    hs=em.HardSwishParams(0, 0, 0, 0, k2, s2, 0), count_out=0,
                                    count_acc=1, rows_in=16, cols_in=25, plane_bytes=3200)
            hdr_e = em.PacketHeader(op=em.OP_POOL, k=k, ncin=2, nco=4, flags=em.F_EMIT,
                                    hs=em.HardSwishParams(0, 0, 0, 0, k2, s2, 0), count_out=1,
                                    rows_in=16, cols_in=25, plane_bytes=3200)
            st = em.CoreState()
            a_h = rng.integers(0, 256, size=em.A_BYTES, dtype=np.uint8)
            a_e = rng.integers(0, 256, size=em.A_BYTES, dtype=np.uint8)
            self.assertIsNone(em.run_packet(em.pack_w_packet(hdr_h, np.zeros(32, np.int32), None),
                                            a_h, st, 0))
            got = em.run_packet(em.pack_w_packet(hdr_e, np.zeros(32, np.int32), None), a_e, st, 0)

            def pooled(a):
                """Both planes of one pool packet, pooled — what the core leaves in blocks 0-1
                of the hold packet and blocks 2-3 of the emit packet."""
                res = np.zeros((2, 5, 20, 8), dtype=np.uint8)
                for b in range(2):
                    plane = a[b * 3200:(b + 1) * 3200].reshape(16, 25, 8).astype(np.int64)
                    sums = np.zeros((5, 20, 8), dtype=np.int64)
                    for dy in range(k):
                        for dx in range(k):
                            sums += plane[dy:dy + 5, dx:dx + 20, :]
                    res[b] = np.clip(np.round(sums * k2 / (1 << s2)).astype(np.int64),
                                     0, 255).astype(np.uint8)
                return res

            ref = np.concatenate([pooled(a_h), pooled(a_e)])
            self.assertTrue(np.array_equal(got.reshape(4, 5, 20, 8), ref), k)


class TestDispatchGate(unittest.TestCase):
    """The compile-time dispatch gate's fail-closed check.

    The gate derives its case set from the container's own packets, so ignite-compile cannot
    emit a packet its kernel has no case for. If one ever reached a gated-out case the core
    would fall through to `default:` - emitting nothing and raising nothing - so the check
    that catches it has to be exercised rather than assumed.
    """

    @staticmethod
    def _wpackets(ops):
        pkts = np.zeros((len(ops), em.W_BYTES), np.uint8)
        words = np.zeros((len(ops), em.HDR_BYTES // 4), np.int32)
        words[:, em.H_OP] = ops
        pkts[:, :em.HDR_BYTES] = words.view(np.uint8).reshape(len(ops), em.HDR_BYTES)
        return pkts.tobytes()

    def test_passes_when_the_kernel_covers_every_packet(self):
        wp = self._wpackets([em.OP_CONV, em.OP_MAXPOOL, em.OP_RESIDUAL])
        check_kernel_covers_packets({"kernel_ops": ["CONV", "MAXPOOL", "RESIDUAL"]}, wp)

    def test_refuses_a_packet_the_kernel_cannot_dispatch(self):
        wp = self._wpackets([em.OP_CONV, em.OP_POOL])
        with self.assertRaises(ValueError) as cm:
            check_kernel_covers_packets({"kernel_ops": ["CONV", "MAXPOOL", "RESIDUAL"]}, wp)
        self.assertIn("POOL", str(cm.exception))

    def test_a_pre_gate_container_carries_no_op_list_and_is_not_checked(self):
        check_kernel_covers_packets({}, self._wpackets([em.OP_POOL]))

    def test_nop_never_needs_a_dispatch_case(self):
        # tools/engine_dispatch_floor*.py rewrite every header to OP_NOP to measure the
        # dispatch floor, and OP_NOP is served by `default:`. Those containers have to stay
        # verifiable against a gated kernel.
        check_kernel_covers_packets({"kernel_ops": ["CONV"]}, self._wpackets([em.OP_NOP] * 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)

