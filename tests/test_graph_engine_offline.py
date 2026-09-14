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
from ignite_xdna.compiler.engine_compile import build_manifest  # noqa: E402
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
        ps = sorted(self.ws.placements.values(), key=lambda p: p.base)
        for a, b in zip(ps, ps[1:]):
            self.assertLessEqual(a.base + a.nbytes, b.base, a.name)
        for p in ps:
            self.assertEqual(p.base % 64, 0)
            self.assertIn(p.halo_value, (0, 128))
        self.assertLess(self.ws.nbytes, 64 * 1024 * 1024)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
