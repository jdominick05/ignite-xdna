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

    def test_04_every_activation_fill_is_one_packet_and_word_aligned(self):
        for s in self.scheds:
            for prog in s.programs:
                for it in prog:
                    if it[0] == "a":
                        for p in it[1]:
                            self.assertEqual(p.nbytes, em.A_BYTES)
                            self.assertEqual(p.offset % 4, 0)
                    elif it[0] == "A":
                        self.assertEqual(it[1].nbytes, 4 * em.A_BYTES)
                    elif it[0] == "o":
                        self.assertIn(it[1].nbytes, (4 * em.O_BYTES, 8 * em.O_BYTES))
                    elif it[0] == "w":
                        self.assertEqual(it[2] % em.W_BYTES, 0)
                        self.assertGreater(it[3], 0)

    def test_05_weight_objects_match_activation_items(self):
        """Every column program delivers exactly as many weight objects as its A items consume."""
        for s in self.scheds:
            for prog in s.programs:
                served = sum(it[3] for it in prog if it[0] == "w")
                a_items = sum(1 for it in prog if it[0] in ("a", "A"))
                self.assertEqual(served, a_items, s.name)
                objects = sum(it[2] // em.W_BYTES for it in prog if it[0] == "w")
                counts = 0
                for it in prog:
                    if it[0] == "w":
                        for k in range(0, it[2], em.W_BYTES):
                            hdr = em.unpack_w_packet(self.store.packet_at(it[1] + k))[0]
                            counts += hdr.count_out + hdr.count_acc
                self.assertEqual(counts, a_items, s.name)
                if a_items:
                    self.assertGreaterEqual(objects, 1)

    def test_06_schedule_size(self):
        rounds = sum(s.rounds for s in self.scheds)
        tasks = sum(program_task_count(p) for s in self.scheds for p in s.programs)
        self.assertEqual(rounds, 1415)
        self.assertLess(tasks, 8000)
        self.assertLess(self.store.nbytes, 12 * 1024 * 1024)

    def test_07_merge_quad_requires_regular_spacing(self):
        from ignite_xdna.compiler.engine_sequence import DmaPattern
        pats = [DmaPattern("ws", 1000 + 400 * i, (8, 5, 160), (2000, 400, 1)) for i in range(4)]
        m = merge_quad(pats)
        self.assertEqual((m.sizes, m.strides), ((4, 8, 5, 160), (400, 2000, 400, 1)))
        pats[3] = DmaPattern("ws", 1000 + 400 * 5, (8, 5, 160), (2000, 400, 1))
        self.assertIsNone(merge_quad(pats))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
