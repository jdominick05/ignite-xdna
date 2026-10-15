"""Offline checks of box fills (``compiler/box_fills.py``), no device and no IRON.

The transform replaces a layer's one-quad activation fills by 3-D boxes the MemTile cuts per core, and merges
adjacent boxes into one shim task. What it must preserve, asserted on the real YOLOv8n and YOLOv8s schedules:
* every converted fill's box, cut, reproduces the fill's per-core objects byte for byte and in order (the
  transform refuses otherwise; here the refusal itself is shown to fire);
* a layer converts only when every fill shares one geometry, and nothing else in its programs changes;
* the cores see the same thing: emulating the boxed schedule leaves the workspace byte-identical to the shipped
  schedule, layer by layer.

The emulator's cut is a MODEL of the MemTile serves, written with the transform; only the device shows it is the
hardware (tools/verify_engine_container.py on a --box-fills container).
"""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.compiler import box_fills as bf  # noqa: E402
from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import graph_ir  # noqa: E402
from ignite_xdna.compiler.engine_sequence import DmaPattern, program_task_count  # noqa: E402

MODELS = {"yolov8n": ROOT / "models" / "yolov8n_cut_xint8.onnx",
          "yolov8s": ROOT / "models" / "yolov8s_cut_xint8.onnx"}
# The engine's dominant YOLOv8s fill: core j takes 4 planes x rows 5j..5j+7 (a 3-row halo).
O4 = DmaPattern("ws", 4096, (4, 4, 8, 200), (3280, 53792, 656, 1))


class BoxFillsOffline(unittest.TestCase):
    _cache = {}

    def _scheds(self, name):
        if not MODELS[name].exists():
            self.skipTest(f"{name} model not present")
        if name not in self._cache:
            ir = graph_ir.lower_yolov8n(MODELS[name])
            ws = es.plan_workspace(ir)
            base, store = es.schedule_graph(ir, ws)
            boxed = copy.deepcopy(base)
            n = bf.apply_box_fills(boxed)
            self._cache[name] = (ir, ws, base, boxed, store, n)
        return self._cache[name]

    def test_01_dominant_shape_is_a_box_that_cuts_back_exactly(self):
        g = bf.geometry_of(O4)
        self.assertEqual(g, bf.BoxGeometry(planes=4, span=23, width=200, step=5, rows=8))
        self.assertEqual(g.box_bytes, 18400)
        bf._check(O4, g)
        self.assertEqual([g.last_byte_end(j) for j in range(4)], [15400, 16400, 17400, 18400])

    def test_02_a_wrong_geometry_is_refused(self):
        with self.assertRaises(AssertionError):
            bf._check(O4, bf.BoxGeometry(planes=4, span=23, width=200, step=4, rows=8))
        self.assertIsNone(bf.geometry_of(DmaPattern("ws", 0, (4, 40, 160), (51200, 640, 1))))

    def test_03_yolov8s_converts_the_uniform_layers_and_cuts_tasks(self):
        _, _, base, boxed, _, n = self._scheds("yolov8s")
        self.assertEqual(n, 28)
        t0 = sum(program_task_count(p) for s in base for p in s.programs)
        t1 = sum(program_task_count(p) for s in boxed for p in s.programs)
        self.assertLess(t1, t0)
        for so, sb in zip(base, boxed):
            if sb.box is None:
                self.assertEqual(so.programs, sb.programs, so.name)
                continue
            for po, pb in zip(so.programs, sb.programs):
                # Everything but the fills is untouched and in order.
                self.assertEqual([it for it in po if it[0] not in ("a", "A")],
                                 [it for it in pb if it[0] != "X"], so.name)
                boxes = sum(it[2] for it in pb if it[0] == "X")
                quads = sum(len(it[1]) if it[0] == "a" else 1 for it in po if it[0] in ("a", "A"))
                self.assertEqual(boxes, quads, so.name)

    def _emulates_identically(self, name):
        ir, ws, base, boxed, store, _ = self._scheds(name)
        rng = np.random.default_rng(11)
        a = ws.halo_fill()
        ws.write_tensor(a, ir.input, rng.integers(128, 256, size=(3, 640, 640), dtype=np.uint8))
        b = a.copy()
        for so, sb in zip(base, boxed):
            es.emulate_layer(so, store, a)
            es.emulate_layer(sb, store, b)
            self.assertTrue(np.array_equal(a, b), so.name)

    def test_04_yolov8n_emulates_byte_identical_to_the_shipped_schedule(self):
        self._emulates_identically("yolov8n")

    def test_05_yolov8s_emulates_byte_identical_to_the_shipped_schedule(self):
        self._emulates_identically("yolov8s")


if __name__ == "__main__":
    unittest.main()
