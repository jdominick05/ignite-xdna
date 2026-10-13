"""Offline checks of Block 5 stream sharing (``compiler/stream_share.py``), no device and no IRON.

The transform moves one activation quad in four onto a column's weight stream. What it must preserve, and what is
asserted here on the real YOLOv8n and YOLOv8s schedules:
* ``split_quads`` cuts a fill into quads whose bytes, in order, are the fill's;
* ``check_stream_share`` passes: every quad reads the bytes it read before and sits on the stream its position
  says, each stream issues in position order, no weight task waits on the stream for a quad queued behind it,
  buffer B never has to hold more than it can, and every hold is what the emitter needs;
* the checker is not vacuous: a quad on the wrong stream, a wrong hold, and a quad moved behind the weight run
  that needs it are each refused;
* the cores see the same thing: emulating the shared schedule leaves the workspace byte-identical to the shipped
  schedule, layer by layer, across all of YOLOv8n.

The emulator's handling of shared items (quads handed to the cores in position order, a quad waiting for its
weight object) is a MODEL of the MemTile chains, written with the transform. These tests show the transform is
exact under that model; only the device shows the model is the hardware.
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

from ignite_xdna.compiler import engine_emulator as em  # noqa: E402
from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import graph_ir  # noqa: E402
from ignite_xdna.compiler import stream_share as ss  # noqa: E402
from ignite_xdna.compiler.engine_sequence import DmaPattern  # noqa: E402

MODELS = {"yolov8n": ROOT / "models" / "yolov8n_cut_xint8.onnx",
          "yolov8s": ROOT / "models" / "yolov8s_cut_xint8.onnx"}


class StreamShareOffline(unittest.TestCase):
    _cache = {}

    def _shared(self, name):
        if not MODELS[name].exists():
            self.skipTest(f"{name} model not present")
        if name not in self._cache:
            ir = graph_ir.lower_yolov8n(MODELS[name])
            ws = es.plan_workspace(ir)
            self._cache[name] = (ir, ws) + tuple(ss.schedule_stream_shared(ir, ws))
        return self._cache[name]

    def test_01_split_quads_keeps_bytes_in_order(self):
        q = ss.QUAD
        # A 4-D fill whose outer dimension enumerates quads, and a 3-D one where it must be cut inside a dimension.
        for p in (DmaPattern("ws", 4096, (3, 4, 8, 800), (8 * 3200, 800, 3200, 1)),
                  DmaPattern("ws", 512, (8, 8, 1600), (12800, 1600, 1))):
            parts = ss.split_quads(p)
            self.assertEqual(len(parts), p.nbytes // q)
            self.assertTrue(all(x.nbytes == q for x in parts))
            self.assertTrue(np.array_equal(np.concatenate([x.indices() for x in parts]), p.indices()))
        with self.assertRaises(ValueError):
            ss.split_quads(DmaPattern("ws", 0, (q + 6400,), (1,)))

    def test_02_checker_passes_and_counts(self):
        for name in ("yolov8n", "yolov8s"):
            with self.subTest(model=name):
                _, _, orig, shared, _, modes = self._shared(name)
                counts = ss.check_stream_share(orig, shared)
                self.assertGreater(counts["stream1_tasks"], 0)
                self.assertEqual(len(modes), len(orig))

    def _mutate_and_expect(self, name, mutate, message):
        _, _, orig, shared, _, _ = self._shared(name)
        bad = [copy.copy(s) for s in shared]
        for s in bad:
            s.programs = [list(p) for p in s.programs]
        self.assertTrue(mutate(bad), "no site for the mutation")
        with self.assertRaisesRegex(AssertionError, message):
            ss.check_stream_share(orig, bad)

    def test_03_checker_refuses_a_quad_on_the_wrong_stream(self):
        def mutate(bad):
            for s in bad:
                for prog in s.programs:
                    for i, it in enumerate(prog):
                        if it[0] == "P":
                            prog[i] = ("A",) + it[1:]
                            return True
            return False
        self._mutate_and_expect("yolov8n", mutate, "wrong stream|position")

    def test_04_checker_refuses_a_wrong_hold(self):
        def mutate(bad):
            for s in bad:
                for prog in s.programs:
                    for i, it in enumerate(prog):
                        if it[0] == "o" and it[2] > 0:
                            prog[i] = ("o", it[1], it[2] - 1)
                            return True
            return False
        self._mutate_and_expect("yolov8n", mutate, "hold")

    def test_05_checker_refuses_a_quad_behind_its_weight_run(self):
        def mutate(bad):
            for s in bad:
                for prog in s.programs:
                    for i in range(1, len(prog)):
                        it = prog[i]
                        if it[0] in ("w", "W") and ss._weight_objects(it) > ss.CORE_W_DEPTH and prog[i - 1][0] == "P":
                            prog[i - 1], prog[i] = prog[i], prog[i - 1]
                            return True
            return False
        self._mutate_and_expect("yolov8s", mutate, "behind|hold")

    def test_06_frame_is_whole_periods_with_a_nop_pad(self):
        _, _, _, shared, store, _ = self._shared("yolov8s")
        for c in range(es.COLS):
            positions = [p for s in shared for it in s.programs[c] if it[0] in ("A", "P") for p in it[2]]
            self.assertEqual(len(positions) % ss.PERIOD, 0, f"column {c}")
            self.assertEqual(sorted(positions), list(range(len(positions))), f"column {c}")
        nop = ss.nop_packet(3)
        hdr = em.unpack_w_packet(nop)[0]
        self.assertEqual((hdr.op, hdr.count_out, hdr.count_acc, hdr.flags), (em.OP_NOP, 0, 3, 0))

    def test_07_yolov8n_emulates_byte_identical_to_the_shipped_schedule(self):
        ir, ws, _, shared, store, _ = self._shared("yolov8n")
        base, base_store = es.schedule_graph(ir, ws)
        rng = np.random.default_rng(11)
        a = ws.halo_fill()
        ws.write_tensor(a, ir.input, rng.integers(128, 256, size=(3, 640, 640), dtype=np.uint8))
        b = a.copy()
        for so, sh in zip(base, shared):
            es.emulate_layer(so, base_store, a)
            es.emulate_layer(sh, store, b)
            self.assertTrue(np.array_equal(a, b), so.name)


if __name__ == "__main__":
    unittest.main()
