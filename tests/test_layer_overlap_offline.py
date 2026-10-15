"""Offline proof for layer overlap (``SequenceEmitter(overlap_layers=True)``), no device and no IRON.

Removing the layer barrier is safe only if no shim task is pushed while an earlier-layer task it conflicts with in
the workspace might still be running: a fill must not read bytes a drain has not finished writing (RAW), and a
drain must not write bytes a fill has not finished reading (WAR) or another drain is still writing (WAW). Twenty
exact dispatches cannot show that - a hazard is a race - and the emulator does not model time. So it is proved here
over the emitted stream itself, for the real schedule of every graph in ``MODELS``: each one the compiler's default
reaches (overlap is on by default wherever a build allows it), plus the 640 classifier, whose host layer keeps it on
the barrier. A new monolithic graph the default reaches belongs in ``MODELS`` before it ships.

The checker is independent of ``compiler/layer_overlap.py``: it uses exact byte sets (``DmaPattern.indices``), not the
tracker's 8-byte cells, and it knows a task is complete only from the awaits the stream actually makes, plus the
rule that a shim channel completes its tasks in order (an awaited task completes every earlier task of its column
and channel). It also checks itself: with the emitter's dependency waits switched off it must find hazards.
"""
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import graph_ir  # noqa: E402
from ignite_xdna.compiler.engine_sequence import (SequenceEmitter, frame_retire_batch,  # noqa: E402
                                                  program_task_count)
from ignite_xdna.compiler.layer_overlap import ConflictTracker  # noqa: E402

# Every graph the compiler turns overlap on for by default (monolithic int8 graph containers): (model, silu_sigmoid).
MODELS = {"yolov8n": (ROOT / "models" / "yolov8n_cut_xint8.onnx", False),
          "yolov8s": (ROOT / "models" / "yolov8s_cut_xint8.onnx", False),
          "yolov8n_silu": (ROOT / "models" / "yolov8n_cut_xint8.onnx", True),
          "yolov8s_silu": (ROOT / "models" / "yolov8s_cut_xint8.onnx", True),
          "yolov8n_pose": (ROOT / "models" / "yolov8n-pose_cut_xint8.onnx", False),
          "sesr_m7": (ROOT / "models" / "sesr_m7_xint8.onnx", False),
          "yolov8n_cls640": (ROOT / "models" / "yolov8n-cls_640_cut_xint8.onnx", False),
          "resnet50_head": (ROOT / "build" / "test_resnet50_head.onnx", False)}


class _Task:
    def __init__(self, idx, col, channel, layer, token, nbytes):
        self.idx, self.col, self.channel, self.layer, self.token = idx, col, channel, layer, token
        self.bytes = nbytes
        self.awaited = False


class _NoDeps(ConflictTracker):
    """The tracker with its answers withheld: what the emitter would do with no dependency waits at all."""

    def deps(self, kind, pattern, layer):
        return {}


class ProofEmitter(SequenceEmitter):
    """The emitter with IRON replaced by a recorder that proves every push hazard-free."""

    def __init__(self, ws_bytes, overlap, withhold=False):
        self._names = {c: {ch: f"{ch}{c}" for ch in ("w", "a", "o")} for c in range(4)}
        self._bytes = {"ws": ws_bytes, "wp": 1 << 30}
        self._await, self._free = self._on_await, self._on_free
        self._overlap = None
        if overlap:
            self._overlap = {"queues": {}, "seq": 0,
                             "tracker": (_NoDeps if withhold else ConflictTracker)(ws_bytes)}
        self.layer = -1
        self.tasks = []
        self.order = defaultdict(list)
        self.done = defaultdict(lambda: -1)   # (col, channel) -> position of the newest awaited task
        self.outstanding = []                 # workspace tasks not yet known complete
        self.hazards = []
        self.live = defaultdict(int)
        self.first_push = {}
        self.overlapped = 0

    def _complete(self, t):
        return self.order[(t.col, t.channel)].index(t) <= self.done[(t.col, t.channel)]

    def transfer(self, alloc, pattern, token, packet=None):
        channel, col = alloc[0], int(alloc[1:])
        nbytes = np.unique(pattern.indices()) if pattern.buffer == "ws" else None
        t = _Task(len(self.tasks), col, channel, self.layer, token, nbytes)
        if self.layer > 0 and self.layer not in self.first_push:
            self.first_push[self.layer] = t.idx
            prev = [o for o in self.tasks if o.layer == self.layer - 1]
            if any(not self._complete(o) for o in prev):
                self.overlapped += 1
        if nbytes is not None:
            self.outstanding = [o for o in self.outstanding if not self._complete(o)]
            for o in self.outstanding:
                if o.layer == t.layer:
                    continue
                kinds = {("a", "o"): "RAW", ("o", "a"): "WAR", ("o", "o"): "WAW"}
                kind = kinds.get((t.channel, o.channel))
                if kind and np.intersect1d(t.bytes, o.bytes, assume_unique=True).size:
                    self.hazards.append((kind, t.layer, t.col, o.layer, o.col))
            self.outstanding.append(t)
        self.tasks.append(t)
        self.order[(col, channel)].append(t)
        self.live[col] += 1
        return t

    def _on_await(self, t):
        assert t.token, "awaited a task without a token"
        assert not t.awaited, "a token awaited twice"
        t.awaited = True
        pos = self.order[(t.col, t.channel)].index(t)
        self.done[(t.col, t.channel)] = max(self.done[(t.col, t.channel)], pos)

    def _on_free(self, t):
        self.live[t.col] -= 1


class LayerOverlapOffline(unittest.TestCase):
    _cache = {}

    def _scheds(self, name):
        model, silu = MODELS[name]
        if not model.exists():
            self.skipTest(f"{name} model not present")
        if name not in self._cache:
            ir = graph_ir.lower_yolov8n(model, silu_sigmoid=silu)
            ws = es.plan_workspace(ir)
            self._cache[name] = (ws, es.schedule_graph(ir, ws)[0], ir)
        return self._cache[name][:2]

    def _ir(self, name):
        self._scheds(name)
        return self._cache[name][2]

    def _emit(self, name, overlap, withhold=False):
        ws, scheds = self._scheds(name)
        em = ProofEmitter(ws.nbytes, overlap, withhold)
        rb = frame_retire_batch(sum(program_task_count(p) for s in scheds for p in s.programs))
        for s in scheds:
            em.layer += 1
            n0 = len(em.tasks)
            em.run_column_programs(s.programs, bd_budget=14, retire_batch=rb)
            self.assertEqual(len(em.tasks) - n0, sum(program_task_count(p) for p in s.programs), s.name)
        em.finish_frame()
        self.assertTrue(all(t.awaited for t in em.tasks if t.token), "a token left unconsumed at frame end")
        self.assertEqual(sum(em.live.values()), 0, "a task left live at frame end")
        return em, scheds

    def test_01_barrier_is_hazard_free(self):
        em, scheds = self._emit("yolov8s", overlap=False)
        self.assertEqual(em.hazards, [])
        # With the barrier, no boundary overlaps: the counter below is meaningful only if it reads 0 here.
        self.assertEqual(em.overlapped, 0)

    def test_02_yolov8s_overlap_hazard_free(self):
        em, _ = self._emit("yolov8s", overlap=True)
        self.assertEqual(em.hazards, [], f"hazards under overlap: {em.hazards[:5]}")

    def test_03_every_default_graph_hazard_free(self):
        # Overlap is the compiler's default wherever a build allows it, so every graph it reaches is proved here:
        # the YOLOv8 detectors with and without the SiLU sigmoid, pose, SESR M7 and both classifiers.
        for name in MODELS:
            if name == "yolov8s":
                continue   # test_02
            with self.subTest(model=name):
                em, _ = self._emit(name, overlap=True)
                self.assertEqual(em.hazards, [], f"{name}: hazards under overlap: {em.hazards[:5]}")

    def test_04_checker_catches_missing_waits(self):
        # The checker's own sensitivity: the same stream with the dependency waits withheld must show hazards,
        # else a pass above would mean nothing.
        em, _ = self._emit("yolov8s", overlap=True, withhold=True)
        self.assertGreater(len(em.hazards), 0, "the checker found no hazard with every dependency wait removed")
        self.assertTrue({h[0] for h in em.hazards} & {"RAW"}, "no read-after-write found without waits")

    def test_05_overlap_happens(self):
        # Without a barrier the change must actually overlap: at a layer's first push, some task of the layer before
        # it is not yet known complete. Required on most boundaries, else the flag does nothing.
        em, scheds = self._emit("yolov8s", overlap=True)
        self.assertGreater(em.overlapped, (len(scheds) - 1) // 2, f"only {em.overlapped} boundaries overlap")

    def test_06_default_decision(self):
        # What the compiler's default (overlap_layers=None) decides, from the same function it calls. Every graph
        # above must be on the grain and unblocked; each excluded mode, a host layer or the bf16 design blocks it.
        from ignite_xdna.compiler.graph_ir import HostLayer
        from ignite_xdna.compiler.layer_overlap import overlap_blockers
        none = {"box fills": False, "activation ring": False}
        hosted = {"yolov8n_cls640"}   # its graph lowers a host layer, so the default keeps its barrier
        for name in MODELS:
            with self.subTest(model=name):
                _, scheds = self._scheds(name)
                has_host = any(isinstance(L, HostLayer) for L in self._ir(name).layers)
                self.assertEqual(has_host, name in hosted, f"{name}: host layer {has_host}")
                self.assertEqual(overlap_blockers(scheds, has_host, True, none),
                                 ["host layers"] if has_host else [], name)
        _, scheds = self._scheds("yolov8n")
        self.assertEqual(overlap_blockers(scheds, False, True, {"box fills": True}), ["box fills"])
        self.assertEqual(overlap_blockers(scheds, True, True, none), ["host layers"])
        self.assertEqual(overlap_blockers(scheds, False, False, none), ["an engine design without transport options"])

    def test_07_off_grain_pattern_blocks(self):
        # A pattern off the 8-byte grain is refused by the tracker and reported by the pre-check, never truncated.
        from ignite_xdna.compiler.layer_overlap import off_grain, pattern_cells

        class _S:
            def __init__(self, pattern):
                self.programs = [[("o", pattern)]]

        _, scheds = self._scheds("yolov8n")
        good = next(item[1] for s in scheds for items in s.programs for item in items if item[0] == "o")
        self.assertIsNone(off_grain([_S(good)]))
        import dataclasses
        bad = dataclasses.replace(good, offset=good.offset + 4)   # 4-byte aligned, as DMA allows; off the 8 grain
        with self.assertRaises(ValueError):
            pattern_cells(bad)
        self.assertIs(off_grain([_S(bad)]), bad)


if __name__ == "__main__":
    unittest.main()
