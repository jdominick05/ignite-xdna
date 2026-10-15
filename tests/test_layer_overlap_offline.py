"""Offline proof for layer overlap (``SequenceEmitter(overlap_layers=True)``), no device and no IRON.

Removing the layer barrier is safe only if no shim task is pushed while an earlier-layer task it conflicts with in
the workspace might still be running: a fill must not read bytes a drain has not finished writing (RAW), and a
drain must not write bytes a fill has not finished reading (WAR) or another drain is still writing (WAW). Twenty
exact dispatches cannot show that - a hazard is a race - and the emulator does not model time. So it is proved here
over the emitted stream itself, for the real YOLOv8n and YOLOv8s schedules.

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

MODELS = {"yolov8n": ROOT / "models" / "yolov8n_cut_xint8.onnx",
          "yolov8s": ROOT / "models" / "yolov8s_cut_xint8.onnx"}


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
        model = MODELS[name]
        if not model.exists():
            self.skipTest(f"{name} model not present")
        if name not in self._cache:
            ir = graph_ir.lower_yolov8n(model)
            ws = es.plan_workspace(ir)
            self._cache[name] = (ws, es.schedule_graph(ir, ws)[0])
        return self._cache[name]

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

    def test_03_yolov8n_overlap_hazard_free(self):
        em, _ = self._emit("yolov8n", overlap=True)
        self.assertEqual(em.hazards, [], f"hazards under overlap: {em.hazards[:5]}")

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


if __name__ == "__main__":
    unittest.main()
