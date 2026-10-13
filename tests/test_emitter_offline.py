"""Offline guard for the shim-task emitter (``SequenceEmitter.run_column_programs``), no device and no IRON.

Nothing else offline runs the emitter: the schedule tests check item lists and their bytes, and only a hardware
compile used to reach the code that turns those items into tasks, tokens, awaits and frees. A mistake there does
not corrupt bytes on paper - it deadlocks a dispatch or overruns a shim's sixteen descriptors - so it is checked
here against the real YOLOv8n and YOLOv8s schedules with the IRON calls replaced by a recorder.

What the hardware requires, and what is asserted:
* at most ``bd_budget`` tasks live on a shim tile, and at most ``queue_depth`` pending on one channel, whenever
  a task is configured;
* only a task that carries a completion token is awaited, each such task exactly once, in issue order per
  channel; a task is freed only once a task at or after it on its channel has been awaited;
* a weight task or a drain issued ahead of its fills (a held task) is awaited only after the activation items
  it serves have been issued. Which items serve which task is modelled here from the item lists alone: each
  activation item serves the oldest still-held task of each kind in its column;
* when a layer's call returns, every task has been freed and every token consumed;
* the tasks issued per column equal ``program_task_count`` of its program.
"""
import sys
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import graph_ir  # noqa: E402
from ignite_xdna.compiler.engine_sequence import SequenceEmitter, program_task_count  # noqa: E402

MODELS = {"yolov8n": ROOT / "models" / "yolov8n_cut_xint8.onnx",
          "yolov8s": ROOT / "models" / "yolov8s_cut_xint8.onnx"}


def item_model(programs):
    """The transfers run_column_programs must make for one layer, in issue order.

    Each entry: column, channel, hold, ``fills`` (activation items of its column issued up to and including the
    one this transfer belongs to) and, for a held task, ``free_at``: the column's activation-item count at which
    its service is complete (None when the layer ends first, which the layer barrier settles)."""
    out = []
    fills = defaultdict(int)
    n = max((len(p) for p in programs), default=0)
    for i in range(n):
        for c, prog in enumerate(programs):
            if i >= len(prog):
                continue
            it = prog[i]
            kind = it[0]
            if kind in ("a", "A"):
                fills[c] += 1
                for _ in (it[1] if kind == "a" else [it[1]]):
                    out.append({"col": c, "channel": "a", "hold": 0, "fills": fills[c]})
            elif kind in ("w", "W", "o"):
                if kind == "w":
                    hold = it[3] if len(it) > 3 else 0
                else:
                    hold = it[2] if len(it) > 2 else 0
                out.append({"col": c, "channel": "o" if kind == "o" else "w", "hold": hold, "fills": fills[c]})
    for c in range(len(programs)):
        acts = sorted({e["fills"] for e in out if e["col"] == c and e["channel"] == "a"})
        for channel in ("w", "o"):
            held = [e for e in out if e["col"] == c and e["channel"] == channel and e["hold"]]
            for e in held:
                e["left"] = e["hold"]
                e["free_at"] = None
            for k in acts:
                for e in held:
                    if e["fills"] < k and e["left"] > 0:
                        e["left"] -= 1
                        if e["left"] == 0:
                            e["free_at"] = k
                        break
    return out


class _Task:
    def __init__(self, seq, entry, token):
        self.seq, self.entry, self.token = seq, entry, token
        self.col, self.channel = entry["col"], entry["channel"]
        self.awaited = self.freed = False


class RecordingEmitter(SequenceEmitter):
    """The emitter with its IRON calls recorded instead of emitted, every hardware rule checked as it runs."""

    def __init__(self, test, bd_budget, queue_depth):
        # SequenceEmitter.__init__ imports IRON and the design; the default path (no ring, no weight buffer)
        # reads only these names.
        self.t = test
        self.bd_budget, self.queue_depth = bd_budget, queue_depth
        self._names = {c: {ch: f"{ch}{c}" for ch in ("w", "a", "o")} for c in range(4)}
        self._await, self._free = self._on_await, self._on_free
        self.seq = 0
        self.live = defaultdict(list)
        self.order = defaultdict(list)
        self.issued_per_col = defaultdict(int)

    def start_layer(self, programs):
        self.expected = item_model(programs)
        self.cursor = 0
        self.final_fills = defaultdict(int)
        for e in self.expected:
            self.final_fills[e["col"]] = max(self.final_fills[e["col"]], e["fills"])

    def transfer(self, alloc, pattern, token):
        channel, col = alloc[0], int(alloc[1:])
        self.t.assertLess(self.cursor, len(self.expected), "more transfers than the items call for")
        exp = self.expected[self.cursor]
        self.t.assertEqual((exp["col"], exp["channel"]), (col, channel), "transfer out of the items' issue order")
        live = self.live[col]
        self.t.assertLess(len(live), self.bd_budget, f"column {col}: more than {self.bd_budget} live tasks")
        pending = sum(1 for x in live if x.channel == channel and not x.awaited)
        self.t.assertLess(pending, self.queue_depth, f"column {col} channel {channel}: queue overrun")
        task = _Task(self.seq, exp, token)
        self.seq += 1
        exp["task"] = task
        self.cursor += 1
        live.append(task)
        self.order[(col, channel)].append(task)
        self.issued_per_col[col] += 1
        return task

    def _fills_now(self, col):
        # An await happens inside ensure(), before the transfer it makes room for: that transfer's item has
        # already been served if it is an activation item. After the last transfer, at the layer barrier, every
        # item of the column has been issued.
        if self.cursor < len(self.expected) and self.expected[self.cursor]["col"] == col:
            return self.expected[self.cursor]["fills"]
        if self.cursor >= len(self.expected):
            return self.final_fills[col]
        prior = [e["fills"] for e in self.expected[:self.cursor] if e["col"] == col]
        return max(prior, default=0)

    def _on_await(self, task):
        self.t.assertTrue(task.token, "awaited a task that carries no completion token")
        self.t.assertFalse(task.awaited, "a token awaited twice")
        earlier = [x for x in self.order[(task.col, task.channel)] if x.seq < task.seq and x.token]
        self.t.assertTrue(all(x.awaited for x in earlier), "tokens awaited out of issue order on one channel")
        e = task.entry
        if e["hold"]:
            barrier = self.cursor >= len(self.expected)
            if not barrier:
                self.t.assertIsNotNone(e["free_at"], f"held {task.channel} task awaited mid-layer, never served")
                self.t.assertGreaterEqual(self._fills_now(task.col), e["free_at"],
                                          f"held {task.channel} task awaited before the fills it serves")
        task.awaited = True

    def _on_free(self, task):
        self.t.assertFalse(task.freed, "a task freed twice")
        later = [x for x in self.order[(task.col, task.channel)] if x.seq >= task.seq and x.awaited]
        self.t.assertTrue(later, "freed a task before any task at or after it on its channel was awaited")
        task.freed = True
        self.live[task.col].remove(task)


class EmitterOffline(unittest.TestCase):
    _cache = {}

    def _scheds(self, name):
        model = MODELS[name]
        if not model.exists():
            self.skipTest(f"{name} model not present")
        if name not in self._cache:
            ir = graph_ir.lower_yolov8n(model)
            ws = es.plan_workspace(ir)
            self._cache[name] = es.schedule_graph(ir, ws)[0]
        return self._cache[name]

    def _run(self, name, bd_budget=14, queue_depth=4, retire_batch=None):
        em = RecordingEmitter(self, bd_budget, queue_depth)
        tasks = 0
        for s in self._scheds(name):
            em.start_layer(s.programs)
            before = dict(em.issued_per_col)
            em.run_column_programs(s.programs, bd_budget=bd_budget, queue_depth=queue_depth,
                                   retire_batch=retire_batch)
            self.assertEqual(em.cursor, len(em.expected), f"{s.name}: transfers missing")
            for c, prog in enumerate(s.programs):
                self.assertEqual(em.issued_per_col[c] - before.get(c, 0), program_task_count(prog),
                                 f"{s.name} column {c}: task count")
            for e in em.expected:
                self.assertTrue(e["task"].freed, f"{s.name}: a task outlives its layer")
                if e["task"].token:
                    self.assertTrue(e["task"].awaited, f"{s.name}: a token left unconsumed")
            tasks += len(em.expected)
        return tasks

    def test_01_yolov8n_default(self):
        self.assertGreater(self._run("yolov8n"), 0)

    def test_02_yolov8s_default(self):
        self.assertGreater(self._run("yolov8s"), 0)

    def test_03_yolov8s_cadence_3(self):
        self._run("yolov8s", retire_batch=3)

    def test_04_yolov8s_tighter_budget(self):
        self._run("yolov8s", bd_budget=10)

    def test_05_a_hold_binds(self):
        # On the shipped YOLOv8n/s schedules no hold ever stops a retirement (counted 2026-10-13: zero refusals),
        # so they cannot show that held tasks wait for their fills. Here one weight task serves five activation
        # items under a three-descriptor budget: the budget fills while the weight task is still held, and the
        # emitter must retire activation tasks instead of awaiting it.
        from ignite_xdna.compiler.engine_sequence import linear
        prog = [("w", 0, 9472, 5)] + [("A", linear("ws", k * 25600, 25600)) for k in range(6)]
        prog.append(("o", linear("ws", 1 << 20, 3200), 0))
        em = RecordingEmitter(self, bd_budget=3, queue_depth=4)
        em.start_layer([prog])
        em.run_column_programs([prog], bd_budget=3, queue_depth=4)
        self.assertEqual(em.cursor, len(em.expected))
        self.assertTrue(all(e["task"].freed for e in em.expected))


if __name__ == "__main__":
    unittest.main()
