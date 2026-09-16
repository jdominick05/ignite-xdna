"""Offline tests for pipelines/npu_power.py: when the NPU's power mode is lowered, and that it is always put back.

No NPU and no xrt-smi: a fake runner answers `examine -r platform`, `examine -r aie-partitions` and `configure --pmode`
with the report formats xrt-smi printed on Desktop 2 (2026-09-16).

    python -m pytest -q tests/test_npu_power_offline.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ignite_npu_power", ROOT / "src" / "ignite_xdna" / "pipelines" / "npu_power.py")
npu_power = importlib.util.module_from_spec(_spec)
sys.modules["ignite_npu_power"] = npu_power
_spec.loader.exec_module(npu_power)

PLATFORM = """
-----------------------------
[003d:00:01.1] : NPU Phoenix
-----------------------------
Platform
  Name                   : NPU Phoenix
  Power Mode             : {mode}
  Total Columns          : 5

Estimated Power          : N/A
"""
NO_CONTEXTS = """
-----------------------------
[003d:00:01.1] : NPU Phoenix
-----------------------------
AIE Partitions
  No hardware contexts running on device
"""
CONTEXT_HEADER = """
-----------------------------
[003d:00:01.1] : NPU Phoenix
-----------------------------
AIE Partitions
  Total Memory Usage: 92 MB
  Partition Index   : 0
    Columns: [1, 2, 3, 4]
    HW Contexts:
      |PID                 |Ctx ID     |Submissions |Migrations  |Err  |Priority |
      |Process Name        |Status     |Completions |Suspensions |     |GOPS     |
      |Memory Usage        |Instr BO   |            |            |     |FPS      |
      |                    |           |            |            |     |Latency  |
      |====================|===========|============|============|=====|=========|
"""
CONTEXT_ROW = """      |{pid:<20}|159        |655         |0           |0    |Normal   |
      |python.exe          |Active     |654         |0           |     |N/A      |
      |92 MB               |640 KB     |            |            |     |N/A      |
      |                    |           |            |            |     |N/A      |
      |--------------------|-----------|------------|------------|-----|---------|
"""


class FakeSmi:
    def __init__(self, mode="Default", pids=(), fail_set=False):
        self.mode, self.pids, self.fail_set, self.calls = mode, list(pids), fail_set, []

    def __call__(self, args):
        self.calls.append(list(args))
        if args[:3] == ["examine", "-r", "platform"]:
            return 0, PLATFORM.format(mode=self.mode)
        if args[:3] == ["examine", "-r", "aie-partitions"]:
            if not self.pids:
                return 0, NO_CONTEXTS
            return 0, CONTEXT_HEADER + "".join(CONTEXT_ROW.format(pid=p) for p in self.pids)
        if args[:2] == ["configure", "--pmode"]:
            if not self.fail_set:
                self.mode = args[2].capitalize()
            return 0, ""
        return 1, "unknown"


def governor(smi, lease, mode="efficiency", fps=30.0, **kw):
    return npu_power.NpuPowerGovernor(mode, 1.0 / fps if fps else 0.0, enabled=kw.pop("enabled", True), runner=smi,
                                      lease=lease, watch_interval_s=0, log=lambda s: None, **kw)


class ParsingTest(unittest.TestCase):
    def test_mode_and_contexts(self):
        smi = FakeSmi(mode="Powersaver", pids=[3664, 4120])
        self.assertEqual(npu_power.read_mode(smi), "powersaver")
        self.assertEqual(npu_power.context_pids(smi), [3664, 4120])
        self.assertEqual(npu_power.context_pids(FakeSmi()), [])

    def test_unavailable_xrt_smi(self):
        missing = lambda args: (127, "FileNotFoundError")  # noqa: E731
        self.assertIsNone(npu_power.read_mode(missing))
        self.assertIsNone(npu_power.context_pids(missing))


class DecisionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lease = Path(self.tmp.name) / "lease.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_yolov8n_at_30_fps_lowers_and_restores(self):
        smi = FakeSmi(pids=[os.getpid()])  # only this process's own context
        g = governor(smi, self.lease)
        # measured YOLOv8n efficiency: G2G 9.52 ms of which dispatch 7.24 ms -> predicted 2.28 + 15.86 = 18.1 ms
        self.assertTrue(g.consider(cpu_ms=2.28, dispatch_ms=7.24, frame=50))
        self.assertEqual(smi.mode, "Powersaver")
        self.assertTrue(self.lease.exists())
        g.restore("test")
        self.assertEqual(smi.mode, "Default")
        self.assertFalse(self.lease.exists())
        self.assertEqual(g.state, "restored")
        g.restore("again")  # idempotent
        self.assertEqual(g.state, "restored")

    def test_yolov8s_at_30_fps_does_not_fit(self):
        smi = FakeSmi()
        g = governor(smi, self.lease)
        self.assertFalse(g.consider(cpu_ms=1.25, dispatch_ms=16.77))  # predicted 38.0 ms > 26.7 ms
        self.assertEqual(smi.mode, "Default")
        self.assertIn("does not fit", g.reason)

    def test_only_efficiency_and_only_paced_and_not_when_off(self):
        for kw, text in (({"mode": "balanced"}, "leaves the NPU"), ({"fps": 0.0}, "not paced"),
                         ({"enabled": False}, "=off")):
            smi = FakeSmi()
            g = governor(smi, self.lease, **kw)
            self.assertFalse(g.consider(2.28, 7.24))
            self.assertEqual(smi.mode, "Default")
            self.assertIn(text, g.reason)
            self.assertFalse([c for c in smi.calls if c[:1] == ["configure"]])

    def test_a_mode_someone_chose_is_left_alone(self):
        smi = FakeSmi(mode="Balanced")
        g = governor(smi, self.lease)
        self.assertFalse(g.consider(2.28, 7.24))
        self.assertEqual(smi.mode, "Balanced")
        self.assertIn("already in balanced", g.reason)

    def test_another_process_on_the_npu_blocks_and_later_restores(self):
        smi = FakeSmi(pids=[os.getpid(), 999999])
        g = governor(smi, self.lease)
        self.assertFalse(g.consider(2.28, 7.24))
        self.assertIn("999999", g.reason)
        smi2 = FakeSmi(pids=[os.getpid()])
        g2 = npu_power.NpuPowerGovernor("efficiency", 1 / 30, enabled=True, runner=smi2,
                                        lease=Path(self.tmp.name) / "lease2.json", watch_interval_s=0.05,
                                        log=lambda s: None)
        self.assertTrue(g2.consider(2.28, 7.24))
        smi2.pids.append(424242)  # another application opens a context; the watcher must notice
        for _ in range(100):
            if g2.state == "restored":
                break
            import time
            time.sleep(0.02)
        self.assertEqual(g2.state, "restored")
        self.assertEqual(smi2.mode, "Default")
        self.assertIn("424242", g2.reason)

    def test_slow_frames_after_lowering_put_the_mode_back(self):
        smi = FakeSmi()
        g = governor(smi, self.lease)
        self.assertTrue(g.consider(2.28, 7.24))
        for _ in range(npu_power.CHECK_FRAMES):
            g.observe(31.0)  # P95 31 ms > 90 % of 33.3 ms
        self.assertEqual(smi.mode, "Default")
        self.assertIn("P95", g.reason)

    def test_fast_frames_after_lowering_keep_it(self):
        smi = FakeSmi()
        g = governor(smi, self.lease)
        self.assertTrue(g.consider(2.28, 7.24))
        for _ in range(npu_power.CHECK_FRAMES):
            g.observe(18.1)
        self.assertEqual(smi.mode, "Powersaver")
        self.assertEqual(g.state, "lowered")
        g.restore()

    def test_restore_leaves_a_mode_changed_by_someone_else(self):
        smi = FakeSmi()
        g = governor(smi, self.lease)
        self.assertTrue(g.consider(2.28, 7.24))
        smi.mode = "Performance"
        g.restore()
        self.assertEqual(smi.mode, "Performance")
        self.assertFalse(self.lease.exists())

    def test_a_failed_set_keeps_default_and_no_lease(self):
        smi = FakeSmi(fail_set=True)
        g = governor(smi, self.lease)
        self.assertFalse(g.consider(2.28, 7.24))
        self.assertIn("could not set", g.reason)
        self.assertFalse(self.lease.exists())


class LeaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lease = Path(self.tmp.name) / "lease.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _dead_pid(self):
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        return p.pid

    def test_a_dead_process_lease_is_restored(self):
        self.lease.write_text('{"pid": %d, "start_time": 1, "previous": "default", "set": "powersaver"}'
                              % self._dead_pid(), encoding="utf-8")
        smi = FakeSmi(mode="Powersaver")
        msg = npu_power.recover_stale_lease(smi, self.lease)
        self.assertIn("restored", msg)
        self.assertEqual(smi.mode, "Default")
        self.assertFalse(self.lease.exists())

    def test_a_governor_recovers_on_creation(self):
        self.lease.write_text('{"pid": %d, "start_time": 1, "previous": "default", "set": "powersaver"}'
                              % self._dead_pid(), encoding="utf-8")
        smi = FakeSmi(mode="Powersaver")
        g = governor(smi, self.lease)
        self.assertIn("restored", g.recovered)
        self.assertEqual(smi.mode, "Default")

    def test_a_live_process_lease_is_left(self):
        start = npu_power.process_start_time(os.getppid())
        self.lease.write_text('{"pid": %d, "start_time": %s, "previous": "default", "set": "powersaver"}'
                              % (os.getppid(), "null" if start is None else start), encoding="utf-8")
        smi = FakeSmi(mode="Powersaver")
        self.assertIsNone(npu_power.recover_stale_lease(smi, self.lease))
        self.assertEqual(smi.mode, "Powersaver")
        self.assertTrue(self.lease.exists())

    def test_a_reused_pid_does_not_protect_a_lease(self):
        start = npu_power.process_start_time(os.getppid())
        self.lease.write_text('{"pid": %d, "start_time": %d, "previous": "default", "set": "powersaver"}'
                              % (os.getppid(), (start or 5) + 12345), encoding="utf-8")
        smi = FakeSmi(mode="Powersaver")
        self.assertIn("restored", npu_power.recover_stale_lease(smi, self.lease))

    def test_a_lease_whose_mode_was_changed_is_dropped_without_touching_the_device(self):
        self.lease.write_text('{"pid": %d, "start_time": 1, "previous": "default", "set": "powersaver"}'
                              % self._dead_pid(), encoding="utf-8")
        smi = FakeSmi(mode="Balanced")
        self.assertIn("dropped", npu_power.recover_stale_lease(smi, self.lease))
        self.assertEqual(smi.mode, "Balanced")
        self.assertFalse([c for c in smi.calls if c[:1] == ["configure"]])


if __name__ == "__main__":
    unittest.main()
