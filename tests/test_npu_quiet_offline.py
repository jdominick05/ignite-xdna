"""Offline checks for the NPU quiet-request protocol (tools/npu_quiet.py). No device: xrt-smi is faked."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import npu_quiet  # noqa: E402


def test_hold_writes_one_file_per_pid_and_release_removes_only_its_own(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    me, other = os.getpid(), 2_147_483_001
    p = npu_quiet.hold("probe x", me)
    q = npu_quiet.hold("someone else", other)
    assert p.name == f"npu-quiet.{me}.request" and p != q
    assert json.loads(p.read_text())["pid"] == me
    assert npu_quiet.release(me) and not p.exists() and q.exists()
    assert not npu_quiet.release(me)                       # already gone
    q.write_text(json.dumps({"pid": me}))                  # a file under another pid's name naming this pid
    assert not npu_quiet.release(other)                    # is not that pid's to remove
    assert q.exists()


def test_requests_marks_dead_pids_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    npu_quiet.hold("live", os.getpid())
    npu_quiet.hold("dead", 2_147_483_003)
    alive = {rec["reason"]: ok for _, rec, ok in npu_quiet.requests()}
    assert alive == {"live": True, "dead": False}


def test_wait_idle_polls_until_the_holder_lets_go(monkeypatch):
    reports = iter(["|1234 |0 | busy"] * 3 + ["No hardware contexts running"])
    monkeypatch.setattr(npu_quiet.time, "sleep", lambda s: None)
    assert npu_quiet.IDLE in npu_quiet.wait_idle(5, lambda: next(reports))


def test_wait_idle_gives_up_at_the_deadline_with_the_busy_report(monkeypatch):
    clock = iter(range(0, 1000, 2))
    monkeypatch.setattr(npu_quiet.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(npu_quiet.time, "sleep", lambda s: None)
    assert npu_quiet.wait_idle(0, lambda: "busy") == "busy"
