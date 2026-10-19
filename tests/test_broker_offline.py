# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline checks of runtime/broker.py with a fake device: results through the real IPC path (blocking and
pipelined), the scheduler's same-context grouping without reordering a client's calls, LRU eviction with the
broker's own instruction re-upload, and the unhealthy latch after a timed-out dispatch. No PyXRT, no NPU."""
import hashlib
import os
import threading
import uuid
from pathlib import Path

import numpy as np
import pytest

from ignite_xdna.runtime import broker as B


class FakeBackend:
    """A context named ...add adds its two uint8 inputs; ...neg negates one. ``gate`` holds the first call."""

    def __init__(self, gate=None, fail=()):
        self.gate, self.fail = gate, set(fail)
        self.log, self.opened, self.closed = [], [], []

    def open(self, xclbin):
        self.opened.append(Path(xclbin).name)
        return dict(name=Path(xclbin).name)

    def instructions(self, h, words):
        return dict(words=words)

    def buffer(self, h, index, nbytes):
        return np.zeros(nbytes, np.uint8)

    @staticmethod
    def view(bo, nbytes):
        return bo[:nbytes]

    def to_device(self, bo):
        pass

    def from_device(self, bo):
        pass

    def start(self, h, ib, nbytes, bos):
        """Runs at wait(), so a later call's copy-in must not have touched this call's buffers by then."""
        return (h, [b.copy() for b in bos], bos)

    def wait(self, run, timeout_ms):
        h, snapshot, bos = run
        if self.gate is not None:
            self.gate.wait()
        self.log.append(h["name"])
        if h["name"] in self.fail:
            return False
        if any(not np.array_equal(a, b) for a, b in zip(snapshot, bos)):
            raise AssertionError("a buffer of a running call was overwritten")
        if h["name"].endswith("add"):
            bos[2][:] = bos[0] + bos[1]
        else:
            bos[1][:] = -bos[0]
        return True

    def close(self, h):
        self.closed.append(h["name"])


def _call(cid, xclbin, arena, a, b=None, insts=b"\x01\x02"):
    spec = [("in", 0, a.nbytes)]
    np.frombuffer(arena.buf, np.uint8)[:a.nbytes] = a
    if b is not None:
        spec.append(("in", 4096, b.nbytes))
        np.frombuffer(arena.buf, np.uint8)[4096:4096 + b.nbytes] = b
    spec.append(("out", 8192, a.nbytes))
    return B._Call(cid, 0, str(Path(xclbin).resolve()), hashlib.sha256(insts).hexdigest(), insts, spec, 1000, arena)


@pytest.fixture
def arena():
    m = B.shared_memory.SharedMemory(create=True, size=1 << 16)
    yield m
    m.close()
    m.unlink()


def test_ipc_roundtrip_results_and_stats():
    address = rf"\\.\pipe\ignite-broker-test-{uuid.uuid4().hex}" if os.name == "nt" else f"/tmp/ib-{uuid.uuid4().hex}"
    fake, ready = FakeBackend(), threading.Event()
    t = threading.Thread(target=B.serve, kwargs=dict(address=address, backend=fake, ready=ready), daemon=True)
    t.start()
    assert ready.wait(10)
    a = np.arange(100, dtype=np.uint8)
    b = np.full(100, 3, np.uint8)
    with B.BrokerClient(address, slot_bytes=1 << 16) as c:
        for _ in range(3):
            out, = c.run("ctx_add", b"insts-add", [a, b], [((100,), np.uint8)])
            assert np.array_equal(out, a + b)
        neg, = c.run("ctx_neg", b"insts-neg", [a], [((10, 10), np.uint8)])
        assert np.array_equal(neg, (-a).reshape(10, 10))
        s = c.stats()
        assert s["ok"] and s["stats"]["calls"] == 4 and s["stats"]["insts_uploads"] == 2
        assert s["stats"]["opens"] == 2 and s["stats"]["switches"] == 1
    rng = np.random.default_rng(0)
    batch = [rng.integers(0, 256, 64, dtype=np.uint8) for _ in range(20)]
    with B.BrokerClient(address, slot_bytes=1 << 14, depth=4) as c:
        tickets = [c.submit("ctx_neg", b"insts-neg", [x], [((64,), np.uint8)]) for x in batch]
        outs = [c.wait(tk) for tk in reversed(tickets)][::-1]   # waiting out of order still returns each its own
        assert all(np.array_equal(o, -x) for (o,), x in zip(outs, batch))
        with pytest.raises(B.BrokerError, match="slot"):
            c.submit("ctx_neg", b"insts-neg", [np.zeros(1 << 14, np.uint8)], [((1 << 14,), np.uint8)])
    B.shutdown(address)
    t.join(10)
    assert sorted(fake.closed) == ["ctx_add", "ctx_neg"]


def test_scheduler_groups_by_context_without_reordering_a_client(arena):
    gate = threading.Event()
    fake = FakeBackend(gate=gate)
    br = B.Broker(fake, max_contexts=4, streak=32)
    a = np.ones(8, np.uint8)
    calls = [_call(0, "warm_add", arena, a, a)]   # the device thread blocks in this one: everything else queues
    br.submit(calls[0])
    while not br.current:
        threading.Event().wait(0.01)
    # client 1: x_add, y_neg; client 2: x_add; client 3: y_neg; client 4: x_add
    order = [(1, "x_add"), (1, "y_neg"), (2, "x_add"), (3, "y_neg"), (4, "x_add")]
    for cid, x in order:
        c = _call(cid, x, arena, a, a if x.endswith("add") else None)
        calls.append(c)
        br.submit(c)
    gate.set()
    for c in calls:
        assert c.done.wait(10) and c.result["ok"], c.result
    br.close()
    # after warm_add (current), the oldest is x_add; then x_add calls that no earlier call of the same client blocks
    # (clients 2 and 4) run before switching; client 1's y_neg runs after its x_add.
    assert fake.log == ["warm_add", "x_add", "x_add", "x_add", "y_neg", "y_neg"]
    assert br.stats["switches"] == 2


def test_streak_limit_forces_a_switch(arena):
    gate = threading.Event()
    fake = FakeBackend(gate=gate)
    br = B.Broker(fake, max_contexts=4, streak=2)
    a = np.ones(8, np.uint8)
    first = _call(0, "p_add", arena, a, a)
    br.submit(first)
    while not br.current:
        threading.Event().wait(0.01)
    calls = [first] + [_call(1, "q_neg", arena, a)] + [_call(10 + i, "p_add", arena, a, a) for i in range(3)]
    for c in calls[1:]:
        br.submit(c)
    gate.set()
    for c in calls:
        assert c.done.wait(10)
    br.close()
    assert fake.log == ["p_add", "p_add", "q_neg", "p_add", "p_add"]


def test_lru_eviction_and_instruction_resend(arena):
    fake = FakeBackend()
    br = B.Broker(fake, max_contexts=2)
    a = np.ones(8, np.uint8)
    for x in ("a_neg", "b_neg", "c_neg"):
        c = _call(0, x, arena, a)
        br.submit(c)
        assert c.done.wait(10) and c.result["ok"]
    assert fake.closed == ["a_neg"] and br.stats["evictions"] == 1
    c = _call(0, "a_neg", arena, a)
    c.insts = None   # sent once already: the broker re-uploads its own copy into the reopened context
    br.submit(c)
    assert c.done.wait(10) and c.result["ok"], c.result
    assert br.stats["insts_uploads"] == 4 and fake.closed == ["a_neg", "b_neg"]
    with pytest.raises(B.BrokerError, match="unknown instruction stream"):
        unknown = _call(0, "a_neg", arena, a, insts=b"never sent")
        unknown.insts = None
        br.submit(unknown)
    br.close()


def test_timeout_latches_unhealthy(arena):
    fake = FakeBackend(fail={"bad_neg"})
    br = B.Broker(fake)
    a = np.ones(8, np.uint8)
    c = _call(0, "bad_neg", arena, a)
    br.submit(c)
    assert c.done.wait(10) and not c.result["ok"] and "unhealthy" in c.result["error"]
    with pytest.raises(B.BrokerError, match="unhealthy"):
        br.submit(_call(0, "ok_neg", arena, a))
    br.close()
