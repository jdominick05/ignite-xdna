# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NPU broker: one process owns the device, client processes submit kernel calls to it.

Every NPU program here opens its own XRT hardware contexts. Two programs at once contend (a foreign context has
already turned measurements into ``ERT_CMD_STATE_TIMEOUT``), the driver allows five contexts per device (the sixth
fails with 0xc01e0009), and alternating calls between two contexts costs 1.0-1.4 ms a switch (BENCHMARKS,
weight-resident Stage A). The broker is the single owner:

- **Contexts** are cached per xclbin, at most ``max_contexts`` (default 4) open, least recently used evicted.
- **Instruction streams** are sent once per client and kept by the broker (by SHA-256), and uploaded once per
  context; after an eviction the broker re-uploads from its own copy.
- **Calls** are the mlir-aie kernel convention: ``kernel(3, instructions, size, *args)``, data args from group id 3.
  Inputs and outputs travel through one shared-memory arena per client (one copy in, one copy out).
- **Pipelining.** A client may have ``depth`` calls in flight (``submit`` / ``wait``), each in its own arena slot;
  replies come back in submission order.
- **Scheduling.** One device thread runs calls in order, except that it keeps running calls for the context it is
  already on (up to ``streak`` in a row) before switching, so interleaved clients do not pay a switch per call.
  Calls from one client always keep their order.
- **Overlap.** The device thread software-pipelines: while call k runs it copies call k+1's inputs in, and after
  starting k+1 it copies k's outputs out. Each (context, argument, size) has two BO sets, used in turn, so the two
  live calls never share a buffer. A call's ``call_s`` is start to observed completion, an upper bound on its
  device time.
- **Safety.** A call that does not complete within its timeout marks the broker unhealthy. After that it refuses
  new calls, because after a timed-out dispatch the device can refuse new contexts for minutes. ``close()``
  releases every BO and context.

Transport: ``multiprocessing.connection`` over a local named pipe. Messages are pickled, so a process that can
connect can run code inside the broker. The authkey keeps out processes that do not hold it: ``serve`` makes a
random one (unless ``IGNITE_BROKER_KEY`` is set) and writes it to ``%LOCALAPPDATA%/ignite-xdna/npu-broker.key``,
which clients read. That file inherits the profile's permissions; it is no defence against code already running as
the same user.

Stopping: the ``shutdown`` command, Ctrl+C or Ctrl+Break. Each lets the running call finish, then releases every
BO and context and exits 0. A client that disconnects mid-run (killed, or no ``bye``) keeps its arena mapped in
the broker until its own queued calls have finished. Usage::

    python -m ignite_xdna.runtime.broker serve [--address ADDR] [--max-contexts 4]

    from ignite_xdna.runtime.broker import BrokerClient
    with BrokerClient() as c:
        out, = c.run(xclbin, insts, [a, b], [(shape, np.uint16)])
    with BrokerClient(depth=8) as c:
        tickets = [c.submit(xclbin, insts, [a], [(shape, np.uint16)]) for a in batch]
        outs = [c.wait(t) for t in tickets]

The device backend is pluggable (``XrtBackend``; tests pass a fake), so the scheduling and bookkeeping are tested
offline. Written on Desktop 2 (2026-10-19); what has been measured on the device is in its logs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import queue
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from multiprocessing import connection, shared_memory
from pathlib import Path

import numpy as np


class BrokerError(RuntimeError):
    pass


DEFAULT_ADDRESS = r"\\.\pipe\ignite-xdna-npu-broker" if os.name == "nt" else "/tmp/ignite-xdna-npu-broker"
KEY_FILE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ignite-xdna" / "npu-broker.key"


def client_key() -> bytes:
    """The broker's authkey: IGNITE_BROKER_KEY, else the key file ``serve`` wrote."""
    if os.environ.get("IGNITE_BROKER_KEY"):
        return os.environ["IGNITE_BROKER_KEY"].encode()
    if KEY_FILE.is_file():
        return KEY_FILE.read_text(encoding="ascii").strip().encode()
    raise BrokerError(f"no broker key: set IGNITE_BROKER_KEY or start the broker (it writes {KEY_FILE})")


def server_key() -> bytes:
    if os.environ.get("IGNITE_BROKER_KEY"):
        return os.environ["IGNITE_BROKER_KEY"].encode()
    import secrets
    key = secrets.token_hex(32)
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(key, encoding="ascii")
    return key.encode()


# ---------------------------------------------------------------------------------------------------- device side

class XrtBackend:
    """pyxrt: contexts, instruction BOs and data BOs. Only the broker's device thread calls it."""

    def __init__(self):
        from .driver import setup_xrt_environment
        setup_xrt_environment()
        import pyxrt
        self.pyxrt = pyxrt
        self.dev = pyxrt.device(0)
        self.to = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.fr = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    def open(self, xclbin: str):
        xb = self.pyxrt.xclbin(xclbin)
        self.dev.register_xclbin(xb)
        ctx = self.pyxrt.hw_context(self.dev, xb.get_uuid())
        kern = self.pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
        return dict(xb=xb, ctx=ctx, kern=kern)

    def instructions(self, h, words: bytes):
        bo = self.pyxrt.bo(self.dev, len(words), self.pyxrt.bo.cacheable, h["kern"].group_id(1))
        bo.write(words, 0)
        bo.sync(self.to)
        return bo

    def buffer(self, h, index: int, nbytes: int):
        return self.pyxrt.bo(self.dev, nbytes, self.pyxrt.bo.host_only, h["kern"].group_id(3 + index))

    @staticmethod
    def view(bo, nbytes: int):
        return np.frombuffer(bo.map(), np.uint8)[:nbytes]

    def to_device(self, bo):
        bo.sync(self.to)

    def from_device(self, bo):
        bo.sync(self.fr)

    def start(self, h, ib, nwords_bytes: int, bos):
        return h["kern"](3, ib, nwords_bytes, *bos)

    def wait(self, run, timeout_ms: int) -> bool:
        return run.wait(timeout_ms) == self.pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED

    def call(self, h, ib, nwords_bytes: int, bos, timeout_ms: int) -> bool:
        return self.wait(self.start(h, ib, nwords_bytes, bos), timeout_ms)

    def close(self, h):
        import gc
        for k in ("kern", "ctx", "xb"):
            h.pop(k, None)
            gc.collect()


@dataclass
class _Context:
    key: str
    handle: dict
    insts: dict = field(default_factory=dict)     # sha256 -> (BO, bytes)
    buffers: dict = field(default_factory=dict)   # (arg index, nbytes, set 0/1) -> BO
    uses: int = 0


@dataclass
class _Call:
    client: int
    seq: int
    xclbin: str
    insts_sha: str
    insts: bytes | None
    args: list            # per data arg: ("in", arena offset, nbytes) or ("out", arena offset, nbytes) or ("io", ...)
    timeout_ms: int
    arena: shared_memory.SharedMemory
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None


class Broker:
    """The device owner. ``submit`` may be called from any thread; one device thread runs the calls."""

    def __init__(self, backend=None, max_contexts=4, streak=32):
        if not 1 <= max_contexts <= 4:
            raise ValueError("max_contexts 1..4 (the driver's fifth context is left for the system)")
        self.backend = backend if backend is not None else XrtBackend()
        self.max_contexts, self.streak = max_contexts, streak
        self.contexts: OrderedDict[str, _Context] = OrderedDict()
        self.streams: dict[str, bytes] = {}   # sha256 -> instruction stream, every one any client sent
        self.pending: deque[_Call] = deque()
        self.cv = threading.Condition()
        self.healthy, self.closing = True, False
        self.current, self.run_len = None, 0
        self.stats = dict(calls=0, switches=0, opens=0, evictions=0, insts_uploads=0, device_s=0.0, queue_s=0.0)
        self.thread = threading.Thread(target=self._loop, name="npu-broker", daemon=True)
        self.thread.start()

    # -- scheduling
    def submit(self, call: _Call):
        with self.cv:
            if not self.healthy:
                raise BrokerError("broker unhealthy after a timed-out dispatch: refusing new calls")
            if self.closing:
                raise BrokerError("broker closing")
            call.result = dict(t_submit=time.perf_counter())
            if call.insts is not None:
                if hashlib.sha256(call.insts).hexdigest() != call.insts_sha:
                    raise BrokerError("instruction stream does not match its hash")
                self.streams[call.insts_sha] = call.insts
            elif call.insts_sha not in self.streams:
                raise BrokerError("unknown instruction stream: send it with the first call that uses it")
            self.pending.append(call)
            self.cv.notify()

    def _pick(self) -> _Call:
        """Oldest call, unless a call for the current context can run without breaking any client's order and the
        streak allows it."""
        first = self.pending[0]
        if self.current is not None and self.run_len < self.streak and first.xclbin != self.current:
            blocked = set()   # clients with an earlier call still queued: their later calls must wait for it
            for c in self.pending:
                if c.xclbin == self.current and c.client not in blocked:
                    self.pending.remove(c)
                    return c
                blocked.add(c.client)
        return self.pending.popleft()

    def _loop(self):
        running = None   # the started call: dict from _prepare plus its run handle
        while True:
            nxt = None
            with self.cv:
                if running is None:
                    while not self.pending and not self.closing:
                        self.cv.wait()
                    if self.closing and not self.pending:
                        return
                if not self.healthy:   # after a timeout nothing more is started: fail what is queued
                    while self.pending:
                        c = self.pending.popleft()
                        c.result.update(ok=False, error="broker unhealthy after a timed-out dispatch")
                        c.done.set()
                elif self.pending:
                    nxt = self._pick()
            if nxt is not None:   # copy-in of the next call overlaps the running one
                try:
                    nxt = self._prepare(nxt, keep=running["call"].xclbin if running else None)
                except Exception as e:  # noqa: BLE001 - the error goes back to the caller
                    nxt.result.update(ok=False, error=f"{type(e).__name__}: {e}")
                    nxt.done.set()
                    nxt = None
            ok = running is None or self._wait(running)
            if nxt is not None:
                if self.healthy:
                    self._start(nxt)
                else:
                    nxt["call"].result.update(ok=False, error="broker unhealthy after a timed-out dispatch")
                    nxt["call"].done.set()
                    nxt = None
            if running is not None:   # copy-out of the finished call overlaps the next one
                if ok:
                    try:
                        self._finish(running)
                    except Exception as e:  # noqa: BLE001
                        running["call"].result.update(ok=False, error=f"{type(e).__name__}: {e}")
                running["call"].done.set()
            running = nxt

    # -- device work (device thread only)
    def _context(self, xclbin: str, keep=None) -> _Context:
        ctx = self.contexts.get(xclbin)
        if ctx is not None:
            self.contexts.move_to_end(xclbin)
            return ctx
        if len(self.contexts) >= self.max_contexts:
            victim = next(k for k in self.contexts if k != keep)   # never the running call's context
            old = self.contexts.pop(victim)
            old.insts.clear()
            old.buffers.clear()
            self.backend.close(old.handle)
            self.stats["evictions"] += 1
        ctx = _Context(xclbin, self.backend.open(xclbin))
        self.contexts[xclbin] = ctx
        self.stats["opens"] += 1
        return ctx

    def _prepare(self, call: _Call, keep=None) -> dict:
        t0 = time.perf_counter()
        call.result["queue_s"] = t0 - call.result["t_submit"]
        self.stats["queue_s"] += call.result["queue_s"]
        ctx = self._context(call.xclbin, keep)
        if call.insts_sha not in ctx.insts:
            words = self.streams[call.insts_sha]
            ctx.insts[call.insts_sha] = (self.backend.instructions(ctx.handle, words), len(words))
            self.stats["insts_uploads"] += 1
        ib, nbytes = ctx.insts[call.insts_sha]
        side = ctx.uses % 2
        ctx.uses += 1
        arena = np.frombuffer(call.arena.buf, np.uint8)
        bos = []
        for i, (kind, off, n) in enumerate(call.args):
            bo = ctx.buffers.get((i, n, side))
            if bo is None:
                bo = ctx.buffers[(i, n, side)] = self.backend.buffer(ctx.handle, i, n)
            if kind in ("in", "io"):
                self.backend.view(bo, n)[:] = arena[off:off + n]
                self.backend.to_device(bo)
            bos.append(bo)
        return dict(call=call, ctx=ctx, ib=ib, nbytes=nbytes, bos=bos, copy_s=time.perf_counter() - t0)

    def _start(self, p: dict):
        call = p["call"]
        if self.current != call.xclbin:
            if self.current is not None:
                self.stats["switches"] += 1
            self.current, self.run_len = call.xclbin, 0
        self.run_len += 1
        p["t1"] = time.perf_counter()
        p["run"] = self.backend.start(p["ctx"].handle, p["ib"], p["nbytes"], p["bos"])

    def _wait(self, p: dict) -> bool:
        call = p["call"]
        ok = self.backend.wait(p["run"], call.timeout_ms)
        p["t2"] = time.perf_counter()
        p.pop("run")
        if not ok:
            with self.cv:
                self.healthy = False
            call.result.update(ok=False, error=f"dispatch did not complete within {call.timeout_ms} ms: broker now "
                                                "unhealthy")
        return ok

    def _finish(self, p: dict):
        call = p["call"]
        t0 = time.perf_counter()
        arena = np.frombuffer(call.arena.buf, np.uint8)
        for bo, (kind, off, n) in zip(p["bos"], call.args):
            if kind in ("out", "io"):
                self.backend.from_device(bo)
                arena[off:off + n] = self.backend.view(bo, n)
        self.stats["calls"] += 1
        self.stats["device_s"] += p["t2"] - p["t1"]
        call.result.update(ok=True, call_s=p["t2"] - p["t1"], copy_s=p["copy_s"] + time.perf_counter() - t0)

    def close(self):
        with self.cv:
            self.closing = True
            self.cv.notify()
        self.thread.join()
        for ctx in self.contexts.values():
            ctx.insts.clear()
            ctx.buffers.clear()
            self.backend.close(ctx.handle)
        self.contexts.clear()


# ---------------------------------------------------------------------------------------------------- transport

def _serve_client(broker: Broker, conn, cid: int):
    """Reader: submits every call at once (a client may pipeline). Sender: replies in submission order."""
    arenas: dict[str, shared_memory.SharedMemory] = {}
    replies: queue.Queue = queue.Queue()   # _Call, a ready reply dict, or None (stop)

    def sender():
        while True:
            item = replies.get()
            if item is None:
                return
            if isinstance(item, _Call):
                item.done.wait()
                r = dict(item.result)
                r.pop("t_submit", None)
                r["seq"] = item.seq
                item = r
            try:
                conn.send(item)
            except OSError:
                return

    out = threading.Thread(target=sender, daemon=True)
    out.start()
    mine: list[_Call] = []   # this client's submitted calls: its arenas stay mapped until each has finished
    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                return
            op = msg.get("op")
            if op == "stats":
                replies.put(dict(ok=True, stats=dict(broker.stats), healthy=broker.healthy,
                                 contexts=list(broker.contexts)))
                continue
            if op == "bye":
                replies.put(dict(ok=True))
                return
            if op != "run":
                replies.put(dict(ok=False, error=f"unknown op {op!r}"))
                continue
            name = msg["arena"]
            if name not in arenas:
                arenas[name] = shared_memory.SharedMemory(name=name)
            call = _Call(cid, msg["seq"], str(Path(msg["xclbin"]).resolve()), msg["insts_sha"], msg.get("insts"),
                         msg["args"], int(msg.get("timeout_ms", 10000)), arenas[name])
            try:
                broker.submit(call)
            except BrokerError as e:
                replies.put(dict(ok=False, error=str(e), seq=msg["seq"]))
                continue
            mine.append(call)
            mine = [c for c in mine if not c.done.is_set()]
            replies.put(call)
    finally:
        for c in mine:
            c.done.wait()
        replies.put(None)
        out.join()
        for a in arenas.values():
            a.close()
        conn.close()


def serve(address=DEFAULT_ADDRESS, authkey=None, max_contexts=4, streak=32, backend=None,
          ready: threading.Event | None = None, stop: threading.Event | None = None, signals=False):
    """Run until ``shutdown``, ``stop`` is set, or (``signals``) Ctrl+C / Ctrl+Break. Returns the broker's stats."""
    authkey = authkey if authkey is not None else server_key()
    stop = stop if stop is not None else threading.Event()
    if signals:
        import signal
        if os.name == "nt":   # a background job's child inherits "ignore Ctrl+C": take it back
            import ctypes
            ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
        for sig in ("SIGINT", "SIGBREAK", "SIGTERM"):
            if hasattr(signal, sig):
                signal.signal(getattr(signal, sig), lambda *_: stop.set())
    broker = Broker(backend, max_contexts, streak)
    listener = connection.Listener(address, authkey=authkey)
    print(f"broker listening on {address} (max {max_contexts} contexts, streak {streak})", flush=True)

    def accept():
        ids = itertools.count(1)
        while not stop.is_set():
            try:
                conn = listener.accept()
                hello = conn.recv()
            except (OSError, EOFError, connection.AuthenticationError):
                continue
            if hello.get("op") == "shutdown":
                conn.send(dict(ok=True))
                conn.close()
                stop.set()
                return
            conn.send(dict(ok=True))
            threading.Thread(target=_serve_client, args=(broker, conn, next(ids)), daemon=True).start()

    threading.Thread(target=accept, name="npu-broker-accept", daemon=True).start()
    if ready:
        ready.set()
    try:
        while not stop.wait(0.2):   # short waits: a signal handler runs between them
            pass
    finally:
        broker.close()
        try:
            listener.close()
        except OSError:
            pass
        print(f"broker stopped: {broker.stats}", flush=True)
    return broker.stats


class BrokerClient:
    """One connection and one shared-memory arena of ``depth`` slots. ``submit`` returns a ticket and blocks only
    when ``depth`` calls are already in flight; ``wait(ticket)`` returns that call's arrays; ``run`` is both."""

    def __init__(self, address=DEFAULT_ADDRESS, authkey=None, slot_bytes=1 << 24, depth=1):
        self.conn = connection.Client(address, authkey=authkey if authkey is not None else client_key())
        self.conn.send(dict(op="hello"))
        self.conn.recv()
        self.depth, self.slot = depth, slot_bytes
        self.arena = shared_memory.SharedMemory(create=True, size=depth * slot_bytes)
        self.sent: set[str] = set()
        self.seq = itertools.count()
        self._insts: dict[str, tuple[bytes, str]] = {}
        self.inflight: deque = deque()   # (seq, specs, n_in, inout, outputs)
        self.done: dict[int, list] = {}
        self.replies: dict[int, dict] = {}
        self.last: dict = {}

    def _stream(self, insts) -> tuple[bytes, str]:
        if isinstance(insts, (str, Path)):
            key = str(Path(insts).resolve())
            if key not in self._insts:
                words = Path(key).read_bytes()
                self._insts[key] = (words, hashlib.sha256(words).hexdigest())
            return self._insts[key]
        words = bytes(insts)
        return words, hashlib.sha256(words).hexdigest()

    def _receive_one(self):
        seq, specs, n_in, inout, outputs = self.inflight.popleft()
        r = self.conn.recv()
        if r.get("seq") != seq:
            raise BrokerError(f"reply for call {r.get('seq')} while waiting for {seq}")
        self.replies[seq] = r
        if not r.get("ok"):
            self.done[seq] = BrokerError(r.get("error", "call failed"))
            return
        arena = np.frombuffer(self.arena.buf, np.uint8)
        res = []
        for (kind, o, n), a in zip(specs[n_in:n_in + len(inout)], inout):
            res.append(arena[o:o + n].copy().view(a.dtype).reshape(a.shape))
        for (kind, o, n), (sh, d) in zip(specs[n_in + len(inout):], outputs):
            res.append(arena[o:o + n].copy().view(d).reshape(sh))
        self.done[seq] = res

    def submit(self, xclbin, insts, inputs, outputs, inout=(), timeout_ms=10000) -> int:
        """inputs: arrays copied in; outputs: (shape, dtype); inout: arrays copied in and returned (after
        ``inputs`` in argument order, before ``outputs``). Returns a ticket for ``wait``."""
        words, sha = self._stream(insts)
        seq = next(self.seq)
        base = (seq % self.depth) * self.slot
        specs, off = [], base
        for kind, n in ([("in", a.nbytes) for a in inputs] + [("io", a.nbytes) for a in inout]
                        + [("out", int(np.prod(sh)) * np.dtype(d).itemsize) for sh, d in outputs]):
            specs.append((kind, off, n))
            off += (n + 4095) // 4096 * 4096
        if off - base > self.slot:
            raise BrokerError(f"call needs {off - base} B, slot is {self.slot} B: make the client with a larger "
                              "slot_bytes")
        while len(self.inflight) >= self.depth:   # this slot's previous call must have been read back
            self._receive_one()
        arena = np.frombuffer(self.arena.buf, np.uint8)
        for (kind, o, n), a in zip(specs, list(inputs) + list(inout)):
            arena[o:o + n] = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        msg = dict(op="run", seq=seq, xclbin=str(xclbin), insts_sha=sha, args=specs, arena=self.arena.name,
                   timeout_ms=timeout_ms)
        if sha not in self.sent:
            msg["insts"] = words
            self.sent.add(sha)
        self.conn.send(msg)
        self.inflight.append((seq, specs, len(inputs), list(inout), list(outputs)))
        return seq

    def wait(self, ticket: int) -> list:
        while ticket not in self.done:
            self._receive_one()
        self.last = self.replies.pop(ticket)
        res = self.done.pop(ticket)
        if isinstance(res, Exception):
            raise res
        return res

    def run(self, xclbin, insts, inputs, outputs, inout=(), timeout_ms=10000):
        return self.wait(self.submit(xclbin, insts, inputs, outputs, inout, timeout_ms))

    def drain(self):
        while self.inflight:
            self._receive_one()

    def stats(self):
        self.drain()
        self.conn.send(dict(op="stats"))
        return self.conn.recv()

    def close(self):
        try:
            self.drain()
            self.conn.send(dict(op="bye"))
            self.conn.recv()
        except (OSError, EOFError, BrokerError):
            pass
        self.conn.close()
        self.arena.close()
        self.arena.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def shutdown(address=DEFAULT_ADDRESS, authkey=None):
    c = connection.Client(address, authkey=authkey if authkey is not None else client_key())
    c.send(dict(op="shutdown"))
    c.recv()
    c.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["serve", "shutdown"])
    ap.add_argument("--address", default=DEFAULT_ADDRESS)
    ap.add_argument("--max-contexts", type=int, default=4)
    ap.add_argument("--streak", type=int, default=32)
    a = ap.parse_args(argv)
    if a.cmd == "shutdown":
        shutdown(a.address)
        return 0
    serve(a.address, max_contexts=a.max_contexts, streak=a.streak, signals=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
