#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Device probe of src/ignite_xdna/runtime/broker.py: what one owner costs, and what it saves under contention.

Workloads (real kernels, seeded inputs; outputs compared bit for bit, not against any maths):
    gemm512    weight-resident generic GEMM 512x512x512   (scratch/egemm-bres, g_512_512_512.xclbin)
    gemm2048   the same xclbin, 2048x512x2048
    attn512    fused local attention, doc512 case          (scratch/egemma-attn-hd/doc512)

Phases, each in its own process(es), NPU idle checked first:
    A  overhead: each workload direct (in-process pyxrt: copy in, sync, call, sync, copy out) against the same calls
       through the broker at depth 1 and depth 4. Per-call wall and device-call medians; output SHA-256 must match.
    B  contention: gemm2048 and attn512 as two concurrent processes. Each child warms up (context, instructions,
       BOs, one call), prints READY and waits; both start on one "go", and the makespan is the go to the later
       child's last completion (perf_counter is system-wide on Windows). Arms, alternated over ``--rounds``:
       direct (each its own context, depth 1), broker depth 1 streak 32, broker depth 4 streak 32, broker depth 4
       streak 1. Depth 1 against direct isolates ownership and grouping from pipelining.
    C  stop paths: a depth-4 client killed mid-run, then a fresh client (bit-equal, broker healthy); Ctrl+Break to
       the broker mid-run (exits 0); NPU idle after each.

    bash scripts/research-iron.sh tools/npu_broker_probe.py run [--calls 200] [--rounds 3]

Written on Desktop 2, 2026-10-19; what has been run is in its logs.
"""
import argparse
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
GEMM = ROOT / "scratch/egemm-bres"
ATTN = ROOT / "scratch/egemma-attn-hd"
XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"


def workload(name):
    """-> (xclbin, insts path, inputs, outputs [(shape, dtype)])."""
    rng = np.random.default_rng(7)
    if name.startswith("gemm"):
        M, K, N = dict(gemm512=(512, 512, 512), gemm2048=(2048, 512, 2048))[name]
        bits = lambda *s: (rng.standard_normal(s).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        return (GEMM / "g_512_512_512.xclbin", GEMM / f"g_{M}_{K}_{N}.bin", [bits(M, K), bits(K, N)],
                [((M, N), np.float32)])
    sys.path.insert(0, str(ROOT / "kernels/embedgemma"))
    import attn_array as A
    q, k, v, L, W = A.case_data("doc512")
    qp, kv, order = A.pack(q, k, v, W)
    return (ATTN / "doc512/final.xclbin", ATTN / "doc512/insts.bin", [qp, kv],
            [((qp.shape[0] * A.OUTW,), np.uint32)])


def sha(arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def barrier(go):
    """With ``go``: report READY and block until the parent's go line."""
    if go:
        print("READY", flush=True)
        sys.stdin.readline()


def direct(name, calls, go=False):
    sys.path.insert(0, str(SRC))
    from ignite_xdna.runtime.broker import XrtBackend
    xclbin, insts, ins, outs = workload(name)
    be = XrtBackend()
    h = be.open(str(xclbin))
    words = insts.read_bytes()
    ib = be.instructions(h, words)
    sizes = [a.nbytes for a in ins] + [int(np.prod(s)) * np.dtype(d).itemsize for s, d in outs]
    bos = [be.buffer(h, i, n) for i, n in enumerate(sizes)]
    wall, dev, res = [], [], None
    if not be.call(h, ib, len(words), bos, 10000):   # warm-up call (BOs touched once)
        raise RuntimeError("warm-up dispatch timed out")
    barrier(go)
    t_all = time.perf_counter()
    for _ in range(calls):
        t0 = time.perf_counter()
        for bo, a in zip(bos, ins):
            be.view(bo, a.nbytes)[:] = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            be.to_device(bo)
        t1 = time.perf_counter()
        if not be.call(h, ib, len(words), bos, 10000):
            raise RuntimeError("dispatch timed out")
        t2 = time.perf_counter()
        res = []
        for bo, (s, d), n in zip(bos[len(ins):], outs, sizes[len(ins):]):
            be.from_device(bo)
            res.append(be.view(bo, n).copy().view(d).reshape(s))
        wall.append(time.perf_counter() - t0)
        dev.append(t2 - t1)
    t_end = time.perf_counter()
    del bos, ib
    be.close(h)
    return dict(mode="direct", work=name, calls=calls, total_s=t_end - t_all, wall_ms=1e3 * float(np.median(wall)),
                call_ms=1e3 * float(np.median(dev)), sha=sha(res), t_start=t_all, t_end=t_end)


def client(name, calls, address, depth, go=False):
    sys.path.insert(0, str(SRC))
    from ignite_xdna.runtime.broker import BrokerClient
    xclbin, insts, ins, outs = workload(name)
    need = sum((a.nbytes + 4095) // 4096 * 4096 for a in ins) + sum(
        (int(np.prod(s)) * np.dtype(d).itemsize + 4095) // 4096 * 4096 for s, d in outs)
    with BrokerClient(address, slot_bytes=need, depth=depth) as c:
        c.run(xclbin, insts, ins, outs)   # warm-up: context, instructions, BOs
        c.run(xclbin, insts, ins, outs)   # both BO sets
        barrier(go)
        t0 = time.perf_counter()
        tickets, res, calls_s = [], None, []
        for _ in range(calls):
            tickets.append(c.submit(xclbin, insts, ins, outs))
            if len(tickets) >= depth:
                res = c.wait(tickets.pop(0))
                calls_s.append(c.last["call_s"])
        while tickets:
            res = c.wait(tickets.pop(0))
            calls_s.append(c.last["call_s"])
        t_end = time.perf_counter()
        st = c.stats()["stats"]
    return dict(mode=f"broker d{depth}", work=name, calls=calls, total_s=t_end - t0,
                wall_ms=1e3 * (t_end - t0) / calls, call_ms=1e3 * float(np.median(calls_s)), sha=sha(res),
                broker=st, t_start=t0, t_end=t_end)


def idle():
    return "No hardware contexts running" in subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"],
                                                            capture_output=True, text=True, timeout=60).stdout


KEY = secrets.token_hex(32)   # this probe's broker key, handed to its broker and clients only
ENV = dict(os.environ, PYTHONPATH=str(SRC), IGNITE_BROKER_KEY=KEY)


def role(args):
    """Run this script as a child in ``args`` (stdin piped for the go line)."""
    return subprocess.Popen([sys.executable, __file__, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=ENV)


def wait_ready(p, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        line = p.stdout.readline()
        if not line:
            raise RuntimeError(f"child exited before READY: {p.stderr.read()[-2000:]}")
        if line.strip() == "READY":
            return
    raise RuntimeError("child not READY in time")


def collect(p, timeout=600):
    out, err = p.communicate(timeout=timeout)
    lines = [ln for ln in out.splitlines() if ln.startswith("{")]
    if p.returncode or not lines:
        raise RuntimeError(f"child failed rc={p.returncode}: {err[-2000:]}")
    return json.loads(lines[-1])


class Broker:
    def __init__(self, streak):
        self.address = rf"\\.\pipe\ignite-broker-probe-{uuid.uuid4().hex}"
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0   # so Ctrl+Break reaches it alone
        self.p = subprocess.Popen([sys.executable, "-m", "ignite_xdna.runtime.broker", "serve", "--address",
                                   self.address, "--streak", str(streak)], stdout=subprocess.PIPE, text=True,
                                  env=ENV, creationflags=flags)
        line = self.p.stdout.readline()
        if "listening" not in line:
            raise RuntimeError(f"broker did not start: {line!r}")

    def stop(self):
        sys.path.insert(0, str(SRC))
        from ignite_xdna.runtime.broker import shutdown
        shutdown(self.address, KEY.encode())
        out = self.p.communicate(timeout=60)[0]
        return out.strip().splitlines()[-1] if out.strip() else ""


def show(r):
    b = r.get("broker")
    extra = f"; broker calls {b['calls']} switches {b['switches']} opens {b['opens']}" if b else ""
    print(f"  {r['mode']:10s} {r['work']:9s} {r['calls']} calls in {r['total_s'] * 1e3:8.1f} ms: per-call wall "
          f"{r['wall_ms']:7.3f} ms, device call {r['call_ms']:7.3f} ms, out {r['sha']}{extra}", flush=True)


def run(calls, rounds):
    if not idle():
        print("NPU not idle: refusing (a foreign context is contention, not a finding)")
        return 3
    ok = True
    print(f"PHASE A: overhead, {calls} calls each", flush=True)
    for w in ("gemm512", "gemm2048", "attn512"):
        d = collect(role(["direct", "--work", w, "--calls", str(calls)]))
        show(d)
        br = Broker(32)
        try:
            for depth in (1, 4):
                r = collect(role(["client", "--work", w, "--calls", str(calls), "--address", br.address,
                                  "--depth", str(depth)]))
                show(r)
                same = r["sha"] == d["sha"]
                ok &= same
                print(f"    output vs direct: {'bit-equal' if same else 'DIFFERENT'}", flush=True)
        finally:
            print(f"    {br.stop()}", flush=True)
    print(f"PHASE B: contention, gemm2048 and attn512 at once, {calls} calls each, {rounds} rounds", flush=True)
    arms = [("direct", None, None), ("broker d1 s32", 1, 32), ("broker d4 s32", 4, 32), ("broker d4 s1", 4, 1)]
    spans = {a[0]: [] for a in arms}
    for rnd in range(rounds):
        for name, depth, streak in (arms if rnd % 2 == 0 else arms[::-1]):
            br = Broker(streak) if depth else None
            try:
                ps = [role(["direct", "--work", w, "--calls", str(calls), "--go"]) if not br else
                      role(["client", "--work", w, "--calls", str(calls), "--address", br.address, "--depth",
                            str(depth), "--go"]) for w in ("gemm2048", "attn512")]
                for p_ in ps:
                    wait_ready(p_)
                t_go = time.perf_counter()
                for p_ in ps:
                    p_.stdin.write("go\n")
                    p_.stdin.flush()
                rs = [collect(p_) for p_ in ps]
            finally:
                tail = br.stop() if br else ""
            span = max(r["t_end"] for r in rs) - t_go
            spans[name].append(span)
            same = rs[0]["sha"] == "cf90e28ca5fc81c0" and rs[1]["sha"] == "e6709b20972a105a"
            ok &= same
            sw = rs[0].get("broker", {}).get("switches")
            print(f"  round {rnd} {name:14s}: makespan {1e3 * span:7.1f} ms; gemm {1e3 * (rs[0]['t_end'] - t_go):7.1f}"
                  f", attn {1e3 * (rs[1]['t_end'] - t_go):7.1f} ms after go; device-call medians "
                  f"{rs[0]['call_ms']:.3f} / {rs[1]['call_ms']:.3f} ms; outputs {'as solo' if same else 'DIFFERENT'}"
                  + (f"; {tail}" if tail else ""), flush=True)
    for name, v in spans.items():
        print(f"  {name:14s}: makespan median {1e3 * float(np.median(v)):7.1f} ms over {len(v)} "
              f"(min {1e3 * min(v):.1f}, max {1e3 * max(v):.1f})", flush=True)

    print("PHASE C: stop paths", flush=True)
    br = Broker(32)
    try:
        victim = role(["client", "--work", "gemm2048", "--calls", "5000", "--address", br.address, "--depth", "4"])
        time.sleep(1.5)
        victim.kill()
        victim.communicate(timeout=60)
        r = collect(role(["client", "--work", "gemm2048", "--calls", "20", "--address", br.address, "--depth", "1"]))
        good = r["sha"] == "cf90e28ca5fc81c0"
        ok &= good
        print(f"  killed a depth-4 client mid-run (rc {victim.returncode}); next client: outputs "
              f"{'as solo' if good else 'DIFFERENT'}, broker calls {r['broker']['calls']}", flush=True)
    finally:
        print(f"    {br.stop()}", flush=True)
    print(f"  idle after: {idle()}", flush=True)
    br = Broker(32)
    busy = role(["client", "--work", "gemm2048", "--calls", "5000", "--address", br.address, "--depth", "4"])
    time.sleep(1.5)
    br.p.send_signal(signal.CTRL_BREAK_EVENT)
    try:
        out = br.p.communicate(timeout=60)[0]
        rc = br.p.returncode
    except subprocess.TimeoutExpired:
        br.p.kill()
        out, rc = br.p.communicate()[0], "timeout (killed)"
    busy.communicate(timeout=60)
    good = rc == 0
    ok &= good
    print(f"  Ctrl+Break to the broker mid-run: exit {rc}; client exit {busy.returncode} (a call error is expected); "
          f"{(out or '').strip().splitlines()[-1] if (out or '').strip() else ''}", flush=True)
    print(f"  idle after: {idle()}", flush=True)
    print(f"idle after: {idle()}", flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "direct", "client"])
    ap.add_argument("--work", default="gemm512")
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--address")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--go", action="store_true", help="child: READY after warm-up, start on a line from stdin")
    a = ap.parse_args()
    if a.cmd == "direct":
        print(json.dumps(direct(a.work, a.calls, a.go)), flush=True)
        return 0
    if a.cmd == "client":
        print(json.dumps(client(a.work, a.calls, a.address, a.depth, a.go)), flush=True)
        return 0
    return run(a.calls, a.rounds)


if __name__ == "__main__":
    sys.exit(main())
