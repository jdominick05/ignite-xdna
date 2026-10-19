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
    B  contention: gemm2048 and attn512 as two concurrent processes, (1) each owning its own context, (2) both as
       broker clients at depth 4, streak 32, (3) the same with streak 1. Wall time of the pair, broker switches.

    bash scripts/research-iron.sh tools/npu_broker_probe.py run [--calls 200]

Written on Desktop 2, 2026-10-19; what has been run is in its logs.
"""
import argparse
import hashlib
import json
import os
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


def direct(name, calls):
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
    total = time.perf_counter() - t_all
    del bos, ib
    be.close(h)
    return dict(mode="direct", work=name, calls=calls, total_s=total, wall_ms=1e3 * float(np.median(wall)),
                call_ms=1e3 * float(np.median(dev)), sha=sha(res))


def client(name, calls, address, depth):
    sys.path.insert(0, str(SRC))
    from ignite_xdna.runtime.broker import BrokerClient
    xclbin, insts, ins, outs = workload(name)
    need = sum((a.nbytes + 4095) // 4096 * 4096 for a in ins) + sum(
        (int(np.prod(s)) * np.dtype(d).itemsize + 4095) // 4096 * 4096 for s, d in outs)
    with BrokerClient(address, slot_bytes=need, depth=depth) as c:
        c.run(xclbin, insts, ins, outs)   # warm-up: context, instructions, BOs
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
        total = time.perf_counter() - t0
        st = c.stats()["stats"]
    return dict(mode=f"broker d{depth}", work=name, calls=calls, total_s=total, wall_ms=1e3 * total / calls,
                call_ms=1e3 * float(np.median(calls_s)), sha=sha(res), broker=st)


def idle():
    return "No hardware contexts running" in subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"],
                                                            capture_output=True, text=True, timeout=60).stdout


def role(args):
    """Run this script as a child in ``args`` and return its JSON line."""
    env = dict(os.environ, PYTHONPATH=str(SRC))
    p = subprocess.Popen([sys.executable, __file__, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env)
    return p


def collect(p, timeout=600):
    out, err = p.communicate(timeout=timeout)
    lines = [ln for ln in out.splitlines() if ln.startswith("{")]
    if p.returncode or not lines:
        raise RuntimeError(f"child failed rc={p.returncode}: {err[-2000:]}")
    return json.loads(lines[-1])


class Broker:
    def __init__(self, streak):
        self.address = rf"\\.\pipe\ignite-broker-probe-{uuid.uuid4().hex}"
        env = dict(os.environ, PYTHONPATH=str(SRC))
        self.p = subprocess.Popen([sys.executable, "-m", "ignite_xdna.runtime.broker", "serve", "--address",
                                   self.address, "--streak", str(streak)], stdout=subprocess.PIPE, text=True,
                                  env=env)
        line = self.p.stdout.readline()
        if "listening" not in line:
            raise RuntimeError(f"broker did not start: {line!r}")

    def stop(self):
        sys.path.insert(0, str(SRC))
        from ignite_xdna.runtime.broker import shutdown
        shutdown(self.address)
        out = self.p.communicate(timeout=60)[0]
        return out.strip().splitlines()[-1] if out.strip() else ""


def show(r):
    b = r.get("broker")
    extra = f"; broker calls {b['calls']} switches {b['switches']} opens {b['opens']}" if b else ""
    print(f"  {r['mode']:10s} {r['work']:9s} {r['calls']} calls in {r['total_s'] * 1e3:8.1f} ms: per-call wall "
          f"{r['wall_ms']:7.3f} ms, device call {r['call_ms']:7.3f} ms, out {r['sha']}{extra}", flush=True)


def run(calls):
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
    print(f"PHASE B: contention, gemm2048 and attn512 at once, {calls} calls each", flush=True)
    ps = [role(["direct", "--work", w, "--calls", str(calls)]) for w in ("gemm2048", "attn512")]
    t0 = time.perf_counter()
    rs = [collect(p) for p in ps]
    print(f"  two direct processes: pair wall {1e3 * (time.perf_counter() - t0):.1f} ms (includes process "
          "start and warm-up)", flush=True)
    for r in rs:
        show(r)
    for streak in (32, 1):
        br = Broker(streak)
        try:
            ps = [role(["client", "--work", w, "--calls", str(calls), "--address", br.address, "--depth", "4"])
                  for w in ("gemm2048", "attn512")]
            t0 = time.perf_counter()
            rs = [collect(p) for p in ps]
            print(f"  broker streak {streak}: pair wall {1e3 * (time.perf_counter() - t0):.1f} ms (includes process "
                  "start and warm-up)", flush=True)
            for r in rs:
                show(r)
        finally:
            print(f"    {br.stop()}", flush=True)
    print(f"idle after: {idle()}", flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "direct", "client"])
    ap.add_argument("--work", default="gemm512")
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--address")
    ap.add_argument("--depth", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "direct":
        print(json.dumps(direct(a.work, a.calls)), flush=True)
        return 0
    if a.cmd == "client":
        print(json.dumps(client(a.work, a.calls, a.address, a.depth)), flush=True)
        return 0
    return run(a.calls)


if __name__ == "__main__":
    sys.exit(main())
