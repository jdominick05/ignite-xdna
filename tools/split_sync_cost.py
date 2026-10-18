#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What does a DirectML + NPU split pay to join the two chips on every GEMV? (LLM study, R5 follow-up)

Stage 2 found the chips' reads add (DirectML + NPU 91.76 GB/s), and R5, the NPU's bandwidth role in
a split, printed OPEN at 1.1004x. The user's decision: measure the synchronization first. In decode,
every GEMV needs the previous one's output, so a split joins the two chips through the host once per
GEMV, 224 times per 7B token. Nothing crosses between the two devices without the host. This measures
that join with trivial kernels, because the join is the cost, not the math:
  npu  a 32 KiB shim -> mem tile -> shim passthrough (the same shape as BENCHMARKS' dispatch-floor
       table), raw pyxrt in this process: write x, sync, start, wait, sync back, read y
  dml  ONNX Runtime DirectML MatMul, x[1,4096] fp16 by W[4096,64], in a worker process
No Python on this machine has both: pyxrt is built for CPython 3.13 (ironenv, whose ONNX Runtime has
no DirectML) and 3.10, and every DirectML-capable environment is 3.12. So the DirectML half runs in a
resnet_env17 worker, and the two halves meet through shared memory with busy-wait flags. The cost of
that handshake alone is measured too (chain e) and subtracted, which favours the split.

Chains, each 224 dependent steps (the next input is built from this step's output):
  a  NPU round trip alone
  b  DirectML round trip alone (timed inside the worker, no handshake)
  e  handshake alone: the worker echoes x back with no DirectML
  c  the join: signal the worker (DirectML starts), run the NPU round trip, wait for the worker,
     combine the two outputs into the next input

    bash scripts/research-iron.sh tools/split_sync_cost.py build     # compile the NPU passthrough
    python tools/split_sync_cost.py prereg                            # the pre-registration text
    bash scripts/research-iron.sh tools/split_sync_cost.py suite     # the sitting (needs SPLIT_WORKER_PYTHON)
    python tools/split_sync_cost.py verdict <suite log>               # the mechanical verdict
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
BUILD = ROOT / "build" / "split_sync_probe"
WORDS = 8192                       # 32 KiB each way, as the dispatch-floor table's passthrough
CHUNK = 1024                       # mem-tile buffer, words; WORDS is a multiple of it
K, N_DML = 4096, 64                # the DirectML half: x[1,K] fp16 by W[K,N_DML]
STEPS = 224                        # GEMVs per 7B token: 32 x (4 + 2 + 1)
STAGE_JOINS = 128                  # dependent stages per token if independent GEMVs share a join:
                                   # 32 x (QKV, O, gate+up, down)
CHAINS, WARMUP_CHAINS = 30, 3
WAIT_MS = 5000
SEED = 20260923

# Pre-registered constants
SAVING_MS = 11.1                   # stage 2's DERIVED maximum saving of a DirectML + NPU split over
                                   # DirectML alone at reduction rates: 47.5 - 36.4 ms per token
KILL_US = SAVING_MS * 1e3 / STEPS  # 49.55 us per join
KILL_STAGE_US = SAVING_MS * 1e3 / STAGE_JOINS   # 86.7 us per join
CPP_SINGLE_US = 108.0              # the repo's C++ host, one dependent dispatch (MEASURED, BENCHMARKS)

# shared memory layout (bytes)
OFF_REQ, OFF_MODE, OFF_DONE, OFF_X = 0, 8, 16, 64
OFF_Y = OFF_X + K * 2
SHM_BYTES = OFF_Y + N_DML * 2
MODE_DML, MODE_ECHO, MODE_QUIT = 0, 1, 2


def design(n: int) -> str:
    """shim MM2S -> mem tile (two CHUNK buffers, lock handshake) -> shim S2MM, one column."""
    L = ["module { aie.device(npu1) {", "%s = aie.tile(0, 0)", "%m = aie.tile(0, 1)",
         "aie.flow(%s, DMA : 0, %m, DMA : 0)", "aie.flow(%m, DMA : 0, %s, DMA : 0)",
         "aie.shim_dma_allocation @in(%s, MM2S, 0)", "aie.shim_dma_allocation @out(%s, S2MM, 0)"]
    for j in range(2):
        L += [f"%b{j} = aie.buffer(%m) {{address = {j * CHUNK * 4} : i32}} : memref<{CHUNK}xi32>",
              f"%f{j} = aie.lock(%m, {2 * j}) {{init = 1 : i32}}",
              f"%r{j} = aie.lock(%m, {2 * j + 1}) {{init = 0 : i32}}"]
    L += ["aie.memtile_dma(%m) {", "%one = arith.constant 1 : i32",
          'aie.dma_start("S2MM", 0, ^in0, ^out)', "^out:", 'aie.dma_start("MM2S", 0, ^out0, ^end)']
    for d, acq, rel in (("in", "f", "r"), ("out", "r", "f")):
        for j in range(2):
            L += [f"^{d}{j}:", f"aie.use_lock(%{acq}{j}, AcquireGreaterEqual, %one)",
                  f"aie.dma_bd(%b{j} : memref<{CHUNK}xi32> offset = 0 len = {CHUNK})",
                  f"aie.use_lock(%{rel}{j}, Release, %one)", f"aie.next_bd ^{d}{1 - j}"]
    L += ["^end:", "aie.end", "}",
          f"aie.runtime_sequence(%x: memref<{n}xi32>, %y: memref<{n}xi32>) {{",
          "%in = aiex.dma_configure_task_for @in {", f"aie.dma_bd(%x : memref<{n}xi32> offset = 0 len = {n})",
          "aie.end", "}", "%out = aiex.dma_configure_task_for @out {",
          f"aie.dma_bd(%y : memref<{n}xi32> offset = 0 len = {n})", "aie.end", "} {issue_token = true}",
          "aiex.dma_start_task(%out)", "aiex.dma_start_task(%in)", "aiex.dma_await_task(%out)",
          "aiex.dma_free_task(%in)", "aiex.dma_free_task(%out)", "}", "}}"]
    return "\n".join(L)


def build():
    from aie.utils.compile.utils import compile_mlir_module
    d = BUILD / f"passthrough_w{WORDS}_c{CHUNK}"
    if not (d / "insts.bin").exists():
        (d / "design.prj").mkdir(parents=True, exist_ok=True)
        module = design(WORDS)
        (d / "probe.mlir").write_text(module, encoding="utf-8")
        compile_mlir_module(module, insts_path=d / "insts.bin", xclbin_path=d / "probe.xclbin", work_dir=d / "design.prj")
    print("ARTIFACT", d.name, "insts_sha256", hashlib.sha256((d / "insts.bin").read_bytes()).hexdigest(),
          "mlir_sha256", hashlib.sha256((d / "probe.mlir").read_bytes()).hexdigest(), flush=True)
    return d


# ---------------------------------------------------------------- the DirectML worker (resnet_env17)

def serve(shm, run_dml) -> None:
    """Answer the coordinator's joins until it sends MODE_QUIT; ctl[2] = ctl[0] is the reply."""
    ctl = np.ndarray((3,), dtype=np.int64, buffer=shm.buf, offset=0)
    xs = np.ndarray((1, K), dtype=np.float16, buffer=shm.buf, offset=OFF_X)
    ys = np.ndarray((1, N_DML), dtype=np.float16, buffer=shm.buf, offset=OFF_Y)
    last = 0
    while True:
        while ctl[0] == last:
            pass
        last = int(ctl[0])
        mode = int(ctl[1])
        if mode == MODE_QUIT:
            ctl[2] = last
            return
        if mode == MODE_DML:
            ys[...] = run_dml(xs)
        else:
            ys[0] = xs[0, :N_DML]
        ctl[2] = last


def dry_worker(shm_name: str) -> int:
    """selftest only: the handshake with no DirectML, touching no chip."""
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        print("READY", sys.version.split()[0], flush=True)
        serve(shm, None)
    finally:
        shm.close()
    return 0


def dml_worker(shm_name: str) -> int:
    import onnx
    import onnxruntime as ort
    from multiprocessing import shared_memory
    from onnx import TensorProto, helper
    rng = np.random.default_rng(SEED)
    w = (rng.standard_normal((K, N_DML)) * 1e-3).astype(np.float16)
    g = helper.make_graph([helper.make_node("MatMul", ["x", "W"], ["y"])], "join",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, K])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, N_DML])],
                          [onnx.numpy_helper.from_array(w, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.enable_mem_pattern = False
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=[("DmlExecutionProvider", {"device_id": 0})])
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        x = (rng.standard_normal((1, K)) * 1e-2).astype(np.float16)
        for _ in range(20):
            sess.run(None, {"x": x})
        # chain b: DirectML round trips alone, no handshake
        chains = []
        for c in range(WARMUP_CHAINS + CHAINS):
            t0 = time.perf_counter_ns()
            for _ in range(STEPS):
                y = sess.run(None, {"x": x})[0]
                x[0, :N_DML] = y[0]
            t1 = time.perf_counter_ns()
            if c >= WARMUP_CHAINS:
                chains.append((t1 - t0) / 1e3)
        print("B_JSON " + json.dumps({"chain_us": chains, "providers": sess.get_providers(),
                                      "onnxruntime": ort.__version__, "python": sys.version.split()[0],
                                      "finite": bool(np.isfinite(x).all())}), flush=True)
        print("READY", flush=True)
        serve(shm, lambda xs: sess.run(None, {"x": xs})[0])
    finally:
        shm.close()
    return 0


# ---------------------------------------------------------------- the coordinator (ironenv, owns the NPU)

def suite() -> int:
    from multiprocessing import shared_memory
    from concurrent_read_bw import gpu_engines
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    from silicon_probe_record import witness
    gpu_engines()                                             # VS Code shares the 780M with DirectML
    host = subprocess.run(["powershell", "-NoProfile", "-File", str(ROOT / "tools/host_load.ps1")],
                          capture_output=True, text=True)
    summary = next(s for s in host.stdout.splitlines() if s.startswith("HOST_LOAD busy_cores="))
    peers = [float(v) for v in re.findall(r"HOST_LOAD_PEER ([\d.]+)", host.stdout)]
    print(summary, "peer_busy_cores", peers, flush=True)
    assert float(re.search(r"busy_cores=([\d.]+)", summary)[1]) < 2, "host busy"
    assert max(peers, default=0) < .5, "a heavy peer holds the CPU"
    worker_py = os.environ.get("SPLIT_WORKER_PYTHON")
    assert worker_py and Path(worker_py).exists(), "set SPLIT_WORKER_PYTHON to resnet_env17's python.exe"
    print("WORKER_ENV", os.environ.get("SPLIT_WORKER_ENV", "?"), "COORDINATOR_PYTHON", sys.version.split()[0], flush=True)
    d = BUILD / f"passthrough_w{WORDS}_c{CHUNK}"
    print("ARTIFACT", d.name, "insts_sha256", hashlib.sha256((d / "insts.bin").read_bytes()).hexdigest(), flush=True)
    nbytes = WORDS * 4
    shm = shared_memory.SharedMemory(create=True, size=SHM_BYTES)
    ctl = np.ndarray((3,), dtype=np.int64, buffer=shm.buf, offset=0)
    xs = np.ndarray((1, K), dtype=np.float16, buffer=shm.buf, offset=OFF_X)
    ys = np.ndarray((1, N_DML), dtype=np.float16, buffer=shm.buf, offset=OFF_Y)
    ctl[:] = 0
    worker = None
    try:
        witness()
        with XrtSiliconHarness(0) as h:
            xrt = h.pyxrt
            h.load_xclbin(str(d / "probe.xclbin"), "MLIR_AIE")
            bo = {}
            bo["instr"], count = h.create_instruction_bo_from_bytes((d / "insts.bin").read_bytes())
            bo["inp"], bo["out"] = h.create_host_bo(nbytes, 3), h.create_host_bo(nbytes, 4)
            bo["run"] = xrt.run(h.kernel)
            for i, val in enumerate((3, bo["instr"], count, bo["inp"], bo["out"])):
                bo["run"].set_arg(i, val)
            to_dev, from_dev = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            done = xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
            rng = np.random.default_rng(SEED)
            x = np.zeros(WORDS, dtype=np.int32)
            x.view(np.float16)[:] = (rng.standard_normal(2 * WORDS) * 1e-2).astype(np.float16)
            mismatches = 0

            def npu_step(v):
                bo["inp"].write(v, 0)
                bo["inp"].sync(to_dev)
                bo["run"].start()
                if bo["run"].wait(WAIT_MS) != done:
                    raise RuntimeError("npu dispatch did not complete")
                bo["out"].sync(from_dev)
                return np.frombuffer(bo["out"].read(nbytes, 0), dtype=np.int32).copy()

            seq = [0]

            def signal(v, mode):
                xs[0] = v.view(np.float16)[:K]
                ctl[1] = mode
                seq[0] += 1
                ctl[0] = seq[0]

            def await_worker():
                while ctl[2] != seq[0]:
                    pass
                return ys[0].copy()

            for _ in range(20):                               # warm up and verify the passthrough
                y = npu_step(x)
                mismatches += int(not np.array_equal(y, x))
            worker = subprocess.Popen([worker_py, str(Path(__file__).resolve()), "dml-worker", "--shm", shm.name],
                                      cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUNBUFFERED": "1"})
            print("WORKER_CMD tools/split_sync_cost.py dml-worker --shm <name>", flush=True)
            wlines = []
            for line in worker.stdout:                        # chain b runs in the worker first
                wlines.append(line.rstrip("\n"))
                if line.startswith("READY"):
                    break
            for s in wlines:
                print("worker| " + s, flush=True)
            if not any(s.startswith("READY") for s in wlines):
                raise RuntimeError("the DirectML worker never reached READY")

            # each chain returns its last NPU input and output; the passthrough must echo it exactly
            def chain_a(v):
                last_in = y = None
                for _ in range(STEPS):
                    last_in = v
                    y = npu_step(v)
                    v = y
                return last_in, y

            def chain_e(v):
                for _ in range(STEPS):
                    signal(v, MODE_ECHO)
                    yd = await_worker()
                    v = v.copy()
                    v.view(np.float16)[:N_DML] = yd
                return None, None

            def chain_c(v):
                last_in = y = None
                for _ in range(STEPS):
                    last_in = v
                    signal(v, MODE_DML)
                    y = npu_step(v)
                    yd = await_worker()
                    v = y.copy()
                    v.view(np.float16)[:N_DML] = yd
                return last_in, y

            times = {"a": [], "e": [], "c": []}
            fns = {"a": chain_a, "e": chain_e, "c": chain_c}
            for rep in range(WARMUP_CHAINS + CHAINS):
                for name in ("a", "e", "c"):                  # interleaved, so drift hits all three
                    t0 = time.perf_counter_ns()
                    last_in, y_last = fns[name](x.copy())
                    t1 = time.perf_counter_ns()
                    if y_last is not None:
                        mismatches += int(not np.array_equal(last_in, y_last))
                    if rep >= WARMUP_CHAINS:
                        times[name].append((t1 - t0) / 1e3)
            signal(x, MODE_QUIT)
            await_worker()
            rest = worker.communicate(timeout=60)[0]
            for s in rest.splitlines():
                print("worker| " + s, flush=True)
            bo.clear()
        print("CHAINS_JSON " + json.dumps({"steps": STEPS, "chain_us": times, "npu_mismatches": mismatches,
                                           "words": WORDS}), flush=True)
        witness()
        gpu_engines()
    finally:
        if worker is not None and worker.poll() is None:
            worker.kill()
        shm.close()
        shm.unlink()
    return 0


def selftest() -> int:
    """Spawn, cross-version shared memory and the handshake chain, with a dry worker; no chip."""
    from multiprocessing import shared_memory
    worker_py = os.environ.get("SPLIT_WORKER_PYTHON", sys.executable)
    shm = shared_memory.SharedMemory(create=True, size=SHM_BYTES)
    ctl = np.ndarray((3,), dtype=np.int64, buffer=shm.buf, offset=0)
    xs = np.ndarray((1, K), dtype=np.float16, buffer=shm.buf, offset=OFF_X)
    ys = np.ndarray((1, N_DML), dtype=np.float16, buffer=shm.buf, offset=OFF_Y)
    ctl[:] = 0
    w = subprocess.Popen([worker_py, str(Path(__file__).resolve()), "dry-worker", "--shm", shm.name], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        print("worker:", w.stdout.readline().strip(), "coordinator:", sys.version.split()[0])
        seq, x = 0, (np.arange(K) % 97).astype(np.float16)
        per = []
        for _ in range(5):
            t0 = time.perf_counter_ns()
            for _ in range(STEPS):
                xs[0] = x
                ctl[1] = MODE_ECHO
                seq += 1
                ctl[0] = seq
                while ctl[2] != seq:
                    pass
                assert np.array_equal(ys[0], x[:N_DML])
                x = np.roll(x, 1)
            per.append((time.perf_counter_ns() - t0) / 1e3 / STEPS)
        ctl[1] = MODE_QUIT
        seq += 1
        ctl[0] = seq
        w.wait(timeout=30)
        print("handshake per step, us:", [round(v, 2) for v in per], "worker exit", w.returncode)
    finally:
        if w.poll() is None:
            w.kill()
        shm.close()
        shm.unlink()
    return 0


# ---------------------------------------------------------------- prereg and verdict

PREREG = f"""The DirectML + NPU split's synchronization cost (LLM study, the R5 follow-up), pre-registered
before any sitting

Question (the user's decision on R5: "measure the sync first")
  Stage 2's R5 printed OPEN at 1.1004x, inside its run-to-run spread. A split decode joins the
  two chips through the host once per GEMV, because each GEMV needs the previous one's output and
  nothing crosses between DirectML and the NPU without the host. At Llama-2-7B that is {STEPS} joins
  per token. The most a DirectML + NPU split could save over DirectML alone, at stage 2's
  reduction rates, is {SAVING_MS} ms per token (DERIVED: 47.5 - 36.4). That figure is itself
  optimistic: GEMV reads slower than a reduction, and no NPU int4 GEMV exists.

Design (tools/split_sync_cost.py), trivial kernels, because the join is the cost and not the math
  NPU half: a {WORDS * 4 // 1024} KiB shim -> mem tile -> shim passthrough (one column, two {CHUNK * 4 // 1024} KiB mem-tile
    buffers). Raw pyxrt, one dispatch at a time: write x, sync to the device, start, wait, sync back,
    read y. This is the same shape as BENCHMARKS' dispatch-floor table.
  DirectML half: ONNX Runtime DirectML MatMul, x[1,{K}] fp16 by W[{K},{N_DML}], CPU fallback disabled,
    session.run with the input and output on the host.
  No Python here has both runtimes (pyxrt: CPython 3.13 and 3.10; DirectML: 3.12 only). So the
  DirectML half runs in a resnet_env17 worker, and the halves meet through shared memory with
  busy-wait flags. Chain e measures that handshake alone, and it is subtracted.
  Chains, each {STEPS} dependent steps; the next input is built from this step's output:
    a  the NPU round trip alone
    b  the DirectML round trip alone, timed inside the worker
    e  the handshake alone: the worker echoes x with no DirectML
    c  the join: signal the worker (DirectML starts), the NPU round trip, wait for the worker,
       then combine both outputs into the next input
  {WARMUP_CHAINS} warmup chains, then {CHAINS} timed chains of each. a, e and c are interleaved; b runs first, in
  the worker. The NPU passthrough is verified on 20 warmup steps and on each chain's last step.
  The sitting: the host-load gate, a GPU engine snapshot before and after, xrt-smi idle
  before and after, recorded by tools/silicon_probe_record.py, announced to the other sessions.

The join cost, named before the sitting
  J  = median over chains of (chain c time / {STEPS})     the per-GEMV join, as measured
  H  = median over chains of (chain e time / {STEPS})     the two-process handshake
  J* = J - H                                              the join a single-process host would pay;
                                                          subtracting H favours the split
Rules
  R1 (the kill line, set by the gate): DEAD iff {STEPS} x J* >= {SAVING_MS} ms, i.e. J* >= {KILL_US:.2f} us. The
     split is then dead for good. Below the line, report the margin only; nothing is established,
     and the user decides.
  R1b (robustness, reported): if independent GEMVs share one join, a token needs {STAGE_JOINS} joins (32 x
     QKV, O, gate+up, down). {STAGE_JOINS} x J* against {SAVING_MS} ms, i.e. J* against {KILL_STAGE_US:.1f} us. If R1 kills and R1b
     does not, the docs say the kill depends on joining per GEMV.
  R2 (reported): J against a, b, max(a, b) and a + b, per step: overlap or serialization.
  INCOMPLETE: an NPU dispatch that does not complete; a passthrough mismatch; DirectML not the
     first provider (CPU fallback is disabled, so creation fails if any node would leave it);
     non-finite DirectML output; the worker never READY; an xrt-smi witness that is not idle; a missing chain.

Written predictions
  Q1 a: 120-220 us per step. BENCHMARKS measured raw pyxrt at about 140 us for one dispatch of this
     shape; here the BO write and the two syncs add to it.
  Q2 b: 100-600 us per step (DirectML session.run with host input and output, never measured here).
  Q3 e: under 10 us per step. A pre-check with no chip (selftest: a dry 3.12 worker, the 3.13
     coordinator, 5 chains of 224 echoes) gave about 8 us per step, including that loop's own
     array checks.
  Q4 J* >= max(a, b) - 20 us: the halves overlap, and the slower one sets the join.
  Q5 R1 DEAD, and R1b DEAD. The prior: the repo's C++ host pays about {CPP_SINGLE_US:.0f} us for one dependent
     NPU dispatch (MEASURED, BENCHMARKS), above both {KILL_US:.1f} and {KILL_STAGE_US:.1f} us. So a faster host language
     would not rescue the split.

Unverified by design: a C++ host for both halves (the C++ NPU figure above is prior evidence, not
remeasured); DirectML IO binding; a device-to-device fence between DirectML and XRT (none is
available to this host); real GEMV halves; attention and the KV cache.
"""


def verdict(log: Path) -> int:
    text = log.read_text(encoding="utf-8")
    problems = []
    rec = next((json.loads(s.split(" ", 1)[1]) for s in text.splitlines() if s.startswith("CHAINS_JSON ")), None)
    b = next((json.loads(s.split(" ", 2)[2]) for s in text.splitlines() if s.startswith("worker| B_JSON ")), None)
    if rec is None or b is None:
        print("PROBLEM missing CHAINS_JSON or B_JSON")
        print("VERDICT INCOMPLETE")
        return 2
    if rec["npu_mismatches"]:
        problems.append(f"{rec['npu_mismatches']} passthrough mismatches")
    # the session disables CPU fallback, so creation fails if any node would leave DirectML; ORT
    # still lists the CPU provider second
    if b["providers"][:1] != ["DmlExecutionProvider"]:
        problems.append(f"DirectML providers {b['providers']}")
    if not b["finite"]:
        problems.append("DirectML chain produced non-finite values")
    for k in ("a", "e", "c"):
        if len(rec["chain_us"][k]) != CHAINS:
            problems.append(f"chain {k}: {len(rec['chain_us'][k])} chains, expected {CHAINS}")
    if len(b["chain_us"]) != CHAINS:
        problems.append(f"chain b: {len(b['chain_us'])} chains")
    if "TIMEOUT" in text or "did not complete" in text:
        problems.append("a dispatch did not complete or the sitting timed out")
    per = {k: [t / STEPS for t in rec["chain_us"][k]] for k in ("a", "e", "c")}
    per["b"] = [t / STEPS for t in b["chain_us"]]
    med = {k: statistics.median(v) for k, v in per.items()}
    print(f"per step, us (median over {CHAINS} chains of {STEPS}; min / max chain)")
    for k, label in (("a", "NPU round trip"), ("b", "DirectML round trip"), ("e", "handshake"), ("c", "join")):
        print(f"  {k} {label:20s} {med[k]:8.1f}   ({min(per[k]):.1f} / {max(per[k]):.1f})")
    if problems:
        for p in problems:
            print("PROBLEM", p)
        print("VERDICT INCOMPLETE")
        return 2
    J, H = med["c"], med["e"]
    Js = J - H
    print(f"\nJ = {J:.1f} us, H = {H:.1f} us, J* = J - H = {Js:.1f} us per join")
    tok = STEPS * Js / 1e3
    print(f"R1 {STEPS} x J* = {tok:.2f} ms per token vs {SAVING_MS} ms: "
          + ("DEAD, the split is dead for good" if tok >= SAVING_MS else
             f"below the line by {SAVING_MS - tok:.2f} ms; nothing is established, the user decides"))
    tok128 = STAGE_JOINS * Js / 1e3
    print(f"R1b {STAGE_JOINS} x J* = {tok128:.2f} ms per token vs {SAVING_MS} ms: " + ("DEAD" if tok128 >= SAVING_MS else "below the line"))
    print(f"R2 J {J:.1f} vs a {med['a']:.1f}, b {med['b']:.1f}, max(a, b) {max(med['a'], med['b']):.1f}, "
          f"a + b {med['a'] + med['b']:.1f} us")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("build", "prereg", "suite", "verdict", "dml-worker", "dry-worker", "selftest"))
    ap.add_argument("log", nargs="?")
    ap.add_argument("--shm")
    a = ap.parse_args()
    if a.mode == "build":
        build()
    elif a.mode == "dml-worker":
        return dml_worker(a.shm)
    elif a.mode == "dry-worker":
        return dry_worker(a.shm)
    elif a.mode == "selftest":
        return selftest()
    elif a.mode == "suite":
        return suite()
    elif a.mode == "verdict":
        return verdict(Path(a.log))
    else:
        print(PREREG)
        d = BUILD / f"passthrough_w{WORDS}_c{CHUNK}"
        print("PINS", d.name, "insts_sha256", hashlib.sha256((d / "insts.bin").read_bytes()).hexdigest(),
              "mlir_sha256", hashlib.sha256((d / "probe.mlir").read_bytes()).hexdigest())
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        print("git HEAD", head, "(plus this log's own commit)")
        print("PREREG_JSON " + json.dumps({"steps": STEPS, "stage_joins": STAGE_JOINS, "saving_ms": SAVING_MS,
                                          "kill_us": KILL_US, "kill_stage_us": KILL_STAGE_US, "chains": CHAINS,
                                          "warmup_chains": WARMUP_CHAINS, "words": WORDS, "chunk": CHUNK,
                                          "k": K, "n_dml": N_DML, "git_head": head}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
