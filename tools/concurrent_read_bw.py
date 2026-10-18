#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Do the CPU's, DirectML's and the NPU's DDR reads add when the three run together? (LLM study stage 2)

All three chips on this APU share one DDR5. Each reader is its own process and uses the fastest
read path measured for that chip alone:
  cpu  ONNX Runtime ReduceSum over a 1 GiB fp32 initializer, 8 threads (60.59 GB/s alone, Phase 1)
  dml  the same graph on DirectML, the Radeon 780M (68.81 GB/s alone, Phase 1)
  npu  tools/npu_read_bw_probe.py's 4x2 1 GiB dispatch, shim MM2S into lockless mem-tile sinks
       (47.62 GB/s alone, stage 1)
The coordinator starts the chosen readers and waits until each has set up and warmed up (READY).
It then hands them all one start and one stop time on perf_counter_ns, which is
QueryPerformanceCounter and system-wide on Windows. Each reader loops whole 1 GiB reads until the
stop time and returns every read's start and end. A reader's rate is the bytes it read inside the
common window [start + TRIM_S, stop - TRIM_S] over the window's length. A read that straddles an
edge counts pro rata.

    python tools/concurrent_read_bw.py prereg                         # the pre-registration text
    python tools/concurrent_read_bw.py suite                          # the sitting (resnet_env17)
    python tools/concurrent_read_bw.py verdict <suite log>            # the mechanical verdict
    python tools/concurrent_read_bw.py reader --kind {cpu,dml,npu}    # spawned by suite
    python tools/concurrent_read_bw.py selftest                       # sleeping readers; no chip
"""
import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

KINDS = ("cpu", "dml", "npu")
CONFIGS = (("cpu",), ("dml",), ("npu",), ("cpu", "dml"), ("cpu", "npu"), ("dml", "npu"), ("cpu", "dml", "npu"))
ORDER = CONFIGS + CONFIGS[::-1]    # each configuration twice, mirrored, so drift cancels in the means
WINDOW_S = 20.0                    # the readers loop this long after the common start
TRIM_S = 1.0                       # dropped at each end; the measured window is 18 s
LEAD_S = 3.0                       # the start time is this far after the last READY
WARMUP = 3
CPU_THREADS = 8                    # as Phase 1's CPU ReduceSum
READBW_MODEL = ROOT / "scratch" / "llm" / "readbw_fp32_65536x4096" / "model.onnx"
READBW_BYTES = 65536 * 4096 * 4
NPU_ARTIFACT = ((4, 2), 1024)      # stage 1's best layout
READY_TIMEOUT_S = 300
RUN_TIMEOUT_S = 180

# Pre-registered constants
SOLO_BEFORE = {"cpu": 60.59, "dml": 68.81, "npu": 47.62}   # MEASURED alone, Phase 1 and stage 1
DDR_GBPS = 96.0                    # DDR5-6000 x 2 channels x 8 B (DERIVED from the configured speed)
ADD_RATIO = 1.10                   # a combination "adds" if its total is >= 1.10x the reference
COVERAGE_MIN = 0.90                # a reader must be mid-read for >= 90% of the window
REPEAT_AGREE = 0.10                # a configuration's two runs must agree within 10% on the total


# ---------------------------------------------------------------- reader (one per process)

def reader(kind: str) -> int:
    import numpy as np
    with contextlib.ExitStack() as stack:
        if kind in ("cpu", "dml"):
            import llm_gemv_bench as bench
            import onnxruntime as ort
            sess = bench.make_session(READBW_MODEL, kind, CPU_THREADS, opt="disable_all")
            feed = {"x": np.zeros(1, np.float32)}
            nbytes = READBW_BYTES
            ident = {"onnxruntime": ort.__version__, "providers": sess.get_providers(),
                     "threads": CPU_THREADS if kind == "cpu" else None}

            def one():
                sess.run(None, feed)
        elif kind == "npu":
            import gc
            import npu_read_bw_probe as probe
            from ignite_xdna.runtime.driver import XrtSiliconHarness
            layout, mib = NPU_ARTIFACT
            d = probe.BUILD / probe.name(layout, mib)
            n = probe.words_per_channel(mib, layout[0] * layout[1])
            nbytes = n * layout[0] * layout[1] * 4
            h = stack.enter_context(XrtSiliconHarness(0))
            # the run and the buffers live in bo and go before the hardware context closes (LIFO)
            bo = {}
            stack.callback(gc.collect)
            stack.callback(bo.clear)
            xrt = h.pyxrt
            h.load_xclbin(str(d / "probe.xclbin"), "MLIR_AIE")
            bo["instr"], count = h.create_instruction_bo_from_bytes((d / "insts.bin").read_bytes())
            bo["inp"] = h.create_host_bo(nbytes, 3)
            step = 1 << 26
            for off in range(0, nbytes, step):
                bo["inp"].write(np.full(min(step, nbytes - off) // 4, 0x5A5A5A5A, dtype=np.uint32), off)
            bo["inp"].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            bo["run"] = xrt.run(h.kernel)
            for i, val in enumerate((3, bo["instr"], count, bo["inp"])):
                bo["run"].set_arg(i, val)
            ident = {"artifact": d.name, "insts_sha256": hashlib.sha256((d / "insts.bin").read_bytes()).hexdigest()}
            done = xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED

            def one():
                bo["run"].start()
                state = bo["run"].wait(probe.WAIT_MS)
                if state != done:
                    raise RuntimeError(f"npu dispatch did not complete: {state}")
        else:
            # harness self-test only (selftest mode): sleeps instead of reading, touches no chip
            nbytes, ident = 1 << 20, {}

            def one():
                time.sleep(0.02)
        for _ in range(WARMUP):
            one()
        print("READY " + json.dumps({"kind": kind, "pid": os.getpid(), "python": sys.version.split()[0],
                                     "bytes": nbytes, **ident}), flush=True)
        start, stop = (int(v) for v in sys.stdin.readline().split())
        while time.perf_counter_ns() < start - 3_000_000:
            time.sleep(0.001)
        while time.perf_counter_ns() < start:
            pass
        t0s, t1s = [], []
        while True:
            t0 = time.perf_counter_ns()
            if t0 >= stop:
                break
            one()
            t1 = time.perf_counter_ns()
            t0s.append((t0 - start) // 1000)
            t1s.append((t1 - start) // 1000)
        print("READER_JSON " + json.dumps({"kind": kind, "bytes": nbytes, "t0_us": t0s, "t1_us": t1s}), flush=True)
    return 0


# ---------------------------------------------------------------- coordinator

def window_stats(nbytes: int, t0s, t1s) -> dict:
    w0, w1 = int(TRIM_S * 1e6), int((WINDOW_S - TRIM_S) * 1e6)
    got = busy = 0.0
    for a, b in zip(t0s, t1s):
        lo, hi = max(a, w0), min(b, w1)
        if hi > lo:
            got += nbytes * (hi - lo) / (b - a)
            busy += hi - lo
    durs = [(b - a) / 1e3 for a, b in zip(t0s, t1s)]
    return {"gbps": got / (w1 - w0) / 1e3, "coverage": busy / (w1 - w0), "reads": len(durs),
            "median_ms": statistics.median(durs) if durs else None}


class Reader:
    def __init__(self, kind: str):
        self.kind, self.lines, self.ready = kind, [], threading.Event()
        script = str(Path(__file__).resolve())
        if kind in ("npu", "idle_iron"):
            cmd = [shutil.which("bash"), "scripts/research-iron.sh", script, "reader", "--kind", kind]
        else:
            cmd = [sys.executable, script, "reader", "--kind", kind]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))
            if line.startswith("READY "):
                self.ready.set()
        self.ready.set()

    def json_line(self, tag: str):
        return next((json.loads(s.split(" ", 1)[1]) for s in self.lines if s.startswith(tag + " ")), None)


def run_config(run: int, kinds) -> None:
    from silicon_probe_record import witness
    print(f"\nRUN_BEGIN {run} {'+'.join(kinds)}", flush=True)
    witness()
    gpu_engines()
    readers = [Reader(k) for k in kinds]
    try:
        limit = time.monotonic() + READY_TIMEOUT_S
        for r in readers:
            r.ready.wait(max(0.0, limit - time.monotonic()))
        missing = [r.kind for r in readers if r.json_line("READY") is None]
        if missing:
            raise RuntimeError(f"not READY: {missing}")
        start = time.perf_counter_ns() + int(LEAD_S * 1e9)
        stop = start + int(WINDOW_S * 1e9)
        for r in readers:
            r.proc.stdin.write(f"{start} {stop}\n")
            r.proc.stdin.flush()
        # one counter sample mid-window: CPU clock relative to base (a drop under load would show a
        # package power limit) and the GPU engines (the 780M busy while DirectML reads)
        while time.perf_counter_ns() < start + int(WINDOW_S * 1e9) // 2:
            time.sleep(0.05)
        print("MID_WINDOW", flush=True)
        paths = ["\\Processor Information(_Total)\\% Processor Performance"]
        # added after sitting 1 (non-deciding): where the DirectML reader's 1 GiB resource sits,
        # dedicated (the 780M's 512 MB carve-out) or shared system memory
        for r in readers:
            if r.kind == "dml":
                pid = r.json_line("READY")["pid"]
                paths += [f"\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage",
                          f"\\GPU Process Memory(pid_{pid}_*)\\Shared Usage"]
        counters(paths)
        gpu_engines()
        for r in readers:
            r.proc.wait(timeout=RUN_TIMEOUT_S)
            r.pump.join(timeout=30)
    finally:
        for r in readers:
            if r.proc.poll() is None:
                r.proc.kill()
        for r in readers:
            print(f"READER_CMD {r.kind} " + subprocess.list2cmdline(r.cmd))
            for s in r.lines:
                if not s.startswith("READER_JSON "):
                    print(f"{r.kind}| {s}")
    rec = {"run": run, "kinds": list(kinds), "readers": {}}
    for r in readers:
        raw = r.json_line("READER_JSON")
        if r.proc.returncode or raw is None:
            print(f"READER_FAILED {r.kind} exit {r.proc.returncode}", flush=True)
            continue
        rec["readers"][r.kind] = {"ready": r.json_line("READY"), "bytes": raw["bytes"], "t0_us": raw["t0_us"],
                                  "t1_us": raw["t1_us"], **window_stats(raw["bytes"], raw["t0_us"], raw["t1_us"])}
    rec["total_gbps"] = sum(v["gbps"] for v in rec["readers"].values())
    print("CONFIG_JSON " + json.dumps(rec), flush=True)
    print("RUN_SUMMARY", run, "+".join(kinds), " ".join(f"{k}={v['gbps']:.2f}(cov {v['coverage']:.3f})"
                                                        for k, v in rec["readers"].items()),
          f"total={rec['total_gbps']:.2f}", flush=True)
    witness()


def counters(paths):
    ps = "(Get-Counter @(" + ",".join(f"'{p}'" for p in paths) + ")).CounterSamples | ForEach-Object { 'COUNTER ' + $_.Path + ' ' + [math]::Round($_.CookedValue, 1) }"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    print(out.strip(), flush=True)


def gpu_engines():
    ps = ("$s = (Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples "
          "| Where-Object CookedValue -gt 1; 'GPU_ENGINES_OVER_1PCT ' + @($s).Count; "
          "$s | ForEach-Object { 'GPU_ENGINE ' + $_.InstanceName + ' ' + [math]::Round($_.CookedValue, 1) }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    print(out.strip(), flush=True)


def suite() -> int:
    host = subprocess.run(["powershell", "-NoProfile", "-File", str(ROOT / "tools/host_load.ps1")],
                          capture_output=True, text=True)
    summary = next(s for s in host.stdout.splitlines() if s.startswith("HOST_LOAD busy_cores="))
    peers = [float(v) for v in re.findall(r"HOST_LOAD_PEER ([\d.]+)", host.stdout)]
    print(summary, "peer_busy_cores", peers, flush=True)
    assert float(re.search(r"busy_cores=([\d.]+)", summary)[1]) < 2, "host busy: the CPU would share DRAM"
    assert max(peers, default=0) < .5, "a heavy peer holds the CPU"
    print("BASH", shutil.which("bash"), "ORT_PYTHON", sys.executable, flush=True)
    for run, kinds in enumerate(ORDER):
        run_config(run, kinds)
    gpu_engines()
    return 0


# ---------------------------------------------------------------- prereg and verdict

def fmt(kinds) -> str:
    return "+".join(kinds)


PREREG = f"""Concurrent DDR reads on the 8700G: CPU, DirectML (Radeon 780M) and NPU (LLM study stage 2),
pre-registered before any sitting

Question (the user's re-scope, item 2)
  Do the chips' DDR reads add when they run together? All three share one DDR5-6000 on two
  channels, {DDR_GBPS:.0f} GB/s theoretical (DERIVED). Each chip's best read rate alone (MEASURED): CPU
  {SOLO_BEFORE['cpu']} GB/s and DirectML {SOLO_BEFORE['dml']} (ORT ReduceSum, Phase 1), NPU {SOLO_BEFORE['npu']} (stage 1). This is the
  combined-chip read ceiling, and the first question under any split of decode across chips.

Design (tools/concurrent_read_bw.py)
  Readers, one process each, set up and warmed up ({WARMUP} reads) before the common start:
    cpu  ORT CPU EP ReduceSum over the 1 GiB fp32 initializer of Phase 1's readbw model, {CPU_THREADS} threads
    dml  the same graph on the DirectML EP, device_id 0 (the Radeon 780M), CPU fallback disabled
    npu  stage 1's c4x2_m1024 artifact: 8 shim MM2S streams, 1 GiB per dispatch, nothing written back
  Each reader reads 1 GiB at a time. The coordinator hands every reader one start time, {LEAD_S:.0f} s
  after the last READY, and one stop time {WINDOW_S:.0f} s later, both on perf_counter_ns. Each reader
  loops until the stop time and returns every read's start and end. Its rate is the bytes read
  inside [start + {TRIM_S:.0f} s, stop - {TRIM_S:.0f} s], with a read that straddles an edge counted pro rata,
  over those {WINDOW_S - 2 * TRIM_S:.0f} s.
  Configurations, each run twice in mirrored order, 14 runs in all:
    {', '.join(fmt(c) for c in ORDER)}
  A configuration's rate is the mean of its two runs; its total is the sum of its readers' rates.
  The sitting: the host-load gate at the start (busy cores < 2, no peer >= 0.5), xrt-smi idle
  before and after every run, a GPU engine snapshot before every run, and the whole sitting
  recorded by tools/silicon_probe_record.py with its device witnesses. It is announced to the
  other sessions, and nothing else runs.
  One counter sample mid-window per run, reported and not deciding: the CPU's % Processor
  Performance (its clock against base) and the GPU engines. The sampler is one short
  PowerShell process on an idle core, the same in every run.

Rules (mechanical; S_x a chip's rate alone in this sitting, P_xy a pair's total, T the three's)
  R1 CPU+DML    ADD iff P_cpu+dml >= {ADD_RATIO} x max(S_cpu, S_dml); otherwise SHARE.
  R2 NPU pairs  for dml+npu and cpu+npu: ADD iff the pair's total >= {ADD_RATIO} x the better of its two
                solo rates; otherwise SHARE.
  R3 NPU onto the other two   ADDS iff T >= {ADD_RATIO} x P_cpu+dml; otherwise NO.
  R4 ceiling    the best total of the 7 configurations, reported against {DDR_GBPS:.0f} GB/s and each solo.
  R5 split      the NPU earns a bandwidth role in a split iff the best total of a configuration
                with it is >= {ADD_RATIO} x the best total of one without it. Otherwise a split that
                includes the NPU is killed on bandwidth. If it passes, the split is OPEN, not
                established: no split decode is measured here, and whether decode kernels return is
                the user's decision (the stage 1 rule kept them out below 59.7 GB/s).
  Reported, not deciding: each chip's rate in each configuration against its rate alone, and
  the mid-window counters.
  Attribution: SHARE names no cause. DRAM itself, the data fabric between the chips and the
  memory controllers, and the package power limit (the 8700G's 65 W, which could lower clocks
  with all three busy) are candidates. A CPU clock that drops in the combined runs would point
  at the last; a steady one argues against it.
  INCOMPLETE    a reader that fails or never reaches READY; a reader mid-read for less than
                {COVERAGE_MIN:.0%} of the window; a configuration whose two totals differ by more than {REPEAT_AGREE:.0%};
                an xrt-smi witness that is not idle (the suite stops); a missing run.

Written predictions (stated expectations; the rules decide)
  Q1 alone, in this harness: CPU 55-65, DML 62-72, NPU 44-50 GB/s.
  Q2 R1 ADD: CPU+DML totals 76-85 GB/s, 79-89% of the theoretical; each chip slows.
  Q3 R2 ADD for dml+npu (76-85) and for cpu+npu (70-85).
  Q4 R3 NO: all three total within 1.05x of CPU+DML; near 80 GB/s one shared limit binds,
     whoever reads.
  Q5 R5: the NPU earns no bandwidth role; the best total with it is < 1.10x the best without it.

Unverified by design: reads through the chips' GEMV kernels (these are reductions and a DMA
sink); writes; the CPU at other thread counts; DirectML fp16 (its ReduceSum anomaly, Phase 1);
any split decode itself.
"""


RERUN = """Concurrent DDR reads, stage 2: the re-run rule, written after sitting 1 and before sitting 2

Sitting 1 was committed as measured at 4566f87 and is INCOMPLETE on its own repeat rule. Its two
DirectML-alone runs read 81.11 and 70.95 GB/s, 13.4% apart, over the 10% limit. Every other
configuration repeated within 4%.

Sitting 2
  The full 14-run matrix again, as pre-registered at 4557ec9
  (concurrent_read_prereg_desktop2_20260923.log): the same readers, windows, mirrored order,
  2 runs per configuration, thresholds and 10% repeat rule. Its logs carry the tag _rerun.

Which sitting decides
  Sitting 2 alone decides R1-R5. Its verdict runs on its own log only.
  If sitting 2 breaks the 10% repeat rule for any configuration, stage 2's verdict is
  INCOMPLETE and R1-R5 are not decided. The spread is reported as a finding, with both
  sittings' run totals side by side. There is no averaging across sittings and no new rule
  after sitting 2's data.
  Sitting 1 stays in the record as measured. Wherever stage 2 is reported, sitting 1's
  DirectML-alone spread is shown beside the verdict, as an observation with its cause
  unattributed. Every figure quoted comes from one sitting; no ratio pairs the two.

Changes made after sitting 1 (none touches a rule's input)
  (a) Non-deciding: the mid-window counter sample also reads the DirectML reader process's GPU
      memory, Dedicated Usage and Shared Usage (Windows GPU Process Memory counters). This
      tests one candidate for the spread: where the 1 GiB resource lands against the 780M's
      512 MB carve-out. It folds into the existing mid-window PowerShell call, so the
      sampling footprint is unchanged.
  (b) Display only: the verdict prints pre-run GPU snapshots as witnesses and leaves out the
      mid-window samples, which are expected to show the chips at work. Everything that gated
      or voided a run in sitting 1 (the host-load gate, xrt-smi idle, coverage, the repeat
      rule) gates sitting 2 identically.
      Checked: re-printing sitting 1's verdict with this change gives the same table, the
      same PROBLEM line and VERDICT INCOMPLETE. Only the witness lines differ: one pre-run
      snapshot, VS Code's 3D engine at 2.2% before run 0.
  Not changed: repeats, windows, order, readers, thresholds, the 10% rule.
  Not possible: closing VS Code's use of the 780M (the user's application). It stays witnessed.

Post hoc from sitting 1, labelled as such (the cause is unattributed)
  Each DirectML-alone run is steady within itself: a median of 13.2 ms per GiB in all four
  quarters of run 1, and 15.1 ms in run 12.
  The mid-window counters do not separate them: CPU % Processor Performance 113.7 vs 114.0,
  and the 780M's 3D engine at 99.4% vs 99.7%.
  Across all runs, the CPU clock held at 112-118% under combined load.
"""


def verdict(log: Path) -> int:
    text = log.read_text(encoding="utf-8")
    runs = [json.loads(s.split(" ", 1)[1]) for s in text.splitlines() if s.startswith("CONFIG_JSON ")]
    problems = []
    if "READER_FAILED" in text or "TIMEOUT" in text:
        problems.append("a reader failed or the sitting timed out")
    by = {}
    for r in runs:
        for k, v in r["readers"].items():
            # recompute from the raw read times; the logged summary must agree
            st = window_stats(v["bytes"], v["t0_us"], v["t1_us"])
            if abs(st["gbps"] - v["gbps"]) > 1e-6:
                problems.append(f"run {r['run']} {k}: recomputed {st['gbps']} != logged {v['gbps']}")
            if st["coverage"] < COVERAGE_MIN:
                problems.append(f"run {r['run']} {k}: coverage {st['coverage']:.3f} < {COVERAGE_MIN}")
        if set(r["readers"]) != set(r["kinds"]):
            problems.append(f"run {r['run']} {fmt(r['kinds'])}: readers {sorted(r['readers'])}")
        by.setdefault(tuple(r["kinds"]), []).append(r)
    rate, total = {}, {}
    print(f"{'configuration':14s} {'run totals GB/s':>18s} {'mean':>7s}  per chip (mean GB/s, share of its solo)")
    for c in CONFIGS:
        rs = by.get(c, [])
        if len(rs) != 2:
            problems.append(f"{fmt(c)}: {len(rs)} runs, expected 2")
            continue
        ts = [r["total_gbps"] for r in rs]
        total[c] = statistics.fmean(ts)
        if abs(ts[0] - ts[1]) / total[c] > REPEAT_AGREE:
            problems.append(f"{fmt(c)}: run totals {ts[0]:.2f} and {ts[1]:.2f} differ > {REPEAT_AGREE:.0%}")
        rate[c] = {k: statistics.fmean(r["readers"][k]["gbps"] for r in rs if k in r["readers"]) for k in c}
    solo = {k: rate[(k,)][k] for k in KINDS if (k,) in rate}
    for c in CONFIGS:
        if c in total:
            per = "  ".join(f"{k} {rate[c][k]:.2f}" + (f" ({rate[c][k] / solo[k]:.0%})" if k in solo else "") for k in c)
            ts = " / ".join(f"{r['total_gbps']:.2f}" for r in by[c])
            print(f"{fmt(c):14s} {ts:>18s} {total[c]:7.2f}  {per}")
    # pre-run snapshots are witnesses; mid-window samples are expected to show the chips at work
    mid = False
    for s in text.splitlines():
        if s.startswith(("RUN_BEGIN", "READER_CMD")):
            mid = False
        elif s.startswith("MID_WINDOW"):
            mid = True
        elif s.startswith("GPU_ENGINES_OVER_1PCT ") and not mid and s.split()[1] != "0":
            print(f"pre-run witness: {s.split()[1]} GPU engine(s) over 1%")
        elif s.startswith("GPU_ENGINE ") and not mid:
            print("  " + s)
    if problems:
        for p in problems:
            print("PROBLEM", p)
        print("VERDICT INCOMPLETE")
        return 2

    def add(t, ref):
        return "ADD" if t >= ADD_RATIO * ref else "SHARE"
    cd, T = total[("cpu", "dml")], total[("cpu", "dml", "npu")]
    print(f"\nR1 cpu+dml {cd:.2f} vs {ADD_RATIO} x max({solo['cpu']:.2f}, {solo['dml']:.2f}): {add(cd, max(solo['cpu'], solo['dml']))} "
          f"({cd / max(solo['cpu'], solo['dml']):.3f}x)")
    for c in (("dml", "npu"), ("cpu", "npu")):
        ref = max(solo[k] for k in c)
        print(f"R2 {fmt(c)} {total[c]:.2f} vs {ADD_RATIO} x {ref:.2f}: {add(total[c], ref)} ({total[c] / ref:.3f}x)")
    print(f"R3 all three {T:.2f} vs {ADD_RATIO} x cpu+dml {cd:.2f}: {'ADDS' if T >= ADD_RATIO * cd else 'NO'} ({T / cd:.3f}x)")
    best = max(total, key=total.get)
    print(f"R4 ceiling {total[best]:.2f} GB/s at {fmt(best)}: {total[best] / DDR_GBPS:.1%} of {DDR_GBPS:.0f} (DERIVED), "
          + ", ".join(f"{total[best] / solo[k]:.2f}x {k} alone" for k in KINDS))
    with_n = max((c for c in total if "npu" in c), key=total.get)
    without = max((c for c in total if "npu" not in c), key=total.get)
    ratio = total[with_n] / total[without]
    print(f"R5 best with the NPU {total[with_n]:.2f} ({fmt(with_n)}) vs best without {total[without]:.2f} ({fmt(without)}) = "
          f"{ratio:.3f}x -> " + ("OPEN, not established: the NPU adds read bandwidth a split could use"
                                  if ratio >= ADD_RATIO else "a split that includes the NPU is killed on bandwidth"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("reader", "prereg", "suite", "verdict", "selftest"))
    ap.add_argument("log", nargs="?")
    ap.add_argument("--kind", choices=KINDS + ("idle", "idle_iron"))
    ap.add_argument("--rerun", action="store_true", help="prereg: print the re-run rule written after sitting 1")
    a = ap.parse_args()
    if a.mode == "reader":
        return reader(a.kind)
    if a.mode == "selftest":
        # pipes, barrier, both launchers and the window arithmetic, with sleeping readers; no chip
        run_config(0, ("idle", "idle_iron"))
        return 0
    if a.mode == "suite":
        return suite()
    if a.mode == "verdict":
        return verdict(Path(a.log))
    print(RERUN if a.rerun else PREREG)
    import npu_read_bw_probe as probe
    d = probe.BUILD / probe.name(*NPU_ARTIFACT)
    print("PINS")
    print("  npu", d.name, "insts_sha256", hashlib.sha256((d / "insts.bin").read_bytes()).hexdigest())
    print("  readbw model", READBW_MODEL.relative_to(ROOT).as_posix(), "sha256",
          hashlib.sha256(READBW_MODEL.read_bytes()).hexdigest(),
          "weights.bin bytes", (READBW_MODEL.parent / "weights.bin").stat().st_size)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print("git HEAD", head, "(plus this log's own commit)")
    print("PREREG_JSON " + json.dumps({"configs": ORDER, "window_s": WINDOW_S, "trim_s": TRIM_S, "lead_s": LEAD_S,
                                      "warmup": WARMUP, "cpu_threads": CPU_THREADS, "solo_before": SOLO_BEFORE,
                                      "ddr_gbps": DDR_GBPS, "add_ratio": ADD_RATIO, "coverage_min": COVERAGE_MIN,
                                      "repeat_agree": REPEAT_AGREE, "git_head": head}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
