#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The two noise sources stage 3 left unattributed, studied as a finding about measuring on this
APU (the user's decision: "study the noise"). Not a yardstick, and it re-scores nothing.

(A) The CPU's 8-thread rows are bimodal from rep to rep (int8 S1 at M = 512: about 4.2 or
    7.2 ms); 16 threads are stable. One row, five ways of placing the threads:
      A0   8 threads, default scheduling (as in stage 3)
      A1   8 threads pinned to 8 distinct physical cores
      A3   8 threads pinned to 7 cores, one of them running two (its SMT siblings)
      A2   8 threads pinned to 4 cores x 2 SMT siblings
      A16  16 threads, default scheduling
      AD   ORT's own default (intra_op_num_threads 0)
    Every rep's time, and once a second every logical CPU's busy share and clock.
(B) DirectML rows held one level within a pass and another in the next, but only once each row
    had its own session (stage 3 sitting 2):
      B1   20 fresh sessions of one row, one after another: open, 3 warmups, 10 reps, the GPU
           memory witness, close
      B2   one session, 20 blocks of 10 reps, the GPU memory witness between blocks

    python tools/measure_noise.py topology                 # SMT siblings and core parking, no timing
    python tools/measure_noise.py prereg                   # the pre-registration text
    python tools/measure_noise.py cpu                      # (A), the sitting (resnet_env17)
    python tools/measure_noise.py dml                      # (B), the sitting (resnet_env17)
    python tools/measure_noise.py verdict CPU_LOG DML_LOG  # the mechanical verdict
    python tools/measure_noise.py selftest                 # synthetic verdicts; no timing
"""
import argparse
import ctypes
import gc
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

ROWS_A = [("int8", "S1", 512), ("fp32", "S1", 512)]      # the first decides; the second is reported
ARMS_A = ["A0", "A1", "A3", "A2", "A16", "AD"]
SESSIONS_A = {"A0": 5, "AD": 5}                          # fresh sessions per arm and pass
SESSIONS_A_DEFAULT = 3
RUN_S = {"A0": 2.0, "AD": 2.0}                           # seconds of timed reps per session
RUN_S_DEFAULT = 1.0
WARMUP_A = 20
SLOW, FAST = 1.30, 1.15                                  # x the run's 10th percentile
MODE_FRAC = 0.10                                         # both clusters hold at least this share
UNIMODAL = 1.15                                          # p90 / p10
LEVEL_MATCH = 0.15                                       # A3's level against A0's slow level
BUSY = 50.0                                              # % busy for a logical CPU in one second
ASSOC = 0.30                                             # slow-share difference, doubled vs not
CLOCK_POINTS = 3.0

ROWS_B = [("fp32", "S2", 2048), ("fp16", "S1", 2048)]    # the first decides; the second is reported
SESSIONS = BLOCKS = 20
REPS_B, WARMUP_B = 10, 3
STEADY = 1.10                                            # within a session or block: p90 / p10
SHIFT = 1.10                                             # across sessions or blocks: max / min
EDGE = 5                                                 # slowest and fastest sessions compared


# ---------------------------------------------------------------- topology and pinning

class _PC(ctypes.Structure):
    _fields_ = [("Flags", ctypes.c_ubyte)]


class _U(ctypes.Union):
    _fields_ = [("ProcessorCore", _PC), ("Reserved", ctypes.c_ulonglong * 2)]


class _SLPI(ctypes.Structure):
    _fields_ = [("ProcessorMask", ctypes.c_size_t), ("Relationship", ctypes.c_int), ("u", _U)]


def cores() -> list:
    """Physical cores as lists of logical CPUs (0-based), from GetLogicalProcessorInformation."""
    k32 = ctypes.windll.kernel32
    n = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformation(None, ctypes.byref(n))
    buf = (_SLPI * (n.value // ctypes.sizeof(_SLPI)))()
    if not k32.GetLogicalProcessorInformation(buf, ctypes.byref(n)):
        raise OSError("GetLogicalProcessorInformation failed")
    return [[i for i in range(64) if e.ProcessorMask >> i & 1] for e in buf if e.Relationship == 0]


def arm_cpus(arm: str, cs: list):
    """The logical CPUs an arm's 8 threads are pinned to (the first one takes the calling thread)."""
    if arm == "A1":
        return [c[0] for c in cs[:8]]
    if arm == "A3":
        return cs[0] + [c[0] for c in cs[1:7]]
    if arm == "A2":
        return [x for c in cs[:4] for x in c]
    return None


def parking() -> dict:
    out = {}
    for name in ("CPMINCORES", "CPMAXCORES"):
        text = subprocess.run(["powercfg", "/qh", "SCHEME_CURRENT", "SUB_PROCESSOR", name],
                              capture_output=True, text=True).stdout
        ac = re.search(r"Current AC Power Setting Index: 0x([0-9a-f]+)", text)
        out[name] = int(ac[1], 16) if ac else None
    scheme = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True).stdout.strip()
    out["scheme"] = scheme.split("(")[-1].rstrip(")") if "(" in scheme else scheme
    return out


def topology() -> int:
    cs = cores()
    print("TOPOLOGY_JSON " + json.dumps({"cores": cs, "parking_ac_percent": parking(),
                                         "arms": {a: arm_cpus(a, cs) for a in ARMS_A}}))
    return 0


def pin_calling_thread(cpu):
    k32 = ctypes.windll.kernel32
    k32.GetCurrentThread.restype = wintypes.HANDLE
    k32.SetThreadAffinityMask.restype = ctypes.c_size_t
    k32.SetThreadAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(ctypes.c_size_t)]
    if cpu is None:
        pm, sm = ctypes.c_size_t(), ctypes.c_size_t()
        if not k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(pm), ctypes.byref(sm)):
            raise OSError("GetProcessAffinityMask failed")
        mask = pm.value
    else:
        mask = 1 << cpu
    if not k32.SetThreadAffinityMask(k32.GetCurrentThread(), mask):
        raise OSError("SetThreadAffinityMask failed")


class _TE32(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ThreadID", wintypes.DWORD),
                ("th32OwnerProcessID", wintypes.DWORD), ("tpBasePri", ctypes.c_long), ("tpDeltaPri", ctypes.c_long),
                ("dwFlags", wintypes.DWORD)]


class _GA(ctypes.Structure):
    _fields_ = [("Mask", ctypes.c_size_t), ("Group", wintypes.WORD), ("Reserved", wintypes.WORD * 3)]


class _PN(ctypes.Structure):
    _fields_ = [("Group", wintypes.WORD), ("Number", ctypes.c_ubyte), ("Reserved", ctypes.c_ubyte)]


def thread_ids() -> set:
    """This process's thread ids (Toolhelp32)."""
    k32 = ctypes.windll.kernel32
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.Thread32First.argtypes = k32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_TE32)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    snap = k32.CreateToolhelp32Snapshot(4, 0)                 # TH32CS_SNAPTHREAD
    te, pid, out = _TE32(), os.getpid(), set()
    te.dwSize = ctypes.sizeof(_TE32)
    ok = k32.Thread32First(snap, ctypes.byref(te))
    while ok:
        if te.th32OwnerProcessID == pid:
            out.add(te.th32ThreadID)
        ok = k32.Thread32Next(snap, ctypes.byref(te))
    k32.CloseHandle(snap)
    return out


def thread_placement(tids) -> list:
    """Each thread's affinity (logical CPUs) and ideal processor: the pinning, read back."""
    k32 = ctypes.windll.kernel32
    k32.OpenThread.restype = wintypes.HANDLE
    k32.GetThreadGroupAffinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(_GA)]
    k32.GetThreadIdealProcessorEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PN)]
    k32.GetThreadSelectedCpuSets.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG), wintypes.ULONG,
                                             ctypes.POINTER(wintypes.ULONG)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    out = []
    for tid in sorted(tids):
        h = k32.OpenThread(0x0040, False, tid)                # THREAD_QUERY_INFORMATION
        if not h:
            continue
        ga, pn, n = _GA(), _PN(), wintypes.ULONG(0)
        mask = ga.Mask if k32.GetThreadGroupAffinity(h, ctypes.byref(ga)) else 0
        ideal = pn.Number if k32.GetThreadIdealProcessorEx(h, ctypes.byref(pn)) else None
        k32.GetThreadSelectedCpuSets(h, None, 0, ctypes.byref(n))   # n = 0: no CPU sets chosen
        k32.CloseHandle(h)
        out.append({"cpus": [i for i in range(64) if mask >> i & 1], "ideal": ideal, "cpu_sets": n.value})
    return out


def cpu_session(model: bytes, threads: int, cpus):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    if cpus:                                              # ORT pins the pool (1-based ids); we pin the caller
        so.add_session_config_entry("session.intra_op_thread_affinities", ";".join(str(c + 1) for c in cpus[1:]))
    return ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------- samplers

class Sampler:
    """Once a second: every logical CPU's % Processor Time (b<i>) and % Processor Performance
    (c<i>), plus the total clock. A sample stamped T covers [T - 1 s, T]."""

    PS = ("$p = @('\\Processor(*)\\% Processor Time', '\\Processor Information(*)\\% Processor Performance'); "
          "Get-Counter -Counter $p -SampleInterval 1 -Continuous | ForEach-Object { "
          "$t = [DateTimeOffset]::Now.ToUnixTimeMilliseconds(); "
          "$s = ($_.CounterSamples | ForEach-Object { $k = if ($_.Path -like '*processor time') {'b'} else {'c'}; "
          "$k + ($_.InstanceName -replace '^0,', '') + '=' + [math]::Round($_.CookedValue, 1) }) -join ' '; "
          "\"$t $s\" }")

    def __init__(self):
        self.samples = []
        self.proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", self.PS], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            parts = line.split()
            if parts and parts[0].isdigit():
                vals = {}
                for kv in parts[1:]:
                    k, _, v = kv.partition("=")
                    k = k.lower()                         # Get-Counter reports the total as "_total"
                    try:
                        vals[k] = float(v)
                    except ValueError:
                        pass
                self.samples.append({"t": int(parts[0]), **vals})

    def wait_first(self, timeout_s=20.0):
        t0 = time.time()
        while not self.samples and time.time() - t0 < timeout_s:
            time.sleep(0.1)

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()


def gpu_memory(pid: int) -> dict:
    ps = (f"(Get-Counter @('\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage', "
          f"'\\GPU Process Memory(pid_{pid}_*)\\Shared Usage')).CounterSamples | ForEach-Object "
          "{ $_.Path + '|' + $_.CookedValue }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    mem = {"dedicated_mb": 0.0, "shared_mb": 0.0}
    for line in out.splitlines():
        path, _, v = line.rpartition("|")
        key = "dedicated_mb" if "dedicated usage" in path.lower() else "shared_mb" if "shared usage" in path.lower() else None
        if key and v.strip():
            mem[key] += float(v) / 2 ** 20
    return {k: round(v, 1) for k, v in mem.items()}


# ---------------------------------------------------------------- the sittings

def header():
    import onnxruntime as ort
    print("HEADER " + json.dumps({"python": platform.python_version(), "onnxruntime": ort.__version__,
                                  "numpy": np.__version__, "env": os.environ.get("CONDA_DEFAULT_ENV", "?"),
                                  "host": platform.node()}, sort_keys=True), flush=True)


def row_model(kind: str, s: str, data: dict):
    from llm_prefill_bench import ort_model
    if kind == "int8":
        return ort_model("int8", data[f"w_{s}_q"], "u8zp")
    return ort_model(kind, data[f"w_{s}"].astype(np.float16 if kind == "fp16" else np.float32))


def row_feed(kind: str, s: str, M: int, data: dict):
    from llm_prefill_bench import X_OF, ZP
    x = X_OF[s]
    if kind == "int8":
        return {"x": (data[f"{x}_q"][:M].astype(np.int16) + ZP).astype(np.uint8)}
    return {"x": np.ascontiguousarray(data[x][:M].astype(np.float16 if kind == "fp16" else np.float32))}


def cpu() -> int:
    from llm_prefill_bench import host_gate, load
    header()
    host_gate()
    cs = cores()
    print("TOPOLOGY_JSON " + json.dumps({"cores": cs, "parking_ac_percent": parking(),
                                         "arms": {a: arm_cpus(a, cs) for a in ARMS_A}}), flush=True)
    data = load(["x4", "x4_q", "w_S1", "w_S1_q"])
    models = {(k, s): row_model(k, s, data) for k, s, _ in ROWS_A}
    # a throwaway session first, so the per-arm thread witness counts only each session's pool and not
    # the threads ORT starts once per process
    k0, s0, M0 = ROWS_A[0]
    cpu_session(models[(k0, s0)], 1, None).run(None, row_feed(k0, s0, M0, data))
    gc.collect()
    sampler = Sampler()
    sampler.wait_first()
    order = [(r, a) for r in ROWS_A for a in ARMS_A]
    try:
        for p, seq in ((1, order), (2, order[::-1])):
            for (kind, s, M), arm in seq:
                cpus = arm_cpus(arm, cs)
                threads = {"A16": 16, "AD": 0}.get(arm, 8)
                feed = row_feed(kind, s, M, data)
                for si in range(SESSIONS_A.get(arm, SESSIONS_A_DEFAULT)):
                    before = thread_ids()
                    sess = cpu_session(models[(kind, s)], threads, cpus)
                    pin_calling_thread(cpus[0] if cpus else None)
                    try:
                        for _ in range(WARMUP_A):
                            sess.run(None, feed)
                        pool = thread_placement(thread_ids() - before)
                        main_at = thread_placement({ctypes.windll.kernel32.GetCurrentThreadId()})
                        ms, ts = [], []
                        w0 = int(time.time() * 1000)
                        end = time.perf_counter() + RUN_S.get(arm, RUN_S_DEFAULT)
                        while time.perf_counter() < end:
                            ts.append(int(time.time() * 1000))
                            t0 = time.perf_counter()
                            sess.run(None, feed)
                            ms.append(round((time.perf_counter() - t0) * 1e3, 3))
                        w1 = int(time.time() * 1000)
                    finally:
                        pin_calling_thread(None)
                    del sess
                    gc.collect()
                    print("ARM_JSON " + json.dumps({"row": f"{kind} {s} M={M}", "arm": arm, "pass": p, "session": si,
                                                    "threads": threads, "cpus": cpus, "pool": pool, "main": main_at,
                                                    "w0": w0, "w1": w1, "ms": ms, "t": ts}), flush=True)
                time.sleep(1.5)                           # a sampler second between arms
    finally:
        time.sleep(1.5)
        sampler.close()
    for smp in sampler.samples:
        print("SAMPLE " + json.dumps(smp))
    return 0


def dml() -> int:
    from concurrent_read_bw import gpu_engines
    from llm_prefill_bench import host_gate, load, make_session
    header()
    host_gate()
    gpu_engines()
    pid = os.getpid()
    data = load(["x4", "w_S1", "w_S2"])
    for kind, s, M in ROWS_B:
        row = f"{kind} {s} M={M}"
        model = row_model(kind, s, data)
        feed = row_feed(kind, s, M, data)
        for i in range(SESSIONS):                         # B1: fresh sessions, one after another
            t0 = time.perf_counter()
            sess = make_session(model, "dml", 0)
            create_ms = (time.perf_counter() - t0) * 1e3
            for _ in range(WARMUP_B):
                sess.run(None, feed)
            ms = []
            for _ in range(REPS_B):
                t0 = time.perf_counter()
                sess.run(None, feed)
                ms.append(round((time.perf_counter() - t0) * 1e3, 3))
            mem = gpu_memory(pid)
            providers = sess.get_providers()
            del sess
            gc.collect()
            print("SESSION_JSON " + json.dumps({"row": row, "i": i, "create_ms": round(create_ms, 1), "ms": ms,
                                                "providers": providers[:1], **mem}), flush=True)
        sess = make_session(model, "dml", 0)              # B2: one session, blocks
        for _ in range(WARMUP_B):
            sess.run(None, feed)
        for b in range(BLOCKS):
            ms = []
            for _ in range(REPS_B):
                t0 = time.perf_counter()
                sess.run(None, feed)
                ms.append(round((time.perf_counter() - t0) * 1e3, 3))
            print("BLOCK_JSON " + json.dumps({"row": row, "b": b, "ms": ms, **gpu_memory(pid)}), flush=True)
        del sess
        gc.collect()
    gpu_engines()
    return 0


# ---------------------------------------------------------------- the verdict

def classify(ms) -> dict:
    a = np.asarray(ms, dtype=float)
    p10, p50, p90 = (float(np.percentile(a, q)) for q in (10, 50, 90))
    slow, fast = a[a > SLOW * p10], a[a <= FAST * p10]
    sf, ff = len(slow) / len(a), len(fast) / len(a)
    mode = ("bimodal" if sf >= MODE_FRAC and ff >= MODE_FRAC else
            "unimodal" if p90 / p10 <= UNIMODAL else "broad")
    return {"n": len(a), "p10": p10, "p50": p50, "p90": p90, "slow_frac": sf, "fast_frac": ff, "mode": mode,
            "fast_level": float(np.median(fast)) if len(fast) else None,
            "slow_level": float(np.median(slow)) if len(slow) else None}


def seconds(arm: dict, samples: list, cs: list) -> list:
    """Per sampler second fully inside the arm's window: its reps' slow share, the busy logical
    CPUs, the physical cores with both siblings busy, and the total clock."""
    out = []
    p10 = float(np.percentile(arm["ms"], 10))
    for smp in samples:
        lo, hi = smp["t"] - 1000, smp["t"]
        if lo < arm["w0"] or hi > arm["w1"]:
            continue
        reps = [m for t, m in zip(arm["t"], arm["ms"]) if lo < t <= hi]
        if not reps:
            continue
        busy = [i for i in range(sum(len(c) for c in cs)) if smp.get(f"b{i}", 0) >= BUSY]
        doubled = sum(1 for c in cs if all(x in busy for x in c))
        out.append({"slow_frac": sum(m > SLOW * p10 for m in reps) / len(reps), "busy": busy, "doubled": doubled,
                    "clock": smp.get("c_total")})
    return out


def verdict_cpu(text: str) -> list:
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    lines_json = [json.loads(s[9:]) for s in text.splitlines() if s.startswith("ARM_JSON ")]
    samples = [json.loads(s[7:]) for s in text.splitlines() if s.startswith("SAMPLE ")]
    topo = next((json.loads(s[14:]) for s in text.splitlines() if s.startswith("TOPOLOGY_JSON ")), None)
    if not lines_json or not topo:
        say("(A) INCOMPLETE: no arm rows or no topology in the log")
        return lines
    cs = topo["cores"]
    say(f"(A) cores {cs}; core parking on AC: {topo['parking_ac_percent']}")
    groups = {}
    for a in lines_json:                                  # an arm and pass pools its sessions
        groups.setdefault((a["row"], a["arm"], a["pass"]), []).append(a)
    arms = []
    for (row, arm, p), ss in groups.items():
        ss.sort(key=lambda a: a.get("session", 0))
        arms.append({"row": row, "arm": arm, "pass": p, "cpus": ss[0]["cpus"], "w0": ss[0]["w0"], "w1": ss[-1]["w1"],
                     "ms": [m for a in ss for m in a["ms"]], "t": [t for a in ss for t in a["t"]], "sessions": ss})

    def ideal_shared(a):
        ideal = [t["ideal"] for t in a.get("pool", []) + a.get("main", []) if t["ideal"] is not None]
        return sum(1 for c in cs if sum(i in c for i in ideal) >= 2)

    res = {}
    for a in arms:
        c = classify(a["ms"])
        res[(a["row"], a["arm"], a["pass"])] = (a, c)
        sec = seconds(a, samples, cs)
        dbl = [s["doubled"] for s in sec]
        nbusy = [len(s["busy"]) for s in sec]
        clk = [s["clock"] for s in sec if s["clock"] is not None]
        say(f"  {a['row']:16s} {a['arm']:3s} pass {a['pass']}  n {c['n']:5d}  p10/p50/p90 {c['p10']:7.2f} "
            f"{c['p50']:7.2f} {c['p90']:7.2f} ms  {c['mode']:8s} slow {c['slow_frac']:.2f}  "
            f"levels {c['fast_level'] or 0:.2f}/{c['slow_level'] or 0:.2f}  busy CPUs {min(nbusy, default=0)}-"
            f"{max(nbusy, default=0)}  doubled cores {min(dbl, default=0)}-{max(dbl, default=0)}  "
            f"clock {min(clk, default=0):.0f}-{max(clk, default=0):.0f}%")
        ss = a["sessions"]
        th = [t for x in ss for t in x.get("pool", []) + x.get("main", [])]
        aff = sorted({tuple(t["cpus"]) for t in ss[0].get("pool", [])})
        say(f"      sessions {len(ss)}: p50 {[round(float(np.median(x['ms'])), 2) for x in ss]} ms; pool sizes "
            f"{[len(x.get('pool', [])) for x in ss]}; cores holding two ideal processors "
            f"{[ideal_shared(x) for x in ss]}; first session's pool affinities {[list(x) for x in aff]}, main "
            f"{ss[0]['main'][0]['cpus'] if ss[0].get('main') else '?'}; threads with CPU sets "
            f"{sum(1 for t in th if t.get('cpu_sets'))}")
    ref = f"{ROWS_A[0][0]} {ROWS_A[0][1]} M={ROWS_A[0][2]}"
    get = lambda arm: [res[k] for k in sorted(res) if k[0] == ref and k[1] == arm]
    a0, a1, a3 = get("A0"), get("A1"), get("A3")
    if not (a0 and a1 and a3):
        say("(A) INCOMPLETE: the reference row lacks A0, A1 or A3")
        return lines
    # pinning control: a pinned arm's busy CPUs must be its own
    for arm in ("A1", "A3", "A2"):
        for a, _ in get(arm):
            sec = seconds(a, samples, cs)
            stray = sorted({i for s in sec for i in s["busy"]} - set(a["cpus"]))
            say(f"  pinning control {arm} pass {a['pass']}: busy outside its CPUs: {stray or 'none'}")
    if not any(c["mode"] == "bimodal" for _, c in a0):
        say(f"(A) NOT REPRODUCED: A0 is not bimodal in either pass of {ref}. No attribution.")
        return lines
    slow0 = statistics.median(c["slow_level"] for _, c in a0 if c["slow_level"])
    if all(c["mode"] == "unimodal" for _, c in a1):
        say("(A) ATTRIBUTED to where ORT's own threads land, not the clock and not a thread outside ORT: A0 is "
            "bimodal and A1 (8 distinct cores, pinned, their siblings free) is unimodal.")
        a3_ok = all(c["mode"] == "unimodal" and abs(c["p50"] / slow0 - 1) <= LEVEL_MATCH for _, c in a3)
        say(f"  SMT sufficiency: A3 (one doubled core) p50 {', '.join(f'{c['p50']:.2f}' for _, c in a3)} ms "
            f"against A0's slow level {slow0:.2f}: " + ("matches within 15%" if a3_ok else "does not match"))
        sec0 = [s for a, _ in a0 for s in seconds(a, samples, cs)]
        with_d = [s["slow_frac"] for s in sec0 if s["doubled"] >= 1]
        without = [s["slow_frac"] for s in sec0 if s["doubled"] == 0]
        if with_d and without:
            diff = statistics.mean(with_d) - statistics.mean(without)
            assoc = diff >= ASSOC
            say(f"  association in A0: slow share {statistics.mean(with_d):.2f} in {len(with_d)} seconds with a "
                f"doubled core, {statistics.mean(without):.2f} in {len(without)} without: "
                + ("holds" if assoc else "does not hold"))
        else:
            assoc = None
            say(f"  association in A0: not testable ({len(with_d)} seconds with a doubled core, {len(without)} without)")
        if a3_ok and assoc:
            say("(A) VERDICT: ORT-thread doubling. One core running two of the 8 threads gives the slow level, "
                "and the slow reps come in the seconds when a core runs two.")
        elif a3_ok:
            say("(A) VERDICT: placement. A doubled core can give the slow level, but the busy map does not tie it "
                "to the slow reps: doubling and scheduler migration are not separated.")
        else:
            say("(A) VERDICT: placement, but one doubled core does not give the slow level: scheduler migration or "
                "another placement effect; not separated.")
    else:
        sec = [s for a, _ in a0 + a1 for s in seconds(a, samples, cs)]
        hi = [s["clock"] for s in sec if s["slow_frac"] >= 0.5 and s["clock"] is not None]
        lo = [s["clock"] for s in sec if s["slow_frac"] <= 0.1 and s["clock"] is not None]
        a2 = get("A2")
        if hi and lo and statistics.mean(lo) - statistics.mean(hi) >= CLOCK_POINTS:
            say(f"(A) VERDICT: where ORT's threads land does not explain it (A1 is not unimodal); the clock is "
                f"implicated: {statistics.mean(hi):.0f}% in slow seconds, {statistics.mean(lo):.0f}% in fast ones.")
        elif a2 and all(c["mode"] == "unimodal" for _, c in a2):
            say("(A) VERDICT: not ORT's own placement (A1 is not unimodal) and the clock does not separate slow "
                "seconds from fast; A2 (4 full cores, 4 idle) holds. The slowdown needs a free sibling beside an "
                "ORT thread: a thread outside ORT on that sibling, not separated from an effect of the number of busy "
                "cores (A1 keeps 8 busy, A2 4).")
        else:
            say("(A) VERDICT: NON-ATTRIBUTION. A1 (pinned to 8 distinct cores) is not unimodal, A2 does not hold, "
                "and the clock does not separate slow seconds from fast ones.")
    for arm, what in (("A16", "16 threads"), ("AD", "ORT's default threading")):
        say(f"  control {arm} ({what}): " + ", ".join(f"pass {a['pass']} {c['mode']} p50 {c['p50']:.2f} ms"
                                                     for a, c in get(arm)))
    # reported, not deciding: do the unpinned sessions whose threads' ideal processors share a core run slower?
    free = [x for arm in ("A0", "AD") for a, _ in get(arm) for x in a["sessions"]]
    two = [float(np.median(x["ms"])) for x in free if ideal_shared(x) >= 1]
    none = [float(np.median(x["ms"])) for x in free if ideal_shared(x) == 0]
    say(f"  reported: unpinned sessions (A0 and AD) with a core holding two ideal processors: {len(two)}, median "
        f"p50 {statistics.median(two) if two else float('nan'):.2f} ms; without: {len(none)}, median p50 "
        f"{statistics.median(none) if none else float('nan'):.2f} ms")
    return lines


def verdict_dml(text: str) -> list:
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    sess = [json.loads(s[13:]) for s in text.splitlines() if s.startswith("SESSION_JSON ")]
    blocks = [json.loads(s[11:]) for s in text.splitlines() if s.startswith("BLOCK_JSON ")]
    rows = [f"{k} {s} M={M}" for k, s, M in ROWS_B]
    for n, row in enumerate(rows):
        ss = [x for x in sess if x["row"] == row]
        bb = [x for x in blocks if x["row"] == row]
        if len(ss) < SESSIONS or len(bb) < BLOCKS:
            say(f"(B) {row}: INCOMPLETE ({len(ss)} sessions, {len(bb)} blocks)")
            continue
        if any(x["providers"] != ["DmlExecutionProvider"] for x in ss):
            say(f"(B) {row}: INCOMPLETE, a session did not place on DirectML")
            continue
        lv = lambda x: float(np.median(x["ms"]))
        steady = lambda x: float(np.percentile(x["ms"], 90) / np.percentile(x["ms"], 10)) <= STEADY
        say(f"(B) {row}" + ("  (decides)" if n == 0 else "  (reported)"))
        for x in ss:
            say(f"  session {x['i']:2d}  level {lv(x):8.2f} ms  {'steady' if steady(x) else 'unsteady'}  create "
                f"{x['create_ms']:7.1f} ms  dedicated {x['dedicated_mb']:7.1f} MB  shared {x['shared_mb']:7.1f} MB")
        for x in bb:
            say(f"  block   {x['b']:2d}  level {lv(x):8.2f} ms  {'steady' if steady(x) else 'unsteady'}  "
                f"dedicated {x['dedicated_mb']:7.1f} MB  shared {x['shared_mb']:7.1f} MB")
        st = [x for x in ss if steady(x)]
        s_ratio = max(map(lv, st)) / min(map(lv, st)) if len(st) >= 2 else 1.0
        b_ratio = max(map(lv, bb)) / min(map(lv, bb))
        repro = s_ratio >= SHIFT
        say(f"  across steady sessions ({len(st)} of {len(ss)}): max/min {s_ratio:.3f}; across blocks of one "
            f"session: {b_ratio:.3f}")
        if n:
            continue
        if not repro:
            say("(B) VERDICT: level shifts NOT REPRODUCED across fresh sessions. No attribution.")
        elif b_ratio < SHIFT:
            srt = sorted(st, key=lv)
            fast, slow = srt[:EDGE], srt[-EDGE:]
            sep = []
            for key in ("dedicated_mb", "shared_mb"):
                f, s_ = [x[key] for x in fast], [x[key] for x in slow]
                if max(f) < min(s_) or max(s_) < min(f):
                    sep.append(key)
            if sep:
                say(f"(B) VERDICT: per-session state, attributed to allocation placement: the {EDGE} slowest and "
                    f"{EDGE} fastest sessions do not overlap in {', '.join(sep)}.")
            else:
                say("(B) VERDICT: per-session state. Fresh sessions land at different levels and one session holds "
                    "its level, but the memory witness does not separate slow sessions from fast: the cause inside "
                    "session creation is unattributed.")
        else:
            say("(B) VERDICT: NON-ATTRIBUTION. Levels shift within one long-lived session too, so it is not "
                "per-session; GPU clock or power state and the driver's queue are not separated.")
    return lines


def verdict(cpu_log: Path, dml_log: Path) -> int:
    ca = cpu_log.read_text(encoding="utf-8", errors="replace")
    da = dml_log.read_text(encoding="utf-8", errors="replace")
    for name, text in (("cpu", ca), ("dml", da)):
        if "EXIT_CODE: 0" not in text:
            print(f"{name} log: no EXIT_CODE 0; INCOMPLETE")
            return 2
    verdict_cpu(ca)
    print()
    verdict_dml(da)
    return 0


# ---------------------------------------------------------------- pre-registration

PREREG = f"""\
Measurement noise on this APU: the two sources stage 3 left unattributed, pre-registered before
any sitting (the user's decision: "study the noise")

This is a finding about measuring on this machine, not about the NPU. It re-scores nothing:
stage 3 stays INCOMPLETE. What it can change is how later pre-registrations time the CPU and
DirectML arms, written as a recommendation.

(A) The CPU's 8-thread bimodality
  Seen: stage 3's 8-thread rows flip from rep to rep. int8 S1 at M = 512 in sitting 1 ranged
  4.1-7.3 ms within a pass, 14 of its 20 reps near 4.2 or near 7.2, in runs of consecutive reps.
  In sitting 2, where each row opened its own session, both passes sat near the slow level
  (medians 7.19 and 7.69 ms). Its 16-thread rows held in both sittings (this row: medians
  4.10-4.35 ms). Stage 3 sitting 2's clock witness read 103-114% throughout.
  Rows (ONNX Runtime CPU EP, the stage 3 inputs and models): int8 MatMulInteger S1 at M = 512
  decides; fp32 MatMul S1 at M = 512 is reported with the same measures.
  Arms, in two passes, the second in reverse order. Each arm and pass is several fresh sessions,
  pooled: A0 and AD {SESSIONS_A['A0']} sessions of {RUN_S['A0']:.0f} s, the others {SESSIONS_A_DEFAULT} of {RUN_S_DEFAULT:.0f} s, each timed after {WARMUP_A} warmups.
  Several, because in stage 3 sitting 2 a row's level held for a whole session: one session per
  arm could sit at one level and hide the other.
    A0   8 threads, default scheduling
    A1   8 threads pinned to 8 distinct physical cores (one SMT sibling of each)
    A3   8 threads pinned to 7 cores: both siblings of one core and one sibling of six others
    A2   8 threads pinned to 4 cores x 2 siblings
    A16  16 threads, default scheduling
    AD   ORT's own default, intra_op_num_threads 0. ORT's documentation says the default makes one
         thread per physical core and pins them. A functional smoke before this pre-registration (a
         64 x 64 model, no timing) read back 7 pool threads plus the caller, every one with all 16
         logical CPUs in its affinity mask and no CPU sets: this build does not pin by default.
  ORT pins its 7 pool threads (session.intra_op_thread_affinities); the calling thread, which also
  computes, is pinned to the arm's first CPU with SetThreadAffinityMask. The CPUs come from
  GetLogicalProcessorInformation at run time and are printed in the log.
  Recorded: every rep's time and start; once a second, every logical CPU's % Processor Time and
  % Processor Performance, and the total clock; once per session, after the warmups, every ORT thread's
  affinity and ideal processor, read back (GetThreadGroupAffinity, GetThreadIdealProcessorEx), which
  checks the pinning and counts the threads ORT's default creates. The sampler is one Get-Counter
  process: a small load outside ORT, present in every arm.
  Measures, per arm and pass (its sessions pooled), against the pooled 10th percentile p10:
    slow reps > {SLOW:.2f} x p10; fast reps <= {FAST:.2f} x p10
    bimodal   if both slow and fast reps are at least {MODE_FRAC:.0%} of the pooled reps
    unimodal  if p90 / p10 <= {UNIMODAL:.2f}; otherwise broad
    levels    the median of the fast reps and of the slow reps
    per second: the slow share of the reps started in it, the logical CPUs at >= {BUSY:.0f}% busy, and
    the physical cores with both siblings busy ("doubled")
  Candidates and what each predicts:
    ORT-thread doubling (the scheduler puts two of ORT's 8 threads on one core's two siblings, and
      the GEMM waits on that core): A0 bimodal; A1 unimodal at A0's fast level; A3 unimodal near
      A0's slow level; A2 unimodal, no faster than A3; AD like A0 (it does not pin); in A0,
      seconds with a doubled core carry more slow reps.
    Scheduler migration (ORT's threads moving between logical CPUs without doubling up): A0
      bimodal; A1 unimodal; A3's level unrelated to A0's slow level; no such association.
    A thread outside ORT on a free sibling (a system or sampler thread sharing an ORT thread's
      core): A1 not unimodal either, since its 8 free siblings stay open to it; A2 unimodal, since
      its 4 cores are full and 4 other cores idle.
    Clock or power state: pinning does not help, so neither A1 nor A2 is unimodal, and slow
      seconds run at a lower total clock.
    Core parking: on AC this plan's minimum parked-cores setting is 100% (printed in the log), so
      parking is not expected to act.
  Rules (the reference row):
    Not reproduced if A0 is bimodal in neither pass: no attribution.
    ORT's own placement, not the clock and not a thread outside ORT, if A0 is bimodal and A1 is
    unimodal in both passes. Then:
      ORT-thread doubling if A3 is unimodal within {LEVEL_MATCH:.0%} of A0's slow level in both passes, and in
      A0 the slow share in seconds with a doubled core exceeds that in seconds without by >= {ASSOC:.2f}.
      If A3 matches but that association is absent or untestable: placement, doubling versus
      migration not separated. If A3 does not match: placement, not a single doubled core; not
      separated.
    If A1 is not unimodal in a pass: the clock is implicated if the total clock in slow seconds
      (slow share >= 0.5, over A0 and A1) is at least {CLOCK_POINTS:.0f} points below that in fast seconds (<= 0.1).
      Otherwise, if A2 is unimodal in both passes: a thread outside ORT on a free sibling, not
      separated from an effect of the number of busy cores (A1 keeps 8 busy, A2 4). Otherwise
      non-attribution.
  Controls and witnesses (printed, not deciding): each pinned arm's busy logical CPUs must be its
  own; A16 and AD with their mode and median; every session's median; the unpinned sessions (A0
  and AD) with a core holding two of their threads' ideal processors against those without.

(B) DirectML's level shifts
  Seen: stage 3 sitting 2's DirectML rows held one level within a pass and another in the next
  (fp32 S2 at M = 2048: medians 141.71 ms, then 107.02). Sitting 2 opened one session per row.
  Sitting 1 kept one session per arm and shape open for the whole sitting, and its DirectML rows
  held (the same row: 152.02 and 150.79 ms).
  Rows (the stage 3 models, device 0, CPU fallback disabled): fp32 MatMul S2 at M = 2048 decides;
  fp16 MatMul S1 at M = 2048 is reported.
    B1  {SESSIONS} fresh sessions one after another: open, {WARMUP_B} warmups, {REPS_B} reps, the GPU memory witness
        (this process's GPU Process Memory, Dedicated and Shared Usage), close
    B2  one session, {WARMUP_B} warmups, then {BLOCKS} blocks of {REPS_B} reps with the memory witness between blocks
        (B1's gaps also hold a session's close and open)
  A session's or block's level is the median of its reps; it is steady if p90 / p10 <= {STEADY:.2f}.
  Candidates: allocation placement (where a session's weights land, dedicated or shared memory);
  GPU clock or power state; the driver's queue.
  Rules (the reference row):
    Reproduced if the steady B1 sessions' levels span max / min >= {SHIFT:.2f}. If not: no attribution.
    Per-session state if reproduced and B2's block levels span < {SHIFT:.2f} (one session holds its level).
      Allocation placement, in addition, if the {EDGE} slowest and {EDGE} fastest steady sessions do not
      overlap in Dedicated or in Shared Usage. Otherwise the per-session cause is unattributed.
    If B2's blocks also span >= {SHIFT:.2f}: not per-session; clock or power state and the driver's queue
      are not separated (non-attribution).

Written predictions (stated expectations; the rules decide, these do not)
  P1 (A) reproduces: A0 bimodal in the int8 row (across its sessions more than within them), A16
     unimodal.
  P2 (A) ORT-thread doubling: A1 unimodal near 4.2 ms; A3 at 7-8.5 ms, near A0's slow level;
     A2 about 8 ms (16 threads ran no faster than 8, so a core running two halves each one's
     share). The association is the weakest part: the flips come in runs of a few reps, tens of
     ms, far shorter than the 1 s busy map, so "not separated" is a likely outcome of that sub-rule.
  P3 The fp32 row shows the same pattern, less cleanly (stage 3's fp32 8-thread rows were broad).
  P4 (B) reproduces across fresh sessions, and one long-lived session holds its level.
  P5 (B) The memory witness separates slow sessions from fast ones: the least certain call here.
  P6 AD behaves like A0 (bimodal), since the smoke read back no pinning: 8 threads, placed freely.

Unverified by design: per-rep thread placement (the busy map is per second, the thread witness
once per session, and an ideal processor is a preference, not a placement); GPU clocks (no counter
for the 780M's clock here); the driver's queue; other CPU stacks and DirectML IO binding.
"""


def prereg() -> int:
    from llm_prefill_bench import INPUTS, sha
    print(PREREG)
    cs = cores()
    print("TOPOLOGY_JSON " + json.dumps({"cores": cs, "parking_ac_percent": parking(),
                                         "arms": {a: arm_cpus(a, cs) for a in ARMS_A}}))
    print("PINS inputs manifest", sha(INPUTS / "manifest.json"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.split("\n")
    n = sum(1 for s in dirty if s.strip())
    print(f"git HEAD {head}" + (f" (+{n} uncommitted paths, this study's own files)" if n else ""))
    print("PREREG_JSON " + json.dumps({
        "rows_a": ROWS_A, "arms_a": ARMS_A, "sessions_a": SESSIONS_A, "sessions_a_default": SESSIONS_A_DEFAULT,
        "run_s": RUN_S, "run_s_default": RUN_S_DEFAULT, "warmup_a": WARMUP_A, "slow": SLOW, "fast": FAST,
        "mode_frac": MODE_FRAC, "unimodal": UNIMODAL, "level_match": LEVEL_MATCH, "busy": BUSY, "assoc": ASSOC,
        "clock_points": CLOCK_POINTS, "rows_b": ROWS_B, "sessions_b": SESSIONS, "blocks": BLOCKS, "reps_b": REPS_B,
        "warmup_b": WARMUP_B, "steady": STEADY, "shift": SHIFT, "edge": EDGE, "git_head": head}))
    return 0


# ---------------------------------------------------------------- selftest (no timing)

def selftest() -> int:
    rng = np.random.default_rng(3)
    cs = [[2 * k, 2 * k + 1] for k in range(8)]
    for name, want in (("doubling", "ORT-thread doubling"), ("clock", "clock is implicated"),
                       ("foreign", "a thread outside ORT on that sibling")):
        out, samples, t = [], [], 1_000_000
        for p in (1, 2):
            for arm in ARMS_A:
                cpus = arm_cpus(arm, cs)
                for si in range(SESSIONS_A.get(arm, SESSIONS_A_DEFAULT)):
                    slow = si % 2 == 0 and (arm == "A0" or (name != "doubling" and arm == "A1"))
                    w0 = t
                    ms, ts = [], []
                    for _sec in range(int(RUN_S.get(arm, RUN_S_DEFAULT))):
                        busy = cpus or ([0, 1] + [c[0] for c in cs[1:7]] if slow else [c[0] for c in cs])
                        for _ in range(100):
                            base = 7.2 if slow or arm in ("A3", "A2") else 4.2
                            ts.append(t + 5)
                            ms.append(float(base * (1 + 0.02 * rng.standard_normal())))
                            t += 10
                        smp = {"t": t, "c_total": 100.0 if (name == "clock" and slow) else 110.0}
                        smp.update({f"b{i}": (100.0 if i in busy else 2.0) for i in range(16)})
                        samples.append(smp)
                    pinned = cpus or ([0, 1] + [2 * k for k in range(1, 7)] if slow else [2 * k for k in range(8)])
                    pool = [{"cpus": [c] if cpus else list(range(16)), "ideal": c, "cpu_sets": 0} for c in pinned[1:]]
                    out.append("ARM_JSON " + json.dumps({"row": "int8 S1 M=512", "arm": arm, "pass": p, "session": si,
                                                         "threads": 8, "cpus": cpus, "pool": pool,
                                                         "main": [{"cpus": pinned[:1], "ideal": pinned[0], "cpu_sets": 0}],
                                                         "w0": w0, "w1": t, "ms": ms, "t": ts}))
                t += 2000
        text = "\n".join(["TOPOLOGY_JSON " + json.dumps({"cores": cs, "parking_ac_percent": {}})] + out
                         + ["SAMPLE " + json.dumps(s) for s in samples])
        print(f"---- synthetic (A): {name}")
        got = verdict_cpu(text)
        assert any(want in g for g in got), (name, got[-3:])
    for name, want in (("placement", "allocation placement"), ("within", "NON-ATTRIBUTION")):
        out = []
        for kind, s, M in ROWS_B:
            row = f"{kind} {s} M={M}"
            for i in range(SESSIONS):
                slow = i % 3 == 0
                lvl = 141.0 if slow else 107.0
                out.append("SESSION_JSON " + json.dumps({"row": row, "i": i, "create_ms": 900.0,
                           "ms": [lvl * (1 + 0.005 * k) for k in range(REPS_B)], "providers": ["DmlExecutionProvider"],
                           "dedicated_mb": 50.0 if slow else 230.0, "shared_mb": 200.0}))
            for b in range(BLOCKS):
                lvl = 107.0 if name == "placement" else (141.0 if b % 2 else 107.0)
                out.append("BLOCK_JSON " + json.dumps({"row": row, "b": b, "ms": [lvl] * REPS_B,
                                                       "dedicated_mb": 230.0, "shared_mb": 200.0}))
        print(f"---- synthetic (B): {name}")
        got = verdict_dml("\n".join(out))
        assert any(want in g for g in got), (name, got[-3:])
    print("selftest: all five synthetic verdicts as expected")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("topology", "prereg", "cpu", "dml", "selftest"):
        sub.add_parser(c)
    v = sub.add_parser("verdict")
    v.add_argument("cpu_log")
    v.add_argument("dml_log")
    a = ap.parse_args()
    if a.cmd == "verdict":
        return verdict(Path(a.cpu_log), Path(a.dml_log))
    return {"topology": topology, "prereg": prereg, "cpu": cpu, "dml": dml, "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
