#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM study, Gemma 3 4B, pre-registration (b): J/GB of each chip's read stream, and whether the
package counters hold the GPU and the NPU.

The bar grew (locked decision 10): energy per token counts, from the package counters only. Before
any energy figure is compared, this asks what one GB read costs each chip above idle, and whether
\\Energy Meter(RAPL_Package0_PKG) sees the Radeon 780M's and the NPU's power at all.

Each arm is one reader process with stage 2's protocol (tools/concurrent_read_bw.py): it sets up,
warms up, prints READY, takes one start and one stop time on perf_counter_ns, loops until the stop
and reports. The coordinator samples package, core, CPU and GPU-engine counters with typeperf at
1 Hz, a 60 s idle before every arm and through its 60 s window, and aligns the rows by wall clock.
  R-cpu, R-dml, R-npu   stage 2's readers, unchanged (1 GiB reads)
  C1                    one Python thread spinning in a cache-resident loop
  G100, G50, G25        a DirectML fp16 chain of 32 MatMuls on 256 x 256 operands, looped
                        back to back, and duty-cycled to 50% and 25% of wall time
  N-c                   stage 3b's NPU bf16 S1 M = 2048 tile (S1-P), looped

    python tools/llm_energy.py prereg                    # the pre-registration text and pins
    python tools/llm_energy.py suite                     # the sitting (resnet_env17)
    python tools/llm_energy.py verdict <suite log>       # the mechanical verdict
    python tools/llm_energy.py reader --kind KIND        # spawned by suite
    python tools/llm_energy.py selftest                  # sleeping readers and synthetic rules; no chip
"""
import argparse
import contextlib
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import concurrent_read_bw as crb  # noqa: E402
from power_probe import CORES, PKG  # noqa: E402

CPU = r"\Processor(_Total)\% Processor Time"
GPU = r"\GPU Engine(*)\Utilization Percentage"
TMP = ROOT / "scratch" / "llm" / "energy"

ARMS = ("R-cpu", "R-dml", "R-npu", "C1", "G100", "G50", "G25", "N-c")
ORDER = [(1, a) for a in ARMS] + [(2, a) for a in ARMS[::-1]]      # two passes, the second mirrored
KIND = {"R-cpu": "cpu", "R-dml": "dml", "R-npu": "npu", "C1": "c1", "G100": "g100", "G50": "g50",
        "G25": "g25", "N-c": "nc"}
READ_ARMS = ("R-cpu", "R-dml", "R-npu")
CRB_KINDS = ("cpu", "dml", "npu")          # run by tools/concurrent_read_bw.py's own reader
IRON_KINDS = ("npu", "nc", "sleep_iron")   # run through scripts/research-iron.sh
DUTY = {"g100": 1.0, "g50": 0.5, "g25": 0.25}
DUTY_PERIOD_S = 0.1

IDLE_S = 60                  # idle samples before every arm
WINDOW_S = 60.0              # an arm's window, from the common start to the stop
TRIM_S = crb.TRIM_S          # 1 s dropped at each end, for the bytes and the power rows alike
LEAD_S = crb.LEAD_S          # the start is 3 s after the counters are up
TP_LEAD_S = 2.5              # typeperf starts after READY (its GPU instances are listed at start)
SETTLE_S = 10.0              # after an arm, before the next idle
SMOKE_S = 10.0               # the G100 fail-fast smoke's window, before the first idle
WARMUP = 3
READY_TIMEOUT_S = 300
RUN_TIMEOUT_S = 180
MIN_ROWS = 50                # an idle or a window with fewer valid power rows voids the arm
crb.WINDOW_S = WINDOW_S      # crb.window_stats reads the window length from its module

G_DIM, G_CHAIN = 256, 32
SEED = 20260924
NC_TILE = ("bf16", "S1-P", 2048)
NC_CHECK_ROWS = 64
NC_INSTS_SHA = "e67727d58b83dd9f996a765ef10d9924f56984bc21a655693d1d30f169c3ea86"    # 3b's NPU log
NPU_READ_INSTS_SHA = "c3aa45a3c6013bbcf1170e0cb9d68cb0a20ecbf063bc6eb2d52621e06314fe3a"  # stage 1, 2

# Pre-registered constants (v2 section 3)
GPU_MIN_W = 2.0              # GPU INSIDE: dR(G100) - dR(C1) >= max(2.0 W, 3 sigma_R), both passes
NPU_MIN_W = 1.0              # NPU INSIDE: dR(N-c) - dR(C1) >= max(1.0 W, 3 sigma_R), both passes
SIGMA_K = 3.0
PASS_AGREE = 0.10            # a read arm's J/GB must agree within 10% between passes
G100_BUSY_MIN = 80.0         # G100's GPU-engine witness, summed over its process's engines
IDLE_SHIFT_W = 2.0
PRED_JGB = {"R-cpu": (0.45, 0.70), "R-dml": (0.12, 0.35), "R-npu": (0.05, 0.20)}
PRED_NPU_MARGIN = (1.0, 4.0)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- readers (one per process)

def g_session(np):
    """The chain: x (256 x 256 fp16) times one orthogonal W, 32 times, IO-bound on the device."""
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(SEED)
    w = np.linalg.qr(rng.standard_normal((G_DIM, G_DIM)))[0].astype(np.float16)
    x = rng.standard_normal((G_DIM, G_DIM)).astype(np.float16)
    nodes, prev = [], "x"
    for i in range(G_CHAIN):
        out = "y" if i == G_CHAIN - 1 else f"h{i}"
        nodes.append(helper.make_node("MatMul", [prev, "w"], [out]))
        prev = out
    vi = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT16, [G_DIM, G_DIM])  # noqa: E731
    g = helper.make_graph(nodes, "g_chain", [vi("x")], [vi("y")], [numpy_helper.from_array(w, "w")])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_mem_pattern = False
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_session_config_entry("ep.dml.enable_graph_capture", "1")
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=[("DmlExecutionProvider", {"device_id": 0})])
    xv = ort.OrtValue.ortvalue_from_numpy(x, "dml", 0)
    yv = ort.OrtValue.ortvalue_from_shape_and_type([G_DIM, G_DIM], np.float16, "dml", 0)
    io = sess.io_binding()
    io.bind_ortvalue_input("x", xv)
    io.bind_ortvalue_output("y", yv)

    def one():
        sess.run_with_iobinding(io)
        io.synchronize_outputs()

    def check():
        ref = x.astype(np.float64)
        for _ in range(G_CHAIN):
            ref = ref @ w.astype(np.float64)
        y = yv.numpy().astype(np.float64)
        if not np.all(np.isfinite(y)):
            raise RuntimeError("the G chain's output is not finite")
        return {"rel_l2": float(np.linalg.norm(y - ref) / np.linalg.norm(ref))}

    ident = {"onnxruntime": ort.__version__, "providers": sess.get_providers(), "graph_capture": True,
             "chain": G_CHAIN, "dim": G_DIM}
    return one, check, ident, (sess, io, xv, yv)


def reader(kind: str) -> int:
    import numpy as np
    with contextlib.ExitStack() as stack:
        check, ident = (lambda: {}), {}
        if kind == "c1":
            state = [12345]

            def one():
                x = state[0]
                for _ in range(20000):
                    x = (x * 1103515245 + 12345) & 0x7FFFFFFF
                state[0] = x
            ident = {"loop": "20000 LCG steps per op, one thread"}
        elif kind in DUTY:
            one, check, ident, _keep = g_session(np)      # the session and its device buffers
        elif kind == "nc":
            import gc
            import ml_dtypes
            import llm_prefill_bench as pb
            from ignite_xdna.runtime.driver import XrtSiliconHarness
            d = pb.tdir(pb.tile(*NC_TILE))
            if sha(d / "insts.bin") != NC_INSTS_SHA:
                raise SystemExit(f"{d.name}: insts.bin differs from stage 3b's")
            data = pb.load(["x4", "w_S1"])
            bf16 = ml_dtypes.bfloat16
            xa = data["x4"][:NC_TILE[2]].astype(bf16)
            wb = data["w_S1"].astype(bf16)
            raw = lambda a: np.ascontiguousarray(a).view(np.uint16)  # noqa: E731
            h = stack.enter_context(XrtSiliconHarness(0))
            # the GEMM and its registry go before the hardware context closes (LIFO)
            reg, gs = {}, []
            stack.callback(gc.collect)
            stack.callback(reg.clear)
            stack.callback(lambda: [g.close() for g in gs])
            gs.append(pb.NpuGemm(h, reg, d, raw(xa), [raw(wb)], NC_TILE[2] * 4096 * 4))

            def one():
                gs[0].step(0, True)

            def check():
                y = gs[0].read(0, np.float32, 4096)[:NC_CHECK_ROWS]
                own = pb.reference(xa[:NC_CHECK_ROWS].astype(np.float32), wb.astype(np.float32))
                rel = pb.errors(y, own)["rel_l2"]
                if not rel <= pb.BF16_ACC_MAX:
                    raise RuntimeError(f"N-c output rel-L2 {rel} > {pb.BF16_ACC_MAX}")
                return {"check_rows": NC_CHECK_ROWS, "rel_l2_own": rel}
            ident = {"artifact": d.name, "insts_sha256": NC_INSTS_SHA, "xclbin_sha256": sha(d / "final.xclbin")}
        else:
            # harness self-test only: sleeps instead of working, touches no chip
            def one():
                time.sleep(0.02)
        for _ in range(WARMUP):
            one()
        ident.update(check())
        print("READY " + json.dumps({"kind": kind, "pid": os.getpid(), "python": sys.version.split()[0], **ident}),
              flush=True)
        start, stop = (int(v) for v in sys.stdin.readline().split())
        while time.perf_counter_ns() < start - 3_000_000:
            time.sleep(0.001)
        while time.perf_counter_ns() < start:
            pass
        duty, period = DUTY.get(kind, 1.0), int(DUTY_PERIOD_S * 1e9)
        w0, w1 = start + int(TRIM_S * 1e9), stop - int(TRIM_S * 1e9)
        ops = busy = 0
        in_ops = 0.0
        durs = []
        while True:
            t0 = time.perf_counter_ns()
            if t0 >= stop:
                break
            if duty < 1.0:
                k = (t0 - start) // period
                if t0 >= start + k * period + int(duty * period):
                    time.sleep(max(0, min(start + (k + 1) * period, stop) - t0) / 1e9)
                    continue
            one()
            t1 = time.perf_counter_ns()
            lo, hi = max(t0, w0), min(t1, w1)
            if hi > lo:
                busy += hi - lo
                in_ops += (hi - lo) / (t1 - t0)
            ops += 1
            durs.append(t1 - t0)
        print("READER_JSON " + json.dumps({"kind": kind, "ops": ops, "ops_in_window": in_ops,
                                           "busy_fraction": busy / (w1 - w0),
                                           "median_op_us": statistics.median(durs) / 1e3 if durs else None,
                                           "window_s": (stop - start) / 1e9}), flush=True)
    return 0


# ---------------------------------------------------------------- coordinator

class Reader:
    """One reader process. Its command is logged relative to the repository, the interpreter by env."""

    def __init__(self, kind: str):
        self.kind, self.lines, self.ready = kind, [], threading.Event()
        script = "tools/concurrent_read_bw.py" if kind in CRB_KINDS else "tools/llm_energy.py"
        args = [script, "reader", "--kind", kind]
        if kind in IRON_KINDS:
            import shutil
            self.shown = ["bash", "scripts/research-iron.sh", *args]
            cmd = [shutil.which("bash"), "scripts/research-iron.sh", *args]
        else:
            self.shown = [f"python[{os.environ.get('CONDA_DEFAULT_ENV', '?')}]", *args]
            cmd = [sys.executable, *args]
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


def typeperf(samples: int, path: Path) -> subprocess.Popen:
    # -sc ends it by itself, so no kill races the last row; -y answers the overwrite prompt
    return subprocess.Popen(["typeperf", PKG, *CORES, CPU, GPU, "-si", "1", "-sc", str(samples), "-y",
                             "-o", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def finish(proc: subprocess.Popen, timeout: float = 60):
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        proc.wait(timeout=15)


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return 0.0


def read_csv(path: Path, pid=None) -> list:
    """[epoch s, package mW, sum of Core0-7 mW, CPU %, GPU busy summed over all engines, and over
    pid's engines] per typeperf row. A row with a blank package or core field (typeperf's first)
    is skipped; a blank GPU field counts 0."""
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    head = [h.lower() for h in rows[0]]
    col = lambda p: next(i for i, h in enumerate(head) if h.endswith(p.lower().lstrip("\\")))  # noqa: E731
    ipkg, icores, icpu = col(PKG), [col(c) for c in CORES], col(CPU)
    igpu = [i for i, h in enumerate(head) if "\\gpu engine(" in h]
    ipid = [i for i in igpu if f"gpu engine(pid_{pid}_" in head[i]] if pid is not None else []
    out = []
    for r in rows[1:]:
        try:
            stamp = datetime.strptime(r[0], "%m/%d/%Y %H:%M:%S.%f").timestamp()
            pkg, cores, cpu = float(r[ipkg]), sum(float(r[i]) for i in icores), float(r[icpu])
        except (ValueError, IndexError):
            continue
        out.append([round(stamp, 3), round(pkg, 1), round(cores, 1), round(cpu, 2),
                    round(sum(_num(r[i]) for i in igpu), 2), round(sum(_num(r[i]) for i in ipid), 2)])
    return out


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def idle(tag: str, seconds: int = IDLE_S) -> list:
    path = TMP / f"{tag}_idle.csv"
    p = typeperf(seconds + 1, path)
    finish(p, seconds + 60)
    return read_csv(path)


def window(tag: str, kind: str, seconds: float = WINDOW_S) -> dict:
    """Start the reader, wait for READY, start the counters, hand out the start and stop, collect."""
    r = Reader(kind)
    rec = {"kind": kind, "cmd": " ".join(r.shown)}
    tp = None
    try:
        if not r.ready.wait(READY_TIMEOUT_S) or r.json_line("READY") is None:
            raise RuntimeError(f"{kind} not READY")
        ready = r.json_line("READY")
        path = TMP / f"{tag}_window.csv"
        tp = typeperf(int(TP_LEAD_S + LEAD_S + seconds + 6), path)
        time.sleep(TP_LEAD_S)
        tt, pc = time.time(), time.perf_counter_ns()
        start = pc + int(LEAD_S * 1e9)
        stop = start + int(seconds * 1e9)
        r.proc.stdin.write(f"{start} {stop}\n")
        r.proc.stdin.flush()
        r.proc.wait(timeout=RUN_TIMEOUT_S + seconds)
        r.pump.join(timeout=30)
        finish(tp)
        tp = None
        rec.update(ready=ready, wall0=tt + (start - pc) / 1e9, window_s=seconds,
                   rows=read_csv(path, ready["pid"]))
    finally:
        if r.proc.poll() is None:
            r.proc.kill()
        if tp is not None:
            finish(tp, 0)
        print(f"READER_CMD {kind} {rec['cmd']}", flush=True)
        for s in r.lines:
            if not s.startswith(("READER_JSON ", "READY ")):
                print(f"{kind}| {s}", flush=True)
    raw = r.json_line("READER_JSON")
    if r.proc.returncode or raw is None:
        rec["failed"] = f"exit {r.proc.returncode}"
    rec["reader"] = raw
    return rec


def pins() -> dict:
    import npu_read_bw_probe as probe
    import llm_prefill_bench as pb
    d_read = probe.BUILD / probe.name(*crb.NPU_ARTIFACT)
    d_nc = pb.tdir(pb.tile(*NC_TILE))
    got = {"npu_read_insts": sha(d_read / "insts.bin"), "npu_read_xclbin": sha(d_read / "probe.xclbin"),
           "nc_insts": sha(d_nc / "insts.bin"), "nc_xclbin": sha(d_nc / "final.xclbin"),
           "readbw_model": sha(crb.READBW_MODEL), "prefill_manifest": sha(pb.INPUTS / "manifest.json")}
    bad = [k for k, want in (("npu_read_insts", NPU_READ_INSTS_SHA), ("nc_insts", NC_INSTS_SHA)) if got[k] != want]
    return {"artifacts": [d_read.name, d_nc.name], "sha256": got, "mismatch": bad}


def suite() -> int:
    from silicon_probe_record import witness
    import llm_prefill_bench as pb
    TMP.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    pb.host_gate()
    p = pins()
    print("PINS_JSON " + json.dumps(p), flush=True)
    if p["mismatch"]:
        print("PIN_MISMATCH", p["mismatch"], flush=True)
        return 3
    print("ORT_PYTHON env", os.environ.get("CONDA_DEFAULT_ENV", "?"), flush=True)
    crb.gpu_engines()
    # the fail-fast smoke: G100 for 10 s before anything is measured; below the witness, stop
    print(f"\nSMOKE_BEGIN {utc()}", flush=True)
    witness()
    try:
        s = window(f"{stamp}_smoke", "g100", SMOKE_S)
    except Exception as e:
        s = {"failed": f"{type(e).__name__}: {str(e)[:300]}"}
    rows = inside(s.get("rows", []), s.get("wall0", 0), SMOKE_S)
    busy = statistics.fmean(r[5] for r in rows) if rows else 0.0
    ok = "failed" not in s and len(rows) >= 3 and busy >= G100_BUSY_MIN
    print("SMOKE_JSON " + json.dumps({"ok": ok, "gpu_pid_busy": busy, "rows": len(rows), "failed": s.get("failed"),
                                      "ready": s.get("ready"), "reader": s.get("reader")}), flush=True)
    witness()
    if not ok:
        print(f"SMOKE_FAIL G100 busy {busy:.1f}% over {len(rows)} rows (< {G100_BUSY_MIN}% or failed): the sitting "
              "stops before any idle", flush=True)
        return 3
    time.sleep(SETTLE_S)
    # xrt-smi idle before and after every arm: each arm's closing witness is the next one's opening,
    # and it runs before the settle, never inside an idle
    for n, (pas, arm) in enumerate(ORDER):
        print(f"\nARM_BEGIN {n} pass {pas} {arm} {utc()}", flush=True)
        tag = f"{stamp}_{n:02d}_{arm}"
        idle_rows = idle(tag)
        rec = {"n": n, "pass": pas, "arm": arm, "idle_utc": utc(), "idle_rows": idle_rows}
        try:
            rec.update(window(tag, KIND[arm]))
        except Exception as e:                       # recorded; the verdict marks the arm INCOMPLETE
            rec["failed"] = f"{type(e).__name__}: {str(e)[:300]}"
        print("ARM_JSON " + json.dumps(rec), flush=True)
        st = arm_stats(rec)
        print("ARM_SUMMARY", n, pas, arm, " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                     for k, v in st.items()), flush=True)
        witness()
        time.sleep(SETTLE_S)
    crb.gpu_engines()
    return 0


# ---------------------------------------------------------------- the numbers, from the rows

def inside(rows, wall0: float, seconds: float) -> list:
    """Rows whose whole second lies in [start + TRIM_S, stop - TRIM_S] (a row stamped t holds the
    second that ends at t, as tools/energy_sitting.py reads typeperf)."""
    return [r for r in rows if wall0 + TRIM_S + 1.0 <= r[0] <= wall0 + seconds - TRIM_S]


def means(rows) -> dict:
    f = lambda i: statistics.fmean(r[i] for r in rows)  # noqa: E731
    return {"pkg_w": f(1) / 1e3, "cores_w": f(2) / 1e3, "cpu": f(3), "gpu_all": f(4), "gpu_pid": f(5), "n": len(rows)}


def arm_stats(rec: dict) -> dict:
    """dP, dCores and dR (R = package - cores) against the arm's own idle; J/GB for a read arm."""
    if rec.get("failed") or not rec.get("rows") or rec.get("reader") is None:
        return {"ok": False, "why": rec.get("failed") or "no window"}
    win = inside(rec["rows"], rec["wall0"], rec["window_s"])
    if len(rec["idle_rows"]) < MIN_ROWS or len(win) < MIN_ROWS:
        return {"ok": False, "why": f"rows idle {len(rec['idle_rows'])} window {len(win)} < {MIN_ROWS}"}
    i, w = means(rec["idle_rows"]), means(win)
    out = {"ok": True, "idle_pkg_w": i["pkg_w"], "idle_r_w": i["pkg_w"] - i["cores_w"], "idle_cpu": i["cpu"],
           "idle_gpu_all": i["gpu_all"], "idle_pkg_sd_w": statistics.pstdev(r[1] for r in rec["idle_rows"]) / 1e3,
           "dP": w["pkg_w"] - i["pkg_w"], "dC": w["cores_w"] - i["cores_w"],
           "dR": (w["pkg_w"] - w["cores_w"]) - (i["pkg_w"] - i["cores_w"]), "cpu": w["cpu"],
           "gpu_all": w["gpu_all"], "gpu_pid": w["gpu_pid"], "rows": len(win)}
    raw = rec["reader"]
    if rec["arm"] in READ_ARMS:
        st = crb.window_stats(raw["bytes"], raw["t0_us"], raw["t1_us"])
        if not st["gbps"] > 0:
            return {"ok": False, "why": "no bytes read in the window"}
        out.update(gbps=st["gbps"], coverage=st["coverage"], jgb=out["dP"] / st["gbps"])
    else:
        out.update(busy_fraction=raw["busy_fraction"], ops_in_window=raw["ops_in_window"])
    return out


def evaluate(arms: dict) -> dict:
    """arms: {(pass, arm): arm_stats}. The v2 rules, mechanically."""
    ok = {k: v for k, v in arms.items() if v.get("ok")}
    idle_r = [v["idle_r_w"] for v in ok.values()]
    sigma = statistics.stdev(idle_r) if len(idle_r) >= 2 else float("nan")
    t_gpu, t_npu = max(GPU_MIN_W, SIGMA_K * sigma), max(NPU_MIN_W, SIGMA_K * sigma)
    res = {"sigma_r": sigma, "t_gpu": t_gpu, "t_npu": t_npu, "gpu": {}, "npu": {}, "reads": {}, "problems": []}
    for key in [(p, a) for p in (1, 2) for a in ARMS]:
        if key not in ok:
            res["problems"].append(f"pass {key[0]} {key[1]}: " + (arms[key].get("why") if key in arms else "missing"))
    for p in (1, 2):
        g = [ok.get((p, a)) for a in ("C1", "G100", "G50", "G25")]
        if None in g:
            res["gpu"][p] = {"state": None}
        else:
            c1, g100, g50, g25 = g
            margin = g100["dR"] - c1["dR"]
            state = ("VOID" if g100["gpu_pid"] < G100_BUSY_MIN
                     else margin >= t_gpu and g25["dR"] < g50["dR"] < g100["dR"])
            res["gpu"][p] = {"state": state, "margin": margin, "busy": g100["gpu_pid"],
                             "dR": [g25["dR"], g50["dR"], g100["dR"]]}
        n = [ok.get((p, a)) for a in ("C1", "N-c")]
        if None in n:
            res["npu"][p] = {"state": None}
        else:
            margin = n[1]["dR"] - n[0]["dR"]
            res["npu"][p] = {"state": margin >= t_npu, "margin": margin}

    def outcome(d):
        states = [d[p]["state"] for p in (1, 2)]
        if False in states:
            return "NOT SHOWN"
        if None in states:
            return "INCOMPLETE"
        if "VOID" in states:
            return "NOT SHOWN (G100 void)"
        return "INSIDE"
    res["GPU"], res["NPU"] = outcome(res["gpu"]), outcome(res["npu"])
    for a in READ_ARMS:
        js = [ok[(p, a)]["jgb"] for p in (1, 2) if (p, a) in ok]
        if len(js) < 2:
            res["reads"][a] = {"state": "INCOMPLETE", "jgb": js}
            continue
        m = statistics.fmean(js)
        res["reads"][a] = {"state": "OK" if abs(js[0] - js[1]) / m <= PASS_AGREE else "INCOMPLETE", "jgb": js,
                           "mean": m}
    return res


def labels(gpu: str, npu: str) -> list:
    g_in, n_in = gpu == "INSIDE", npu == "INSIDE"
    if "INCOMPLETE" in (gpu, npu):
        return ["an INCOMPLETE outcome labels nothing: every energy comparison it touches is UNDECIDED"]
    out = []
    if g_in and n_in:
        out.append("NPU vs DirectML: comparable, labelled. The controls show a component scaling with each chip's "
                   "work is inside the package, not that all of it is.")
    elif n_in:
        out.append("NPU vs DirectML: DirectML energy is 'package-reported, may be low'. An NPU WIN over DirectML "
                   "stands; an NPU LOSS to it is UNDECIDED.")
    elif g_in:
        out.append("NPU vs DirectML: an NPU WIN is UNDECIDED; an NPU LOSS stands.")
    else:
        out.append("NPU vs DirectML: every comparison is UNDECIDED, in either direction.")
    out.append("NPU vs CPU (the cores are package 0's sub-domains, inside by construction): "
               + ("comparable, labelled." if n_in else "an NPU WIN is UNDECIDED; an NPU LOSS stands."))
    out.append("On every figure: completeness is never shown (package counters only), and DRAM power is outside "
               "the package; shared DRAM pulls ratios toward 1 and undercounts a compaction's saving.")
    return out


def predictions(res: dict) -> list:
    out = []
    for i, a in enumerate(READ_ARMS, 1):
        lo, hi = PRED_JGB[a]
        r = res["reads"][a]
        if r["state"] != "OK":
            out.append((f"P{i}", f"{a} {lo}-{hi} J/GB", "UNSCORED (arm INCOMPLETE)"))
        else:
            out.append((f"P{i}", f"{a} {lo}-{hi} J/GB", f"{'HIT' if lo <= r['mean'] <= hi else 'MISS'} "
                                                         f"({r['mean']:.3f})"))
    g = res["GPU"]
    out.append(("P4", "GPU INSIDE", "UNSCORED (INCOMPLETE)" if g == "INCOMPLETE" else
                ("HIT" if g == "INSIDE" else f"MISS ({g})")))
    n = res["NPU"]
    if n == "INCOMPLETE":
        out.append(("P5", "NPU INSIDE, margin 1-4 W", "UNSCORED (INCOMPLETE)"))
    else:
        m = statistics.fmean(res["npu"][p]["margin"] for p in (1, 2) if "margin" in res["npu"][p])
        lo, hi = PRED_NPU_MARGIN
        out.append(("P5", "NPU INSIDE, margin 1-4 W", f"{'HIT' if n == 'INSIDE' and lo <= m <= hi else 'MISS'} "
                                                      f"({n}, mean margin {m:.2f} W)"))
    return out


def verdict(log: Path) -> int:
    text = log.read_text(encoding="utf-8")
    lines = text.splitlines()
    smoke = next((json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith("SMOKE_JSON ")), None)
    recs = [json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith("ARM_JSON ")]
    arms = {(r["pass"], r["arm"]): arm_stats(r) for r in recs}
    res = evaluate(arms)
    if smoke is None or not smoke["ok"]:
        res["problems"].append("the G100 smoke did not pass")
    else:
        print(f"Smoke: G100 GPU busy {smoke['gpu_pid_busy']:.1f}% over {smoke['rows']} rows (>= {G100_BUSY_MIN}%)")
    if "TIMEOUT" in text:
        res["problems"].append("the sitting timed out")
    pin = next((json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith("PINS_JSON ")), None)
    if pin is None or pin["mismatch"]:
        res["problems"].append(f"pins: {pin and pin['mismatch']}")
    print(f"\n{'pass':>4} {'arm':6} {'idle W':>7} {'idle R':>7} {'dPKG W':>7} {'dCores':>7} {'dR W':>6} {'CPU%':>5} "
          f"{'GPUpid':>6} {'GB/s':>6} {'J/GB':>6}  witness")
    medians = []
    for r in recs:
        st = arms[(r["pass"], r["arm"])]
        if not st["ok"]:
            print(f"{r['pass']:>4} {r['arm']:6} INCOMPLETE: {st['why']}")
            continue
        shift = ""
        if len(medians) >= 3 and abs(st["idle_pkg_w"] - statistics.median(medians)) > IDLE_SHIFT_W:
            shift = f"  idle shifted {st['idle_pkg_w'] - statistics.median(medians):+.2f} W"
        medians.append(st["idle_pkg_w"])
        extra = (f"{st['gbps']:6.2f} {st['jgb']:6.3f}  cov {st['coverage']:.3f}" if "jgb" in st
                 else f"{'':6} {'':6}  busy {st['busy_fraction']:.3f}")
        print(f"{r['pass']:>4} {r['arm']:6} {st['idle_pkg_w']:7.2f} {st['idle_r_w']:7.2f} {st['dP']:7.2f} "
              f"{st['dC']:7.2f} {st['dR']:6.2f} {st['cpu']:5.1f} {st['gpu_pid']:6.1f} {extra}"
              f"  idle CPU {st['idle_cpu']:.1f}% sd {st['idle_pkg_sd_w']:.2f} W GPU(all) {st['idle_gpu_all']:.1f}%{shift}")
    print(f"\nsigma_R over {sum(1 for v in arms.values() if v.get('ok'))} idle windows: {res['sigma_r']:.3f} W; "
          f"GPU line max({GPU_MIN_W}, 3 sigma) = {res['t_gpu']:.3f} W; NPU line max({NPU_MIN_W}, 3 sigma) = "
          f"{res['t_npu']:.3f} W")
    said = {True: "met", False: "not met", "VOID": "G100 void"}
    for p in (1, 2):
        g, n = res["gpu"][p], res["npu"][p]
        if g["state"] is not None:
            print(f"pass {p} GPU: dR(G100) - dR(C1) = {g['margin']:.3f} W; dR G25/G50/G100 = "
                  + " / ".join(f"{x:.3f}" for x in g["dR"]) + f"; G100 busy {g['busy']:.1f}% -> {said[g['state']]}")
        if n["state"] is not None:
            print(f"pass {p} NPU: dR(N-c) - dR(C1) = {n['margin']:.3f} W -> {said[n['state']]}")
    print(f"\nGPU {res['GPU']}\nNPU {res['NPU']}")
    for a in READ_ARMS:
        r = res["reads"][a]
        print(f"J/GB {a}: " + " / ".join(f"{j:.3f}" for j in r["jgb"])
              + (f", mean {r['mean']:.3f}" if "mean" in r else "") + f" -> {r['state']}")
    print("\nLabels")
    for s in labels(res["GPU"], res["NPU"]):
        print("  " + s)
    print("\nPredictions")
    for tag, what, score in predictions(res):
        print(f"  {tag} {what}: {score}")
    for p in res["problems"]:
        print("PROBLEM", p)
    print("VERDICT", "INCOMPLETE" if res["problems"] else "COMPLETE")
    return 2 if res["problems"] else 0


# ---------------------------------------------------------------- prereg and selftest

PREREG = f"""J/GB of each chip's read stream, and whether the package counters hold the GPU and the NPU
(LLM study, Gemma 3 4B pre-registration (b), locked decision 10), pre-registered before any sitting

Question
  The bar grew (locked decision 10): "energy per token and freeing the gpu both count". Energy
  comes from the package counters only (the user: "package counters only"; no wall meter, ADLX
  unplanned). Before any energy figure is compared:
  Q1 what one GB read costs each chip above idle (J/GB), on its fastest measured read path;
  Q2 whether \\Energy Meter(RAPL_Package0_PKG) holds the 780M's and the NPU's power, since a chip
     whose power the counter misses would look free.

Method
  Energy = the integral of (P_pkg - P_idle) over an arm's window. P_pkg is typeperf's
  \\Energy Meter(RAPL_Package0_PKG)\\Power (milliwatts, tools/power_probe.py) at 1 Hz, with the
  eight \\Energy Meter(RAPL_Package0_CoreN_CORE)\\Power, \\Processor(_Total)\\% Processor Time and
  \\GPU Engine(*)\\Utilization Percentage beside it.
  Every arm gets its own {IDLE_S} s idle baseline just before it, with nothing launched. An idle more
  than {IDLE_SHIFT_W} W from the median of the sitting's earlier idles (3 or more) is flagged, as
  tools/energy_sitting.py does; every arm is read against its own idle.
  Above-idle is the repo's convention: idle power is charged to no one. No bare wattage is quoted.
  Read arms (tools/concurrent_read_bw.py's readers, unchanged, with a {WINDOW_S:.0f} s common window
  instead of 20):
    R-cpu  ORT ReduceSum over Phase 1's 1 GiB fp32 initializer, {crb.CPU_THREADS} threads
    R-dml  the same graph on DirectML, device_id 0 (the Radeon 780M), CPU fallback disabled
    R-npu  stage 1's c4x2_m1024 artifact (4x2, 1 GiB per dispatch), insts SHA re-verified
  J/GB = mean(P_pkg - P_idle) / (GB/s read in the window). dSum(cores) and the residual
  R = P_pkg - Sum(Core0-7), as dR against the idle, are reported beside it.
  Controls (new, small):
    C1     one CPU thread in a cache-resident loop, matching a host submit thread. It gives the
           non-core rise one busy core causes.
    G100 / G50 / G25   a DirectML fp16 chain of {G_CHAIN} MatMuls on {G_DIM}x{G_DIM} operands, sized to
           stay in the 780M's 2 MB L2 (DRAM traffic minimal: INFERRED, not measured). Looped at
           100%, and duty-cycled to 50% and 25% of wall time.
           Witness: \\GPU Engine(pid_*)\\Utilization Percentage, summed over the process's
           engines; G100 needs >= {G100_BUSY_MIN:.0f}% or it is void.
    N-c    stage 3b's NPU bf16 S1 M = 2048 artifact (S1-P, SHA-pinned), looped. A compute-heavy
           NPU load; its DRAM traffic is not measured.

Rules (sigma_R = the sample standard deviation of the residual's per-window mean across the
sitting's idle windows)
  GPU INSIDE if both passes show dR(G100) - dR(C1) >= max({GPU_MIN_W} W, 3 sigma_R) and
    dR(G25) < dR(G50) < dR(G100). Otherwise GPU NOT SHOWN.
  NPU INSIDE if dR(N-c) - dR(C1) >= max({NPU_MIN_W} W, 3 sigma_R) in both passes. Otherwise NOT SHOWN.
  What each outcome means for later comparisons:
    GPU NOT SHOWN: DirectML energy is "package-reported, may be low". An NPU energy WIN over
      DirectML still stands, since DirectML's true figure can only be higher. An NPU LOSS to it
      is UNDECIDED.
    NPU NOT SHOWN: the reverse. An NPU WIN is UNDECIDED; a loss stands.
    Both NOT SHOWN: every NPU-vs-DirectML energy comparison is UNDECIDED, in either direction.
    The CPU arms' core power is inside the package by construction: the Core0-7 meters are
      package 0's core sub-domains (RAPL_Package0_CoreN). So NPU-vs-CPU comparisons follow the
      NPU's label alone.
    Both INSIDE: the controls show that a component scaling with each chip's work is inside the
      package. They do NOT show that all of it is. That, and DRAM power (outside the package),
      are labelled on every figure. DRAM is roughly shared when arms read the same bytes: it
      pulls ratios toward 1 and undercounts a compaction's saving.
  Package counters only (the user's decision): no wall meter, and ADLX stays unplanned.
  Completeness is therefore never shown. Every energy figure carries these labels, and the NOT
  SHOWN cases end UNDECIDED as above.

Order
  An idle before each arm, running {', '.join(ARMS)},
  then mirrored. Two passes, about 40 min in all. Every read arm's J/GB must agree within {PASS_AGREE:.0%} between passes, or that arm
  is INCOMPLETE. NPU checks: xrt-smi idle before and after every arm, and the device witness
  (tools/silicon_probe_record.py --device), as in stage 1.

Written predictions (stated expectations; the rules decide)
  P1 R-cpu {PRED_JGB['R-cpu'][0]}-{PRED_JGB['R-cpu'][1]} J/GB.
  P2 R-dml {PRED_JGB['R-dml'][0]}-{PRED_JGB['R-dml'][1]} J/GB, package-reported.
  P3 R-npu {PRED_JGB['R-npu'][0]}-{PRED_JGB['R-npu'][1]} J/GB.
  P4 GPU INSIDE.
  P5 NPU INSIDE, with dR(N-c) - dR(C1) at {PRED_NPU_MARGIN[0]:.0f}-{PRED_NPU_MARGIN[1]:.0f} W (the mean of the two passes).

Details v2 left open, fixed here before any run (named to the gate with this commit)
  D1 G's chain: x ({G_DIM}x{G_DIM} fp16, seeded) times one seeded orthogonal W, {G_CHAIN} times, so the values
     stay finite. x and y are bound on the device (IO binding) and DirectML graph capture is on
     (ep.dml.enable_graph_capture, present in this ORT build), so a loop iteration is one
     replay and a wait for it (synchronize_outputs), with no host copy. Reason: a DirectML round
     trip with host input and output costs 168.8 us (MEASURED, the sync stage), which could keep
     a chain of this size under the 80% witness (INFERRED). The output is checked finite once,
     after the warm-up; its rel-L2 against float64 is reported, not deciding.
  D2 G50 and G25: within each {DUTY_PERIOD_S * 1e3:.0f} ms period the chain loops for the first 50% or 25% of
     it and the process sleeps for the rest. The measured busy fraction is reported, not ruled.
  D3 C1: pure Python, 20000 LCG steps per op, one thread, in resnet_env17's interpreter.
  D4 N-c: 3b's own dispatch step (sync A to the device, run, wait, sync C back), one hardware
     context for the whole window. After the warm-up, the first {NC_CHECK_ROWS} output rows are checked
     against the product of the bf16-rounded inputs (rel-L2 <= 1e-4, stage 3's bound), or the
     reader fails.
  D5 Readers: stage 2's protocol. Each arm's reader starts after the idle; typeperf starts
     after READY (it lists GPU-engine instances when it starts), the common start is {LEAD_S:.0f} s after
     the counters, and the window is {WINDOW_S:.0f} s. Bytes count inside [start + {TRIM_S:.0f} s, stop - {TRIM_S:.0f} s], pro
     rata (crb.window_stats). A power row stamped t holds the second ending at t, so it counts
     if [t - 1, t] lies inside that span (energy_sitting's rule), by wall clock (time.time()
     and perf_counter_ns() read together once per arm). Rows with a blank package or core
     field are dropped; a blank GPU field counts 0.
  D6 An idle or a window with fewer than {MIN_ROWS} valid power rows voids that arm (INCOMPLETE); a
     reader that fails voids its arm. A rule that needs a voided arm is INCOMPLETE, unless a
     completed pass already fails it (then NOT SHOWN).
  D7 {SETTLE_S:.0f} s settle after each arm before the next idle. The host-load gate runs once at the start
     (busy cores < 2, no peer >= 0.5), as in stage 3; each idle's CPU % is reported.
  Reading of v2, flagged: "every arm's J/GB within 10%" can only apply to the three read arms (the
  controls read no GB). The controls' pass agreement is the "both passes" clause of the INSIDE
  rules.
Addition to v2 (named to the gate before the START REQUEST)
  A1 A {SMOKE_S:.0f} s G100 smoke at the sitting's start, before the first idle: if G100's GPU busy is
     below {G100_BUSY_MIN:.0f}%, or its reader fails, the sitting stops there and nothing is measured. It
     changes no arm and no rule; it keeps a void G100 from costing the whole sitting.

Unverified by design: DRAM power (outside the package); that all of any chip's power is inside
(completeness); the 780M's L2 residency of G's chain; N-c's DRAM traffic; any energy per token
(that is (c) and (d)); other read paths, thread counts and dtypes.
"""


def prereg() -> int:
    print(PREREG)
    p = pins()
    print("PINS")
    print("  R-npu", p["artifacts"][0], "insts_sha256", p["sha256"]["npu_read_insts"], "xclbin_sha256",
          p["sha256"]["npu_read_xclbin"])
    print("  N-c  ", p["artifacts"][1], "insts_sha256", p["sha256"]["nc_insts"], "xclbin_sha256",
          p["sha256"]["nc_xclbin"])
    print("  readbw model", crb.READBW_MODEL.relative_to(ROOT).as_posix(), "sha256", p["sha256"]["readbw_model"])
    print("  prefill inputs manifest sha256", p["sha256"]["prefill_manifest"])
    if p["mismatch"]:
        print("PIN_MISMATCH", p["mismatch"])
        return 3
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print("git HEAD", head, "(plus this log's own commit)")
    print("PREREG_JSON " + json.dumps({
        "order": ORDER, "idle_s": IDLE_S, "window_s": WINDOW_S, "trim_s": TRIM_S, "lead_s": LEAD_S,
        "settle_s": SETTLE_S, "smoke_s": SMOKE_S, "warmup": WARMUP, "duty": DUTY, "duty_period_s": DUTY_PERIOD_S,
        "g": [G_CHAIN, G_DIM], "seed": SEED, "nc_tile": NC_TILE, "min_rows": MIN_ROWS, "gpu_min_w": GPU_MIN_W,
        "npu_min_w": NPU_MIN_W, "sigma_k": SIGMA_K, "pass_agree": PASS_AGREE, "g100_busy_min": G100_BUSY_MIN,
        "idle_shift_w": IDLE_SHIFT_W, "pred_jgb": PRED_JGB, "pred_npu_margin": PRED_NPU_MARGIN,
        "pins": p["sha256"], "git_head": head}))
    return 0


def synthetic(pas: int, arm: str, dP: float, dC: float, busy: float = 90.0, gbps: float = 60.0,
              idle_r: float = 20.0) -> dict:
    """An ARM_JSON-shaped record whose stats come out as given (for the selftest's rules)."""
    t = 1_000_000.0
    idle_rows = [[t - 200 + i, 40000.0, 40000.0 - idle_r * 1e3, 5.0, 2.0, 0.0] for i in range(IDLE_S)]
    rows = [[t + i, 40000.0 + dP * 1e3, 40000.0 - idle_r * 1e3 + dC * 1e3, 50.0, 90.0, busy] for i in range(-3, 64)]
    rec = {"pass": pas, "arm": arm, "idle_rows": idle_rows, "rows": rows, "wall0": t, "window_s": WINDOW_S}
    if arm in READ_ARMS:
        n = int(gbps * WINDOW_S)                              # 1 GB reads, back to back
        step = int(WINDOW_S * 1e6 / n)
        rec["reader"] = {"bytes": 10 ** 9, "t0_us": [i * step for i in range(n)],
                         "t1_us": [(i + 1) * step for i in range(n)]}
    else:
        rec["reader"] = {"busy_fraction": 1.0, "ops_in_window": 100.0}
    return rec


def selftest() -> int:
    fails = []

    def expect(what, got, want):
        print(f"  {what}: {got}" + ("" if got == want else f"  (expected {want})"))
        if got != want:
            fails.append(what)

    # the row window: a row stamped t covers (t - 1, t]
    rows = [[100.0 + i, 0, 0, 0, 0, 0] for i in range(12)]
    expect("rows inside a 10 s window from 100.0", [r[0] for r in inside(rows, 100.0, 10.0)],
           [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0])

    def run(spec):
        arms = {}
        for (p, a), kw in spec.items():
            rec = synthetic(p, a, **kw)
            arms[(p, a)] = arm_stats(rec)
        return evaluate(arms)
    base = {"R-cpu": dict(dP=30.0, dC=27.0, gbps=60.0), "R-dml": dict(dP=15.0, dC=1.0, gbps=65.0),
            "R-npu": dict(dP=5.0, dC=0.5, gbps=45.0), "C1": dict(dP=9.0, dC=8.0),
            "G100": dict(dP=12.0, dC=1.0), "G50": dict(dP=6.0, dC=0.5), "G25": dict(dP=3.0, dC=0.25),
            "N-c": dict(dP=5.0, dC=1.0)}
    both = {(p, a): kw for p in (1, 2) for a, kw in base.items()}
    r = run(both)
    expect("both INSIDE: GPU", r["GPU"], "INSIDE")
    expect("both INSIDE: NPU", r["NPU"], "INSIDE")
    expect("J/GB R-cpu", round(r["reads"]["R-cpu"]["mean"], 3), 0.5)
    expect("sigma_R with identical idles", r["sigma_r"], 0.0)
    expect("P1 on 0.5 J/GB", predictions(r)[0][2], "HIT (0.500)")
    expect("P5 on margin 3 W", predictions(r)[4][2], "HIT (INSIDE, mean margin 3.00 W)")
    s = dict(both)
    s[(2, "G100")] = dict(dP=12.0, dC=1.0, busy=70.0)
    expect("G100 at 70% in pass 2", run(s)["GPU"], "NOT SHOWN (G100 void)")
    s = dict(both)
    s[(1, "G50")] = dict(dP=2.0, dC=0.5)
    expect("G25 >= G50 in pass 1", run(s)["GPU"], "NOT SHOWN")
    s = dict(both)
    s[(1, "N-c")] = dict(dP=1.5, dC=1.0)
    expect("N-c below C1 in pass 1", run(s)["NPU"], "NOT SHOWN")
    s = dict(both)
    del s[(2, "C1")]
    r = run(s)
    expect("C1 missing in pass 2: GPU", r["GPU"], "INCOMPLETE")
    expect("C1 missing in pass 2: problems", len(r["problems"]), 1)
    s = dict(both)
    s[(2, "R-dml")] = dict(dP=18.0, dC=1.0, gbps=65.0)
    expect("R-dml passes 20% apart", run(s)["reads"]["R-dml"]["state"], "INCOMPLETE")
    expect("labels GPU NOT SHOWN, NPU INSIDE", labels("NOT SHOWN", "INSIDE")[0][:40],
           "NPU vs DirectML: DirectML energy is 'pac")
    # the harness: sleeping readers through both launchers, real counters, short windows; no chip
    TMP.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S") + "_selftest"
    for kind in ("sleep", "sleep_iron"):
        irows = idle(f"{stamp}_{kind}", 4)
        rec = window(f"{stamp}_{kind}", kind, 6.0)
        win = inside(rec.get("rows", []), rec.get("wall0", 0), 6.0)
        busy = (rec.get("reader") or {}).get("busy_fraction", 0)
        print(f"  harness {kind}: idle rows {len(irows)}, window rows {len(win)}, busy {busy:.3f}, "
              f"failed {rec.get('failed')}")
        if len(irows) < 3 or len(win) < 2 or rec.get("failed") or not 0.9 < busy <= 1.0:
            fails.append(f"harness {kind}")
    print("SELFTEST", "FAIL " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("reader", "prereg", "suite", "verdict", "selftest"))
    ap.add_argument("log", nargs="?")
    ap.add_argument("--kind", choices=("c1", "g100", "g50", "g25", "nc", "sleep", "sleep_iron"))
    a = ap.parse_args()
    if a.mode == "reader":
        return reader(a.kind)
    if a.mode == "suite":
        return suite()
    if a.mode == "verdict":
        return verdict(Path(a.log))
    if a.mode == "selftest":
        return selftest()
    return prereg()


if __name__ == "__main__":
    sys.exit(main())
