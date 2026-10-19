#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""EmbeddingGemma 2 at 2048 tokens: the 780M alone, the NPU hybrid alone, and both at once, with package energy.
This is the data for the iGPU/NPU "logical device" decision: does the NPU add throughput beside the GPU, or save
energy per document?

Arms, each in its own window after its own idle (pass 2 runs them in reverse order, so drift shows):
  G    tools/embedgemma_gpu_loop.py: llama-server on ROCm0, the GPU baseline's command, one document per request
       (-fa off, as the baseline; --gpu-fa on runs the same sitting with flash attention)
  N    tools/embedgemma_hybrid.py --loop-go under scripts/research-lowlevel.sh --npu: the best bf16 hybrid
       (fused GeGLU, NPU local attention, native QKV and row ops), its 56-document gate first, then --batch 2 groups
  GN   both, started N first (the wrapper needs a clear host at start), each in its own loop over the same window
Power: typeperf at 1 Hz on the RAPL package counter (locked decision 10: package counters only; the 780M and the NPU
are both inside it, BENCHMARKS "J/GB of each chip's read stream"). An arm's idle is IDLE seconds with no worker,
right before it. Rows whose whole second lies in the window, TRIM seconds in from each end, are averaged; documents
are counted over the same trimmed interval, a span straddling an edge credited by its overlap fraction.

Readouts per arm and pass: docs/s per device, ms/doc (1 / rate), dPKG (W above the arm's idle), J/doc above idle
(dPKG / docs per s) and with idle (PKG / docs per s), CPU %, and busy per adapter and engine type.

Rules, fixed before the first sitting (the verdict reads them mechanically):
  T  throughput role: the docs/s of GN (both devices) >= 1.10 x G's, in both passes.
  E  energy role: N's J/doc above idle <= 0.90 x G's, in both passes; or GN's J/doc above idle <= 0.90 x G's
     in both passes (the mixed system cheaper per document than the GPU alone).
  Each of T and E reads NO when the opposite holds in both passes (GN < 1.10 x G; N and GN > 0.90 x G), and
  UNDECIDED otherwise. A pass with a failed arm, a failed gate or under 10 power rows is not read.
  The contention witness (scripts/research-lowlevel.sh) flags llama-server in GN by design; anything else foreign
  voids the arm.

    python tools/embedgemma_concurrency_energy.py --out results/aie/<name>.json [--passes 2] [--window 60]
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from power_probe import PKG  # noqa: E402

CPU = r"\Processor(_Total)\% Processor Time"
GPU = r"\GPU Engine(*)\Utilization Percentage"   # every engine, by adapter LUID and type (ROCm uses "compute 1")
BASE = Path.home() / "gemma4-embedding-baseline/logs"
DOCS = BASE / "docs-heldout-v1.json"
REF = BASE / "hf-fp32-cpu-20261018T174113Z.npz"
PY = Path.home() / "miniforge3/envs/mlir-aie-iron/python.exe"
TRIM, MIN_ROWS = 2.0, 10
LOG_DIR = "results/aie"   # the NPU arms' wrapper logs; --log-dir for a smoke run
GPU_FA = "off"            # llama-server -fa for the G arms (--gpu-fa); "off" is the GPU baseline's command


def npu_cmd(log, go, out):
    return ["bash", "scripts/research-lowlevel.sh", "--log", str(log), "--npu", "--wait-clear", "60", "--seconds",
            "2400", "--rss-gib", "16", "--", "bash", "scripts/research-iron.sh", "tools/embedgemma_hybrid.py",
            "--backend", "npu", "--batch", "2", "--round-dll", "scratch/egemma-native/embedgemma_round.dll",
            "--builds", "scratch/egemma-gemm-epi", "--npu-geglu", "6", "--attention", "npu-local",
            "--attn-build", "scratch/egemma-attn-strided", "--attn-pack-dll", "scratch/egemma-native/egattn_pack.dll",
            "--native-qkv", "scratch/egemma-native/eg_qkv_norm_rope.dll",
            "--native-rowops", "scratch/egemma-native/eg_rowops.dll", "--docs", str(DOCS), "--reference", str(REF),
            "--out", str(out), "--loop-go", str(go)]


def gpu_cmd(go, out):
    return [str(PY), "-I", "tools/embedgemma_gpu_loop.py", "--docs", str(DOCS), "--go", str(go), "--out", str(out),
            "--fa", GPU_FA]


def typeperf(samples, path):
    return subprocess.Popen(["typeperf", PKG, CPU, GPU, "-si", "1", "-sc", str(samples), "-y", "-o", str(path)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_csv(path):
    import csv
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    head = [h.lower() for h in rows[0]]
    ipkg = next(i for i, h in enumerate(head) if h.endswith(PKG.lower().lstrip("\\")))
    icpu = next(i for i, h in enumerate(head) if h.endswith(CPU.lower().lstrip("\\")))
    groups = {}   # "luid_..._engtype_X" -> columns; ROCm's queues need not be the 3D engine
    for i, h in enumerate(head):
        if "\\gpu engine(" in h and "_luid_" in h and "_engtype_" in h:
            luid = h.split("_luid_")[1].split("_phys")[0]
            groups.setdefault(f"{luid}_{h.split('_engtype_')[1].split(')')[0]}", []).append(i)
    out = []
    for r in rows[1:]:
        try:
            t = datetime.strptime(r[0], "%m/%d/%Y %H:%M:%S.%f").timestamp()
            eng = {g: round(sum(float(r[i]) for i in c if r[i].strip()), 2) for g, c in groups.items()}
            out.append([t, float(r[ipkg]) / 1e3, float(r[icpu]), eng])
        except (ValueError, IndexError):
            continue
    return out


def docs_in(spans, a, b):
    n = 0.0
    for t0, t1, k in spans:
        if t1 > t0:
            n += k * max(0.0, min(t1, b) - max(t0, a)) / (t1 - t0)
    return n


def arm(name, tmp, window, idle_s, stamp):
    """One arm: idle, start workers, wait READY, hand out the window, collect."""
    rec = dict(arm=name, idle_s=idle_s, window_s=window)
    ip = tmp / f"{name}_idle.csv"
    typeperf(idle_s + 1, ip).wait(timeout=idle_s + 60)
    rec["idle_rows"] = read_csv(ip)
    procs, outs, gos = {}, {}, {}
    if "N" in name:
        gos["npu"], outs["npu"] = tmp / f"{name}_npu.go", tmp / f"{name}_npu.json"
        rec["npu_log"] = f"{LOG_DIR}/embedgemma_concurrency_{name.lower()}_npu_desktop2_{stamp}.log"
        procs["npu"] = subprocess.Popen(npu_cmd(rec["npu_log"], gos["npu"], outs["npu"]), cwd=ROOT,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if "G" in name:
        gos["gpu"], outs["gpu"] = tmp / f"{name}_gpu.go", tmp / f"{name}_gpu.json"
        procs["gpu"] = subprocess.Popen(gpu_cmd(gos["gpu"], outs["gpu"]), cwd=ROOT,
                                        stdout=(tmp / f"{name}_gpu.stdout").open("w"), stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 1500
    while not all(Path(str(g) + ".ready").exists() for g in gos.values()):
        dead = [k for k, p in procs.items() if p.poll() is not None]
        if dead or time.monotonic() > deadline:
            for p in procs.values():
                if p.poll() is None:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            rec["failed"] = f"not ready: {dead or 'timeout'}"
            return rec
        time.sleep(0.5)
    wp = tmp / f"{name}_window.csv"
    tp = typeperf(int(window + 10), wp)
    time.sleep(2.5)
    start = time.time() + 2.0
    stop = start + window
    for g in gos.values():
        g.write_text(f"{start} {stop}\n", encoding="utf-8")
    rec.update(start=start, stop=stop)
    for k, p in procs.items():
        try:
            p.wait(timeout=window + 600)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            p.wait(timeout=30)
        rec[f"{k}_exit"] = p.returncode
    tp.wait(timeout=120)
    rec["window_rows"] = read_csv(wp)
    for k, o in outs.items():
        rec[k] = json.loads(Path(str(o) + (".loop.json" if k == "npu" else "")).read_text(encoding="utf-8")) \
            if Path(str(o) + (".loop.json" if k == "npu" else "")).exists() else None
    if "npu" in outs and outs["npu"].exists():
        h = json.loads(outs["npu"].read_text(encoding="utf-8"))
        rec["npu_gate"] = dict(passed=h["passed"], cos_min=h["cos_min"], nn_kept=h["nn_kept"])
    return rec


def stats(rec):
    if rec.get("failed") or any(rec.get(k) is None for k in ("npu", "gpu") if k.upper()[0] in rec["arm"]):
        return dict(ok=False, why=rec.get("failed") or "no loop record")
    if "N" in rec["arm"] and not rec.get("npu_gate", {}).get("passed"):
        return dict(ok=False, why="npu gate")
    a, b = rec["start"] + TRIM, rec["stop"] - TRIM
    win = [r for r in rec["window_rows"] if a + 1.0 <= r[0] <= b]
    idle = rec["idle_rows"][1:]
    if len(win) < MIN_ROWS or len(idle) < MIN_ROWS:
        return dict(ok=False, why=f"rows window {len(win)} idle {len(idle)}")
    pi, pw = statistics.fmean(r[1] for r in idle), statistics.fmean(r[1] for r in win)
    out = dict(ok=True, idle_w=pi, idle_sd_w=statistics.pstdev(r[1] for r in idle), pkg_w=pw, dP_w=pw - pi,
               cpu=statistics.fmean(r[2] for r in win), rows=len(win))
    keys = {k for r in win for k in r[3]}
    busy = {k: statistics.fmean(r[3].get(k, 0.0) for r in win) for k in keys}
    out["engine_busy"] = {k: round(v, 1) for k, v in sorted(busy.items(), key=lambda kv: -kv[1]) if v >= 1.0}
    total = 0.0
    for k in ("gpu", "npu"):
        if rec.get(k):
            r = docs_in(rec[k]["spans"], a, b) / (b - a)
            out[f"{k}_docs_s"], out[f"{k}_ms_doc"] = r, 1000 / r if r else None
            total += r
    out["docs_s"] = total
    out["j_doc_above_idle"] = out["dP_w"] / total
    out["j_doc_with_idle"] = pw / total
    return out


def verdict(passes):
    def read(cond_yes, cond_no):
        ys = [cond_yes(p) for p in passes]
        ns = [cond_no(p) for p in passes]
        return "YES" if all(ys) else "NO" if all(ns) else "UNDECIDED"
    ok = [p for p in passes if all(p[k]["ok"] for k in ("G", "N", "GN"))]
    if len(ok) < len(passes) or not ok:
        return dict(T="UNDECIDED", E="UNDECIDED", why="a pass has a failed arm")
    T = read(lambda p: p["GN"]["docs_s"] >= 1.10 * p["G"]["docs_s"], lambda p: p["GN"]["docs_s"] < 1.10 * p["G"]["docs_s"])
    jg = lambda p: p["G"]["j_doc_above_idle"]  # noqa: E731
    E = read(lambda p: p["N"]["j_doc_above_idle"] <= 0.90 * jg(p), lambda p: p["N"]["j_doc_above_idle"] > 0.90 * jg(p))
    E2 = read(lambda p: p["GN"]["j_doc_above_idle"] <= 0.90 * jg(p), lambda p: p["GN"]["j_doc_above_idle"] > 0.90 * jg(p))
    e = "YES" if "YES" in (E, E2) else "NO" if E == E2 == "NO" else "UNDECIDED"
    return dict(T=T, E=e, E_npu_alone=E, E_mixed=E2)


def main():
    global LOG_DIR, GPU_FA  # noqa: PLW0603
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--idle", type=int, default=60)
    ap.add_argument("--log-dir", default="results/aie")
    ap.add_argument("--gpu-fa", choices=["on", "off"], default="off", help="llama-server -fa in the G arms")
    a = ap.parse_args()
    LOG_DIR, GPU_FA = a.log_dir, a.gpu_fa
    if a.out.exists():
        raise FileExistsError(a.out)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tmp = ROOT / "scratch" / "egemma-energy" / stamp
    tmp.mkdir(parents=True)
    recs, passes = [], []
    for p in range(a.passes):
        order = ["G", "N", "GN"] if p % 2 == 0 else ["GN", "N", "G"]
        ps = {}
        for name in order:
            ptmp = tmp / f"p{p + 1}"
            ptmp.mkdir(exist_ok=True)
            r = arm(name, ptmp, a.window, a.idle, f"{stamp}_p{p + 1}")
            r["pass"] = p + 1
            s = stats(r)
            r["stats"] = ps[name] = s
            recs.append(r)
            print(f"pass {p + 1} {name}: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                                       for k, v in s.items()}), flush=True)
        passes.append(ps)
    v = verdict(passes)
    print("VERDICT " + json.dumps(v), flush=True)
    a.out.write_text(json.dumps(dict(stamp=stamp, gpu_fa=GPU_FA, window_s=a.window, idle_s=a.idle, trim_s=TRIM, verdict=v,
                                     passes=passes, arms=recs, npu_command=npu_cmd("LOG", "GO", "OUT"),
                                     gpu_command=gpu_cmd("GO", "OUT")), indent=1) + "\n", encoding="utf-8")
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
