#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The LLM study's stage (e): does an NPU decode free the GPU, or the CPU, where the CPU and
DirectML decodes of Gemma 3 4B do not? (locked decision 10; plan v4, approved 2026-09-24)

    bash scripts/research-iron.sh tools/llm_freeing.py build   # the NPU proxy's xclbin (compile only)
    python tools/llm_freeing.py dryrun                         # the one disclosed proxy dry run (pre-prereg)
    python tools/llm_freeing.py dryrun-verdict <log>           # the dry run's go/no-go, re-read
    python tools/llm_freeing.py prereg                         # the pre-registration and its pins
    python tools/llm_freeing.py loadcheck                      # every child to READY and a 2 s go (pre-sitting)
    python tools/llm_freeing.py suite                          # the sitting (resnet_env17)
    python tools/llm_freeing.py verdict <suite log>            # the mechanical verdict
    python tools/llm_freeing.py selftest                       # synthetic rules; no model, no chip
    python tools/llm_freeing.py reader --arm {CPU,DML,CPU-c} --threads N [--unpaced]   # spawned by suite
    python tools/llm_freeing.py wload --kind {W1,W2,Wcpu}                              # spawned by suite

The NPU arm is a PROXY: no NPU int4 GEMV exists. It is stage 1's shim-read design
(tools/npu_read_bw_probe.py) at 4 columns x 2 MM2S channels, reading one H4 token's bytes in
238 dependent dispatches (34 layers x 7 GEMVs), each from its own host buffer, paced on an
absolute 190 ms schedule by the C++ host (kernels/dispatch_floor/dispatch_runner.cpp --paced).
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "llm_freeing"
TMP = ROOT / "scratch" / "llm" / "freeing"
OUT = ROOT / "results" / "llm"
for _p in (ROOT / "tools", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

GB_TOKEN_H4 = 2_182_348_800        # (c)'s H4 bytes per token (DERIVED from the release's shapes)
DISPATCHES = 238                   # 34 layers x 7 GEMVs
LAYOUT = (4, 2)                    # stage 1's best layout: 47.62 GB/s (MEASURED)
PERIOD_MS = 190.0                  # 5.26 tok/s: the pace (plan v4, section 0.3)
FLOOR_TOKPS = 5.0                  # the user's floor; "holds" is achieved >= 5.00 over the window


def proxy_words_per_channel() -> int:
    """Words per MM2S channel so that DISPATCHES dispatches read at least one H4 token, rounded up
    to 1024 words as stage 1 rounds (npu_read_bw_probe.words_per_channel)."""
    ch = LAYOUT[0] * LAYOUT[1]
    return -(-GB_TOKEN_H4 // DISPATCHES // ch // 4 // 1024) * 1024


PROXY_WPC = proxy_words_per_channel()                  # 286,720
PROXY_WORDS = PROXY_WPC * LAYOUT[0] * LAYOUT[1]        # 2,293,760 int32 per dispatch (9,175,040 B)
PROXY_NAME = f"proxy_c{LAYOUT[0]}x{LAYOUT[1]}_w{PROXY_WPC}"
PROXY_DIR = BUILD / PROXY_NAME

# ---------------------------------------------------------------- the protocol (plan v4, U1-U6 as recommended)

CAL_THREADS = (1, 2, 4, 8)         # U6: the CPU arm's N is the smallest of these that clears CAL_MIN_TOKPS
CAL_MIN_TOKPS = 6.58               # 1.25 x 5.26 tok/s, UNPACED, in both calibration runs
CAL_RUNS = 2
DML_THREADS = 1                    # U6: the DirectML arm's host, 1 intra-op thread, spinning off, unpinned
EVEN = [0, 2, 4, 6, 8, 10, 12, 14]   # Wcpu (and (c)'s CPU setup): the caller on 0, the pool on the rest
ODD = [1, 3, 5, 7, 9, 11, 13, 15]    # the CPU arm: the caller on 1, its N - 1 pool threads on 3, 5, ...
ARM_MODEL = {"CPU": "C0-H4", "DML": "D-H4", "CPU-c": "C0-H4"}
ARMS = ("NPU", "CPU", "DML")
WS = ("W1", "W2", "Wcpu")
MATRIX = [(a, None) for a in ARMS] + [(None, w) for w in WS] + [(a, w) for w in WS for a in ARMS]
ORDER = [(1, a, w) for a, w in MATRIX] + [(2, a, w) for a, w in MATRIX[::-1]]
WORST = [(None, "Wcpu"), ("CPU-c", None), ("CPU-c", "Wcpu")]      # report-only: (c)'s same-core setup
WORST_ORDER = [(1, a, w) for a, w in WORST] + [(2, a, w) for a, w in WORST[::-1]]
GAP_S = 5.0                        # between windows
TP_LEAD_S = 2.5                    # the counters start after every READY, this long before the start
GO_LEAD_S = 0.5
SETTLE_S = 10.0
WINDOW_S = 60.0
TAIL_S = 3.0
RUNNER_S = 80.0                    # the proxy's --seconds from its READY: 2.5 + 0.5 + 10 + 60 + 3 = 76 s, + 4
READY_TIMEOUT_S = 600
RUNNER_READY_S = 120
TP_SAMPLES = 600                   # typeperf's cap; it is stopped when the window's processes exit
MIN_ROWS = 50                      # counter rows, and fault-sampler seconds, inside a 60 s window
WARMUP = {"W1": 3, "W2": 50, "Wcpu": 10}   # W2's is 4b_g2g's default

# ---------------------------------------------------------------- the rules (plan v4, section 5)

K_MIN = 0.90
K_AGREE = 0.05                     # a K whose passes differ by more than this (absolute) is INCOMPLETE
W_AGREE = 0.10                     # a W's alone rate must agree within 10% between the passes
TIE_NUM, TIE_DEN = 11, 10          # the study's 1.10 line; exactly 1.10x counts
TIE_EPS = 1e-12
HF_MAX = 25.0                      # U5: a measured process's own hard faults per second over the window
PAGES_MAX = 1000.0                 # U5: the system's Pages Input/sec over the window (chosen after (c)'s 625)
MEM_BIG_GB = 4.0                   # a process this large at the start refuses the sitting (as (c))
READING_WPM = 238                  # U2's report-only column: 238 words per minute
READING_TOKPS = (5.98, 6.21)       # at 1.51-1.57 Gemma tokens per word on (c)'s pinned text (the prereg derives it)

# predictions (plan v4, section 7), scored by the verdict; the joint score is OPEN and has none
PRED_K = {("W1", "NPU"): (0.90, 0.96), ("W1", "CPU"): (0.87, 0.94), ("W1", "DML"): (0.60, 0.80),
          ("W2", "NPU"): (0.95, None), ("W2", "CPU"): (0.90, None), ("W2", "DML"): (None, 0.90),
          ("Wcpu", "DML"): (0.90, None), ("Wcpu", "NPU"): (0.90, None), ("Wcpu", "CPU"): (0.80, 0.95)}
PRED_CAL_N = (2, 4)

# ---------------------------------------------------------------- the pins

READBW_MODEL = "scratch/llm/readbw_fp32_65536x4096/model.onnx"       # stage 2's W1 model (1 GiB fp32)
READBW_BYTES = 65536 * 4096 * 4
YOLO_MODEL = "models/yolov8s-worldv2_cut.onnx"
YOLO_TEXT_NPZ = "models/yolow_text.npz"
YOLO_TEXT_ENCODER = "models/yolow_text_encoder.onnx"
BUS_JPG = "assets/bus.jpg"
FILE_PIN = {
    YOLO_MODEL: "7a9c0c0076f52506f843d171c3102d631f85a3828f0fa974cdbe5b46b4bca777",
    YOLO_TEXT_NPZ: "9738bb5f998f1828bc3a504a087a07d7e8011b73fd306e98d1894d571767a782",
    YOLO_TEXT_ENCODER: "d2481c13bd50f7333e21eb3d5eef70804bae671b27377bbbb8f0e093bd3ff709",
    BUS_JPG: "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63",
    READBW_MODEL: "1871f393faa54fde25815f47858b1c14455cc844dff5a0251a33f4ce4a4457a1",
    "scratch/llm/readbw_fp32_65536x4096/weights.bin": "b825ea3b48eead2a7ff8185ff5ac1e8f34432ebc75011ec54f338bbdb34b03e4",
    # the proxy and its host, as llm_freeing_build_desktop2_20260924.log printed them
    f"build/llm_freeing/{PROXY_NAME}/insts.bin": "234670b4ae0d4e0017d5ed9bc47ed83fd3aab65f40c0241df84b02c7af8b0e6a",
    f"build/llm_freeing/{PROXY_NAME}/probe.xclbin": "bb835441eeaf8f0d90aea816845a416a9fc3fd08da6e412b4d5ddd5c3d0460df",
    f"build/llm_freeing/{PROXY_NAME}/probe.mlir": "853486f911ec02b1672d4e742dc31c70b11feb59bf106c399cf544b809e058ab",
    "kernels/dispatch_floor/dispatch_runner.cpp": "27be3db3328ce7ec3c4188bf37c70aa908b024f796f2b49070e1909f2ee6fdc9",
    "kernels/dispatch_floor/dispatch_runner.exe": "f904c4709aecf40a954ab051ae874a3e38e08c73104d90d942b97e018951aec8",
}
DECODE_MODELS = ("C0-H4", "D-H4")  # pinned by gemma_decode_suite.MODEL_SHA ((c)'s build log)
LOG_PIN = {                        # {name: (sha256 of the working copy, of the LF blob)}
    "llm_freeing_build_desktop2_20260924.log": (
        "8b41056e036e4c3c0e44eae0c7ceef09ef233ef985487be82b6644703fb06eaf",
        "6f904a0399d195a18dadc3541050344302f9dbcc92370e47b44110404ae56f83"),
    "llm_freeing_dryrun_desktop2_20260924.log": (
        "845efb16b36f7ce945922ae44fe3cf1d5c28ba9bf075b95ded549f144bdd79fd",
        "845efb16b36f7ce945922ae44fe3cf1d5c28ba9bf075b95ded549f144bdd79fd"),
}
NEUTRAL_FILE = ROOT / "scratch" / "neutral_names.txt"   # local and git-ignored; its names are never logged


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sha256_lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def say(tag: str, obj) -> None:
    print(f"{tag} " + json.dumps(obj), flush=True)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wait_until(t: float) -> None:
    d = t - time.time()
    if d > 0:
        time.sleep(d)


# ---------------------------------------------------------------- hygiene: names and paths at the source

_NEUTRAL = {"loaded": False, "re": None, "count": 0}
_ROOT_RE = re.compile(r"(?i)" + r"[\\/]+".join(re.escape(p.rstrip("\\/")) for p in ROOT.parts))


def neutral_list(path: Path = None) -> list:
    path = path or NEUTRAL_FILE
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [s.strip() for s in lines if s.strip() and not s.strip().startswith("#")]


def set_neutral(names) -> None:
    names = [n for n in names if n]
    _NEUTRAL.update(loaded=True, count=len(names), re=re.compile(
        r"(?i)(?<![A-Za-z0-9])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9])") if names else None)


def neutralize(s: str) -> str:
    """The worktree root becomes <repo>, and each name on the local list becomes <tool> (so an image
    name "x.exe" reads "<tool>.exe")."""
    if not _NEUTRAL["loaded"]:
        set_neutral(neutral_list())
    s = _ROOT_RE.sub("<repo>", s)
    return _NEUTRAL["re"].sub("<tool>", s) if _NEUTRAL["re"] else s


def neutral_counters() -> dict:
    import gemma_decode_suite as gds
    return {pid: (neutralize(v[0]), *v[1:]) for pid, v in gds.proc_counters().items()}


def host_load() -> str:
    out = subprocess.run(["powershell", "-NoProfile", "-File", str(ROOT / "tools" / "host_load.ps1")],
                         capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = "\n".join(neutralize(s) for s in out.splitlines())
    print(out, flush=True)
    return out


def host_gate() -> bool:
    """(c)'s gate (llm_prefill_bench.host_gate): under 2 busy cores and no peer at >= 0.5 cores."""
    out = host_load()
    busy = re.search(r"HOST_LOAD busy_cores=([\d.]+)", out)
    peers = [float(v) for v in re.findall(r"HOST_LOAD_PEER ([\d.]+)", out)]
    ok = busy is not None and float(busy[1]) < 2 and max(peers, default=0.0) < 0.5
    print("HOST_GATE", "CLEAR" if ok else "REFUSE", flush=True)
    return ok


# ---------------------------------------------------------------- the proxy: build, runner, dry run

def build():
    """Compile only: stage 1's design at the proxy's size, into build/llm_freeing/."""
    import npu_read_bw_probe as rb
    from aie.utils.compile.utils import compile_mlir_module
    d = PROXY_DIR
    if not (d / "insts.bin").exists():
        (d / "design.prj").mkdir(parents=True, exist_ok=True)
        module = rb.design(LAYOUT, PROXY_WPC)
        (d / "probe.mlir").write_text(module, encoding="utf-8")
        print("BUILD", PROXY_NAME, flush=True)
        compile_mlir_module(module, insts_path=d / "insts.bin", xclbin_path=d / "probe.xclbin",
                            work_dir=d / "design.prj")
    else:
        print("EXISTS", PROXY_NAME, flush=True)
    print("PROXY_ARTIFACT_JSON", json.dumps({
        "name": PROXY_NAME, "layout": list(LAYOUT), "words_per_channel": PROXY_WPC,
        "bytes_per_dispatch": PROXY_WORDS * 4, "dispatches_per_token": DISPATCHES,
        "bytes_per_token": PROXY_WORDS * 4 * DISPATCHES, "h4_bytes_per_token": GB_TOKEN_H4,
        "insts_sha256": sha256(d / "insts.bin"), "xclbin_sha256": sha256(d / "probe.xclbin"),
        "mlir_sha256": sha256(d / "probe.mlir")}), flush=True)
    runner()


RUNNER_CPP = ROOT / "kernels" / "dispatch_floor" / "dispatch_runner.cpp"
RUNNER_EXE = ROOT / "kernels" / "dispatch_floor" / "dispatch_runner.exe"


def runner() -> dict:
    """The C++ host's source and binary (scripts/build_dispatch_runner.bat), by SHA-256."""
    rec = {"cpp": RUNNER_CPP.relative_to(ROOT).as_posix(), "cpp_sha256": sha256(RUNNER_CPP),
           "exe": RUNNER_EXE.relative_to(ROOT).as_posix(),
           "exe_sha256": sha256(RUNNER_EXE) if RUNNER_EXE.exists() else None}
    print("RUNNER_JSON", json.dumps(rec), flush=True)
    return rec


def proxy_cmd(seconds: float, warmup_tokens: int = 3) -> list:
    """The paced proxy, with paths relative to the repository root (the runner prints them)."""
    rel = lambda p: p.relative_to(ROOT).as_posix()  # noqa: E731
    return [rel(RUNNER_EXE), "--xclbin", rel(PROXY_DIR / "probe.xclbin"), "--insts", rel(PROXY_DIR / "insts.bin"),
            "--paced", "--per-token", str(DISPATCHES), "--words", str(PROXY_WORDS),
            "--period-ms", f"{PERIOD_MS:g}", "--seconds", f"{seconds:g}", "--warmup-tokens", str(warmup_tokens)]


def dryrun(seconds: float = 60.0) -> int:
    """The one disclosed dry run (plan v4), before the prereg: go/no-go only. It answers whether the
    runner's pacing mode holds the 190 ms period at 238 x 9.17 MB, a size stage 1 never timed."""
    print("DRYRUN: the (e) proxy, pre-prereg, go/no-go only (plan v4 as approved 2026-09-24)", flush=True)
    host_load()
    runner()
    cmd = proxy_cmd(seconds)
    print("PROXY_CMD", " ".join(cmd), flush=True)
    raw = []
    # executed by absolute path (Windows resolves a relative program against the parent, not cwd);
    # the log shows the relative form above, and the runner prints only the relative xclbin/insts
    p = subprocess.Popen([str(RUNNER_EXE)] + cmd[1:], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    try:
        for s in p.stdout:
            s = s.rstrip("\n")
            raw.append(s)
            print(s, flush=True)
        rc = p.wait(timeout=seconds + 600)
    finally:
        if p.poll() is None:
            p.kill()
    print("RUNNER_EXIT", rc, flush=True)
    if rc != 0:
        print("DRYRUN NO-GO: the runner exited non-zero")
        return 2
    return dryrun_lines(raw)


def paced_tokens(lines) -> list:
    return [json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith("TOKEN_JSON ")]


def dryrun_verdict(log: Path) -> int:
    return dryrun_lines(log.read_text(encoding="utf-8").splitlines())


def dryrun_lines(lines) -> int:
    """The dry run's go/no-go (plan v4): GO if every token's work fits the 190 ms period and the
    achieved rate over the run holds the floor; otherwise NO-GO, and the stage stops for the user.
    (The sitting counts tokens by token_stats(), not by this.)"""
    toks = paced_tokens(lines)
    done = next((json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith("PACED_DONE_JSON ")), None)
    if not toks or done is None:
        print("DRYRUN NO-GO: the runner did not finish (no tokens or no PACED_DONE_JSON)")
        return 2
    work = [t["work_us"] / 1e3 for t in toks]
    span_s = (toks[-1]["t1_ns"] - toks[0]["t0_ns"]) / 1e9
    tokps = len(toks) / (span_s + PERIOD_MS / 1e3 - work[-1] / 1e3) if span_s > 0 else 0.0
    rec = {"tokens": len(toks), "work_ms_median": statistics.median(work), "work_ms_max": max(work),
           "work_ms_min": min(work), "over_period": sum(w > PERIOD_MS for w in work),
           "achieved_tokps": tokps, "late_starts": done["late_starts"]}
    print("DRYRUN_JSON", json.dumps(rec))
    go = rec["over_period"] == 0 and tokps >= FLOOR_TOKPS
    print(f"work per token: median {rec['work_ms_median']:.1f} ms, max {rec['work_ms_max']:.1f} ms against the "
          f"{PERIOD_MS:g} ms period; achieved {tokps:.3f} tok/s against the {FLOOR_TOKPS:g} floor")
    print("DRYRUN", "GO" if go else "NO-GO")
    return 0 if go else 2


# ---------------------------------------------------------------- the GenAI decode arms (a reader process each)

def pool_cpus(arm: str, threads: int) -> list:
    return {"CPU": ODD[1:threads], "CPU-c": EVEN[1:], "DML": []}[arm]


def caller_cpu(arm: str):
    return {"CPU": ODD[0], "CPU-c": EVEN[0], "DML": None}[arm]


def session_options(arm: str, threads: int) -> dict:
    """The GenAI overlay's decoder session_options. CPU-c is (c)'s setup exactly
    (gemma_decode_suite.og_model(arm, pinned=True)); the rivals (U6) run with intra-op spinning off,
    the CPU arm's pool on the odd CPUs (1-based ids for ORT), none for N = 1."""
    if arm == "CPU-c":
        return {"intra_op_num_threads": len(EVEN), "inter_op_num_threads": 1,
                "config_entries": {"session.intra_op_thread_affinities": ";".join(str(c + 1) for c in EVEN[1:])}}
    ce = {"session.intra_op.allow_spinning": "0"}
    if pool_cpus(arm, threads):
        ce["session.intra_op_thread_affinities"] = ";".join(str(c + 1) for c in pool_cpus(arm, threads))
    return {"intra_op_num_threads": threads, "inter_op_num_threads": 1, "config_entries": ce}


def og_model(arm: str, threads: int):
    import onnxruntime_genai as og
    import gemma_decode as gd
    cfg = og.Config(str(gd.ONNX / ARM_MODEL[arm]))
    cfg.overlay(json.dumps({"model": {"decoder": {"session_options": session_options(arm, threads)}}}))
    return og.Model(cfg)


def reader(arm: str, threads: int, paced: bool) -> int:
    """One arm in one window: load, an 8-token check, READY; on "go <start> <stop>" a greedy decode
    from (c)'s pinned 128-token prompt, a fresh generator every 896 generated tokens, each token on
    the absolute schedule start + k x 190 ms (paced) or back to back (unpaced), until stop.
    READER_JSON carries every token's wall-clock start and completion."""
    import gc
    import numpy as np
    import psutil
    import onnxruntime_genai as og
    import gemma_decode as gd
    import gemma_decode_suite as gds
    import measure_noise as mn
    ids = gds.pinned_ids()
    prompt = np.asarray(ids[:gds.PROMPT_TOKENS], dtype=np.int32)
    pool = pool_cpus(arm, threads)
    if caller_cpu(arm) is not None:
        mn.pin_calling_thread(caller_cpu(arm))
    before = mn.thread_ids()
    t0 = time.time()
    model = og_model(arm, threads)
    load_s = time.time() - t0
    check, _, _, _ = gds.decode(model, list(ids[:gds.PROMPT_TOKENS]), 8)
    placement = mn.thread_placement(mn.thread_ids() - before)
    single = sorted(p["cpus"][0] for p in placement if len(p["cpus"]) == 1)
    say("READY", {"pid": os.getpid(), "arm": arm, "model": ARM_MODEL[arm], "threads": threads, "paced": paced,
                  "session_options": session_options(arm, threads), "caller_cpu": caller_cpu(arm), "pool_cpus": pool,
                  "single_cpu_threads": single, "pin_ok": all(c in single for c in pool), "load_s": round(load_s, 2),
                  "check_tokens": check, "available_gb": round(psutil.virtual_memory().available / 1e9, 2),
                  "onnxruntime_dll": gd.loaded_dll("onnxruntime.dll"),
                  "directml_dll": gd.loaded_dll("DirectML.dll")})
    go = sys.stdin.readline().split()
    if len(go) != 3 or go[0] != "go":
        return 2
    start, stop = float(go[1]), float(go[2])
    period = PERIOD_MS / 1e3
    gen = params = None
    produced = generators = late = k = 0
    t0s, t1s, first = [], [], []
    wait_until(start)
    while True:
        slot = start + k * period if paced else time.time()
        if slot >= stop:
            break
        if paced:
            now = time.time()
            if now < slot:
                time.sleep(slot - now)
            elif k:
                late += 1
        a = time.time()
        if gen is None or produced >= gds.GEN_TOKENS:     # a restart: its prompt pass is part of this token
            gen = params = None
            params = og.GeneratorParams(model)
            params.set_search_options(do_sample=False, max_length=gds.POSITIONS, min_length=gds.POSITIONS)
            gen = og.Generator(model, params)
            gen.append_tokens(prompt)
            produced = 0
            generators += 1
        gen.generate_next_token()
        tok = int(gen.get_next_tokens()[0])
        b = time.time()
        produced += 1
        if generators == 1 and len(first) < 16:
            first.append(tok)
        t0s.append(round(a, 4))
        t1s.append(round(b, 4))
        k += 1
    stopped = time.time()
    gen = params = None
    del model
    gc.collect()
    say("READER_JSON", {"arm": arm, "pid": os.getpid(), "tokens": k, "generators": generators, "late_starts": late,
                        "first_tokens": first, "stopped": round(stopped, 3), "t0": t0s, "t1": t1s})
    return 0


# ---------------------------------------------------------------- the workloads (a process each)

UNIT_SIZE = {"W1": READBW_BYTES / 1e9, "W2": 1.0, "Wcpu": 1.0}
UNIT_NAME = {"W1": "GB/s", "W2": "frames/s", "Wcpu": "inferences/s"}


def wload(kind: str) -> int:
    """W1: stage 2's DirectML read loop (a 1 GiB fp32 ReduceSum). W2: 4b_g2g's DirectML frame loop on
    bus.jpg with COCO's 80 names (letterbox, network, decode and NMS). Wcpu: the same model on an ORT
    CPU session, 8 intra-op threads (the caller on logical CPU 0, the pool on 2, ..., 14), looping one
    letterboxed bus.jpg. READY, then on "go <start> <stop>" back-to-back units until stop; W_JSON
    carries every unit's wall-clock start and end."""
    import gc
    import numpy as np
    import psutil
    ident = {}
    if kind == "W1":
        import llm_gemv_bench as bench
        sess = bench.make_session(ROOT / READBW_MODEL, "dml", 8, opt="disable_all")
        feed = {"x": np.zeros(1, np.float32)}
        ident = {"providers": sess.get_providers()}

        def one():
            sess.run(None, feed)
    else:
        import cv2
        import npu.yolow as yw
        img = cv2.imread(str(ROOT / BUS_JPG))
        model = ROOT / YOLO_MODEL
        if kind == "W2":
            from ignite_xdna.pipelines.yolow_pipeline import YoloWorldDecoder
            from ignite_xdna.pipelines.yolow_text import YoloWorldText
            from npu.session import build_session
            names = list(yw.COCO_CLASSES)
            text = YoloWorldText(ROOT / YOLO_TEXT_ENCODER, ROOT / YOLO_TEXT_NPZ)
            emb, _ = text.vocabulary(names)
            scales, biases = text.contrastive_scales, text.contrastive_biases
            del text                                     # the text encoder's CPU session goes before READY
            gc.collect()
            sess = build_session(model, "dml", "unused", None, log_severity=3)
            imgsz = yw.input_size(sess.get_inputs()[0].shape, str(model))
            order = yw.head_order(sess, imgsz)
            inp = sess.get_inputs()[0].name
            dec = YoloWorldDecoder(emb, names, scales, biases, imgsz=imgsz, conf_thres=0.25, iou_thres=0.7)
            last = {}

            def one():
                x, pad, scale = yw.letterbox(img, imgsz)
                raw = sess.run(None, {inp: x})
                last["detections"] = len(dec.postprocess(dec.decode([raw[i] for i in order], None, 0.25), pad, scale))
            ident = {"providers": sess.get_providers(), "classes": len(names)}
        else:
            import onnxruntime as ort
            import measure_noise as mn
            mn.pin_calling_thread(EVEN[0])
            before = mn.thread_ids()
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.intra_op_num_threads = len(EVEN)
            so.inter_op_num_threads = 1
            so.add_session_config_entry("session.intra_op_thread_affinities", ";".join(str(c + 1) for c in EVEN[1:]))
            sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
            imgsz = yw.input_size(sess.get_inputs()[0].shape, str(model))
            x, _, _ = yw.letterbox(img, imgsz)
            inp = sess.get_inputs()[0].name

            def one():
                sess.run(None, {inp: x})
    for _ in range(WARMUP[kind]):
        one()
    if kind == "Wcpu":
        single = sorted(p["cpus"][0] for p in mn.thread_placement(mn.thread_ids() - before) if len(p["cpus"]) == 1)
        ident = {"providers": sess.get_providers(), "threads": len(EVEN), "caller_cpu": EVEN[0], "pool_cpus": EVEN[1:],
                 "single_cpu_threads": single, "pin_ok": all(c in single for c in EVEN[1:])}
    if kind == "W2":
        ident["detections_last_warmup"] = last.get("detections")
    say("READY", {"kind": kind, "pid": os.getpid(), "warmup": WARMUP[kind], **ident,
                  "available_gb": round(psutil.virtual_memory().available / 1e9, 2)})
    go = sys.stdin.readline().split()
    if len(go) != 3 or go[0] != "go":
        return 2
    start, stop = float(go[1]), float(go[2])
    t0s, t1s = [], []
    wait_until(start)
    while True:
        a = time.time()
        if a >= stop:
            break
        one()
        t0s.append(round(a, 4))
        t1s.append(round(time.time(), 4))
    say("W_JSON", {"kind": kind, "pid": os.getpid(), "units": len(t0s), "t0": t0s, "t1": t1s})
    return 0


# ---------------------------------------------------------------- the numbers (pure; the selftest covers them)

def token_stats(t0s, t1s, a: float, b: float) -> dict:
    """An arm's tok/s: the tokens whose completion lies in (a, b], over b - a."""
    inwin = [(x, y) for x, y in zip(t0s, t1s) if a < y <= b]
    work = sorted((y - x) * 1e3 for x, y in inwin)
    return {"tokens": len(inwin), "rate": len(inwin) / (b - a),
            "work_ms_median": round(statistics.median(work), 2) if work else None,
            "work_ms_p95": round(work[min(len(work) - 1, int(0.95 * len(work)))], 2) if work else None,
            "work_ms_max": round(work[-1], 2) if work else None}


def unit_stats(t0s, t1s, a: float, b: float, size: float = 1.0) -> dict:
    """A W's rate: each unit's overlap with [a, b] as a fraction of the unit, summed, times the unit's
    size, over b - a; also per second of the window."""
    got, per = 0.0, [0.0] * int(round(b - a))
    for x, y in zip(t0s, t1s):
        if y <= x:
            continue
        lo, hi = max(x, a), min(y, b)
        if hi <= lo:
            continue
        got += (hi - lo) / (y - x)
        for s in range(max(0, int(lo - a)), min(len(per), int(hi - a) + 1)):
            l2, h2 = max(lo, a + s), min(hi, a + s + 1)
            if h2 > l2:
                per[s] += (h2 - l2) / (y - x)
    ms = sorted((y - x) * 1e3 for x, y in zip(t0s, t1s) if a < y <= b)
    return {"rate": got * size / (b - a), "units": round(got, 3), "per_second": [round(v * size, 3) for v in per],
            "unit_ms_median": round(statistics.median(ms), 3) if ms else None,
            "covered": bool(t0s) and t0s[0] <= a and t1s[-1] >= b}


def faults_window(rows, a: float, b: float, pids: dict, top: int = 5) -> dict:
    """From FaultSampler rows [(t, {pid: (name, faults, hard faults, read bytes)})]: the seconds whose
    whole second lies in [a, b] (inside()'s rule), each measured process's own hard faults/s (pids:
    {role: pid}), and the top processes by hard faults/s, [name, pid, hard/s, faults/s, read B/s]."""
    agg, n = {}, 0
    for t, d in rows:
        if a + 1.0 <= t <= b:
            n += 1
            for pid, (name, pf, hf, rb) in d.items():
                x = agg.setdefault(pid, [name, 0, 0, 0])
                x[1], x[2], x[3] = x[1] + pf, x[2] + hf, x[3] + rb
    s = max(n, 1)
    ent = [[nm, pid, round(hf / s, 2), round(pf / s, 1), round(rb / s)] for pid, (nm, pf, hf, rb) in agg.items()]
    return {"seconds": n, "hard_per_s": {r: round(agg[p][2] / s, 2) if p in agg else 0.0 for r, p in pids.items()},
            "top_hard": sorted(ent, key=lambda e: (-e[2], -e[3]))[:top]}


def window_state(rec: dict) -> tuple:
    """OK, VOID (the witnesses) or FAILED (a process); a window that is not OK feeds nothing."""
    if rec.get("failed"):
        return "FAILED", rec["failed"]
    why = []
    for key in ("arm_stats", "w_stats"):
        st = rec.get(key)
        if st is None:
            continue
        if st.get("missing") or st.get("exit") != 0:
            return "FAILED", f"{key}: {st.get('missing') or 'exit ' + str(st.get('exit'))}"
        if not st.get("covered"):
            why.append(f"{key}: the process did not run through the window")
        if st.get("pin_ok") is False:
            why.append(f"{key}: the pinning was not read back")
    c, f = rec.get("counters", {}), rec.get("faults", {})
    if c.get("rows", 0) < MIN_ROWS:
        why.append(f"counter rows {c.get('rows', 0)} < {MIN_ROWS}")
    if f.get("seconds", 0) < MIN_ROWS:
        why.append(f"fault-sampler seconds {f.get('seconds', 0)} < {MIN_ROWS}")
    for role, hf in f.get("hard_per_s", {}).items():
        if hf > HF_MAX:
            why.append(f"the {role} process's hard faults {hf}/s > {HF_MAX:g}")
    if c.get("pages_in_mean", 0.0) > PAGES_MAX:
        why.append(f"system pages input/s {c['pages_in_mean']} > {PAGES_MAX:g}")
    return ("VOID", "; ".join(why)) if why else ("OK", "")


def calibrate(rates: dict):
    """rates: {N: [run rates, None for a run that is not OK]}. The smallest N whose runs all reach
    CAL_MIN_TOKPS; None if none does (the sitting then takes N = 8 and says so)."""
    for n in CAL_THREADS:
        r = rates.get(n, [])
        if len(r) == CAL_RUNS and all(v is not None and v >= CAL_MIN_TOKPS for v in r):
            return n
    return None


# the rules, tri-state: True, False, or None (undecided: an input is INCOMPLETE)

def t_and(*xs):
    return False if any(x is False for x in xs) else (True if all(x is True for x in xs) else None)


def t_or(*xs):
    return True if any(x is True for x in xs) else (False if all(x is False for x in xs) else None)


def t_not(x):
    return None if x is None else not x


def k_ge(k, x):
    return None if k is None else k * (1 + TIE_EPS) >= x


def k_lt(k, x):
    return t_not(k_ge(k, x))


def ratio_ge(kn, kr):
    """K(NPU) >= 1.10 x K(rival); exactly 1.10x counts."""
    return None if kn is None or kr is None else kn * TIE_DEN * (1 + TIE_EPS) >= kr * TIE_NUM


def label(x) -> str:
    return {True: "PASS", False: "FAIL", None: "INCOMPLETE"}[x]


def index(recs, block: str = "matrix") -> dict:
    return {(r["pass"], r["arm"], r["w"]): r for r in recs if r["block"] == block}


def _ok(r) -> bool:
    return r is not None and r.get("state") == "OK"


def w_rate(idx, p, arm, w):
    r = idx.get((p, arm, w))
    return r["w_stats"]["rate"] if _ok(r) else None


def a_rate(idx, p, arm, w):
    r = idx.get((p, arm, w))
    return r["arm_stats"]["rate"] if _ok(r) else None


def w_agree(idx, w) -> dict:
    r = [w_rate(idx, p, None, w) for p in (1, 2)]
    if None in r:
        return {"ok": None, "alone": r, "rel": None}
    m = (r[0] + r[1]) / 2
    rel = abs(r[0] - r[1]) / m if m > 0 else float("inf")
    return {"ok": rel <= W_AGREE * (1 + TIE_EPS), "alone": r, "rel": rel}


def k_value(idx, arm, w) -> dict:
    ag = w_agree(idx, w)
    ks = []
    for p in (1, 2):
        co, al = w_rate(idx, p, arm, w), ag["alone"][p - 1]
        ks.append(co / al if co is not None and al else None)
    out = {"passes": ks, "alone": ag["alone"], "alone_rel": ag["rel"], "k": None}
    if ag["ok"] is None:
        return {**out, "why": f"a {w}-alone window is not OK"}
    if ag["ok"] is False:
        return {**out, "why": f"{w} alone disagrees between the passes ({ag['rel']:.3f} > {W_AGREE:g})"}
    if None in ks:
        return {**out, "why": "a co-run window is not OK"}
    if abs(ks[0] - ks[1]) > K_AGREE + 1e-12:
        return {**out, "why": f"the passes' K differ by {abs(ks[0] - ks[1]):.3f} > {K_AGREE:g}"}
    return {**out, "k": (ks[0] + ks[1]) / 2, "why": ""}


def holds(idx, arm, w=None):
    """>= 5.00 tok/s in that window in both passes; a valid window below 5.00 decides False."""
    v = [a_rate(idx, p, arm, w) for p in (1, 2)]
    if any(x is not None and x < FLOOR_TOKPS for x in v):
        return False
    return True if all(x is not None for x in v) else None


def frees(idx, w, rival) -> tuple:
    kn, kr = k_value(idx, "NPU", w)["k"], k_value(idx, rival, w)["k"]
    parts = {f"the NPU holds with {w}": holds(idx, "NPU", w),
             f"K_{w}(NPU) >= {K_MIN:.2f}": k_ge(kn, K_MIN),
             f"K_{w}(NPU) >= 1.10 x K_{w}({rival}), or {rival} does not hold with {w}":
                 t_or(ratio_ge(kn, kr), t_not(holds(idx, rival, w)))}
    return t_and(*parts.values()), parts


def joint(idx) -> tuple:
    k = lambda arm, w: k_value(idx, arm, w)["k"]  # noqa: E731
    parts = {}
    for w in WS:
        parts[f"the NPU holds with {w}, K >= {K_MIN:.2f}"] = t_and(holds(idx, "NPU", w), k_ge(k("NPU", w), K_MIN))
    parts["CPU gives up Wcpu (K < 0.90, or K(NPU) >= 1.10 x K(CPU), or CPU does not hold)"] = t_or(
        k_lt(k("CPU", "Wcpu"), K_MIN), ratio_ge(k("NPU", "Wcpu"), k("CPU", "Wcpu")), t_not(holds(idx, "CPU", "Wcpu")))
    for w in ("W1", "W2"):
        parts[f"DML gives up {w} (K < 0.90, or K(NPU) >= 1.10 x K(DML), or DML does not hold)"] = t_or(
            k_lt(k("DML", w), K_MIN), ratio_ge(k("NPU", w), k("DML", w)), t_not(holds(idx, "DML", w)))
    return t_and(*parts.values()), parts


def in_range(v, rng) -> bool:
    lo, hi = rng
    return (lo is None or v >= lo) and (hi is None or (v <= hi if lo is not None else v < hi))


def rng_text(rng) -> str:
    lo, hi = rng
    return f">= {lo:.2f}" if hi is None else (f"< {hi:.2f}" if lo is None else f"{lo:.2f}-{hi:.2f}")


def predictions(idx, cal_n) -> list:
    out = []
    for arm in ARMS:
        h = holds(idx, arm)
        out.append((f"{arm} holds alone", "yes", "UNSCORED (INCOMPLETE)" if h is None else ("HIT" if h else "MISS")))
    out.append(("the CPU arm's calibrated N", "2 or 4", "UNSCORED (no calibration)" if cal_n is None else
                f"{'HIT' if cal_n in PRED_CAL_N else 'MISS'} ({cal_n})"))
    for (w, arm), rng in PRED_K.items():
        kv = k_value(idx, arm, w)["k"]
        out.append((f"K_{w}({arm})", rng_text(rng), "UNSCORED (INCOMPLETE)" if kv is None else
                    f"{'HIT' if in_range(kv, rng) else 'MISS'} ({kv:.3f})"))
    for what, got in (("frees the GPU on W1", frees(idx, "W1", "CPU")[0]), ("frees the CPU", frees(idx, "Wcpu", "DML")[0])):
        out.append((what, "FAIL", "UNSCORED (INCOMPLETE)" if got is None else f"{'HIT' if got is False else 'MISS'} "
                                                                             f"({label(got)})"))
    out.append(("the joint score", "OPEN (no prediction)", label(joint(idx)[0])))
    return out


# ---------------------------------------------------------------- the coordinator

class Proc:
    """One child process; the command it logs is relative to the repository, the interpreter by env."""

    def __init__(self, cmd: list, shown: str):
        self.shown, self.lines, self.ready, self.ready_wall = shown, [], threading.Event(), None
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()

    def _pump(self):
        for line in self.proc.stdout:
            s = line.rstrip("\n")
            self.lines.append(s)
            if s == "READY" or s.startswith("READY "):
                self.ready_wall = time.time()
                self.ready.set()
        self.ready.set()

    def json(self, tag: str):
        return next((json.loads(s.split(" ", 1)[1]) for s in self.lines if s.startswith(tag + " ")), None)

    def go(self, start: float, stop: float):
        self.proc.stdin.write(f"go {start:.6f} {stop:.6f}\n")
        self.proc.stdin.flush()


def py_proc(args: list) -> Proc:
    args = ["tools/llm_freeing.py", *args]
    return Proc([sys.executable, *args], f"python[{os.environ.get('CONDA_DEFAULT_ENV', '?')}] " + " ".join(args))


def runner_proc() -> Proc:
    cmd = proxy_cmd(RUNNER_S)
    return Proc([str(RUNNER_EXE)] + cmd[1:], " ".join(cmd))


QUIET = ("READER_JSON ", "W_JSON ", "TOKEN_JSON ")    # summarized in WINDOW_JSON instead


def cpu_seconds(pids: dict) -> dict:
    import psutil
    out = {}
    for role, pid in pids.items():
        try:
            t = psutil.Process(pid).cpu_times()
            out[role] = t.user + t.system
        except psutil.Error:
            pass
    return out


LUIDS_780M = None


def arm_record(p, arm: str, a: float, b: float) -> dict:
    if p is None:
        return {"missing": "no process"}
    if arm == "NPU":
        toks, done = paced_tokens(p.lines), p.json("PACED_DONE_JSON")
        if done is None:
            return {"missing": "no PACED_DONE_JSON", "exit": p.proc.returncode}
        t0, t1 = [t["t0_ns"] / 1e9 for t in toks], [t["t1_ns"] / 1e9 for t in toks]
        st = token_stats(t0, t1, a, b)
        st.update(exit=p.proc.returncode, tokens_total=len(toks), late_starts=done["late_starts"],
                  setup=p.json("PACED_SETUP_JSON"), ready_wall=round(p.ready_wall, 3),
                  covered=p.ready_wall + RUNNER_S >= b + 1.0 and bool(t0) and t0[0] <= a)
        return st
    r, ready = p.json("READER_JSON"), p.json("READY")
    if r is None or ready is None:
        return {"missing": "no READER_JSON", "exit": p.proc.returncode}
    st = token_stats(r["t0"], r["t1"], a, b)
    st.update(exit=p.proc.returncode, tokens_total=r["tokens"], generators=r["generators"], late_starts=r["late_starts"],
              first_tokens=r["first_tokens"], pin_ok=ready["pin_ok"], single_cpu_threads=ready["single_cpu_threads"],
              load_s=ready["load_s"], covered=bool(r["t0"]) and r["t0"][0] <= a and r["stopped"] >= b)
    return st


def w_record(p, w: str, a: float, b: float) -> dict:
    if p is None:
        return {"missing": "no process"}
    r, ready = p.json("W_JSON"), p.json("READY")
    if r is None or ready is None:
        return {"missing": "no W_JSON", "exit": p.proc.returncode}
    st = unit_stats(r["t0"], r["t1"], a, b, UNIT_SIZE[w])
    st.update(exit=p.proc.returncode, unit=UNIT_NAME[w], units_total=r["units"], pin_ok=ready.get("pin_ok"))
    return st


def run_window(stamp: str, n: int, block: str, pas: int, arm, w, threads=None, paced: bool = True) -> dict:
    import gemma_decode_suite as gds
    import psutil
    from silicon_probe_record import witness
    tag = f"{n:02d}_{block}_p{pas}_{arm or 'none'}_{w or 'none'}"
    print(f"\nWINDOW_BEGIN {tag} {utc()}", flush=True)
    witness()
    time.sleep(GAP_S)
    rec = {"n": n, "block": block, "pass": pas, "arm": arm, "w": w, "threads": threads, "paced": paced,
           "available_gb_before": round(psutil.virtual_memory().available / 1e9, 2), "utc": utc()}
    procs, tp, fs = {}, None, None
    path = TMP / f"{stamp}_{tag}.csv"
    a = b = None
    try:
        if w:
            procs["w"] = py_proc(["wload", "--kind", w])
        if arm in ARM_MODEL:
            procs["arm"] = py_proc(["reader", "--arm", arm, "--threads", str(threads)] + ([] if paced else ["--unpaced"]))
        deadline = time.monotonic() + READY_TIMEOUT_S
        for p in procs.values():
            p.ready.wait(max(0.0, deadline - time.monotonic()))
        missing = [k for k, p in procs.items() if p.json("READY") is None]
        if missing:
            raise RuntimeError(f"not READY: {missing}")
        if arm == "NPU":                                  # last: it paces from its own READY
            procs["arm"] = runner_proc()
            if not procs["arm"].ready.wait(RUNNER_READY_S) or procs["arm"].ready_wall is None:
                raise RuntimeError("the proxy did not report READY")
        pids = {k: p.proc.pid for k, p in procs.items()}
        rec["pids"] = pids
        tp = gds.typeperf(TP_SAMPLES, path)
        fs = gds.FaultSampler(counters=neutral_counters).start()
        time.sleep(TP_LEAD_S)
        start = time.time() + GO_LEAD_S
        a, b = start + SETTLE_S, start + SETTLE_S + WINDOW_S
        stop = b + TAIL_S
        rec["window"] = [round(a, 3), round(b, 3)]
        for k, p in procs.items():
            if not (k == "arm" and arm == "NPU"):
                p.go(start, stop)
        wait_until(a)
        ca = cpu_seconds(pids)
        wait_until(b)
        cb = cpu_seconds(pids)
        rec["process_cpu_pct"] = {k: round((cb[k] - ca[k]) / (b - a) * 100, 1) for k in pids if k in ca and k in cb}
        for p in procs.values():
            p.proc.wait(timeout=max(60.0, stop - time.time() + 300))
            p.pump.join(timeout=30)
        time.sleep(1.5)                                   # the window's last second is sampled
        gds.stop(tp, 0)
        tp = None
        fs.stop()
        rows = gds.inside(gds.read_csv(path, None, LUIDS_780M)["rows"], a, b)
        f = lambda i: round(statistics.fmean(r[i] for r in rows), 2) if rows else None  # noqa: E731
        rec["counters"] = {"rows": len(rows), "cpu_total_pct": f(3), "pkg_w": round(f(1) / 1e3, 2) if rows else None,
                           "pages_in_mean": f(8) or 0.0, "pages_in_max": max((r[8] for r in rows), default=None),
                           "avail_mb_min": min((r[7] for r in rows), default=None)}
        g, grows = gds.gpu_frame(path)
        gw = {k: gds.gpu_window(g, grows, set(LUIDS_780M), pid, a, b) for k, pid in pids.items()}
        any_w = next(iter(gw.values()), {})
        rec["gpu"] = {"rows": any_w.get("rows"), "gpu780_all": any_w.get("gpu780_all"),
                      "other_adapters": any_w.get("other_adapters"),
                      "by_role": {k: v.get("reader") for k, v in gw.items()}}
        rec["faults"] = faults_window(fs.rows, a, b, pids)
    except Exception as e:                               # recorded; the window is FAILED and feeds nothing
        rec["failed"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        for p in procs.values():
            if p.proc.poll() is None:
                p.proc.kill()                             # our own child, by its pid
        if tp is not None:
            gds.stop(tp, 0)
        if fs is not None:
            fs.stop()
        for k, p in procs.items():
            print(f"PROC_CMD {k} {p.shown}", flush=True)
            for s in p.lines:
                if not s.startswith(QUIET):
                    print(neutralize(f"{k}| {s}"), flush=True)
    if a is not None:
        if arm:
            rec["arm_stats"] = arm_record(procs.get("arm"), arm, a, b)
        if w:
            rec["w_stats"] = w_record(procs.get("w"), w, a, b)
    rec["state"], rec["why"] = window_state(rec)
    say("WINDOW_JSON", rec)
    parts = [f"{rec['state']}"]
    if rec.get("arm_stats", {}).get("rate") is not None:
        parts.append(f"{arm} {rec['arm_stats']['rate']:.3f} tok/s (work median {rec['arm_stats']['work_ms_median']} ms)")
    if rec.get("w_stats", {}).get("rate") is not None:
        parts.append(f"{w} {rec['w_stats']['rate']:.3f} {UNIT_NAME[w]}")
    if rec.get("counters"):
        parts.append(f"pages {rec['counters']['pages_in_mean']}/s, hard {rec.get('faults', {}).get('hard_per_s')}")
    print("WINDOW_SUMMARY", tag, "; ".join(parts) + (f"  ({rec['why']})" if rec["why"] else ""), flush=True)
    if arm == "NPU":
        witness()
    return rec


def check_pins() -> dict:
    import gemma_decode as gd
    import gemma_decode_suite as gds
    got, bad = {}, []
    for arm in DECODE_MODELS:
        for f, want in gds.MODEL_SHA[arm].items():
            h = sha256(gd.ONNX / arm / f)
            got[f"{arm}/{f}"] = h
            if h != want:
                bad.append(f"{arm}/{f}")
    for rel, want in FILE_PIN.items():
        p = ROOT / rel
        h = sha256(p) if p.exists() else None
        got[rel] = h
        if h != want:
            bad.append(rel)
    return {"sha256": got, "mismatch": bad}


class Live:
    """The sitting's lines, also written as they come to a scratch file: the committed log is written
    by silicon_probe_record when the sitting ends, so a killed wrapper loses nothing."""

    def __init__(self, path: Path):
        self.f, self.out = path.open("x", encoding="utf-8", newline="\n"), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()

    def __getattr__(self, name):                         # anything else (encoding, isatty, ...) is the stream's
        return getattr(self.out, name)


def suite() -> int:
    import gemma_decode_suite as gds
    global LUIDS_780M
    TMP.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    sys.stdout = Live(TMP / f"{stamp}_suite_live.log")
    names = neutral_list()
    set_neutral(names)
    print(f"SUITE (e), stamp {stamp}, {utc()}", flush=True)
    print(f"NEUTRAL_NAMES {len(names)} (a local, git-ignored list; the names themselves are not logged)", flush=True)
    if not names:
        print("NEUTRAL_REFUSE the local name list is empty or missing", flush=True)
        return 3
    if not host_gate():
        return 3
    m = gds.memory_start()
    for key in ("processes_over_1gb", "refuse"):
        for e in m[key]:
            e["name"] = neutralize(e["name"])
    say("MEMORY_START_JSON", m)
    if m["refuse"]:
        print(f"MEMORY_REFUSE a process holds >= {MEM_BIG_GB:g} GB private at the start: {m['refuse']}", flush=True)
        return 3
    ad = gds.dxgi_adapters()
    LUIDS_780M = sorted(x["luid"] for x in ad if (x["vendor"], x["device"]) == gds.GPU_780M_ID and not x["flags"] & 2)
    say("ADAPTERS_JSON", {"adapters": ad, "luids_780m": LUIDS_780M})
    hw = [x for x in ad if not x["flags"] & 2]
    if not hw or hw[0] is not ad[0] or any((x["vendor"], x["device"]) != gds.GPU_780M_ID for x in hw):
        print(f"ADAPTER_REFUSE a hardware adapter other than the 780M, or adapter 0 not the 780M: {ad}", flush=True)
        return 3
    p = check_pins()
    say("PINS_JSON", p)
    if p["mismatch"]:
        print("PIN_MISMATCH", p["mismatch"], flush=True)
        return 3
    say("PROTOCOL_JSON", protocol())
    say("COUNTER_FILES_JSON", {"dir": TMP.relative_to(ROOT).as_posix(), "stamp": stamp,
                               "names": "<stamp>_<n>_<block>_p<pass>_<arm>_<w>.csv"})
    n = 0
    rates = {}
    for threads in CAL_THREADS:                          # U6: the calibration, unpaced, each N twice
        for run in range(1, CAL_RUNS + 1):
            rec = run_window(stamp, n, "cal", run, "CPU", None, threads, paced=False)
            n += 1
            rates.setdefault(threads, []).append(rec["arm_stats"]["rate"] if _ok(rec) else None)
        say("CAL_JSON", {"threads": threads, "rates": rates[threads], "min": CAL_MIN_TOKPS})
        if calibrate({threads: rates[threads]}) is not None:
            break
    chosen = calibrate(rates)
    say("CALIBRATION_JSON", {"threads": chosen if chosen is not None else CAL_THREADS[-1], "passed": chosen is not None,
                             "rates": rates, "min": CAL_MIN_TOKPS})
    if chosen is None:
        print(f"CALIBRATION_NONE no N reached {CAL_MIN_TOKPS} tok/s in both runs; the CPU arm runs at N = "
              f"{CAL_THREADS[-1]}", flush=True)
    threads = chosen if chosen is not None else CAL_THREADS[-1]
    for pas, arm, w in ORDER:                            # the matrix: 15 windows per pass, pass 2 mirrored
        run_window(stamp, n, "matrix", pas, arm, w, threads if arm == "CPU" else (DML_THREADS if arm == "DML" else None))
        n += 1
    for pas, arm, w in WORST_ORDER:                      # report-only: (c)'s same-core setup against Wcpu
        run_window(stamp, n, "worst", pas, arm, w, len(EVEN) if arm == "CPU-c" else None)
        n += 1
    print(f"\nSUITE_DONE {n} windows {utc()}", flush=True)
    return 0


LOADCHECK_CASES = (("CPU N=1 unpaced", ["reader", "--arm", "CPU", "--threads", "1", "--unpaced"]),
                   ("CPU N=4 paced", ["reader", "--arm", "CPU", "--threads", "4"]),
                   ("DML", ["reader", "--arm", "DML", "--threads", str(DML_THREADS)]),
                   ("CPU-c", ["reader", "--arm", "CPU-c", "--threads", str(len(EVEN))]),
                   ("W1", ["wload", "--kind", "W1"]), ("W2", ["wload", "--kind", "W2"]), ("Wcpu", ["wload", "--kind", "Wcpu"]))
LOADCHECK_GO_S = 2.0
LOADCHECK_PROXY_S = 3.0


def loadcheck() -> int:
    """The load check (pre-sitting, disclosed, enters no rule; the gate's approval of 2026-09-24): every
    child the sitting spawns, started to READY by the sitting's own code path, sent the sitting's go line
    with its stop 2 s after its start (so its loop and its final record run once), and parsed by the
    sitting's record code. No window, no counters, no fault sampler."""
    import gemma_decode_suite as gds
    from silicon_probe_record import witness
    global LUIDS_780M
    TMP.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    sys.stdout = Live(TMP / f"{stamp}_loadcheck_live.log")
    names = neutral_list()
    set_neutral(names)
    print(f"LOADCHECK (e), pre-sitting, go/no-go only; enters no rule. {utc()}", flush=True)
    print(f"  Each child: READY, then the go line with its stop {LOADCHECK_GO_S:g} s after its start, then its record; "
          f"the proxy runs a {LOADCHECK_PROXY_S:g} s schedule. No window, no counters, no fault sampler.", flush=True)
    print(f"NEUTRAL_NAMES {len(names)} (a local, git-ignored list; the names themselves are not logged)", flush=True)
    bad = [] if names else ["the local name list is empty or missing"]
    if not host_gate():
        bad.append("the host-load gate")
    m = gds.memory_start()
    for key in ("processes_over_1gb", "refuse"):
        for e in m[key]:
            e["name"] = neutralize(e["name"])
    say("MEMORY_START_JSON", m)
    ad = gds.dxgi_adapters()
    LUIDS_780M = sorted(x["luid"] for x in ad if (x["vendor"], x["device"]) == gds.GPU_780M_ID and not x["flags"] & 2)
    say("ADAPTERS_JSON", {"adapters": ad, "luids_780m": LUIDS_780M})
    nc = neutral_counters()                              # the fault sampler's source, called once (no sampling)
    print("NEUTRAL_COUNTERS", json.dumps({"processes": len(nc), "self": (nc.get(os.getpid()) or ["?"])[0]}), flush=True)

    def finish(p, key, kind, start, stop) -> dict:
        try:
            p.proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            p.proc.kill()                                # our own child, by its pid
            p.proc.wait(timeout=30)
        p.pump.join(timeout=30)
        print(f"PROC_CMD {key} {p.shown}", flush=True)
        for s in p.lines:
            if not s.startswith(QUIET):
                print(neutralize(f"{key}| {s}"), flush=True)
        if start is None:
            return {"exit": p.proc.returncode, "record": None}
        st = (arm_record(p, kind, start, stop) if kind in ARM_MODEL or kind == "NPU" else w_record(p, kind, start, stop))
        rec = {"arm_stats" if kind in ARM_MODEL or kind == "NPU" else "w_stats": st, "counters": {"rows": MIN_ROWS},
               "faults": {"seconds": MIN_ROWS, "hard_per_s": {}}}
        window_state(rec)                                # its code path runs; with no window its label means nothing
        return {"exit": p.proc.returncode, "record": {k: v for k, v in st.items() if k not in ("per_second", "setup")},
                "window_state_code_ran": True}

    for name, args in LOADCHECK_CASES:
        kind = args[2]                                   # --arm CPU/DML/CPU-c or --kind W1/W2/Wcpu
        p = py_proc(args)
        ready = p.ready.wait(READY_TIMEOUT_S) and p.json("READY") is not None
        start = stop = None
        cpu = {}
        if ready:
            cpu = cpu_seconds({"child": p.proc.pid})
            start = time.time() + GO_LEAD_S
            stop = start + LOADCHECK_GO_S
            p.go(start, stop)
        res = finish(p, "arm" if args[0] == "reader" else "w", kind, start, stop)
        rec = {"case": name, "ready": ready, "cpu_seconds_read": ready and "child" in cpu, **res}
        say("LOADCHECK_JSON", rec)
        st = res.get("record") or {}
        if not ready or res["exit"] != 0 or not st or st.get("missing") or st.get("pin_ok") is False:
            bad.append(name)
    witness()
    p = Proc([str(RUNNER_EXE)] + proxy_cmd(LOADCHECK_PROXY_S)[1:], " ".join(proxy_cmd(LOADCHECK_PROXY_S)))
    ready = p.ready.wait(RUNNER_READY_S) and p.ready_wall is not None
    a = p.ready_wall
    res = finish(p, "arm", "NPU", a, (a + LOADCHECK_PROXY_S) if a else None)
    rec = {"case": "NPU proxy", "ready": ready, **res,
           "tokens": len(paced_tokens(p.lines)), "done": p.json("PACED_DONE_JSON")}
    say("LOADCHECK_JSON", rec)
    if not ready or res["exit"] != 0 or rec["done"] is None or not rec["tokens"]:
        bad.append("NPU proxy")
    witness()
    for b in bad:
        print("LOADCHECK_PROBLEM", b, flush=True)
    print("LOADCHECK", "GO" if not bad else "NO-GO", flush=True)
    return 0 if not bad else 2


def protocol() -> dict:
    return {"period_ms": PERIOD_MS, "floor": FLOOR_TOKPS, "cal_threads": CAL_THREADS, "cal_min_tokps": CAL_MIN_TOKPS,
            "cal_runs": CAL_RUNS, "dml_threads": DML_THREADS, "even": EVEN, "odd": ODD, "arm_model": ARM_MODEL,
            "order": ORDER, "worst_order": WORST_ORDER, "gap_s": GAP_S, "tp_lead_s": TP_LEAD_S, "go_lead_s": GO_LEAD_S,
            "settle_s": SETTLE_S, "window_s": WINDOW_S, "tail_s": TAIL_S, "runner_s": RUNNER_S, "min_rows": MIN_ROWS,
            "warmup": WARMUP, "k_min": K_MIN, "k_agree": K_AGREE, "w_agree": W_AGREE, "tie": [TIE_NUM, TIE_DEN],
            "hf_max": HF_MAX, "pages_max": PAGES_MAX, "mem_big_gb": MEM_BIG_GB, "reading_tokps": READING_TOKPS,
            "proxy": {"name": PROXY_NAME, "dispatches": DISPATCHES, "bytes_per_dispatch": PROXY_WORDS * 4,
                      "bytes_per_token": PROXY_WORDS * 4 * DISPATCHES, "cmd": " ".join(proxy_cmd(RUNNER_S))},
            "pred_k": {f"{w}/{a}": list(r) for (w, a), r in PRED_K.items()}, "pred_cal_n": PRED_CAL_N}


# ---------------------------------------------------------------- the verdict

def verdict(log: Path) -> int:
    lines = log.read_text(encoding="utf-8").splitlines()
    grab = lambda tag: [json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith(tag + " ")]  # noqa: E731
    problems = []
    pins = next(iter(grab("PINS_JSON")), None)
    if pins is None or pins["mismatch"]:
        problems.append(f"pins: {pins and pins['mismatch']}")
    recs = grab("WINDOW_JSON")
    cal = next(iter(grab("CALIBRATION_JSON")), None)
    print(f"(e) verdict over {log.name} (sha256 {sha256(log)} working copy, {sha256_lf(log)} LF blob)")
    print("\nCalibration (U6): the CPU arm alone, unpaced; N is the smallest whose two runs reach "
          f">= {CAL_MIN_TOKPS} tok/s")
    for r in (x for x in recs if x["block"] == "cal"):
        st = r.get("arm_stats", {})
        print(f"  N={r['threads']} run {r['pass']}: {r['state']:6} "
              + (f"{st['rate']:.3f} tok/s, work median {st['work_ms_median']} ms, process CPU "
                 f"{r.get('process_cpu_pct', {}).get('arm')}%" if st.get("rate") is not None else "")
              + (f"  ({r['why']})" if r["why"] else ""))
    cal_n = cal["threads"] if cal and cal["passed"] else None
    print(f"  -> N = {cal['threads'] if cal else '?'}" + ("" if cal_n is not None else "  (no N passed: the 8-thread arm)"))
    idx = index(recs)
    print(f"\nThe matrix: {len(idx)} of {len(ORDER)} windows logged")
    print(f"{'pass':>4} {'arm':5} {'w':5} {'state':7} {'arm tok/s':>9} {'work ms':>8} {'W rate':>9} {'unit':13} "
          f"{'pages/s':>8} {'hard/s arm,w':>13} {'CPU%':>5} {'780M':>6}")
    for pas, arm, w in ORDER:
        r = idx.get((pas, arm, w))
        if r is None:
            print(f"{pas:>4} {arm or '-':5} {w or '-':5} MISSING")
            problems.append(f"window pass {pas} {arm} {w} missing")
            continue
        st, ws = r.get("arm_stats", {}), r.get("w_stats", {})
        hf = r.get("faults", {}).get("hard_per_s", {})
        print(f"{pas:>4} {arm or '-':5} {w or '-':5} {r['state']:7} "
              + (f"{st['rate']:9.3f} {st['work_ms_median'] or 0:8.1f} " if st.get("rate") is not None else f"{'-':>9} {'-':>8} ")
              + (f"{ws['rate']:9.3f} {UNIT_NAME[w]:13} " if ws.get("rate") is not None else f"{'-':>9} {'':13} ")
              + f"{r.get('counters', {}).get('pages_in_mean', 0):8.1f} "
              + f"{str(hf.get('arm', '-')) + ',' + str(hf.get('w', '-')):>13} "
              + f"{r.get('counters', {}).get('cpu_total_pct') or 0:5.1f} {r.get('gpu', {}).get('gpu780_all') or 0:6.1f}"
              + (f"  ({r['why']})" if r["why"] else ""))
        if r["state"] != "OK":
            top = r.get("faults", {}).get("top_hard")
            if top:
                print(f"       top hard faults/s: {top}")
    print(f"\nHolds (>= {FLOOR_TOKPS:.2f} tok/s in both passes); the pace is {1000 / PERIOD_MS:.2f}; the reading-rate "
          f"column (U2, report-only) is {READING_TOKPS[0]:.2f}-{READING_TOKPS[1]:.2f} tok/s")
    for arm in ARMS:
        cap = [idx.get((p, arm, None), {}).get("arm_stats", {}).get("work_ms_median") for p in (1, 2)]
        caps = ", ".join(f"{1000 / c:.1f}" for c in cap if c)
        print(f"  {arm}: alone {label(holds(idx, arm))}; " + ", ".join(f"with {w} {label(holds(idx, arm, w))}" for w in WS)
              + f"; unpaced capacity from its alone work per token: {caps or '-'} tok/s")
    print("\nK_W(arm) = W's rate with the arm / alone, per pass; K is the passes' mean if they differ by <= 0.05")
    for w in WS:
        ag = w_agree(idx, w)
        print(f"  {w} alone: " + " / ".join("-" if x is None else f"{x:.3f}" for x in ag["alone"]) + " " + UNIT_NAME[w]
              + ("" if ag["rel"] is None else f", passes differ by {ag['rel']:.3f} (<= {W_AGREE:g} to agree)"))
        for arm in ARMS:
            kv = k_value(idx, arm, w)
            print(f"    K_{w}({arm}) = " + ("INCOMPLETE" if kv["k"] is None else f"{kv['k']:.3f}") + "  passes "
                  + " / ".join("-" if x is None else f"{x:.3f}" for x in kv["passes"]) + (f"  ({kv['why']})" if kv["why"] else ""))
    res = {}
    print("\nThe rules")
    for w in ("W1", "W2"):
        res[f"gpu_{w}"], parts = frees(idx, w, "CPU")
        print(f"  frees the GPU on {w} (against the CPU arm): {label(res[f'gpu_{w}'])}")
        for k, v in parts.items():
            print(f"      {k}: {label(v)}")
    gpu = t_and(res["gpu_W1"], res["gpu_W2"])
    print(f"  frees the GPU (unqualified: W1 and W2): {label(gpu)}")
    res["cpu"], parts = frees(idx, "Wcpu", "DML")
    print(f"  frees the CPU on Wcpu (against the DirectML arm): {label(res['cpu'])}")
    for k, v in parts.items():
        print(f"      {k}: {label(v)}")
    res["joint"], parts = joint(idx)
    print(f"  frees both at once (joint): {label(res['joint'])}")
    for k, v in parts.items():
        print(f"      {k}: {label(v)}")
    roles = [s for s, x in (("frees the GPU", gpu), ("frees the CPU", res["cpu"]), ("frees both at once", res["joint"]))
             if x is True]
    roles += [f"frees the GPU on {w} only" for w in ("W1", "W2") if res[f"gpu_{w}"] is True and gpu is not True]
    print("  role: " + ("; ".join(roles) if roles else "none earned") + " (the proxy is not a role: a PASS only "
          "justifies building the GEMV, which the user decides)")
    worst = index(recs, "worst")
    if worst:
        print("\nReport-only, not a rival: (c)'s CPU setup (8 threads on 0, 2, ..., 14, default spinning) against Wcpu")
        for p in (1, 2):
            al, co, ar = worst.get((p, None, "Wcpu")), worst.get((p, "CPU-c", "Wcpu")), worst.get((p, "CPU-c", None))
            k = (co["w_stats"]["rate"] / al["w_stats"]["rate"]) if _ok(al) and _ok(co) else None
            print(f"  pass {p}: K_Wcpu(CPU-c) = " + ("-" if k is None else f"{k:.3f}")
                  + (f"; its tok/s alone {ar['arm_stats']['rate']:.3f}" if _ok(ar) else "")
                  + (f", with Wcpu {co['arm_stats']['rate']:.3f}" if _ok(co) else ""))
    print("\nPredictions")
    for tag, what, score in predictions(idx, cal_n):
        print(f"  {tag}: {what}: {score}")
    undecided = [k for k, v in (("frees the GPU on W1", res["gpu_W1"]), ("on W2", res["gpu_W2"]),
                                ("frees the CPU", res["cpu"]), ("joint", res["joint"])) if v is None]
    for p in problems:
        print("PROBLEM", p)
    if undecided:
        print("UNDECIDED", ", ".join(undecided))
    print("VERDICT", "INCOMPLETE" if problems or undecided else "COMPLETE")
    return 2 if problems or undecided else 0


# ---------------------------------------------------------------- the prereg

PREREG = f"""Gemma 3 4B decode, stage (e): does an NPU decode free the GPU, or the CPU, where the CPU and
DirectML decodes do not? (LLM study, locked decision 10; plan v4, approved by the user on 2026-09-24
with U1-U6 as recommended), pre-registered before the sitting

Question
  An NPU arm earns a role if it frees the GPU, or frees the CPU, against the rival that frees the same
  resource. Its distinct claim is freeing both at once: the CPU decode leaves the GPU free, and the
  DirectML decode leaves most CPU cores free. The floor is {FLOOR_TOKPS:.2f} tok/s (U2). Energy is not part
  of (e). U1: (e) runs now, with no third (c) sitting.

The arms: Gemma 3 4B at H4 ({GB_TOKEN_H4:,} B of weights per token), paced on an absolute {PERIOD_MS:g} ms
schedule ({1000 / PERIOD_MS:.2f} tok/s: a pacer capped at exactly 5 could never show ">= 5"). Each is one
long-lived session per window.
  CPU   C0-H4 (U3; (c)'s one complete H4 arm) in its most-freeing configuration (U6): the caller on logical
        CPU 1 and N - 1 pool threads on 3, 5, ... (the odd CPUs, the SMT siblings of Wcpu's), intra-op
        spinning off (session.intra_op.allow_spinning = 0 through genai_config's session_options.config_entries,
        the path (c)'s build measured to reach ORT), the masks read back. N is calibrated in the sitting.
  DML   D-H4 on the 780M: 1 intra-op thread, spinning off, unpinned (its host thread mostly waits).
  NPU   a PROXY (U4), since no NPU int4 GEMV exists: stage 1's shim-read design at 4 columns x 2 channels,
        reading one H4 token's bytes in {DISPATCHES} dependent dispatches of {PROXY_WORDS * 4:,} B ({PROXY_WORDS * 4 * DISPATCHES:,} B per
        token, {PROXY_WORDS * 4 * DISPATCHES / GB_TOKEN_H4:.4f}x H4), each from its own host buffer, by the C++ runner's pacing mode,
        unpinned. Its host thread is a cost against Wcpu that a real NPU decode would also pay, so it is
        measured, not corrected out. A GEMV reads at least these bytes and adds synchronization, so a proxy
        FAIL is robust on the GPU side; a proxy PASS is not a role, only a reason to build the GEMV, which the
        user decides. The proxy has no accuracy axis.
  The GenAI arms start a fresh generator from (c)'s pinned 128-token prompt and restart it after 896
  generated tokens (before position 1024), so F3 and F4 change nothing measured; the prompt passes are
  part of the paced stream.

The workloads, each also measured alone in the same pass
  W1    the GPU bandwidth worst case: stage 2's DirectML read loop (a 1 GiB fp32 ReduceSum), in GB/s.
  W2    a GPU application: YOLO-World v2 FP32 on DirectML, 4b_g2g's frame loop (letterbox, network,
        decode and NMS) on bus.jpg with COCO's 80 names, flat out, in frames/s. It uses the CPU for its own
        pre- and post-processing. The text encoder's CPU session is released before READY.
  Wcpu  CPU-bound: the same model on an ORT CPU session, 8 intra-op threads, the caller on logical CPU 0
        and the pool on 2, 4, ..., 14 (masks read back), looping a letterboxed bus.jpg, in inferences/s.

The sitting (resnet_env17; a START REQUEST to BFP16 first; no other load is launched)
  Start: the host-load gate, the memory start (a process of >= {MEM_BIG_GB:g} GB private refuses the sitting), the
  DXGI adapter refusal, every pin below, and xrt-smi idle.
  Calibration (U6): the CPU arm alone and UNPACED at N = 1, 2, 4, 8, each in {CAL_RUNS} windows, from N = 1 up;
  N is the first whose runs both reach >= {CAL_MIN_TOKPS} tok/s (1.25 x 5.26: <= 152 ms of work per 190 ms).
  A calibration window that is not OK does not pass. If no N passes, N = 8 and the report says so. Each
  run's tok/s and the arm's process CPU% (the spin entry's effect) are logged.
  The matrix, per pass: each arm alone (3), each W alone (3), each W with each arm (9): 15 windows. Pass 2
  is pass 1 reversed. The order is in PROTOCOL_JSON.
  A window: its processes start and load and report READY; the proxy starts last and paces from its own
  READY; the counters and the fault sampler start; {TP_LEAD_S:g} s later every other process gets a common start;
  a {SETTLE_S:g} s settle; the {WINDOW_S:g} s measured window [a, b]; a {TAIL_S:g} s tail. {GAP_S:g} s between windows.
  Report-only, after the matrix, not a rival: (c)'s CPU setup (8 threads, the caller on 0 and the pool on
  2, ..., 14, the same CPUs as Wcpu, default spinning), paced, against Wcpu: Wcpu alone, that arm alone,
  together, then mirrored. Stopping before it changes no rule.
  Time (DERIVED): a window is about 88 s plus its processes' load (10-30 s), so 12-16 min of calibration,
  50-60 of matrix and 10-12 of worst case, about 75-90 min in all (plan v4 estimated about 100).

The numbers
  An arm's tok/s: the tokens whose completion lies in (a, b], over 60 s.
  A W's rate: each unit's overlap with [a, b] as a fraction of that unit, summed, times the unit (1 GiB for
  W1), over 60 s.
  K_W(arm) = W's rate in the co-run window / W's rate alone, within one pass.

The rules (tri-state: a decided failing input fails a condition; an undecided one leaves it INCOMPLETE)
  Agreement: a W's alone rate must agree within 10% between the passes (|r1 - r2| / mean), or every K on
    that W is INCOMPLETE. A K whose two passes differ by more than {K_AGREE:g} (absolute) is INCOMPLETE; otherwise
    K is the passes' mean.
  Holds: an arm holds alone if it reaches >= {FLOOR_TOKPS:.2f} tok/s in its alone window in both passes; it holds
    with W only if it reaches >= {FLOOR_TOKPS:.2f} in that co-run window in both passes.
  Frees the GPU, on W in (W1, W2), all three of: the NPU holds with W; K_W(NPU) >= {K_MIN:.2f}; K_W(NPU) >= 1.10 x
    K_W(CPU arm), or the CPU arm does not hold with W. Unqualified, it needs both W1 and W2; otherwise it is
    stated with its workload.
  Frees the CPU, on Wcpu: the same three conditions, against the DirectML arm.
  Joint, "frees both at once", both of: the NPU holds, with K >= {K_MIN:.2f}, on W1, W2 and Wcpu; and each rival
    gives up the resource its decode occupies: the CPU arm on Wcpu (K < {K_MIN:.2f}, or K_Wcpu(NPU) >= 1.10 x
    K_Wcpu(CPU)), and the DirectML arm on W1 and on W2 (K < {K_MIN:.2f}, or K_W(NPU) >= 1.10 x K_W(DML)).
    Written into the rule here, as the per-resource rules have it: a rival that does not hold with that W
    also gives it up.
  One-sided results earn that role alone, with the other resource's K's beside it. The per-resource rule
  fails if both leave the resource equally free, even when the joint score passes. Exactly 1.10x counts
  (integer-safe comparisons, as (c)).
  Each arm's tok/s is reported beside the {FLOOR_TOKPS:.2f} floor and beside the reading-rate column (U2,
  report-only; derived below). The pace sits under that column by design, so each arm's unpaced capacity
  (1 / its alone work per token) is printed beside it.

Witnesses
  Memory, a rule (U5). A window is VOID if a measured process (the arm's or the W's, by pid) has its own
  hard faults above {HF_MAX:g}/s over the window, or the system's \\Memory\\Pages Input/sec over the window exceeds
  {PAGES_MAX:,.0f}. The backstop is there because DirectML's GPU allocations are paged by the kernel's video memory
  manager, whose faults may be charged to System (pid 4) rather than the reader (INFERRED risk), leaving the
  per-pid rule blind for D-H4, W1 and W2. 1,000 is 22x below a paging decode's >= 22,000 pages/s at the floor.
  Disclosed: 1,000 was chosen after seeing (c)'s maximum window of 625, for a new stage. Pages Input/sec is
  otherwise report-only. Each window logs its top 5 processes by hard faults/s.
  A window is also VOID if its counters or its fault sampler hold under {MIN_ROWS} seconds inside it, if a pinned
  process's affinities are not read back, or if a process did not run through it (the proxy's schedule must
  reach past its end); FAILED if a process exits non-zero or never reports. A window that is not OK feeds
  nothing; there is no re-run of a window and no third pass without the user.
  Report-only: the 780M's busy per process (its LUIDs from DXGI at the start; the counters start after every
  READY), the other adapters' busy (the NPU), CPU %, each measured process's CPU time over the window, and
  xrt-smi idle before every window and after every NPU window.
  Hygiene: image names on a local, git-ignored list are replaced by <tool> at the source (the host-load lines,
  the memory start and the fault attribution), and the worktree root by <repo>. The list's length is logged;
  the sitting refuses to start without it.

Predictions (written before the sitting; plan v4 section 7)
  Every arm holds alone. The DirectML arm is the risk: at about 1/3 duty the 780M may clock down; (b)'s G50 ran
    each chain 4.1-4.3x longer (no clock sampled), and 4.1x of D-H4's 63-64 ms/token alone is near 260 ms.
  The CPU arm's calibrated N is 2 or 4 (INFERRED; 1 thread's decode rate is unmeasured).
  W1: K(NPU) about 0.93 (scored 0.90-0.96), K(CPU) 0.87-0.94 (DERIVED roughly, from stage 2's 76% beside the NPU
    and 79% beside the CPU, scaled by each arm's read duty), so freeing the GPU on W1 FAILS on the 1.10 rule;
    K(DML) 0.60-0.80 (a guess, INFERRED).
  W2 (INFERRED): K(NPU) >= 0.95, K(CPU) >= 0.90 with the rival on 1-4 odd CPUs, K(DML) < 0.90.
  Wcpu (INFERRED): K(DML) and K(NPU) >= 0.90, so freeing the CPU FAILS; K(CPU) 0.80-0.95 with N threads on the SMT
    siblings.
  The joint score: OPEN, no prediction. With the most-freeing CPU arm its CPU-side condition needs K_Wcpu(CPU) <
    0.90 or K_Wcpu(NPU) >= 1.10 x K_Wcpu(CPU).

The dry run (pre-prereg, disclosed; plan v4): one run of the proxy alone, 60 s, under a START REQUEST, go/no-go
only: does the runner's pacing mode hold 190 ms at 238 x 9.17 MB, a size stage 1 never timed? Its log,
llm_freeing_dryrun_desktop2_20260924.log, is committed with this prereg; re-read below. Its numbers enter no rule.

Logs: llm_freeing_suite_desktop2_<date>.log (scripts/llm-study.sh freeing-suite; a live copy goes to
scratch/llm/freeing/<stamp>_suite_live.log as it runs) and llm_freeing_verdict_desktop2_<date>.log
(freeing-verdict). The counter files: scratch/llm/freeing/<stamp>_<n>_<block>_p<pass>_<arm>_<w>.csv.

Unverified going in: whether the spin entry changes C0-H4's paced CPU% (the calibration logs it); C0-H4's decode
rate at 1, 2 and 4 threads; the 780M's clocks under a paced decode (no clock counter); whether DirectML's paging
is charged to System (why the backstop exists); W2's and Wcpu's model identity with the 2026-09-17 sitting's
(INFERRED from the mtime; pinned here by SHA-256).
"""


def text_ratio() -> dict:
    """U2's column: Gemma tokens per word over (c)'s pinned 1023 text tokens (after BOS), decoded, with
    wikitext's " @-@ ", " @,@ " and " @.@ " joined back; counted as whitespace words holding a letter or
    digit, and as alphabetic runs (the planning count of 2026-09-24, now logged)."""
    import gemma_decode_suite as gds
    ids = gds.pinned_ids()
    text = gds.quiet_tokenizer().decode(ids[1:])
    norm = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    words = [w for w in norm.split() if re.search(r"[A-Za-z0-9]", w)]
    alpha = re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", norm)
    n = len(ids) - 1
    tpw = (n / len(words), n / len(alpha))
    return {"tokens": n, "words": len(words), "alpha_runs": len(alpha), "tokens_per_word": [round(x, 4) for x in tpw],
            "reading_tokps": [round(READING_WPM / 60 * x, 2) for x in tpw], "ids_sha": gds.ids_sha(ids)}


def prereg() -> int:
    print(PREREG)
    bad = []
    say("PROTOCOL_JSON", protocol())
    p = check_pins()
    for k, h in p["sha256"].items():
        print(f"  PIN {k} {h}" + ("" if k not in p["mismatch"] else "  MISMATCH"))
    bad += [f"pin {m}" for m in p["mismatch"]]
    for name, (crlf, lf) in LOG_PIN.items():
        f = OUT / name
        got = (sha256(f), sha256_lf(f)) if f.exists() else (None, None)
        print(f"  LOG {name}: sha256 {got[0]} (CRLF working copy), {got[1]} (LF blob)"
              + ("" if got == (crlf, lf) else "  MISMATCH"))
        if got != (crlf, lf):
            bad.append(f"log {name}")
    print("\nPRE-PREREG DRY RUN, re-read (go/no-go only; enters no rule):")
    if dryrun_verdict(OUT / "llm_freeing_dryrun_desktop2_20260924.log") != 0:
        bad.append("the dry run's log does not read GO")
    r = text_ratio()
    say("READING_RATE_JSON", r)
    if tuple(r["reading_tokps"]) != READING_TOKPS:
        bad.append(f"READING_TOKPS {READING_TOKPS} differs from the text's {r['reading_tokps']}")
    names = neutral_list()
    print(f"NEUTRAL_NAMES {len(names)} (a local, git-ignored list; the names themselves are not logged)")
    if not names:
        bad.append("the local name list is empty or missing")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print("git HEAD", head, "(plus this log's own commit)")
    for b in bad:
        print("PIN_MISMATCH", b)
    return 3 if bad else 0


# ---------------------------------------------------------------- selftest

def selftest() -> int:
    fails = []

    def expect(what, got, want):
        print(f"  {what}: {got}" + ("" if got == want else f"  (expected {want})"))
        if got != want:
            fails.append(what)

    expect("the proxy: words per channel, bytes per token", (PROXY_WPC, PROXY_WORDS * 4 * DISPATCHES), (286720, 2183659520))
    expect("the matrix: 15 windows a pass, 30 in all, pass 2 mirrored",
           (len(MATRIX), len(ORDER), [x[1:] for x in ORDER[15:]] == [x[1:] for x in ORDER[:15]][::-1]), (15, 30, True))
    expect("the worst case: 6 windows, mirrored", (len(WORST_ORDER), WORST_ORDER[3][1:] == WORST_ORDER[2][1:]), (6, True))
    import gemma_decode as gd
    expect("EVEN is (c)'s PIN_CPUS", EVEN == gd.PIN_CPUS, True)
    so = session_options("CPU", 1)
    expect("CPU N = 1: no pool, so no affinities entry; spinning off", (so["intra_op_num_threads"],
           "session.intra_op_thread_affinities" in so["config_entries"], so["config_entries"]["session.intra_op.allow_spinning"]),
           (1, False, "0"))
    expect("CPU N = 4: the pool on 3, 5, 7 (1-based 4;6;8)",
           session_options("CPU", 4)["config_entries"]["session.intra_op_thread_affinities"], "4;6;8")
    expect("DML: 1 thread, spinning off, unpinned", (session_options("DML", 1), pool_cpus("DML", 1), caller_cpu("DML")),
           ({"intra_op_num_threads": 1, "inter_op_num_threads": 1, "config_entries": {"session.intra_op.allow_spinning": "0"}},
            [], None))
    expect("CPU-c is (c)'s overlay: 8 threads, 3;5;...;15, no spin entry", session_options("CPU-c", 8),
           {"intra_op_num_threads": 8, "inter_op_num_threads": 1,
            "config_entries": {"session.intra_op_thread_affinities": "3;5;7;9;11;13;15"}})
    # tokens and units
    t0 = [1000.0 + k * 0.19 for k in range(400)]
    t1 = [x + 0.076 for x in t0]
    st = token_stats(t0, t1, 1010.0, 1070.0)
    expect("a paced stream: 316 completions in 60 s -> 5.267 tok/s", (st["tokens"], round(st["rate"], 3)), (316, 5.267))
    st = unit_stats([0.5, 1.5, 2.5], [1.5, 2.5, 3.5], 1.0, 3.0)
    expect("units straddling both edges count by overlap (0.5 + 1 + 0.5 over 2 s)", (st["units"], st["rate"],
                                                                                    st["per_second"]), (2.0, 1.0, [1.0, 1.0]))
    expect("... and a W that starts after a is not covered", unit_stats([1.2], [1.4], 1.0, 3.0)["covered"], False)
    # faults
    rows = [(100.0 + i, {7: ("svc.exe", 60, 30, 0), 11: ("reader.exe", 2, 1, 0)}) for i in range(70)]
    f = faults_window(rows, 105.0, 165.0, {"arm": 11, "w": 42})
    expect("faults: 60 seconds, the arm's own 1/s, an absent W 0, svc.exe on top", (f["seconds"], f["hard_per_s"],
           f["top_hard"][0][:3]), (60, {"arm": 1.0, "w": 0.0}, ["svc.exe", 7, 30.0]))
    # window state and the memory rule
    base = {"arm_stats": {"exit": 0, "covered": True, "pin_ok": True, "rate": 5.26},
            "w_stats": {"exit": 0, "covered": True, "pin_ok": None, "rate": 70.0},
            "counters": {"rows": 60, "pages_in_mean": 1000.0}, "faults": {"seconds": 60, "hard_per_s": {"arm": 25.0, "w": 3.0}}}
    expect("25/s and 1,000 pages/s exactly stand", window_state(base)[0], "OK")
    x = json.loads(json.dumps(base))
    x["faults"]["hard_per_s"]["w"] = 25.01
    expect("a W's own hard faults at 25.01/s VOID the window", window_state(x)[0], "VOID")
    x = json.loads(json.dumps(base))
    x["counters"]["pages_in_mean"] = 1000.5
    expect("the system backstop at 1,000.5 pages/s VOIDs it", window_state(x)[0], "VOID")
    x = json.loads(json.dumps(base))
    x["counters"]["rows"] = 49
    expect("49 counter rows VOID it", window_state(x)[0], "VOID")
    x = json.loads(json.dumps(base))
    x["arm_stats"]["pin_ok"] = False
    expect("an arm whose pinning is not read back VOIDs it", window_state(x)[0], "VOID")
    x = json.loads(json.dumps(base))
    x["arm_stats"]["exit"] = 1
    expect("a process exiting non-zero FAILs it", window_state(x)[0], "FAILED")
    # calibration
    expect("calibration: N = 1 fails a run, N = 2 passes both", calibrate({1: [7.0, 6.5], 2: [8.0, 8.1], 4: [12, 12]}), 2)
    expect("calibration: a run that is not OK does not pass", calibrate({1: [7.0, None], 2: [6.58, 6.6]}), 2)
    expect("calibration: none passes -> None (the sitting takes 8)", calibrate({8: [6.0, 7.0]}), None)

    # the rules on synthetic matrices
    def matrix(wr, ar):
        """wr: {(arm, w): (pass-1 W rate, pass-2 W rate)} with arm None alone; ar: {(arm, w): (tok/s, tok/s)}."""
        recs = []
        for p in (1, 2):
            for arm, w in MATRIX:
                r = {"block": "matrix", "pass": p, "arm": arm, "w": w, "state": "OK"}
                if w:
                    r["w_stats"] = {"rate": wr[(arm, w)][p - 1]}
                if arm:
                    r["arm_stats"] = {"rate": ar.get((arm, w), (5.26, 5.26))[p - 1]}
                recs.append(r)
        return index(recs)

    def wr_of(ks):
        """ks: {(arm, w): K}; every W alone at 100 in both passes."""
        out = {(None, w): (100.0, 100.0) for w in WS}
        for (arm, w), k in ks.items():
            out[(arm, w)] = (100.0 * k, 100.0 * k)
        return out

    good = {("NPU", w): 0.97 for w in WS}
    good.update({("CPU", "W1"): 0.80, ("CPU", "W2"): 0.85, ("CPU", "Wcpu"): 0.70,
                 ("DML", "W1"): 0.60, ("DML", "W2"): 0.70, ("DML", "Wcpu"): 0.97})
    idx = matrix(wr_of(good), {})
    expect("frees the GPU on W1 and W2 (0.97 vs 0.80, 0.85)", (frees(idx, "W1", "CPU")[0], frees(idx, "W2", "CPU")[0]),
           (True, True))
    expect("frees the CPU fails: DML leaves it as free (0.97 vs 0.97)", frees(idx, "Wcpu", "DML")[0], False)
    expect("joint PASS: the NPU >= 0.90 everywhere, CPU on Wcpu 0.70, DML on W1/W2 < 0.90", joint(idx)[0], True)
    t = dict(good)
    t[("NPU", "W1")], t[("CPU", "W1")] = 0.99, 0.90
    expect("exactly 1.10x (0.99 vs 0.90) counts", frees(matrix(wr_of(t), {}), "W1", "CPU")[0], True)
    t[("NPU", "W1")] = 0.989
    expect("just under 1.10x (0.989 vs 0.90) fails", frees(matrix(wr_of(t), {}), "W1", "CPU")[0], False)
    t = dict(good)
    t[("CPU", "W1")] = 0.95
    expect("the CPU at 0.95 on W1 but not holding (4.9 tok/s in pass 2): the NPU frees the GPU on W1",
           frees(matrix(wr_of(t), {("CPU", "W1"): (5.3, 4.9)}), "W1", "CPU")[0], True)
    expect("... and holding, it does not (0.97 < 1.10 x 0.95)", frees(matrix(wr_of(t), {}), "W1", "CPU")[0], False)
    t = dict(good)
    t[("NPU", "W2")] = 0.89
    expect("K_W2(NPU) 0.89 < 0.90 fails the GPU on W2 and the joint", (frees(matrix(wr_of(t), {}), "W2", "CPU")[0],
                                                                       joint(matrix(wr_of(t), {}))[0]), (False, False))
    expect("the NPU not holding with Wcpu (4.99) fails the joint",
           joint(matrix(wr_of(good), {("NPU", "Wcpu"): (5.26, 4.99)}))[0], False)
    wr = wr_of(good)
    wr[("NPU", "W1")] = (97.0, 91.0)                    # K 0.97 and 0.91: differ by 0.06
    kv = k_value(matrix(wr, {}), "NPU", "W1")
    expect("K passes 0.97 / 0.91 differ by 0.06 > 0.05: INCOMPLETE", kv["k"], None)
    expect("... so the GPU on W1 is undecided", frees(matrix(wr, {}), "W1", "CPU")[0], None)
    wr = wr_of(good)
    wr[("NPU", "W1")] = (97.0, 92.0)
    expect("K passes 0.97 / 0.92 differ by 0.05: the mean", round(k_value(matrix(wr, {}), "NPU", "W1")["k"], 4), 0.945)
    wr = wr_of(good)
    wr[(None, "W2")] = (100.0, 88.0)                     # 12.8% apart
    expect("W2 alone 100 / 88 (0.128 apart) leaves every K on W2 INCOMPLETE",
           [k_value(matrix(wr, {}), a, "W2")["k"] for a in ARMS], [None, None, None])
    wr = wr_of(good)
    wr[(None, "W2")] = (100.0, 90.5)
    expect("W2 alone 100 / 90.5 (0.0997 apart) agrees", w_agree(matrix(wr, {}), "W2")["ok"], True)
    idx = matrix(wr_of(good), {})
    idx[(2, "NPU", "W1")] = dict(idx[(2, "NPU", "W1")], state="VOID")
    expect("a VOID co-run window: K INCOMPLETE and holds undecided", (k_value(idx, "NPU", "W1")["k"], holds(idx, "NPU", "W1")),
           (None, None))
    expect("... a valid pass below the floor still decides False", holds(matrix(wr_of(good), {("DML", None): (4.2, 5.3)}),
                                                                          "DML"), False)
    expect("tri-state: and/or", (t_and(True, None), t_and(False, None), t_or(False, None), t_or(True, None)),
           (None, False, None, True))
    expect("predictions: in_range edges", (in_range(0.90, (0.90, 0.96)), in_range(0.90, (None, 0.90)),
                                           in_range(0.95, (0.95, None))), (True, False, True))
    # the verdict end to end, on a synthetic sitting log (every printed path; no chip, no model)
    import contextlib
    import io
    import tempfile

    def full(r, n):
        r = dict(r, n=n, threads=None, why="", counters={"rows": 60, "pages_in_mean": 12.0, "cpu_total_pct": 20.0},
                 faults={"seconds": 60, "hard_per_s": {"arm": 0.0, "w": 0.0}, "top_hard": []}, gpu={"gpu780_all": 30.0})
        if "arm_stats" in r:
            r["arm_stats"] = dict(r["arm_stats"], work_ms_median=75.0)
        return r
    recs = [full(r, i) for i, r in enumerate(matrix(wr_of(good), {}).values())]
    recs += [full({"block": "cal", "pass": k, "arm": "CPU", "w": None, "state": "OK",
                   "arm_stats": {"rate": 7.0}}, 90 + k) for k in (1, 2)]
    for p in (1, 2):
        recs += [full({"block": "worst", "pass": p, "arm": a, "w": w, "state": "OK",
                       **({"w_stats": {"rate": 100.0 if a is None else 55.0}} if w else {}),
                       **({"arm_stats": {"rate": 5.26}} if a else {})}, 100 + p) for a, w in WORST]
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "suite.log"
        log.write_text("\n".join(["PINS_JSON " + json.dumps({"sha256": {}, "mismatch": []}),
                                  "CALIBRATION_JSON " + json.dumps({"threads": 1, "passed": True}),
                                  *("WINDOW_JSON " + json.dumps(r) for r in recs)]) + "\n", encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = verdict(log)
        text = buf.getvalue()
    expect("the verdict on a synthetic sitting: COMPLETE, the GPU freed, the CPU not, joint PASS",
           (rc, "VERDICT COMPLETE" in text, "frees the GPU (unqualified: W1 and W2): PASS" in text,
            "frees the CPU on Wcpu (against the DirectML arm): FAIL" in text, "frees both at once (joint): PASS" in text,
            "K_Wcpu(CPU-c) = 0.550" in text), (0, True, True, True, True, True))
    # hygiene
    set_neutral(["foo", "bar-cli"])
    expect("neutralize: an image name, a host-load line, a path",
           (neutralize("Foo.exe"), neutralize("HOST_LOAD_TOP 0.03 bar-cli:61164 ws_mb=46"), neutralize("food.exe"),
            neutralize(str(ROOT / "tools" / "x.py")), neutralize(ROOT.as_posix().upper() + "/y")),
           ("<tool>.exe", "HOST_LOAD_TOP 0.03 <tool>:61164 ws_mb=46", "food.exe", "<repo>\\tools\\x.py", "<repo>/y"))
    set_neutral([])
    expect("an empty list replaces no name", neutralize("foo.exe"), "foo.exe")
    _NEUTRAL["loaded"] = False
    print("SELFTEST", "FAIL " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("build", "runner", "dryrun", "dryrun-verdict", "prereg", "loadcheck", "suite",
                                     "verdict", "selftest", "reader", "wload"))
    ap.add_argument("log", nargs="?")
    ap.add_argument("--arm", choices=tuple(ARM_MODEL))
    ap.add_argument("--threads", type=int)
    ap.add_argument("--unpaced", action="store_true")
    ap.add_argument("--kind", choices=WS)
    a = ap.parse_args()
    if a.mode == "build":
        build()
    elif a.mode == "runner":
        runner()
    elif a.mode == "dryrun":
        sys.exit(dryrun())
    elif a.mode == "dryrun-verdict":
        sys.exit(dryrun_verdict(Path(a.log)))
    elif a.mode == "verdict":
        sys.exit(verdict(Path(a.log)))
    elif a.mode == "reader":
        sys.exit(reader(a.arm, a.threads, not a.unpaced))
    elif a.mode == "wload":
        sys.exit(wload(a.kind))
    else:
        sys.exit({"prereg": prereg, "loadcheck": loadcheck, "suite": suite, "selftest": selftest}[a.mode]())


if __name__ == "__main__":
    main()
