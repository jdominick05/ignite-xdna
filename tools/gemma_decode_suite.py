#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM study, Gemma 3 4B, pre-registration (c): decode speed, energy per token and accuracy on the
CPU and DirectML with ONNX Runtime GenAI 0.11.2, and (d)'s mechanical rule for NPU-only decode.

The five models are (c)'s build (tools/gemma_decode.py, its log pins them): C0-H4, C4-H4, D-H4,
C0-H16, D-H16. This file runs them and never edits them; the build's file stays byte-identical to
the one that wrote the recheck log (the build log's writer lacked only the smoke's BOS-first row and
B3's unread-placement check).
  suite     the timing and energy sitting: every arm twice (the second pass mirrored), a 60 s idle
            before each; a reader process per arm and pass decodes greedily from a pinned 128-token
            prompt to 1024 positions (896 generated), and the window is generated tokens 257-896
  accuracy  untimed: every arm teacher-forced through GenAI's decode path, one token per step, over
            the pinned 1024-token text, against transformers' fp32 Gemma 3 with the release's weights
  verdict   the mechanical verdict over both logs, with (d)'s rules

    python tools/gemma_decode_suite.py prereg <build log> <recheck log>  # the pre-registration and pins
    python tools/gemma_decode_suite.py suite                         # the sitting (resnet_env17)
    python tools/gemma_decode_suite.py accuracy                      # the accuracy pass (not timed)
    python tools/gemma_decode_suite.py verdict <suite log> <accuracy log>
    python tools/gemma_decode_suite.py prereg-rerun                  # sitting 2's rule and its pins
    python tools/gemma_decode_suite.py verdict --rerun <sitting-2 suite log> <sitting 1's accuracy log>
    python tools/gemma_decode_suite.py posthoc-gpu <suite log> <counter dir> <stamp>   # POST HOC re-read
    python tools/gemma_decode_suite.py reader --arm ARM --pass N     # spawned by suite
    python tools/gemma_decode_suite.py selftest                      # synthetic rules; no model, no chip
"""
import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import gemma_decode as gd  # noqa: E402

# ---------------------------------------------------------------- the protocol ((A), the gate's decision)

PROMPT_TOKENS = 128          # BOS + the pinned text's first 127 tokens
POSITIONS = 1024             # every sequence ends at 1024 positions: the 1024-key window never binds
GEN_TOKENS = POSITIONS - PROMPT_TOKENS          # 896 generated, greedy, min_length = max_length
WIN_FROM, WIN_TO = 256, 896  # the window runs from generated token 256's completion to 896's: 640 tokens
WIN_TOKENS = WIN_TO - WIN_FROM
ARM_ORDER = ("C4-H4", "D-H4", "C0-H4", "D-H16", "C0-H16")
ORDER = [(1, a) for a in ARM_ORDER] + [(2, a) for a in ARM_ORDER[::-1]]
SEQS = {"cpu": 3, "dml": 5}  # CPU: 3 sequences in one session; DirectML: 5 fresh sessions, one each
CHECK_TOKENS = 8             # the reader's untimed check run before READY
FORCE_CHECK_N = 32           # the accuracy pass's forced-vs-appended check on each CPU arm (positions + 1)
# ---------------------------------------------------------------- sitting 2 (the gate's re-run rule, 2026-09-24)
# Sitting 1 (99a7eec) is INCOMPLETE and stays so; sitting 2 alone decides, with sitting 1's accuracy log
# (untimed, deterministic). Each sitting-2 arm's greedy tokens must equal sitting 1's, or the arm is VOID.
TOKENS_SHA_S1 = {            # from gemma_decode_suite_desktop2_20260924.log: every sequence, both passes
    "C4-H4": "c78d37dd4ef54142973d8add8104686158f8bacf708ba89c9c729a9ca1c55467",
    "C0-H4": "43eb1ccaa3d0d6a8ce0a28efdff5e2cb6e1e5a8ef4073a04abe1d08bedf6b8d4",
    "D-H4": "43eb1ccaa3d0d6a8ce0a28efdff5e2cb6e1e5a8ef4073a04abe1d08bedf6b8d4",
    "C0-H16": "38bf3d11228efcdb88840169a358cdff04dd3850cd7d69b036dc0a76b1cc2e0b",
    "D-H16": "38bf3d11228efcdb88840169a358cdff04dd3850cd7d69b036dc0a76b1cc2e0b",
}
S1_LOGS_LF = {               # sha256 of sitting 1's logs with CRLF read as LF (the committed blobs, 99a7eec)
    "gemma_decode_suite_desktop2_20260924.log": "19af6e168d9d164730b490ebabdfd140e56402986fb6c988ac3422da9dc4c7d4",
    "gemma_decode_accuracy_desktop2_20260924.log": "7a579b84bdb022e77d4a267d0b25690cfb99eb2594bda4071528c7000fe2d5dd",
}
GPU_780M_ID = (0x1002, 0x15BF)   # the Radeon 780M's DXGI vendor and device id; its LUIDs are the GPU witness
LUIDS_780M = None                # set by suite() from DXGI at the start (DXGI lists the 780M twice here)
B3_CPU_ALLOWED = ("/model/attn_mask_reformat/attn_mask_subgraph/Gather",        # B3': GQA's total_sequence_length
                  "/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast")   # producers, the only CPU nodes
MEM_BIG_GB = 4.0             # a process this large at the sitting's start refuses it: a builder peaked at 19-21 GB
                             # and one arm's session holds 2-6 GB; the desktop's largest was 0.95 GB (2026-09-24)
IDLE_S = 60
SETTLE_S = 10.0
TP_LEAD_S = 2.5              # typeperf starts after READY (its GPU instances are listed at start)
TP_MAX_S = 3600              # typeperf's sample cap; it is stopped when the reader exits
READY_TIMEOUT_S = 600
RUN_TIMEOUT_S = 3000
MIN_ROWS = 50                # an idle, or an arm-pass's windows together, below this voids the arm-pass
PASS_AGREE = 0.10            # tok/s and J/token must agree within 10% between the passes
IDLE_SHIFT_W = 2.0
PIN = True                   # the build log's AFFINITY_JSON: GenAI 0.11.2 takes the affinities (CPUs 2-14)
PIN_CPUS = gd.PIN_CPUS       # 0, 2, ..., 14: the caller on 0, the 7 pool threads on the rest

# ---------------------------------------------------------------- pins (filled from the build log)

TEXT_PIN = {"repo": "Salesforce/wikitext", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "file": "wikitext-2-raw-v1/test-00000-of-00001.parquet", "size": 732610,
            "sha256": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"}
IDS_SHA = "c87fda3e20a29a5fd9af2104661438ff63a3c29cd496cb4d95a9000820dfb15e"  # the 1024 ids, int32 little-endian
MODEL_SHA = {                # {arm: {file: sha256}} from the build log's MODEL_SHA_JSON (the recheck's is equal)
    "C0-H4": {
        "added_tokens.json": "f8730690481ac3d7ef2249793e7c876ca8d32c14821fe52cfdba1f8521b61c5c",
        "chat_template.jinja": "e0a1e0983c75b0198f1045cbb20c64506a9ffb30925bd44feba0b91acf2d4f43",
        "genai_config.json": "14102d8ca482efabc04b50dec7ae00ca287187f5e627afba33256fc14281f2bc",
        "model.onnx": "b3f63d47ba1e475d4ca82e3561b883577d50d8a38b7beca2fb3c99715fe776ef",
        "model.onnx.data": "9cf93b883014655b6b85fa5541787f1297f7f451fb5028e08e395e4606a688c1",
        "special_tokens_map.json": "88d16a8f41b25d29d30b7865c65404fa9527bbc9c5a652f68a1c25c4b0ae0fd1",
        "tokenizer.json": "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
        "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        "tokenizer_config.json": "6eac11d7a3e6c40d7e0f19195baf61435c1350c9e5da4fea4c097779c65b2fb6",
    },
    "C4-H4": {
        "added_tokens.json": "f8730690481ac3d7ef2249793e7c876ca8d32c14821fe52cfdba1f8521b61c5c",
        "chat_template.jinja": "e0a1e0983c75b0198f1045cbb20c64506a9ffb30925bd44feba0b91acf2d4f43",
        "genai_config.json": "14102d8ca482efabc04b50dec7ae00ca287187f5e627afba33256fc14281f2bc",
        "model.onnx": "cc93a59575e65ad6b6f4b9e6b47991d4d36aa99fc27c6c948684e1eb4bfed4d4",
        "model.onnx.data": "9cf93b883014655b6b85fa5541787f1297f7f451fb5028e08e395e4606a688c1",
        "special_tokens_map.json": "88d16a8f41b25d29d30b7865c65404fa9527bbc9c5a652f68a1c25c4b0ae0fd1",
        "tokenizer.json": "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
        "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        "tokenizer_config.json": "6eac11d7a3e6c40d7e0f19195baf61435c1350c9e5da4fea4c097779c65b2fb6",
    },
    "D-H4": {
        "added_tokens.json": "f8730690481ac3d7ef2249793e7c876ca8d32c14821fe52cfdba1f8521b61c5c",
        "chat_template.jinja": "e0a1e0983c75b0198f1045cbb20c64506a9ffb30925bd44feba0b91acf2d4f43",
        "genai_config.json": "3ffb436f47461b4fe81d23616d97aa69a1ed24cf0c685ee4ab6c20b5d002583d",
        "model.onnx": "09b9a2d0ed34033113f11f471aa81b3085d4e72b5d410a24ecc52f77e2de76e3",
        "model.onnx.data": "46b5050670f6bcae88233fccb35f5e761436429144e8d28b38e1314cab2a5571",
        "special_tokens_map.json": "88d16a8f41b25d29d30b7865c65404fa9527bbc9c5a652f68a1c25c4b0ae0fd1",
        "tokenizer.json": "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
        "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        "tokenizer_config.json": "6eac11d7a3e6c40d7e0f19195baf61435c1350c9e5da4fea4c097779c65b2fb6",
    },
    "C0-H16": {
        "added_tokens.json": "f8730690481ac3d7ef2249793e7c876ca8d32c14821fe52cfdba1f8521b61c5c",
        "chat_template.jinja": "e0a1e0983c75b0198f1045cbb20c64506a9ffb30925bd44feba0b91acf2d4f43",
        "genai_config.json": "14102d8ca482efabc04b50dec7ae00ca287187f5e627afba33256fc14281f2bc",
        "model.onnx": "33c47ffa21a77bda96870e97ad6ce6df9213fca93c1e7e624091d3d2ac4e019c",
        "model.onnx.data": "11edd445f01760ab1d44d8197fd75e40ae805b968522848d101373c804fb8380",
        "special_tokens_map.json": "88d16a8f41b25d29d30b7865c65404fa9527bbc9c5a652f68a1c25c4b0ae0fd1",
        "tokenizer.json": "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
        "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        "tokenizer_config.json": "6eac11d7a3e6c40d7e0f19195baf61435c1350c9e5da4fea4c097779c65b2fb6",
    },
    "D-H16": {
        "added_tokens.json": "f8730690481ac3d7ef2249793e7c876ca8d32c14821fe52cfdba1f8521b61c5c",
        "chat_template.jinja": "e0a1e0983c75b0198f1045cbb20c64506a9ffb30925bd44feba0b91acf2d4f43",
        "genai_config.json": "3ffb436f47461b4fe81d23616d97aa69a1ed24cf0c685ee4ab6c20b5d002583d",
        "model.onnx": "422b5501095b2eafa777451e0c4ef7f7da378e5bd51ec1013f81b0d0dfe9e4fb",
        "model.onnx.data": "5e06ff4b9439b0b0d0e9302841d4796bffb876ab2693e6ebff7684e78538cd40",
        "special_tokens_map.json": "88d16a8f41b25d29d30b7865c65404fa9527bbc9c5a652f68a1c25c4b0ae0fd1",
        "tokenizer.json": "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
        "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        "tokenizer_config.json": "6eac11d7a3e6c40d7e0f19195baf61435c1350c9e5da4fea4c097779c65b2fb6",
    },
}

# ---------------------------------------------------------------- (b)'s results and (d)'s constants

JGB_NPU = 0.328              # R-npu, J/GB, MEASURED (1e546f0)
NPU_GBPS = 47.62             # the NPU's read rate, GB/s, MEASURED (stage 1)
LABEL_B = {"GPU": "INSIDE", "NPU": "INSIDE"}     # (b)'s verdict, 1e546f0
GB_TOKEN = {"H4": (1_804_861_440 + 377_487_360) / 1e9, "H16": (1_804_861_440 + 1_342_177_280) / 1e9}
HEAD_OF = {a: ("H16" if gd.ARMS[a][2] else "H4") for a in gd.ARMS}
KL_FLOOR = 1e-6
TIE_NUM, TIE_DEN = 11, 10    # the study's 1.10 line; exactly 1.10x is OPEN (integer-safe comparisons)
FLOOR_TOKPS = 5.0            # the user's floor, reported beside every tok/s

PRED_TOKPS = {"D-H4": (20, 30), "C4-H4": (12, 22), "C0-H4": (6, 15), "D-H16": (14, 22), "C0-H16": (5, 12)}
PRED_JTOK = {"C0-H4": (2.5, 4.5), "C4-H4": (2.5, 4.5), "C0-H16": (2.5, 7.0), "D-H4": (1.6, 3.0), "D-H16": (1.6, 3.0)}

CPU = r"\Processor(_Total)\% Processor Time"
GPU = r"\GPU Engine(*)\Utilization Percentage"
AVAIL = r"\Memory\Available MBytes"
PAGES_IN = r"\Memory\Pages Input/sec"
# The memory witness (the gate's addition): a sequence's window whose mean Pages Input/sec exceeds
# PAGE_MAX is VOID, and with it the arm-pass. 100 pages/s is 0.4 MB/s; a decode whose weights paged
# would re-read at least 1% of a token's >= 1.8 GB per token, 18 MB = ~4,400 pages, i.e. >= 22,000
# pages/s at the 5 tok/s floor, while a desktop's background hard faults run in the tens per second.
PAGE_MAX = 100.0
TMP = ROOT / "scratch" / "llm" / "decode"


def sha256(path: Path) -> str:
    return gd.sha256(path)


def say(tag: str, obj) -> None:
    print(f"{tag} " + json.dumps(obj), flush=True)


# ---------------------------------------------------------------- the text

def quiet_tokenizer():
    """The text-only copy's tokenizer, loaded with transformers' logging at error: 4.57.6's "incorrect
    regex pattern" warning (logger.warning in tokenization_utils_base) quotes the local path, which
    sitting 1's logs carried; the prereg's TOKENIZER_CHECK_JSON shows the warning changes no id."""
    from transformers import AutoTokenizer
    from transformers.utils import logging as tlog
    v = tlog.get_verbosity()
    tlog.set_verbosity_error()
    try:
        return AutoTokenizer.from_pretrained(str(gd.TEXT))
    finally:
        tlog.set_verbosity(v)


def text_ids() -> list:
    """BOS + the first 1023 tokens of wikitext-2-raw-v1's test split, joined as the HF perplexity
    guide joins it, by the text-only copy's tokenizer (Gemma 3's)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = Path(hf_hub_download(TEXT_PIN["repo"], TEXT_PIN["file"], revision=TEXT_PIN["revision"], repo_type="dataset"))
    if p.stat().st_size != TEXT_PIN["size"] or sha256(p) != TEXT_PIN["sha256"]:
        sys.exit("the text differs from its pin")
    text = "\n\n".join(pq.read_table(p).column("text").to_pylist())
    tok = quiet_tokenizer()
    ids = [2] + tok(text[:20000], add_special_tokens=False)["input_ids"][:POSITIONS - 1]
    assert len(ids) == POSITIONS
    return ids


def ids_sha(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def pinned_ids() -> list:
    ids = text_ids()
    if IDS_SHA is not None and ids_sha(ids) != IDS_SHA:
        sys.exit(f"the token ids differ from their pin ({ids_sha(ids)} != {IDS_SHA})")
    return ids


# ---------------------------------------------------------------- GenAI sessions

def og_model(arm: str, pinned: bool):
    import onnxruntime_genai as og
    cfg = og.Config(str(gd.ONNX / arm))
    if pinned:
        cfg.overlay(json.dumps({"model": {"decoder": {"session_options": {
            "intra_op_num_threads": len(PIN_CPUS), "inter_op_num_threads": 1,
            "config_entries": {"session.intra_op_thread_affinities": ";".join(str(c + 1) for c in PIN_CPUS[1:])}}}}}))
    return og.Model(cfg)


def decode(model, prompt, gen_tokens: int):
    """Greedy decode to len(prompt) + gen_tokens positions (min_length = max_length); each generated
    token's completion on perf_counter_ns, with the wall clock mapped from one common origin."""
    import onnxruntime_genai as og
    params = og.GeneratorParams(model)
    n = len(prompt) + gen_tokens
    params.set_search_options(do_sample=False, max_length=n, min_length=n)
    gen = og.Generator(model, params)
    wall0, pc0 = time.time(), time.perf_counter_ns()
    gen.append_tokens(np.asarray(prompt, dtype=np.int32))
    t, out = [], []
    while len(out) < gen_tokens:
        gen.generate_next_token()
        out.append(int(gen.get_next_tokens()[0]))
        t.append(time.perf_counter_ns())
        if gen.is_done() and len(out) < gen_tokens:
            break
    del gen, params
    return out, t, wall0, pc0


def reader(arm: str, pas: int) -> int:
    """One arm-pass: load, a check run, READY, then on "go" the sequences; SEQ_JSON per sequence, and
    SESSION_RELEASED (available RAM) after each session is released."""
    import gc
    import measure_noise as mn
    import psutil
    ep = gd.ARMS[arm][0]
    pinned = bool(PIN) and ep == "cpu"
    ids = pinned_ids()
    prompt = ids[:PROMPT_TOKENS]
    if pinned:
        mn.pin_calling_thread(PIN_CPUS[0])
    before = mn.thread_ids()
    t0 = time.time()
    model = og_model(arm, pinned)
    load_s = time.time() - t0
    check, _, _, _ = decode(model, prompt, CHECK_TOKENS)
    placement = mn.thread_placement(mn.thread_ids() - before)
    single = sorted(p["cpus"][0] for p in placement if len(p["cpus"]) == 1)
    pin_ok = (not pinned) or all(c in single for c in PIN_CPUS[1:])
    say("READY", {"pid": os.getpid(), "arm": arm, "pass": pas, "pinned": pinned, "pin_ok": pin_ok,
                  "single_cpu_threads": single, "load_s": round(load_s, 2), "check_tokens": check,
                  "available_gb": round(psutil.virtual_memory().available / 1e9, 2),
                  "onnxruntime_dll": gd.loaded_dll("onnxruntime.dll")})
    if sys.stdin.readline().strip() != "go":
        return 2
    seqs = []

    def release(after_seq: int):
        nonlocal model
        del model
        gc.collect()
        say("SESSION_RELEASED", {"after_seq": after_seq, "available_gb": round(psutil.virtual_memory().available / 1e9, 2)})

    for s in range(SEQS[ep]):
        if ep == "dml" and s > 0:                       # a fresh session per sequence, the last one released first
            release(s - 1)
            t0 = time.time()
            model = og_model(arm, pinned)
            load_s = time.time() - t0
        out, t, wall0, pc0 = decode(model, prompt, GEN_TOKENS)
        rec = {"seq": s, "generated": len(out), "load_s": round(load_s, 2),
               "tokens_sha": ids_sha(out), "first_tokens": out[:16]}
        if len(out) == GEN_TOKENS:
            a, b = t[WIN_FROM - 1], t[WIN_TO - 1]
            rec.update(win_s=(b - a) / 1e9, tokps=WIN_TOKENS / ((b - a) / 1e9),
                       win_wall=[wall0 + (a - pc0) / 1e9, wall0 + (b - pc0) / 1e9],
                       step_ms_median=statistics.median(np.diff(t[WIN_FROM - 1:WIN_TO]) / 1e6))
        seqs.append(rec)
        say("SEQ_JSON", rec)
    release(len(seqs) - 1)
    say("READER_JSON", {"arm": arm, "pass": pas, "seqs": len(seqs)})
    return 0


# ---------------------------------------------------------------- the coordinator

class Reader:
    """One reader process; its command is logged relative to the repository, the interpreter by env."""

    def __init__(self, arm: str, pas: int):
        self.lines, self.ready = [], threading.Event()
        args = ["tools/gemma_decode_suite.py", "reader", "--arm", arm, "--pass", str(pas)]
        self.shown = [f"python[{os.environ.get('CONDA_DEFAULT_ENV', '?')}]", *args]
        self.proc = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
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

    def json_lines(self, tag: str) -> list:
        return [json.loads(s.split(" ", 1)[1]) for s in self.lines if s.startswith(tag + " ")]


def typeperf(samples: int, path: Path) -> subprocess.Popen:
    from power_probe import CORES, PKG
    return subprocess.Popen(["typeperf", PKG, *CORES, CPU, AVAIL, PAGES_IN, GPU, "-si", "1", "-sc", str(samples), "-y",
                             "-o", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop(proc: subprocess.Popen, timeout: float = 60) -> None:
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        proc.wait(timeout=15)


# ---------------------------------------------------------------- the page-in attribution witness (re-run (b))

def proc_counters() -> dict:
    """{pid: (image name, page faults, hard faults, IO read bytes)}, cumulative, for every process, from
    one NtQuerySystemInformation(SystemProcessInformation) call: the kernel counts behind
    \\Process(*)\\Page Faults/sec and \\Process(*)\\IO Read Bytes/sec. typeperf fixes a wildcard's
    instances when it starts, so it cannot see a process launched inside a window; this can. About
    11 ms of one core per call on this desktop (a psutil sweep took 2.5 s)."""
    import ctypes
    from ctypes import wintypes

    class USTR(ctypes.Structure):
        _fields_ = [("Length", ctypes.c_ushort), ("MaximumLength", ctypes.c_ushort), ("Buffer", ctypes.c_void_p)]

    class SPI(ctypes.Structure):                   # SYSTEM_PROCESS_INFORMATION (x64), up to its IO counters
        _fields_ = [("NextEntryOffset", ctypes.c_ulong), ("NumberOfThreads", ctypes.c_ulong),
                    ("WorkingSetPrivateSize", ctypes.c_longlong), ("HardFaultCount", ctypes.c_ulong),
                    ("NumberOfThreadsHighWatermark", ctypes.c_ulong), ("CycleTime", ctypes.c_ulonglong),
                    ("CreateTime", ctypes.c_longlong), ("UserTime", ctypes.c_longlong), ("KernelTime", ctypes.c_longlong),
                    ("ImageName", USTR), ("BasePriority", ctypes.c_long), ("UniqueProcessId", ctypes.c_void_p),
                    ("InheritedFromUniqueProcessId", ctypes.c_void_p), ("HandleCount", ctypes.c_ulong),
                    ("SessionId", ctypes.c_ulong), ("UniqueProcessKey", ctypes.c_void_p),
                    ("PeakVirtualSize", ctypes.c_size_t), ("VirtualSize", ctypes.c_size_t),
                    ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t), ("PrivatePageCount", ctypes.c_size_t),
                    ("ReadOperationCount", ctypes.c_longlong), ("WriteOperationCount", ctypes.c_longlong),
                    ("OtherOperationCount", ctypes.c_longlong), ("ReadTransferCount", ctypes.c_longlong),
                    ("WriteTransferCount", ctypes.c_longlong), ("OtherTransferCount", ctypes.c_longlong)]

    nt = ctypes.WinDLL("ntdll")
    nt.NtQuerySystemInformation.restype = ctypes.c_long
    size = 1 << 20
    while True:
        buf, need = ctypes.create_string_buffer(size), wintypes.ULONG(0)
        st = nt.NtQuerySystemInformation(5, buf, size, ctypes.byref(need)) & 0xFFFFFFFF
        if st == 0:
            break
        if st != 0xC0000004:                         # STATUS_INFO_LENGTH_MISMATCH: grow and retry
            raise OSError(f"NtQuerySystemInformation 0x{st:08x}")
        size = max(need.value, size) * 2
    out, off, base = {}, 0, ctypes.addressof(buf)
    while True:
        e = SPI.from_address(base + off)
        name = ctypes.wstring_at(e.ImageName.Buffer, e.ImageName.Length // 2) if e.ImageName.Buffer else "Idle"
        out[e.UniqueProcessId or 0] = (name, e.PageFaultCount, e.HardFaultCount, e.ReadTransferCount)
        if not e.NextEntryOffset:
            return out
        off += e.NextEntryOffset


class FaultSampler:
    """Every process's page faults, hard faults and IO read bytes once a second (proc_counters), kept as
    per-second deltas; run in every idle and every arm-pass alike, so its own cost cancels in dP.
    Report-only: no rule reads it."""

    def __init__(self, counters=proc_counters):
        self.counters, self.rows, self.halt = counters, [], threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        prev = self.counters()
        while not self.halt.wait(1.0 - time.time() % 1.0):
            now, cur = time.time(), self.counters()
            d = {}
            for pid, (name, pf, hf, rb) in cur.items():
                p = prev.get(pid)
                if p is None or p[0] != name:            # a process new since the last second: all its counts
                    p = (name, 0, 0, 0)
                dd = (pf - p[1], hf - p[2], rb - p[3])
                if any(dd):
                    d[pid] = (name, *dd)
            self.rows.append((round(now, 3), d))
            prev = cur

    def stop(self):
        self.halt.set()
        self.thread.join(5)

    def window(self, a: float, b: float, top: int = 5) -> dict:
        """The top processes by page faults/s and by IO read bytes/s over the seconds whose whole second
        lies in [a, b] (inside()'s rule), each entry [name, pid, faults/s, hard faults/s, read bytes/s]."""
        agg, n = {}, 0
        for t, d in self.rows:
            if a + 1.0 <= t <= b:
                n += 1
                for pid, (name, pf, hf, rb) in d.items():
                    x = agg.setdefault(pid, [name, 0, 0, 0])
                    x[1], x[2], x[3] = x[1] + pf, x[2] + hf, x[3] + rb
        s = max(n, 1)
        ent = [[nm, pid, round(pf / s, 1), round(hf / s, 1), round(rb / s)] for pid, (nm, pf, hf, rb) in agg.items()]
        return {"seconds": n, "top_page_faults": sorted(ent, key=lambda e: -e[2])[:top],
                "top_io_read": sorted(ent, key=lambda e: -e[4])[:top]}


# the engine type may hold a space ("Compute 0", "Timer 0": the 780M's DirectML engines); sitting 1's
# (\w+) dropped those columns, so its "780M pid" read 0 (the re-run rule's change (a))
LUID_RE = re.compile(r"gpu engine\(pid_(\d+)_luid_(0x[0-9a-f]+_0x[0-9a-f]+)_phys_\d+_eng_\d+_engtype_([^)]+)\)")


def gpu_columns(head: list) -> dict:
    """{index: (pid, luid, engtype)} for every GPU-engine column; the 780M is the LUID with a 3D
    engine, and any other LUID (the NPU lists only Compute) is reported as other."""
    out = {}
    for i, h in enumerate(head):
        m = LUID_RE.search(h.lower())
        if m:
            out[i] = (int(m.group(1)), m.group(2), m.group(3))
    return out


def dxgi_adapters() -> list:
    """DXGI's adapters in EnumAdapters1 order (ORT's DirectML device_id indexes it:
    dml_provider_factory.cc:499-502, rel-1.23.0), each LUID in the GPU Engine counters' spelling."""
    import ctypes
    import uuid
    from ctypes import POINTER, byref, c_void_p, wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("d1", ctypes.c_ulong), ("d2", ctypes.c_ushort), ("d3", ctypes.c_ushort), ("d4", ctypes.c_ubyte * 8)]

    class LUID(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.LONG)]

    class DESC1(ctypes.Structure):
        _fields_ = [("desc", ctypes.c_wchar * 128), ("vendor", ctypes.c_uint), ("device", ctypes.c_uint),
                    ("subsys", ctypes.c_uint), ("rev", ctypes.c_uint), ("vram", ctypes.c_size_t),
                    ("sysmem", ctypes.c_size_t), ("shared", ctypes.c_size_t), ("luid", LUID), ("flags", ctypes.c_uint)]

    def method(obj, idx, *argtypes):
        vt = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
        return ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)(vt[idx])

    u = uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387")          # IID_IDXGIFactory1
    iid = GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))
    factory, out = c_void_p(), []
    ctypes.windll.dxgi.CreateDXGIFactory1(byref(iid), byref(factory))
    for i in range(16):
        ad = c_void_p()
        try:
            method(factory, 12, ctypes.c_uint, POINTER(c_void_p))(factory, i, byref(ad))     # EnumAdapters1
        except OSError:                                                                    # DXGI_ERROR_NOT_FOUND
            break
        d = DESC1()
        method(ad, 10, POINTER(DESC1))(ad, byref(d))                                       # GetDesc1
        out.append({"index": i, "description": d.desc, "vendor": d.vendor, "device": d.device, "flags": d.flags,
                    "luid": f"0x{d.luid.high & 0xffffffff:08x}_0x{d.luid.low:08x}"})
    return out


def read_csv(path: Path, pid=None, luids780=None) -> dict:
    """typeperf rows as [epoch s, package mW, sum of Core0-7 mW, CPU %, 780M busy (all), 780M busy
    (pid), other adapters' busy (all), available MB, pages input/s], and the LUIDs seen. The 780M's
    columns are those of luids780 (DXGI's, at the sitting's start); without it, any LUID with a 3D
    engine. A row with a blank package or core field is skipped; a blank GPU or memory field counts 0."""
    from power_probe import CORES, PKG
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    head = [h.lower() for h in rows[0]]
    col = lambda p: next(i for i, h in enumerate(head) if h.endswith(p.lower().lstrip("\\")))  # noqa: E731
    ipkg, icores, icpu = col(PKG), [col(c) for c in CORES], col(CPU)
    iavail, ipages = col(AVAIL), col(PAGES_IN)
    g = gpu_columns(head)
    gpu_luids = set(luids780) if luids780 else {luid for _, luid, eng in g.values() if eng == "3d"}
    i780 = [i for i, (_, luid, _) in g.items() if luid in gpu_luids]
    ipid = [i for i in i780 if g[i][0] == pid] if pid is not None else []
    iother = [i for i, (_, luid, _) in g.items() if luid not in gpu_luids]

    def num(s: str) -> float:
        try:
            return float(s)
        except ValueError:
            return 0.0
    out = []
    for r in rows[1:]:
        try:
            stamp = datetime.strptime(r[0], "%m/%d/%Y %H:%M:%S.%f").timestamp()
            pkg, cores, cpu = float(r[ipkg]), sum(float(r[i]) for i in icores), float(r[icpu])
        except (ValueError, IndexError):
            continue
        out.append([round(stamp, 3), round(pkg, 1), round(cores, 1), round(cpu, 2),
                    round(sum(num(r[i]) for i in i780 if i < len(r)), 2),
                    round(sum(num(r[i]) for i in ipid if i < len(r)), 2),
                    round(sum(num(r[i]) for i in iother if i < len(r)), 2),
                    round(num(r[iavail]), 0), round(num(r[ipages]), 2)])
    return {"rows": out, "gpu_luids": sorted(gpu_luids),
            "luids_3d_seen": sorted({luid for _, luid, eng in g.values() if eng == "3d"}),
            "other_luids": sorted({g[i][1] for i in iother})}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def idle(tag: str) -> dict:
    path = TMP / f"{tag}_idle.csv"
    fs = FaultSampler().start()                        # run in the idle too, so its cost cancels in dP
    p = typeperf(IDLE_S + 1, path)
    stop(p, IDLE_S + 60)
    fs.stop()
    return read_csv(path, None, LUIDS_780M)


def arm_pass(tag: str, arm: str, pas: int) -> dict:
    r = Reader(arm, pas)
    rec = {"cmd": " ".join(r.shown)}
    tp = fs = None
    try:
        if not r.ready.wait(READY_TIMEOUT_S) or not r.json_lines("READY"):
            raise RuntimeError(f"{arm} not READY")
        rec["ready"] = r.json_lines("READY")[0]
        path = TMP / f"{tag}_window.csv"
        tp = typeperf(TP_MAX_S, path)
        fs = FaultSampler().start()
        time.sleep(TP_LEAD_S)
        r.proc.stdin.write("go\n")
        r.proc.stdin.flush()
        r.proc.wait(timeout=RUN_TIMEOUT_S)
        r.pump.join(timeout=30)
        time.sleep(2.0)                                  # the last window's closing second is sampled
        stop(tp, 0)
        tp = None
        fs.stop()
        rec["counters"] = read_csv(path, rec["ready"]["pid"], LUIDS_780M)
    finally:
        if r.proc.poll() is None:
            r.proc.kill()
        if tp is not None:
            stop(tp, 0)
        if fs is not None:
            fs.stop()
        print(f"READER_CMD {arm} pass {pas} {rec['cmd']}", flush=True)
        for s in r.lines:
            if not s.startswith(("SEQ_JSON ", "READY ", "READER_JSON ")):
                print(f"{arm}| {s}", flush=True)
    rec["seqs"] = r.json_lines("SEQ_JSON")
    rec["releases"] = r.json_lines("SESSION_RELEASED")
    if fs is not None:                                   # the attribution witness, one entry per sequence window
        rec["pagein_attr"] = [{"seq": s["seq"], **fs.window(*s["win_wall"])} for s in rec["seqs"] if "win_wall" in s]
    if r.proc.returncode or not r.json_lines("READER_JSON"):
        rec["failed"] = f"exit {r.proc.returncode}"
    return rec


def attributions(rec: dict) -> list:
    """The attribution witness for each of an arm-pass's windows whose mean pages input/s is over
    PAGE_MAX (the windows arm_stats voids), with that mean."""
    rows = rec.get("counters", {}).get("rows", [])
    attr = {a["seq"]: a for a in rec.get("pagein_attr", [])}
    out = []
    for s in rec.get("seqs", []):
        if "win_wall" not in s:
            continue
        rs = inside(rows, *s["win_wall"])
        m = statistics.fmean(r[8] for r in rs) if rs else 0.0
        if m > PAGE_MAX:
            out.append({"arm": rec["arm"], "pass": rec["pass"], "seq": s["seq"], "pages_in_mean": round(m, 1),
                        **attr.get(s["seq"], {"missing": True})})
    return out


def check_pins() -> dict:
    bad = []
    got = {}
    for arm, files in MODEL_SHA.items():
        for f, want in files.items():
            h = sha256(gd.ONNX / arm / f)
            got[f"{arm}/{f}"] = h
            if h != want:
                bad.append(f"{arm}/{f}")
    if not MODEL_SHA:
        bad.append("no model pins")
    if PIN is None:
        bad.append("PIN not set from the build log")
    return {"sha256": got, "mismatch": bad}


def memory_start() -> dict:
    """The gate's start condition: the available RAM (what \\Memory\\Available MBytes reports) and every
    process of >= 1 GB private; one of >= MEM_BIG_GB refuses the sitting."""
    import psutil
    big = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            gb = p.memory_info().private / 1e9
        except (psutil.Error, AttributeError):
            continue
        if gb >= 1.0 and p.pid != os.getpid():
            big.append({"pid": p.pid, "name": p.info["name"], "private_gb": round(gb, 2)})
    big.sort(key=lambda b: -b["private_gb"])
    return {"available_gb": round(psutil.virtual_memory().available / 1e9, 2), "processes_over_1gb": big,
            "refuse": [b for b in big if b["private_gb"] >= MEM_BIG_GB]}


def suite() -> int:
    from silicon_probe_record import witness
    import llm_prefill_bench as pb
    TMP.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    pb.host_gate()
    m = memory_start()
    say("MEMORY_START_JSON", m)
    if m["refuse"]:
        print(f"MEMORY_REFUSE a process holds >= {MEM_BIG_GB:g} GB private at the start: {m['refuse']}", flush=True)
        return 3
    global LUIDS_780M
    ad = dxgi_adapters()
    LUIDS_780M = sorted(a["luid"] for a in ad if (a["vendor"], a["device"]) == GPU_780M_ID and not a["flags"] & 2)
    say("ADAPTERS_JSON", {"adapters": ad, "luids_780m": LUIDS_780M})
    hw = [a for a in ad if not a["flags"] & 2]                     # DXGI_ADAPTER_FLAG_SOFTWARE
    if not hw or hw[0] is not ad[0] or any((a["vendor"], a["device"]) != GPU_780M_ID for a in hw):
        print(f"ADAPTER_REFUSE a hardware adapter other than the 780M, or adapter 0 not the 780M: {ad}", flush=True)
        return 3
    p = check_pins()
    say("PINS_JSON", p)
    if p["mismatch"]:
        print("PIN_MISMATCH", p["mismatch"], flush=True)
        return 3
    ids = pinned_ids()
    say("TEXT_JSON", {"ids_sha": ids_sha(ids), "prompt": ids[:PROMPT_TOKENS]})
    print("PROTOCOL", json.dumps({"prompt": PROMPT_TOKENS, "positions": POSITIONS, "generated": GEN_TOKENS,
                                  "window": [WIN_FROM + 1, WIN_TO], "seqs": SEQS, "order": ORDER, "pin": PIN}), flush=True)
    say("COUNTER_FILES_JSON", {"dir": TMP.relative_to(ROOT).as_posix(), "stamp": stamp,
                               "names": "<stamp>_<n>_<arm>_{idle,window}.csv"})
    for n, (pas, arm) in enumerate(ORDER):
        print(f"\nARM_BEGIN {n} pass {pas} {arm} {utc()}", flush=True)
        witness()
        time.sleep(SETTLE_S)
        tag = f"{stamp}_{n:02d}_{arm}"
        import psutil
        rec = {"n": n, "pass": pas, "arm": arm, "available_gb_before": round(psutil.virtual_memory().available / 1e9, 2),
               "idle_utc": utc(), "idle": idle(tag)}
        try:
            rec.update(arm_pass(tag, arm, pas))
        except Exception as e:                           # recorded; the verdict marks the arm-pass INCOMPLETE
            rec["failed"] = f"{type(e).__name__}: {str(e)[:300]}"
        say("ARM_JSON", rec)
        st = arm_stats(rec)
        print("ARM_SUMMARY", n, pas, arm, " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                     for k, v in st.items() if k != "seq_tokps"), flush=True)
        for a in attributions(rec):
            say("PAGEIN_ATTRIBUTION", a)
    witness()
    return 0


# ---------------------------------------------------------------- the accuracy pass

def reference_hidden(mm, by, st, ids):
    """gemma_decode.reference_logits' model (transformers' fp32 Gemma 3 with the release's weights,
    streamed layer by layer) up to the final norm: [positions, hidden] fp32."""
    import torch
    import transformers.models.gemma3.modeling_gemma3 as mg
    captured, real = {}, mg.Gemma3ForCausalLM.forward

    def forward(self, input_ids=None, **kw):                    # the text model only; the head is applied by the caller
        out = self.model(input_ids=input_ids, use_cache=False)
        captured["h"] = out.last_hidden_state[0].float().numpy().copy()
        return type("Out", (), {"logits": torch.zeros(1, 1, 1)})()

    mg.Gemma3ForCausalLM.forward = forward
    try:
        gd.reference_logits(mm, by, st, ids)
    finally:
        mg.Gemma3ForCausalLM.forward = real
    return captured["h"]


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return z - (z.max() + np.log(np.exp(z - z.max()).sum()))


def arm_heads(mm, by) -> dict:
    """Each arm's head as it reads it, fp32 [V, hidden]: the release's F16 head for H16, the
    builder's int4 head dequantized for H4 (one entry per distinct head)."""
    out = {}
    for arm in gd.ARMS:
        if HEAD_OF[arm] == "H16":
            continue
        g = gd.Graph(gd.ONNX / arm)
        hn = next(n for n in g.m.graph.node if n.name.startswith("/lm_head/MatMul"))
        b, s = g.read(hn.input[1]), g.read(hn.input[2]).astype(np.float32).ravel()
        out[arm] = gd.dequant(gd.unpack_ort(b), s)
    return out


def forced_logits(gen, ids, n: int, on_row=None) -> list:
    """GenAI's decode path, one token per step, teacher-forced: the logits after position i are read
    (get_logits), then replaced by a one-hot on id i+1, and a greedy generate_next_token appends it.
    GenAI 0.11.2 refuses a second append_tokens on DirectML ("Continuous decoding is not supported"),
    so every arm is forced this way. Returns the generated sequence."""
    one = None
    gen.append_tokens(np.asarray(ids[:1], dtype=np.int32))
    for i in range(n - 1):
        z = np.asarray(gen.get_logits(), dtype=np.float32)
        if on_row is not None:
            on_row(i, z.reshape(-1))
        if one is None:
            one = np.full(z.shape, -1e4, dtype=np.float32)
        one.reshape(-1)[ids[i + 1]] = 0.0
        gen.set_logits(one)
        gen.generate_next_token()
        one.reshape(-1)[ids[i + 1]] = -1e4
    return [int(t) for t in gen.get_sequence(0)]


def force_check(arm: str, ids, n: int = FORCE_CHECK_N) -> dict:
    """A CPU arm's forced logits against append_tokens one id at a time (both the decode path; the
    CPU allows continuous decoding): they must be equal."""
    import onnxruntime_genai as og
    model = og_model(arm, False)
    rows = {"forced": [], "appended": []}
    for how in rows:
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=len(ids), min_length=len(ids))
        gen = og.Generator(model, params)
        if how == "forced":
            forced_logits(gen, ids, n, lambda i, z: rows["forced"].append(z.copy()))
        else:
            gen.append_tokens(np.asarray(ids[:1], dtype=np.int32))
            for i in range(n - 1):
                rows["appended"].append(np.asarray(gen.get_logits(), dtype=np.float32).reshape(-1).copy())
                gen.append_tokens(np.asarray([ids[i + 1]], dtype=np.int32))
        del gen, params
    del model
    f, a = np.stack(rows["forced"]), np.stack(rows["appended"])
    return {"arm": arm, "positions": n - 1, "max_abs_diff": float(np.abs(f - a).max()),
            "argmax_equal": bool((f.argmax(1) == a.argmax(1)).all())}


def teacher_forced(arm: str, ids, refs: dict, pinned: bool) -> dict:
    """The forced decode path (forced_logits): the logits after position i against each reference
    row i (KL, top-1, the next token's log-probability)."""
    import onnxruntime_genai as og
    model = og_model(arm, pinned)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(ids), min_length=len(ids))
    gen = og.Generator(model, params)
    kl = {k: [] for k in refs}
    top1, lp, nonfinite = [], [], [0]

    def row(i, z32):
        z = z32.astype(np.float64)
        if not np.isfinite(z).all():
            nonfinite[0] += 1
            z = np.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
        lz = log_softmax(z)
        for k, ref in refs.items():
            la = log_softmax(ref[i])
            kl[k].append(float((np.exp(la) * (la - lz)).sum()))
        top1.append(int(z.argmax() == refs["primary"][i].argmax()))
        lp.append(float(lz[ids[i + 1]]))

    seq = forced_logits(gen, ids, len(ids), row)
    del gen, params, model
    res = {"arm": arm, "positions": len(ids) - 1, "nonfinite_positions": nonfinite[0],
           "forced_sequence_equal": seq == list(ids),
           "top1_agree": float(np.mean(top1)), "ppl": float(np.exp(-np.mean(lp)))}
    for k, v in kl.items():
        v = np.asarray(v)
        res[f"kl_{k}_mean"] = float(v.mean())
        res[f"kl_{k}_p50"] = float(np.median(v))
        res[f"kl_{k}_p99"] = float(np.quantile(v, 0.99))
        res[f"kl_{k}_max"] = float(v.max())
    return res


def accuracy() -> int:
    import torch
    say("MEMORY_START_JSON", memory_start())                      # logged only: the pass is untimed
    ids = pinned_ids()
    say("TEXT_JSON", {"ids_sha": ids_sha(ids)})
    mm, by, st, _ = gd.open_release(gd.release_paths())
    h = reference_hidden(mm, by, st, ids)
    head = torch.from_numpy(gd.gguf_head(mm, by).astype(np.float32))
    ht = torch.from_numpy(h)
    primary = torch.nn.functional.linear(ht, head).numpy()
    ref_lp = [log_softmax(primary[i])[ids[i + 1]] for i in range(len(ids) - 1)]
    say("REFERENCE_JSON", {"positions": len(ids) - 1, "ppl": float(np.exp(-np.mean(ref_lp))),
                           "finite": bool(np.isfinite(primary).all())})
    heads = arm_heads(mm, by)
    for arm in (a for a in gd.ARMS if gd.ARMS[a][0] == "cpu"):
        say("FORCE_CHECK_JSON", force_check(arm, ids))
    for arm in gd.ARMS:
        refs = {"primary": primary}
        if arm in heads:                                   # the secondary: the same model with the arm's head
            refs["secondary"] = torch.nn.functional.linear(ht, torch.from_numpy(heads[arm])).numpy()
        t0 = time.time()
        res = teacher_forced(arm, ids, refs, bool(PIN) and gd.ARMS[arm][0] == "cpu")
        res["seconds"] = round(time.time() - t0, 1)
        if "secondary" not in refs:
            res.update({k.replace("primary", "secondary"): v for k, v in res.items() if k.startswith("kl_primary")})
        say("ACC_JSON", res)
        del refs
    return 0


# ---------------------------------------------------------------- the numbers, from the rows

def inside(rows, a: float, b: float) -> list:
    """Rows whose whole second lies in [a, b] (a row stamped t holds the second that ends at t)."""
    return [r for r in rows if a + 1.0 <= r[0] <= b]


def arm_stats(rec: dict) -> dict:
    if rec.get("failed") or "counters" not in rec:
        return {"ok": False, "why": rec.get("failed") or "no counters"}
    ep = gd.ARMS[rec["arm"]][0]
    seqs = rec.get("seqs", [])
    if len(seqs) != SEQS[ep] or any(s["generated"] != GEN_TOKENS for s in seqs):
        return {"ok": False, "why": f"sequences {len(seqs)}/{SEQS[ep]}, generated "
                                    f"{[s['generated'] for s in seqs]} (want {GEN_TOKENS} each)"}
    if not rec["ready"].get("pin_ok", True):
        return {"ok": False, "why": f"pinning not read back: {rec['ready'].get('single_cpu_threads')}"}
    rows = rec["counters"]["rows"]
    per_seq = [inside(rows, *s["win_wall"]) for s in seqs]
    win = [r for rs in per_seq for r in rs]
    irows = rec["idle"]["rows"]
    if len(irows) < MIN_ROWS or len(win) < MIN_ROWS:
        return {"ok": False, "why": f"rows idle {len(irows)} window {len(win)} < {MIN_ROWS}"}
    f = lambda rs, i: statistics.fmean(r[i] for r in rs)  # noqa: E731
    paging = [f(rs, 8) if rs else 0.0 for rs in per_seq]
    if max(paging) > PAGE_MAX:                           # the memory witness: a window that paged is VOID
        return {"ok": False, "why": f"paging: window pages input/s {[round(p, 1) for p in paging]} > {PAGE_MAX:g}"}
    idle_w, win_w = f(irows, 1) / 1e3, f(win, 1) / 1e3
    tokps_seq = [s["tokps"] for s in seqs]
    agg = WIN_TOKENS * len(seqs) / sum(s["win_s"] for s in seqs)
    dP = win_w - idle_w
    return {"ok": True, "tokps": statistics.median(tokps_seq), "tokps_agg": agg, "seq_tokps": tokps_seq,
            "idle_w": idle_w, "idle_cpu": f(irows, 3), "dP": dP, "dC": (f(win, 2) - f(irows, 2)) / 1e3,
            "jtok": dP / agg, "cpu": f(win, 3), "gpu780_all": f(win, 4), "gpu780_pid": f(win, 5),
            "other_all": f(win, 6), "rows": len(win), "tokens_sha": sorted({s["tokens_sha"] for s in seqs}),
            "pages_in_max_window": max(paging), "pages_in_idle": f(irows, 8),
            "avail_mb_min": min(r[7] for r in win), "avail_gb_before": rec.get("available_gb_before")}


def combine(stats: dict) -> dict:
    """stats: {(pass, arm): arm_stats}. An arm is OK if both passes are, and tok/s and J/token agree
    within PASS_AGREE; its value is the passes' mean."""
    out = {}
    for arm in gd.ARMS:
        p = [stats.get((k, arm), {"ok": False, "why": "missing"}) for k in (1, 2)]
        if not all(x["ok"] for x in p):
            out[arm] = {"state": "INCOMPLETE", "why": "; ".join(f"pass {k + 1}: {x['why']}" for k, x in enumerate(p)
                                                                if not x["ok"])}
            continue
        v = {}
        for key in ("tokps", "jtok"):
            a, b = p[0][key], p[1][key]
            m = (a + b) / 2
            v[key] = m
            v[f"{key}_agree"] = abs(a - b) / m if m > 0 else float("inf")
        ok = v["tokps_agree"] <= PASS_AGREE and v["jtok_agree"] <= PASS_AGREE
        out[arm] = {"state": "OK" if ok else "INCOMPLETE", **v,
                    "why": "" if ok else f"passes disagree (tok/s {v['tokps_agree']:.3f}, J/token {v['jtok_agree']:.3f})"}
    return out


# ---------------------------------------------------------------- (d)

def kl_eff(kl: float) -> float:
    return max(kl, KL_FLOOR)


TIE_EPS = 1e-12              # relative: a tie at exactly 1.10x in decimal inputs is kept a tie


def at_least_as_accurate(kl_a: float, kl_b: float) -> bool:
    """A counts as at least as accurate as B if KL_A <= 1.10 x KL_B (both floored at 1e-6)."""
    return kl_eff(kl_a) * TIE_DEN <= kl_eff(kl_b) * TIE_NUM * (1 + TIE_EPS)


def rivals(arms: dict, head: str, assume: str = "generous") -> tuple:
    """arms: {arm: {"kl": float or None (non-finite), "tokps", "jtok", "head"}}. The NPU on `head` is
    assumed as accurate as the best measured arm on that head (generous) or the worst; the rivals
    are every finite arm at least that accurate."""
    on = [a for a, v in arms.items() if v["head"] == head and v["kl"] is not None]
    if not on:
        return None, []
    ref = (min if assume == "generous" else max)(arms[a]["kl"] for a in on)
    return ref, [a for a, v in arms.items() if v["kl"] is not None and at_least_as_accurate(v["kl"], ref)]


def rule_speed(ceiling: float, rival_tokps: float) -> str:
    """KILL if the NPU's ceiling is below 1.10x the fastest rival; exactly 1.10x is OPEN."""
    return "OPEN" if ceiling * TIE_DEN * (1 + TIE_EPS) >= rival_tokps * TIE_NUM else "KILL"


def rule_energy(e_npu: float, rival_jtok: float) -> str:
    """KILL if 1.10x the NPU's floor exceeds the rivals' lowest J/token; exactly 1.10x is OPEN."""
    return "KILL" if e_npu * TIE_NUM > rival_jtok * TIE_DEN * (1 + TIE_EPS) else "OPEN"


def rule_d(arms: dict) -> list:
    out = []
    for head in ("H4", "H16"):
        ceiling = NPU_GBPS / GB_TOKEN[head]
        e_npu = JGB_NPU * GB_TOKEN[head]
        for assume in ("generous", "worst"):
            ref, rv = rivals(arms, head, assume)
            if ref is None:
                out.append({"head": head, "assume": assume, "speed": "UNDECIDED", "energy": "UNDECIDED", "rivals": []})
                continue
            fast = max(rv, key=lambda a: arms[a]["tokps"])
            frugal = min(rv, key=lambda a: arms[a]["jtok"])
            out.append({"head": head, "assume": assume, "kl_ref": ref, "rivals": rv,
                        "ceiling": ceiling, "fastest": fast, "fastest_tokps": arms[fast]["tokps"],
                        "speed": rule_speed(ceiling, arms[fast]["tokps"]),
                        "e_npu": e_npu, "frugal": frugal, "frugal_jtok": arms[frugal]["jtok"],
                        "energy": rule_energy(e_npu, arms[frugal]["jtok"])})
    return out


def labels() -> list:
    return ["(b): GPU INSIDE and NPU INSIDE (1e546f0): NPU-vs-DirectML and NPU-vs-CPU energy compare, labelled. "
            "A component scaling with each chip's work is inside the package, not shown to be all of it.",
            "Package counters only: completeness is never shown, and DRAM power is outside the package; shared "
            "DRAM pulls ratios toward 1."]


def predictions(res: dict, acc: dict) -> list:
    out = []
    for arm in ARM_ORDER:
        r = res[arm]
        lo, hi = PRED_TOKPS[arm]
        out.append((f"speed {arm}", f"{lo}-{hi} tok/s", "UNSCORED (INCOMPLETE)" if r["state"] != "OK" else
                    f"{'HIT' if lo <= r['tokps'] <= hi else 'MISS'} ({r['tokps']:.2f})"))
    for arm in ARM_ORDER:
        r = res[arm]
        lo, hi = PRED_JTOK[arm]
        out.append((f"energy {arm}", f"{lo}-{hi} J/token", "UNSCORED (INCOMPLETE)" if r["state"] != "OK" else
                    f"{'HIT' if lo <= r['jtok'] <= hi else 'MISS'} ({r['jtok']:.3f})"))
    if all(a in acc for a in gd.ARMS):
        kl = {a: acc[a]["kl_primary_mean"] for a in gd.ARMS}
        out.append(("accuracy 1", "C0-H16 has the lowest primary KL",
                    "HIT" if min(kl, key=kl.get) == "C0-H16" else f"MISS (lowest {min(kl, key=kl.get)})"))
        out.append(("accuracy 2", "D-H4's KL >= 1.10 x D-H16's",
                    f"{'HIT' if kl['D-H4'] * TIE_DEN >= kl['D-H16'] * TIE_NUM else 'MISS'} "
                    f"({kl['D-H4']:.3g} vs {kl['D-H16']:.3g})"))
        sec = max(acc[a]["kl_secondary_mean"] for a in gd.ARMS)
        out.append(("accuracy 3", "C0-H4's primary KL > every arm's secondary KL",
                    f"{'HIT' if kl['C0-H4'] > sec else 'MISS'} ({kl['C0-H4']:.3g} vs max secondary {sec:.3g})"))
    return out


def sha256_lf(path: Path) -> str:
    """sha256 of a text file with CRLF read as LF: git's blob of a log the runner wrote with CRLF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def tokens_void(recs: list) -> dict:
    """Sitting 2's rule (c): {arm: why} for every arm whose sequences' tokens differ from sitting 1's."""
    out = {}
    for arm in gd.ARMS:
        got = sorted({s.get("tokens_sha") for r in recs if r["arm"] == arm for s in r.get("seqs", [])})
        if got and got != [TOKENS_SHA_S1[arm]]:
            out[arm] = f"tokens_sha {[g[:16] for g in got]} differs from sitting 1's {TOKENS_SHA_S1[arm][:16]}"
    return out


def verdict(suite_log: Path, acc_log: Path, rerun: bool = False) -> int:
    s_lines = suite_log.read_text(encoding="utf-8").splitlines()
    a_lines = acc_log.read_text(encoding="utf-8").splitlines()
    problems = []
    if rerun:
        print(f"SITTING 2 (the re-run rule): this sitting alone decides; the accuracy is sitting 1's, "
              f"{acc_log.name} (LF sha256 {sha256_lf(acc_log)[:16]}...)")
        if sha256_lf(acc_log) != S1_LOGS_LF.get(acc_log.name):
            problems.append(f"the accuracy log is not sitting 1's pinned one ({acc_log.name})")
    pins = next((json.loads(s.split(" ", 1)[1]) for s in s_lines if s.startswith("PINS_JSON ")), None)
    if pins is None or pins["mismatch"]:
        problems.append(f"pins: {pins and pins['mismatch']}")
    recs = [json.loads(s.split(" ", 1)[1]) for s in s_lines if s.startswith("ARM_JSON ")]
    stats = {(r["pass"], r["arm"]): arm_stats(r) for r in recs}
    res = combine(stats)
    if rerun:
        for arm, why in tokens_void(recs).items():
            res[arm] = {"state": "VOID", "why": why}
    for r in recs:
        for a in attributions(r):
            if "missing" not in a:
                print("PAGEIN_ATTRIBUTION", json.dumps(a))
    acc = {r["arm"]: r for r in (json.loads(s.split(" ", 1)[1]) for s in a_lines if s.startswith("ACC_JSON "))}
    ref = next((json.loads(s.split(" ", 1)[1]) for s in a_lines if s.startswith("REFERENCE_JSON ")), None)
    fc = {r["arm"]: r for r in (json.loads(s.split(" ", 1)[1]) for s in a_lines if s.startswith("FORCE_CHECK_JSON "))}
    for arm in (a for a in gd.ARMS if gd.ARMS[a][0] == "cpu"):
        if arm not in fc or fc[arm]["max_abs_diff"] != 0:
            problems.append(f"{arm}: forced logits differ from appended ones ({fc.get(arm)})")
    print(f"{'pass':>4} {'arm':7} {'tok/s':>7} {'agg':>7} {'idle W':>7} {'dPKG W':>7} {'dCores':>7} {'J/tok':>7} "
          f"{'CPU%':>5} {'780M pid':>8} {'rows':>5}")
    med = []
    for r in recs:
        st = stats[(r["pass"], r["arm"])]
        if not st["ok"]:
            print(f"{r['pass']:>4} {r['arm']:7} INCOMPLETE: {st['why']}")
            continue
        shift = ""
        if len(med) >= 3 and abs(st["idle_w"] - statistics.median(med)) > IDLE_SHIFT_W:
            shift = f"  idle shifted {st['idle_w'] - statistics.median(med):+.2f} W"
        med.append(st["idle_w"])
        print(f"{r['pass']:>4} {r['arm']:7} {st['tokps']:7.2f} {st['tokps_agg']:7.2f} {st['idle_w']:7.2f} {st['dP']:7.2f} "
              f"{st['dC']:7.2f} {st['jtok']:7.3f} {st['cpu']:5.1f} {st['gpu780_pid']:8.1f} {st['rows']:5d}  seqs "
              + " / ".join(f"{x:.2f}" for x in st["seq_tokps"]) + shift)
    print(f"\n{'arm':7} {'state':10} {'tok/s':>7} {'J/token':>8} {'KL mean':>10} {'KL sec':>10} {'top-1':>6} {'ppl':>7}")
    arms_d = {}
    for arm in ARM_ORDER:
        r, a = res[arm], acc.get(arm)
        kl = None
        if a is None:
            problems.append(f"no accuracy row for {arm}")
        elif not a.get("forced_sequence_equal"):
            problems.append(f"{arm}: the forced sequence differs from the pinned ids")
        elif a["nonfinite_positions"] == 0:
            kl = a["kl_primary_mean"]
        line = f"{arm:7} {r['state']:10} "
        line += (f"{r['tokps']:7.2f} {r['jtok']:8.3f} " if r["state"] == "OK" else f"{'-':>7} {'-':>8} ")
        line += (f"{a['kl_primary_mean']:10.3g} {a['kl_secondary_mean']:10.3g} {a['top1_agree']:6.3f} {a['ppl']:7.3f}"
                 + (f"  NON-FINITE at {a['nonfinite_positions']} positions" if a["nonfinite_positions"] else "")
                 if a else "no accuracy row")
        if r["state"] != "OK":
            line += f"  ({r['why']})"
            problems.append(f"{arm}: {r['why']}")
        print(line + (f"  below the {FLOOR_TOKPS:g} tok/s floor" if r["state"] == "OK" and r["tokps"] < FLOOR_TOKPS else ""))
        if r["state"] == "OK" and a is not None:
            arms_d[arm] = {"kl": kl, "tokps": r["tokps"], "jtok": r["jtok"], "head": HEAD_OF[arm]}
    if ref:
        print(f"reference perplexity over {ref['positions']} positions: {ref['ppl']:.3f}")
    print("\n(d): NPU-only decode, per head format; the NPU's ceiling at 47.62 GB/s and its floor at 0.328 J/GB")
    for d in (rule_d(arms_d) if not problems else []):
        if d["speed"] == "UNDECIDED":
            print(f"  {d['head']} ({d['assume']}): no measured arm on {d['head']} -> UNDECIDED")
            continue
        print(f"  {d['head']} ({d['assume']}): KL* {d['kl_ref']:.3g}, rivals {', '.join(d['rivals'])}")
        print(f"    speed: ceiling {d['ceiling']:.2f} tok/s vs 1.10 x {d['fastest']} {d['fastest_tokps']:.2f} = "
              f"{d['fastest_tokps'] * 1.1:.2f} -> {d['speed']}")
        print(f"    energy: 1.10 x E_npu {d['e_npu']:.3f} = {d['e_npu'] * 1.1:.3f} J/token vs {d['frugal']} "
              f"{d['frugal_jtok']:.3f} -> {d['energy']}" + ("  (an energy OPEN is not a finding: see the prereg)"
                                                            if d["energy"] == "OPEN" else ""))
    print("\nLabels")
    for s in labels():
        print("  " + s)
    print("\nPredictions")
    for tag, what, score in predictions(res, acc):
        print(f"  {tag}: {what}: {score}")
    for p in problems:
        print("PROBLEM", p)
    print("VERDICT", "INCOMPLETE" if problems else "COMPLETE")
    return 2 if problems else 0


# ---------------------------------------------------------------- sitting 1's GPU witness, re-read (POST HOC)

def gpu_frame(path: Path) -> tuple:
    """A counter file's GPU-engine columns ({index: (pid, luid, engtype)}, the fixed regex) and its rows
    [(epoch s, {index: busy %})]."""
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    g = gpu_columns([h.lower() for h in rows[0]])
    out = []
    for r in rows[1:]:
        try:
            t = datetime.strptime(r[0], "%m/%d/%Y %H:%M:%S.%f").timestamp()
        except (ValueError, IndexError):
            continue
        vals = {}
        for i in g:
            try:
                vals[i] = float(r[i]) if i < len(r) else 0.0
            except ValueError:
                vals[i] = 0.0
        out.append((t, vals))
    return g, out


def gpu_window(g: dict, rows: list, luids: set, pid, a=None, b=None) -> dict:
    """Mean busy % over the rows (those inside [a, b] by inside()'s rule, if given): the 780M's engines
    summed, the reader's own (and per engine type), every other pid on the 780M (top 3), and the
    other adapters."""
    rs = [v for t, v in rows if a is None or a + 1.0 <= t <= b]
    if not rs:
        return {"rows": 0}
    m = {i: statistics.fmean(v.get(i, 0.0) for v in rs) for i in g}
    on780 = [i for i in g if g[i][1] in luids]
    own = [i for i in on780 if g[i][0] == pid]
    others = {}
    for i in on780:
        if g[i][0] != pid:
            others[g[i][0]] = others.get(g[i][0], 0.0) + m[i]
    by_eng = {}
    for i in own:
        by_eng[g[i][2]] = round(by_eng.get(g[i][2], 0.0) + m[i], 2)
    return {"rows": len(rs), "gpu780_all": round(sum(m[i] for i in on780), 2), "reader": round(sum(m[i] for i in own), 2),
            "reader_by_engine": {k: v for k, v in by_eng.items() if v}, "other_pids_top": [
                [p, round(x, 2)] for p, x in sorted(others.items(), key=lambda kv: -kv[1])[:3]],
            "other_adapters": round(sum(m[i] for i in g if g[i][1] not in luids), 2)}


def posthoc_gpu(suite_log: Path, csv_dir: Path, stamp: str) -> int:
    """POST HOC, decides nothing (the gate, 2026-09-24): sitting 1's counter files re-read with the
    fixed GPU regex, per arm-pass idle and per sequence window."""
    lines = suite_log.read_text(encoding="utf-8").splitlines()
    ad = next(json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith("ADAPTERS_JSON "))
    luids = set(ad["luids_780m"])
    print(f"POST HOC, decides nothing: {suite_log.name} (LF sha256 {sha256_lf(suite_log)}), its counter files "
          f"{csv_dir.as_posix()}/{stamp}_*.csv, re-read with the fixed regex (engine types with a space, "
          f"\"Compute 0\"). The 780M: {sorted(luids)}. Busy % is summed over engines, a mean over the rows "
          f"inside each window (inside()'s rule). The pids of other processes are all the counters name.")
    for f in sorted(csv_dir.glob(f"{stamp}_*.csv")):
        say("CSV_SHA_JSON", {"file": f.name, "bytes": f.stat().st_size, "sha256_crlf_working_copy": sha256(f),
                             "sha256_lf_blob": sha256_lf(f)})
    for s in lines:
        if not s.startswith("ARM_JSON "):
            continue
        rec = json.loads(s.split(" ", 1)[1])
        n, arm, pas, pid = rec["n"], rec["arm"], rec["pass"], rec.get("ready", {}).get("pid")
        gi, ri = gpu_frame(csv_dir / f"{stamp}_{n:02d}_{arm}_idle.csv")
        gw, rw = gpu_frame(csv_dir / f"{stamp}_{n:02d}_{arm}_window.csv")
        out = {"n": n, "pass": pas, "arm": arm, "reader_pid": pid, "idle": gpu_window(gi, ri, luids, None),
               "windows": [{"seq": q["seq"], "tokps": round(q["tokps"], 2), **gpu_window(gw, rw, luids, pid, *q["win_wall"])}
                           for q in rec.get("seqs", []) if "win_wall" in q]}
        say("POSTHOC_GPU_JSON", out)
        print(f"  {n} pass {pas} {arm:7} idle 780M {out['idle'].get('gpu780_all', 0):6.2f}  windows: " + " | ".join(
            f"{w['tokps']:.2f} tok/s: 780M {w['gpu780_all']:.1f}, reader {w['reader']:.1f}, others "
            f"{sum(x for _, x in w['other_pids_top']):.2f}" for w in out["windows"] if w.get("rows")))
    return 0


# ---------------------------------------------------------------- prereg and selftest

PREREG = f"""Gemma 3 4B decode on the CPU and DirectML: speed, energy per token and accuracy, and (d)'s rule
for NPU-only decode (LLM study, pre-registration (c) with (d), locked decision 10), pre-registered
before any sitting

Question
  The bar (locked decision 10): an NPU arm earns a role for Gemma 3 4B decode if it is faster, or
  more accurate, or uses less energy per token, or frees the GPU, or frees the CPU, than both the
  CPU (ONNX Runtime) and DirectML on the Radeon 780M. The floor is {FLOOR_TOKPS:g} tok/s (the user's number).
  (c) measures the two rivals on the release's own weights: decode tok/s, J/token (package
  counters, (b)'s method and labels) and accuracy against the release. (d) applies a mechanical
  rule to (b) and (c): is there a speed or an energy route for NPU-only decode at this model's
  bytes? Freeing the GPU and the CPU is (e), pre-registered after (c).

The models: (c)'s build (tools/gemma_decode.py; its log is quoted under BUILD FACTS below)
  The release is google/gemma-3-4b-it-qat-q4_0-gguf @ 15f73f5e, gemma-3-4b-it-q4_0.gguf: 238 q4_0
  linears, a tied F16 head [262,144 x 2,560] and 205 F32 norms. onnxruntime-genai 0.11.2's builder
  (onnxruntime-genai-directml-ryzenai on ONNX Runtime 1.23.3.dev20260320, resnet_env17) built five
  models from a text-only copy of google/gemma-3-4b-it-qat-q4_0-unquantized @ 7c0881d8 whose
  embedding is the release's F16 head. The builder's model load was wrapped, so that every linear
  enters as the GGUF's own codes and scales through the builder's pre-quantized MatMulNBits path.
  Every weight was then read back against the release by value:
    linears     all 3,208,642,560 dequantized weights equal the GGUF's d * (q - 8), in every model
    embedding   the release's F16 head, all 262,144 rows (fp32 on the CPU, fp16 on DirectML); the
                checkpoint's 64 extra rows (padding and the image token, id 262,144) are not in it
    head        H16: the release's F16 head (fp32 on the CPU, which holds F16 exactly; fp16 on
                DirectML). H4: the builder's own RTN int4 (block 32, symmetric) of the release's
                F16 head, a lossy step the release does not take; its rel-L2 is quoted below
    norms       N1: the GGUF's F32 norms, written and read back (the builder writes the
                checkpoint's + 1, which differs; B1's norm check)
  Three defects of the installed stack were corrected in the graphs and verified:
    F1  onnxruntime-genai 0.11.2 ignores rope_scaling (linear, factor 8): the global layers' cos/sin
        caches are written with transformers' values (fixed on genai main, 4859742e)
    F2  it rounds the embedding scale to 50.6: set to sqrt(2560) (fp32 50.596443; fp16 is 50.59375
        either way, so the DirectML arms do not change)
    F3  ORT 1.23.3's GroupQueryAttention CPU kernel keeps local_window_size + 1 keys; the 1.23
        releases predate the fix (8af9f58086, #25927) and the installed build behaves unfixed
        (MEASURED on the selftest's tiny model). The CPU arms' 29 local layers get 1023, so they keep
        1024 keys as transformers does. The DirectML arms keep 1024.
    F4  (a scope limit, not a fix) DirectML's GroupQueryAttention reads no local_window_size (the
        ORT main and rel-1.23.0 sources): through ORT's DirectML EP, Gemma 3 past 1024 positions is
        a different model. That is a limit of DirectML's stack, not of the 780M. D-H16's KL within
        and past the window at 1100 tokens is quoted below; it is outside every verdict.
  The fidelity gate: C0-H16 against transformers' fp32 Gemma3ForCausalLM with the release's
  weights (the GGUF's dequantized linears, F16 head and F32 norms, the norms applied as the GGUF's
  values bit for bit), streamed one layer at a time, at 64 and 1100 tokens: max KL per position
  <= {gd.FID_KL_MAX:g}. Its staged passes (as built, N1, N1+F1+F2, all four) are quoted below.
  B3', DirectML placement (the gate's amendment of B3, 2026-09-24): every node of D-H4 and D-H16 on
  DmlExecutionProvider (ORT's verbose placement log) except the two that make GroupQueryAttention's
  total_sequence_length, input 6, which DirectML's GQA registers with requiredConstantCpuInputs(6)
  (onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/OperatorRegistration.cpp:1179,
  ORT main and rel-1.23.0), so ORT places its producers on the CPU by design. The CPU node set
  must EQUAL, by name, {{{B3_CPU_ALLOWED[0]} (Shape(attention_mask)[1]),
  {B3_CPU_ALLOWED[1]} (to int32)}}, and the Cast's only consumers must be the 34 GQA nodes'
  input 6 (read from each graph by this prereg, B3_PRIME_JSON). Anything else on the CPU voids the
  arm. The two nodes read no weight and no activation, only the mask's shape, and their cost is
  inside every timed DirectML window. The build log's B3 subprocesses exited 0xC0000142
  (STATUS_DLL_INIT_FAILED) before ONNX Runtime loaded (spawned after a host memory guard had killed
  the console-owning wrapper; the cause is INFERRED), so B3, the smoke and the SHA-256s were run
  again on the same files (the recheck log; its MODEL_SHA_JSON must equal the build log's).
  The smoke: {gd.SMOKE_TOKENS} greedy tokens per arm through GenAI. 0.11.2's og.Tokenizer.encode adds no BOS
  for this tokenizer (transformers' adds it), and Gemma 3 without BOS degenerates (the build log:
  ' is' {gd.SMOKE_TOKENS} times on every arm); the recheck gives each arm both rows. The sitting and the
  accuracy pass prepend BOS themselves.

Arms
  C0-H4   CPU, MatMulNBits accuracy_level 0 (fp32 compute), H4 head
  C4-H4   CPU, accuracy_level 4 (int8 compute; the builder's CPU default), H4 head
  D-H4    DirectML on the 780M (fp16 io and compute), H4 head
  C0-H16  CPU, accuracy_level 0, H16 head held in fp32: the release at fp32 compute
  D-H16   DirectML, H16 head
  Bytes per token (DERIVED): linears 1,804,861,440 B; H4 head 377,487,360 B with fp16 scales
  (the CPU arms store fp32 scales, 419,430,400 B); H16 head 1,342,177,280 B. So 2.182 GB (H4) and
  3.147 GB (H16), the figures (d) uses for an NPU reading fp16 or bf16 scales.

Protocol (the gate's decision (A): every arm stays within 1024 positions)
  The text: {TEXT_PIN['repo']} @ {TEXT_PIN['revision'][:8]}, {TEXT_PIN['file']}
  (sha256 {TEXT_PIN['sha256'][:16]}...), its lines joined by a blank line as the Hugging Face
  perplexity guide joins them, tokenized by the release's tokenizer, BOS + the first 1023 tokens.
  The ids are pinned by SHA-256 (TEXT_JSON below). Every arm and the reference see exactly one BOS
  (id 2, prepended by hand; the text is tokenized with add_special_tokens=False), and IDS_SHA pins
  identical token ids across the arms, the passes and the accuracy pass. transformers 4.57.6 warns
  of "an incorrect regex pattern" (a Mistral check) when it loads this tokenizer; the pinned ids
  equal SentencePiece's on the release's tokenizer.model and the raw tokenizer.json's
  (TOKENIZER_CHECK_JSON, checked by this prereg), so the warning changes no id.
  Decode: the first {PROMPT_TOKENS} ids as the prompt, greedy, min_length = max_length = {POSITIONS} positions,
  so {GEN_TOKENS} tokens are generated. The window runs from generated token {WIN_FROM}'s completion to token
  {WIN_TO}'s ({WIN_TOKENS} tokens), each completion on perf_counter_ns after generate_next_token and
  get_next_tokens; warm-up is by position (stage 3b's ramp lesson).
  Within 1024 positions Gemma 3's 1024-key window never binds: the query at position 1023 sees keys
  0-1023 in transformers and in every arm. So F3's edit and F4's gap change no (c) measurement; F3
  matters only to the >= 1100-token fidelity gate.
  CPU arms: one session per arm-pass, {SEQS['cpu']} sequences in it. If GenAI 0.11.2 takes the thread
  affinities (the build's AFFINITY_JSON; PIN below), the session runs {len(PIN_CPUS)} intra-op threads, the
  caller pinned to logical CPU {PIN_CPUS[0]} and the pool to {', '.join(str(c) for c in PIN_CPUS[1:])} (one per physical core); the
  pinning is read back per session and a mismatch voids the arm-pass (as in the noise study).
  DirectML arms: {SEQS['dml']} fresh sessions per arm-pass, one sequence each, on the 780M (GenAI creates its
  own DirectML device from provider options {{"dml": {{}}}}; the sitting refuses to start unless every
  hardware adapter DXGI lists is the 780M, ADAPTERS_JSON), each session released (and the release
  logged with the available RAM) before the next is created.
  An arm-pass's tok/s is the median of its sequences' window tok/s. Every sequence must generate all
  {GEN_TOKENS} tokens, or the arm-pass is INCOMPLETE.
  Order: {', '.join(ARM_ORDER)}, then mirrored: two passes. Before each arm-pass the host
  witness (xrt-smi and the host load, tools/silicon_probe_record.py) and a {SETTLE_S:.0f} s settle, then a
  {IDLE_S} s idle with nothing launched; the reader process starts after the idle, loads its model, runs
  an untimed check of {CHECK_TOKENS} tokens and reports READY; the counters start then, and the sequences after
  {TP_LEAD_S} s. The host-load gate (busy cores < 2, no peer >= 0.5) runs once at the start, and so
  does the memory start (the gate's condition): the available RAM and every process of >= 1 GB
  private are logged (MEMORY_START_JSON), and a process of >= {MEM_BIG_GB:g} GB refuses the sitting (a builder
  peaked at 19-21 GB, one arm's session holds 2-6 GB; the desktop's largest was 0.95 GB).
  tok/s and J/token must each agree within {PASS_AGREE:.0%} between the passes, or the arm is
  INCOMPLETE; an arm's value is the mean of its two passes. Every figure is reported beside the
  {FLOOR_TOKPS:g} tok/s floor.
  Every model file's SHA-256 is verified at the start (MODEL_SHA, from the build log); any mismatch
  stops the sitting.

Energy ((b)'s method)
  typeperf at 1 Hz: \\Energy Meter(RAPL_Package0_PKG)\\Power, the eight core meters, % Processor
  Time, \\Memory\\Available MBytes, \\Memory\\Pages Input/sec and \\GPU Engine(*)\\Utilization
  Percentage. A row stamped t holds the second ending at t; it counts if that second lies inside a
  sequence's window (by wall clock, mapped once per sequence from perf_counter_ns).
  J/token of an arm-pass = (mean package power over the rows inside its sequences' windows - its
  idle's mean) / (its window tokens / its windows' total time). The rise of the core meters is
  reported beside it. Fewer than {MIN_ROWS} rows in the idle or in the arm-pass's windows voids it. An
  idle more than {IDLE_SHIFT_W:g} W from the running median of the earlier idles is flagged.
  Labels ((b), 1e546f0: GPU INSIDE and NPU INSIDE): NPU-vs-DirectML and NPU-vs-CPU energy compare,
  labelled; a component scaling with each chip's work is inside the package, not shown to be all of
  it; DRAM power is outside the package (package counters only: completeness is never shown), and
  shared DRAM pulls ratios toward 1.
  GPU witness: the 780M's LUIDs are the DXGI adapters with its vendor and device id (0x1002,
  0x15bf), enumerated at the sitting's start (ADAPTERS_JSON). DXGI lists the 780M twice on this
  desktop, and the Microsoft Basic Render Driver (a software adapter) also shows a 3D engine while
  anything renders through it, so "the LUID with a 3D engine" is not unique. The 780M's engines are
  summed, the reader's own apart; every other LUID (the NPU, the software adapter) is reported as
  other. DXGI's adapter 0, ORT's DirectML device 0 (EnumAdapters1(device_id),
  dml_provider_factory.cc:499-502, rel-1.23.0), must be the 780M or the sitting refuses to start.
  Memory witness (the gate's addition): the available RAM before each arm-pass and after each
  session release, and \\Memory\\Pages Input/sec over each sequence's window. A window whose mean
  exceeds {PAGE_MAX:g} pages/s is VOID, and with it the arm-pass. Reason: {PAGE_MAX:g} pages/s is 0.4 MB/s;
  a decode whose weights paged would re-read at least 1% of a token's >= 1.8 GB each token (18 MB,
  about 4,400 pages), >= 22,000 pages/s at the {FLOOR_TOKPS:g} tok/s floor, while this desktop's background
  hard faults are far below the threshold (an idle sample in the dry run: 0 pages/s in five
  seconds, 40 in one). (The build log's builder runs peaked at 19.0-20.9 GB private and left as
  little as 0.74 GB available, 0.03 GB in the stopped run; the sitting holds one model at a time,
  2-6 GB, and on this APU the 780M's memory is system RAM.)

Accuracy (a separate, untimed pass: gemma_decode_suite.py accuracy)
  Changed from the draft, named to the gate: each arm is teacher-forced through GenAI's own decode
  path, one token per step, not a plain ORT prefill, because a prefill runs other kernels
  (MatMulNBits at M > 1, GQA's prompt path) than the timed decode does. The forcing: after BOS is
  appended, at each position get_logits is read, the logits are then set (set_logits) to a one-hot
  on the next pinned id, and a greedy generate_next_token appends that id. GenAI 0.11.2 refuses a
  second append_tokens on DirectML ("Continuous decoding is not supported on the selected device
  type (DirectML)", the dry run), so every arm is forced this way. Checked in the accuracy log: each
  arm's generated sequence must equal the pinned ids, and on each CPU arm the forced logits must
  equal those of append_tokens one id at a time over the first {FORCE_CHECK_N - 1} positions (FORCE_CHECK_JSON,
  max |diff| 0); either failing withholds the verdict. (Before this prereg, functional only: C4-H4
  equal bit for bit over 39 positions, and force_check itself 0 on C0-H4, C0-H16 and C4-H4 over 31;
  D-H4's forced sequence equal to the ids, its forced rows within KL 3e-5 of a single prefill of
  the same ids. That check also printed D-H4 against C4-H4 over 39 positions, top-1 0.974 and max
  KL 0.0127: seen before this prereg, used for nothing.) The accuracy pass logs its memory start
  (MEMORY_START_JSON) too.
  Over the pinned 1024 ids: {POSITIONS - 1} positions, each predicting the next id.
  Primary reference, common to all arms: the fidelity gate's model (transformers' fp32 Gemma 3 with
  the release's weights) on the same ids, its final-norm hidden states times the release's F16 head.
  Secondary: the same hidden states times each arm's own head (the H4 arms' int4 head dequantized
  from its model; for H16 the secondary is the primary). KL against it isolates the arm's compute
  error. Reported; it decides nothing.
  Metric: mean over positions of KL(reference || arm), in nats. Top-1 agreement with the primary and
  each arm's perplexity are reported beside it, with the reference's. The vocabulary is the
  release's 262,144 tokens in every arm and in the reference.
  "At least as accurate": arm A is at least as accurate as B if KL_A <= 1.10 x KL_B, any KL below
  {KL_FLOOR:g} taken as {KL_FLOOR:g} so that near-exact arms tie. A non-finite logit at any position is an accuracy
  failure: that arm is less accurate than every finite arm.

Written predictions (stated expectations; the rules decide)
  Speed (tok/s): D-H4 {PRED_TOKPS['D-H4'][0]}-{PRED_TOKPS['D-H4'][1]}; C4-H4 {PRED_TOKPS['C4-H4'][0]}-{PRED_TOKPS['C4-H4'][1]}; C0-H4 {PRED_TOKPS['C0-H4'][0]}-{PRED_TOKPS['C0-H4'][1]}; D-H16 {PRED_TOKPS['D-H16'][0]}-{PRED_TOKPS['D-H16'][1]}; C0-H16 {PRED_TOKPS['C0-H16'][0]}-{PRED_TOKPS['C0-H16'][1]}.
  Energy (J/token, package-reported), above the read floors DERIVED from (b)'s J/GB at full-rate
  streaming (CPU 0.880 x 2.182 = 1.92 for H4 and 0.880 x 4.489 = 3.95 for C0-H16's fp32 head;
  DirectML 0.748 x 2.182 = 1.63 for H4 and 0.748 x 3.147 = 2.35 for H16): the CPU arms 2.5-4.5
  (C0-H16 up to 7), the DirectML arms 1.6-3.0.
  Accuracy: C0-H16 has the lowest primary KL; D-H4's KL >= 1.10 x D-H16's (the int4 head costs
  more than DirectML's fp16 compute); C0-H4's primary KL, about the H4 head's cost alone, exceeds
  every arm's secondary (compute-only) KL. Non-finite logits on DirectML are possible (the
  builder's own fp16 warning); not scored.

(d): mechanical, on (b) and (c), for an NPU reading head format H in {{H4, H16}}
  No NPU decode arm exists, so its accuracy is assumed generously: KL*_H is the lowest primary KL
  among the measured arms reading H. The rivals are every finite measured arm, of any format, at
  least as accurate as KL*_H (KL <= 1.10 x KL*_H); the arm that sets KL*_H is always one. A KILL
  under this assumption is robust; an OPEN needs a real NPU arm to show its own KL. The same rules
  are also reported with the NPU assumed only as accurate as the least accurate measured arm on H.
  Speed: the NPU's ceiling is {NPU_GBPS} GB/s (MEASURED, stage 1) / GB per token: {NPU_GBPS / GB_TOKEN['H4']:.2f} tok/s (H4) and
    {NPU_GBPS / GB_TOKEN['H16']:.2f} (H16), DERIVED. KILL if the ceiling < 1.10 x the fastest rival's tok/s; otherwise OPEN
    (exactly 1.10x is OPEN, stage 3's keep rule).
  Energy: the NPU's floor is E_npu(H) = {JGB_NPU} J/GB (R-npu, MEASURED, 1e546f0) x GB per token:
    {JGB_NPU * GB_TOKEN['H4']:.3f} J/token (H4) and {JGB_NPU * GB_TOKEN['H16']:.3f} (H16), DERIVED. KILL if 1.10 x E_npu(H) > the rivals' lowest
    J/token; otherwise OPEN.
  Stated in advance (DERIVED from (b), the gate's addition): the energy rule cannot KILL here. Every
    rival's read floor (1.63 J/token at the lowest, DirectML H4) is already above 1.10 x E_npu
    ({1.1 * JGB_NPU * GB_TOKEN['H4']:.3f} for H4, {1.1 * JGB_NPU * GB_TOKEN['H16']:.3f} for H16). So (d)'s energy result will be OPEN whatever (c)
    measures, and an energy OPEN is not a finding. The energy question is decided only by a real
    NPU decode arm.
  Budget (a SKETCH, all DERIVED and untested): to stay 1.10x below DirectML's 0.748 J/GB, an NPU
    decode must stay under about 0.68 J/GB. Its headroom over R-npu's 0.328 is about 0.35 J/GB,
    about 16.7 W at 47.62 GB/s. N-c's compute added about 5.6 W over C1 in (b); R-npu's host added
    only 0.23 W of cores (XRT's waits do not busy-poll); a busy-waiting host thread (C1: about
    +10 W of cores) would eat most of the headroom.
  Each outcome carries (b)'s labels. The selftest covers both directions and the edges of both
  rules and of eligibility (a faster but less accurate arm excluded, two KLs under the floor tying,
  a non-finite arm excluded, exact 1.10x OPEN).

Unverified by design: DRAM power (outside the package) and completeness; any NPU decode (no NPU
arm exists); positions past 1024 (F4 makes DirectML a different model there); other prompts, batch
sizes and sampling; GenAI's prefill speed (not timed); anything about (e).
"""


def build_facts(log: Path) -> dict:
    """The build log's facts, quoted for the prereg: nothing is transcribed by hand."""
    lines = log.read_text(encoding="utf-8").splitlines()
    grab = lambda tag: [json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith(tag + " ")]  # noqa: E731
    fid = grab("FIDELITY_JSON")
    rel = {(r["arm"], tuple(r["fixes"])): r for r in grab("RELEASE_JSON")}
    final = {a: r for (a, fx), r in rel.items() if fx == tuple(gd.FIXES)}
    facts = {"b1": grab("B1_JSON")[-1], "b1_common": grab("B1_COMMON_JSON")[-1],
             "reference_norms": grab("REFERENCE_NORMS_JSON")[-1],
             "fidelity": [{"tag": f["tag"], "pass": f["pass"],
                           "passes": [{k: p[k] for k in ("arm", "tokens", "max_kl", "max_kl_within_window",
                                                         "max_kl_past_window", "top1_agree")} for p in f["passes"]]}
                          for f in fid],
             "release": {a: {"ok": r["ok"] if "ok" in r else None, "linears_exact": r["linears"]["exact"],
                             "linears": r["linears"]["weights"], "norms_on_disk": r["norms"]["exact_on_disk"],
                             "gqa_windows": r["gqa_windows"], "embed_scale_on_disk": r["embed_scale"]["on_disk"],
                             "head": {k: v for k, v in r["head"].items() if k in ("format", "exact", "n", "rel_l2_vs_release",
                                                                               "scales_dtype", "bytes")}}
                         for a, r in final.items()},
             "b3": grab("B3_JSON"), "affinity": grab("AFFINITY_JSON")[-1], "smoke": grab("SMOKE_JSON"),
             "model_sha": grab("MODEL_SHA_JSON")[-1], "builders": grab("BUILDER_JSON"),
             "done": next((s for s in lines if s.startswith("BUILD_DONE")), None)}
    return facts


def b3_prime(arm: str, row) -> dict:
    """B3' (the gate, 2026-09-24): every node of a DirectML arm on DmlExecutionProvider but the two
    that make GroupQueryAttention's total_sequence_length (input 6; DirectML's GQA registers it with
    requiredConstantCpuInputs(6), OperatorRegistration.cpp:1179 on ORT main and rel-1.23.0): the
    Gather of Shape(attention_mask)[1] and its Cast to int32, by name, the Cast's only consumers every
    GQA node's input 6. Anything else on the CPU voids the arm. Read from the recheck's B3 row and the
    graph (no weights loaded)."""
    import onnx
    from onnx import numpy_helper
    if row is None or row["status"] not in ("ALL_DML", "CPU_NODES"):
        return {"arm": arm, "ok": False, "why": f"placement {row and row['status']}"}
    names = sorted(re.sub(r"^\w+ \((.*)\)$", r"\1", s) for s in row["cpu_nodes"])
    n_cpu = row["placed"].get("CPUExecutionProvider", 0)
    if n_cpu == 0:
        return {"arm": arm, "ok": True, "cpu_nodes": [], "why": "ALL_DML"}
    g = onnx.load(str(gd.ONNX / arm / "model.onnx"), load_external_data=False).graph
    node = {n.name: n for n in g.node}
    prod = {o: n for n in g.node for o in n.output}
    inits = {t.name: t for t in g.initializer}

    def const(name):
        if name in prod and prod[name].op_type == "Constant":
            return numpy_helper.to_array(prod[name].attribute[0].t).tolist()
        return numpy_helper.to_array(inits[name]).tolist() if name in inits else None

    def uses(out):
        return sorted((n.op_type, n.name, i) for n in g.node for i, x in enumerate(n.input) if x == out)

    gather, cast = (node.get(n) for n in B3_CPU_ALLOWED)
    shape = prod.get(gather.input[0]) if gather is not None else None
    n_gqa = sum(n.op_type == "GroupQueryAttention" for n in g.node)
    checks = {
        "cpu_set_equal": names == sorted(B3_CPU_ALLOWED) and n_cpu == len(B3_CPU_ALLOWED),
        "gather_of_shape_mask_1": (gather is not None and gather.op_type == "Gather" and shape is not None
                                   and shape.op_type == "Shape" and list(shape.input) == ["attention_mask"]
                                   and const(gather.input[1]) == 1),
        "cast_to_int32": (cast is not None and cast.op_type == "Cast" and gather is not None
                          and list(cast.input) == [gather.output[0]]
                          and any(a.name == "to" and a.i == onnx.TensorProto.INT32 for a in cast.attribute)),
        "gather_feeds_only_the_cast": gather is not None and [u[1] for u in uses(gather.output[0])] == [B3_CPU_ALLOWED[1]],
        "cast_feeds_only_gqa_input_6": (cast is not None and n_gqa == gd.LAYERS
                                        and [(u[0], u[2]) for u in uses(cast.output[0])] == [("GroupQueryAttention", 6)] * n_gqa),
    }
    bad = [k for k, v in checks.items() if not v]
    return {"arm": arm, "ok": not bad, "cpu_nodes": names, "gqa_nodes": n_gqa, "checks": checks,
            "why": "B3': only GQA's total_sequence_length producers on the CPU" if not bad else f"failed {bad}"}


def recheck_facts(log: Path) -> dict:
    """The recheck log's B3, smoke, SHA-256s and end (gemma_decode.py build --steps b3,smoke,pins)."""
    lines = log.read_text(encoding="utf-8").splitlines()
    grab = lambda tag: [json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith(tag + " ")]  # noqa: E731
    return {"b3": grab("B3_JSON"), "smoke": grab("SMOKE_JSON"), "model_sha": (grab("MODEL_SHA_JSON") or [None])[-1],
            "done": next((s for s in lines if s.startswith("BUILD_DONE")), None)}


def prereg(build_log: Path, recheck_log: Path) -> int:
    print(PREREG)
    facts = build_facts(build_log)
    rc = recheck_facts(recheck_log)
    facts["build_b3"] = [{k: r[k] for k in ("arm", "exit", "status")} for r in facts["b3"]]
    facts["build_smoke"] = [{k: r[k] for k in ("arm", "tokens", "text")} for r in facts["smoke"]]
    facts["b3"], facts["smoke"], facts["recheck_done"] = rc["b3"], rc["smoke"], rc["done"]
    print(f"BUILD FACTS, quoted from {build_log.as_posix()} (sha256 {sha256(build_log)}); B3 and the smoke from "
          f"{recheck_log.as_posix()} (sha256 {sha256(recheck_log)})")
    say("BUILD_FACTS_JSON", {k: v for k, v in facts.items() if k != "model_sha"})
    for r in rc["b3"]:
        print(f"  B3 {r['arm']}: {r['status']} {r['placed']}" + (f", CPU ops {r['cpu_ops']}" if r["cpu_ops"] else ""))
    for r in rc["smoke"]:
        print(f"  smoke {r['arm']} ({'BOS first' if r['bos_first'] else 'as encoded'}): {r['text']!r}")
    for f in facts["fidelity"]:
        for p in f["passes"]:
            print(f"  fidelity [{f['tag']}] {p['arm']} {p['tokens']} tokens: max KL {p['max_kl']:.3g} (within the "
                  f"window {p['max_kl_within_window']:.3g}" + (f", past it {p['max_kl_past_window']:.3g}"
                                                               if p["max_kl_past_window"] is not None else "") + ")")
    bad = []
    want_sha = {a: {f: v["sha256"] for f, v in files.items()} for a, files in facts["model_sha"].items()}
    if MODEL_SHA != want_sha:
        bad.append("MODEL_SHA differs from the build log's MODEL_SHA_JSON")
    if PIN is None or PIN != facts["affinity"]["takes_affinities"]:
        bad.append(f"PIN {PIN} differs from the build's affinity read-back {facts['affinity']['takes_affinities']}")
    if not (facts["done"] or "").startswith("BUILD_DONE OK"):
        bad.append(f"the build did not end OK: {facts['done']}")
    if not (rc["done"] or "").startswith("BUILD_DONE OK"):
        bad.append(f"the recheck did not end OK: {rc['done']}")
    if rc["model_sha"] != facts["model_sha"]:
        bad.append("the recheck's MODEL_SHA_JSON differs from the build log's: the models changed")
    for arm in ("D-H4", "D-H16"):
        v = b3_prime(arm, next((r for r in rc["b3"] if r["arm"] == arm), None))
        say("B3_PRIME_JSON", v)
        print(f"  B3' {arm}: {'PASS' if v['ok'] else 'VOID'} ({v['why']})")
        if not v["ok"]:
            bad.append(f"B3' {arm}: {v['why']}; the arm is void")
    ids = text_ids()
    if ids_sha(ids) != IDS_SHA:
        bad.append(f"IDS_SHA {IDS_SHA} differs from the text's {ids_sha(ids)}")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    p = Path(hf_hub_download(TEXT_PIN["repo"], TEXT_PIN["file"], revision=TEXT_PIN["revision"], repo_type="dataset"))
    text = "\n\n".join(pq.read_table(p).column("text").to_pylist())
    tok = AutoTokenizer.from_pretrained(str(gd.TEXT))
    n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
    n_words = len(text.split())
    import sentencepiece as spm
    from tokenizers import Tokenizer
    sp = spm.SentencePieceProcessor(model_file=str(gd.TEXT / "tokenizer.model")).encode(text[:20000])
    raw = Tokenizer.from_file(str(gd.TEXT / "tokenizer.json")).encode(text[:20000], add_special_tokens=False).ids
    tc = {"sentencepiece_equal": sp[:POSITIONS - 1] == ids[1:], "raw_tokenizer_json_equal": raw[:POSITIONS - 1] == ids[1:]}
    say("TOKENIZER_CHECK_JSON", tc)
    if not all(tc.values()):
        bad.append(f"the pinned ids differ from an independent tokenizer: {tc}")
    say("TEXT_JSON", {"pin": TEXT_PIN, "ids_sha": ids_sha(ids), "prompt": ids[:PROMPT_TOKENS],
                      "tokens_per_word_whole_test_split": n_tok / n_words, "tokens": n_tok, "words": n_words,
                      "prompt_text": tok.decode(ids[1:PROMPT_TOKENS])})
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    say("CONSTANTS_JSON", {"protocol": {"prompt": PROMPT_TOKENS, "positions": POSITIONS, "generated": GEN_TOKENS,
                                        "window": [WIN_FROM + 1, WIN_TO], "seqs": SEQS, "order": ORDER,
                                        "idle_s": IDLE_S, "settle_s": SETTLE_S, "min_rows": MIN_ROWS,
                                        "pass_agree": PASS_AGREE, "page_max": PAGE_MAX},
                           "pin": PIN, "pin_cpus": PIN_CPUS, "model_sha": MODEL_SHA, "ids_sha": IDS_SHA,
                           "d": {"jgb_npu": JGB_NPU, "npu_gbps": NPU_GBPS, "gb_token": GB_TOKEN, "kl_floor": KL_FLOOR,
                                 "tie": [TIE_NUM, TIE_DEN]},
                           "pred_tokps": PRED_TOKPS, "pred_jtok": PRED_JTOK, "git_head": head})
    print("git HEAD", head, "(plus this log's own commit)")
    for b in bad:
        print("PIN_MISMATCH", b)
    return 3 if bad else 0


RERUN = f"""Gemma 3 4B decode, pre-registration (c): the rule for sitting 2, committed before it (the gate's
decision of 2026-09-24, after sitting 1 at 99a7eec)

Sitting 1 (gemma_decode_suite, _accuracy and _verdict_desktop2_20260924.log) is INCOMPLETE and stays
so in the record: three arm-passes void on the memory witness (C4-H4 pass 1, D-H4 pass 2, C0-H16 pass 2),
and D-H16's passes disagree (13.35 against 10.85 tok/s). C0-H4 alone completed there; its values are
recomputed from that log below. Sitting 2 runs the full matrix again and ALONE decides: no sitting-1
arm-pass enters its verdict. If sitting 2 is INCOMPLETE, it stands, and there is no third run without the
user.

Unchanged from the prereg at 68e9cfe (gemma_decode_prereg_desktop2_20260924.log):
  - the arms, the order and the lengths (a {PROMPT_TOKENS}-token prompt, {GEN_TOKENS} generated, window
    {WIN_FROM + 1}-{WIN_TO});
  - the {PAGE_MAX:g} pages/s void, not moved after seeing data;
  - the {PASS_AGREE:.0%} pass agreement, B3', and the start refusals (a process of >= {MEM_BIG_GB:g} GB; the
    780M's adapters);
  - the energy method, the predictions and (d).

Changes, and nothing else:
  (a) The GPU witness regex accepts engine types with a space ("Compute 0", "Compute 1", "Timer 0", the
      780M's DirectML engines). Sitting 1's (\\w+) dropped those columns, so its "780M pid" read 0. A
      selftest case has a spaced name. Report-only, as before.
  (b) A page-in attribution witness, report-only.
      - What: every process's page faults, hard faults and IO read bytes, once a second, from one
        NtQuerySystemInformation(SystemProcessInformation) call. These are the kernel counts behind
        \\Process(*)\\Page Faults/sec and \\Process(*)\\IO Read Bytes/sec.
      - Why not the PDH counters: typeperf fixes a wildcard's instances when it starts, so it would miss
        a process launched inside a window. A psutil sweep cost 2.5 s of CPU; this call costs about 11 ms.
      - When: in every idle and every arm-pass alike, so its cost cancels in dP.
      - Logged: for every window over {PAGE_MAX:g} pages/s, the top 5 processes by page faults/s and by IO
        read bytes/s (PAGEIN_ATTRIBUTION). Each entry also carries that process's hard faults/s. The
        verdict prints them beside the voids.
  (c) No accuracy pass. Sitting 1's accuracy log stands (untimed and deterministic), pinned below by its
      sha256. Each sitting-2 arm's greedy tokens (the tokens_sha of every sequence, both passes) must
      equal sitting 1's for that arm, or the arm is VOID (TOKENS_SHA_S1 below). This ties the two sittings
      without a cross-sitting timing ratio.
  Hygiene (the gate's addendum, report-only): the tokenizer loads with transformers' logging at error.
  4.57.6's "incorrect regex pattern" warning, which quoted the local path on 24 lines of sitting 1's and
  the build's logs, therefore no longer reaches a log. The prereg's TOKENIZER_CHECK_JSON showed that the
  warning changes no id.

The logs: gemma_decode_suite_rerun_desktop2_<date>.log (scripts/llm-study.sh decode-rerun) and
gemma_decode_verdict_rerun_desktop2_<date>.log (decode-rerun-verdict, over it and sitting 1's accuracy
log). As in sitting 1: BFP16 holds the machine, and nothing else runs.
"""


def prereg_rerun() -> int:
    print(RERUN)
    out = ROOT / "results" / "llm"
    bad = []
    for name, want in S1_LOGS_LF.items():
        got = sha256_lf(out / name)
        print(f"  {name}: sha256 {sha256(out / name)} (CRLF working copy), {got} (LF blob)")
        if got != want:
            bad.append(f"{name} differs from its pin")
    s1 = [json.loads(s.split(" ", 1)[1]) for s in (out / "gemma_decode_suite_desktop2_20260924.log")
          .read_text(encoding="utf-8").splitlines() if s.startswith("ARM_JSON ")]
    for arm in gd.ARMS:
        got = sorted({q["tokens_sha"] for r in s1 if r["arm"] == arm for q in r["seqs"]})
        print(f"  TOKENS_SHA_S1 {arm}: {TOKENS_SHA_S1[arm]}; sitting 1 had {len(got)} distinct value(s)"
              + (", equal" if got == [TOKENS_SHA_S1[arm]] else f", DIFFERENT {got}"))
        if got != [TOKENS_SHA_S1[arm]]:
            bad.append(f"TOKENS_SHA_S1 {arm} is not sitting 1's single value")
    c0 = combine({(r["pass"], r["arm"]): arm_stats(r) for r in s1})["C0-H4"]
    say("SITTING1_C0_H4_JSON", {k: c0[k] for k in ("state", "tokps", "jtok", "tokps_agree", "jtok_agree") if k in c0})
    ids = pinned_ids()
    print(f"  IDS_SHA {ids_sha(ids)} ({'equal to the pin' if ids_sha(ids) == IDS_SHA else 'DIFFERENT'})")
    if ids_sha(ids) != IDS_SHA:
        bad.append("IDS_SHA")
    p = out / "gemma_decode_prereg_desktop2_20260924.log"
    print(f"  the prereg: {p.name}, sha256 {sha256(p)} (CRLF working copy), {sha256_lf(p)} (LF blob)")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    say("RERUN_CONSTANTS_JSON", {"tokens_sha_s1": TOKENS_SHA_S1, "s1_logs_lf": S1_LOGS_LF, "page_max": PAGE_MAX,
                                 "pass_agree": PASS_AGREE, "order": ORDER, "seqs": SEQS, "git_head": head})
    print("git HEAD", head, "(plus this log's own commit)")
    for b in bad:
        print("PIN_MISMATCH", b)
    return 3 if bad else 0


def selftest() -> int:
    fails = []

    def expect(what, got, want):
        print(f"  {what}: {got}" + ("" if got == want else f"  (expected {want})"))
        if got != want:
            fails.append(what)

    # (d)'s speed rule, the gate's example and the edges
    expect("speed: ceiling 15.1 vs rival 14.5 -> KILL (15.1 < 15.95)", rule_speed(15.1, 14.5), "KILL")
    expect("speed: ceiling 16.0 vs 14.5 -> OPEN", rule_speed(16.0, 14.5), "OPEN")
    expect("speed: exactly 1.10x (16.5 vs 15) -> OPEN", rule_speed(16.5, 15.0), "OPEN")
    expect("speed: a rival faster than the ceiling -> KILL", rule_speed(15.1, 16.0), "KILL")
    expect("energy: E_npu 1.00 vs rival 1.05 -> KILL", rule_energy(1.00, 1.05), "KILL")
    expect("energy: 1.00 vs 1.20 -> OPEN", rule_energy(1.00, 1.20), "OPEN")
    expect("energy: exactly 1.10x (1.00 vs 1.10) -> OPEN", rule_energy(1.00, 1.10), "OPEN")
    # eligibility
    arms = {"A": {"kl": 0.010, "tokps": 30.0, "jtok": 1.0, "head": "H4"},
            "B": {"kl": 0.020, "tokps": 40.0, "jtok": 0.5, "head": "H4"},
            "C": {"kl": 1e-8, "tokps": 8.0, "jtok": 4.0, "head": "H16"},
            "D": {"kl": 5e-7, "tokps": 12.0, "jtok": 3.0, "head": "H16"},
            "E": {"kl": None, "tokps": 90.0, "jtok": 0.1, "head": "H16"}}
    ref, rv = rivals(arms, "H4")
    expect("a faster but less accurate arm is excluded (B at 2x A's KL)", sorted(rv), ["A", "C", "D"])
    ref, rv = rivals(arms, "H16")
    expect("two KLs under the 1e-6 floor tie; a non-finite arm is excluded", sorted(rv), ["C", "D"])
    ref, rv = rivals(arms, "H4", "worst")
    expect("assumed as accurate as the worst H4 arm: B joins", sorted(rv), ["A", "B", "C", "D"])
    expect("at_least_as_accurate at exactly 1.10x", at_least_as_accurate(1.1e-3, 1e-3), True)
    expect("at_least_as_accurate just above 1.10x", at_least_as_accurate(1.1001e-3, 1e-3), False)
    expect("speed: exactly 1.10x in inexact decimals (15.4 vs 14.0) -> OPEN", rule_speed(15.4, 14.0), "OPEN")
    expect("energy: exactly 1.10x in inexact decimals (0.7 vs 0.77) -> OPEN", rule_energy(0.7, 0.77), "OPEN")
    d = {x["head"] + "/" + x["assume"]: x for x in rule_d(arms)}
    expect("H16 generous: fastest rival D 12.0 vs ceiling 15.13 -> OPEN; energy: 1.10 x 1.032 > 3.0? no -> OPEN",
           (d["H16/generous"]["speed"], d["H16/generous"]["energy"]), ("OPEN", "OPEN"))
    expect("H4 worst: B at 40 tok/s beats 21.82 -> KILL; B at 0.5 J/token < 1.10 x 0.716 -> KILL",
           (d["H4/worst"]["speed"], d["H4/worst"]["energy"]), ("KILL", "KILL"))
    expect("the ceilings (H4, H16)", (round(NPU_GBPS / GB_TOKEN["H4"], 2), round(NPU_GBPS / GB_TOKEN["H16"], 2)),
           (21.82, 15.13))
    expect("E_npu (H4, H16)", (round(JGB_NPU * GB_TOKEN["H4"], 3), round(JGB_NPU * GB_TOKEN["H16"], 3)), (0.716, 1.032))
    expect("the energy rule cannot KILL on the rivals' read floors (DirectML H4 1.63 J/token)",
           (rule_energy(JGB_NPU * GB_TOKEN["H4"], 0.748 * GB_TOKEN["H4"]),
            rule_energy(JGB_NPU * GB_TOKEN["H16"], 0.748 * GB_TOKEN["H4"])), ("OPEN", "OPEN"))
    # rows and windows
    rows = [[100.0 + i, 20000.0 + (5000 if 110 <= 100 + i <= 130 else 0), 5000.0, 10.0, 0.0, 0.0, 0.0] for i in range(60)]
    expect("inside: a row stamped t holds (t-1, t]", [r[0] for r in inside(rows, 110.2, 113.0)], [112.0, 113.0])
    head = ["(PDH-CSV 4.0)", r"\\H\GPU Engine(pid_42_luid_0x00000000_0x0000BADF_phys_0_eng_0_engtype_3D)\Utilization Percentage",
            r"\\H\GPU Engine(pid_42_luid_0x00000000_0x5736A8F5_phys_0_eng_0_engtype_Compute)\Utilization Percentage",
            r"\\H\GPU Engine(pid_7_luid_0x00000000_0x0000BADF_phys_0_eng_1_engtype_Compute)\Utilization Percentage"]
    g = gpu_columns([h.lower() for h in head])
    expect("GPU columns by LUID (the 3D LUID is the 780M)", sorted({v[1] for v in g.values() if v[2] == "3d"}),
           ["0x00000000_0x0000badf"])
    # (a): an engine type with a space, as the 780M names its DirectML engines, is a 780M column
    spaced = r"\\H\GPU Engine(pid_42_luid_0x00000000_0x0000BADF_phys_0_eng_2_engtype_Compute 0)\Utilization Percentage"
    expect("(a) a spaced engine type parses (pid, luid, 'compute 0')", list(gpu_columns([spaced.lower()]).values()),
           [(42, "0x00000000_0x0000badf", "compute 0")])
    import tempfile
    from power_probe import CORES, PKG
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.csv"
        cols = ["(PDH-CSV 4.0)", "\\\\H" + PKG, *("\\\\H" + c for c in CORES), "\\\\H" + CPU, "\\\\H" + AVAIL,
                "\\\\H" + PAGES_IN, spaced, head[2]]
        vals = ["09/24/2026 07:00:01.000", "20000", *(["100"] * len(CORES)), "10", "20000", "3", "80", "5"]
        p.write_text(",".join(f'"{c}"' for c in cols) + "\n" + ",".join(f'"{v}"' for v in vals) + "\n", encoding="utf-8")
        r = read_csv(p, 42, ["0x00000000_0x0000badf"])
        expect("(a) read_csv: a 'Compute 0' column counts in 780M all and pid, the NPU's in other",
               r["rows"][0][4:7], [80.0, 80.0, 5.0])
    # (b): the attribution witness, aggregated over a window by inside()'s rule
    fs = FaultSampler(counters=lambda: {})
    fs.rows = [(100.0 + i, {9: ("svc.exe", 50, 40, 4096), 5: ("reader.exe", 1, 0, 0)}) for i in range(10)]
    fs.rows[3][1][11] = ("scan.exe", 900, 800, 1 << 20)
    w = fs.window(101.0, 105.0)
    expect("(b) window: 4 seconds, top faults scan.exe (225/s) then svc.exe", (w["seconds"], [e[0] for e in w["top_page_faults"][:2]],
           w["top_page_faults"][0][2]), (4, ["scan.exe", "svc.exe"], 225.0))
    me = proc_counters().get(os.getpid())
    expect("(b) proc_counters sees this process by name", bool(me) and me[0].lower().startswith("python"), True)
    # (c): sitting 2's tokens must equal sitting 1's, arm by arm
    recs2 = [{"arm": a, "seqs": [{"tokens_sha": TOKENS_SHA_S1[a]}]} for a in gd.ARMS]
    expect("(c) equal tokens: no arm VOID", tokens_void(recs2), {})
    recs2[1]["seqs"].append({"tokens_sha": "0" * 64})
    expect("(c) one differing sequence VOIDs its arm only", sorted(tokens_void(recs2)), [recs2[1]["arm"]])
    # arm_stats and combine on synthetic records
    def rec(pas, arm, tokps, dpw):
        ep = gd.ARMS[arm][0]
        seqs, t = [], 1000.0
        for s in range(SEQS[ep]):
            w = WIN_TOKENS / tokps
            seqs.append({"seq": s, "generated": GEN_TOKENS, "tokens_sha": "x", "win_s": w, "tokps": tokps,
                         "win_wall": [t, t + w]})
            t += w + 30
        rows = [[1000.0 + i, 15000.0 + dpw * 1000, 4000.0, 20.0, 0.0, 0.0, 0.0, 9000.0, 3.0]
                for i in range(int(t - 1000) + 5)]
        return {"pass": pas, "arm": arm, "ready": {"pin_ok": True}, "seqs": seqs,
                "idle": {"rows": [[i, 15000.0, 3000.0, 1.0, 0.0, 0.0, 0.0, 9000.0, 2.0] for i in range(60)]},
                "counters": {"rows": rows}}
    st = arm_stats(rec(1, "C4-H4", 16.0, 30.0))
    expect("arm_stats: J/token = dP / tok/s (30 W at 16 tok/s)", round(st["jtok"], 4), round(30.0 / 16.0, 4))
    stats = {(p, a): arm_stats(rec(p, a, 16.0 if p == 1 else 17.0, 30.0)) for p in (1, 2) for a in gd.ARMS}
    c = combine(stats)
    expect("combine: passes 16 and 17 tok/s agree within 10%", c["C4-H4"]["state"], "OK")
    stats[(2, "D-H4")] = arm_stats(rec(2, "D-H4", 25.0, 30.0))
    expect("combine: passes 16 and 25 tok/s disagree -> INCOMPLETE", combine(stats)["D-H4"]["state"], "INCOMPLETE")
    short = rec(1, "C0-H4", 10.0, 30.0)
    short["seqs"][1]["generated"] = 500
    expect("a short sequence voids the arm-pass", arm_stats(short)["ok"], False)
    paged = rec(1, "D-H16", 20.0, 30.0)
    a, b = paged["seqs"][2]["win_wall"]
    for r in paged["counters"]["rows"]:
        if a + 1 <= r[0] <= b:
            r[8] = 150.0
    expect("the memory witness: one window at 150 pages/s voids the arm-pass", arm_stats(paged)["ok"], False)
    expect("... and at 3 pages/s it stands", arm_stats(rec(1, "D-H16", 20.0, 30.0))["ok"], True)
    # B3' on a synthetic graph: the mask subgraph feeding every GQA node's input 6
    import tempfile
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    def graph(extra_consumer: bool):
        gth, cst = B3_CPU_ALLOWED
        nodes = [helper.make_node("Shape", ["attention_mask"], ["s"], name="/model/attn_mask_reformat/attn_mask_subgraph/Shape"),
                 helper.make_node("Constant", [], ["/model/constants/INT64/1"],
                                  value=numpy_helper.from_array(np.array(1, dtype=np.int64))),
                 helper.make_node("Gather", ["s", "/model/constants/INT64/1"], [gth + "/output_0"], name=gth, axis=0),
                 helper.make_node("Cast", [gth + "/output_0"], [cst + "/output_0"], name=cst, to=TensorProto.INT32)]
        for layer in range(gd.LAYERS):
            nodes.append(helper.make_node("GroupQueryAttention", ["q", "k", "v", "", "", "sl", cst + "/output_0"],
                                          [f"o{layer}"], name=f"/model/layers.{layer}/attn/GQA", domain="com.microsoft"))
        if extra_consumer:
            nodes.append(helper.make_node("Identity", [cst + "/output_0"], ["x"], name="/extra"))
        g = helper.make_graph(nodes, "b3", [helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, None])],
                              [helper.make_tensor_value_info("o0", TensorProto.FLOAT, None)])
        return helper.make_model(g)

    row = {"status": "CPU_NODES", "cpu_nodes": [f"Gather ({B3_CPU_ALLOWED[0]})", f"Cast ({B3_CPU_ALLOWED[1]})"],
           "placed": {"DmlExecutionProvider": 100, "CPUExecutionProvider": 2}}
    real_onnx = gd.ONNX
    with tempfile.TemporaryDirectory() as td:
        gd.ONNX = Path(td)
        try:
            for arm, extra in (("ok", False), ("extra", True)):
                (Path(td) / arm).mkdir()
                onnx.save(graph(extra), str(Path(td) / arm / "model.onnx"))
            expect("B3': only the two named producers of GQA input 6 -> PASS", b3_prime("ok", row)["ok"], True)
            expect("B3': the Cast feeding another node -> VOID", b3_prime("extra", row)["ok"], False)
            third = dict(row, cpu_nodes=row["cpu_nodes"] + ["MatMulNBits (/model/layers.0/mlp/MatMul)"],
                         placed={"DmlExecutionProvider": 99, "CPUExecutionProvider": 3})
            expect("B3': a third CPU node -> VOID", b3_prime("ok", third)["ok"], False)
            expect("B3': an unread placement -> VOID", b3_prime("ok", {"status": "UNPARSED"})["ok"], False)
        finally:
            gd.ONNX = real_onnx
    print("SELFTEST", "FAIL " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("prereg", "prereg-rerun", "suite", "accuracy", "verdict", "posthoc-gpu", "reader",
                                     "selftest"))
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--arm", choices=tuple(gd.ARMS))
    ap.add_argument("--pass", dest="pas", type=int, choices=(1, 2))
    ap.add_argument("--rerun", action="store_true", help="verdict: sitting 2's rule (tokens_sha against sitting 1's)")
    a = ap.parse_args()
    if a.mode == "reader":
        return reader(a.arm, a.pas)
    if a.mode == "verdict":
        return verdict(Path(a.logs[0]), Path(a.logs[1]), a.rerun)
    if a.mode == "prereg-rerun":
        return prereg_rerun()
    if a.mode == "posthoc-gpu":
        return posthoc_gpu(Path(a.logs[0]), Path(a.logs[1]), a.logs[2])
    if a.mode == "prereg":
        return prereg(Path(a.logs[0]), Path(a.logs[1]))
    return {"suite": suite, "accuracy": accuracy, "selftest": selftest}[a.mode]()


if __name__ == "__main__":
    sys.exit(main())
