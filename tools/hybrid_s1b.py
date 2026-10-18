#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid stack, S1b: int8 per token (N2) serves the prompt's KV, and R0's continuation decides. CPU only, no
NPU and no GPU. The plan is PREREG below; the rules are this file's verdict code (VERDICT_FUNCS, the imported S1
kernels in S1_VERDICT_FUNCS, verdict_constants()), frozen by VERDICT_CODE_SHA256.

S1's built models are reused, read-only (tools/hybrid_s1.py, frozen at c67d6d2): R0 (the reference), N2 (the
deciding arm), N3 and R (report-only). Each window is [BOS] + 3,071 new tokens of WikiText-2, after S1's span:
  the arm runs the 2,048-id prompt and keeps its KV; R0 runs ids 2,048-3,071 on that KV (set C, deciding) and
  id 2,047 on the KV sliced to 0-2,046 (P-R0); the arm's own logits at 2,047 are P-arm. R0 one-shot over all
  3,072 ids is the reference.

    python tools/hybrid_s1b.py selftest   # synthetic checks: no model, no chip
    python tools/hybrid_s1b.py prereg     # the plan, the 84 windows, PROTOCOL_JSON, the frozen hashes, S1's blob
    python tools/hybrid_s1b.py check      # the model pins, C5' and the R0 control (children)
    python tools/hybrid_s1b.py states     # the arms' prompts and R0's continuations, in batches (children)
    python tools/hybrid_s1b.py verdict    # the metrics and the frozen verdict
Every heavy child holds one session (or the head), refuses below S1's MIN_AVAIL_GB available, and ends itself
(rc 4) if the available memory falls below S1's WATCH_MIN_GB while it runs (S1's amendment 1).
"""
import argparse
import inspect
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gemma_decode_suite as gds  # noqa: E402
import hybrid_s1 as s1  # noqa: E402

WORK = ROOT / "scratch/llm/hybrid_s1b"
CHECK = WORK / "check"
STATES = WORK / "states"
KVDIR = WORK / "kv"
MODELS = s1.MODELS                                       # S1's built models, read-only
RESULTS = s1.RESULTS

# ---------------------------------------------------------------- the protocol (pre-registered)

ARMS = ("N2", "N3", "R")                                 # the arms whose prompt KV R0 continues
DECIDING = ("N2",)
REPORT_ONLY = ("N3", "R")
BOS, PROMPT = s1.BOS, s1.SEQ_LEN                         # 2, 2,048
S1_END = s1.STRIDE * s1.N_SEQ                            # 2,047 x 16 = 32,752: S1's span is T[0 : S1_END]
WIN_LEN = 3072                                           # [BOS] + 3,071 tokens
WIN_STRIDE = WIN_LEN - 1
N_WIN = 84
SET_C = (2048, 3070)                                     # positions i (the logits after ids[0..i]), inclusive
P_POS = 2047
BANDS = s1.BANDS                                         # L (0-1,023), H (1,024-2,046): the replication
Q2_SPLIT = 2560                                          # Q2: positions 2,048-2,559 against 2,560-3,070
CONTROL_WINDOWS = (0, 12, 24, 36, 48, 60, 72, 83)
CONTROL_PATHS = ("a", "b", "c")
CONTROL_KL_MAX = s1.C6_KL_MAX                            # 1e-5, S1's C6 limit
T_KL, T_TOP1 = s1.T_KL, s1.T_TOP1                        # 0.0123, 0.956: (c)'s C0-H4, carried to set C unchanged
SEED, BOOT_N, CHUNK = s1.SEED, s1.BOOT_N, s1.CHUNK
BATCH = 8
MIN_DISK_GB = 40.0                                       # states: the batch's KV (13.7 GB) and the states (7.3 GB)
S1_COMMIT = "c67d6d2"
S1_BUILD_LOG = "hybrid_s1_build_desktop2_20260924_r2.log"
MODEL_PIN = {                                            # the r2 BUILD_JSON's files, and S1's SRC_PIN for base
    "R0.onnx": (432992, "a0a7d785e8dafb387189d76e2e3aedcd4425a26a130ec0142b1e13e0049b1904"),
    "R.onnx": (432992, "b960a9193a9ab3e19fe1d823bf828ae52523eb60a3349da27011bc6e7ee1b94e"),
    "N2.onnx": (903332, "7ac950ad4978205e3afcdde62674b09b8efc0787374c4c456810b9b8b626d1ac"),
    "N3.onnx": (462635, "c3ad9f64d469bf7c8c148062bdc26c5d1e9269579049c8e33764d78a63abfef6"),
    "int8.onnx.data": (3216719872, "ceef81a2b70f4651b83bae28e04d18edcddebb26bc409492653cf477de338a5b"),
    "bf16.onnx.data": (6417285120, "5499847b8e9abe29d0ccf8b645424367a961a180d23d44d46adef0570978a1be"),
    "base.onnx.data": s1.SRC_PIN["model.onnx.data"],
}
PICK = "N3 (S1's pick; S1b does not change it: any change to the hybrid's NPU numerics is the user's decision)"
PREDICTIONS = [
    ("Q1", "set C (deciding): N2's mean KL <= 0.0123, its top-1 within 0.956 +/- 0.01, and set C NARROW on top-1; "
           "the side of the line is not predicted"),
    ("Q2", "set C: N2's mean KL over positions 2,048-2,559 exceeds its mean over 2,560-3,070"),
    ("Q3", "P-arm (report-only): N2 FAILS"),
    ("Q4", "N2's mean KL orders as set C < P-R0 < P-arm"),
    ("Q5", "N3 PASSES set C, and reads within both thresholds on P-arm and P-R0"),
    ("Q6", "R PASSES set C"),
    ("Q7", "the R0 control passes all three paths on all 8 windows, with paths (a) and (c) sha-equal on every "
           "window; path (b)'s sha-equality is not predicted"),
    ("Q8", "the replication FAILS N2 again in bands L and H, with mean KL in 0.04-0.08 and top-1 in 0.88-0.93 in "
           "each"),
]


def protocol() -> dict:
    return {"stage": "hybrid S1b", "plan": "v2 (scratch sha256 1da1e402)", "arms": ARMS, "deciding": DECIDING,
            "report_only": REPORT_ONLY, "reference": "R0 one-shot over the window's 3,072 ids",
            "bos": BOS, "prompt": PROMPT, "s1_end": S1_END, "win_len": WIN_LEN, "win_stride": WIN_STRIDE,
            "n_win": N_WIN, "set_c": SET_C, "p_pos": P_POS, "bands": BANDS, "q2_split": Q2_SPLIT,
            "control_windows": CONTROL_WINDOWS, "control_paths": {
                "a": "R0's own full-prompt KV, then R0's continuation, against the one-shot at 2,048-3,070",
                "b": "R0's KV sliced to 0-2,046, then one decode step for id 2,047, against the one-shot at 2,047",
                "c": "R0's 2,048-id prompt hidden states against the one-shot's first 2,048 positions"},
            "control_kl_max": CONTROL_KL_MAX, "t_kl": T_KL, "t_top1": T_TOP1, "seed": SEED, "boot_n": BOOT_N,
            "chunk": CHUNK, "batch": BATCH, "min_avail_gb": s1.MIN_AVAIL_GB, "watch_min_gb": s1.WATCH_MIN_GB,
            "min_disk_gb": MIN_DISK_GB, "session": s1.SESSION, "text_pin": gds.TEXT_PIN,
            "s1_commit": S1_COMMIT, "s1_build_log": S1_BUILD_LOG, "model_pin": MODEL_PIN,
            "saved": {"N2": "all 2,048 prompt positions", "N3": "position 2,047", "R": "position 2,047"},
            "pick": PICK}


# ---------------------------------------------------------------- the text (84 new windows)

def offsets(n: int = N_WIN, start: int = S1_END, stride: int = WIN_STRIDE, length: int = WIN_LEN - 1) -> list:
    """Window w takes T[start + stride*w : start + stride*w + length], after its BOS (stride = length here)."""
    return [(start + stride * w, start + stride * w + length) for w in range(n)]


def window_asserts(offs: list, n_tokens: int, s1_end: int, prompt_shas: list, s1_seq_shas: list) -> dict:
    """The three non-overlap asserts, and the text's bound."""
    after = all(a >= s1_end for a, _ in offs)
    so = sorted(offs)
    disjoint = all(so[i][1] <= so[i + 1][0] for i in range(len(so) - 1)) and all(a < b for a, b in so)
    inside = all(b <= n_tokens for _, b in offs)
    no_s1 = not (set(prompt_shas) & set(s1_seq_shas))
    return {"start_after_s1": after, "pairwise_disjoint": disjoint, "inside_text": inside,
            "no_s1_prompt_sha": no_s1, "ok": after and disjoint and inside and no_s1}


def windows(T: list) -> list:
    wins = [[BOS] + T[a:b] for a, b in offsets()]
    assert all(len(w) == WIN_LEN for w in wins)
    return wins


def text_pins(T: list) -> dict:
    offs = offsets()
    wins = windows(T)
    s1_seq = s1.text_pins(T)["seq_sha"]
    prompt_sha = [gds.ids_sha(w[:PROMPT]) for w in wins]
    return {"text_tokens": len(T), "s1_end": S1_END, "offsets": offs,
            "window_sha": [gds.ids_sha(w) for w in wins], "prompt_sha": prompt_sha,
            "all_sha": gds.ids_sha([i for w in wins for i in w]), "first": offs[0][0], "end": offs[-1][1],
            "left": len(T) - offs[-1][1],
            "asserts": window_asserts(offs, len(T), S1_END, prompt_sha, s1_seq)}


def pinned_windows() -> list:
    """The ids, checked against the committed S1b prereg log's TEXT_JSON."""
    T = s1.text_tokens()
    got = text_pins(T)
    want = s1.logged_json(s1.latest("hybrid_s1b_prereg_*.log"), "TEXT_JSON")
    for k in ("text_tokens", "offsets", "window_sha", "prompt_sha", "all_sha"):
        if json.loads(json.dumps(got[k])) != want[k]:
            sys.exit(f"the ids differ from the S1b prereg's TEXT_JSON ({k})")
    if not got["asserts"]["ok"]:
        sys.exit("the windows fail their asserts")
    return windows(T)


# ---------------------------------------------------------------- the metrics (frozen: VERDICT_FUNCS)

def slice_kv(kv: dict, n: int) -> dict:
    """The past KV kept to its first n positions (axis 2 of [1, kv_heads, positions, head_dim])."""
    return {k: np.ascontiguousarray(v[:, :, :n, :]) for k, v in kv.items()}


def window_metrics(ho: np.ndarray, arms: dict, ids, head: np.ndarray) -> dict:
    """One window, every arm, against R0 one-shot `ho` [3,072, hidden]. arms[a] holds "prompt" (all 2,048 rows,
    or only row 2,047 when "prompt_rows" is "last"), "cont" (R0's continuation on the arm's KV, 1,024 rows from
    position 2,048) and "pr0" (R0's decode step at 2,047 on the KV sliced to 0-2,046, 1 row).
    Returns, per arm, "C", "P_arm", "P_R0" and, for a full prompt, "L" and "H": kl, top1, nll, finite."""
    out = {a: {} for a in arms}

    def score(lo: int, hi: int, get: dict, key: str) -> None:
        acc = {a: {"kl": [], "top1": [], "nll": [], "finite": True} for a in get}
        for r in range(lo, hi + 1, CHUNK):
            e = min(r + CHUNK, hi + 1)
            nxt = np.asarray(ids[r + 1:e + 1])
            z0 = s1.logits_chunk(ho[r:e], head)
            lp0 = s1.log_softmax_rows(z0)
            p0 = np.exp(lp0)
            top0 = np.argmax(z0, axis=1)
            for a, g in get.items():
                h = g(r, e)
                k = s1.kl_top1_chunk(lp0, p0, top0, s1.logits_chunk(h, head), nxt)
                acc[a]["finite"] &= k["finite"] and bool(np.isfinite(h).all())
                for f in ("kl", "top1", "nll"):
                    acc[a][f].append(k[f])
        for a, v in acc.items():
            out[a][key] = {f: np.concatenate(v[f]) for f in ("kl", "top1", "nll")} | {"finite": v["finite"]}

    score(SET_C[0], SET_C[1], {a: (lambda r, e, a=a: arms[a]["cont"][r - SET_C[0]:e - SET_C[0]]) for a in arms}, "C")
    score(P_POS, P_POS, {a: (lambda r, e, a=a: arms[a]["prompt"][-1:] if arms[a]["prompt_rows"] == "last"
                             else arms[a]["prompt"][r:e]) for a in arms}, "P_arm")
    score(P_POS, P_POS, {a: (lambda r, e, a=a: arms[a]["pr0"][0:1]) for a in arms}, "P_R0")
    full = {a: (lambda r, e, a=a: arms[a]["prompt"][r:e]) for a in arms if arms[a]["prompt_rows"] == "all"}
    if full:
        for band, (lo, hi) in BANDS.items():
            score(lo, hi, full, band)
    return out


def set_aggregate(per_win: list) -> dict:
    """per_win: one {kl, top1, nll, finite} per window, for one arm and set. S1's aggregate: pooled means, the
    bootstrap over windows."""
    kl = np.stack([w["kl"] for w in per_win])
    t1 = np.stack([w["top1"] for w in per_win])
    nl = np.stack([w["nll"] for w in per_win])
    return s1.aggregate(kl, t1, nl) | {"finite": all(w["finite"] for w in per_win), "windows": len(per_win)}


def halves(per_win: list) -> list:
    """Q2: set C's mean KL over positions 2,048-2,559, then over 2,560-3,070."""
    kl = np.stack([w["kl"] for w in per_win])
    cut = Q2_SPLIT - SET_C[0]
    return [float(kl[:, :cut].mean()), float(kl[:, cut:].mean())]


def rule_set(a: dict) -> tuple:
    """PASS iff mean KL <= T_KL and top-1 >= T_TOP1; FAIL otherwise, or on a non-finite value. NARROW if the 95%
    bootstrap interval holds a threshold: reported, it changes nothing. The point estimate decides."""
    if not a["finite"]:
        return "FAIL", False
    ok = a["kl_mean"] <= T_KL and a["top1"] >= T_TOP1
    narrow = a["kl_ci"][0] <= T_KL <= a["kl_ci"][1] or a["top1_ci"][0] <= T_TOP1 <= a["top1_ci"][1]
    return ("PASS" if ok else "FAIL"), narrow


def control_outcome(rows: list) -> dict:
    """rows: one {"w", "a", "b", "c"} per control window, each path {"max_kl", "sha_equal"}. PASS iff every
    control window is present and every path's max KL is finite and <= CONTROL_KL_MAX."""
    got = {r["w"] for r in rows}
    failed = [[r["w"], p] for r in rows for p in CONTROL_PATHS
              if not (np.isfinite(r[p]["max_kl"]) and r[p]["max_kl"] <= CONTROL_KL_MAX)]
    missing = sorted(set(CONTROL_WINDOWS) - got)
    return {"outcome": "PASS" if not failed and not missing else "FAIL", "failed": failed, "missing": missing,
            "sha_equal": {p: all(r[p]["sha_equal"] for r in rows) for p in CONTROL_PATHS} if rows else {}}


def outcome_n2(metrics: dict, complete: bool, c5: str) -> dict:
    """N2 on set C only. C5' failing or a missing window leaves N2 INCOMPLETE."""
    if c5 != "PASS" or not complete or "C" not in metrics:
        return {"outcome": "INCOMPLETE", "narrow": False}
    o, nb = rule_set(metrics["C"])
    return {"outcome": o, "narrow": nb}


def score_predictions(metrics: dict, control: dict, n2_halves: list) -> dict:
    def s(cond, need):
        if not all(need):
            return "NOT SCORED"
        return "HIT" if cond() else "MISS"
    m2, m3, mr = metrics.get("N2", {}), metrics.get("N3", {}), metrics.get("R", {})
    have = lambda m, *ks: all(k in m for k in ks)  # noqa: E731
    return {
        "Q1": s(lambda: m2["C"]["kl_mean"] <= T_KL and abs(m2["C"]["top1"] - T_TOP1) <= 0.01
                and m2["C"]["top1_ci"][0] <= T_TOP1 <= m2["C"]["top1_ci"][1], [have(m2, "C")]),
        "Q2": s(lambda: n2_halves[0] > n2_halves[1], [n2_halves is not None]),
        "Q3": s(lambda: rule_set(m2["P_arm"])[0] == "FAIL", [have(m2, "P_arm")]),
        "Q4": s(lambda: m2["C"]["kl_mean"] < m2["P_R0"]["kl_mean"] < m2["P_arm"]["kl_mean"],
                [have(m2, "C", "P_R0", "P_arm")]),
        "Q5": s(lambda: all(rule_set(m3[k])[0] == "PASS" for k in ("C", "P_arm", "P_R0")),
                [have(m3, "C", "P_arm", "P_R0")]),
        "Q6": s(lambda: rule_set(mr["C"])[0] == "PASS", [have(mr, "C")]),
        "Q7": s(lambda: control["outcome"] == "PASS" and control["sha_equal"]["a"] and control["sha_equal"]["c"],
                [bool(control.get("sha_equal"))]),
        "Q8": s(lambda: all(rule_set(m2[b])[0] == "FAIL" and 0.04 <= m2[b]["kl_mean"] <= 0.08
                            and 0.88 <= m2[b]["top1"] <= 0.93 for b in BANDS), [have(m2, *BANDS)]),
    }


VERDICT_FUNCS = ("offsets", "window_asserts", "slice_kv", "window_metrics", "set_aggregate", "halves", "rule_set",
                 "control_outcome", "outcome_n2", "score_predictions")
S1_VERDICT_FUNCS = ("log_softmax_rows", "logits_chunk", "kl_top1_chunk", "band_rows", "bootstrap", "aggregate",
                    "max_kl")


def verdict_constants() -> dict:
    return {"ARMS": ARMS, "DECIDING": DECIDING, "REPORT_ONLY": REPORT_ONLY, "BOS": BOS, "PROMPT": PROMPT,
            "S1_END": S1_END, "WIN_LEN": WIN_LEN, "WIN_STRIDE": WIN_STRIDE, "N_WIN": N_WIN, "SET_C": SET_C,
            "P_POS": P_POS, "BANDS": BANDS, "Q2_SPLIT": Q2_SPLIT, "CONTROL_WINDOWS": CONTROL_WINDOWS,
            "CONTROL_PATHS": CONTROL_PATHS, "CONTROL_KL_MAX": CONTROL_KL_MAX, "T_KL": T_KL, "T_TOP1": T_TOP1,
            "SEED": SEED, "BOOT_N": BOOT_N, "CHUNK": CHUNK, "MODEL_PIN": MODEL_PIN, "PICK": PICK,
            "TEXT_PIN": gds.TEXT_PIN, "S1_COMMIT": S1_COMMIT}


def verdict_sources() -> list:
    """(name, source) for every function the verdict code is: S1b's own, then the imported S1 kernels.
    inspect.getsource reads with universal newlines, so CRLF and LF checkouts agree."""
    g = globals()
    return [(f, inspect.getsource(g[f])) for f in VERDICT_FUNCS] + \
           [(f"hybrid_s1.{f}", inspect.getsource(getattr(s1, f))) for f in S1_VERDICT_FUNCS]


def verdict_code_sha() -> str:
    src = "".join(s for _, s in verdict_sources())
    return s1.sha_bytes((src + json.dumps(verdict_constants(), sort_keys=True)).encode("utf-8"))


def hashes() -> dict:
    return {"PREREG_TEXT_SHA256": s1.sha_bytes(PREREG.encode("utf-8")),
            "PROTOCOL_JSON_SHA256": s1.sha_bytes(json.dumps(protocol(), sort_keys=True).encode("utf-8")),
            "VERDICT_CODE_SHA256": verdict_code_sha()}


def print_hashes() -> None:
    for k, v in hashes().items():
        print(f"{k} {v}", flush=True)


# ---------------------------------------------------------------- the plan (PREREG)

PREREG = r"""S1b PLAN AND PREREG: int8 per token (N2) serves the prompt's KV, and the continuation decides.
Plan v2 (2026-09-24), accepted by the gate as the plan text. CPU emulation throughout: no NPU and no GPU runs.
Tags: MEASURED (S1's committed logs), DERIVED (arithmetic on them), ESTIMATE. GB = 1e9 bytes.

## 0. Why S1b (disclosed)

- S1 (closed at c67d6d2) failed N2 at every prompt position, against R0 on 16 prompts: band L mean KL 0.06077,
  top-1 0.8949; band H 0.05204, 0.9107 (MEASURED).
- S1's band C was report-only: 4 sequences, the arm's prompt KV, then R0 over positions 2,048-3,070. It read N2
  at mean KL 0.00962 and top-1 0.9580, 0.002 above the top-1 line (MEASURED).
- That band-C reading motivated S1b. It is 4 sequences with no interval, and it upgrades nothing. S1b asks the
  question properly, on new text.

## 1. The question

The user approved this configuration: the NPU only builds the prompt's KV, and the GPU generates exactly. N2's
arithmetic would serve the prompt: the NPU runs the seven weight GEMMs per layer, and the 780M runs attention
(building the KV from N2's k and v projections) and the generation (design v3). Does the generation then stay
within S1's bar?
- Set C, DECIDING: the continuation, positions 2,048-3,070. R0 runs teacher-forced on the arm's full
  2,048-position prompt KV, against R0 one-shot.
- Set P, REPORT-ONLY: the last prompt position (2,047), measured both ways:
  - P-arm: the arm's own logits at 2,047;
  - P-R0: R0 on the arm's KV sliced to positions 0-2,046, with one decode step for id 2,047. This is exact under
    causal attention: the arm's KV at positions 0-2,046 does not depend on id 2,047, so the slice equals the KV
    of a 2,047-id prompt. The arms' activation scales keep it so: per token (N2), per element (N3), per 32-block
    of a row (R); a per-tensor scale (N1) would not. The R0 control checks the slice path.
- The consequence: a set C PASS with a P-arm FAIL means N2 can serve the prompt's KV only if the GPU computes
  the last prompt position (the NPU prefills 0-2,046, and the GPU runs id 2,047 as its first decode step). P-R0
  is that hybrid's first token, report-only here; that reading goes to the user as the gate's design note.

## 2. Text: new windows only

- WikiText-2 raw test, the same pin, tokenizer and joined split as S1: 292,282 tokens (S1's TEXT_JSON).
- S1's span is T[0 : S1_END], S1_END = STRIDE x N_SEQ = 2,047 x 16 = 32,752 (imported from S1). S1's band C
  (T[0 : 9,212]) lies inside it.
- Window w (w = 0..83) is [BOS] + T[32,752 + 3,071w : 32,752 + 3,071(w+1)]: 3,072 ids. The prompt is ids 0-2,047,
  the continuation ids 2,048-3,071. 84 whole windows end at 290,716 and leave 1,566 tokens (DERIVED).
- TEXT_JSON prints every offset and every window's and prompt's sha, and three asserts: every window starts at
  or after 32,752; the windows are pairwise disjoint; no prompt's 2,048 ids hash to any of S1's seq_sha. The
  prereg stops if any fails. Every later stage checks its ids against this TEXT_JSON.

## 3. The measure

- The arm runs the whole 2,048-id prompt in one session and keeps its KV. It saves the hidden states it is scored
  on: all 2,048 positions for N2 (the L/H replication), position 2,047 for N3 and R.
- Set C: R0 runs ids 2,048-3,071 with the arm's full KV as past; its states at 2,048-3,070 are scored (3,071 has
  no next id in the window, as in S1's band C). 1,023 positions per window.
- P-R0: R0 runs id 2,047 with the arm's KV sliced to 0-2,046 as past. 1 position per window; P-arm likewise.
- The reference is R0 one-shot over all 3,072 ids. Its first 2,048 positions are the reference for P-arm, P-R0
  and the L/H replication; control (c) checks them against R0's own 2,048-id prompt run.
- The metrics are S1's frozen kernels (imported; their sources are in VERDICT_CODE): KL(ref || arm) in float64
  over all 262,144 entries, top-1 agreement, the arm's perplexity on the next ids, the fp32 head (the release's
  F16 token_embd upcast, its sha checked against S1's r2 build log) applied in the verdict. Per set: the pooled
  mean, p50, p99 and max, and a 95% bootstrap interval over the per-window means (10,000 resamples, seed
  20260924).
- Scope: (a) R0 stands in for the generation's numerics; the 780M's runtime is S0's subject. (b) The arm's KV is
  the hybrid's KV in form: k and v projected in the arm's numerics, attention in the reference graph; the 780M's
  own attention precision is not modelled. (c) The head is fp32 here. (d) CPU emulation; N2's tie to the NPU is
  DERIVED (S1's wording).

## 4. The arms and the control

- N2: DECIDING, on set C.
- N3: control, REPORT-ONLY. S1's PASS arm through the same prompt -> R0 path.
- R: REPORT-ONLY. All 84 windows.
- N1: DROPPED (S1: mean KL 3.5 at prompt positions, 0.570 in band C; no branch of the pick uses it).
- The replication, REPORT-ONLY: N2's bands L (0-1,023) and H (1,024-2,046) on the new windows, against the
  one-shot's prompt positions. It re-reads S1's FAIL on new text and decides nothing.
- The R0 control, in the check stage before any states, on windows 0, 12, 24, 36, 48, 60, 72 and 83:
  (a) R0's own full-prompt KV, then R0's continuation, against the one-shot at 2,048-3,070;
  (b) R0's KV sliced to 0-2,046 plus one decode step for id 2,047, against the one-shot at 2,047;
  (c) R0's 2,048-id prompt hidden states against the one-shot's first 2,048 positions.
  Each path is held to S1's C6 limit, max per-position KL <= 1e-5, with the sha-equality of the compared rows
  reported beside it. Any failure is a STOP: a pipeline defect, not a verdict.
  For the record: S1's C6 was 1 sequence (R0 chunked against one-shot), within the same limit. Band C's R0 row
  was 4 sequences, 4,092 positions, with a mean of exactly 0.0 (BAND_C_JSON). It was not C6 on 4 sequences.
- C5': N2 runs window 0 twice, and the two prompt states' sha256 must match. A failure leaves N2 INCOMPLETE (as
  S1's C5), and the states stage does not run.

## 5. The rule

- Thresholds, S1's, unchanged: mean KL <= 0.0123 and top-1 >= 0.956. They were anchored on prompt positions
  ((c)'s C0-H4), and are carried to set C unchanged.
- Set C decides N2: PASS if both hold on set C, FAIL otherwise, and FAIL on a non-finite value. NARROW names a
  threshold inside the 95% bootstrap interval; it is reported and changes nothing. The point estimate decides.
- Report-only, with the same PASS or FAIL wording marked report-only: P-arm and P-R0 for N2, N3 and R (set P's
  top-1 moves in steps of 1/84: 81/84 = 0.964 passes, 80/84 = 0.952 fails; its KL is one heavy-tailed position
  per window); set C for N3 and R; N2's bands L and H.
- INCOMPLETE: any window missing from set C for N2 after one re-run of its child, or C5' failing. rc 4 (the
  memory gate) is not re-run: the stage stops, "memory gate; not a verdict", as in S1's amendment 1.
- STOP: the R0 control fails any path on any of its windows; a frozen hash, model sha or text pin differs; or
  tools/hybrid_s1.py's blob differs from c67d6d2's.
- The verdict names N2 PASS, FAIL or INCOMPLETE on set C, with NARROW, and reports P both ways. It does NOT
  change the pick: S1's pick, N3, stands. Any change to the hybrid's NPU numerics is the user's decision, made
  with the interval in hand.

## 6. N = 84

- Every whole window. A re-run is a fresh sitting on the same pre-registered windows, so a reserve has no use.
- The spread, DERIVED from S1's published intervals (half-width x sqrt(16) / 1.96): N2's per-sequence top-1 SD is
  about 0.013 (band L) and 0.012 (band H); its per-sequence KL SD about 0.010, a CV near 17%.
- Set C's top-1 (ESTIMATE, assuming the continuation's per-window spread is like the prompt bands'): the SE at 84
  windows lies between the binomial floor sqrt(0.958 x 0.042 / 1,023) / sqrt(84) and the spread 0.0125 /
  sqrt(84), 0.0007-0.0014, so the 95% half-width is 0.0013-0.0027. Band C's gap to the line is 0.002, so a true
  value near band C's would likely be NARROW even at 84 windows; 84 is all the text allows.
- Set P (report-only): at a true top-1 of 0.91, the binomial SE is 0.031, a half-width of about 0.06.

## 7. Predictions (they decide nothing; PREDICTIONS_JSON scores them mechanically)

- Q1 (set C): N2's mean KL <= 0.0123 (band C read 0.00962); its top-1 within 0.956 +/- 0.01; set C NARROW on
  top-1. The side of the line is not predicted.
- Q2 (set C): N2's KL falls with position: its mean over 2,048-2,559 exceeds its mean over 2,560-3,070. The
  local layers' 1,024 window holds fewer of the arm's KV positions as the continuation grows; the global layers
  hold all of them.
- Q3 (P, report-only): N2 FAILS P-arm (S1's band H top-1 0.9107 at prompt positions).
- Q4: N2's mean KL orders as set C < P-R0 < P-arm. P-R0's past is all the arm's KV, like set C's first position,
  while set C's mean runs over positions whose local window holds less of it; P-arm adds N2's own GEMMs at the
  scored position.
- Q5: N3 PASSES set C, and reads within both thresholds on P-arm and P-R0.
- Q6: R PASSES set C.
- Q7: the R0 control passes all three paths on all 8 windows; (a) and (c) sha-equal on every window, as band C's
  R0 row (exactly 0.0) suggests; (b) within 1e-5, its sha-equality not predicted (a 1-id step may take a
  different MatMul path).
- Q8: the replication FAILS N2 again in bands L and H, with mean KL in 0.04-0.08 and top-1 in 0.88-0.93 in each.
- Overall: N2's set C verdict is predicted NARROW, its side not predicted. The pick stays N3 either way.

## 8. Checks

- tools/hybrid_s1.py: every stage prints its blob (git hash-object) beside c67d6d2's (git rev-parse
  c67d6d2:tools/hybrid_s1.py) and stops unless they are equal. The prereg prints S1b's VERDICT_CODE and the list
  of the imported S1 sources it hashes, each with its own sha256.
- Models: S1's R0.onnx, R.onnx, N2.onnx and N3.onnx and their data files (int8.onnx.data, bf16.onnx.data,
  base.onnx.data) are reused, not rebuilt. MODEL_PIN holds their sizes and shas; the prereg checks MODEL_PIN
  against the r2 build log's BUILD_JSON (and base.onnx.data against S1's SRC_PIN), and check and states hash
  every file against MODEL_PIN before any session.
- Text: the section 2 asserts, and TEXT_JSON against this prereg in every later stage.
- Memory: every heavy child gets S1's 15 GB start gate and 5 GB watchdog (rc 4 is not a verdict). One model per
  child. PRIORITY_BASED sessions (S1's session_options). The states stage refuses below 40 GB free disk.
- Completeness: 84 windows x each arm's P-arm state, P-R0 state and continuation, and the reference, each pinned
  by sha256 in the states log. The KV is deleted after each batch, and the log says so.

## 9. Tool, runs and logs

- tools/hybrid_s1b.py imports tools/hybrid_s1.py read-only. VERDICT_CODE_SHA256 hashes the sources of its own
  VERDICT_FUNCS and of the imported S1 kernels (S1_VERDICT_FUNCS: log_softmax_rows, logits_chunk, kl_top1_chunk,
  band_rows, bootstrap, aggregate, max_kl), plus verdict_constants(). The blob check is the second guard.
- Batches of 8 windows (the last holds 4): for each, the N2, N3 and R children (prompt, states and KV), then one R0
  child (the one-shots, then for each arm's KV the continuation and the P-R0 step), then the batch's KV deleted.
  KV per arm and window is 570 MB (MEASURED in S1): 13.7 GB per batch, about 148 GB written in all (DERIVED).
- Runtime ESTIMATE from S1's MEASURED states: about 120 s per window, 2.8 h for the states; about 10 min for the
  check; 35-45 min and about 5 GB for the verdict (about 430k scored positions).
- Logs: results/llm/hybrid_s1b_{selftest,prereg,check,states,verdict}_desktop2_YYYYMMDD.log, UTF-8, the profile
  path scrubbed, never overwritten (--tag for a re-run). Every RAM-heavy stage is a START REQUEST, confirmed
  before it starts, and a FINISHED after.

## 10. What S1b would not establish

- The 780M's generation numerics (S0), or its attention precision.
- An NPU run: CPU emulation, and N2's NPU tie is DERIVED.
- Speed or energy.
- Other text or tokenizers, prompts other than 2,048 ids, or continuations beyond 1,023 positions.
- Free-running generation: the continuation is teacher-forced.
"""


# ---------------------------------------------------------------- guards

def s1_blob() -> dict:
    def git(*a):
        r = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, encoding="utf-8")
        return r.stdout.strip() if r.returncode == 0 else None
    at, now = git("rev-parse", f"{S1_COMMIT}:tools/hybrid_s1.py"), git("hash-object", "tools/hybrid_s1.py")
    return {"commit": S1_COMMIT, "blob_at_commit": at, "blob_now": now, "equal": at is not None and at == now}


def frozen_ok() -> bool:
    p = s1.latest("hybrid_s1b_prereg_*.log")
    text = p.read_text(encoding="utf-8")
    want = dict(re.findall(r"^(PREREG_TEXT_SHA256|PROTOCOL_JSON_SHA256|VERDICT_CODE_SHA256) ([0-9a-f]{64})$",
                           text, re.M))
    got = hashes()
    ok = all(want.get(k) == v for k, v in got.items())
    blob = s1_blob()
    s1.say("FROZEN_JSON", {"prereg_log": p.name, "prereg_log_lf_sha256": s1.sha_lf(p), "equal": ok, **got})
    s1.say("S1_BLOB_JSON", blob)
    return ok and blob["equal"]


def pins_ok() -> bool:
    """Hash every reused model file against MODEL_PIN (about 17 GB read)."""
    res, ok = {}, True
    for f, (size, sha) in MODEL_PIN.items():
        p = MODELS / f
        got = s1.sha_file(p) if p.exists() else None
        good = p.exists() and p.stat().st_size == size and got == sha
        res[f] = {"bytes": p.stat().st_size if p.exists() else None, "sha256": got, "ok": good}
        ok &= good
    s1.say("MODEL_PIN_JSON", res)
    return ok


# ---------------------------------------------------------------- children

_STDOUT_JSON = []


def child(args: list, retries: int = 0) -> int:
    """Run a child of this tool, its output into this log, and keep its tagged JSON for the parent."""
    for attempt in range(retries + 1):
        sys.stdout.flush()
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve())] + args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end="", flush=True)
            _STDOUT_JSON.append(line.rstrip("\n"))
        rc = p.wait()
        print(f"CHILD {' '.join(args)} rc {rc}" + (f" (attempt {attempt + 1})" if retries else ""), flush=True)
        if rc == 0 or rc == 4:
            return rc
    return rc


def stdout_json(tag: str) -> list:
    return [json.loads(s.split(" ", 1)[1]) for s in _STDOUT_JSON if s.startswith(tag + " ")]


def batches() -> list:
    return [list(range(b, min(b + BATCH, N_WIN))) for b in range(0, N_WIN, BATCH)]


def c5_child() -> int:
    s1.start_gate("N2 (C5')")
    wins = pinned_windows()
    sess = s1.session(MODELS / "N2.onnx")
    h1, _ = s1.run(sess, wins[0][:PROMPT])
    h2, _ = s1.run(sess, wins[0][:PROMPT])
    s1.say("C5P_JSON", {"sha_run1": s1.arr_sha(h1), "sha_run2": s1.arr_sha(h2),
                        "outcome": "PASS" if s1.arr_sha(h1) == s1.arr_sha(h2) else "FAIL"})
    s1.say("MEM_JSON", {"child": "N2 (C5')", **s1.own_memory()})
    return 0


def control_child() -> int:
    s1.start_gate("R0 control (a, b, c)")
    wins = pinned_windows()
    sess = s1.session(MODELS / "R0.onnx")
    for w in CONTROL_WINDOWS:
        ids = wins[w]
        t0 = time.perf_counter()
        ho, _ = s1.run(sess, ids)
        hp, kv = s1.run(sess, ids[:PROMPT], want_kv=True)
        hc, _ = s1.run(sess, ids[PROMPT:], past=kv)
        hd, _ = s1.run(sess, ids[P_POS:PROMPT], past=slice_kv(kv, P_POS))
        del kv
        for k, a in (("oneshot", ho), ("prompt", hp), ("cont", hc), ("dec", hd)):
            np.save(CHECK / f"ctl_w{w:02d}_{k}.npy", a)
        s1.say("CONTROL_RUN_JSON", {"w": w, "oneshot": s1.arr_sha(ho), "prompt": s1.arr_sha(hp),
                                    "cont": s1.arr_sha(hc), "dec": s1.arr_sha(hd),
                                    "rows": [len(ho), len(hp), len(hc), len(hd)],
                                    "seconds": round(time.perf_counter() - t0, 1)})
    s1.say("MEM_JSON", {"child": "R0 control", **s1.own_memory()})
    return 0


def ccompare_child() -> int:
    s1.start_gate("control compare (head)")
    head, _, _ = s1.load_head()
    b = s1.logged_json(RESULTS / S1_BUILD_LOG, "BUILD_JSON")
    if s1.arr_sha(head) != b["head_sha256"]:
        print("CONTROL STOP: the head differs from S1's r2 build", flush=True)
        return 2
    n = SET_C[1] - SET_C[0] + 1
    for w in CONTROL_WINDOWS:
        ho, hp, hc, hd = (np.load(CHECK / f"ctl_w{w:02d}_{k}.npy") for k in ("oneshot", "prompt", "cont", "dec"))
        pairs = {"a": (ho[SET_C[0]:SET_C[1] + 1], hc[:n]), "b": (ho[P_POS:P_POS + 1], hd[:1]),
                 "c": (ho[:PROMPT], hp)}
        row = {"w": w}
        for p, (ref, got) in pairs.items():
            finite = bool(np.isfinite(ref).all() and np.isfinite(got).all())
            row[p] = {"max_kl": s1.max_kl(ref, got, head) if finite else float("inf"),
                      "sha_equal": s1.arr_sha(ref) == s1.arr_sha(got), "rows": len(got)}
        s1.say("CONTROL_JSON", row)
    return 0


def prompt_child(arm: str, b: str) -> int:
    s1.start_gate(f"prompt {arm} batch {b}")
    wins = pinned_windows()
    ws = batches()[int(b)]
    d, kd = STATES / arm, KVDIR / arm
    d.mkdir(parents=True, exist_ok=True)
    kd.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sess = s1.session(MODELS / f"{arm}.onnx")
    s1.say("SESSION_JSON", {"arm": arm, "batch": int(b), "seconds": round(time.perf_counter() - t0, 1),
                            **s1.own_memory()})
    for w in ws:
        t1 = time.perf_counter()
        h, kv = s1.run(sess, wins[w][:PROMPT], want_kv=True)
        keep = h if arm == "N2" else h[P_POS:P_POS + 1]
        np.save(d / f"w{w:02d}_prompt.npy", keep)
        np.savez(kd / f"w{w:02d}.npz", **kv)
        del kv
        s1.say("PROMPT_JSON", {"arm": arm, "w": w, "rows": len(keep), "sha256": s1.arr_sha(keep),
                               "finite": bool(np.isfinite(h).all()), "seconds": round(time.perf_counter() - t1, 1)})
    s1.say("MEM_JSON", {"child": f"prompt {arm} batch {b}", **s1.own_memory()})
    return 0


def r0_child(b: str) -> int:
    s1.start_gate(f"R0 batch {b}")
    wins = pinned_windows()
    ws = batches()[int(b)]
    d = STATES / "R0"
    d.mkdir(parents=True, exist_ok=True)
    sess = s1.session(MODELS / "R0.onnx")
    for w in ws:
        t1 = time.perf_counter()
        h, _ = s1.run(sess, wins[w])
        np.save(d / f"w{w:02d}_oneshot.npy", h)
        s1.say("ONESHOT_JSON", {"w": w, "rows": len(h), "sha256": s1.arr_sha(h), "finite": bool(np.isfinite(h).all()),
                                "seconds": round(time.perf_counter() - t1, 1)})
    for arm in ARMS:
        for w in ws:
            f = KVDIR / arm / f"w{w:02d}.npz"
            if not f.exists():                           # its prompt child failed twice: that window is missing
                s1.say("CONT_SKIPPED_JSON", {"arm": arm, "w": w, "why": "no KV from the arm's prompt run"})
                continue
            t1 = time.perf_counter()
            kv = dict(np.load(f))
            hc, _ = s1.run(sess, wins[w][PROMPT:], past=kv)
            hd, _ = s1.run(sess, wins[w][P_POS:PROMPT], past=slice_kv(kv, P_POS))
            del kv
            np.save(STATES / arm / f"w{w:02d}_cont.npy", hc)
            np.save(STATES / arm / f"w{w:02d}_pr0.npy", hd)
            s1.say("CONT_JSON", {"arm": arm, "w": w, "rows": len(hc), "sha256": s1.arr_sha(hc),
                                 "finite": bool(np.isfinite(hc).all()), "seconds": round(time.perf_counter() - t1, 1)})
            s1.say("PR0_JSON", {"arm": arm, "w": w, "rows": len(hd), "sha256": s1.arr_sha(hd),
                                "finite": bool(np.isfinite(hd).all())})
    s1.say("MEM_JSON", {"child": f"R0 batch {b}", **s1.own_memory()})
    return 0


# ---------------------------------------------------------------- stages

def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def prereg() -> int:
    print(PREREG, flush=True)
    s1.say("PROTOCOL_JSON", protocol())
    s1.say("PREDICTIONS_JSON", [{"id": q, "text": t} for q, t in PREDICTIONS])
    T = s1.text_tokens()
    pins = text_pins(T)
    s1.say("TEXT_JSON", pins)
    a = pins["asserts"]
    print(f"TEXT: {pins['text_tokens']} tokens; {N_WIN} windows of {WIN_LEN} ids from {pins['first']} to "
          f"{pins['end']}, {pins['left']} tokens left; asserts: start at or after {S1_END} {a['start_after_s1']}, "
          f"pairwise disjoint {a['pairwise_disjoint']}, inside the text {a['inside_text']}, no S1 prompt sha "
          f"{a['no_s1_prompt_sha']}", flush=True)
    b = s1.logged_json(RESULTS / S1_BUILD_LOG, "BUILD_JSON")
    pin_cmp = {f: {"pin": list(v), "logged": ([b["files"][f]["bytes"], b["files"][f]["sha256"]] if f in b["files"]
                                              else list(s1.SRC_PIN["model.onnx.data"]))}
               for f, v in MODEL_PIN.items()}
    pins_equal = all(c["pin"] == c["logged"] for c in pin_cmp.values())
    s1.say("MODEL_PIN_JSON", {"source": f"{S1_BUILD_LOG} BUILD_JSON (base.onnx.data: hybrid_s1 SRC_PIN)",
                              "source_lf_sha256": s1.sha_lf(RESULTS / S1_BUILD_LOG), "equal": pins_equal,
                              "files": pin_cmp})
    s1.say("VERDICT_SOURCES_JSON", [{"name": n, "sha256": s1.sha_bytes(s.encode("utf-8"))}
                                    for n, s in verdict_sources()])
    blob = s1_blob()
    s1.say("S1_BLOB_JSON", blob)
    print_hashes()
    stop = [k for k, ok in (("text asserts", a["ok"]), ("model pins", pins_equal), ("S1 blob", blob["equal"]))
            if not ok]
    if stop:
        print(f"PREREG STOP: {', '.join(stop)}", flush=True)
        return 1
    print("PREREG OK", flush=True)
    return 0


def check() -> int:
    print(f"CHECK (hybrid S1b), no chip. {stamp()}", flush=True)
    print_hashes()
    if not frozen_ok():
        print("CHECK STOP: the frozen hashes or S1's blob differ", flush=True)
        return 2
    if not pins_ok():
        print("CHECK STOP: a model file differs from MODEL_PIN", flush=True)
        return 2
    if CHECK.exists():
        shutil.rmtree(CHECK)
    CHECK.mkdir(parents=True)
    rcs = {}
    for c in (["_c5"], ["_control"], ["_ccompare"]):
        rcs[c[0]] = child(c, retries=1)
        if rcs[c[0]] == 4:
            print("CHECK STOP: memory gate; not a verdict, re-run later", flush=True)
            return 4
    c5 = stdout_json("C5P_JSON")
    ctl = control_outcome(stdout_json("CONTROL_JSON"))
    res = {"pins": True, "C5p": c5[-1]["outcome"] if c5 else "MISSING", "control": ctl, "rc": rcs}
    res["stop"] = ctl["outcome"] != "PASS"
    s1.say("CHECK_JSON", res)
    if res["stop"]:
        print("CHECK STOP: the R0 control failed (a pipeline defect, not a verdict)", flush=True)
        return 2
    if res["C5p"] != "PASS":
        print("CHECK STOP (C5'): N2 is INCOMPLETE, and the states stage does not run", flush=True)
        return 2
    print("CHECK OK", flush=True)
    return 0


def states() -> int:
    print(f"STATES (hybrid S1b), no chip. {stamp()}", flush=True)
    print_hashes()
    if not frozen_ok():
        print("STATES STOP: the frozen hashes or S1's blob differ", flush=True)
        return 2
    c = s1.logged_json(s1.latest("hybrid_s1b_check_*.log"), "CHECK_JSON")
    s1.say("GATES_READ_JSON", {"control": c["control"]["outcome"], "C5p": c["C5p"]})
    if c["stop"] or c["C5p"] != "PASS":
        print("STATES STOP: the check is not OK", flush=True)
        return 2
    if not pins_ok():
        print("STATES STOP: a model file differs from MODEL_PIN", flush=True)
        return 2
    free = shutil.disk_usage(WORK.parent).free / 1e9
    s1.say("DISK_JSON", {"free_gb": round(free, 1), "min_gb": MIN_DISK_GB})
    if free < MIN_DISK_GB:
        print(f"STATES STOP: {free:.1f} GB free disk, below {MIN_DISK_GB} GB", flush=True)
        return 2
    for p in (STATES, KVDIR):
        if p.exists():
            shutil.rmtree(p)
    rcs = {}
    for b, ws in enumerate(batches()):
        print(f"BATCH {b}: windows {ws[0]}-{ws[-1]} {stamp()}", flush=True)
        for arm in ARMS:
            rcs[f"{arm}/{b}"] = child(["_prompt", arm, str(b)], retries=1)
            if rcs[f"{arm}/{b}"] == 4:
                print("STATES STOP: memory gate; not a verdict, re-run later", flush=True)
                return 4
        rcs[f"R0/{b}"] = child(["_r0", str(b)], retries=1)
        if rcs[f"R0/{b}"] == 4:
            print("STATES STOP: memory gate; not a verdict, re-run later", flush=True)
            return 4
        if KVDIR.exists():
            shutil.rmtree(KVDIR)
        print(f"KV_DELETED batch {b} {not KVDIR.exists()}", flush=True)
    s1.say("STATES_DONE_JSON", {"rc": rcs})
    ok = all(v == 0 for v in rcs.values())
    print("STATES", "OK" if ok else "INCOMPLETE", flush=True)
    return 0 if ok else 2


def verdict() -> int:
    print(f"VERDICT (hybrid S1b), no chip. {stamp()}", flush=True)
    print_hashes()
    if not frozen_ok():
        print("VERDICT REFUSED: the frozen hashes or S1's blob differ", flush=True)
        return 3
    s1.start_gate("verdict")
    cl, sl = s1.latest("hybrid_s1b_check_*.log"), s1.latest("hybrid_s1b_states_*.log")
    for p in (cl, sl):
        print(f"INPUT_LOG {p.name}: LF sha256 {s1.sha_lf(p)}", flush=True)
    c = s1.logged_json(cl, "CHECK_JSON")
    if c["control"]["outcome"] != "PASS":
        print("VERDICT STOP: the R0 control failed", flush=True)
        return 3
    prm = {(r["arm"], r["w"]): r for r in s1.logged_json(sl, "PROMPT_JSON", every=True)}
    cnt = {(r["arm"], r["w"]): r for r in s1.logged_json(sl, "CONT_JSON", every=True)}
    pr0 = {(r["arm"], r["w"]): r for r in s1.logged_json(sl, "PR0_JSON", every=True)}
    one = {r["w"]: r for r in s1.logged_json(sl, "ONESHOT_JSON", every=True)}
    wins = pinned_windows()
    head, _, _ = s1.load_head()
    if s1.arr_sha(head) != s1.logged_json(RESULTS / S1_BUILD_LOG, "BUILD_JSON")["head_sha256"]:
        print("VERDICT REFUSED: the head differs from S1's r2 build", flush=True)
        return 3

    def load(path: Path, rec: dict):
        if rec is None or not path.exists():
            return None
        a = np.load(path)
        return a if s1.arr_sha(a) == rec["sha256"] else None

    per = {a: {} for a in ARMS}
    missing = {a: [] for a in ARMS}
    for w, ids in enumerate(wins):
        ho = load(STATES / "R0" / f"w{w:02d}_oneshot.npy", one.get(w))
        if ho is None:
            for a in ARMS:
                missing[a].append(w)
            continue
        arms = {}
        for a in ARMS:
            hp = load(STATES / a / f"w{w:02d}_prompt.npy", prm.get((a, w)))
            hc = load(STATES / a / f"w{w:02d}_cont.npy", cnt.get((a, w)))
            hd = load(STATES / a / f"w{w:02d}_pr0.npy", pr0.get((a, w)))
            if hp is None or hc is None or hd is None:
                missing[a].append(w)
                continue
            arms[a] = {"prompt": hp, "prompt_rows": "all" if a == "N2" else "last", "cont": hc, "pr0": hd}
        if arms:
            for a, sets in window_metrics(ho, arms, ids, head).items():
                for k, v in sets.items():
                    per[a].setdefault(k, []).append(v)
        print(f"WINDOW {w} scored", flush=True)
    metrics = {a: {k: set_aggregate(v) for k, v in per[a].items()} for a in ARMS if not missing[a]}
    complete = {a: not missing[a] for a in ARMS}
    n2 = outcome_n2(metrics.get("N2", {}), complete["N2"], c["C5p"])
    n2_halves = halves(per["N2"]["C"]) if complete["N2"] else None
    sc = score_predictions(metrics, c["control"], n2_halves)
    report(metrics, n2, c, n2_halves, missing, sc)
    s1.say("VERDICT_JSON", {"N2_set_C": n2["outcome"], "narrow": n2["narrow"], "pick": PICK,
                            "complete": complete})
    s1.say("PREDICTIONS_SCORED_JSON", sc)
    return 0 if n2["outcome"] != "INCOMPLETE" else 2


def report(metrics, n2, c, n2_halves, missing, sc) -> None:
    print(f"\nTHE R0 CONTROL (check log): {c['control']['outcome']}; sha-equal per path {c['control']['sha_equal']}; "
          f"C5' {c['C5p']}", flush=True)
    print(f"\nAgainst R0 one-shot (the rule: mean KL <= {T_KL} and top-1 >= {T_TOP1}; set C decides N2; the rest "
          f"is report-only)")
    print(f"  {'arm':4s} {'set':6s} {'mean KL':>10s} {'95% CI':>23s} {'p50':>9s} {'p99':>9s} {'max':>9s} "
          f"{'top-1':>7s} {'95% CI':>17s} {'ppl':>8s}  outcome")
    for a in ARMS:
        if a not in metrics:
            print(f"  {a:4s} INCOMPLETE: windows missing {missing[a]}")
            continue
        for k in ("C", "P_arm", "P_R0", "L", "H"):
            if k not in metrics[a]:
                continue
            m = metrics[a][k]
            o, nb = rule_set(m)
            tag = "DECIDING" if (a, k) == ("N2", "C") else "report-only"
            print(f"  {a:4s} {k:6s} {m['kl_mean']:10.5f} [{m['kl_ci'][0]:.5f}, {m['kl_ci'][1]:.5f}] "
                  f"{m['kl_p50']:9.5f} {m['kl_p99']:9.4f} {m['kl_max']:9.3f} {m['top1']:7.4f} "
                  f"[{m['top1_ci'][0]:.4f}, {m['top1_ci'][1]:.4f}] {m['ppl']:8.4f}  {o}"
                  f"{' NARROW' if nb else ''} ({tag})")
    if n2_halves is not None:
        print(f"\nN2 set C by position (Q2): mean KL {n2_halves[0]:.5f} over 2,048-2,559, {n2_halves[1]:.5f} over "
              f"2,560-3,070")
    print(f"\nN2 ON SET C: {n2['outcome']}{' (NARROW)' if n2['narrow'] else ''}")
    print(f"THE PICK: {PICK}")
    print("\nPredictions (they decide nothing):")
    for q, text in PREDICTIONS:
        print(f"  {q} {sc[q]}: {text}")
    s1.say("METRICS_JSON", metrics)
    s1.say("N2_HALVES_JSON", n2_halves)


# ---------------------------------------------------------------- selftest

def selftest() -> int:
    fails = []

    def expect(what, got, want=True):
        print(f"  {what}: {got}" + ("" if got == want else f"  (expected {want})"), flush=True)
        if got != want:
            fails.append(what)

    rng = np.random.default_rng(1)
    n_tok = 292282
    print("The windows and the three non-overlap asserts (synthetic ids of the real text's length):")
    offs = offsets()
    expect("84 windows, the first at 32,752", (len(offs), offs[0][0]), (84, 32752))
    expect("the last window ends at 290,716, leaving 1,566 tokens", (offs[-1][1], n_tok - offs[-1][1]), (290716, 1566))
    T = rng.integers(3, 1000, n_tok).tolist()
    s1_seq = [gds.ids_sha([BOS] + T[s1.STRIDE * s:s1.STRIDE * (s + 1)]) for s in range(s1.N_SEQ)]
    wins = [[BOS] + T[a:b] for a, b in offs]
    psha = [gds.ids_sha(w[:PROMPT]) for w in wins]
    expect("every window is 3,072 ids", all(len(w) == WIN_LEN for w in wins))
    expect("the real windows pass all asserts", window_asserts(offs, n_tok, S1_END, psha, s1_seq)["ok"])
    bad = offsets(stride=WIN_STRIDE - 1)
    expect("the deliberate case overlaps by one token", bad[0][1] - bad[1][0], 1)
    expect("a deliberate overlap (stride 3,070, windows of 3,071) fails pairwise_disjoint",
           window_asserts(bad, n_tok, S1_END, psha, s1_seq)["pairwise_disjoint"], False)
    early = offsets(start=S1_END - 1)
    expect("a start at 32,751 fails start_after_s1", window_asserts(early, n_tok, S1_END, psha, s1_seq)["start_after_s1"],
           False)
    expect("85 windows overrun the text", window_asserts(offsets(n=85), n_tok, S1_END, psha, s1_seq)["inside_text"],
           False)
    expect("a prompt equal to an S1 sequence fails no_s1_prompt_sha",
           window_asserts(offs, n_tok, S1_END, psha[:5] + [s1_seq[3]] + psha[6:], s1_seq)["no_s1_prompt_sha"], False)

    print("The KV slice to 0-2,046 on a small causal attention (numpy): prompt, slice, one step = the one-shot row:")
    Hd, V = 8, 50
    Wq, Wk, Wv = (rng.standard_normal((Hd, Hd)) for _ in range(3))
    emb = rng.standard_normal((V, Hd))

    def attend(x, past=None):
        """Causal attention over [past + x]; returns (outputs for x, the KV of past + x as [1, 1, n, Hd])."""
        q, k, v = x @ Wq, x @ Wk, x @ Wv
        if past is not None:
            k = np.concatenate([past["k"][0, 0], k])
            v = np.concatenate([past["v"][0, 0], v])
        n0 = len(k) - len(x)
        s = q @ k.T / np.sqrt(Hd)
        s = np.where(np.arange(len(k))[None, :] <= (n0 + np.arange(len(x)))[:, None], s, -np.inf)
        p = np.exp(s - s.max(axis=1, keepdims=True))
        return (p / p.sum(axis=1, keepdims=True)) @ v, {"k": k[None, None], "v": v[None, None]}
    ids = rng.integers(0, V, WIN_LEN)
    x = emb[ids]
    one, _ = attend(x)
    _, kv = attend(x[:PROMPT])
    sl = slice_kv(kv, P_POS)
    step, _ = attend(x[P_POS:PROMPT], past=sl)
    expect("the slice keeps 2,047 positions, contiguous", (sl["k"].shape[2], sl["k"].flags["C_CONTIGUOUS"]), (2047, True))
    expect("one step on the sliced KV equals the one-shot at 2,047", bool(np.allclose(step[0], one[P_POS], rtol=0,
                                                                                    atol=1e-12)))
    cont, _ = attend(x[PROMPT:], past=kv)
    expect("the continuation on the full KV equals the one-shot at 2,048-3,071", bool(np.allclose(cont, one[PROMPT:],
                                                                                              rtol=0, atol=1e-12)))

    print("The set C, P-arm, P-R0 and L/H slicing on synthetic hidden states (a small head):")
    head = rng.standard_normal((V, Hd)).astype(np.float32)
    ho = rng.standard_normal((WIN_LEN, Hd)).astype(np.float32)
    idl = rng.integers(0, V, WIN_LEN).tolist()

    def arms_equal():
        return {"N2": {"prompt": ho[:PROMPT].copy(), "prompt_rows": "all", "cont": ho[PROMPT:].copy(),
                       "pr0": ho[P_POS:P_POS + 1].copy()},
                "N3": {"prompt": ho[P_POS:P_POS + 1].copy(), "prompt_rows": "last", "cont": ho[PROMPT:].copy(),
                       "pr0": ho[P_POS:P_POS + 1].copy()}}
    m = window_metrics(ho, arms_equal(), idl, head)
    expect("sets per arm", (sorted(m["N2"]), sorted(m["N3"])), (["C", "H", "L", "P_R0", "P_arm"], ["C", "P_R0", "P_arm"]))
    expect("positions: C 1,023, P 1 and 1, L 1,024, H 1,023",
           [len(m["N2"][k]["kl"]) for k in ("C", "P_arm", "P_R0", "L", "H")], [1023, 1, 1, 1024, 1023])
    expect("equal states give KL 0 and top-1 1 everywhere",
           all(np.abs(m[a][k]["kl"]).max() < 1e-9 and m[a][k]["top1"].min() == 1 for a in m for k in m[a]))

    def hit(mutate, arm="N2"):
        a = arms_equal()
        mutate(a[arm])
        mm = window_metrics(ho, a, idl, head)[arm]
        return sorted(k for k in mm if np.abs(mm[k]["kl"]).max() > 1e-6)
    expect("changing prompt row 2,047 moves P-arm only", hit(lambda a: a["prompt"].__setitem__(P_POS, 5.0)), ["P_arm"])
    expect("changing prompt row 100 moves band L only", hit(lambda a: a["prompt"].__setitem__(100, 5.0)), ["L"])
    expect("changing prompt row 2,046 moves band H only", hit(lambda a: a["prompt"].__setitem__(2046, 5.0)), ["H"])
    expect("changing continuation row 0 moves set C only", hit(lambda a: a["cont"].__setitem__(0, 5.0)), ["C"])
    expect("changing continuation row 1,023 (position 3,071, unscored) moves nothing",
           hit(lambda a: a["cont"].__setitem__(1023, 5.0)), [])
    expect("changing the P-R0 row moves P-R0 only", hit(lambda a: a["pr0"].__setitem__(0, 5.0)), ["P_R0"])
    expect("a last-row-only prompt: its row moves P-arm only", hit(lambda a: a["prompt"].__setitem__(0, 5.0), "N3"),
           ["P_arm"])
    ci = m["N2"]["C"]
    lp = s1.log_softmax_rows(s1.logits_chunk(ho[2048:2049], head))
    expect("set C's first row scores id 2,049 (the next id)", bool(np.isclose(ci["nll"][0], -lp[0, idl[2049]])))
    h2 = halves([m["N2"]["C"]] * 3)
    expect("halves split set C at 2,560", len(range(SET_C[0], Q2_SPLIT)) + len(range(Q2_SPLIT, SET_C[1] + 1)), 1023)
    expect("halves of equal states are 0", bool(abs(h2[0]) < 1e-9 and abs(h2[1]) < 1e-9))

    print("The rule, NARROW, P's granularity, INCOMPLETE and the pick:")

    def agg(kl, t1, finite=True, ci=None, tci=None):
        return {"kl_mean": kl, "top1": t1, "kl_ci": ci or [kl, kl], "top1_ci": tci or [t1, t1], "finite": finite}
    expect("rule: PASS at the thresholds", rule_set(agg(T_KL, T_TOP1))[0], "PASS")
    expect("rule: FAIL above T_KL", rule_set(agg(0.0124, 0.99))[0], "FAIL")
    expect("rule: FAIL below T_TOP1", rule_set(agg(0.001, 0.955))[0], "FAIL")
    expect("rule: FAIL when non-finite", rule_set(agg(0.0, 1.0, finite=False))[0], "FAIL")
    expect("rule: NARROW when the interval holds T_TOP1", rule_set(agg(0.005, 0.958, tci=[0.955, 0.961])), ("PASS", True))
    expect("rule: NARROW when the interval holds T_KL", rule_set(agg(0.013, 0.97, ci=[0.011, 0.014])), ("FAIL", True))
    expect("P granularity: 81/84 passes", rule_set(agg(0.001, 81 / 84))[0], "PASS")
    expect("P granularity: 80/84 fails", rule_set(agg(0.001, 80 / 84))[0], "FAIL")
    good = {"C": agg(0.009, 0.96), "P_arm": agg(0.05, 0.90), "P_R0": agg(0.02, 0.94)}
    expect("N2: a set C PASS with a P-arm FAIL is a PASS (P is report-only)", outcome_n2(good, True, "PASS")["outcome"],
           "PASS")
    expect("N2: a missing window is INCOMPLETE", outcome_n2(good, False, "PASS")["outcome"], "INCOMPLETE")
    expect("N2: C5' failing is INCOMPLETE", outcome_n2(good, True, "FAIL")["outcome"], "INCOMPLETE")
    expect("N2: set C FAIL", outcome_n2({"C": agg(0.02, 0.96)}, True, "PASS")["outcome"], "FAIL")
    expect("the pick is S1's N3, fixed", PICK.startswith("N3 (S1's pick; S1b does not change it"), True)

    print("The R0 control: each path failing alone is a STOP:")

    def ctl_rows(bad_w=None, bad_p=None, val=2e-5):
        rows = []
        for w in CONTROL_WINDOWS:
            r = {"w": w}
            for p in CONTROL_PATHS:
                r[p] = {"max_kl": val if (w, p) == (bad_w, bad_p) else 0.0, "sha_equal": (w, p) != (bad_w, bad_p)}
            rows.append(r)
        return rows
    expect("all paths at 0: PASS", control_outcome(ctl_rows())["outcome"], "PASS")
    for p in CONTROL_PATHS:
        expect(f"path ({p}) above 1e-5 on one window: FAIL", control_outcome(ctl_rows(48, p))["outcome"], "FAIL")
    expect("a non-finite path: FAIL", control_outcome(ctl_rows(0, "b", float("inf")))["outcome"], "FAIL")
    expect("exactly 1e-5: PASS", control_outcome(ctl_rows(0, "a", 1e-5))["outcome"], "PASS")
    expect("a missing control window: FAIL", control_outcome(ctl_rows()[:-1])["outcome"], "FAIL")

    print("The predictions' scoring:")
    mets = {"N2": {"C": agg(0.0096, 0.957, tci=[0.954, 0.960]), "P_arm": agg(0.05, 0.90), "P_R0": agg(0.02, 0.95),
                   "L": agg(0.06, 0.895), "H": agg(0.05, 0.91)},
            "N3": {k: agg(0.0001, 0.995) for k in ("C", "P_arm", "P_R0")}, "R": {"C": agg(0.001, 0.99)}}
    ctl = control_outcome(ctl_rows())
    expect("all HIT", score_predictions(mets, ctl, [0.012, 0.008]), {q: "HIT" for q, _ in PREDICTIONS})
    expect("Q2 MISS when KL rises", score_predictions(mets, ctl, [0.008, 0.012])["Q2"], "MISS")
    expect("Q7 MISS when (c) is not sha-equal",
           score_predictions(mets, control_outcome(ctl_rows(12, "c", 0.0)), [0.012, 0.008])["Q7"], "MISS")
    expect("NOT SCORED without N2", score_predictions({}, ctl, None)["Q1"], "NOT SCORED")

    print("The memory watchdog (S1's amendment 1), in a child whose main thread sleeps 3 s:")
    for min_gb, want_rc in ((1e6, 4), (0.0, 0)):
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_watchdog", repr(min_gb)],
                           capture_output=True, text=True, encoding="utf-8", timeout=300)
        expect(f"threshold {min_gb:g} GB: rc", r.returncode, want_rc)
        expect(f"threshold {min_gb:g} GB: MEMORY_ABORT_JSON printed", "MEMORY_ABORT_JSON" in r.stdout, want_rc == 4)

    print("The frozen hashes and the verdict sources:")
    names = [n for n, _ in verdict_sources()]
    expect("VERDICT_CODE covers S1b's functions and the imported S1 kernels", names,
           list(VERDICT_FUNCS) + [f"hybrid_s1.{f}" for f in S1_VERDICT_FUNCS])
    own = s1.sha_bytes(("".join(inspect.getsource(globals()[f]) for f in VERDICT_FUNCS)
                        + json.dumps(verdict_constants(), sort_keys=True)).encode("utf-8"))
    expect("the imported sources change VERDICT_CODE", own != verdict_code_sha())
    for kk, v in hashes().items():
        print(f"  {kk} {v}")
    expect("PREREG holds the plan", PREREG.startswith("S1b PLAN AND PREREG"))
    blob = s1_blob()
    expect("tools/hybrid_s1.py's blob equals c67d6d2's", blob["equal"])
    print("SELFTEST", "OK" if not fails else f"FAILED: {fails}", flush=True)
    return 0 if not fails else 1


# ---------------------------------------------------------------- main

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("selftest", "prereg", "check", "states", "verdict", "_c5", "_control",
                                     "_ccompare", "_prompt", "_r0", "_watchdog"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.mode == "_watchdog":
        return s1.watchdog_child(a.args[0])
    if a.mode == "_prompt":
        return prompt_child(a.args[0], a.args[1])
    if a.mode == "_r0":
        return r0_child(a.args[0])
    return {"selftest": selftest, "prereg": prereg, "check": check, "states": states, "verdict": verdict,
            "_c5": c5_child, "_control": control_child, "_ccompare": ccompare_child}[a.mode]()


if __name__ == "__main__":
    sys.exit(main())
