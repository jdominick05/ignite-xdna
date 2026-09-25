#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid stack, F2-A: F2's arithmetic (F2-E's pick: form B, FLUSH_B2, S = ROW) in all 238 linears of Gemma 3 4B,
at model level, on new text. CPU only: no NPU and no GPU. The plan is PREREG below (the accepted text, LF e2408569);
the rules are this file's verdict code (VERDICT_FUNCS, the imported S1 and S1b kernels, verdict_constants()), frozen
by VERDICT_CODE_SHA256.

The F2 graph is S1's base (the head cut from (c)'s C0-H16), with every MatMulNBits replaced by the F2 subgraph:
  per input row, s_x = amax / 127 per 32-block (fp32), Sx the smallest fp32 with 255 Sx >= max s_x, qx = max(1, c)
  with c the exact ceiling of s_x / Sx, codes RNE(x / (qx Sx)) in float64; W' = qw (c - 8) int16 from hybrid_f2e's
  weight grid; I = MatMul(double) of X' = qx code and W', exact; fp32(I), then f2_flush_b.cc's B2 in ORDER[2].
Two verdicts, KV first: F2-A/KV on set C (S1b's rule), F2-A/FULL on bands L and H (S1's rule). R0 is the reference.

    python tools/hybrid_f2a.py selftest   # synthetic only: no model, no text, no x16
    python tools/hybrid_f2a.py prereg     # the plan, the validation text's 16 windows, PROTOCOL_JSON, the frozen hashes
    python tools/hybrid_f2a.py build      # the F2 graph, its weight grid, and the probe on S1's sequence 0 (children)
    python tools/hybrid_f2a.py check      # the anchors K1-K8 (children)
    python tools/hybrid_f2a.py states     # F2, R and N2 prompts, then R0's continuations, in batches (children)
    python tools/hybrid_f2a.py verdict    # the metrics and the two frozen verdicts
Every heavy child holds one session (or the head), refuses below S1's MIN_AVAIL_GB available, and ends itself (rc 4)
if the available memory falls below S1's WATCH_MIN_GB while it runs. A child that exits nonzero stops its stage.
"""
import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gemma_compress as gc  # noqa: E402
import gemma_decode as gd  # noqa: E402
import gemma_decode_suite as gds  # noqa: E402
import hybrid_f2e as f2e  # noqa: E402
import hybrid_s1 as s1  # noqa: E402
import hybrid_s1b as s1b  # noqa: E402
import hybrid_u6e as u6e  # noqa: E402
import llm_prefill3c as p3c  # noqa: E402

WORK = ROOT / "scratch/llm/hybrid_f2a"
MODELS = WORK / "models"
CHECK = WORK / "check"
STATES = WORK / "states"
KVDIR = WORK / "kv"
PROF = WORK / "prof"
S1_MODELS = s1.MODELS                                    # S1's built models, read-only
RESULTS = s1.RESULTS

# ---------------------------------------------------------------- the pins (LF sha256 unless named)

PREREG_SHA = "e240856939dd21e966a8bdd96776ba1f0a11017f354bb634c568ea7fa20d8242"   # v2, accepted: PREREG hashes to it
PLAN_FILE = ROOT / "scratch/llm/hybrid_f2_plan_draft.md"                                 # git-ignored
PLAN_SHA = "b98787f451c11fbc9e847a4cb973b8f12b4988e3dce7729435d1fd34334a07c5"
ADDENDUM = ROOT / "scratch/llm/hybrid_f2e_addendum.md"                                   # git-ignored
ADDENDUM_SHA = "1ad61601f743525b6c08c2de38a86bad6f32f742e756a25333d5b257f2f165ec"
F2E_LOG = RESULTS / "hybrid_f2e_run_desktop2_20260925.log"                               # 96f83a8
F2E_LOG_SHA = "da97478e6894a6c7904616cf12038cc031c38d493282bba76dd07e1ceca73291"
F2E_PICK = "F2-E PICK: form B, FLUSH_B2, S = ROW,"
F2E_TOOL = ROOT / "tools/hybrid_f2e.py"                                                   # frozen at f88fe73
F2E_TOOL_SHA = "dac40cdffa052169e0a1a6a8c9de112a3883066e7da158f210640e5e43c83f78"
U6E_TOOL = ROOT / "tools/hybrid_u6e.py"
U6E_TOOL_SHA = "b88d23d30c6c2ec3ffa55407dd4969957acf98e788fca47f1a703feec146e892"
F20B_TOOL = ROOT / "tools/hybrid_f2_0b.py"
F20B_TOOL_SHA = "7a953146eaccc4b143bb4cf3072a84f10cad10f4f54bab1ec956d3588fcda354"
SOURCE = ROOT / "kernels/f2_epilogue/f2_flush_b.cc"
SOURCE_SHA = "caa62ce055f6e1a3e8de350349ac76bef3f8d9dd521416c9fb8db0c64a2e18d4"
PINS = [("the F2 plan", PLAN_FILE, PLAN_SHA), ("the F2-E addendum", ADDENDUM, ADDENDUM_SHA),
        ("F2-E's run log", F2E_LOG, F2E_LOG_SHA), ("the F2-E tool", F2E_TOOL, F2E_TOOL_SHA),
        ("the U6-E tool", U6E_TOOL, U6E_TOOL_SHA), ("the F2-0b tool", F20B_TOOL, F20B_TOOL_SHA),
        ("f2_flush_b.cc", SOURCE, SOURCE_SHA)]
BLOBS = {"tools/hybrid_s1.py": "c67d6d2", "tools/hybrid_s1b.py": "b777070"}
S1_BUILD_LOG = s1b.S1_BUILD_LOG                          # the r2 build: the model pins and the head
MODEL_PIN = {f: s1b.MODEL_PIN[f] for f in ("R0.onnx", "R.onnx", "N2.onnx", "int8.onnx.data", "base.onnx.data")}
C5_SHA = "a5597e1ffad816ad48a7dfb911883e3ec99bc454018890701fdca1e7a359ae83"   # S1's C5, check log line 18
VAL_PIN = {"repo": "Salesforce/wikitext", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
           "file": "wikitext-2-raw-v1/validation-00000-of-00001.parquet", "size": 657209,
           "sha256": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"}
ORT_VERSION = u6e.ORT_VERSION

# ---------------------------------------------------------------- the protocol (pre-registered)

ARM_KEY, FLUSH = "B/B2/ROW", "B2"                        # F2-E's pick (its log, line 133)
ARMS = ("F2", "R", "N2")                                 # the arms whose prompt KV R0 continues
DECIDING = ("F2",)
REPORT_ONLY = ("R", "N2")
BOS, PROMPT = s1.BOS, s1.SEQ_LEN                         # 2, 2,048
WIN_LEN = 3072                                           # [BOS] + 3,071 tokens
WIN_STRIDE = WIN_LEN - 1
N_WIN = 16
SET_C, P_POS = (2048, 3070), 2047
BANDS = s1.BANDS                                         # L (0-1,023), H (1,024-2,046)
Q2_SPLIT = 2560                                          # set C's halves: 2,048-2,559 and 2,560-3,070
SETS = ("C", "P_arm", "P_R0", "L", "H")
CONTROL_WINDOWS = (0, 2, 4, 6, 8, 10, 12, 14)
CONTROL_PATHS = ("a", "b", "c")
CONTROL_KL_MAX = s1.C6_KL_MAX                            # 1e-5
T_KL, T_TOP1 = s1.T_KL, s1.T_TOP1                        # 0.0123, 0.956
SEED, BOOT_N, CHUNK = s1.SEED, s1.BOOT_N, s1.CHUNK
BATCH = 8
MIN_DISK_GB = 25.0                                       # states: a batch's KV (13.7 GB) and the states (about 2 GB)
PROBE_LINE_S = 420.0                                     # the gate's line: 13.1 TFLOP / 420 s, about 31 GFLOPS
PROBE_CEILING_S = 1800.0                                 # the runaway ceiling only; the 420 s line is the build's STOP
K4_TOL = 1e-6
K4_PATH = 'arms["B/B2/ROW"]["vs_l0"]'
QX_MAX, CODE_MAX = f2e.QX_MAX, f2e.CODE_MAX              # 255, 127
STEP = 1.0 + 3.0 * 2.0 ** -25                            # one fp32 ulp up, by RNE (the prereg, section 2.2)
F2_TAG = "/f2/"
GUARD_NAME = "f2/guard"
W_BYTES = 6_417_285_120                                  # f2.onnx.data: the 238 W' at 2 B, no padding
FA_P2_FACTOR, FA_P3_TOP1 = 2.0, 0.98

RULE_KV = ("F2-A/KV, on set C (positions 2,048-3,070, R0 teacher-forced on the arm's full 2,048-position prompt KV): "
           "S1b's rule, verbatim. PASS iff mean KL <= 0.0123 and top-1 agreement >= 0.956 on set C; FAIL otherwise, "
           "and FAIL on a non-finite value. NARROW names a threshold inside the 95% bootstrap interval: it is reported "
           "and changes nothing; the point estimate decides. INCOMPLETE: a window's set C states for the F2 arm are "
           "missing when the gate rules that the verdict runs")
RULE_FULL = ("F2-A/FULL, on bands L (0-1,023) and H (1,024-2,046), the F2 arm's own logits: S1's rule, verbatim. PASS "
             "iff in both bands, pooled over the 16 windows, mean KL <= 0.0123 and top-1 agreement >= 0.956; FAIL "
             "otherwise, or on any non-finite hidden state or logit. NARROW names each band whose bootstrap interval "
             "holds a threshold; the verdict stands on the point estimate. INCOMPLETE: a window's F2 prompt states are "
             "missing when the gate rules that the verdict runs")
HEADLINE_ORDER = ("F2-A/KV", "F2-A/FULL")
COMBINATIONS = {
    "KV PASS, FULL FAIL": "N2's position: F2 qualifies for the KV-serving role on the same pre-registered basis as N2. "
                          "Any move is the user's.",
    "KV FAIL": "F2 does not qualify for the role N2 fills.",
    "BOTH PASS": "F2 qualifies for the KV-serving role, and also holds at prompt positions.",
    "KV INCOMPLETE": "F2-A/KV is INCOMPLETE: no combination is read.",
    "FULL INCOMPLETE": "F2-A/KV reads PASS and F2-A/FULL is INCOMPLETE: no combination is read.",
}
NO_PICK_CHANGE = ("The verdict does not change the N2 pick. Neither verdict is ever re-scored, and a NARROW changes "
                  "neither.")
PREDICTIONS = [
    ("FA-P1a", "F2-A/FULL: PASS in both bands, not NARROW"),
    ("FA-P1b", "F2-A/KV: PASS on set C, not NARROW"),
    ("FA-P2", "F2's mean KL is at most 2.0 x R's on the same windows, in both bands and on set C (a KL ratio)"),
    ("FA-P3", "F2's top-1 is >= 0.98 in both bands"),
    ("FA-P4", "F2's set C mean KL is below N2's on the same windows"),
    ("FA-P5", "R PASSES under both rules (report-only arm)"),
    ("FA-P6", "N2 FAILS bands L and H, and PASSES set C; its NARROW is not predicted (report-only arm)"),
    ("FA-P7", "F2 reads within both thresholds on P-R0 (report-only)"),
    ("FA-P8", "the guard reads 0 in every F2 run"),
]
ARM_TEXT = ("F2-E's pick, form B, FLUSH_B2, S = ROW, in all 238 MatMulNBits (q, k, v, o, gate, up, down x 34 layers). "
            "Per input row: s_x = fp32(amax / 127) per 32-block (S1's q8_ort); Sx = the smallest fp32 with 255 x Sx "
            ">= the row's max s_x: q = double(max s_x) / 255, f = Cast_f32(q), Sx = f if 255 x double(f) >= max s_x, "
            "else Cast_f32(double(f) x (1 + 3 x 2^-25)); c = the exact ceiling of s_x / Sx (Ceil, then hybrid_f2e's "
            "ceil_exact corrections, in order), c = 0 on a zero row; qx = Clip(c, 1, 255); codes = Clip(RNE(x64 / "
            "(qx x Sx)), -127, 127), 0 where qx x Sx is 0; X' = qx x code (double). Offline: Dw and qw by hybrid_f2e's "
            "w_grid at ROW; W' = qw x (c - 8), int16 [K, N], external (f2.onnx.data); Dw fp32 [N], inline. I = "
            "MatMul(double) of X' and Cast_double(W'), exact; fp32(I) = Cast_f32(I), RNE; then FLUSH_B2: bf16 pieces "
            "by Cast(float -> bfloat16 -> float) with an exact fp32 Sub, P by ORDER[2]['P'], y from an fp32 0 by "
            "ORDER[2]['Y'], one fp32 Add each. The activation side is built once per input and read by every linear "
            "on it (q, k, v; gate, up), as S1's N1 and N2 shared their quantization")
GUARD_TEXT = ("each input's count = the codes whose rounded value exceeds 127 in magnitude before the clip + the "
              "blocks with c > 255 + the rows with 255 x Sx < max s_x (int64); each of the 238 linears contributes its "
              "input's count, and the 238 are summed (Concat, ReduceSum) into the graph output f2/guard. Every run "
              "must read 0: a nonzero reading is a STOP. The weight side is checked in the build, offline, by "
              "hybrid_f2e's w_stops on all 238 linears")
ANCHORS = {
    "K1": "the pins (texts, tools, blobs, the F2-E log, the GGUF, x16 and W, S1's models, the F2 files by the build "
          "log, the validation text, the tokenizer, ORT) and the frozen triple",
    "K2": "graph identity: F2 equals R0.onnx node for node outside the 238 linears, inputs equal, outputs equal but "
          "for f2/guard, shared initializers on the same bytes of the same data file; each W' and Dw equal their "
          "re-derivation from the GGUF tensor the node names, by sha",
    "K3": "the census: the build's profiled pass on S1's test sequence 0 executes 238 MatMul with double inputs and no "
          "MatMulNBits, no fused op (S1's FUSED_FORBIDDEN), every op as the graph holds it, every op outside the F2 "
          "subgraphs at R0's count",
    "K4": "21 one-linear probes by the same builder on x16 (layers 0, 16, 33 x the seven linears) equal hybrid_f2e's "
          "B/B2/ROW emulation bit for bit on F2-E's 256 rows, and their rel-L2 against L0 reproduces the F2-E run "
          f"log's CASE_JSON {K4_PATH}, keyed by (layer, name), within {K4_TOL:g} relative",
    "K5": f"R0 on S1's test sequence 0 reproduces S1's C5 hidden-state sha {C5_SHA}",
    "K6": "the F2 arm runs window 0's prompt twice: the two hidden-state shas are equal",
    "K7": "the R0 control on windows 0, 2, ..., 14, paths (a), (b) and (c), each against R0 one-shot at max "
          "per-position KL <= 1e-5, sha-equality reported",
    "K8": "the guard reads 0 in every F2 run of the build, the check and the states",
}
REPORT_ONLY_LIST = ["N2's set C beside F2-A/KV", "the R arm: bands L and H, set C and set P, both rules' would-be "
                    "outcomes", "the N2 arm: bands L and H and set P, both rules' would-be outcomes",
                    "F2's set P both ways", "F2's set C in two halves (2,048-2,559, 2,560-3,070)",
                    "perplexity per arm, R0's too", "per-window means", "the bootstrap intervals",
                    "K4's values and worst relative difference", "the guard's counts"]
NO_RERUNS = ("a child that exits nonzero stops its stage, which comes to the gate as it stands; rc 4 (the memory gate "
             "or the watchdog) reads 'memory gate; not a verdict', and a memory-gate stop that produced no output may "
             "be restarted only on the gate's ruling; no logged stage is repeated otherwise; the verdict runs once")


def protocol() -> dict:
    return {"stage": "hybrid F2-A", "prereg_lf_sha256": PREREG_SHA, "arm": {"name": "F2", "pick": ARM_KEY,
                                                                              "text": ARM_TEXT, "linears": 238},
            "flush_order": {"t": f2e.FLUSH_ORDERS[FLUSH]["t"], "P": f2e.FLUSH_ORDERS[FLUSH]["P"],
                            "Y": f2e.FLUSH_ORDERS[FLUSH]["Y"]},
            "step": STEP, "guard": GUARD_TEXT, "reference": "R0 (S1's R0.onnx): one-shot over each window's 3,072 ids",
            "arms": ARMS, "deciding": DECIDING, "report_only": REPORT_ONLY,
            "text": {"val_pin": VAL_PIN, "joining": "the text column joined with two newlines, tokenized with "
                                                    "add_special_tokens=False by S1's tokenizer"},
            "windows": {"n": N_WIN, "len": WIN_LEN, "stride": WIN_STRIDE, "bos": BOS, "prompt": PROMPT,
                        "form": "window w = [BOS] + T[3,071 w : 3,071 (w + 1)]"},
            "sets": {"C": SET_C, "L": BANDS["L"], "H": BANDS["H"], "P": P_POS, "q2_split": Q2_SPLIT},
            "verdicts": [{"name": "F2-A/KV", "set": "C", "positions": SET_C, "rule": RULE_KV},
                         {"name": "F2-A/FULL", "sets": ["L", "H"], "positions": [BANDS["L"], BANDS["H"]],
                          "rule": RULE_FULL}],
            "headline_order": HEADLINE_ORDER, "combinations": COMBINATIONS, "no_pick_change": NO_PICK_CHANGE,
            "thresholds": {"t_kl": T_KL, "t_top1": T_TOP1},
            "bootstrap": {"over": "windows", "resamples": BOOT_N, "seed": SEED, "interval": 0.95},
            "report_only_list": REPORT_ONLY_LIST, "anchors": ANCHORS,
            "k4": {"path": K4_PATH, "tol": K4_TOL, "rows": "hybrid_f2e.ROWS (256)", "layers": f2e.LAYERS},
            "control": {"windows": CONTROL_WINDOWS, "paths": CONTROL_PATHS, "kl_max": CONTROL_KL_MAX},
            "predictions": [{"id": q, "text": t} for q, t in PREDICTIONS],
            "pins": {what: {"file": p.relative_to(ROOT).as_posix(), "lf_sha256": sha} for what, p, sha in PINS},
            "blobs": BLOBS, "model_pin": MODEL_PIN, "s1_build_log": S1_BUILD_LOG, "c5_sha": C5_SHA,
            "gguf_pin": {k: gc.GGUF_PIN[k] for k in ("repo", "file", "revision", "size", "sha256")},
            "x16_sha": u6e.X16_SHA, "w_sha": u6e.W_SHA, "ort_version": ORT_VERSION, "session": s1.SESSION,
            "memory": {"min_avail_gb": s1.MIN_AVAIL_GB, "watch_min_gb": s1.WATCH_MIN_GB, "min_disk_gb": MIN_DISK_GB,
                       "one_model_per_child": True},
            "probe_line_s": PROBE_LINE_S, "probe_ceiling_s": PROBE_CEILING_S, "batch": BATCH, "w_bytes": W_BYTES,
            "no_reruns": NO_RERUNS}


class Stop(Exception):
    """A pipeline check failed: the stage ends, and the reason goes to the gate."""


def lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- the F2 subgraph (one builder for model and probes)

def _t(name, dtype, dims, vals):
    from onnx import helper
    return helper.make_tensor(name, dtype, list(dims), vals)


def f2_consts() -> list:
    from onnx import TensorProto as P
    return [_t("f2/c127f", P.FLOAT, [], [127.0]), _t("f2/zerof", P.FLOAT, [], [0.0]),
            _t("f2/c255d", P.DOUBLE, [], [255.0]), _t("f2/stepd", P.DOUBLE, [], [STEP]),
            _t("f2/zerod", P.DOUBLE, [], [0.0]), _t("f2/oned", P.DOUBLE, [], [1.0]),
            _t("f2/lod", P.DOUBLE, [], [-127.0]), _t("f2/hid", P.DOUBLE, [], [127.0]),
            _t("f2/ax_last", P.INT64, [1], [-1]), _t("f2/ax_kb", P.INT64, [1], [-2]),
            _t("f2/shape_blocks", P.INT64, [4], [0, 0, -1, 32]), _t("f2/shape_k", P.INT64, [3], [0, 0, -1]),
            _t("f2/shape_row", P.INT64, [3], [0, 0, 1]), _t("f2/shape_one", P.INT64, [1], [1])]


def used_f2_consts(nodes) -> list:
    names = {i for n in nodes for i in n.input}
    return [t for t in f2_consts() if t.name in names]


def split_nodes(v: str, p: str, t: int) -> tuple:
    """t bf16 pieces of fp32 v, as u6e.split: each piece is RNE of the running residual, the Sub exact."""
    from onnx import TensorProto as P, helper
    n, out, r = [], [], v
    for i in range(1, t + 1):
        n.append(helper.make_node("Cast", [r], [f"{p}/b{i}"], name=f"{p}/CastBf{i}", to=P.BFLOAT16))
        n.append(helper.make_node("Cast", [f"{p}/b{i}"], [f"{p}/p{i}"], name=f"{p}/CastF{i}", to=P.FLOAT))
        out.append(f"{p}/p{i}")
        if i < t:
            n.append(helper.make_node("Sub", [r, f"{p}/p{i}"], [f"{p}/r{i}"], name=f"{p}/Sub{i}"))
            r = f"{p}/r{i}"
    return n, out


def step_nodes(f: str, f64: str, out: str, p: str) -> list:
    """The one-ulp step: Cast_f32(double(f) x (1 + 3 x 2^-25))."""
    from onnx import TensorProto as P, helper
    return [helper.make_node("Mul", [f64, "f2/stepd"], [p + "/fst64"], name=p + "/MulStep"),
            helper.make_node("Cast", [p + "/fst64"], [out], name=p + "/CastStep", to=P.FLOAT)]


def act_nodes(a: str, sx_in: str = None, Sx_in: str = None) -> tuple:
    """The activation side for input a [B, S, K] fp32: X' [B, S, K] double, Sx's pieces, the guard's count.
    sx_in / Sx_in replace the computed s_x [B, S, kb, 1] / Sx [B, S, 1, 1] (the selftest's doctored probes)."""
    from onnx import TensorProto as P, helper
    p = f"/f2/act{a}"
    n = [helper.make_node("Reshape", [a, "f2/shape_blocks"], [p + "/xb"], name=p + "/ReshapeBlocks")]
    if sx_in is None:
        n += [helper.make_node("Abs", [p + "/xb"], [p + "/abs"], name=p + "/Abs"),
              helper.make_node("ReduceMax", [p + "/abs", "f2/ax_last"], [p + "/amax"], name=p + "/ReduceMaxBlock",
                               keepdims=1),
              helper.make_node("Div", [p + "/amax", "f2/c127f"], [p + "/sx"], name=p + "/Sx32")]
        sx = p + "/sx"
    else:
        sx = sx_in
    n += [helper.make_node("ReduceMax", [sx, "f2/ax_kb"], [p + "/mx"], name=p + "/ReduceMaxRow", keepdims=1),
          helper.make_node("Cast", [p + "/mx"], [p + "/mx64"], name=p + "/CastMx", to=P.DOUBLE)]
    if Sx_in is None:
        n += [helper.make_node("Div", [p + "/mx64", "f2/c255d"], [p + "/q64"], name=p + "/Quotient"),
              helper.make_node("Cast", [p + "/q64"], [p + "/f"], name=p + "/CastF", to=P.FLOAT),
              helper.make_node("Cast", [p + "/f"], [p + "/f64"], name=p + "/CastF64", to=P.DOUBLE),
              helper.make_node("Mul", [p + "/f64", "f2/c255d"], [p + "/f255"], name=p + "/Check255"),
              helper.make_node("GreaterOrEqual", [p + "/f255", p + "/mx64"], [p + "/fok"], name=p + "/CheckOk")]
        n += step_nodes(p + "/f", p + "/f64", p + "/fst", p)
        n.append(helper.make_node("Where", [p + "/fok", p + "/f", p + "/fst"], [p + "/Sx"], name=p + "/RoundUp"))
        Sx = p + "/Sx"
    else:
        Sx = Sx_in
    n += [helper.make_node("Cast", [Sx], [p + "/Sx64"], name=p + "/CastSx", to=P.DOUBLE),
          helper.make_node("Mul", [p + "/Sx64", "f2/c255d"], [p + "/S255"], name=p + "/GuardS255"),
          helper.make_node("Less", [p + "/S255", p + "/mx64"], [p + "/sxbad"], name=p + "/GuardSx"),
          helper.make_node("Cast", [sx], [p + "/sx64"], name=p + "/CastSx64", to=P.DOUBLE),
          helper.make_node("Greater", [p + "/Sx64", "f2/zerod"], [p + "/pos"], name=p + "/Pos"),
          helper.make_node("Where", [p + "/pos", p + "/Sx64", "f2/oned"], [p + "/Sd"], name=p + "/GuardDiv"),
          helper.make_node("Div", [p + "/sx64", p + "/Sd"], [p + "/ratio"], name=p + "/Ratio"),
          helper.make_node("Ceil", [p + "/ratio"], [p + "/c0"], name=p + "/Ceil"),
          helper.make_node("Mul", [p + "/c0", p + "/Sx64"], [p + "/c0S"], name=p + "/UpProd"),
          helper.make_node("Less", [p + "/c0S", p + "/sx64"], [p + "/upb"], name=p + "/UpTest"),
          helper.make_node("Cast", [p + "/upb"], [p + "/up"], name=p + "/UpCast", to=P.DOUBLE),
          helper.make_node("Add", [p + "/c0", p + "/up"], [p + "/c1"], name=p + "/Up"),
          helper.make_node("Sub", [p + "/c1", "f2/oned"], [p + "/c1m"], name=p + "/DownMinus"),
          helper.make_node("Mul", [p + "/c1m", p + "/Sx64"], [p + "/c1mS"], name=p + "/DownProd"),
          helper.make_node("GreaterOrEqual", [p + "/c1mS", p + "/sx64"], [p + "/dnb"], name=p + "/DownTest"),
          helper.make_node("GreaterOrEqual", [p + "/c1", "f2/oned"], [p + "/c1ge"], name=p + "/DownAtLeast1"),
          helper.make_node("And", [p + "/c1ge", p + "/dnb"], [p + "/dn2"], name=p + "/DownAnd"),
          helper.make_node("Cast", [p + "/dn2"], [p + "/dn"], name=p + "/DownCast", to=P.DOUBLE),
          helper.make_node("Sub", [p + "/c1", p + "/dn"], [p + "/c2"], name=p + "/Down"),
          helper.make_node("Where", [p + "/pos", p + "/c2", "f2/zerod"], [p + "/c"], name=p + "/ZeroRow"),
          helper.make_node("Greater", [p + "/c", "f2/c255d"], [p + "/cbad"], name=p + "/GuardC"),
          helper.make_node("Clip", [p + "/c", "f2/oned", "f2/c255d"], [p + "/qx"], name=p + "/Qx"),
          helper.make_node("Mul", [p + "/qx", p + "/Sx64"], [p + "/eff"], name=p + "/Eff"),
          helper.make_node("Greater", [p + "/eff", "f2/zerod"], [p + "/effpos"], name=p + "/EffPos"),
          helper.make_node("Where", [p + "/effpos", p + "/eff", "f2/oned"], [p + "/effd"], name=p + "/EffGuard"),
          helper.make_node("Cast", [p + "/xb"], [p + "/x64"], name=p + "/CastX", to=P.DOUBLE),
          helper.make_node("Div", [p + "/x64", p + "/effd"], [p + "/xs"], name=p + "/DivX"),
          helper.make_node("Round", [p + "/xs"], [p + "/xr"], name=p + "/Round"),
          helper.make_node("Where", [p + "/effpos", p + "/xr", "f2/zerod"], [p + "/xr0"], name=p + "/CodeZero"),
          helper.make_node("Abs", [p + "/xr0"], [p + "/xra"], name=p + "/CodeAbs"),
          helper.make_node("Greater", [p + "/xra", "f2/hid"], [p + "/clipbad"], name=p + "/GuardCode"),
          helper.make_node("Clip", [p + "/xr0", "f2/lod", "f2/hid"], [p + "/code"], name=p + "/Clip"),
          helper.make_node("Mul", [p + "/code", p + "/qx"], [p + "/xq"], name=p + "/XPrime"),
          helper.make_node("Reshape", [p + "/xq", "f2/shape_k"], [p + "/xp"], name=p + "/ReshapeK")]
    for i, bad in enumerate((p + "/clipbad", p + "/cbad", p + "/sxbad"), 1):
        n += [helper.make_node("Cast", [bad], [f"{p}/n{i}"], name=f"{p}/CountCast{i}", to=P.INT64),
              helper.make_node("ReduceSum", [f"{p}/n{i}"], [f"{p}/s{i}"], name=f"{p}/CountSum{i}", keepdims=1)]
    n += [helper.make_node("Concat", [p + "/s1", p + "/s2", p + "/s3"], [p + "/s123"], name=p + "/CountConcat", axis=0),
          helper.make_node("ReduceSum", [p + "/s123"], [p + "/s"], name=p + "/CountAll", keepdims=1),
          helper.make_node("Reshape", [p + "/s", "f2/shape_one"], [p + "/count"], name=p + "/CountOne"),
          helper.make_node("Reshape", [Sx, "f2/shape_row"], [p + "/Sxr"], name=p + "/ReshapeSx")]
    an, apieces = split_nodes(p + "/Sxr", p + "/a", f2e.FLUSH_ORDERS[FLUSH]["t"])
    n += an
    names = {"xp": p + "/xp", "a": apieces, "count": p + "/count", "Sx": Sx, "c": p + "/c", "qx": p + "/qx",
             "code": p + "/code", "sx": sx, "counts": [p + "/s1", p + "/s2", p + "/s3"]}
    return n, names


def lin_nodes(name: str, act: dict, wp: str, dw: str, y: str) -> list:
    """One linear: I = MatMul(X', double(W')), fp32(I), then the flush (FLUSH_ORDERS[FLUSH]) into y."""
    from onnx import TensorProto as P, helper
    q = name + "/f2"
    o = f2e.FLUSH_ORDERS[FLUSH]
    t = o["t"]
    n = [helper.make_node("Cast", [wp], [q + "/w64"], name=q + "/CastW", to=P.DOUBLE),
         helper.make_node("MatMul", [act["xp"], q + "/w64"], [q + "/I"], name=q + "/MatMul"),
         helper.make_node("Cast", [q + "/I"], [q + "/fI"], name=q + "/CastI", to=P.FLOAT)]
    hn, h = split_nodes(q + "/fI", q + "/h", t)
    wn, w = split_nodes(dw, q + "/w", t)
    n += hn + wn
    a = act["a"]
    (j, k), rest = o["P"][0], o["P"][1:]
    n.append(helper.make_node("Mul", [a[j - 1], w[k - 1]], [q + "/P0"], name=q + "/P0"))
    for i, (j, k) in enumerate(rest, 1):
        n += [helper.make_node("Mul", [a[j - 1], w[k - 1]], [f"{q}/Pm{i}"], name=f"{q}/Pm{i}"),
              helper.make_node("Add", [f"{q}/P{i - 1}", f"{q}/Pm{i}"], [f"{q}/P{i}"], name=f"{q}/P{i}")]
    pn, pp = split_nodes(f"{q}/P{len(rest)}", q + "/p", t)
    n += pn
    prev = "f2/zerof"
    for i, (u, v) in enumerate(o["Y"]):
        out = y if i == len(o["Y"]) - 1 else f"{q}/Y{i}"
        n += [helper.make_node("Mul", [h[u - 1], pp[v - 1]], [f"{q}/Ym{i}"], name=f"{q}/Ym{i}"),
              helper.make_node("Add", [prev, f"{q}/Ym{i}"], [out], name=f"{q}/Y{i}")]
        prev = out
    return n


def guard_nodes(counts: list, out: str = GUARD_NAME) -> list:
    from onnx import helper
    return [helper.make_node("Concat", list(counts), ["/f2/guard/all"], name="/f2/guard/Concat", axis=0),
            helper.make_node("ReduceSum", ["/f2/guard/all"], [out], name="/f2/guard/ReduceSum", keepdims=0)]


def wprime(codes: np.ndarray, d: np.ndarray) -> tuple:
    """codes [N, kb, 32] (0..15), d [N * kb] fp16 bits -> (W' int16 [K, N], Dw fp32 [N], hybrid_f2e's w_grid)."""
    N, kb, _ = codes.shape
    dw = gc.f16_to_f32(d).reshape(N, kb)
    g = f2e.w_grid(dw, "ROW")
    wq = g["qw"][:, :, None] * (codes.astype(np.int64) - 8)
    if np.abs(wq).max(initial=0) > 8 * f2e.QW_MAX:
        raise Stop("a W' value exceeds 1,016 in magnitude")
    Wp = np.ascontiguousarray(wq.reshape(N, kb * 32).T.astype(np.int16))
    Dw = np.ascontiguousarray(g["Dw"][:, 0].astype(np.float32))
    return Wp, Dw, g


def f2_replace(m, wp_of, data):
    """Every MatMulNBits -> the F2 subgraph. wp_of(node_name) -> (W' int16 [K, N], Dw fp32 [N]); W' goes to `data`
    (an s1.DataFile), Dw inline. The activation side is built once per input (the `done` map, as S1's surgery)."""
    from onnx import NodeProto, TensorProto as P, helper, numpy_helper
    g = m.graph
    new, inits, done, counts = [], [], {}, []
    for n0 in g.node:
        n = NodeProto()
        n.CopyFrom(n0)
        if n.op_type != "MatMulNBits":
            new.append(n)
            continue
        a, y = n.input[0], n.output[0]
        attrs = {x.name: x.i for x in n.attribute}
        K, N = attrs["K"], attrs["N"]
        if a not in done:
            an, names = act_nodes(a)
            new += an
            done[a] = names
        Wp, Dw = wp_of(n.name)
        assert Wp.shape == (K, N) and Wp.dtype == np.int16 and Dw.shape == (N,) and Dw.dtype == np.float32, n.name
        tw = data.add(n.name + "/f2/wp", Wp, P.INT16, [K, N])
        td = numpy_helper.from_array(Dw, n.name + "/f2/dw")
        inits += [tw, td]
        new += lin_nodes(n.name, done[a], tw.name, td.name, y)
        counts.append(done[a]["count"])
    new += guard_nodes(counts)
    old = {x.input[k] for x in g.node if x.op_type == "MatMulNBits" for k in (1, 2)}
    keep = []
    for t0 in g.initializer:
        if t0.name not in old:
            t = P()
            t.CopyFrom(t0)
            keep.append(t)
    del g.node[:]
    g.node.extend(new)
    del g.initializer[:]
    g.initializer.extend(keep + used_f2_consts(new) + inits)
    g.output.append(helper.make_tensor_value_info(GUARD_NAME, P.INT64, []))
    return m


def linear_model(K: int, N: int, Wp: np.ndarray, Dw: np.ndarray, extra=(), sx_in=False, Sx_in=False):
    """A one-linear F2 probe: x [1, M, K] fp32 -> y [1, M, N] fp32 and f2/guard; extra names the act_nodes outputs
    to expose ("Sx", "c", "qx", "code", "counts"). sx_in / Sx_in make s_x / Sx graph inputs (doctored probes)."""
    from onnx import TensorProto as P, helper, numpy_helper
    ins = [helper.make_tensor_value_info("x", P.FLOAT, [1, "M", K])]
    if sx_in:
        ins.append(helper.make_tensor_value_info("sx", P.FLOAT, [1, "M", K // 32, 1]))
    if Sx_in:
        ins.append(helper.make_tensor_value_info("Sx", P.FLOAT, [1, "M", 1, 1]))
    an, names = act_nodes("x", "sx" if sx_in else None, "Sx" if Sx_in else None)
    nodes = an + lin_nodes("/lin", names, "wp", "dw", "y") + guard_nodes([names["count"]])
    outs = [helper.make_tensor_value_info("y", P.FLOAT, [1, "M", N]),
            helper.make_tensor_value_info(GUARD_NAME, P.INT64, [])]
    types = {"Sx": P.FLOAT, "c": P.DOUBLE, "qx": P.DOUBLE, "code": P.DOUBLE}
    for e in extra:
        if e == "counts":
            outs += [helper.make_tensor_value_info(c, P.INT64, None) for c in names["counts"]]
        elif not (e == "Sx" and Sx_in):
            outs.append(helper.make_tensor_value_info(names[e], types[e], None))
    inits = used_f2_consts(nodes) + [numpy_helper.from_array(Wp, "wp"), numpy_helper.from_array(Dw, "dw")]
    g = helper.make_graph(nodes, "f2_linear", ins, outs, inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)], ir_version=10), names


def sess_of(m, threads=s1.THREADS):
    import onnxruntime as ort
    return ort.InferenceSession(m.SerializeToString(), s1.session_options(threads), providers=["CPUExecutionProvider"])


def run_f2(sess, ids, past: dict = None, want_kv: bool = False) -> tuple:
    """s1.run's feed, plus the guard: (hidden [n, 2560] fp32, kv or None, guard int)."""
    n = len(ids)
    plen = 0 if past is None else past["past_key_values.0.key"].shape[2]
    feed = {"input_ids": np.asarray([ids], dtype=np.int64), "attention_mask": np.ones((1, plen + n), dtype=np.int64)}
    for i in sess.get_inputs():
        if i.name.startswith("past_key_values"):
            feed[i.name] = past[i.name] if past is not None else np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32)
    names = [s1.HIDDEN_NAME, GUARD_NAME] + (s1.present_names(sess) if want_kv else [])
    out = sess.run(names, feed)
    kv = None
    if want_kv:
        kv = {nm.replace("present.", "past_key_values."): o for nm, o in zip(names[2:], out[2:])}
    return np.ascontiguousarray(out[0][0], dtype=np.float32), kv, int(out[1])


# ---------------------------------------------------------------- K2 and K3's comparators

def k2_compare(base, arm) -> dict:
    """F2 against R0: nodes equal outside the F2 subgraphs, inputs equal, outputs equal but for f2/guard (last),
    shared initializers equal (same location, offset and length), every linear rewired to its F2 subgraph."""
    bnb = [n for n in base.graph.node if n.op_type == "MatMulNBits"]
    bn = [n for n in base.graph.node if n.op_type != "MatMulNBits"]
    an = [n for n in arm.graph.node if F2_TAG not in n.name and n.op_type != "MatMulNBits"]
    res = {"base_has_no_f2_names": not any(F2_TAG in n.name for n in base.graph.node),
           "inputs_equal": [i.SerializeToString() for i in base.graph.input] ==
           [i.SerializeToString() for i in arm.graph.input],
           "outputs_equal_but_guard": [o.SerializeToString() for o in base.graph.output] ==
           [o.SerializeToString() for o in arm.graph.output if o.name != GUARD_NAME]
           and [o.name for o in arm.graph.output].count(GUARD_NAME) == 1
           and arm.graph.output[-1].name == GUARD_NAME,
           "nodes_equal": [n.SerializeToString() for n in bn] == [n.SerializeToString() for n in an],
           "no_matmulnbits": not any(n.op_type == "MatMulNBits" for n in arm.graph.node)}
    old = {x.input[k] for x in bnb for k in (1, 2)}
    b_init = {t.name: t.SerializeToString() for t in base.graph.initializer if t.name not in old}
    a_init = {t.name: t.SerializeToString() for t in arm.graph.initializer}
    res["shared_initializers_equal"] = all(a_init.get(k) == v for k, v in b_init.items())
    extra = [k for k in a_init if k not in b_init]
    res["new_initializers_are_f2"] = all(k.startswith("f2/") or F2_TAG in k for k in extra) and not (set(a_init) & old)
    made = {o for n in arm.graph.node if F2_TAG in n.name for o in n.output}
    mm = {n.name: n for n in arm.graph.node if n.op_type == "MatMul" and F2_TAG in n.name}
    wired = len(mm) == len(bnb)
    for b in bnb:
        mnode = mm.get(b.name + "/f2/MatMul")
        wired &= mnode is not None and mnode.input[1] == b.name + "/f2/w64" and b.output[0] in made
        wired &= b.name + "/f2/wp" in a_init and b.name + "/f2/dw" in a_init
    res["linears_rewired"] = wired
    res["linears"] = len(bnb)
    res["ok"] = all(v for k, v in res.items() if k != "linears")
    return res


def census_of(prof_path: str) -> dict:
    """From an ORT profile: executed nodes by op type, and the MatMuls whose inputs are all double."""
    ev = json.loads(Path(prof_path).read_text(encoding="utf-8"))
    ops, dbl = {}, 0
    for e in ev:
        if e.get("cat") == "Node" and str(e.get("name", "")).endswith("_kernel_time"):
            a = e.get("args", {})
            op = a.get("op_name")
            ops[op] = ops.get(op, 0) + 1
            if op == "MatMul":
                types = [k for d in a.get("input_type_shape", []) for k in d]
                dbl += bool(types) and all(t == "double" for t in types)
    return {"executed": dict(sorted(ops.items())), "matmul_double": dbl}


def census_ok(c: dict, static: dict, r0_static: dict, f2_static_rest: dict, linears: int) -> dict:
    """K3: the executed ops equal the graph's; `linears` MatMuls, all double; no MatMulNBits; no fused op; every op
    outside the F2 subgraphs at R0's count (MatMulNBits aside)."""
    ex = c["executed"]
    r0_rest = {k: v for k, v in r0_static.items() if k != "MatMulNBits"}
    res = {"executed_equal_static": ex == static, "matmul": ex.get("MatMul", 0) == linears,
           "matmul_double": c["matmul_double"] == linears, "no_matmulnbits": "MatMulNBits" not in ex,
           "no_fused": not any(f in ex for f in s1.FUSED_FORBIDDEN), "rest_equal_r0": f2_static_rest == r0_rest}
    res["ok"] = all(res.values())
    return res


def static_rest(m) -> dict:
    """Op counts of the nodes outside the F2 subgraphs."""
    c = {}
    for n in m.graph.node:
        if F2_TAG not in n.name and n.op_type != "Constant":
            c[n.op_type] = c.get(n.op_type, 0) + 1
    return dict(sorted(c.items()))


# ---------------------------------------------------------------- the text (16 windows of the validation split)

def val_file() -> Path:
    """The validation parquet at the pinned revision, fetched without a token. A Hub that asks for one is a STOP."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError,
                                        RepositoryNotFoundError)
    t = VAL_PIN
    try:
        p = Path(hf_hub_download(t["repo"], t["file"], revision=t["revision"], repo_type="dataset", token=False))
    except (GatedRepoError, RepositoryNotFoundError) as e:
        raise Stop(f"the Hub asked for a token or hid the dataset ({type(e).__name__}); come to the gate") from None
    except LocalEntryNotFoundError:
        raise Stop("the validation text is not in the local cache and was not fetched (offline, or the Hub "
                   "unreachable)") from None
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in (401, 403):
            raise Stop(f"the Hub asked for a token (HTTP {code}); come to the gate") from None
        raise Stop(f"the download failed (HTTP {code})") from None
    except Exception as e:
        raise Stop(f"the download failed ({type(e).__name__})") from None
    size, sha = p.stat().st_size, s1.sha_file(p)
    s1.say("VAL_FILE_JSON", {"file": t["file"], "revision": t["revision"], "bytes": size, "sha256": sha,
                             "bytes_pinned": t["size"], "sha256_pinned": t["sha256"],
                             "equal": size == t["size"] and sha == t["sha256"]})
    if size != t["size"] or sha != t["sha256"]:
        raise Stop("the validation text differs from its pin")
    return p


def val_tokens() -> list:
    import pyarrow.parquet as pq
    text = "\n\n".join(pq.read_table(val_file()).column("text").to_pylist())
    return gds.quiet_tokenizer()(text, add_special_tokens=False)["input_ids"]


def tokenizer_pins() -> dict:
    """S1's tokenizer pins (the text-only copy's files: sizes, LFS sha256, git blob sha1, config.json)."""
    res, t = {}, gd.TEXT
    for f, (size, sha) in gd.CK_LFS.items():
        if f.endswith(".safetensors"):
            continue
        p = t / f
        res[f] = p.exists() and p.stat().st_size == size and s1.sha_file(p) == sha
    for f, (size, sha1) in gd.CK_BLOB.items():
        if f in ("config.json", "model.safetensors.index.json"):
            continue
        p = t / f
        ok = p.exists() and p.stat().st_size == size
        if ok:
            b = p.read_bytes()
            ok = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == sha1
        res[f] = ok
    want = s1.logged_json(s1.C_BUILD_LOG, "TEXTONLY_JSON")["config"]
    p = t / "config.json"
    res["config.json"] = p.exists() and json.loads(p.read_text(encoding="utf-8")) == want
    return {"files": res, "ok": all(res.values())}


def prior_prompt_shas() -> tuple:
    s1_seq = s1.logged_json(s1.latest("hybrid_s1_prereg_*.log"), "TEXT_JSON")["seq_sha"]
    s1b_prompt = s1.logged_json(s1.latest("hybrid_s1b_prereg_*.log"), "TEXT_JSON")["prompt_sha"]
    return s1_seq, s1b_prompt


def text_pins(T: list, prior: tuple) -> dict:
    offs = s1b.offsets(N_WIN, 0, WIN_STRIDE, WIN_LEN - 1)
    wins = [[BOS] + T[a:b] for a, b in offs]
    prompt_sha = [gds.ids_sha(w[:PROMPT]) for w in wins]
    asserts = s1b.window_asserts(offs, len(T), 0, prompt_sha, prior[0] + prior[1])
    asserts["windows_full"] = all(len(w) == WIN_LEN for w in wins)
    asserts["ok"] = asserts["ok"] and asserts["windows_full"]
    return {"text_tokens": len(T), "offsets": offs, "window_sha": [gds.ids_sha(w) for w in wins],
            "prompt_sha": prompt_sha, "all_sha": gds.ids_sha([i for w in wins for i in w]), "first": offs[0][0],
            "end": offs[-1][1], "left": len(T) - offs[-1][1], "asserts": asserts,
            "prior": {"s1_seq_sha": len(prior[0]), "s1b_prompt_sha": len(prior[1])}}


def pinned_windows() -> list:
    """The 16 windows, checked against the committed F2-A prereg log's TEXT_JSON."""
    tp = tokenizer_pins()
    if not tp["ok"]:
        raise Stop(f"a tokenizer file differs from S1's pins: {[k for k, v in tp['files'].items() if not v]}")
    T = val_tokens()
    got = text_pins(T, prior_prompt_shas())
    want = s1.logged_json(s1.latest("hybrid_f2a_prereg_*.log"), "TEXT_JSON")
    for k in ("text_tokens", "offsets", "window_sha", "prompt_sha", "all_sha"):
        if json.loads(json.dumps(got[k])) != want[k]:
            raise Stop(f"the ids differ from the F2-A prereg's TEXT_JSON ({k})")
    if not got["asserts"]["ok"]:
        raise Stop("the windows fail their asserts")
    return [[BOS] + T[a:b] for a, b in got["offsets"]]


# ---------------------------------------------------------------- the metrics (frozen: VERDICT_FUNCS)

def control_outcome(rows: list) -> dict:
    """rows: one {"w", "a", "b", "c"} per control window, each path {"max_kl", "sha_equal"}. PASS iff every
    control window is present and every path's max KL is finite and <= CONTROL_KL_MAX."""
    got = {r["w"] for r in rows}
    failed = [[r["w"], p] for r in rows for p in CONTROL_PATHS
              if not (np.isfinite(r[p]["max_kl"]) and r[p]["max_kl"] <= CONTROL_KL_MAX)]
    missing = sorted(set(CONTROL_WINDOWS) - got)
    return {"outcome": "PASS" if not failed and not missing else "FAIL", "failed": failed, "missing": missing,
            "sha_equal": {p: all(r[p]["sha_equal"] for r in rows) for p in CONTROL_PATHS} if rows else {}}


def outcome_kv(m: dict, complete: bool) -> dict:
    """F2-A/KV: S1b's rule_set on set C. INCOMPLETE if a window's set C is missing."""
    if not complete or "C" not in m:
        return {"outcome": "INCOMPLETE", "narrow": False, "label": "INCOMPLETE"}
    o, nb = s1b.rule_set(m["C"])
    return {"outcome": o, "narrow": bool(nb), "label": o + (" NARROW" if nb else "")}


def outcome_full(m: dict, complete: bool) -> dict:
    """F2-A/FULL: S1's rule on bands L and H (a non-finite value in either band is a FAIL). INCOMPLETE if a
    window's prompt states are missing."""
    if not complete or "L" not in m or "H" not in m:
        return {"outcome": "INCOMPLETE", "narrow": [], "label": "INCOMPLETE"}
    o, nb = s1.rule({"finite": bool(m["L"]["finite"] and m["H"]["finite"]), "L": m["L"], "H": m["H"]})
    return {"outcome": o, "narrow": nb, "label": o + (" NARROW" if nb else "")}


def combination(kv: str, full: str) -> str:
    """kv, full: each verdict's outcome (PASS, FAIL or INCOMPLETE); NARROW changes neither."""
    if kv == "INCOMPLETE":
        return COMBINATIONS["KV INCOMPLETE"]
    if kv == "FAIL":
        return COMBINATIONS["KV FAIL"]
    if full == "PASS":
        return COMBINATIONS["BOTH PASS"]
    if full == "FAIL":
        return COMBINATIONS["KV PASS, FULL FAIL"]
    return COMBINATIONS["FULL INCOMPLETE"]


def headline(kv: dict, full: dict) -> list:
    """The headline, KV first (HEADLINE_ORDER), then the combination and the N2 note."""
    return [f"F2-A/KV (set C, S1b's rule): {kv['label']}", f"F2-A/FULL (bands L and H, S1's rule): {full['label']}",
            combination(kv["outcome"], full["outcome"]), NO_PICK_CHANGE]


def score_predictions(metrics: dict, kv: dict, full: dict, would: dict, guards_zero) -> dict:
    """FA-P1a...FA-P8, HIT or MISS, NOT SCORED where an input is missing. would[arm] = {"KV": .., "FULL": ..}."""
    def s(cond, need):
        if not all(need):
            return "NOT SCORED"
        return "HIT" if cond() else "MISS"
    f2, r, n2 = metrics.get("F2", {}), metrics.get("R", {}), metrics.get("N2", {})
    have = lambda m, *ks: all(k in m for k in ks)  # noqa: E731
    wr, wn = would.get("R", {}), would.get("N2", {})
    return {
        "FA-P1a": s(lambda: full["label"] == "PASS", [full["outcome"] != "INCOMPLETE"]),
        "FA-P1b": s(lambda: kv["label"] == "PASS", [kv["outcome"] != "INCOMPLETE"]),
        "FA-P2": s(lambda: all(f2[k]["kl_mean"] <= FA_P2_FACTOR * r[k]["kl_mean"] for k in ("L", "H", "C")),
                   [have(f2, "L", "H", "C"), have(r, "L", "H", "C")]),
        "FA-P3": s(lambda: all(f2[k]["top1"] >= FA_P3_TOP1 for k in ("L", "H")), [have(f2, "L", "H")]),
        "FA-P4": s(lambda: f2["C"]["kl_mean"] < n2["C"]["kl_mean"], [have(f2, "C"), have(n2, "C")]),
        "FA-P5": s(lambda: wr["KV"]["outcome"] == "PASS" and wr["FULL"]["outcome"] == "PASS",
                   [bool(wr) and wr["KV"]["outcome"] != "INCOMPLETE" and wr["FULL"]["outcome"] != "INCOMPLETE"]),
        "FA-P6": s(lambda: wn["FULL"]["outcome"] == "FAIL" and wn["KV"]["outcome"] == "PASS",
                   [bool(wn) and wn["KV"]["outcome"] != "INCOMPLETE" and wn["FULL"]["outcome"] != "INCOMPLETE"]),
        "FA-P7": s(lambda: f2["P_R0"]["finite"] and f2["P_R0"]["kl_mean"] <= T_KL and f2["P_R0"]["top1"] >= T_TOP1,
                   [have(f2, "P_R0")]),
        "FA-P8": s(lambda: guards_zero, [guards_zero is not None]),
    }


VERDICT_FUNCS = ("control_outcome", "outcome_kv", "outcome_full", "combination", "headline", "score_predictions")
S1B_VERDICT_FUNCS = ("offsets", "window_asserts", "slice_kv", "window_metrics", "set_aggregate", "halves", "rule_set")
S1_VERDICT_FUNCS = ("log_softmax_rows", "logits_chunk", "kl_top1_chunk", "band_rows", "bootstrap", "aggregate", "rule",
                    "max_kl")


def verdict_constants() -> dict:
    return {"ARMS": ARMS, "DECIDING": DECIDING, "REPORT_ONLY": REPORT_ONLY, "BOS": BOS, "PROMPT": PROMPT,
            "WIN_LEN": WIN_LEN, "WIN_STRIDE": WIN_STRIDE, "N_WIN": N_WIN, "SET_C": SET_C, "P_POS": P_POS,
            "BANDS": BANDS, "Q2_SPLIT": Q2_SPLIT, "SETS": SETS, "CONTROL_WINDOWS": CONTROL_WINDOWS,
            "CONTROL_PATHS": CONTROL_PATHS, "CONTROL_KL_MAX": CONTROL_KL_MAX, "T_KL": T_KL, "T_TOP1": T_TOP1,
            "SEED": SEED, "BOOT_N": BOOT_N, "CHUNK": CHUNK, "MODEL_PIN": MODEL_PIN, "VAL_PIN": VAL_PIN,
            "RULE_KV": RULE_KV, "RULE_FULL": RULE_FULL, "HEADLINE_ORDER": HEADLINE_ORDER,
            "COMBINATIONS": COMBINATIONS, "NO_PICK_CHANGE": NO_PICK_CHANGE, "PREDICTIONS": PREDICTIONS,
            "FA_P2_FACTOR": FA_P2_FACTOR, "FA_P3_TOP1": FA_P3_TOP1, "K4_TOL": K4_TOL, "K4_PATH": K4_PATH,
            "C5_SHA": C5_SHA, "ARM_KEY": ARM_KEY, "S1B_CONSTANTS": {"SET_C": s1b.SET_C, "P_POS": s1b.P_POS,
                                                                     "BANDS": s1b.BANDS, "CHUNK": s1b.CHUNK,
                                                                     "Q2_SPLIT": s1b.Q2_SPLIT}}


def verdict_sources() -> list:
    """(name, source) for every function the verdict code is: F2-A's own, then the imported S1b and S1 kernels.
    inspect.getsource reads with universal newlines, so CRLF and LF checkouts agree."""
    g = globals()
    return [(f, inspect.getsource(g[f])) for f in VERDICT_FUNCS] + \
           [(f"hybrid_s1b.{f}", inspect.getsource(getattr(s1b, f))) for f in S1B_VERDICT_FUNCS] + \
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


# ---------------------------------------------------------------- the plan (PREREG: the accepted text, verbatim)

PREREG = r"""# F2-A, pre-registration text, draft v2: F2's arithmetic at model level, two verdicts on new text (2026-09-25)

Status: TEXT ONLY. Nothing is coded, built or run, and no new text is downloaded or tokenized. The user's decision
(2026-09-25, relayed by the gate): "Plan F2-A". v2 applies the gate's rulings and fixes on v1 (LF a0cbad8c, ACCEPTED
WITH RULINGS AND FIXES). Section 14 holds the rulings as applied, and section 15 the changes. The gate re-reads the
changed lines; then the tool and its stages go as a diff for the gate's go, and each run needs the gate's go. Push
is the gate's.

**This text amends the F2 plan's section 4** (plan lines 266–272: an S1-pattern run with S1's bar, on U6-A's
pattern, which is bands L and H alone). By the gate's ruling 1, F2-A's deciding set is two pre-registered verdicts:
F2-A/KV on set C with S1b's rule, and F2-A/FULL on bands L and H with S1's rule, in place of S1's bar alone. The
plan file stays unedited at b98787f4, as the F2-E addendum amended sections 1 and 3.

Sources:
- the F2 plan (v2 final, git-ignored, LF b98787f4): section 4's outline, and section 3's rule;
- the F2-E addendum (LF 1ad61601) and F2-E's run log (96f83a8, LF da97478e);
- the S1 plan and prereg (tools/hybrid_s1.py's PREREG, frozen at c67d6d2), and the S1b plan and prereg
  (tools/hybrid_s1b.py, frozen at b777070);
- the U6 plan (LF 830eb8b9), section 4's U6-A;
- docs/BENCHMARKS.md and docs/DECISIONS.md at 120bf29.

Tags: MEASURED (a committed log), DERIVED (arithmetic on measured or SPEC figures), SPEC (a vendor source or the
repo's code, read on the date given), INFERRED, ESTIMATE. GB = 1e9 bytes.

## 0. What F2-A asks, and what it cannot change

- **Two questions, two pre-registered verdicts** (the gate's ruling 1, 2026-09-25; the user's to override before
  the tool is frozen, as F2_FACTOR was):
  - **F2-A/KV:** when F2's arithmetic builds the prompt's KV in every linear the NPU would run, does the
    continuation stay within S1b's rule on set C? That is the role F2 would play in the user's hybrid: the NPU builds
    the prompt's KV, and the GPU does the rest. It is the basis the user's N2 move rests on (S1b: set C PASS while L
    and H FAIL).
  - **F2-A/FULL:** does F2's arithmetic also keep the model within S1's bar at every prompt position (bands L and
    H)? That is the stricter question: could F2 also produce the prompt's own logits?
  - Neither is a sub-case of the other, and neither is ever re-scored. The headline prints KV first.
- **The arm is F2-E's pick:** form B, FLUSH_B2, S = ROW, E_F2 23.28125 (F2-E's log, line 133; MEASURED pick,
  DERIVED E_F2). F2-E screened it per linear: 1.047–1.189 × r_R over 21 cases, on layer 16's inputs only (MEASURED).
  F2-A is the model-level test that screen could not be.
- **F2-A cannot change the N2 pick.** F2's best modelled energy, 2.7292 mJ with idle per prompt token per layer
  (compute only, at an assumed power; DERIVED), is 1.67× N-i8's MEASURED 1.6334 (DECISIONS at 120bf29).
  - F2's case is weight memory (the user's decision, 2026-09-25). F2-A answers only whether F2's arithmetic holds
    at model level, a precondition of that case. It measures no bytes.
- **The F2 plan's factor** (F2_FACTOR 2.0, "the user may override it before F2-A's prereg") has done its job: F2-E's
  pick stands at 2.0 and at 1.41. F2-A decides on S1b's and S1's rules, not on that factor.

## 1. The arm, its linears, and what stays the reference's

- **The linears: all 238 MatMulNBits nodes, the seven linears (q, k, v, o, gate, up, down) of all 34 layers.**
  - These are the hybrid's NPU GEMMs: the NPU runs the seven weight GEMMs per layer, and the 780M runs attention
    (the U6 plan, section 0; S1b's plan, section 1). This is S1's replace scope (S1 plan, section 4).
  - The head is not among them. It is applied in the verdict, in fp32, identically for every model (S1's C3).
- **What stays the reference's (R0's; the gate's ruling 2):** everything outside the 238 linears, in fp32 on the
  CPU EP, node for node as in S1's graphs: the embedding, the norms, QK-norm, RoPE, GQA, GELU and the residuals.
  - R0 also runs the continuation (set C) and the last prompt position (P-R0), standing in for the 780M, as in S1b
    (its scope (a)). N2's move rests on that basis.
  - No continuation runs R's arithmetic (accuracy level 4). That would break comparability with S1b's basis. Only
    the report-only R arm runs level 4, in its own prompt.
- **F2's arithmetic, per linear** (F2-E's, at S = ROW; the activation side is per row, so the arm is
  batch-invariant, and the KV slice of section 4 is exact, as S1b's argument for N2; DERIVED):
  1. **s_x, MLAS's block scale:** per 32-block of the row, amax / 127 in fp32 (S1's q8_ort).
  2. **Sx, rounded up:** the smallest fp32 with 255 · Sx ≥ the row's max s_x (section 2.2).
  3. **qx = max(1, c),** with c the exact ceiling of s_x / Sx (section 2.2).
  4. **The codes:** RNE(x / (qx · Sx)), in float64, clipped to ±127 as a guard.
  5. **The weights, offline in the build:** Dw rounded up (127 · Dw ≥ max |d_w| over the row of K), qw = RNE(d_w /
     Dw), by hybrid_f2e's own functions (frozen at f88fe73). The 4-bit codes are the release's.
  6. **I = Σ_k X′[m,k] · W′[k,n],** with X′ = qx · code and W′ = qw · (c − 8), exact (section 2.1). It is F2-E's I,
     since Σ_b qx_b · qw_b · i_b regroups to this sum.
  7. **The flush:** fp32(I) = RNE(I), then f2_flush_b.cc's B2, in ORDER[2]'s term order, once per output (S = ROW).
- **This subgraph is the host quantizer F2-E said a host must reproduce** (BENCHMARKS at 120bf29, F2-E's "What this
  does not establish"): its codes are computed in float64, and Sx is rounded up.

## 2. The arm's graph

The graph is S1's base (S1's base_cut of (c)'s C0-H16, reused), with each MatMulNBits replaced by the F2 subgraph.
S1's code is imported read-only; the replace step is new code in tools/hybrid_f2a.py.

### 2.1 The exact integer sum: a float64 MatMul (the gate's ruling 4)

- **Neither factor fits int8,** so MatMulInteger cannot carry I: |X′| ≤ 127 × 255 = 32,385 and |W′| ≤ 8 × 127 =
  1,016 (DERIVED).
- **X′ and W′ as float64, and one MatMul(double) per linear.**
  - ORT's CPU EP registers MatMul for tensor(double), and for int32, int64, uint32 and uint64 (SPEC:
    docs/OperatorKernels.md at rel-1.23.2, read 2026-09-25; this build is 1.23.3.dev20260320, and the selftest
    confirms it).
  - **Exact in any order** (DERIVED): each product is an integer with |X′ · W′| ≤ 32,903,160 < 2^25, and each
    partial sum an integer of magnitude at most 10,240 × 32,903,160, about 3.4e11 < 2^39 (the F2 plan, section 1:
    320 blocks × 1,052,901,120). Every value is an integer below 2^53, so float64 holds it exactly, whatever the
    summation order, blocking or FMA.
  - The selftest checks I against Python integers, as F2-E's did, and keeps MatMul(int64)'s result on record.
- **W′ is stored as int16** (|W′| ≤ 1,016) and cast to float64 at run time, as N3 casts its bf16 W.
  - 3,208,642,560 weights (94,371,840 per layer × 34; DERIVED) at 2 B is 6,417,285,120 B. That is the byte count
    of S1's bf16.onnx.data (MEASURED, the r2 build log's BUILD_JSON), the same element count at 2 B.
  - Dw is one fp32 per output column per linear, about 4 MB in all (DERIVED).
- **X′** is qx · code in float64, from the codes of 1.4.

### 2.2 The round-up and the exact ceiling in ONNX (the gate's ruling 5)

ONNX has no nextafter, and Cast(double → float) rounds to nearest (SPEC). F2-E's Sx is the smallest fp32 at or
above the quotient.
- **Sx:**
  - q = double(max s_x) / 255 in float64, and f = Cast_f32(q);
  - if 255 · double(f) ≥ double(max s_x), then Sx = f. The product is exact: 8 bits × 24 bits;
  - else Sx = Cast_f32(double(f) · (1 + 3 · 2^-25)), one fp32 ulp above f.
- **Why that steps exactly one ulp** (DERIVED, for a positive normal f with ulp u; re-derived by the gate):
  - f · 3 · 2^-25 lies in [0.75 u, 1.5 u), and the float64 product is exact (24 + 26 bits);
  - so f + that offset sits within 0.5 u of f + u, never at a tie, and RNE takes f + u. At a binade's top, f + u
    is the power of two, and the same holds;
  - Cast-then-step equals F2-E's round-up (DERIVED). Let t be the true quotient, so q = RN64(t) and f = RN32(q).
    Both roundings are monotone and every fp32 is a float64, so no fp32 lies strictly between f and t. If the check
    passes, f ≥ t and f is the smallest fp32 ≥ t. If it fails, f < t, and the step lands on the smallest fp32 ≥ t;
  - a subnormal f is not stepped by this. It would fail the guard (2.3) and STOP. A subnormal Sx needs a row's max
    s_x below 255 × 2^-126, about 3.0e-36 (DERIVED); a row of exact zeros is handled below.
- **The exact ceiling:** c0 = Ceil(double(s_x) / double(Sx)); then c = c0 + [c0 · Sx < s_x] − [(c0 − 1) · Sx ≥ s_x].
  Each product is exact in float64 (9 bits × 24). The correction terms are hybrid_f2e's ceil_exact.
- **A zero row** (every s_x zero, so Sx = 0) takes qx = 1 and codes 0, with the divisor guarded as S1's N2 guards a
  zero scale. Its outputs are 0.
- The selftest must show the in-graph Sx, c and qx equal to hybrid_f2e's fp32_up, scale_up and ceil_exact, bit for
  bit, before the tool is frozen.

### 2.3 The guard (the gate's ruling 6)

F2-E made any clip or scale fault on real data a STOP (the gate's fix F3). F2-A carries it into the graph:
- **Each F2 subgraph counts, per run:**
  - the codes whose rounded value exceeds ±127 before the clip;
  - the blocks with c > 255;
  - the rows whose Sx fails 255 · Sx ≥ max s_x.
- **The 238 counts are summed into one extra graph output, int64, and every run must read 0.** A nonzero reading
  is a STOP. It is DERIVED never to fire (F2-E's addendum, section 2.2).
- **The count is a pipeline fact, not an accuracy figure,** so the states print it (blind-safe; ruling 6).
- **The weight side is checked in the build, offline,** by hybrid_f2e's scale_faults and its qw-clip STOP, on all
  238 linears. Any fault is a STOP.

### 2.4 The flush

- fp32(I) = Cast_f32(I), RNE (SPEC; the selftest checks ties).
- B2 as f2_flush_b.cc (LF caa62ce0) and hybrid_f2e's flush:
  - the split is Cast(float → bfloat16 → float) with an exact fp32 Sub;
  - products of bf16 pieces are exact in fp32;
  - the adds are fp32 Adds, one node each, in ORDER[2]'s order.
- ORT_DISABLE_ALL (S1's session options) means no node is fused, so no add is contracted.
- The Cast to bfloat16 rounds to nearest even on this build (MEASURED: S1's C1 checked N3's cast X against
  ml_dtypes' RNE).

## 3. The reference, and its anchors (the check stage; all before any states)

**The reference is R0,** S1's R0.onnx (MatMulNBits at accuracy level 0: exact d · (q − 8) against fp32
activations), reused and not rebuilt.
- It is refused unless its sha equals the r2 BUILD_JSON: R0.onnx a0a7d785…, with the base data per S1's SRC_PIN.
- The scored reference is R0 one-shot over each window's 3,072 ids (S1b's form). Its first 2,048 positions are the
  reference for bands L and H, and its positions 2,047–3,070 for set P and set C.

**The anchors.** Each prints PASS or FAIL in the check log, and its values go into the verdict log, with two
exceptions printed in the check log: K4's worst difference, which concerns only F2-E's published cases, and K7's max
KL, R0 against R0. Any FAIL is a STOP.
- **K1, the pins** (section 8), and the frozen triple.
- **K2, graph identity (S1's C3):**
  - the F2 graph equals R0 node for node outside the 238 replaced linears;
  - its shared initializers point at the same bytes of the same data file;
  - each linear's W′ and Dw equal their re-derivation from the GGUF tensor the node names, by sha.
- **K3, the census (S1's SURVIVE):** the build's profiled pass on S1's test sequence 0 executes 238 MatMul(double)
  and no MatMulNBits, and no fused op (S1's FUSED_FORBIDDEN). Every op outside the F2 subgraphs has R0's count.
- **K4, the link to F2-E, bit for bit:**
  - For each of the 21 cases F2-E read (layers 0, 16 and 33 × the seven linears), a one-linear probe is built by
    the same F2 builder. It runs on layer 16's captured x16_* (S1's sequence 0, U6-E's X16_SHA pins).
  - Its output on F2-E's 256 rows must equal hybrid_f2e's B/B2/ROW emulation bit for bit.
  - Its rel-L2 against L0 on those rows must reproduce F2-E's logged r_F2 (the run log's CASE_JSON, 96f83a8)
    within 1e-6 relative. The worst relative difference is printed.
  - This ties F2-A's arm to the arm F2-E screened. It covers the grid, the integer sum, the conversion and the
    bf16 casts on real inputs.
- **K5, R0's tie to S1:** R0 on S1's test sequence 0 reproduces S1's C5 hidden-state sha, a5597e1f… (MEASURED, S1's
  check log, line 18). That re-proves R0 in this sitting's environment. S1's own states made the same tie to its C5
  (BENCHMARKS, S1's checks).
- **K6, determinism (S1's C5'):** the F2 arm runs window 0 twice, and the two hidden-state shas must be equal.
- **K7, the R0 control (S1b's), which guards F2-A/KV's path:**
  - on windows 0, 2, 4, 6, 8, 10, 12 and 14, paths (a), (b) and (c): (a) R0's prompt KV then R0's continuation,
    (b) the KV sliced to 0–2,046 plus one decode step, (c) R0's 2,048-id prompt against the one-shot's first
    2,048;
  - each against R0 one-shot, at max per-position KL ≤ 1e-5, with sha-equality reported. Each path's max KL is
    printed, as S1b's check printed it: R0 against R0, with no arm figure in it.
  - No R control (ruling 2).
- **K8, the guard,** in every F2 run of the check and the states: 0.

## 4. Text, sample, positions and windows

- **Corpus: WikiText-2 raw, validation split, new to this study** (U6-A's proposal). S1 and S1b read the test split.
  - Salesforce/wikitext @ b08601e04326c79dfdd32d625aee71d232d685c3;
  - wikitext-2-raw-v1/validation-00000-of-00001.parquet;
  - 657,209 B, sha256 204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c (SPEC: the Hub's API
    listing at that revision, read 2026-09-25). The same listing gives the test file's size and sha exactly as S1
    pinned them (732,610 B, 5f1bea06…).
  - It is not in the local cache. The prereg downloads it at the pinned revision (ruling 10), refuses a size or sha
    mismatch, and prints both. The dataset is public: if the Hub asks for a token, the prereg STOPs and comes to the
    gate. No token is ever printed.
- **Tokenizer and joining:** S1's exactly. The text column is joined with "\n\n" and tokenized with
  add_special_tokens=False. The tokenizer files are S1's pins. The prereg prints the token count; it is not known
  now (ESTIMATE, scaling the test split's 292,282 tokens by file size: about 262,000).
- **The windows (S1b's form):** window w is [BOS] + T[3,071 · w : 3,071 · (w + 1)], 3,072 ids. The prompt is ids
  0–2,047, and the continuation is ids 2,048–3,071.
  - **N = 16, w = 0..15** (S1's N; ruling 3). That uses 49,136 tokens.
  - A NARROW stands as the verdict. A larger sample later is a new pre-registered experiment on the user's go,
    never a re-run (ruling 3).
  - The prereg asserts that the windows are pairwise disjoint, and that no window's 2,048-id prompt hashes to any
    of S1's seq_sha or S1b's window prompt shas. TEXT_JSON pins the token count, the offsets and every window's
    sha.
- **The positions** (i is the logits after ids[0..i], predicting ids[i + 1]):
  - **Set C:** i = 2,048–3,070, R0 run teacher-forced on the arm's full 2,048-position prompt KV. It decides
    **F2-A/KV**.
  - **Band L:** i = 0–1,023. **Band H:** i = 1,024–2,046. These are S1's bands, on the F2 arm's own logits. They
    decide **F2-A/FULL**.
  - **Set P:** i = 2,047, both ways, report-only as in S1b. P-arm is the arm's own logits. P-R0 is R0 on the arm's
    KV sliced to 0–2,046, one decode step for id 2,047: the hybrid's first token under the user's N2 design. At N =
    16, set P's top-1 moves in steps of 1/16.

## 5. The two rules, and what each outcome means

Both rules are the gate's ruling 1, pre-registered here and frozen in the tool (section 8). The bootstrap for each
is over windows: 10,000 resamples, seed 20260924, a 95% interval.
- **F2-A/KV, on set C: S1b's rule, verbatim.** PASS if both hold on set C: mean KL ≤ 0.0123 and top-1 agreement ≥
  0.956. FAIL otherwise, and FAIL on a non-finite value.
  - NARROW names a threshold that lies inside the 95% bootstrap interval. It is reported and changes nothing, as in
    S1. The point estimate decides.
  - INCOMPLETE: a window's set C states for the F2 arm are missing when the gate rules that the verdict runs.
- **F2-A/FULL, on bands L and H: S1's rule, verbatim.** Against R0, in both bands L and H, pooled over the 16
  windows:
  - PASS if mean KL ≤ 0.0123 and top-1 agreement ≥ 0.956;
  - FAIL otherwise, or on any non-finite hidden state or logit;
  - NARROW if a threshold lies inside the bootstrap interval; the verdict stands on the point estimate;
  - INCOMPLETE: a window's F2 prompt states are missing when the gate rules that the verdict runs.
- **Each verdict reads PASS, PASS NARROW, FAIL, FAIL NARROW or INCOMPLETE.** The headline prints F2-A/KV's line
  first, then F2-A/FULL's.
- **STOP** (a pipeline defect, not a verdict; it goes to the gate as it stands):
  - a failed anchor (K1–K8);
  - a scale fault or a clip in the build's weight grid;
  - a frozen hash, model sha or text pin that differs;
  - a child that exits nonzero;
  - the Hub asking for a token.
- **What each combination means:**
  - **KV PASS, FULL FAIL:** N2's position. F2 qualifies for the KV-serving role on the same pre-registered basis as
    N2. Any move is the user's.
  - **KV FAIL** (whatever FULL reads): F2 does not qualify for the role N2 fills.
  - **Both PASS:** F2 qualifies for the KV-serving role, and also holds at prompt positions.
  - Neither verdict is ever re-scored, and a NARROW does not change either.
- **The verdict does not change the N2 pick.**

## 6. Report-only

- **N2's set C on the new windows, printed beside F2-A/KV** (ruling 7). It is the user's pick, on the same text:
  the like-for-like reading.
- **The R arm** (S1's R.onnx, accuracy level 4, reused by sha): bands L and H, set C and set P, with its would-be
  outcomes under both rules. It is the arithmetic F2 approximates. S1 read it at mean KL 0.00108 / 0.00098 and
  top-1 0.9868 / 0.9862 on the test split (MEASURED, report-only).
- **The N2 arm** (S1's N2.onnx, reused by sha): bands L and H and set P too, with its would-be outcomes. N3 stays
  out.
- **F2's set P,** both ways, each with its mean, top-1 and interval, and the would-be PASS or FAIL marked
  report-only.
- **F2's set C in two halves,** 2,048–2,559 and 2,560–3,070, as S1b reported N2's. The deciding figure is the
  whole set.
- **Perplexity** per arm, R0's too; per-window means; the bootstrap intervals.
- **K4's worst relative difference,** and the guard's counts.

## 7. Predictions (ESTIMATE; scored HIT or MISS; they decide nothing)

- **FA-P1a (F2-A/FULL):** PASS in both bands, not NARROW.
- **FA-P1b (F2-A/KV):** PASS on set C, not NARROW.
- **FA-P2, the square law the F2 plan's factor rested on** (plan section 3, INFERRED): F2's mean KL is at most
  2.0 × R's on the same windows, in both bands and on set C.
  - This 2.0 is a KL ratio. The plan's F2_FACTOR 2.0 is a departure ratio, which under the square law would allow
    about 4× in KL.
  - F2-E's worst per-linear departure ratio, 1.1891, squares to 1.41. The gap to 2.0 leaves room for compounding
    across 238 linears, which the per-linear screen does not bound.
- **FA-P3:** F2's top-1 is ≥ 0.98 in both bands.
- **FA-P4:** F2's set C mean KL is below N2's on the same windows (both report their figures; F2's deciding, N2's
  report-only).
- **FA-P5:** R PASSES under both rules (report-only arm).
- **FA-P6:** N2 FAILS bands L and H again (S1's and S1b's replication read FAIL), and PASSES set C (S1b read PASS
  on the test split); its NARROW is not predicted (report-only arm).
- **FA-P7:** F2 reads within both thresholds on P-R0 (report-only).
- **FA-P8:** the guard reads 0 in every F2 run.

The side of the line is predicted for FA-P1a, FA-P1b, FA-P5 and FA-P6. The figures behind them are the ones cited,
not new readings.

## 8. Pins (checked in the prereg and in every later stage; a mismatch is a STOP)

- **The texts:** this text (its LF, filled when the gate accepts it), the F2 plan (b98787f4) and the F2-E
  addendum (1ad61601).
- **F2-E's run log** (96f83a8, LF da97478e): its PICK line must read "form B, FLUSH_B2, S = ROW", and its CASE_JSON
  feeds K4.
- **The tools, imported read-only:**
  - hybrid_f2e.py (LF dac40cdf), hybrid_u6e.py (LF b88d23d3) and hybrid_f2_0b.py (LF 7a953146);
  - hybrid_s1.py, asserted equal to its blob at c67d6d2;
  - hybrid_s1b.py, asserted equal to its blob at b777070;
  - f2_flush_b.cc (LF caa62ce0), the orders' source.
- **The model:**
  - the GGUF (3,155,051,328 B, sha256 76aed0a8…);
  - U6-E's X16_SHA and W_SHA, for K4;
  - S1's models by the r2 BUILD_JSON: R0.onnx a0a7d785…, R.onnx b960a919…, N2.onnx with int8.onnx.data ceef81a2…,
    and the base data per S1's SRC_PIN.
- **The text:** the validation parquet (section 4); S1's test parquet, for K3 and K5; S1's tokenizer pins.
- **The runtime:** ORT 1.23.3.dev20260320.
- **The F2 graph's own files** (F2.onnx and f2.onnx.data) are hashed as the build writes them, and pinned in its
  BUILD_JSON. Every later stage refuses others.
- **The frozen triple, as S1 and S1b:** PREREG_TEXT_SHA256 (this text inside the tool), PROTOCOL_JSON_SHA256 and
  VERDICT_CODE_SHA256.
  - **The two-verdict rule is in both** (fix F2): PROTOCOL_JSON carries each verdict's set, rule, thresholds,
    bootstrap and the headline order (KV first); VERDICT_CODE hashes both rule functions and the combination
    wording.
  - VERDICT_CODE hashes the sources of every verdict function used, the imported S1 and S1b ones included
    (inspect.getsource), as S1b's does.
  - All three are printed in every F2-A log, and the verdict refuses a mismatch.

## 9. Blind-safety

- The two rules, the thresholds, the windows, the arm, the report-only list, the predictions and every print are
  fixed in the tool, and hashed into the frozen triple, before any real input is read.
- **The selftest is synthetic only:** no model, no text, no x16.
- **The prereg reads only the text,** to pin it. It prints the token count, the offsets and the shas; no model
  runs.
- **The build and the check print pipeline facts only:**
  - shas, op counts, the guard, memory and times;
  - K2–K8 as PASS or FAIL. K4's worst difference concerns F2-E's published cases on S1's sequence 0 only, and K7's
    max KL is R0 against R0.
- **The states write the hidden states and KV to disk, unread.** They print rc, memory, times and the guard only.
  The guard's count is a pipeline fact (ruling 6).
- **No KL, top-1, agreement or perplexity figure on the new text exists before the verdict.** No interim reading is
  taken or reported.
- **No re-runs** (ruling 8):
  - a child that exits nonzero stops its stage, and the stage comes to the gate as it stands;
  - rc 4, the memory gate or the watchdog, reads "memory gate; not a verdict", and the stage stops. A memory-gate
    stop that produced no output may be restarted only on the gate's ruling;
  - no logged stage is repeated otherwise.
- The verdict runs once, on the gate's go.

## 10. The selftest (synthetic only)

1. **The F2 subgraph against hybrid_f2e's emulation (frozen), bit for bit,** through ORT's CPU EP on one-linear
   probes:
   - on synthetic X and Q4_0 weights at K = 2,048, 2,560 and 10,240, with a small N;
   - with plants: a zero block, a zero row, blocks at qx = 1, codes at ±127, an Sx that needs the step, and large
     and small magnitudes.
2. **The one-ulp step** equals hybrid_f2e's step_up (nextafter) on 10^6 random positive normal fp32. It is also
   checked on every binade's first and last value across the normal range.
3. **The in-graph Sx, c and qx** equal hybrid_f2e's fp32_up, scale_up and ceil_exact on planted quotients: integer
   quotients, and quotients a hair above an integer.
4. **The integer sum:** MatMul(double) equals Python-int I at the bounds: |X′| = 32,385, |W′| = 1,016, K = 10,240,
   and |I| near its maximum. MatMul(int64)'s result is checked and kept on record (ruling 4).
5. **The casts:** Cast(double → float) and Cast(float → bfloat16) round to nearest even on planted ties, against
   numpy and ml_dtypes.
6. **The guard:** it reads 0 on clean inputs, and counts each planted violation, one per kind, fed through a probe
   with a doctored scale input.
7. **The windows:** the offsets and the disjointness asserts on a synthetic token list, fed a deliberately
   overlapping case.
8. **The verdict code:**
   - each rule on synthetic metrics: PASS, FAIL, both NARROWs, a non-finite value, and INCOMPLETE;
   - the headline order (KV first) and each combination's wording;
   - set L, H, C and P slicing on synthetic logits, and the KV slice to 0–2,046;
   - the bootstrap;
   - the imported kernels, hashed by source.
9. **K2's and K3's comparators** on a small synthetic graph with a planted extra node, and a planted fused op.
10. **The watchdog and the start gate** (S1's), and determinism.

## 11. Cost, and the START REQUESTs to BFP16

The F2 child's cost is an ESTIMATE until the build measures it. The build's probe measures the F2 arm's session
memory and one 2,048-id prompt on S1's test sequence 0, as S1's PROBE_JSON did. START REQUEST 2 is sized from that
probe, and the probe goes to the gate first.
- **The double GEMM path** (SPEC, ORT's source at rel-1.23.2, read 2026-09-25): the MatMul kernel's generic
  Compute calls math::MatMul<T> (core/providers/cpu/math/matmul.cc:124). math::MatMul<double> calls MlasGemm when
  MLAS_SUPPORTS_GEMM_DOUBLE is defined, and Eigen otherwise (core/util/math_cpu.cc:188-195). mlas.h defines it for
  MLAS_TARGET_AMD64 (core/mlas/inc/mlas.h:83-85). So on this machine MatMul(double) is MLAS's double GEMM. Its speed
  here is unmeasured.
- **The probe's stop line** (the gate's note, 2026-09-25): if the probe reads one 2,048-id F2 prompt slower than
  about 7 min (under about 31 GFLOPS: 13.1 TFLOP / 420 s; DERIVED), it STOPs and comes to the gate before START
  REQUEST 2. N, or the int64 route, is revisited then, not now.
- **The build** writes f2.onnx.data, 6,417,285,120 B (DERIVED, section 2.1), to scratch/llm/hybrid_f2a/models.
  - It runs hybrid_f2e's weight grid over the 238 linears.
  - It profiles one pass per new model.
  - ESTIMATE: 20–40 min, and the RAM of the probes below.
- **The F2 prompt child:**
  - **RAM (ESTIMATE):** about 9–12 GB peak working set and 3–6 GB private. N3, with a data file of the same size
    cast at run time, read 9.01 GB working set and 2.85 GB private (MEASURED, S1b). F2 adds float64 transients: the
    largest W′ is 26,214,400 × 8 B = 210 MB, and X′ at 2,048 × 10,240 × 8 B is 168 MB (DERIVED).
  - **Time (ESTIMATE):** about 45–150 s per 2,048-id prompt.
    - The float64 MatMuls are 6.57e12 multiply-adds, 13.1 TFLOP per prompt (DERIVED: 3,208,642,560 × 2,048 × 2).
    - The rate is an assumed 90–300 GFLOPS (ESTIMATE), plus the casts.
    - N3's fp32 prompt took 24.35 s (MEASURED, S1b's median).
- **The reused children, per window** (MEASURED medians, S1b's states log):
  - R0 one-shot 32.0 s; each continuation about 10.2–10.5 s;
  - R's prompt 17.6 s, and N2's 13.1 s;
  - the P-R0 steps, under 1 s each (ESTIMATE).
- **Per window,** with F2, R and N2: about 140–250 s (ESTIMATE). **16 windows: about 40–70 min.**
- **The check:**
  - K4's 21 probes are light;
  - K5 and K6 are three prompt runs;
  - K7 is 8 windows at about 85–93 s each (MEASURED: S1b's control, 85.2–93.3 s per window, its check log).
  - About 25–40 min (ESTIMATE).
- **Disk:**
  - the F2 data, 6.42 GB;
  - the KV at 570 MB per arm and window (MEASURED, S1): 13.7 GB per batch of 8 windows × 3 arms, deleted after each
    batch;
  - the hidden states, about 2.0 GB (DERIVED: 16 × 12,288 positions × 10,240 B).
- **The verdict:** about 147,000 scored positions, against S1's 184,000 (DERIVED):
  - L and H, 3 × 16 × 2,047;
  - set C, 3 × 16 × 1,023;
  - set P, 96.

  About 15–25 min at about 5 GB (ESTIMATE, from S1's plan).
- **Memory rules, S1's:** one model per child; a 15 GB start gate; a 5 GB watchdog; PRIORITY_BASED sessions,
  ORT_DISABLE_ALL, 8 intra-op threads.
- **BFP16:**
  - an FYI for the selftest and the prereg (light);
  - START REQUEST 1 for the build and the check;
  - START REQUEST 2 for the states, sized from the build's probe;
  - START REQUEST 3 for the verdict;
  - FINISHED after each.

## 12. What F2-A cannot establish

- **Silicon.** No F2 kernel runs on the NPU. F2's NPU tie is DERIVED: F2-0b's compiled flush, emulated in its order.
  The lane order and the RNE add are not tested on a core.
- **Energy.** Nothing is measured. F2's modelled 2.7292 mJ against N-i8's measured 1.6334 stands.
- **The N2 pick.** F2-A cannot change it.
- **F2's memory case.** F2-A measures accuracy, not bytes.
- **The 780M's own numerics.** R0 stands in for its last-position step and its continuation. Its attention
  precision is not modelled.
- **The host quantizer's cost and fp32 form.** The codes are computed in float64 here.
- **Free-running generation.** The continuation is teacher-forced.
- **Other text, tokenizers or prompt lengths.** That includes 8,192-token prompts, and continuations beyond 1,023
  positions.

## 13. Files and order

- **New:** tools/hybrid_f2a.py (AGPL-3.0-or-later, written fresh), with modes selftest, prereg, build, check,
  states and verdict, and their child modes. It imports hybrid_s1, hybrid_s1b, hybrid_f2e and hybrid_u6e
  read-only.
- **The runner:** stages f2a-selftest, f2a-prereg, f2a-build, f2a-check, f2a-states and f2a-verdict in
  scripts/hybrid-stack.sh, edited only between runs. S1's and S1b's stages stay byte-identical.
- **Scratch** (git-ignored): scratch/llm/hybrid_f2a/{models,check,states,kv}.
- **Logs:** results/llm/hybrid_f2a_{selftest,prereg,build,check,states,verdict}_desktop2_YYYYMMDD.log. UTF-8, the
  profile path scrubbed, never overwritten, each committed alone.
- **The order:**
  1. this text to the gate; its rulings; the revision; the gate's acceptance (LF);
  2. the diff (the tool and the stages), with the selftest's output, for the gate's go;
  3. the frozen tool, committed with its selftest log. Then the prereg run (the text pins), committed before
     anything heavy;
  4. START REQUEST 1, then the build and the check, then FINISHED; the logs committed; a pipeline-facts report,
     with the probe's measured cost;
  5. the gate's go, then START REQUEST 2, the states, and FINISHED; the log committed; a pipeline-facts report;
  6. the gate's go, then START REQUEST 3, the verdict (once), and FINISHED; the log committed; the report.

## 14. The gate's rulings on v1's open points (2026-09-25), as applied

1. **The deciding set:** two pre-registered verdicts, not one and not a conjunction. F2-A/KV on set C with S1b's
   rule, and F2-A/FULL on bands L and H with S1's rule. KV prints first. P-R0 stays report-only. The user may
   override this before the tool is frozen. Applied in the header and sections 0, 4, 5, 7 and 8.
2. **"What stays R's":** R0, the reference path. No R-continuation variant, and no R control in K7. Sections 1
   and 3.
3. **N = 16:** accepted. A NARROW stands as the verdict; a larger sample is a new pre-registered experiment on the
   user's go, never a re-run. Section 4.
4. **MatMul(double):** accepted, on the exactness argument. The selftest keeps MatMul(int64)'s record. Sections
   2.1 and 10.
5. **The one-ulp step:** accepted as DERIVED. It must pass the selftest bit for bit against hybrid_f2e before the
   tool is frozen. Section 2.2.
6. **The in-graph guard,** with F3's STOP semantics: accepted. Its count is a pipeline fact, printed in the states.
   Sections 2.3 and 9.
7. **R and N2 as report-only arms:** accepted. N2's set C is printed beside F2-A/KV. N3 stays out. Section 6.
8. **No re-runs:** accepted. A memory-gate stop (rc 4) that produced no output may be restarted only on the gate's
   ruling. Section 9.
9. **No C4:** K5 is the tie. Section 3.
10. **The download in the prereg stage:** accepted. If the Hub asks for a token, STOP and come to the gate; never
    print a token. Section 4.

## 15. Changes after the gate's read of LF a0cbad8c (ACCEPTED WITH RULINGS AND FIXES)

- **Header, fix F1b:** the amendment of the F2 plan's section 4 (lines 266–272), with the plan file unedited.
- **Section 0, fix F1:** two questions and two verdicts, KV first.
- **Section 1:** ruling 2's reading; no R continuation; the note that the regrouped sum is F2-E's I.
- **Section 2.1:** the partial-sum bound written as 10,240 × 32,903,160; the int64 record.
- **Section 2.2:** the equivalence of cast-then-step with F2-E's round-up.
- **Section 2.3:** the guard's count is printed in the states.
- **Section 3:** the reference positions for set P and set C; K7 guards F2-A/KV's path; no R control.
- **Section 4:** set C decides F2-A/KV and L and H decide F2-A/FULL; set P's steps of 1/16; ruling 3's NARROW and
  larger-sample wording; ruling 10's token STOP.
- **Section 5, fix F1:** the two rules, verbatim from S1b and S1, with their NARROW, INCOMPLETE, the headline order
  and each combination's meaning. "The verdict does not change the N2 pick" is kept.
- **Section 6:** N2's set C beside F2-A/KV; F2's set C moved from report-only to deciding.
- **Section 7, fix F1:** FA-P1 split into FA-P1a (FULL) and FA-P1b (KV); FA-P4 (F2 against N2 on set C) added; the
  report-only predictions renumbered.
- **Section 8, fix F2:** the two-verdict rule in PROTOCOL_JSON and VERDICT_CODE.
- **Section 9:** ruling 8's restart wording; ruling 6's guard print.
- **Section 10:** each rule, the headline order and the combinations in the verdict code's selftest; the int64
  record.
- **Section 11:** the double GEMM path read from ORT's source (MLAS on AMD64, not Eigen), and the gate's 7-minute
  stop line for the probe.
- **Section 13:** unchanged. Section 14 replaces v1's open points with the rulings.
"""


# ---------------------------------------------------------------- guards

def blob(path: str, commit: str) -> dict:
    def git(*a):
        r = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, encoding="utf-8")
        return r.stdout.strip() if r.returncode == 0 else None
    at, now = git("rev-parse", f"{commit}:{path}"), git("hash-object", path)
    return {"file": path, "commit": commit, "blob_at_commit": at, "blob_now": now,
            "equal": at is not None and at == now}


def pin_rows() -> list:
    rows = []
    for what, p, sha in PINS:
        cur = lf_sha(p) if p.exists() else None
        rows.append({"what": what, "file": p.relative_to(ROOT).as_posix(), "pinned": sha, "lf_sha256": cur,
                     "equal": cur == sha})
    return rows


def f2e_pick_ok() -> dict:
    lines = F2E_LOG.read_text(encoding="utf-8").splitlines() if F2E_LOG.exists() else []
    pick = [ln for ln in lines if ln.startswith("F2-E PICK:")]
    out = [json.loads(ln.split(" ", 1)[1]) for ln in lines if ln.startswith("OUTPUT_JSON ")]
    ok = len(pick) == 1 and pick[0].startswith(F2E_PICK) and len(out) == 1 and out[0].get("pick") == ARM_KEY
    return {"pick_line": pick[0] if pick else None, "output_pick": out[0].get("pick") if out else None, "ok": ok}


def guards_ok() -> bool:
    """The imported code and the texts are the pinned ones: LF pins, the S1 and S1b blobs, F2-E's pick."""
    rows = pin_rows()
    blobs = [blob(p, c) for p, c in BLOBS.items()]
    pk = f2e_pick_ok()
    s1.say("PIN_JSON", rows)
    s1.say("BLOB_JSON", blobs)
    s1.say("F2E_PICK_JSON", pk)
    return all(r["equal"] for r in rows) and all(b["equal"] for b in blobs) and pk["ok"]


def frozen_ok() -> bool:
    p = s1.latest("hybrid_f2a_prereg_*.log")
    text = p.read_text(encoding="utf-8")
    want = dict(re.findall(r"^(PREREG_TEXT_SHA256|PROTOCOL_JSON_SHA256|VERDICT_CODE_SHA256) ([0-9a-f]{64})$",
                           text, re.M))
    got = hashes()
    ok = all(want.get(k) == v for k, v in got.items()) and got["PREREG_TEXT_SHA256"] == PREREG_SHA
    s1.say("FROZEN_JSON", {"prereg_log": p.name, "prereg_log_lf_sha256": s1.sha_lf(p), "equal": ok, **got})
    return guards_ok() and ok


def models_ok(build: dict) -> bool:
    """K1's model pins: S1's reused files against MODEL_PIN, the linked base data, the F2 files against BUILD_JSON."""
    res, ok = {}, True
    for f, (size, sha) in MODEL_PIN.items():
        p = S1_MODELS / f
        got = s1.sha_file(p) if p.exists() else None
        good = p.exists() and p.stat().st_size == size and got == sha
        res[f] = {"bytes": p.stat().st_size if p.exists() else None, "sha256": got, "ok": good}
        ok &= good
    link = MODELS / "base.onnx.data"
    same = link.exists() and os.path.samefile(link, S1_MODELS / "base.onnx.data")
    res["f2a/base.onnx.data"] = {"same_file_as_s1": same, "ok": same}
    ok &= same
    for f in ("F2.onnx", "f2.onnx.data"):
        p = MODELS / f
        want = build["files"].get(f, {})
        got = s1.sha_file(p) if p.exists() else None
        good = p.exists() and p.stat().st_size == want.get("bytes") and got == want.get("sha256")
        res[f"f2a/{f}"] = {"bytes": p.stat().st_size if p.exists() else None, "sha256": got, "ok": good}
        ok &= good
    s1.say("MODEL_PIN_JSON", res)
    return ok


# ---------------------------------------------------------------- children

_STDOUT_JSON = []


def child(args: list) -> int:
    """Run a child of this tool once (no re-run), its output into this log, and keep its tagged JSON."""
    sys.stdout.flush()
    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve())] + args, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True)
        _STDOUT_JSON.append(line.rstrip("\n"))
    rc = p.wait()
    print(f"CHILD {' '.join(args)} rc {rc}", flush=True)
    return rc


def stdout_json(tag: str) -> list:
    return [json.loads(s.split(" ", 1)[1]) for s in _STDOUT_JSON if s.startswith(tag + " ")]


def last_json(tag: str) -> dict:
    got = stdout_json(tag)
    if not got:
        raise Stop(f"no {tag} from the children")
    return got[-1]


def stop_rc(stage: str, rc: int, what: str) -> int:
    if rc == 4:
        print(f"{stage} STOP: memory gate; not a verdict ({what}). A stop that produced no output may be restarted "
              "only on the gate's ruling", flush=True)
        return 4
    print(f"{stage} STOP: {what} exited rc {rc}; the stage comes to the gate as it stands", flush=True)
    return 2


def batches() -> list:
    return [list(range(b, min(b + BATCH, N_WIN))) for b in range(0, N_WIN, BATCH)]


def line_timer(seconds: float, tag: str):
    """The probe's runaway ceiling: if the prompt is still running at `seconds`, print the STOP and end (rc 5). The
    420 s line is not this timer: the prompt runs to its end, and the build STOPs on the measured seq0_s."""
    def fire():
        print(f"PROBE STOP: {tag}: the prompt passed the {seconds:,g} s ceiling unfinished (the {PROBE_LINE_S:.0f} s "
              "line is the gate's STOP)", flush=True)
        s1.say("PROBE_CEILING_JSON", {"child": tag, "ceiling_s": seconds, "memory": s1.own_memory()})
        sys.stdout.flush()
        os._exit(5)
    t = threading.Timer(seconds, fire)
    t.daemon = True
    t.start()
    return t


def probe_child() -> int:
    s1.start_gate("probe F2")
    seqs, _ = s1.pinned_sequences()
    PROF.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sess = s1.session(MODELS / "F2.onnx", profile=PROF / "F2")
    t1 = time.perf_counter()
    mem_load = s1.own_memory()
    timer = line_timer(PROBE_CEILING_S, "probe F2")
    h, _, guard = run_f2(sess, seqs[0])
    t2 = time.perf_counter()
    timer.cancel()
    prof = sess.end_profiling()
    c = census_of(prof)
    s1.say("PROBE_JSON", {"arm": "F2", "session_s": round(t1 - t0, 1), "seq0_s": round(t2 - t1, 1),
                          "executed": c["executed"], "matmul_double": c["matmul_double"],
                          "finite": bool(np.isfinite(h).all()), "guard": guard, "memory_after_load": mem_load,
                          "memory_after_run": s1.own_memory()})
    return 0


def k2_child() -> int:
    import onnx
    from onnx import numpy_helper
    s1.start_gate("K2 (graph identity, W' and Dw)")
    f2 = onnx.load(str(MODELS / "F2.onnx"), load_external_data=False)
    r0 = onnx.load(str(S1_MODELS / "R0.onnx"), load_external_data=False)
    graph = k2_compare(r0, f2)
    mm, by = s1.open_gguf(hash_check=True)
    init = {t.name: t for t in f2.graph.initializer}
    raw = np.memmap(MODELS / "f2.onnx.data", dtype=np.uint8, mode="r")
    bad, n = [], 0
    for nb in [x for x in r0.graph.node if x.op_type == "MatMulNBits"]:
        L, gg = s1.gguf_of(nb.name)
        codes, d = gd.gguf_linear(mm, by[f"blk.{L}.{gg}.weight"])
        Wp, Dw, _ = wprime(codes, d)
        tw, td = init.get(nb.name + "/f2/wp"), init.get(nb.name + "/f2/dw")
        ok = tw is not None and td is not None
        if ok:
            e = s1.ext(tw)
            b = raw[int(e["offset"]):int(e["offset"]) + int(e["length"])]
            ok = e.get("location") == "f2.onnx.data" and list(tw.dims) == list(Wp.shape) and \
                s1.sha_bytes(b.tobytes()) == s1.arr_sha(Wp) and \
                s1.arr_sha(numpy_helper.to_array(td)) == s1.arr_sha(Dw)
        if not ok:
            bad.append(nb.name)
        n += 1
    del raw
    s1.say("K2_JSON", {"graph": graph, "linears": n, "weights_mismatched": bad,
                       "outcome": "PASS" if graph["ok"] and n == gd.LAYERS * 7 and not bad else "FAIL"})
    return 0


def f2e_cases() -> dict:
    """F2-E's run log: (layer, name) -> CASE_JSON's arms["B/B2/ROW"]["vs_l0"] (K4_PATH)."""
    out = {}
    for ln in F2E_LOG.read_text(encoding="utf-8").splitlines():
        if ln.startswith("CASE_JSON "):
            c = json.loads(ln.split(" ", 1)[1])
            out[(int(c["layer"]), c["name"])] = c["arms"][ARM_KEY]["vs_l0"]
    return out


def k4_child() -> int:
    s1.start_gate("K4 (21 one-linear probes)")
    mm, by = s1.open_gguf(hash_check=True)
    xs, ws = u6e.current_pins(mm, by)
    if xs != u6e.X16_SHA or ws != u6e.W_SHA:
        print("K4 STOP: an x16 input or a weight differs from U6-E's pins", flush=True)
        return 2
    logged = f2e_cases()
    X16 = {x: np.load(s1.CHECK / f"x16_{x}.npy") for x in p3c.INPUTS}
    rows = np.asarray(f2e.ROWS)
    vals, guards = [], []
    for L in f2e.LAYERS:
        for name, gg, K, N, x in p3c.LINEARS:
            codes, d = gd.gguf_linear(mm, by[f"blk.{L}.{gg}.weight"])
            Wp, Dw, g = wprime(codes, d)
            st = f2e.w_stops(g)
            if st:
                print(f"K4 STOP: layer {L} {name}: {st}", flush=True)
                return 2
            m, _ = linear_model(K, N, Wp, Dw)
            y, guard = sess_of(m).run(["y", GUARD_NAME], {"x": X16[x][None]})
            yr = np.ascontiguousarray(y[0][rows])
            Xr = np.ascontiguousarray(X16[x][rows])
            q, sx = s1.q8_ort(Xr)
            ye = f2e.emulate(s1.blocks(Xr), q, sx, codes, gc.f16_to_f32(d).reshape(N, K // 32), "B", "ROW", FLUSH)[0]
            y0 = u6e.nb_run(X16[x], codes, d, K, N, 0)[rows]
            r = s1.rel_l2(yr, y0)
            lg = logged.get((L, name))
            rd = abs(r - lg) / abs(lg) if lg else float("inf")
            vals.append({"layer": L, "name": name, "bits_equal": bool(np.array_equal(yr.view(np.uint32),
                                                                                      ye.view(np.uint32))),
                         "vs_l0": r, "logged": lg, "rel_diff": rd, "guard": int(guard)})
            guards.append(int(guard))
    CHECK.mkdir(parents=True, exist_ok=True)
    blob_ = json.dumps(vals, sort_keys=True).encode("utf-8")
    (CHECK / "k4_values.json").write_bytes(blob_)
    s1.say("K4_JSON", {"cases": len(vals), "bits_equal": sum(v["bits_equal"] for v in vals),
                       "reproduced": sum(v["rel_diff"] <= K4_TOL for v in vals),
                       "worst_rel_diff": max(v["rel_diff"] for v in vals), "guards": guards,
                       "values_file": "k4_values.json", "values_sha256": s1.sha_bytes(blob_), "path": K4_PATH})
    s1.say("MEM_JSON", {"child": "K4", **s1.own_memory()})
    return 0


def k5_child() -> int:
    s1.start_gate("R0 (K5)")
    seqs, _ = s1.pinned_sequences()
    sess = s1.session(S1_MODELS / "R0.onnx")
    h, _ = s1.run(sess, seqs[0])
    s1.say("K5_JSON", {"sha": s1.arr_sha(h), "want": C5_SHA, "outcome": "PASS" if s1.arr_sha(h) == C5_SHA else "FAIL"})
    s1.say("MEM_JSON", {"child": "R0 (K5)", **s1.own_memory()})
    return 0


def k6_child() -> int:
    s1.start_gate("F2 (K6)")
    wins = pinned_windows()
    sess = s1.session(MODELS / "F2.onnx")
    t0 = time.perf_counter()
    h1, _, g1 = run_f2(sess, wins[0][:PROMPT])
    t1 = time.perf_counter()
    h2, _, g2 = run_f2(sess, wins[0][:PROMPT])
    t2 = time.perf_counter()
    s1.say("K6_JSON", {"sha_run1": s1.arr_sha(h1), "sha_run2": s1.arr_sha(h2), "guards": [g1, g2],
                       "seconds": [round(t1 - t0, 1), round(t2 - t1, 1)],
                       "outcome": "PASS" if s1.arr_sha(h1) == s1.arr_sha(h2) else "FAIL"})
    s1.say("MEM_JSON", {"child": "F2 (K6)", **s1.own_memory()})
    return 0


def control_child() -> int:
    s1.start_gate("R0 control (a, b, c)")
    wins = pinned_windows()
    sess = s1.session(S1_MODELS / "R0.onnx")
    for w in CONTROL_WINDOWS:
        ids = wins[w]
        t0 = time.perf_counter()
        ho, _ = s1.run(sess, ids)
        hp, kv = s1.run(sess, ids[:PROMPT], want_kv=True)
        hc, _ = s1.run(sess, ids[PROMPT:], past=kv)
        hd, _ = s1.run(sess, ids[P_POS:PROMPT], past=s1b.slice_kv(kv, P_POS))
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


def finite_of(arm: str, a) -> dict:
    """R's, N2's and R0's finiteness, printed in the states. The deciding arm's is an outcome (RULE_KV and RULE_FULL
    read a non-finite value as FAIL), so the states print none for it (the prereg, section 9): the verdict computes
    it from the arrays."""
    return {} if arm in DECIDING else {"finite": bool(np.isfinite(a).all())}


def prompt_child(arm: str, b: str) -> int:
    s1.start_gate(f"prompt {arm} batch {b}")
    wins = pinned_windows()
    ws = batches()[int(b)]
    d, kd = STATES / arm, KVDIR / arm
    d.mkdir(parents=True, exist_ok=True)
    kd.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sess = s1.session(MODELS / "F2.onnx" if arm == "F2" else S1_MODELS / f"{arm}.onnx")
    s1.say("SESSION_JSON", {"arm": arm, "batch": int(b), "seconds": round(time.perf_counter() - t0, 1),
                            **s1.own_memory()})
    for w in ws:
        t1 = time.perf_counter()
        if arm == "F2":
            h, kv, guard = run_f2(sess, wins[w][:PROMPT], want_kv=True)
            if guard != 0:
                s1.say("GUARD_JSON", {"arm": arm, "w": w, "guard": guard})
                print(f"F2 GUARD STOP: window {w} read {guard}; nothing of this window is written", flush=True)
                return 2
        else:
            h, kv = s1.run(sess, wins[w][:PROMPT], want_kv=True)
            guard = None
        np.save(d / f"w{w:02d}_prompt.npy", h)
        np.savez(kd / f"w{w:02d}.npz", **kv)
        del kv
        s1.say("PROMPT_JSON", {"arm": arm, "w": w, "rows": len(h), "sha256": s1.arr_sha(h), "guard": guard,
                               **finite_of(arm, h), "seconds": round(time.perf_counter() - t1, 1)})
    s1.say("MEM_JSON", {"child": f"prompt {arm} batch {b}", **s1.own_memory()})
    return 0


def r0_child(b: str) -> int:
    s1.start_gate(f"R0 batch {b}")
    wins = pinned_windows()
    ws = batches()[int(b)]
    d = STATES / "R0"
    d.mkdir(parents=True, exist_ok=True)
    sess = s1.session(S1_MODELS / "R0.onnx")
    for w in ws:
        t1 = time.perf_counter()
        h, _ = s1.run(sess, wins[w])
        np.save(d / f"w{w:02d}_oneshot.npy", h)
        s1.say("ONESHOT_JSON", {"w": w, "rows": len(h), "sha256": s1.arr_sha(h), "finite": bool(np.isfinite(h).all()),
                                "seconds": round(time.perf_counter() - t1, 1)})
    for arm in ARMS:
        for w in ws:
            f = KVDIR / arm / f"w{w:02d}.npz"
            if not f.exists():
                s1.say("CONT_SKIPPED_JSON", {"arm": arm, "w": w, "why": "no KV from the arm's prompt run"})
                continue
            t1 = time.perf_counter()
            kv = dict(np.load(f))
            hc, _ = s1.run(sess, wins[w][PROMPT:], past=kv)
            hd, _ = s1.run(sess, wins[w][P_POS:PROMPT], past=s1b.slice_kv(kv, P_POS))
            del kv
            np.save(STATES / arm / f"w{w:02d}_cont.npy", hc)
            np.save(STATES / arm / f"w{w:02d}_pr0.npy", hd)
            s1.say("CONT_JSON", {"arm": arm, "w": w, "rows": len(hc), "sha256": s1.arr_sha(hc),
                                 **finite_of(arm, hc), "seconds": round(time.perf_counter() - t1, 1)})
            s1.say("PR0_JSON", {"arm": arm, "w": w, "rows": len(hd), "sha256": s1.arr_sha(hd), **finite_of(arm, hd)})
    s1.say("MEM_JSON", {"child": f"R0 batch {b}", **s1.own_memory()})
    return 0


def gate_child(min_gb: str) -> int:
    """The selftest's start-gate check: S1's start_gate with the floor set to min_gb."""
    s1.MIN_AVAIL_GB = float(min_gb)
    s1.start_gate("selftest")
    print("GATE_PASSED", flush=True)
    return 0


def line_child(seconds: str) -> int:
    """The selftest's stop-line check: the main thread sleeps 2 s under a line of `seconds`."""
    t = line_timer(float(seconds), "selftest")
    time.sleep(2)
    t.cancel()
    print("LINE_MAIN_DONE", flush=True)
    return 0


# ---------------------------------------------------------------- stages

def stage_header(title: str) -> None:
    print(f"{title} (hybrid F2-A), no chip. {stamp()}", flush=True)
    print_hashes()


def prereg() -> int:
    print(PREREG, flush=True)
    s1.say("PROTOCOL_JSON", protocol())
    s1.say("PREDICTIONS_JSON", [{"id": q, "text": t} for q, t in PREDICTIONS])
    try:
        return prereg_body()
    except Stop as e:
        print(f"PREREG STOP: {e}", flush=True)
        return 2


def prereg_body() -> int:
    stop = []
    if not guards_ok():
        stop.append("a pin, a blob or F2-E's pick")
    tp = tokenizer_pins()
    s1.say("TOKENIZER_JSON", tp)
    if not tp["ok"]:
        stop.append("the tokenizer pins")
    T = val_tokens()
    pins = text_pins(T, prior_prompt_shas())
    s1.say("TEXT_JSON", pins)
    a = pins["asserts"]
    print(f"TEXT: {pins['text_tokens']} tokens in the joined validation split; {N_WIN} windows of {WIN_LEN} ids from "
          f"{pins['first']} to {pins['end']}, {pins['left']} tokens left; asserts: pairwise disjoint "
          f"{a['pairwise_disjoint']}, inside the text {a['inside_text']}, every window whole {a['windows_full']}, no "
          f"prompt sha among S1's {pins['prior']['s1_seq_sha']} and S1b's {pins['prior']['s1b_prompt_sha']} "
          f"{a['no_s1_prompt_sha']}", flush=True)
    if not a["ok"]:
        stop.append("the window asserts")
    b = s1.logged_json(RESULTS / S1_BUILD_LOG, "BUILD_JSON")
    cmp_ = {f: {"pin": list(v), "logged": ([b["files"][f]["bytes"], b["files"][f]["sha256"]] if f in b["files"]
                                           else list(s1.SRC_PIN["model.onnx.data"]))} for f, v in MODEL_PIN.items()}
    equal = all(c["pin"] == c["logged"] for c in cmp_.values())
    s1.say("MODEL_PIN_JSON", {"source": f"{S1_BUILD_LOG} BUILD_JSON (base.onnx.data: hybrid_s1 SRC_PIN)",
                              "source_lf_sha256": s1.sha_lf(RESULTS / S1_BUILD_LOG), "equal": equal, "files": cmp_})
    if not equal:
        stop.append("the model pins")
    s1.say("VERDICT_SOURCES_JSON", [{"name": n, "sha256": s1.sha_bytes(s.encode("utf-8"))}
                                    for n, s in verdict_sources()])
    print_hashes()
    same = hashes()["PREREG_TEXT_SHA256"] == PREREG_SHA
    print(f"PREREG_TEXT_EQUALS_ACCEPTED {same} (the accepted text's LF sha256 {PREREG_SHA})", flush=True)
    if not same:
        stop.append("the prereg text")
    if stop:
        print(f"PREREG STOP: {', '.join(stop)}", flush=True)
        return 1
    print("PREREG OK", flush=True)
    return 0


def build() -> int:
    stage_header("BUILD")
    try:
        return build_body()
    except Stop as e:
        print(f"BUILD STOP: {e}", flush=True)
        return 2


def build_body() -> int:
    import onnx
    if not frozen_ok():
        raise Stop("the frozen hashes, a pin, a blob or F2-E's pick differ")
    if u6e.ort_version() != ORT_VERSION:
        raise Stop(f"onnxruntime {u6e.ort_version()} is not {ORT_VERSION}")
    src = s1.verify_sources()
    s1.say("SOURCES_JSON", src)
    if not all(v["ok"] for v in src.values()):
        raise Stop("a copied source differs from its pin (S1's verify_sources)")
    mm, by = s1.open_gguf(hash_check=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    for f in ("F2.onnx", "f2.onnx.data"):
        if (MODELS / f).exists():
            raise Stop(f"{f} exists; the build never replaces a model file")
    link = MODELS / "base.onnx.data"
    if not link.exists():
        os.link(S1_MODELS / "base.onnx.data", link)
    same = os.path.samefile(link, S1_MODELS / "base.onnx.data")
    s1.say("LINK_JSON", {"file": "base.onnx.data", "hard_link_to_s1": same})
    if not same:
        raise Stop("scratch/llm/hybrid_f2a/models/base.onnx.data is not S1's file")
    grid = {"linears": 0, "dw_steps": 0, "scale_faults": 0, "qw_clips": 0}
    stops = []

    def wp_of(name):
        L, gg = s1.gguf_of(name)
        codes, d = gd.gguf_linear(mm, by[f"blk.{L}.{gg}.weight"])
        Wp, Dw, g = wprime(codes, d)
        st = f2e.w_stops(g)
        if st:
            stops.append({"node": name, "stops": st})
        grid["linears"] += 1
        grid["dw_steps"] += g["steps"]
        grid["scale_faults"] += g["scale_faults"]
        grid["qw_clips"] += g["clip"]
        return Wp, Dw
    t0 = time.perf_counter()
    data = s1.DataFile(MODELS / "f2.onnx.data")
    f2 = f2_replace(s1.base_cut(S1_MODELS / "c0h16.onnx", "base.onnx.data"), wp_of, data)
    files = {"f2.onnx.data": data.close()}
    s1.say("WEIGHT_GRID_JSON", {**grid, "stops": stops, "seconds": round(time.perf_counter() - t0, 1)})
    if stops or grid["linears"] != gd.LAYERS * 7:
        raise Stop("the weight grid has a scale fault or a qw clip (hybrid_f2e's w_stops), or a linear is missing")
    if files["f2.onnx.data"]["bytes"] != W_BYTES:
        raise Stop(f"f2.onnx.data is {files['f2.onnx.data']['bytes']} B, not {W_BYTES}")
    onnx.save(f2, str(MODELS / "F2.onnx"))
    p = MODELS / "F2.onnx"
    files["F2.onnx"] = {"file": p.name, "bytes": p.stat().st_size, "sha256": s1.sha_file(p)}
    r0 = onnx.load(str(S1_MODELS / "R0.onnx"), load_external_data=False)
    graph = k2_compare(r0, f2)
    static, r0_static, rest = s1.static_counts(f2), s1.static_counts(r0), static_rest(f2)
    s1.say("STATIC_COUNTS_JSON", {"F2": static, "R0": r0_static, "F2_outside_f2": rest})
    s1.say("GRAPH_JSON", graph)
    del mm
    rc = child(["_probe"])
    if rc != 0:
        return stop_rc("BUILD", rc, "the probe")
    pr = last_json("PROBE_JSON")
    cen = census_ok({"executed": pr["executed"], "matmul_double": pr["matmul_double"]}, static, r0_static, rest,
                    gd.LAYERS * 7)
    s1.say("CENSUS_JSON", cen)
    under = pr["seq0_s"] <= PROBE_LINE_S
    ok = graph["ok"] and cen["ok"] and pr["finite"] and pr["guard"] == 0 and under
    s1.say("BUILD_JSON", {"files": files, "grid": grid, "graph_ok": graph["ok"], "census": cen, "probe_rc": rc,
                          "probe": {k: pr[k] for k in ("session_s", "seq0_s", "finite", "guard", "memory_after_load",
                                                       "memory_after_run")},
                          "probe_under_line": under, "probe_line_s": PROBE_LINE_S, "ok": ok})
    if not under:
        print(f"BUILD STOP: the probe's prompt took {pr['seq0_s']} s, over the {PROBE_LINE_S:.0f} s line; it comes "
              "to the gate before START REQUEST 2", flush=True)
        return 2
    print("BUILD", "OK" if ok else "STOP (the graph, the census, the probe's finiteness or the guard)", flush=True)
    return 0 if ok else 2


def check() -> int:
    stage_header("CHECK")
    try:
        return check_body()
    except Stop as e:
        print(f"CHECK STOP: {e}", flush=True)
        return 2


def check_body() -> int:
    if not frozen_ok():
        raise Stop("the frozen hashes, a pin, a blob or F2-E's pick differ")
    b = s1.logged_json(s1.latest("hybrid_f2a_build_*.log"), "BUILD_JSON")
    if not b["ok"]:
        raise Stop("the build is not OK")
    if u6e.ort_version() != ORT_VERSION:
        raise Stop(f"onnxruntime {u6e.ort_version()} is not {ORT_VERSION}")
    k1 = models_ok(b)
    print(f"K1 {'PASS' if k1 else 'FAIL'}: the pins, the blobs, F2-E's pick, the frozen triple, S1's models, the "
          "linked base data and the F2 files", flush=True)
    if not k1:
        raise Stop("K1 FAIL")
    if CHECK.exists():
        shutil.rmtree(CHECK)
    CHECK.mkdir(parents=True)
    rcs = {}
    for c in (["_k2"], ["_k4"], ["_k5"], ["_k6"], ["_control"], ["_ccompare"]):
        rcs[c[0]] = child(c)
        if rcs[c[0]] != 0:
            return stop_rc("CHECK", rcs[c[0]], c[0])
    k2, k4, k5, k6 = (last_json(t) for t in ("K2_JSON", "K4_JSON", "K5_JSON", "K6_JSON"))
    ctl = control_outcome(stdout_json("CONTROL_JSON"))
    k4_ok = k4["cases"] == 21 and k4["bits_equal"] == 21 and k4["reproduced"] == 21
    guards = [b["probe"]["guard"]] + k4["guards"] + k6["guards"]
    res = {"K1": "PASS", "K2": k2["outcome"], "K3": "PASS" if b["census"]["ok"] else "FAIL",
           "K4": "PASS" if k4_ok else "FAIL", "K5": k5["outcome"], "K6": k6["outcome"], "K7": ctl["outcome"],
           "K8": "PASS" if all(g == 0 for g in guards) else "FAIL"}
    print(f"K2 {res['K2']}: F2 equals R0 outside the 238 linears; W' and Dw equal their re-derivation from the GGUF "
          f"by sha ({k2['linears']} linears, {len(k2['weights_mismatched'])} mismatched)", flush=True)
    print(f"K3 {res['K3']}: the build's profiled pass on S1's test sequence 0 (238 MatMul, all double; no "
          "MatMulNBits; no fused op; every op outside the F2 subgraphs at R0's count)", flush=True)
    print(f"K4 {res['K4']}: 21 one-linear probes; bit for bit equal to hybrid_f2e's B/B2/ROW on F2-E's 256 rows: "
          f"{k4['bits_equal']} of {k4['cases']}; rel-L2 against L0 reproduces the F2-E run log's CASE_JSON "
          f"{K4_PATH}, keyed by (layer, name), within {K4_TOL:g} relative: {k4['reproduced']} of {k4['cases']} "
          f"(worst relative difference {k4['worst_rel_diff']:.3e})", flush=True)
    print(f"K5 {res['K5']}: R0 on S1's test sequence 0 against S1's C5 sha", flush=True)
    print(f"K6 {res['K6']}: the F2 arm on window 0's prompt, twice", flush=True)
    worst = {p: max((r[p]["max_kl"] for r in stdout_json("CONTROL_JSON")), default=None) for p in CONTROL_PATHS}
    print(f"K7 {res['K7']}: the R0 control on windows {', '.join(map(str, CONTROL_WINDOWS))}, R0 against R0; max KL "
          f"per path {worst}; sha-equal per path {ctl['sha_equal']}", flush=True)
    print(f"K8 {res['K8']}: the guard in {len(guards)} F2 runs (the build's probe, K4's 21, K6's 2): "
          f"{'all 0' if res['K8'] == 'PASS' else guards}", flush=True)
    res["stop"] = any(v != "PASS" for v in res.values())
    res.update({"control": ctl, "rc": rcs, "k4_values_sha256": k4["values_sha256"], "guards": guards})
    s1.say("CHECK_JSON", res)
    print("CHECK", "STOP (an anchor failed; a pipeline defect, not a verdict)" if res["stop"] else "OK", flush=True)
    return 2 if res["stop"] else 0


def states() -> int:
    stage_header("STATES")
    try:
        return states_body()
    except Stop as e:
        print(f"STATES STOP: {e}", flush=True)
        return 2


def states_body() -> int:
    if not frozen_ok():
        raise Stop("the frozen hashes, a pin, a blob or F2-E's pick differ")
    b = s1.logged_json(s1.latest("hybrid_f2a_build_*.log"), "BUILD_JSON")
    c = s1.logged_json(s1.latest("hybrid_f2a_check_*.log"), "CHECK_JSON")
    s1.say("GATES_READ_JSON", {"build_ok": b["ok"], **{k: c[k] for k in ANCHORS}})
    if not b["ok"] or c["stop"]:
        raise Stop("the build or the check is not OK")
    if not models_ok(b):
        raise Stop("a model file differs from its pin")
    free = shutil.disk_usage(WORK.parent).free / 1e9
    s1.say("DISK_JSON", {"free_gb": round(free, 1), "min_gb": MIN_DISK_GB})
    if free < MIN_DISK_GB:
        raise Stop(f"{free:.1f} GB free disk, below {MIN_DISK_GB} GB")
    for p in (STATES, KVDIR):
        if p.exists():
            shutil.rmtree(p)
    rcs = {}
    for bi, ws in enumerate(batches()):
        print(f"BATCH {bi}: windows {ws[0]}-{ws[-1]} {stamp()}", flush=True)
        for arm in ARMS:
            rcs[f"{arm}/{bi}"] = rc = child(["_prompt", arm, str(bi)])
            if rc != 0:
                return stop_rc("STATES", rc, f"_prompt {arm} {bi}")
        rcs[f"R0/{bi}"] = rc = child(["_r0", str(bi)])
        if rc != 0:
            return stop_rc("STATES", rc, f"_r0 {bi}")
        if KVDIR.exists():
            shutil.rmtree(KVDIR)
        print(f"KV_DELETED batch {bi} {not KVDIR.exists()}", flush=True)
    guards = [r["guard"] for r in stdout_json("PROMPT_JSON") if r["arm"] == "F2"]
    s1.say("STATES_DONE_JSON", {"rc": rcs, "f2_guards": guards})
    ok = all(v == 0 for v in rcs.values()) and len(guards) == N_WIN and all(g == 0 for g in guards)
    print(f"GUARD: {'0' if ok else guards} in all {len(guards)} F2 prompt runs", flush=True)
    print("STATES", "OK" if ok else "INCOMPLETE", flush=True)
    return 0 if ok else 2


def verdict() -> int:
    stage_header("VERDICT")
    try:
        return verdict_body()
    except Stop as e:
        print(f"VERDICT STOP: {e}", flush=True)
        return 3


def verdict_body() -> int:
    if not frozen_ok():
        print("VERDICT REFUSED: the frozen hashes, a pin, a blob or F2-E's pick differ", flush=True)
        return 3
    s1.start_gate("verdict")
    bl, cl, sl = (s1.latest(f"hybrid_f2a_{k}_*.log") for k in ("build", "check", "states"))
    for p in (bl, cl, sl):
        print(f"INPUT_LOG {p.name}: LF sha256 {s1.sha_lf(p)}", flush=True)
    b, c = s1.logged_json(bl, "BUILD_JSON"), s1.logged_json(cl, "CHECK_JSON")
    if not b["ok"] or c["stop"]:
        raise Stop("the build or the check is not OK")
    prm = {(r["arm"], r["w"]): r for r in s1.logged_json(sl, "PROMPT_JSON", every=True)}
    cnt = {(r["arm"], r["w"]): r for r in s1.logged_json(sl, "CONT_JSON", every=True)}
    pr0 = {(r["arm"], r["w"]): r for r in s1.logged_json(sl, "PR0_JSON", every=True)}
    one = {r["w"]: r for r in s1.logged_json(sl, "ONESHOT_JSON", every=True)}
    wins = pinned_windows()
    head, _, _ = s1.load_head()
    if s1.arr_sha(head) != s1.logged_json(RESULTS / S1_BUILD_LOG, "BUILD_JSON")["head_sha256"]:
        raise Stop("the head differs from S1's r2 build")

    def load(path: Path, rec: dict):
        if rec is None or not path.exists():
            return None
        a = np.load(path)
        return a if s1.arr_sha(a) == rec["sha256"] else None

    all_arms = ARMS + ("R0",)
    per = {a: {} for a in all_arms}
    missing = {a: {k: [] for k in SETS} for a in all_arms}
    for w, ids in enumerate(wins):
        ho = load(STATES / "R0" / f"w{w:02d}_oneshot.npy", one.get(w))
        if ho is None:
            for a in all_arms:
                for k in SETS:
                    missing[a][k].append(w)
            print(f"WINDOW {w}: no reference", flush=True)
            continue
        arms = {"R0": {"prompt": ho[:PROMPT], "prompt_rows": "all", "cont": ho[PROMPT:], "pr0": ho[P_POS:P_POS + 1]}}
        gaps = {}
        for a in ARMS:
            hp = load(STATES / a / f"w{w:02d}_prompt.npy", prm.get((a, w)))
            hc = load(STATES / a / f"w{w:02d}_cont.npy", cnt.get((a, w)))
            hd = load(STATES / a / f"w{w:02d}_pr0.npy", pr0.get((a, w)))
            if hp is None:
                for k in SETS:
                    missing[a][k].append(w)
                continue
            gaps[a] = []
            if hc is None:
                hc = np.zeros((WIN_LEN - PROMPT, hp.shape[1]), np.float32)
                gaps[a].append("C")
            if hd is None:
                hd = np.zeros((1, hp.shape[1]), np.float32)
                gaps[a].append("P_R0")
            arms[a] = {"prompt": hp, "prompt_rows": "all", "cont": hc, "pr0": hd}
        for a, sets in s1b.window_metrics(ho, arms, ids, head).items():
            for k, v in sets.items():
                if k in gaps.get(a, []):
                    missing[a][k].append(w)
                else:
                    per[a].setdefault(k, []).append(v)
        print(f"WINDOW {w} scored", flush=True)
    metrics = {a: {k: s1b.set_aggregate(v) for k, v in per[a].items() if len(v) == N_WIN} for a in all_arms}
    complete = {a: {k: len(per[a].get(k, [])) == N_WIN for k in SETS} for a in all_arms}
    kv = outcome_kv(metrics["F2"], complete["F2"]["C"])
    full = outcome_full(metrics["F2"], complete["F2"]["L"] and complete["F2"]["H"])
    would = {a: {"KV": outcome_kv(metrics[a], complete[a]["C"]),
                 "FULL": outcome_full(metrics[a], complete[a]["L"] and complete[a]["H"])} for a in REPORT_ONLY}
    f2_halves = s1b.halves(per["F2"]["C"]) if complete["F2"]["C"] else None
    sg = [r["guard"] for r in s1.logged_json(sl, "PROMPT_JSON", every=True) if r["arm"] == "F2"]
    guards = c.get("guards", []) + sg
    guards_zero = all(g == 0 for g in guards) if len(sg) == N_WIN else None
    sc = score_predictions(metrics, kv, full, would, guards_zero)
    report(metrics, kv, full, would, f2_halves, missing, sc, c, cl, per, guards)
    s1.say("VERDICT_JSON", {"F2-A/KV": kv["label"], "F2-A/FULL": full["label"],
                            "combination": combination(kv["outcome"], full["outcome"]),
                            "no_pick_change": NO_PICK_CHANGE, "complete": complete})
    s1.say("PREDICTIONS_SCORED_JSON", sc)
    return 0 if kv["outcome"] != "INCOMPLETE" and full["outcome"] != "INCOMPLETE" else 2


def report(metrics, kv, full, would, f2_halves, missing, sc, c, cl, per, guards) -> None:
    kf = CHECK / "k4_values.json"
    if kf.exists() and s1.sha_bytes(kf.read_bytes()) == c.get("k4_values_sha256"):
        print(f"\nK4's values (the check's {kf.name}, sha256 equal to the check log's): rel-L2 against L0, ours "
              f"against the F2-E run log's CASE_JSON {K4_PATH}")
        for v in json.loads(kf.read_text(encoding="utf-8")):
            print(f"  layer {v['layer']:2d} {v['name']:5s} {v['vs_l0']:.6e} logged {v['logged']:.6e} relative "
                  f"difference {v['rel_diff']:.2e} bits {'equal' if v['bits_equal'] else 'DIFFER'}")
    else:
        print("\nK4's values: the check's file is missing or differs from the check log's sha (report-only)")
    rows = s1.logged_json(cl, "CONTROL_JSON", every=True)
    print("K7's values (R0 against R0): " + "; ".join(
        f"w{r['w']} " + ", ".join(f"{p} {r[p]['max_kl']:.3g}" for p in CONTROL_PATHS) for r in rows))
    print(f"THE GUARD: {'0 in every one of' if all(g == 0 for g in guards) else 'NONZERO among'} {len(guards)} F2 runs "
          "(the build's probe, K4, K6 and the states)")
    print("\n" + "\n".join(headline(kv, full)))
    n2c = metrics.get("N2", {}).get("C")
    if n2c:
        o, nb = s1b.rule_set(n2c)
        print(f"N2's set C beside it (report-only, the like-for-like reading): mean KL {n2c['kl_mean']:.5f} "
              f"[{n2c['kl_ci'][0]:.5f}, {n2c['kl_ci'][1]:.5f}], top-1 {n2c['top1']:.4f} [{n2c['top1_ci'][0]:.4f}, "
              f"{n2c['top1_ci'][1]:.4f}], would-be {o}{' NARROW' if nb else ''}")
    print(f"\nAgainst R0 one-shot (the rule's thresholds: mean KL <= {T_KL} and top-1 >= {T_TOP1}; F2's set C decides "
          "F2-A/KV, F2's bands L and H decide F2-A/FULL; the rest is report-only; per band and set, the would-be side "
          "of the line by S1b's rule_set)")
    print(f"  {'arm':4s} {'set':6s} {'mean KL':>10s} {'95% CI':>23s} {'p50':>9s} {'p99':>9s} {'max':>9s} "
          f"{'top-1':>7s} {'95% CI':>17s} {'ppl':>8s}  side")
    for a in ARMS:
        for k in ("C", "L", "H", "P_arm", "P_R0"):
            m = metrics[a].get(k)
            if m is None:
                print(f"  {a:4s} {k:6s} INCOMPLETE: windows missing {missing[a][k]}")
                continue
            o, nb = s1b.rule_set(m)
            role = "DECIDING F2-A/KV" if (a, k) == ("F2", "C") else "DECIDING F2-A/FULL" if a == "F2" and k in BANDS \
                else "report-only"
            print(f"  {a:4s} {k:6s} {m['kl_mean']:10.5f} [{m['kl_ci'][0]:.5f}, {m['kl_ci'][1]:.5f}] "
                  f"{m['kl_p50']:9.5f} {m['kl_p99']:9.4f} {m['kl_max']:9.3f} {m['top1']:7.4f} "
                  f"[{m['top1_ci'][0]:.4f}, {m['top1_ci'][1]:.4f}] {m['ppl']:8.4f}  {o}{' NARROW' if nb else ''} "
                  f"({role})")
    for a in REPORT_ONLY:
        print(f"  {a} would-be (report-only): under S1b's rule on set C {would[a]['KV']['label']}; under S1's rule on "
              f"bands L and H {would[a]['FULL']['label']}")
    print("  R0's perplexity on the next ids: " + ", ".join(
        f"{k} {metrics['R0'][k]['ppl']:.4f}" for k in SETS if k in metrics["R0"]))
    if f2_halves is not None:
        print(f"\nF2's set C by position (report-only): mean KL {f2_halves[0]:.5f} over 2,048-2,559, "
              f"{f2_halves[1]:.5f} over 2,560-3,070")
    print("\nPredictions (they decide nothing):")
    for q, text in PREDICTIONS:
        print(f"  {q} {sc[q]}: {text}")
    s1.say("METRICS_JSON", metrics)
    s1.say("PER_WINDOW_JSON", {a: {k: {"kl": [float(w["kl"].mean()) for w in v], "top1": [float(w["top1"].mean())
                                                                                          for w in v]}
                                   for k, v in per[a].items()} for a in per})
    s1.say("F2_HALVES_JSON", f2_halves)
    s1.say("MISSING_JSON", missing)


# ---------------------------------------------------------------- selftest (synthetic only)

def selftest() -> int:
    import tempfile
    import ml_dtypes
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto as P, helper
    fails = []

    def expect(what, got, want=True):
        print(f"  {what}: {got}" + ("" if got == want else f"  (expected {want})"), flush=True)
        if got != want:
            fails.append(what)

    def bits(a):
        return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)

    def graph_steps(mx):
        """The rows where the graph's one-ulp step fires: 255 x RN32(RN64(max / 255)) < max."""
        m64 = np.asarray(mx, dtype=np.float32).astype(np.float64)
        f = (m64 / 255.0).astype(np.float32).astype(np.float64)
        return int(np.count_nonzero(255.0 * f < m64))

    print(f"F2-A SELFTEST (synthetic only: no model, no text, no x16). {stamp()}", flush=True)
    rng = np.random.default_rng(SEED)
    print("0. The pins:")
    expect("PREREG is the accepted text (LF e2408569)", s1.sha_bytes(PREREG.encode("utf-8")) == PREREG_SHA)
    rows = pin_rows()
    expect("the imported tools, the source, the texts and F2-E's log are the pinned ones",
           [r["what"] for r in rows if not r["equal"]], [])
    blobs = [blob(p, cm) for p, cm in BLOBS.items()]
    expect("tools/hybrid_s1.py and tools/hybrid_s1b.py equal their blobs at c67d6d2 and b777070",
           [b["equal"] for b in blobs], [True, True])
    expect("F2-E's pick line reads form B, FLUSH_B2, S = ROW", f2e_pick_ok()["ok"])
    expect("the source's orders equal hybrid_f2e's", f2e.order_check()["ok"])
    expect("S1b's module constants equal F2-A's (window_metrics and halves read them)",
           (s1b.SET_C, s1b.P_POS, s1b.BANDS, s1b.CHUNK, s1b.Q2_SPLIT), (SET_C, P_POS, BANDS, CHUNK, Q2_SPLIT))
    expect("onnxruntime is the pinned build", ort.__version__, ORT_VERSION)

    print("1. The F2 subgraph against hybrid_f2e's emulation, bit for bit (ORT, one-linear probes, planted rows):")

    def planted(K, M):
        kb = K // 32
        X = (rng.standard_normal((M, K)) * np.exp(rng.standard_normal((M, 1)))).astype(np.float32)
        X[0, :64] *= 300.0                                             # an outlier row
        X[1] = 0.0                                                     # a zero row
        X[2, 96:128] = 0.0                                             # a zero block
        X[3] *= np.float32(1e-4)
        X[3, :32] *= np.float32(1e7)                                   # one large block: qx = 1 elsewhere
        X[4] *= np.float32(1e4)                                        # large
        X[5] *= np.float32(1e-4)                                       # small
        found = 0
        for _ in range(20000):                                         # rows whose Sx needs the step
            r = (rng.standard_normal(K) * np.exp(rng.uniform(-8, 8))).astype(np.float32)
            _, sxr = s1.q8_ort(r[None])
            mx = sxr.max()
            f = np.float32(np.float64(mx) / 255.0)
            if 255.0 * np.float64(f) < np.float64(mx):
                X[6 + found] = r
                found += 1
                if found == 3:
                    break
        return X, kb, found

    def weights(N, kb):
        codes = rng.integers(0, 16, (N, kb, 32)).astype(np.uint8)
        d = (np.abs(rng.standard_normal(N * kb)) * 0.01 + 1e-4).astype(np.float16)
        d = d.reshape(N, kb)
        d[0] = 0.0                                                     # a zero column: Dw = 0
        d[1, 3] = 0.0                                                  # a zero weight block
        return codes, d.reshape(-1).view(np.uint16)

    runs = {}
    for K in (2048, 2560, 10240):
        M, N = 40, 24
        X, kb, found = planted(K, M)
        codes, d = weights(N, kb)
        Wp, Dw, g = wprime(codes, d)
        m, _ = linear_model(K, N, Wp, Dw, extra=("Sx", "c", "qx", "code"))
        y, guard, Sx, cc, qx, code = sess_of(m, 2).run(None, {"x": X[None]})
        q, sx = s1.q8_ort(X)
        dw = gc.f16_to_f32(d).reshape(N, kb)
        ye, Ie, ag, wg = f2e.emulate(s1.blocks(X), q, sx, codes, dw, "B", "ROW", FLUSH)
        c_ref = f2e.ceil_exact(sx.astype(np.float64), np.repeat(ag["Sx"], kb, axis=1).astype(np.float64),
                               f2e.quotient(sx.astype(np.float64), np.repeat(ag["Sx"], kb, axis=1).astype(np.float64)))
        runs[K] = (m, X, y)
        expect(f"K = {K}: y equals the emulation bit for bit",
               bool(np.array_equal(bits(y[0]), bits(ye))))
        expect(f"K = {K}: Sx, c, qx and the codes equal hybrid_f2e's",
               bool(np.array_equal(bits(Sx.reshape(M, 1)), bits(ag["Sx"])) and np.array_equal(cc.reshape(M, kb), c_ref)
                    and np.array_equal(qx.reshape(M, kb), ag["qx"]) and np.array_equal(code[0], ag["codes"])))
        nst = graph_steps(sx.max(axis=1))
        print(f"  K = {K}: the graph's step fires on {nst} of {M} rows (3 planted); hybrid_f2e's step_up fallback on "
              f"{ag['steps']}", flush=True)
        expect(f"K = {K}: the plants are present (3 stepped rows, qx = 1 blocks, codes at +-127, a zero row at 0)",
               (found, nst >= 3, int((ag["qx"] == 1).sum()) > kb, int((np.abs(ag["codes"]) == 127).sum()) > 0,
                bool(np.all(y[0][1] == 0)), bool(np.all(Ie[:, 1] == 0))), (3, True, True, True, True, True))
        expect(f"K = {K}: the guard reads 0 on clean inputs", int(guard), 0)

    print("2. The one-ulp step against hybrid_f2e's step_up (nextafter), 10^6 positive normals and every binade's "
          "ends:")
    rand = rng.integers(0x00800000, 0x7F800000, 1_000_000, dtype=np.uint32).view(np.float32)
    ends = np.array([e << 23 for e in range(1, 255)] + [(e << 23) | 0x7FFFFF for e in range(1, 255)],
                    dtype=np.uint32).view(np.float32)
    fv = np.concatenate([rand, ends])
    sn = [helper.make_node("Cast", ["f"], ["/st/f64"], name="/st/Cast", to=P.DOUBLE)] + \
        step_nodes("f", "/st/f64", "fs", "/st")
    gm = helper.make_graph(sn, "step", [helper.make_tensor_value_info("f", P.FLOAT, [None])],
                           [helper.make_tensor_value_info("fs", P.FLOAT, [None])], used_f2_consts(sn))
    got = sess_of(helper.make_model(gm, opset_imports=[helper.make_opsetid("", 21)], ir_version=10), 2).run(
        None, {"f": fv})[0]
    with np.errstate(over="ignore"):
        want, nsteps = f2e.step_up(fv, np.full(fv.size, np.inf, np.float32), QX_MAX)
    expect("the step equals step_up's nextafter on all 1,000,508 values (FLT_MAX steps to inf both ways)",
           bool(np.array_equal(got.view(np.uint32), want.view(np.uint32))) and nsteps == fv.size)

    print("3. The in-graph Sx, c and qx against fp32_up, scale_up and ceil_exact (planted quotients, random rows):")
    base = (np.float32(2.0 ** -10) * (1 + rng.integers(0, 64, 400) / np.float32(64))).astype(np.float32)
    k = rng.integers(1, 255, 400)                                     # 1..254: the hair above stays under the max
    on = (base.astype(np.float64) * k).astype(np.float32)
    above = np.nextafter(on, np.float32(np.inf))
    sxp = np.zeros((400, 4, 1), np.float32)
    sxp[:, 0, 0] = (base.astype(np.float64) * 255).astype(np.float32)   # the row max: Sx = base exactly
    sxp[:, 1, 0], sxp[:, 2, 0], sxp[:, 3, 0] = on, above, 0.0
    rnd = np.exp(rng.uniform(-30, 10, (10000, 16, 1))).astype(np.float32)
    rnd[:50] = 0.0                                                    # zero rows
    rnd[50:100, 5:] = 0.0                                             # zero blocks
    for tag, sxv in (("planted", sxp), ("random", rnd)):
        Mv, kbv = sxv.shape[0], sxv.shape[1]
        Kv = kbv * 32
        m3, _ = linear_model(Kv, 4, np.zeros((Kv, 4), np.int16), np.ones(4, np.float32), extra=("Sx", "c", "qx"),
                             sx_in=True)
        out = sess_of(m3, 2).run(None, {"x": np.zeros((1, Mv, Kv), np.float32), "sx": sxv[None]})
        Sx_o, c_o, qx_o = out[2].reshape(Mv), out[3].reshape(Mv, kbv), out[4].reshape(Mv, kbv)
        mx = sxv[:, :, 0].max(axis=1)
        Sx_r, steps = f2e.scale_up(mx, QX_MAX)
        Sb = np.repeat(Sx_r[:, None], kbv, axis=1).astype(np.float64)
        s64 = sxv[:, :, 0].astype(np.float64)
        c_r = f2e.ceil_exact(s64, Sb, f2e.quotient(s64, Sb))
        up = f2e.fp32_up(mx.astype(np.float64) / QX_MAX)
        nst = graph_steps(mx)
        eq_up = bool(np.array_equal(Sx_o.view(np.uint32), up.view(np.uint32)))
        expect(f"{tag}: Sx equals scale_up bit for bit, and fp32_up of the float64 quotient where step_up's fallback "
               f"did not fire (the graph's step fired on {nst} of {Mv} rows; step_up's fallback on {steps})",
               bool(np.array_equal(Sx_o.view(np.uint32), Sx_r.view(np.uint32))) and (steps > 0 or eq_up)
               and (tag == "planted" or nst > 1000))
        expect(f"{tag}: c equals ceil_exact, qx equals max(1, min(c, 255))",
               bool(np.array_equal(c_o, c_r) and np.array_equal(qx_o, np.maximum(1, np.minimum(c_r, QX_MAX)))))
        if tag == "planted":
            expect("planted: k on integer quotients, k + 1 a hair above, 255 on the row max, 0 on a zero block",
                   bool(np.array_equal(c_o[:, 1], k) and np.array_equal(c_o[:, 2], k + 1)
                        and np.all(c_o[:, 0] == 255) and np.all(c_o[:, 3] == 0)))

    print("4. The integer sum: MatMul(double) against Python integers at the bounds; MatMul(int64) on record:")
    Kb, Mb, Nb = 10240, 3, 4
    Xp = np.empty((Mb, Kb), np.float64)
    Xp[0], Xp[1] = 32385.0, -32385.0
    Xp[2] = rng.integers(-32385, 32386, Kb)
    Wb = np.empty((Kb, Nb), np.int16)
    Wb[:, 0], Wb[:, 1] = 1016, -1016
    Wb[:, 2] = rng.integers(-1016, 1017, Kb)
    Wb[:, 3] = np.where(np.arange(Kb) % 2 == 0, 1016, -1016)
    ref = [[sum(int(a) * int(b) for a, b in zip(Xp[i].astype(np.int64), Wb[:, j].astype(np.int64))) for j in range(Nb)]
           for i in range(Mb)]
    outs = {}
    for ty, name in ((P.DOUBLE, "double"), (P.INT64, "int64")):
        nodes = [helper.make_node("Cast", ["w"], ["wd"], name="/i/CastW", to=ty),
                 helper.make_node("Cast", ["x"], ["xd"], name="/i/CastX", to=ty),
                 helper.make_node("MatMul", ["xd", "wd"], ["I"], name="/i/MatMul")]
        gi = helper.make_graph(nodes, "isum", [helper.make_tensor_value_info("x", P.DOUBLE, [Mb, Kb]),
                                               helper.make_tensor_value_info("w", P.INT16, [Kb, Nb])],
                               [helper.make_tensor_value_info("I", ty, [Mb, Nb])])
        outs[name] = sess_of(helper.make_model(gi, opset_imports=[helper.make_opsetid("", 21)], ir_version=10),
                             2).run(None, {"x": Xp, "w": Wb})[0]
    exact = [[int(v) for v in r] for r in outs["double"].tolist()]
    expect("MatMul(double) equals the Python-integer sum exactly, |I| up to 10,240 x 32,385 x 1,016",
           exact == ref and max(abs(v) for r in ref for v in r) == 10240 * 32385 * 1016)
    expect("MatMul(int64) equals it too (kept on record)", [[int(v) for v in r] for r in outs["int64"].tolist()], ref)
    gm = rng.standard_normal((2048, 2560))
    gw = rng.standard_normal((2560, 2560))
    gg = helper.make_graph([helper.make_node("MatMul", ["a", "b"], ["c"], name="/g/MatMul")], "dgemm",
                           [helper.make_tensor_value_info("a", P.DOUBLE, [2048, 2560]),
                            helper.make_tensor_value_info("b", P.DOUBLE, [2560, 2560])],
                           [helper.make_tensor_value_info("c", P.DOUBLE, [2048, 2560])])
    sg = sess_of(helper.make_model(gg, opset_imports=[helper.make_opsetid("", 21)], ir_version=10))
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        sg.run(None, {"a": gm, "b": gw})
        ts.append(time.perf_counter() - t0)
    print(f"  report-only, a pipeline fact: MatMul(double) at 2,048 x 2,560 x 2,560 on 8 threads, median "
          f"{sorted(ts)[1]:.3f} s, {2 * 2048 * 2560 * 2560 / sorted(ts)[1] / 1e9:.0f} GFLOPS (the build's probe "
          "measures the model's)", flush=True)

    print("5. The casts round to nearest even on planted ties (against numpy, ml_dtypes, F2-0b's rne and u6e.bf16):")
    ties_i = np.array([2 ** 24 + 1, 2 ** 24 + 3, 2 ** 25 + 2, 2 ** 25 + 6, -(2 ** 24 + 1), 2 ** 38 + 2 ** 14,
                       2 ** 38 + 3 * 2 ** 14, 2 ** 39 - 1], dtype=np.int64)
    f = np.abs(rng.standard_normal(2000)).astype(np.float32) + np.float32(1e-3)
    mid = f.astype(np.float64) + np.spacing(f).astype(np.float64) / 2        # exact halfway, in float64
    dv = np.concatenate([ties_i.astype(np.float64), mid])
    cn = [helper.make_node("Cast", ["d"], ["o"], name="/c/Cast", to=P.FLOAT)]
    gd_ = helper.make_graph(cn, "d2f", [helper.make_tensor_value_info("d", P.DOUBLE, [None])],
                            [helper.make_tensor_value_info("o", P.FLOAT, [None])])
    o = sess_of(helper.make_model(gd_, opset_imports=[helper.make_opsetid("", 21)], ir_version=10), 2).run(
        None, {"d": dv})[0]
    expect("Cast(double -> float) equals numpy's RNE, and F2-0b's rne on the integer ties",
           bool(np.array_equal(o.view(np.uint32), dv.astype(np.float32).view(np.uint32))
                and np.array_equal(o[:len(ties_i)].view(np.uint32), f2e.f20b.rne(ties_i).view(np.uint32))))
    expect("the tie 2^24 + 1 rounds to 2^24 (even), 2^24 + 3 to 2^24 + 4", (float(o[0]), float(o[1])),
           (2.0 ** 24, 2.0 ** 24 + 4))
    tb = np.array([1 + 2 ** -8, 1 + 3 * 2 ** -8, -(1 + 2 ** -8), 1 + 2 ** -8 + 2 ** -20, 3.14159, 1e-30, 65504.0],
                  dtype=np.float32)
    rb = (rng.integers(0x00800000, 0x7F000000, 3000, dtype=np.uint32) & np.uint32(0xFFFF0000)) | np.uint32(0x8000)
    xv = np.concatenate([tb, rb.view(np.float32), rng.standard_normal(1000).astype(np.float32)])
    sp, pieces = split_nodes("v", "/b", 1)
    gb = helper.make_graph(sp, "bf", [helper.make_tensor_value_info("v", P.FLOAT, [None])],
                           [helper.make_tensor_value_info(pieces[0], P.FLOAT, [None])])
    ob = sess_of(helper.make_model(gb, opset_imports=[helper.make_opsetid("", 21)], ir_version=10), 2).run(
        None, {"v": xv})[0]
    expect("Cast(float -> bfloat16 -> float) equals ml_dtypes' RNE and u6e.bf16 bit for bit (3,000 planted ties)",
           bool(np.array_equal(ob.view(np.uint32), xv.astype(ml_dtypes.bfloat16).astype(np.float32).view(np.uint32))
                and np.array_equal(ob.view(np.uint32), u6e.bf16(xv).view(np.uint32))))
    sp2, p2 = split_nodes("v", "/b2", 2)
    gb2 = helper.make_graph(sp2, "bf2", [helper.make_tensor_value_info("v", P.FLOAT, [None])],
                            [helper.make_tensor_value_info(n_, P.FLOAT, [None]) for n_ in p2])
    ob2 = sess_of(helper.make_model(gb2, opset_imports=[helper.make_opsetid("", 21)], ir_version=10), 2).run(
        None, {"v": xv})
    us = u6e.split(xv, 2)
    expect("the 2-piece split equals u6e.split bit for bit", all(np.array_equal(a.view(np.uint32), b_.view(np.uint32))
                                                                   for a, b_ in zip(ob2, us)))

    print("6. The guard: 0 on clean inputs; each planted violation counted (a doctored s_x; a doctored Sx):")
    K6, M6, N6 = 128, 2, 3
    x6 = np.full((M6, K6), 0.5, np.float32)
    x6[:, 0] = 3.0                                                     # block 0 holds the row max: s_x = 3 / 127
    x6[:, 32:] = 1.0
    _, sx6 = s1.q8_ort(x6)
    Wz, Dz = np.zeros((K6, N6), np.int16), np.ones(N6, np.float32)
    mc, _ = linear_model(K6, N6, Wz, Dz, extra=("counts",), sx_in=True)
    clean = sess_of(mc, 2).run(None, {"x": x6[None], "sx": sx6[None, :, :, None]})
    doc = sx6.copy()
    doc[0, 0] = np.float32(0.99 * 3.0 / 127)                          # row 0: one code rounds to 128
    bad = sess_of(mc, 2).run(None, {"x": x6[None], "sx": doc[None, :, :, None]})
    expect("clean: guard 0, counts (code, c, Sx) = (0, 0, 0)",
           (int(clean[1]), [int(v.sum()) for v in clean[2:5]]), (0, [0, 0, 0]))
    expect("a doctored s_x (0.99 x amax / 127 on one block): one code over 127, and nothing else",
           (int(bad[1]), [int(v.sum()) for v in bad[2:5]]), (1, [1, 0, 0]))
    ms, _ = linear_model(K6, N6, Wz, Dz, extra=("counts",), Sx_in=True)
    Sx_true, _ = f2e.scale_up(sx6.max(axis=1), QX_MAX)
    Sx_doc = Sx_true.copy()
    Sx_doc[1] = np.nextafter(Sx_doc[1], np.float32(0))               # row 1: one ulp under the smallest valid
    ok6 = sess_of(ms, 2).run(None, {"x": x6[None], "Sx": Sx_true[None, :, None, None]})
    bd6 = sess_of(ms, 2).run(None, {"x": x6[None], "Sx": Sx_doc[None, :, None, None]})
    expect("the true Sx fed in: guard 0", int(ok6[1]), 0)
    expect("a doctored Sx one ulp under: the Sx count and c > 255 on the row max block (one event: s_x / Sx > 255 "
           "there), no code over", (int(bd6[1]), [int(v.sum()) for v in bd6[2:5]]), (2, [0, 1, 1]))

    print("7. The windows: offsets and the non-overlap asserts on a synthetic token list; the download's STOP:")
    n_tok = 60000
    T = rng.integers(3, 1000, n_tok).tolist()
    offs = s1b.offsets(N_WIN, 0, WIN_STRIDE, WIN_LEN - 1)
    expect("16 windows at 0, 3,071, ..., the last ending at 49,136", (len(offs), offs[0], offs[1][0], offs[-1][1]),
           (16, (0, 3071), 3071, 49136))
    tp = text_pins(T, ([], []))
    expect("the synthetic windows pass every assert", tp["asserts"]["ok"])
    expect("each window is [BOS] + 3,071 tokens; the prompt its first 2,048 ids",
           tp["prompt_sha"][1] == gds.ids_sha([BOS] + T[3071:3071 + 2047]))
    bad_offs = s1b.offsets(N_WIN, 0, WIN_STRIDE - 1, WIN_LEN - 1)
    expect("pairwise_disjoint on a deliberate overlap, stride 3,070 (False = caught)",
           s1b.window_asserts(bad_offs, n_tok, 0, tp["prompt_sha"], [])["pairwise_disjoint"], False)
    expect("the sha assert on a prompt equal to a prior prompt (False = caught)",
           s1b.window_asserts(offs, n_tok, 0, tp["prompt_sha"], [tp["prompt_sha"][4]])["no_s1_prompt_sha"], False)
    expect("text_pins' asserts on a text too short for 16 windows (False = caught)",
           text_pins(T[:40000], ([], []))["asserts"]["ok"], False)
    expect("text_pins' sha assert on an S1 sha equal to window 0's prompt (False = caught)",
           text_pins(T, ([gds.ids_sha([BOS] + T[:2047])], []))["asserts"]["no_s1_prompt_sha"], False)
    other = ([gds.ids_sha(rng.integers(3, 1000, 2048).tolist())], [gds.ids_sha(rng.integers(3, 1000, 2048).tolist())])
    expect("unrelated S1 and S1b shas pass", text_pins(T, other)["asserts"]["no_s1_prompt_sha"], True)
    import huggingface_hub
    import requests
    from unittest import mock
    said = []
    for exc in (requests.ConnectionError("BODY"), requests.Timeout("BODY"), OSError("BODY")):
        with mock.patch.object(huggingface_hub, "hf_hub_download", side_effect=exc):
            try:
                val_file()
                said.append("no STOP")
            except Stop as e:
                said.append(str(e))
    expect("the download failing mid-way (a mocked ConnectionError, Timeout and OSError; no network): a STOP naming "
           "the class only", said, [f"the download failed ({c})" for c in ("ConnectionError", "Timeout", "OSError")])

    print("8. The verdict code: the rules, the headline and combinations, the slicing, the KV slice, the bootstrap:")

    def agg(kl, t1, finite=True, ci=None, tci=None):
        return {"kl_mean": kl, "top1": t1, "kl_ci": ci or [kl, kl], "top1_ci": tci or [t1, t1], "finite": finite}
    good = {"C": agg(0.005, 0.97), "L": agg(0.004, 0.975), "H": agg(0.004, 0.975)}
    expect("KV: PASS", outcome_kv(good, True)["label"], "PASS")
    expect("KV: FAIL above T_KL", outcome_kv({"C": agg(0.0124, 0.99)}, True)["label"], "FAIL")
    expect("KV: PASS NARROW", outcome_kv({"C": agg(0.005, 0.958, tci=[0.955, 0.961])}, True)["label"], "PASS NARROW")
    expect("KV: FAIL NARROW", outcome_kv({"C": agg(0.013, 0.97, ci=[0.011, 0.014])}, True)["label"], "FAIL NARROW")
    expect("KV: FAIL when non-finite", outcome_kv({"C": agg(0.0, 1.0, finite=False)}, True)["label"], "FAIL")
    expect("KV: INCOMPLETE when a window is missing", outcome_kv(good, False)["label"], "INCOMPLETE")
    expect("FULL: PASS", outcome_full(good, True)["label"], "PASS")
    expect("FULL: FAIL when band H fails", outcome_full({**good, "H": agg(0.02, 0.97)}, True)["label"], "FAIL")
    expect("FULL: PASS NARROW (band L's interval holds T_TOP1)",
           outcome_full({**good, "L": agg(0.004, 0.958, tci=[0.955, 0.961])}, True),
           {"outcome": "PASS", "narrow": ["L"], "label": "PASS NARROW"})
    expect("FULL: FAIL NARROW", outcome_full({**good, "H": agg(0.013, 0.97, ci=[0.011, 0.014])}, True)["label"],
           "FAIL NARROW")
    expect("FULL: FAIL when a band is non-finite", outcome_full({**good, "L": agg(0.0, 1.0, finite=False)}, True)
           ["label"], "FAIL")
    expect("FULL: INCOMPLETE", outcome_full(good, False)["label"], "INCOMPLETE")
    P_, F_, I_ = ({"outcome": o, "narrow": False, "label": o} for o in ("PASS", "FAIL", "INCOMPLETE"))
    expect("the headline prints KV first, then FULL, the combination and the N2 note",
           [ln.split(" (")[0] for ln in headline(P_, F_)[:2]] + headline(P_, F_)[2:],
           ["F2-A/KV", "F2-A/FULL", COMBINATIONS["KV PASS, FULL FAIL"], NO_PICK_CHANGE])
    expect("combinations: KV PASS / FULL FAIL, KV FAIL (either FULL), both PASS, the INCOMPLETEs",
           [combination(*x) for x in (("PASS", "FAIL"), ("FAIL", "PASS"), ("FAIL", "INCOMPLETE"), ("PASS", "PASS"),
                                      ("INCOMPLETE", "PASS"), ("PASS", "INCOMPLETE"))],
           [COMBINATIONS[k] for k in ("KV PASS, FULL FAIL", "KV FAIL", "KV FAIL", "BOTH PASS", "KV INCOMPLETE",
                                      "FULL INCOMPLETE")])
    expect("NARROW changes no combination", combination("PASS", "FAIL") == combination(
        outcome_kv({"C": agg(0.005, 0.958, tci=[0.955, 0.961])}, True)["outcome"], "FAIL"))
    Hd, V = 8, 50
    head = rng.standard_normal((V, Hd)).astype(np.float32)
    ho = rng.standard_normal((WIN_LEN, Hd)).astype(np.float32)
    idl = rng.integers(0, V, WIN_LEN).tolist()

    def arms_eq():
        return {a: {"prompt": ho[:PROMPT].copy(), "prompt_rows": "all", "cont": ho[PROMPT:].copy(),
                    "pr0": ho[P_POS:P_POS + 1].copy()} for a in ARMS + ("R0",)}
    mw = s1b.window_metrics(ho, arms_eq(), idl, head)
    expect("every arm, R0's pseudo-arm too, gets C, P_arm, P_R0, L and H",
           all(sorted(mw[a]) == sorted(SETS) for a in mw), True)
    expect("positions: C 1,023, P 1 and 1, L 1,024, H 1,023",
           [len(mw["F2"][k]["kl"]) for k in ("C", "P_arm", "P_R0", "L", "H")], [1023, 1, 1, 1024, 1023])
    expect("equal states: KL 0 and top-1 1 everywhere",
           all(np.abs(mw[a][k]["kl"]).max() < 1e-9 and mw[a][k]["top1"].min() == 1 for a in mw for k in mw[a]))

    def hit(mut):
        a = arms_eq()
        mut(a["F2"])
        mm_ = s1b.window_metrics(ho, a, idl, head)["F2"]
        return sorted(k for k in mm_ if np.abs(mm_[k]["kl"]).max() > 1e-6)
    expect("prompt row 2,047 moves P-arm only", hit(lambda a: a["prompt"].__setitem__(P_POS, 5.0)), ["P_arm"])
    expect("prompt row 100 moves band L only", hit(lambda a: a["prompt"].__setitem__(100, 5.0)), ["L"])
    expect("prompt row 2,046 moves band H only", hit(lambda a: a["prompt"].__setitem__(2046, 5.0)), ["H"])
    expect("continuation row 0 moves set C only", hit(lambda a: a["cont"].__setitem__(0, 5.0)), ["C"])
    expect("continuation row 1,023 (position 3,071, unscored) moves nothing",
           hit(lambda a: a["cont"].__setitem__(1023, 5.0)), [])
    expect("the P-R0 row moves P-R0 only", hit(lambda a: a["pr0"].__setitem__(0, 5.0)), ["P_R0"])
    hv = s1b.halves([mw["F2"]["C"]] * 3)
    expect("halves split set C at 2,560 and read 0 on equal states", (len(range(SET_C[0], Q2_SPLIT)),
                                                                        abs(hv[0]) < 1e-9 and abs(hv[1]) < 1e-9),
           (512, True))
    Wq, Wk, Wv = (rng.standard_normal((Hd, Hd)) for _ in range(3))
    emb = rng.standard_normal((V, Hd))

    def attend(x, past=None):
        q_, k_, v_ = x @ Wq, x @ Wk, x @ Wv
        if past is not None:
            k_ = np.concatenate([past["k"][0, 0], k_])
            v_ = np.concatenate([past["v"][0, 0], v_])
        n0 = len(k_) - len(x)
        s_ = q_ @ k_.T / np.sqrt(Hd)
        s_ = np.where(np.arange(len(k_))[None, :] <= (n0 + np.arange(len(x)))[:, None], s_, -np.inf)
        pr = np.exp(s_ - s_.max(axis=1, keepdims=True))
        return (pr / pr.sum(axis=1, keepdims=True)) @ v_, {"k": k_[None, None], "v": v_[None, None]}
    xe = emb[rng.integers(0, V, WIN_LEN)]
    full_, _ = attend(xe)
    _, kvp = attend(xe[:PROMPT])
    step_, _ = attend(xe[P_POS:PROMPT], past=s1b.slice_kv(kvp, P_POS))
    expect("the KV slice to 0-2,046 plus one step equals the one-shot at 2,047",
           bool(np.allclose(step_[0], full_[P_POS], rtol=0, atol=1e-12)))
    expect("the bootstrap is deterministic (seed 20260924, 10,000 resamples)",
           s1.bootstrap(np.arange(16.0)) == s1.bootstrap(np.arange(16.0)))

    def ctl_rows(bad_w=None, bad_p=None, val=2e-5):
        out_ = []
        for w in CONTROL_WINDOWS:
            r = {"w": w}
            for p in CONTROL_PATHS:
                r[p] = {"max_kl": val if (w, p) == (bad_w, bad_p) else 0.0, "sha_equal": (w, p) != (bad_w, bad_p)}
            out_.append(r)
        return out_
    expect("control: all 0 PASS; each path failing alone FAIL; a missing window FAIL; exactly 1e-5 PASS",
           [control_outcome(ctl_rows())["outcome"]] + [control_outcome(ctl_rows(8, p))["outcome"]
                                                       for p in CONTROL_PATHS]
           + [control_outcome(ctl_rows()[:-1])["outcome"], control_outcome(ctl_rows(0, "a", 1e-5))["outcome"]],
           ["PASS", "FAIL", "FAIL", "FAIL", "FAIL", "PASS"])
    mets = {"F2": {"C": agg(0.002, 0.985), "L": agg(0.0015, 0.99), "H": agg(0.0015, 0.99),
                   "P_R0": agg(0.003, 1.0), "P_arm": agg(0.003, 1.0)},
            "R": {"C": agg(0.0015, 0.99), "L": agg(0.0011, 0.987), "H": agg(0.001, 0.986)},
            "N2": {"C": agg(0.009, 0.958), "L": agg(0.06, 0.895), "H": agg(0.05, 0.91)}}
    wo = {a: {"KV": outcome_kv(mets[a], True), "FULL": outcome_full(mets[a], True)} for a in ("R", "N2")}
    sc = score_predictions(mets, outcome_kv(mets["F2"], True), outcome_full(mets["F2"], True), wo, True)
    expect("predictions: all HIT on a set built to hit", sc, {q: "HIT" for q, _ in PREDICTIONS})
    sc2 = score_predictions(mets, P_, {"outcome": "PASS", "narrow": ["L"], "label": "PASS NARROW"}, wo, False)
    expect("predictions: FA-P1a MISS on PASS NARROW, FA-P8 MISS on a nonzero guard", (sc2["FA-P1a"], sc2["FA-P8"]),
           ("MISS", "MISS"))
    expect("predictions: NOT SCORED without the arms", score_predictions({}, I_, I_, {}, None)["FA-P2"], "NOT SCORED")
    names = [n for n, _ in verdict_sources()]
    expect("VERDICT_CODE covers F2-A's functions and the imported S1b and S1 kernels", names,
           list(VERDICT_FUNCS) + [f"hybrid_s1b.{f}" for f in S1B_VERDICT_FUNCS]
           + [f"hybrid_s1.{f}" for f in S1_VERDICT_FUNCS])
    own = s1.sha_bytes(("".join(inspect.getsource(globals()[f]) for f in VERDICT_FUNCS)
                        + json.dumps(verdict_constants(), sort_keys=True)).encode("utf-8"))
    expect("the imported sources change VERDICT_CODE", own != verdict_code_sha())

    print("9. K2's and K3's comparators, and the surgery end to end, on a synthetic two-layer graph:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td = Path(td)
        K2_, S_ = 64, 5
        wts, nodes, inits = {}, [], []
        data = s1.DataFile(td / "model.onnx.data")
        prev = "x"
        for L in range(2):
            for proj, inp in (("q", prev), ("k", prev), ("o", None)):
                nm = f"/model/layers.{L}/attn/{proj}_proj/MatMulNBits"
                codes = rng.integers(0, 16, (K2_, K2_ // 32, 32)).astype(np.uint8)
                d = (np.abs(rng.standard_normal(K2_ * K2_ // 32)) * 0.01 + 0.001).astype(np.float16).view(np.uint16)
                wts[nm] = (codes, d)
                a = inp if inp is not None else f"/model/layers.{L}/add/output_0"
                qb = data.add(f"model.layers.{L}.{proj}.qweight", gd.pack_ort(codes), P.UINT8,
                              list(gd.pack_ort(codes).shape))
                sc_ = data.add(f"model.layers.{L}.{proj}.scales", gc.f16_to_f32(d).astype(np.float32), P.FLOAT,
                               [K2_ * K2_ // 32])
                inits += [qb, sc_]
                nodes.append(helper.make_node("MatMulNBits", [a, qb.name, sc_.name], [nm + "/output_0"], name=nm,
                                              domain="com.microsoft", K=K2_, N=K2_, bits=4, block_size=32,
                                              accuracy_level=0))
                if proj == "k":
                    nodes.append(helper.make_node("Add", [f"/model/layers.{L}/attn/q_proj/MatMulNBits/output_0",
                                                          nm + "/output_0"], [f"/model/layers.{L}/add/output_0"],
                                                  name=f"/model/layers.{L}/add"))
            prev = f"/model/layers.{L}/attn/o_proj/MatMulNBits/output_0"
        nodes.append(helper.make_node("Identity", [prev], [s1.HIDDEN_NAME],
                                      name="/model/layers.34/final_norm_layernorm"))
        hw = data.add(s1.HEAD_W, rng.standard_normal((K2_, 16)).astype(np.float32), P.FLOAT, [K2_, 16])
        inits.append(hw)
        nodes.append(helper.make_node("MatMul", [s1.HIDDEN_NAME, s1.HEAD_W], ["logits"], name=s1.HEAD_NODE))
        data.close()
        g = helper.make_graph(nodes, "syn", [helper.make_tensor_value_info("x", P.FLOAT, [1, "S", K2_])],
                              [helper.make_tensor_value_info("logits", P.FLOAT, [1, "S", 16])], inits)
        onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft",
                                                                                                         1)],
                                    ir_version=10), str(td / "model.onnx"))
        base = s1.base_cut(td / "model.onnx", "model.onnx.data", K2_)
        fdata = s1.DataFile(td / "f2.onnx.data")
        arm = f2_replace(s1.base_cut(td / "model.onnx", "model.onnx.data", K2_), lambda n_: wprime(*wts[n_])[:2],
                         fdata)
        fd = fdata.close()
        onnx.save(arm, str(td / "F2.onnx"))
        expect("f2.onnx.data holds the six W' with no padding (4096-multiples)", fd["bytes"], 6 * K2_ * K2_ * 2)
        cmpk = k2_compare(base, arm)
        expect("K2 on the synthetic F2 graph", cmpk["ok"])
        extra = onnx.load(str(td / "F2.onnx"), load_external_data=False)
        extra.graph.node.insert(3, helper.make_node("Identity", ["x"], ["planted"], name="/model/planted"))
        expect("K2's ok on a graph with a planted extra node (False = caught)", k2_compare(base, extra)["ok"], False)
        renamed = onnx.load(str(td / "F2.onnx"), load_external_data=False)
        next(n for n in renamed.graph.node if n.name == "/model/layers.0/add").name = "changed"
        expect("K2's ok on a graph with a changed node (False = caught)", k2_compare(base, renamed)["ok"], False)
        nog = onnx.load(str(td / "F2.onnx"), load_external_data=False)
        nog.graph.output.pop()
        expect("K2's ok on a graph missing the guard output (False = caught)", k2_compare(base, nog)["ok"], False)
        so = s1.session_options(None)
        so.enable_profiling = True
        so.profile_file_prefix = str(td / "prof_F2")
        ss = ort.InferenceSession(str(td / "F2.onnx"), so, providers=["CPUExecutionProvider"])
        xin = rng.standard_normal((1, S_, K2_)).astype(np.float32)
        xin[0, 2] = 0.0                                                # a zero row through the whole graph
        hid, gv = ss.run([s1.HIDDEN_NAME, GUARD_NAME], {"x": xin})
        prof = ss.end_profiling()
        cen = census_of(prof)
        r0m = base                                                     # the synthetic base is at accuracy level 0
        c_ok = census_ok(cen, s1.static_counts(arm), s1.static_counts(r0m), static_rest(arm), 6)
        expect("K3 on the synthetic graph's profile: 6 MatMul, all double; executed equals static; rest equals R0's",
               c_ok["ok"])
        planted_ = {"executed": {**cen["executed"], "FusedMatMul": 1}, "matmul_double": cen["matmul_double"]}
        expect("K3's no_fused on a profile with a planted fused op (False = caught)",
               census_ok(planted_, s1.static_counts(arm), s1.static_counts(r0m), static_rest(arm), 6)["no_fused"],
               False)
        half = {"executed": cen["executed"], "matmul_double": cen["matmul_double"] - 1}
        expect("K3's ok on a profile with a MatMul that is not double (False = caught)",
               census_ok(half, s1.static_counts(arm), s1.static_counts(r0m), static_rest(arm), 6)["ok"], False)

        def emu(X, nm):
            codes, d = wts[nm]
            q_, sx_ = s1.q8_ort(X)
            return f2e.emulate(s1.blocks(X), q_, sx_, codes, gc.f16_to_f32(d).reshape(K2_, K2_ // 32), "B", "ROW",
                               FLUSH)[0]
        h = xin[0]
        for L in range(2):
            qv = emu(h, f"/model/layers.{L}/attn/q_proj/MatMulNBits")
            kv_ = emu(h, f"/model/layers.{L}/attn/k_proj/MatMulNBits")
            h = emu((qv + kv_).astype(np.float32), f"/model/layers.{L}/attn/o_proj/MatMulNBits")
        expect("the surgery end to end: the graph's hidden states equal the chained emulation bit for bit",
               bool(np.array_equal(bits(hid[0]), bits(h))))
        expect("the graph's guard reads 0 (each of the 6 linears contributes its input's count)", int(gv), 0)
        expect("the activation side is built once per input: 4 of them for 6 linears",
               sum(1 for n_ in arm.graph.node if n_.name.endswith("/ReshapeBlocks")), 4)
        del ss

    print("10. The watchdog and the start gate (S1's), the probe's runaway ceiling, the states' finiteness, and "
          "determinism:")
    for mode, arg, want_rc, marker, done in (("_watchdog", 1e6, 4, "MEMORY_ABORT_JSON", "WATCHDOG_MAIN_DONE"),
                                             ("_watchdog", 0.0, 0, "MEMORY_ABORT_JSON", "WATCHDOG_MAIN_DONE"),
                                             ("_gate", 1e6, 4, "MEMORY_REFUSE", "GATE_PASSED"),
                                             ("_gate", 0.0, 0, "MEMORY_REFUSE", "GATE_PASSED"),
                                             ("_line", 0.3, 5, "PROBE STOP", "LINE_MAIN_DONE"),
                                             ("_line", 60.0, 0, "PROBE STOP", "LINE_MAIN_DONE")):
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), mode, repr(arg)], capture_output=True,
                           text=True, encoding="utf-8", timeout=300)
        expect(f"{mode} {arg:g}: rc {want_rc}, {marker} {'printed' if want_rc else 'absent'}, the main thread "
               f"{'stopped' if want_rc else 'finished'}", (r.returncode, marker in r.stdout, done in r.stdout),
               (want_rc, want_rc != 0, want_rc == 0))
    expect("the probe's timer is the runaway ceiling, above the line the build STOPs on (ceiling, line)",
           (PROBE_CEILING_S, PROBE_LINE_S, PROBE_CEILING_S > PROBE_LINE_S), (1800.0, 420.0, True))
    bad_, good_ = np.array([[0.0, np.nan]], np.float32), np.zeros((1, 2), np.float32)
    expect("the states print finiteness for R, N2 and R0, and none for the deciding arm F2 (the prereg, section 9)",
           (finite_of("F2", bad_), finite_of("F2", good_), finite_of("R", bad_), finite_of("N2", good_),
            finite_of("R0", good_)), ({}, {}, {"finite": False}, {"finite": True}, {"finite": True}))
    m1, X1, y1 = runs[2560]
    y2 = sess_of(m1, 2).run(["y"], {"x": X1[None]})[0]
    y3 = sess_of(m1, 8).run(["y"], {"x": X1[None]})[0]
    expect("deterministic: a second session, and 8 threads against 2, give the same bits",
           bool(np.array_equal(bits(y1), bits(y2)) and np.array_equal(bits(y1), bits(y3))))

    print("The frozen hashes:")
    for kk, v in hashes().items():
        print(f"  {kk} {v}")
    s1.say("MEM_JSON", {"child": "F2-A selftest", **s1.own_memory()})
    print("SELFTEST", "OK" if not fails else f"FAILED: {fails}", flush=True)
    return 0 if not fails else 1


# ---------------------------------------------------------------- main

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("selftest", "prereg", "build", "check", "states", "verdict", "_probe", "_k2",
                                     "_k4", "_k5", "_k6", "_control", "_ccompare", "_prompt", "_r0", "_watchdog",
                                     "_gate", "_line"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.mode == "_watchdog":
        return s1.watchdog_child(a.args[0])
    if a.mode == "_gate":
        return gate_child(a.args[0])
    if a.mode == "_line":
        return line_child(a.args[0])
    if a.mode == "_prompt":
        return prompt_child(a.args[0], a.args[1])
    if a.mode == "_r0":
        return r0_child(a.args[0])
    return {"selftest": selftest, "prereg": prereg, "build": build, "check": check, "states": states,
            "verdict": verdict, "_probe": probe_child, "_k2": k2_child, "_k4": k4_child, "_k5": k5_child,
            "_k6": k6_child, "_control": control_child, "_ccompare": ccompare_child}[a.mode]()


if __name__ == "__main__":
    sys.exit(main())
