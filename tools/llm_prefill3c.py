#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM study stage 3c: prefill weight GEMMs at Gemma 3 4B's shapes, M = 2048 and 8192, on speed, accuracy
and energy per prompt token (plan v2; the user, 2026-09-24: "Go with U1-U12").

One layer's seven linears (the release's blk.16, all Q4_0) on three chips: the CPU (ONNX Runtime, pinned
8 and 16 threads), DirectML on the Radeon 780M, and the NPU (mlir-aie whole_array). The rivals run one
session per arm holding the seven MatMuls; the NPU runs gate and up as 2560-wide column slices and down
split-K. The pre-registration is PREREG below; the rules are verdict()'s code.

    python tools/llm_prefill3c.py models              # step 2: the CPU and DirectML layer models, their
                                                      #   DirectML placement and a small output check (resnet_env17)
    python tools/llm_prefill3c.py placement MODEL     # (models' subprocess) one model's DirectML placement, verbose
    python tools/llm_prefill3c.py insts-fit           # stage 3's 28 NPU builds against the insts.bin fit (read-only)
    python tools/llm_prefill3c.py prereg              # the plan: its text, PROTOCOL_JSON, the predictions, the step-2
                                                      #   and insts-fit logs' hashes and the dropped-arm list
    python tools/llm_prefill3c.py build               # step 4, IRON env: the NPU xclbins, compile only, one row each
    python tools/llm_prefill3c.py inputs              # step 4: X, W, int8 copies, references, exact int32 SHAs
    python tools/llm_prefill3c.py verdict LOG [LOG]   # the mechanical verdict over the sittings' WINDOW_JSON records
    python tools/llm_prefill3c.py selftest            # tiny models and synthetic verdicts; no chip, no GPU

Code-only commits after the second plan commit (413a562) add the rest: step 4's builds and inputs first,
then the window-state code, the load check and the sitting. Each keeps PREREG, PROTOCOL_JSON and
VERDICT_CODE_SHA256 byte-identical, and every later 3c log prints the three hashes.
"""
import argparse
import hashlib
import itertools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

WORK = ROOT / "scratch/llm/prefill3c"
MODELS = WORK / "models"
BUILD3 = ROOT / "build/llm_prefill"                   # stage 3's NPU builds (read-only here)
RESULTS = ROOT / "results/llm"

# ---------------------------------------------------------------- the protocol (pre-registered)

LAYER = 16
# name, GGUF tensor, K, N, input
LINEARS = [("q", "attn_q", 2560, 2048, "x_attn"), ("k", "attn_k", 2560, 1024, "x_attn"),
           ("v", "attn_v", 2560, 1024, "x_attn"), ("o", "attn_output", 2048, 2560, "x_o"),
           ("gate", "ffn_gate", 2560, 10240, "x_ffn"), ("up", "ffn_up", 2560, 10240, "x_ffn"),
           ("down", "ffn_down", 10240, 2560, "x_down")]
INPUTS = {"x_attn": 2560, "x_o": 2048, "x_ffn": 2560, "x_down": 10240}
LAYERS = 34
MS = (2048, 8192)
SEED = 20260924
ZP = 128                                              # C-i8: u8 x s8 with A's zero point 128 = s8 x s8
THREADS = (8, 16)
# the arms, pass 1's order; pass 2 is its reverse. N-w4 is report-only.
ORDER = ["C-fp32@8", "C-fp32@16", "C-i8@8", "C-i8@16", "C-nb0@8", "C-nb0@16", "C-nb4@8", "C-nb4@16",
         "D-fp16", "D-fp32", "D-i8", "D-nb16", "N-bf16", "N-i8", "N-w4"]
REPORT_ONLY = ("N-w4",)
NPU_ARMS = ("N-bf16", "N-i8")
MODEL_OF = {"C-fp32": "fp32", "C-i8": "i8_u8", "C-nb0": "nb0", "C-nb4": "nb4",
            "D-fp16": "fp16", "D-fp32": "fp32", "D-i8": "i8_s8", "D-nb16": "nb16"}
DML_MODELS = ("fp16", "fp32", "i8_s8", "nb16")
IDLE_S, WARMUP_S, WINDOW_S, SETTLE_S, TRIM_S = 60, 20, 60, 10, 1
MIN_ROWS, MIN_ITERATIONS = 50, 5
REPEAT_MAX = 0.10                                     # |p1 - p2| <= 0.10 x (p1 + p2) / 2, 3b's form
MARGIN = 1.10                                         # faster / fewer joules: by at least 1.10x
ACC_TIE = 1.10                                        # at least as accurate: rel-L2 <= 1.10 x the NPU arm's
BF16_ACC_MAX = 1e-4                                   # N-bf16 against its own bf16-rounded product
HF_MAX, PAGES_MAX = 25.0, 1000.0                      # (e)'s memory rule (U10)
RERUN_VOID = 1                                        # U12: at most one re-run of a VOID window, same pass
NPU_TILES = {"bf16": {"P": (32, 64, 128, 1), "F": (64, 64, 64, 1)},
             "i8": {"P": (64, 128, 64, 1), "F": (64, 64, 64, 0)},
             "w4": {"P": (64, 128, 64, 1)}}           # m, k, n, c_single_buffer; w4: native, unroll2
NPU_PIECE = 2560                                      # gate/up column slices and down's split-K pieces
NPU_CONTEXTS = 4                                      # q; k and v; o; the 2560 pieces
NPU_SHAPES = ((2560, 2048), (2560, 1024), (2048, 2560), (2560, 2560))   # K x N: q; k and v; o; the pieces
INSTS_FIT = (16, 2576)                                # insts.bin = 16 + 2576 * M / (8 m) B (stage 3's 28 builds)


def protocol() -> dict:
    return {"stage": "3c", "plan": "v2", "layer": LAYER, "linears": LINEARS, "inputs": INPUTS, "layers": LAYERS,
            "ms": MS, "seed": SEED, "order": ORDER, "report_only": REPORT_ONLY, "npu_arms": NPU_ARMS,
            "model_of": MODEL_OF, "threads": THREADS, "zp": ZP,
            "cpu_pin": "8: one logical CPU per physical core; 16: every logical CPU; the caller pinned to the first",
            "idle_s": IDLE_S, "warmup_s": WARMUP_S, "window_s": WINDOW_S, "settle_s": SETTLE_S, "trim_s": TRIM_S,
            "min_rows": MIN_ROWS, "min_iterations": MIN_ITERATIONS, "repeat_max": REPEAT_MAX, "margin": MARGIN,
            "acc_tie": ACC_TIE, "bf16_acc_max": BF16_ACC_MAX, "hf_max": HF_MAX, "pages_max": PAGES_MAX,
            "rerun_void": RERUN_VOID, "npu_tiles": NPU_TILES, "npu_piece": NPU_PIECE, "npu_contexts": NPU_CONTEXTS,
            "insts_fit": INSTS_FIT, "e_unit": "J per prompt token per layer, above idle (printed in mJ)",
            "t_unit": "ms per layer iteration, median over the window", "npu_shapes": NPU_SHAPES,
            "wiring_max": WIRING_MAX, "void_rel_l2_max": {arm: void_bound(arm) for arm in ORDER}}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def say(tag: str, obj) -> None:
    print(f"{tag} " + json.dumps(obj), flush=True)


def layer_flops(M: int) -> float:
    return 2.0 * M * sum(K * N for _, _, K, N, _ in LINEARS)


# ---------------------------------------------------------------- the release's layer

def release_layer(layer: int = LAYER) -> dict:
    """{name: (codes [N, K/32, 32] uint8, d [N * K/32] f16 bits)} for the layer's seven linears, from the
    pinned GGUF, verified by size and SHA-256 first."""
    import gemma_compress as gc
    import gemma_decode as gd
    from huggingface_hub import hf_hub_download
    g = gc.GGUF_PIN
    p = Path(hf_hub_download(g["repo"], g["file"], revision=g["revision"]))
    got = sha(p)
    say("GGUF_JSON", {"file": g["file"], "revision": g["revision"], "size": p.stat().st_size, "sha256": got,
                      "ok": p.stat().st_size == g["size"] and got == g["sha256"]})
    if p.stat().st_size != g["size"] or got != g["sha256"]:
        sys.exit("the GGUF differs from its pin; nothing is built")
    mm, _, _, tensors = gc.read_gguf(p)
    by = {t["name"]: t for t in tensors}
    out = {}
    for name, gg, K, N, _ in LINEARS:
        t = by[f"blk.{layer}.{gg}.weight"]
        assert t["tname"] == "Q4_0" and t["dims"] == [K, N], t
        out[name] = gd.gguf_linear(mm, t)
        say("LINEAR_JSON", {"name": name, "tensor": t["name"], "type": t["tname"], "K": K, "N": N})
    return out


def dense(codes: np.ndarray, d: np.ndarray) -> np.ndarray:
    """W [K, N] fp32, exactly d * (q - 8)."""
    import gemma_compress as gc
    import gemma_decode as gd
    return np.ascontiguousarray(gd.dequant(codes, gc.f16_to_f32(d)).T)


def quant_w(w: np.ndarray):
    """int8 per output column, symmetric (3's form)."""
    s = np.abs(w.astype(np.float64)).max(axis=0) / 127.0
    return np.clip(np.rint(w.astype(np.float64) / s), -127, 127).astype(np.int8), s


def quant_x(x: np.ndarray):
    """int8 per tensor, symmetric (3's form)."""
    s = float(np.abs(x).max()) / 127.0
    return np.clip(np.rint(x.astype(np.float64) / s), -127, 127).astype(np.int8), s


# ---------------------------------------------------------------- the rivals' layer models

def layer_model(kind: str, lin: dict, linears=LINEARS) -> bytes:
    """One graph with the layer's seven linears in order (q, k, v, o, gate, up, down), four inputs and
    seven outputs. kind: fp32 | fp16 (MatMul), i8_u8 (MatMulInteger u8 x s8, zero point 128), i8_s8
    (MatMulInteger s8 x s8), nb0 | nb4 (MatMulNBits, fp32 x and scales, accuracy_level 0 | 4), nb16
    (MatMulNBits, fp16 x and scales). lin: {name: (codes, d)}; the int8 weights are quantized here."""
    import gemma_compress as gc
    import gemma_decode as gd
    from onnx import TensorProto, helper, numpy_helper
    xt = {"fp32": TensorProto.FLOAT, "fp16": TensorProto.FLOAT16, "i8_u8": TensorProto.UINT8,
          "i8_s8": TensorProto.INT8, "nb0": TensorProto.FLOAT, "nb4": TensorProto.FLOAT,
          "nb16": TensorProto.FLOAT16}[kind]
    yt = TensorProto.INT32 if kind.startswith("i8") else xt
    used = sorted({x for *_, x in linears}, key=list(INPUTS).index)
    ks = {x: next(K for _, _, K, _, xx in linears if xx == x) for x in used}
    nodes, inits = [], []
    if kind == "i8_u8":
        inits.append(numpy_helper.from_array(np.array(ZP, dtype=np.uint8), "azp"))
    for name, _, K, N, x in linears:
        codes, d = lin[name]
        if kind in ("nb0", "nb4", "nb16"):
            sc = gc.f16_to_f32(d) if kind != "nb16" else np.asarray(d, dtype=np.uint16).view(np.float16)
            inits += [numpy_helper.from_array(gd.pack_ort(codes), f"b_{name}"),
                      numpy_helper.from_array(np.ascontiguousarray(sc.reshape(-1)), f"s_{name}")]
            nodes.append(helper.make_node("MatMulNBits", [x, f"b_{name}", f"s_{name}"], [f"y_{name}"],
                                          domain="com.microsoft", K=K, N=N, bits=4, block_size=32,
                                          accuracy_level=4 if kind == "nb4" else 0, name=f"mm_{name}"))
            continue
        w = dense(codes, d)
        if kind.startswith("i8"):
            wq, _ = quant_w(w)
            inits.append(numpy_helper.from_array(wq, f"w_{name}"))
            ins = [x, f"w_{name}"] + (["azp"] if kind == "i8_u8" else [])
            nodes.append(helper.make_node("MatMulInteger", ins, [f"y_{name}"], name=f"mm_{name}"))
        else:
            inits.append(numpy_helper.from_array(w.astype(np.float16 if kind == "fp16" else np.float32), f"w_{name}"))
            nodes.append(helper.make_node("MatMul", [x, f"w_{name}"], [f"y_{name}"], name=f"mm_{name}"))
    g = helper.make_graph(nodes, f"gemma3_layer{LAYER}_{kind}",
                          [helper.make_tensor_value_info(x, xt, ["M", ks[x]]) for x in used],
                          [helper.make_tensor_value_info(f"y_{name}", yt, ["M", N]) for name, _, _, N, _ in linears],
                          inits)
    ops = [helper.make_opsetid("", 17)] + ([helper.make_opsetid("com.microsoft", 1)] if kind.startswith("nb") else [])
    return helper.make_model(g, opset_imports=ops, ir_version=9).SerializeToString()


def feed_for(kind: str, xs: dict) -> dict:
    """A session feed from fp32 inputs: rounded to the arm's dtype, or int8 per tensor (u8 = s8 + 128)."""
    out = {}
    for x, a in xs.items():
        if kind.startswith("i8"):
            q, _ = quant_x(a)
            out[x] = (q.astype(np.int16) + ZP).astype(np.uint8) if kind == "i8_u8" else q
        else:
            out[x] = a.astype(np.float16 if kind in ("fp16", "nb16") else np.float32)
    return out


def check_outputs(kind: str, ys: list, xs: dict, lin: dict, linears=LINEARS) -> dict:
    """rel-L2 per linear against the float64 product of the fp32 inputs and the exact W; int8 outputs
    dequantized with their scales first."""
    errs = {}
    for (name, _, K, N, x), y in zip(linears, ys):
        w = dense(*lin[name])
        ref = xs[x].astype(np.float64) @ w.astype(np.float64)
        if kind.startswith("i8"):
            _, sx = quant_x(xs[x])
            _, sw = quant_w(w)
            y = y.astype(np.float64) * (sx * sw)[None, :]
        e = y.astype(np.float64) - ref
        errs[name] = float(np.linalg.norm(e) / np.linalg.norm(ref))
    return errs


# step 2's model check: a wiring check (a miswired graph reads rel-L2 near 1), not an accuracy measurement;
# no rule and no prediction reads it
WIRING_MAX = {"fp32": 1e-4, "nb0": 1e-4, "fp16": 1e-2, "nb16": 1e-2, "nb4": 0.2, "i8_u8": 0.2, "i8_s8": 0.2}
CHECK_M = 64


def placement(kind: str) -> int:
    """A subprocess: one DirectML session with verbose logs and the CPU EP listed, so ORT names every node's
    provider (gemma_decode's B3 method)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 0
    so.log_verbosity_level = 0
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    ort.InferenceSession(str(MODELS / f"{kind}.onnx"), so,
                         providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
    print("PLACEMENT_SESSION_OK", flush=True)
    return 0


def read_placement(kind: str) -> dict:
    p = subprocess.run([sys.executable, str(ROOT / "tools/llm_prefill3c.py"), "placement", kind], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = p.stdout + p.stderr
    placed = {ep: int(n) for ep, n in
              re.findall(r"(?:All nodes placed on|Node\(s\) placed on) \[(\w+)\]\. Number of nodes: (\d+)", text)}
    cpu_nodes, cur = [], None
    for s in text.splitlines():
        m = re.search(r"placed on \[(\w+)\]", s)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"^\s*(\w+) \((.*)\)\s*$", s.split("] ", 1)[-1])
        if m and cur == "CPUExecutionProvider":
            cpu_nodes.append(f"{m.group(1)} ({m.group(2)})")
    status = ("UNPARSED" if not placed or "PLACEMENT_SESSION_OK" not in text
              else "ALL_DML" if set(placed) == {"DmlExecutionProvider"} else "CPU_NODES")
    return {"model": kind, "exit": p.returncode, "status": status, "placed": placed, "cpu_nodes": cpu_nodes[:20],
            "lines": [s.split("] ", 1)[-1].strip() for s in text.splitlines() if "placed on [" in s][:10]}


def models() -> int:
    """Step 2: build the rivals' layer models from the release's blk.16, read DirectML's placement, and run
    each model once at M = 64 as a build check. No timing; enters no rule. An arm whose model cannot be
    placed wholly on DirectML is DROPPED and named."""
    import onnxruntime as ort
    import llm_prefill_bench as s3
    print(f"MODELS (3c step 2), no timing, enters no rule. {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
          flush=True)
    print(f"onnxruntime {ort.__version__}; providers {ort.get_available_providers()}", flush=True)
    say("PROTOCOL_JSON", protocol())
    lin = release_layer()
    MODELS.mkdir(parents=True, exist_ok=True)
    kinds = sorted(set(MODEL_OF.values()))
    for kind in kinds:
        t0 = time.perf_counter()
        blob = layer_model(kind, lin)
        p = MODELS / f"{kind}.onnx"
        p.write_bytes(blob)
        say("MODEL_JSON", {"model": kind, "bytes": p.stat().st_size, "sha256": sha(p),
                           "seconds": round(time.perf_counter() - t0, 1)})
    # DirectML placement: the verbose read, then a strict session (no CPU fallback) must open
    dropped = {}
    for kind in DML_MODELS:
        pl = read_placement(kind)
        try:
            s3.make_session(str(MODELS / f"{kind}.onnx"), "dml", 0)
            pl["strict_session"] = "OPENED"
        except Exception as ex:                                        # noqa: BLE001
            pl["strict_session"] = f"REFUSED: {type(ex).__name__}: {str(ex).splitlines()[0][:160]}"
        say("PLACEMENT_JSON", pl)
        if pl["status"] != "ALL_DML" or pl["strict_session"] != "OPENED":
            for arm, k in MODEL_OF.items():
                if k == kind and arm.startswith("D-"):
                    dropped[arm] = f"{kind}: placement {pl['status']}, strict session {pl['strict_session']}"
    # the negative control: a string op DirectML has no kernel for is refused, not run on the CPU
    from onnx import TensorProto, helper
    g = helper.make_graph([helper.make_node("StringNormalizer", ["s"], ["o"], case_change_action="LOWER")], "neg",
                          [helper.make_tensor_value_info("s", TensorProto.STRING, [2])],
                          [helper.make_tensor_value_info("o", TensorProto.STRING, [2])])
    try:
        s3.make_session(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
                        .SerializeToString(), "dml", 0)
        neg = "FAIL: a StringNormalizer session opened, so the strict session proves nothing"
    except Exception as ex:                                            # noqa: BLE001
        neg = f"PASS: refused ({type(ex).__name__})"
    say("NEGATIVE_CONTROL_JSON", {"result": neg})
    # the wiring check at M = 64 (its own seed, not the sitting's X): every model on its chip, once. Its
    # rel-L2s are a wiring check, own seed, not a 3c result, and are never cited (R6)
    rng = np.random.default_rng(SEED + 1)
    xs = {x: rng.standard_normal((CHECK_M, K), dtype=np.float32) for x, K in INPUTS.items()}
    exact = {}                                                         # the exact int8 product, in float64
    for name, _, K, N, x in LINEARS:
        xq, _ = quant_x(xs[x])
        wq, _ = quant_w(dense(*lin[name]))
        exact[name] = (xq.astype(np.float64) @ wq.astype(np.float64)).astype(np.int32)
    bad = []
    for arm, kind in MODEL_OF.items():
        if arm in dropped:
            continue
        ep = "cpu" if arm.startswith("C-") else "dml"
        sess = s3.make_session(str(MODELS / f"{kind}.onnx"), ep, 8)
        ys = sess.run([f"y_{n}" for n, *_ in LINEARS], feed_for(kind, xs))
        del sess
        errs = check_outputs(kind, ys, xs, lin)
        rec = {"arm": arm, "model": kind, "ep": ep, "M": CHECK_M, "rel_l2": errs,
               "finite": all(bool(np.isfinite(y).all()) for y in ys), "wiring_max": WIRING_MAX[kind]}
        if kind.startswith("i8"):
            rec["int32_exact"] = all(np.array_equal(y, exact[n]) for (n, *_), y in zip(LINEARS, ys))
        rec["ok"] = rec["finite"] and max(errs.values()) < WIRING_MAX[kind]
        say("MODEL_CHECK_JSON", rec)
        bad += [] if rec["ok"] else [arm]
    say("DROPPED_ARMS_JSON", dropped)
    for b in bad:
        print("MODEL_CHECK_PROBLEM", b, flush=True)
    ok = not bad and neg.startswith("PASS")
    print("MODELS", "OK" if ok else "PROBLEM", f"(dropped: {', '.join(dropped) or 'none'})", flush=True)
    return 0 if ok else 2


# ---------------------------------------------------------------- stage 3's builds against the fit

def insts_fit() -> int:
    """Read-only: every stage-3 build's insts.bin against 16 + 2576 * M / (8 m)."""
    rows = []
    for d in sorted(BUILD3.iterdir()):
        m = re.fullmatch(r"(bf16|i8)_M(\d+)_K(\d+)_N(\d+)_m(\d+)k(\d+)n(\d+)cs(\d)", d.name)
        f = d / "insts.bin"
        if not m or not f.exists():
            continue
        dt = m.group(1)
        M, K, N, mm, k, n, cs = map(int, m.groups()[1:])
        size = f.stat().st_size
        fit = INSTS_FIT[0] + INSTS_FIT[1] * M // (8 * mm)
        rows.append({"dt": dt, "M": M, "K": K, "N": N, "m": mm, "k": k, "n": n, "cs": cs, "insts_bytes": size,
                     "fit_bytes": fit, "sha256": sha(f)})
        say("INSTS_ROW", rows[-1])
    exact = sum(r["insts_bytes"] == r["fit_bytes"] for r in rows)
    big = max(rows, key=lambda r: r["insts_bytes"])["insts_bytes"] if rows else 0
    say("INSTS_FIT_JSON", {"builds": len(rows), "exact": exact, "largest": big,
                           "largest_builds": [f"{r['dt']} M{r['M']} K{r['K']} N{r['N']} m{r['m']}" for r in rows
                                              if r["insts_bytes"] == big]})
    for dt, M in (("i8", 8192), ("bf16", 8192), ("i8", 2048), ("bf16", 2048)):
        m = NPU_TILES[dt]["P"][0]
        say("INSTS_PREDICTED_3C", {"dt": dt, "M": M, "tile": "P", "m": m,
                                   "insts_bytes": INSTS_FIT[0] + INSTS_FIT[1] * M // (8 * m)})
    return 0 if rows and exact == len(rows) else 2


# ---------------------------------------------------------------- step 4: the NPU builds (compile only)
#
# The gate's condition on this code (after 413a562): exactly one BUILD_ROW_JSON per build, no retries. A P
# build the verifier refuses gets its F build (v2 §5); N-w4 has no F (U4: dropped and stated). A build that
# fails for any other reason (a crash, OOM, a tool error) STOPS the stage (BUILD_STOP_JSON, exit 3) and goes to
# the gate; it is not retried, and it prints no BUILD_ROW_JSON. A refusal is: the field precheck (stage 3's
# constraints(), SILICON 2.6), a design assert, or an aiecc failure carrying an MLIR op-verification
# diagnostic. Nothing else is.

BUILD = ROOT / "build/llm_prefill3c"
W4_DIR = ROOT / "kernels/w4a8_array"
L1_BYTES, STACK = 65536, 3328
DTYPE_OUT = {"bf16": "f32", "i8": "i32"}
VERIFIER_DIAG = re.compile(r"error: '[\w.]+' op ")


def build_dir(dt: str, M: int, K: int, N: int, tile: str) -> Path:
    m, k, n, cs = NPU_TILES[dt][tile]
    return BUILD / f"{dt}_M{M}_K{K}_N{N}_m{m}k{k}n{n}cs{cs}{'_native_unroll2' if dt == 'w4' else ''}"


def fields(dt: str, M: int, K: int, N: int, m: int, k: int, n: int, cs: int):
    """whole_array's asserts and the three DMA field limits: stage 3's constraints() exactly for bf16 and i8,
    with B at half a byte per weight for w4 (packed int4; then L1 is w4a8's own l1_estimate)."""
    a_b = 2 if dt == "bf16" else 1
    b_b = {"bf16": 2, "i8": 1, "w4": 0.5}[dt]
    l1 = int(2 * m * k * a_b + 2 * k * n * b_b + (1 if cs else 2) * m * n * 4 + STACK)
    bad = []
    if M % (m * 4) or (M // (m * 4)) % 2:
        bad.append("M")
    if K % k or N % (n * 4):
        bad.append("K/N")
    if m % 16 or k % 8 or n % 16:
        bad.append("kernel dims")
    if l1 > L1_BYTES:
        bad.append(f"L1 {l1}")
    if m * 4 * N > 2 ** 20:
        bad.append("C step over 2^20 words")
    if N // (n * 4) > 64:
        bad.append("A repeat over 64")
    if int(k * n * b_b) // 4 > 16383:
        bad.append("B tile over 16383 words")
    return bad, l1, {"c_step_words": m * 4 * N, "a_repeat": N // (n * 4), "b_tile_words": int(k * n * b_b) // 4}


def refusal_kind(ex: BaseException):
    """A verifier refusal's kind, or None: then the build failed for another reason and the stage stops."""
    if isinstance(ex, AssertionError):
        return "design assert"
    msg = str(ex)
    if isinstance(ex, RuntimeError) and msg.startswith("[aiecc] Compilation failed") and VERIFIER_DIAG.search(msg):
        return "aiecc verifier"
    return None


def build_one(dt: str, M: int, K: int, N: int, tile: str, wa, w4) -> dict:
    m, k, n, cs = NPU_TILES[dt][tile]
    bad, l1, fw = fields(dt, M, K, N, m, k, n, cs)
    d = build_dir(dt, M, K, N, tile)
    row = {"dt": dt, "M": M, "K": K, "N": N, "tile": tile, "m": m, "k": k, "n": n, "cs": cs, "l1": l1, "fields": fw,
           "verifier_ok": False, "refusal": None, "insts_bytes": None, "fit_bytes": INSTS_FIT[0] + INSTS_FIT[1] * M // (8 * m),
           "xclbin_sha256": None, "insts_sha256": None, "seconds": None, "dir": d.relative_to(ROOT).as_posix()}
    if dt == "w4":
        row.update(arm="native", mode="unroll2", rev=w4.source_rev())
    if bad:
        row["refusal"] = "precheck: " + ", ".join(bad)
        return row
    d.mkdir(parents=True)
    t0 = time.perf_counter()
    try:
        if dt == "w4":
            spec = w4.whole_array_w4a8.specialize(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=4, arm="native",
                                                  mode="unroll2", rev=w4.source_rev(), c_single_buffer=bool(cs))
        else:
            spec = wa.whole_array.specialize(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=4, dtype_in_str=dt,
                                             dtype_out_str=DTYPE_OUT[dt], c_single_buffer=bool(cs))
        spec.compile(xclbin_path=d / "final.xclbin", inst_path=d / "insts.bin")
    except Exception as ex:                                            # noqa: BLE001
        row["seconds"] = round(time.perf_counter() - t0, 1)
        kind = refusal_kind(ex)
        text = str(ex).strip().splitlines()
        if kind is None:
            row["stop"] = f"{type(ex).__name__}: " + " | ".join(text[:6])[:600]
            return row
        diag = next((s for s in text if VERIFIER_DIAG.search(s)), text[0] if text else "")
        row["refusal"] = f"{kind}: {diag[:300]}"
        return row
    row.update(verifier_ok=True, seconds=round(time.perf_counter() - t0, 1),
               insts_bytes=(d / "insts.bin").stat().st_size, insts_sha256=sha(d / "insts.bin"),
               xclbin_sha256=sha(d / "final.xclbin"))
    return row


def plan_hashes() -> None:
    """Every 3c log after the second plan commit prints these (the gate diffs them)."""
    print(f"PREREG_TEXT_SHA256 {hashlib.sha256(PREREG.encode('utf-8')).hexdigest()}", flush=True)
    print(f"PROTOCOL_JSON_SHA256 {hashlib.sha256(json.dumps(protocol()).encode('utf-8')).hexdigest()}", flush=True)
    print(f"VERDICT_CODE_SHA256 {verdict_code_sha()}", flush=True)


def build() -> int:
    """Step 4, compile only (IRON env, no chip): every NPU xclbin through the aiecc verifier. bf16 and i8 at
    the four NPU shapes and both M on their P tiles, F only where P is refused; N-w4's native unroll2 P tile
    at the same shapes. One BUILD_ROW_JSON per build (Q1 reads them); a non-refusal failure stops."""
    import llm_prefill_bench as s3
    print(f"BUILD (3c step 4), compile only, no chip. {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    plan_hashes()
    if BUILD.exists() and any(BUILD.iterdir()):
        sys.exit(f"{BUILD.relative_to(ROOT).as_posix()} is not empty: one row per build, no retries (a re-build is "
                 "the gate's call)")
    sys.path.insert(0, str(s3.WA_DIR))
    sys.path.insert(0, str(W4_DIR))
    import aie.iron as iron
    import whole_array as wa
    import whole_array_w4a8 as w4
    iron.set_current_device(wa._device_for("npu", 4))
    BUILD.mkdir(parents=True, exist_ok=True)
    rows = []
    for dt in ("bf16", "i8", "w4"):
        for M in MS:
            for K, N in NPU_SHAPES:
                for tile in ("P", "F"):
                    row = build_one(dt, M, K, N, tile, wa, w4)
                    if "stop" in row:
                        say("BUILD_STOP_JSON", row)
                        print("BUILD STOPPED: not a verifier refusal; not retried. It goes to the gate.", flush=True)
                        return 3
                    rows.append(row)
                    say("BUILD_ROW_JSON", row)
                    if row["verifier_ok"] or dt == "w4":
                        break                                          # F only where P is refused; none for w4
    p = [r for r in rows if r["tile"] == "P"]
    say("BUILD_SUMMARY_JSON", {"rows": len(rows), "p_builds": len(p), "p_verified": sum(r["verifier_ok"] for r in p),
                               "refused": [r["dir"] for r in rows if not r["verifier_ok"]],
                               "f_builds": [r["dir"] for r in rows if r["tile"] == "F"],
                               "on_fit": sum(r["insts_bytes"] == r["fit_bytes"] for r in rows),
                               "largest_insts": max((r["insts_bytes"] or 0) for r in rows),
                               "note": "Q1 is scored by the verdict from the BUILD_ROW_JSON rows (A3); this line decides nothing"})
    return 0 if all(r["verifier_ok"] for r in p) else 1


# ---------------------------------------------------------------- step 4: the inputs and references

INPUT_DIR = WORK / "inputs"
SHA_FORM = "sha256 of the C-order little-endian int32 array [M, N], as np.ascontiguousarray(y).tobytes()"


def int32_sha(y: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(y, dtype=np.int32).tobytes()).hexdigest()


def exact_int(a_q: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The exact int8 product as int32, by float64 BLAS: exact while every |sum| < 2^53 (127 x 127 x 10240
    = 1.7e8 here)."""
    return np.rint(a_q.astype(np.float64) @ b.astype(np.float64)).astype(np.int32)


def inputs() -> int:
    """Step 4 (resnet_env17, heavy CPU, no chip): X per input at [8192, K] (seeded; M = 2048 takes the first
    rows), W from blk.16 (exact d.(q - 8)), the int8 copies (X per M on its own rows, W per column), N-w4's
    B = q - 8 as int8 [K, N], the float64 references per linear, the float64 products of the bf16-rounded X
    and W, and the exact int32 products' SHAs per linear and M (C-i8, D-i8, N-i8: X_q . W_q; N-w4: X_q . B).
    Every file into a SHA-256 manifest; nothing is ever replaced."""
    import ml_dtypes
    import gemma_compress as gc
    print(f"INPUTS (3c step 4), no chip. {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    plan_hashes()
    if INPUT_DIR.exists() and any(INPUT_DIR.iterdir()):
        sys.exit(f"{INPUT_DIR.relative_to(ROOT).as_posix()} is not empty; inputs are never replaced")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    bf16 = ml_dtypes.bfloat16
    files, exact = {}, {}

    def save(name: str, a: np.ndarray) -> None:
        p = INPUT_DIR / f"{name}.npy"
        np.save(p, a)
        files[name] = sha(p)
        say("INPUT_FILE_JSON", {"name": name, "dtype": str(a.dtype), "shape": list(a.shape), "bytes": p.stat().st_size,
                                "sha256": files[name]})

    lin = release_layer()
    rng = np.random.default_rng(SEED)
    Mx = max(MS)
    xs = {x: rng.standard_normal((Mx, K), dtype=np.float32) for x, K in INPUTS.items()}
    xq = {}
    for x, a in xs.items():
        save(x, a)
        for M in MS:
            q, s = quant_x(a[:M])
            xq[(x, M)] = q
            save(f"{x}_q_M{M}", q)
            save(f"s{x}_M{M}", np.array([s], dtype=np.float64))
    for name, _, K, N, x in LINEARS:
        t0 = time.perf_counter()
        codes, d = lin[name]
        w = dense(codes, d)
        wq, sw = quant_w(w)
        b4 = np.ascontiguousarray((codes.reshape(N, K).astype(np.int16) - 8).astype(np.int8).T)
        save(f"w_{name}", w)
        save(f"w_{name}_q", wq)
        save(f"sw_{name}", sw)
        save(f"b4_{name}", b4)
        save(f"ref_{name}", xs[x].astype(np.float64) @ w.astype(np.float64))
        save(f"ref_bf16_{name}", xs[x].astype(bf16).astype(np.float64) @ w.astype(bf16).astype(np.float64))
        for M in MS:
            exact[f"i8_{name}_M{M}"] = int32_sha(exact_int(xq[(x, M)], wq))
            exact[f"w4_{name}_M{M}"] = int32_sha(exact_int(xq[(x, M)], b4))
        say("LINEAR_DONE_JSON", {"name": name, "K": K, "N": N, "seconds": round(time.perf_counter() - t0, 1)})
    manifest = {"stage": "3c", "layer": LAYER, "seed": SEED, "rows": Mx, "prefix_rows": min(MS),
                "gguf_sha256": gc.GGUF_PIN["sha256"], "bf16": f"ml_dtypes {ml_dtypes.__version__}, round to nearest even",
                "exact_int32_sha_form": SHA_FORM, "files": files, "exact_int32_sha256": exact}
    mp = INPUT_DIR / "manifest.json"
    mp.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    say("EXACT_INT32_SHA256_JSON", exact)
    print(f"INPUTS_MANIFEST_SHA256 {sha(mp)} ({len(files)} files)", flush=True)
    return 0


# ---------------------------------------------------------------- the rules (verdict)
#
# Where v2's text leaves a choice, the code reads it as READINGS states (R1-R6, with the gate's A1-A4; printed
# in the prereg log). The functions in VERDICT_FUNCS and the constants in verdict_constants() are frozen at the
# second plan commit: VERDICT_CODE_SHA256 (A4) is printed in every later 3c log.

OUTPUT_CHECK = ("fails (VOID) on: a non-finite output; a worst-linear rel-L2 above the arm's wiring bound (A1: "
                "WIRING_MAX by model kind, N-i8 at int8's); N-bf16 over 1e-4 rel-L2 from its own bf16 product; any "
                "int8 arm's (C-i8, D-i8, N-i8) int32 not identical to the exact int8 product; N-w4's int32 not "
                "identical to the exact int8 x (q - 8) product (A2). Otherwise its per-linear rel-L2s are recorded. "
                "The int8 control (Q9) is every int8 window's SHAs, VOID or not.")


def void_bound(arm: str):
    """A1: the arm's wiring bound (R6's, fixed before the wiring run), or None (N-bf16, N-w4: R4's checks)."""
    kind = MODEL_OF.get(arm.split("@")[0]) or {"N-i8": "i8_s8"}.get(arm)
    return WIRING_MAX.get(kind) if kind else None


def state_of(w: dict) -> str:
    """A window's state, with A1 applied here too: an OK window whose worst linear exceeds its arm's wiring
    bound is VOID (the sitting voids it live; the verdict enforces it whatever the log says)."""
    b = void_bound(w["arm"])
    if w["state"] == "OK" and b is not None and w.get("err") is not None and worst(w["err"]) > b:
        return "VOID"
    return w["state"]


def taken(ws: list):
    """U12: a pass's window is its first; a VOID first window may be replaced by one re-run."""
    ws = sorted(ws, key=lambda w: w["position"])
    if not ws:
        return None
    if state_of(ws[0]) == "VOID" and len(ws) > 1 and ws[1].get("rerun"):
        return ws[1]
    return ws[0]


def worst(err) -> float:
    return max(err.values()) if isinstance(err, dict) else float(err)


def arm_record(recs: list, arm: str) -> dict:
    """An arm at one M: valid passes, state (COMPLETE, BROKEN or MISSING), each metric's passes and repeat,
    the values each rule may take (the mean, or both pass values), and its error for membership."""
    passes = {}
    for p in (1, 2):
        w = taken([r for r in recs if r["arm"] == arm and r["pass"] == p])
        passes[p] = w if w and state_of(w) == "OK" else None
    errs = [worst(w["err"]) for w in passes.values() if w and w.get("err") is not None]
    out = {"arm": arm, "valid": sum(v is not None for v in passes.values()), "err": max(errs) if errs else None}
    if out["valid"] < 2:
        out.update(state="MISSING", values={"T": None, "E": None})
        return out
    for m in ("T", "E"):
        a, b = passes[1][m], passes[2][m]
        out[m] = {"p1": a, "p2": b, "gap": abs(a - b) / ((a + b) / 2), "holds": abs(a - b) <= REPEAT_MAX * (a + b) / 2}
    ok = out["T"]["holds"] and out["E"]["holds"]
    out["state"] = "COMPLETE" if ok else "BROKEN"
    out["values"] = {m: [(out[m]["p1"] + out[m]["p2"]) / 2] if ok else [out[m]["p1"], out[m]["p2"]] for m in ("T", "E")}
    return out


def rule(arms: dict, npu: str, metric: str, err_ref=None) -> dict:
    """Does `npu` beat both chips on `metric` (lower is better): rival >= MARGIN x npu for every arm of that
    chip whose rel-L2 <= ACC_TIE x err_ref (the NPU arm's own error, or C-nb4's for N-w4's reading)?
    KEEP needs both chips. U8: every combination of the BROKEN arms' pass values (the NPU arm's included)
    must give one outcome."""
    a = arms[npu]
    if a["state"] == "MISSING":
        return {"outcome": "INCOMPLETE", "why": f"{npu} has {a['valid']} valid pass(es)"}
    ref = a["err"] if err_ref is None else err_ref
    rivals = {k: v for k, v in arms.items() if k[:2] in ("C-", "D-")}
    broken = [k for k, v in rivals.items() if v["state"] == "BROKEN"]
    outcomes, combos = set(), 0
    for nv in a["values"][metric]:
        for combo in itertools.product(*[rivals[k]["values"][metric] for k in broken]):
            combos += 1
            val = dict(zip(broken, combo))
            beats = {}
            for chip in "CD":
                possible = {True}
                for k, v in rivals.items():
                    if k[0] != chip:
                        continue
                    member = None if v["err"] is None or ref is None else v["err"] <= ACC_TIE * ref
                    if member is False:
                        continue                                       # less accurate: not in the set
                    if v["state"] == "MISSING":
                        possible = possible | {False}                  # unknown, may be in the set: may block
                        continue
                    if nv * MARGIN > val.get(k, v["values"][metric][0]):
                        possible = {False}                             # a member the NPU arm does not beat
                        break
                beats[chip] = possible
            outcomes |= {bc and bd for bc in beats["C"] for bd in beats["D"]}
    if len(outcomes) == 1:
        return {"outcome": "KEEP" if outcomes.pop() else "KILL", "combos": combos, "broken": broken}
    return {"outcome": "INCOMPLETE", "why": "the combinations disagree, or a MISSING rival could block",
            "combos": combos, "broken": broken}


def evaluate(recs: list) -> dict:
    """recs: WINDOW_JSON records from both sittings. Per M: the arms, each NPU arm's rules, the role, and
    N-w4's report-only reading."""
    out = {}
    for M in MS:
        rm = [r for r in recs if r["M"] == M]
        arms = {arm: arm_record(rm, arm) for arm in ORDER if any(r["arm"] == arm for r in rm)}
        res = {"arms": arms, "rules": {}}
        rival_errs = [v["err"] for k, v in arms.items() if k[:2] in ("C-", "D-") and v["err"] is not None]
        for npu in NPU_ARMS:
            if npu not in arms:
                res["rules"][npu] = {r: {"outcome": "INCOMPLETE", "why": "no windows"} for r in ("speed", "energy")}
            else:
                res["rules"][npu] = {"speed": rule(arms, npu, "T"), "energy": rule(arms, npu, "E")}
            e = arms[npu]["err"] if npu in arms else None
            res["rules"][npu]["accuracy"] = {
                "outcome": "KEEP" if e is not None and rival_errs and e * ACC_TIE < min(rival_errs) else "KILL",
                "note": "cannot be met by construction: C-fp32 and D-fp32 are in the set"}
        outs = [res["rules"][n][r]["outcome"] for n in NPU_ARMS for r in ("speed", "energy")]
        res["role"] = "KEEP" if "KEEP" in outs else "KILL" if all(o == "KILL" for o in outs) else "INCOMPLETE"
        nb4 = [arms[k]["err"] for k in ("C-nb4@8", "C-nb4@16") if k in arms and arms[k]["err"] is not None]
        if "N-w4" in arms and nb4:
            res["w4"] = {m: rule(arms, "N-w4", m, err_ref=max(nb4)) for m in ("T", "E")}
            for m in res["w4"]:
                res["w4"][m]["reading"] = {"KILL": "FAIL", "KEEP": "OPEN"}.get(res["w4"][m]["outcome"], "INCOMPLETE")
        out[M] = res
    return out


def int8_control(recs: list) -> dict:
    """The int32 SHAs per linear of every int8 window that reached its output check (C-i8@8, C-i8@16, D-i8,
    N-i8), VOID or not, per M: identical if all agree (a window voided for a non-exact int32 breaks it)."""
    res = {}
    for M in MS:
        shas = {f"{r['arm']} p{r['pass']} @{r['position']}": r["i8_sha"] for r in recs
                if r["M"] == M and r.get("i8_sha")}
        res[M] = {"windows": len(shas), "identical": bool(shas) and len({json.dumps(v) for v in shas.values()}) == 1}
    return res


# ---------------------------------------------------------------- the predictions (scored by the verdict)

PREDICTIONS = [
    ("Q1", "every P tile passes the aiecc verifier at every shape and both M; insts.bin equals the fit at every 3c build"),
    ("Q2", "the NPU arm holds its four contexts and the M = 8192 buffers; (low confidence) one XRT buffer serves runs "
           "in two contexts"),
    ("Q3", "N-i8's T: 110-135 ms at M = 2048 and 430-520 ms at M = 8192"),
    ("Q4", "report-only, cross-sitting: C-i8's TOPS in sitting B <= in sitting A; D-fp16 likewise"),
    ("Q5", "speed at M = 2048: KILL for both NPU arms; N-i8 at 0.80-1.10x the CPU's best int8; N-bf16 loses to D-fp16 "
           "if D-fp16 is at least as accurate, else to the remaining rivals"),
    ("Q6", "speed at M = 8192: KILL; N-i8 at 0.9-1.2x the faster of C-i8 and C-nb4 at the better thread count"),
    ("Q7", "energy: KEEP for N-i8 at both M, at >= 1.5x fewer J/token than the best rival"),
    ("Q8", "energy: KEEP for N-bf16 at both M; the binding rival D-fp16 or D-nb16, at 1.2-1.6x"),
    ("Q9", "accuracy: the NPU wins nowhere; N-bf16 2e-3..3e-3, N-i8 1e-2..3e-2, D-fp16 2e-4..6e-4, D-fp32 < 1e-5, "
           "C-fp32 < 1e-6, C-nb0 < 1e-6, C-nb4 3e-3..1e-2, D-nb16 3e-4..6e-4; the int8 control holds at both M"),
    ("Q10", "D-nb16 slower than D-fp16 at both M; C-nb4 at 0.7-1.1x C-i8's speed at the same thread count"),
    ("Q11", "report-only: N-w4 at M = 8192 is 1.1-1.4x faster than N-i8's layer; its speed reading FAIL at 2048, "
            "OPEN at 8192"),
    ("Q12", "every DirectML arm holds the repeat rule at both M"),
]
# How each is scored, where v2's words leave a choice: "C-i8" and "C-nb4" mean the better of the arm's two
# thread counts (U11 made them two arms); "N-bf16 loses to D-fp16" means it does not beat D-fp16 by the
# speed rule's 1.10x; "the best rival" is the best of the NPU arm's at-least-as-accurate set (the rule's);
# a prediction whose inputs are missing is NOT SCORED, not MISS.
Q9_RANGES = {"N-bf16": (2e-3, 3e-3), "N-i8": (1e-2, 3e-2), "D-fp16": (2e-4, 6e-4), "D-fp32": (None, 1e-5),
             "C-fp32@8": (None, 1e-6), "C-fp32@16": (None, 1e-6), "C-nb0@8": (None, 1e-6), "C-nb0@16": (None, 1e-6),
             "C-nb4@8": (3e-3, 1e-2), "C-nb4@16": (3e-3, 1e-2), "D-nb16": (3e-4, 6e-4)}
NS = "NOT SCORED"


def in_range(x, lo, hi) -> bool:
    return x is not None and (lo is None or x >= lo) and (hi is None or x <= hi)


def q1(rows) -> list:
    """A3: Q1 over every 3c build row (BUILD_ROW_JSON: dt bf16 | i8 | w4, M, K, N, tile P | F, m, k, n,
    verifier_ok, insts_bytes): every planned P build (NPU_SHAPES x both M x bf16, i8 and N-w4's w4) passed
    the aiecc verifier, and every build, F and w4 included, has insts.bin equal to the fit. A planned P
    build with no row is unknown."""
    if not rows:
        return [None]
    conds = []
    for dt in ("bf16", "i8", "w4"):
        for M in MS:
            for K, N in NPU_SHAPES:
                p = [r for r in rows if (r["dt"], r["M"], r["K"], r["N"], r["tile"]) == (dt, M, K, N, "P")]
                conds.append(bool(p[-1]["verifier_ok"]) if p else None)
    conds += [r.get("insts_bytes") == INSTS_FIT[0] + INSTS_FIT[1] * r["M"] // (8 * r["m"]) for r in rows]
    return conds


def score(ev: dict, ctl: dict, build=None, load=None) -> dict:
    """build: the build log's BUILD_ROW_JSON rows; load: the load check's LOADCHECK_SUMMARY_JSON (contexts,
    buffers_8192, shared_buffer: booleans). A prediction is MISS if any part is known false, else NOT SCORED
    if any part is unknown, else HIT."""
    s = {}

    def val(M, arm, m):
        a = ev[M]["arms"].get(arm)
        return a["values"][m][0] if a and a["state"] == "COMPLETE" else None

    def best_T(M, names):
        got = [val(M, n, "T") for n in names]
        return None if None in got else min(got)

    def outcome(M, npu, r):
        return ev[M]["rules"][npu][r]["outcome"]

    def verdict_of(conds):
        return "MISS" if any(c is not None and not c for c in conds) else NS if None in conds else "HIT"

    s["Q1"] = verdict_of(q1(build))
    s["Q2"] = verdict_of([load.get("contexts"), load.get("buffers_8192"), load.get("shared_buffer")] if load else [None])
    t2, t8 = val(2048, "N-i8", "T"), val(8192, "N-i8", "T")
    s["Q3"] = verdict_of([None if t2 is None else in_range(t2, 110, 135), None if t8 is None else in_range(t8, 430, 520)])
    ci8 = {M: best_T(M, ("C-i8@8", "C-i8@16")) for M in MS}
    df16 = {M: val(M, "D-fp16", "T") for M in MS}
    s["Q4"] = verdict_of([None if None in ci8.values() else ci8[8192] / 8192 >= ci8[2048] / 2048,
                          None if None in df16.values() else df16[8192] / 8192 >= df16[2048] / 2048])
    ni8 = val(2048, "N-i8", "T")
    bf, dfe, bfe = val(2048, "N-bf16", "T"), ev[2048]["arms"].get("D-fp16", {}).get("err"), \
        ev[2048]["arms"].get("N-bf16", {}).get("err")
    dmember = None if dfe is None or bfe is None else dfe <= ACC_TIE * bfe
    if dmember is None:
        bf_half = None
    elif dmember:
        bf_half = None if None in (df16[2048], bf) else (outcome(2048, "N-bf16", "speed") == "KILL"
                                                          and df16[2048] < MARGIN * bf)
    else:
        bf_half = outcome(2048, "N-bf16", "speed") == "KILL"
    s["Q5"] = verdict_of([outcome(2048, "N-i8", "speed") == "KILL", bf_half,
                          None if None in (ci8[2048], ni8) else in_range(ci8[2048] / ni8, 0.80, 1.10)])
    riv8 = best_T(8192, ("C-i8@8", "C-i8@16", "C-nb4@8", "C-nb4@16"))
    ni8_8 = val(8192, "N-i8", "T")
    s["Q6"] = verdict_of([outcome(8192, "N-i8", "speed") == "KILL",
                          None if None in (riv8, ni8_8) else in_range(riv8 / ni8_8, 0.9, 1.2)])

    def margin(M, npu):
        """(the smallest E among the NPU arm's at-least-as-accurate rivals / its E, that rival)"""
        arms = ev[M]["arms"]
        a = arms.get(npu)
        if not a or a["state"] != "COMPLETE" or a["err"] is None:
            return None, None
        riv = [(min(v["values"]["E"]), k) for k, v in arms.items() if k[:2] in ("C-", "D-") and v["state"] != "MISSING"
               and v["err"] is not None and v["err"] <= ACC_TIE * a["err"]]
        if not riv:
            return None, None
        e, k = min(riv)
        return e / a["values"]["E"][0], k
    m7 = [margin(M, "N-i8")[0] for M in MS]
    s["Q7"] = verdict_of([outcome(M, "N-i8", "energy") == "KEEP" for M in MS]
                         + [None if x is None else x >= 1.5 for x in m7])
    m8 = [margin(M, "N-bf16") for M in MS]
    s["Q8"] = verdict_of([outcome(M, "N-bf16", "energy") == "KEEP" for M in MS]
                         + [None if x is None else in_range(x, 1.2, 1.6) and k in ("D-fp16", "D-nb16") for x, k in m8])
    q9 = [None if ev[M]["arms"][k]["err"] is None else in_range(ev[M]["arms"][k]["err"], *rg)
          for M in MS for k, rg in Q9_RANGES.items() if k in ev[M]["arms"]]
    q9 += [ctl[M]["identical"] if ctl[M]["windows"] else None for M in MS]
    q9 += [outcome(M, n, "accuracy") == "KILL" for M in MS for n in NPU_ARMS]
    s["Q9"] = verdict_of(q9)
    q10 = [None if None in (val(M, "D-nb16", "T"), val(M, "D-fp16", "T")) else val(M, "D-nb16", "T") > val(M, "D-fp16", "T")
           for M in MS]
    q10 += [None if None in (val(M, f"C-i8@{t}", "T"), val(M, f"C-nb4@{t}", "T"))
            else in_range(val(M, f"C-i8@{t}", "T") / val(M, f"C-nb4@{t}", "T"), 0.7, 1.1) for M in MS for t in THREADS]
    s["Q10"] = verdict_of(q10)
    w4 = val(8192, "N-w4", "T")
    rd = [ev[M].get("w4", {}).get("T", {}).get("reading") for M in MS]
    s["Q11"] = verdict_of([None if None in (w4, ni8_8) else in_range(ni8_8 / w4, 1.1, 1.4),
                           None if rd[0] in (None, "INCOMPLETE") else rd[0] == "FAIL",
                           None if rd[1] in (None, "INCOMPLETE") else rd[1] == "OPEN"])
    st = [ev[M]["arms"][k]["state"] for M in MS for k in ("D-fp16", "D-fp32", "D-i8", "D-nb16") if k in ev[M]["arms"]]
    s["Q12"] = "MISS" if "BROKEN" in st else NS if "MISSING" in st or not st else "HIT"
    return s


def parse(paths) -> tuple:
    """WINDOW_JSON records, BUILD_ROW_JSON rows (None if there are none) and the LOADCHECK_SUMMARY_JSON."""
    recs, build, load = [], [], None
    for p in paths:
        for s in Path(p).read_text(encoding="utf-8").splitlines():
            if s.startswith("WINDOW_JSON "):
                recs.append(json.loads(s.split(" ", 1)[1]))
            elif s.startswith("BUILD_ROW_JSON "):
                build.append(json.loads(s.split(" ", 1)[1]))
            elif s.startswith("LOADCHECK_SUMMARY_JSON "):
                load = json.loads(s.split(" ", 1)[1])
    return recs, build or None, load


# A4: the verdict code, frozen at the second plan commit. The gate's list (taken, worst, arm_record, rule,
# evaluate, int8_control, in_range, score; REPEAT_MAX through RERUN_VOID, Q9_RANGES, WIRING_MAX), plus what
# they call or read: void_bound, state_of, q1, parse; MS, ORDER, NPU_ARMS, MODEL_OF, NPU_SHAPES, INSTS_FIT.
VERDICT_FUNCS = ("taken", "worst", "arm_record", "rule", "evaluate", "int8_control", "in_range", "score",
                 "void_bound", "state_of", "q1", "parse")


def verdict_constants() -> dict:
    return {"REPEAT_MAX": REPEAT_MAX, "MARGIN": MARGIN, "ACC_TIE": ACC_TIE, "BF16_ACC_MAX": BF16_ACC_MAX,
            "HF_MAX": HF_MAX, "PAGES_MAX": PAGES_MAX, "RERUN_VOID": RERUN_VOID, "Q9_RANGES": Q9_RANGES,
            "WIRING_MAX": WIRING_MAX, "MS": MS, "ORDER": ORDER, "NPU_ARMS": NPU_ARMS, "MODEL_OF": MODEL_OF,
            "NPU_SHAPES": NPU_SHAPES, "INSTS_FIT": INSTS_FIT}


def verdict_code_sha() -> str:
    """sha256 over the sources of VERDICT_FUNCS (inspect.getsource: universal newlines, so CRLF and LF
    checkouts agree) and the JSON of verdict_constants()."""
    import inspect
    g = globals()
    src = "".join(inspect.getsource(g[f]) for f in VERDICT_FUNCS)
    return hashlib.sha256((src + json.dumps(verdict_constants(), sort_keys=True)).encode("utf-8")).hexdigest()


def fmt(x, spec=".2f"):
    return "-" if x is None else format(x, spec)


def report(ev: dict, ctl: dict, sc: dict) -> None:
    for M in MS:
        res = ev[M]
        print(f"\nM = {M} (layer work {layer_flops(M) / 1e9:.1f} GFLOP, DERIVED)")
        print(f"  {'arm':10s} {'state':8s} {'T ms p1 / p2':>19s} {'T':>8s} {'TOPS':>6s} {'E mJ/tok p1 / p2':>19s} "
              f"{'E':>7s} {'rel-L2':>9s}")
        for arm, a in res["arms"].items():
            t, e = a.get("T", {}), a.get("E", {})
            tv = a["values"]["T"][0] if a["state"] == "COMPLETE" else None
            ev_ = a["values"]["E"][0] if a["state"] == "COMPLETE" else None
            print(f"  {arm:10s} {a['state']:8s} {fmt(t.get('p1')):>9s} / {fmt(t.get('p2')):<7s} {fmt(tv):>8s} "
                  f"{fmt(layer_flops(M) / (tv * 1e9) if tv else None):>6s} "
                  f"{fmt(None if e.get('p1') is None else 1e3 * e['p1'], '.4f'):>9s} / "
                  f"{fmt(None if e.get('p2') is None else 1e3 * e['p2'], '.4f'):<7s} "
                  f"{fmt(None if ev_ is None else 1e3 * ev_, '.4f'):>7s} {fmt(a['err'], '.2e'):>9s}")
        for npu, rr in res["rules"].items():
            print(f"  {npu}: speed {rr['speed']['outcome']}, energy {rr['energy']['outcome']}, accuracy "
                  f"{rr['accuracy']['outcome']} ({rr['accuracy']['note']})")
        print(f"  THE ROLE AT M = {M}: {res['role']}")
        if "w4" in res:
            print(f"  N-w4 (report-only; it bounds route (i) only): speed {res['w4']['T']['reading']}, "
                  f"energy {res['w4']['E']['reading']}")
        print("  int8 control: " + ("no int8 windows" if not ctl[M]["windows"] else
                                    f"{'identical' if ctl[M]['identical'] else 'NOT identical'} over "
                                    f"{ctl[M]['windows']} int8 windows (VOID or not)"))
    print("\nPredictions (they decide nothing):")
    for q, text in PREDICTIONS:
        print(f"  {q} {sc[q]}: {text}")


LABELS = ("Energy labels, (b)'s: above idle, with idle charged to no one; package counters only (the user's rule), "
          "so completeness is never shown; DRAM is outside the package, which pulls ratios toward 1; E is DERIVED "
          "from measured power and rate. (b) showed the 780M and the NPU inside the package counter (U9).")


def verdict(paths) -> int:
    print(f"VERDICT_CODE_SHA256 {verdict_code_sha()}", flush=True)
    recs, build, load = parse(paths)
    ev = evaluate(recs)
    ctl = int8_control(recs)
    sc = score(ev, ctl, build, load)
    report(ev, ctl, sc)
    print("\n" + LABELS)
    say("VERDICT_JSON", {str(M): {"role": ev[M]["role"],
                                  "rules": {n: {k: v["outcome"] for k, v in r.items()} for n, r in ev[M]["rules"].items()}}
                         for M in MS})
    say("PREDICTIONS_SCORED_JSON", sc)
    return 0 if all(ev[M]["role"] != "INCOMPLETE" for M in MS) else 2


# ---------------------------------------------------------------- the plan (pre-registration)

PREREG = """\
STAGE 3c PLAN AND PREREG: prefill weight GEMMs at Gemma 3 4B's shapes, M = 2048 and 8192, on speed,
accuracy and energy per prompt token

This is plan v2 as the user approved it on 2026-09-24 (the user's words, relayed by the gate: "Go with
U1-U12"). Its text is v2's, with two edits agreed in review: the scoped route (i) text in §3, and
"(accuracy cannot, §6)" in §6. The §9 heading records the user's decisions.
- v2 took the gate's review of v1: five blocking items (B1-B5) and fourteen fixes (W1-W14), all
  applied. Changes from v1 are marked [v2].
- This text is committed before any 3c NPU build or sitting; Q1 is locked here. The rules are
  tools/llm_prefill3c.py's verdict code.

Order [v2, B1 and W14]:
1. The user's decisions (§9).
2. The ONNX models for the CPU and DirectML arms, and the DirectML placement read at build time.
   - Each DirectML arm's session is opened once on the 780M, and its placement is logged. There is
     no timing. It is a light GPU load, done under a START REQUEST.
   - An arm that cannot be placed wholly on DirectML is dropped and named in the plan commit.
   - No prediction reads this step.
3. **The plan commit.** The tool with the rules and every prediction (Q1-Q12), the dropped-arm list,
   and results/llm/llm_prefill3c_insts_fit_desktop2_<date>.log.
   - That log holds stage 3's 28 builds as (dtype, M, K, N, m, k, n, cs, insts.bin bytes). They are
     read-only from build/llm_prefill, with no chip.
   - No NPU build for 3c exists yet, so Q1 is scored against builds made after it.
4. The NPU compile-only builds and field-width check (§5), then the inputs and references (§2).
   Both are heavy CPU loads, under a START REQUEST.
5. **The pins-only commit.** The xclbin, model and manifest SHAs, and the build log. It changes no
   rule and no prediction: the PROTOCOL_JSON and the rule text are byte-identical to step 3's, and
   the gate can diff them.
6. A load check (pre-sitting, enters no rule), under a START REQUEST. Its log is committed before
   the first sitting.
7. Sitting A (M = 2048) and sitting B (M = 8192), each under its own START REQUEST, each committed
   as measured, then FINISHED (U7).
8. The verdict, the report, and the write-up for the gate.

## 0. Why 3c, and what it answers from the brief

- From M = 512 to 2048 in 3b (MEASURED; Llama-2-7B shapes):
  - NPU int8 rose from 3.28 to 3.90 TOPS;
  - CPU int8 fell from 4.17 to 3.82 TOPS;
  - DirectML fp16 fell from 3.32 to 2.86 TFLOPS.
  [v2, W8] The NPU's speed over the CPU's went 0.79 -> 1.02. Two points are not a trend; they are
  the reason to test M = 8192.
- Prefill energy was never measured. Stages 3 and 3b predate locked decision 10's energy term, and
  they are not re-scored.
- (b) measured the package's rise above idle for a looped NPU bf16 GEMM (3b's S1-P tile at
  2048 x 4096 x 4096): 21.12 / 21.25 W. One Python thread spinning in cache read 25.22 / 25.11 W
  (MEASURED).
  - So a GEMM that ties on time could still win on joules. That is the hypothesis 3c tests.
  - The "+5.6 W for the NPU's compute" is (b)'s residual statistic, labelled POST HOC there. No 3c
    rule reads it.
- Every 3b lesson is pre-registered:
  - DirectML warms up by time (§4);
  - the CPU is pinned and read back (§3);
  - the NPU's slices are consumed where they lie, with every host cost inside the loop (§3, U3);
  - context switches are paid inside the loop (§4).
- 3c is a new experiment. It does not re-score 3 or 3b, and no 3 or 3b row enters it.

## 1. The question (locked decision 10)

At Gemma 3 4B's weight-GEMM shapes, for M = 2048 and 8192 prompt tokens, does an NPU arm earn a
prefill role?
- It earns one by being faster, more accurate, or using less energy per prompt token than both the
  CPU (ONNX Runtime) and DirectML on the 780M.
- Each is judged against every arm of that chip that is at least as accurate.
- Freeing the GPU is (e)'s question and is not measured here.

## 2. Workload and inputs

**Shapes.** One layer's seven linears as the pinned release stores them: all Q4_0, in all 34
layers. The GGUF is google/gemma-3-4b-it-qat-q4_0-gguf @ 15f73f5e, sha256 76aed0a8… (MEASURED in
(a)). Shapes are K x N, y = x·W:

| Linear | K x N | Input |
|---|---|---|
| q | 2560 x 2048 | X_attn |
| k | 2560 x 1024 | X_attn |
| v | 2560 x 1024 | X_attn |
| o | 2048 x 2560 | X_o |
| gate | 2560 x 10240 | X_ffn |
| up | 2560 x 10240 | X_ffn |
| down | 10240 x 2560 | X_down |

- That is 94,371,840 weights per layer.
- Layer work = 2·M·94,371,840: 386.5 GFLOP at M = 2048 and 1,546.2 GFLOP at 8192 (DERIVED).
- The model's 34 layers are x34 (DERIVED).
- [v2, B5] There are four inputs, as in a real layer:
  - q, k and v share one normed input;
  - gate and up share another;
  - o and down each take their own.

**Weight GEMMs only.** Attention, the norms, RoPE, GELU, the head and the activations between GEMMs
are outside every bracket, for every arm.
- F4 is why attention stays out: through ORT's DirectML EP, Gemma 3 past 1024 positions is a
  different model, because DirectML's GroupQueryAttention reads no local_window_size.
- The weight GEMMs do not depend on position, so M = 8192 rows are the same computation on every
  chip. That does not make an 8192-token DirectML prefill a Gemma 3 prefill. 3c says nothing about
  one.

**W is real (U2).**
- W is the release's blk.16 (a middle layer), dequantized exactly: d·(q − 8) is exact in fp32.
- The q4_0 arms carry no weight error against the reference. Every other arm rounds W to its own
  dtype.
- The functions exist: gemma_decode.gguf_linear, then dequant, transposed to [K, N].

**X is random, as in 3 and 3b.**
- X ~ N(0, 1) fp32 at [8192, K], one array per input, seeded. M = 2048 uses the first 2048 rows.
- The limit, stated: real hidden states have outlier channels that per-tensor int8 activations
  handle worse. 3c's int8 errors are this input's, not a real prompt's.

**Quantization.**
- int8 arms quantize X symmetric per tensor and W symmetric per output column, as 3/3b did. The
  conversions are outside every bracket.
- [v2, W12] Each M quantizes its own rows. M = 2048's int8 X takes its per-tensor scale from its
  own 2048 rows, not from the 8192, as a 2048-token prefill would. So M = 2048's int8 X is not a
  prefix of M = 8192's, and the int8 control is per M.

**The manifest** (new; stage 3's generator stops at M_MAX = 2048). Under a SHA-256 manifest pinned
in step 5:
- X, W, and the int8 copies per M;
- the float64 references y = X·W per linear;
- [v2, W4] the float64 products of the bf16-rounded X and W, per linear, for N-bf16's own check.

Sizes and reuse (DERIVED):
- Each reference set is about 1.95 GB at M = 8192: q 134, k 67, v 67, o 168, gate 671, up 671 and
  down 168 MB.
- Rounding to bf16 is elementwise, so M = 2048's references and bf16 products are the first 2048
  rows of M = 8192's.
- X at M = 8192 is 570 MB fp32.

**Accuracy.**
- rel-L2 against the float64 reference. An arm's error is its worst linear at that M.
- Two arms tie if one's rel-L2 is within 1.10x of the other's (3's ACC_TIE).
- N-bf16 must also be within 1e-4 of its own bf16 product (3's check).
- The int8 control: the int32 outputs of C-i8, D-i8 and N-i8 are identical, per linear and M, by
  SHA. The NPU's split-K down sums its int32 partials exactly, so the control holds for it too.

## 3. Arms

Every arm has its weights resident, and its input and output in host memory each call, as 3 and
3b.

**[v2, B5] The rivals run one session per arm, holding the seven MatMuls.**
- The graph's node order is q, k, v, o, gate, up, down. It has the four inputs, and all seven
  outputs return to the host each call. ORT's sequential executor runs the nodes in that order
  (INFERRED).
- This is how ONNX Runtime runs a real layer, and it is the rival's most favourable form:
  - it has one thread pool, not seven pools of 8 or 16 pinned threads on 16 logical CPUs, where
    pool q still spins while pool k starts;
  - on DirectML it has one fence round trip per layer, not seven.
- The rivals' per-linear times are dropped. No rule reads them. The NPU logs its own per-piece
  times.

**CPU (ONNX Runtime 1.23.3, resnet_env17)**, pinned as in 3b: 8 threads one per physical core, or
16 one per logical CPU, with the caller pinned too.
- Affinities are read back after the warm-up, and a mismatch voids the window.
- Intra-op spinning is at ORT's default, as in 3b. In a back-to-back loop the pool's idle gaps are
  short, so its energy effect is INFERRED small; that is a stated limit.
- The arms:
  - C-fp32: MatMul.
  - C-i8: MatMulInteger, u8 x s8, zero point 128.
  - C-nb0: MatMulNBits on the release's own codes (3-input, block 32, fp32 scales,
    accuracy_level 0; fp32 compute). Built as (c) built C0: the codes injected, no requant.
  - C-nb4: the same, with accuracy_level 4 (int8 compute). ORT quantizes the activations per
    32-block internally, and that error is its own.
- Each runs at 8 and at 16 threads as two arms of the CPU (U11), e.g. C-i8@8 and C-i8@16. Every
  rule's "every arm of that chip" then covers both counts: speed is judged against the faster, and
  energy against the one with fewer joules.
  - This is the gate's rule from (e) v3: the rival gets its most favourable configuration for the
    axis being scored.
  - In 3b, 8 and 16 threads were within 0.2-6.5% on int8 time. The power each draws was never
    measured.
  - Fewer than 8 threads is not tried, and that is a stated limit.

**DirectML (the 780M, ORT 1.23.3 DirectML build).** disable_cpu_ep_fallback = 1. Placement is read
at build time (Order, step 2) and logged again in every window.
- D-fp16: MatMul.
- D-fp32: MatMul.
- D-i8: MatMulInteger, s8 x s8.
- D-nb16: MatMulNBits on the release's codes, with fp16 activations and fp16 scales.
- **Sessions: one fresh session per arm and pass, so each pass value is one draw of the session
  level.** This is the noise study's second option (BENCHMARKS, "Recommendation for later
  pre-registrations": "keep one session per row ... and say its level is one draw from a spread of
  this size").
  - Its first option, the median over several fresh sessions, does not fit a continuous 60 s
    energy loop.
  - It did not save 3b either: 3b's broken row moved between passes, not between its 5 sessions.
  - A between-pass level shift is what U8 absorbs, when it cannot change the outcome.

**NPU (mlir-aie whole_array; raw pyxrt; 3's NpuGemm).** The best tiles on record (P); F only where P
is refused at compile.
- N-bf16: bf16 -> f32, P = 32/64/128 cs1.
- N-i8: int8 -> i32, P = 64/128/64 cs1.
- The layout (U3):
  - q, k, v and o run unsliced.
  - gate and up run as four 2560-wide column slices each.
  - down runs split-K as four 2560-row pieces.
  - All twelve pieces are one shape, 2560 x 2560, and one xclbin serves them. The arm holds four
    hardware contexts (q; k and v; o; the 2560 pieces), under the measured cap of 5.
- [v2, W13] The host sums down's four partials (f32 for bf16, int32 for int8) serially, after the
  fourth piece returns. They do not overlap any run, so N-i8's and N-bf16's T are conservative on
  this point.
- [v2, B5] Uploads:
  - The NPU syncs each input once per context that reads it:
    - X_ffn once for all eight gate and up slices (one context);
    - X_o once;
    - each down piece's own quarter of X_down, the same bytes as the rival's one X_down upload.
  - X_attn is read by two contexts (q; k and v). The load check tests whether one XRT buffer can be
    passed to runs in two contexts.
    - If it can, X_attn is synced once, as the rival uploads it once.
    - If it cannot, it is synced twice, and that is stated as an NPU handicap: 10.5 / 42 MB bf16 at
      M = 2048 / 8192 (DERIVED).

**Report-only, not an arm (U4): N-w4, a speed and energy bound for a q4_0 NPU kernel.**
- kernels/w4a8_array's native int8 x int4 whole-array design at its best tile, 64/128/64 cs1
  native unroll2 (6,206.73 GOPS at 2048³, 1.287x upstream int8; MEASURED under IRON's inner
  bracket, not comparable). It is re-timed here in 3's sync bracket, on the same 2560-piece
  layout.
- B = the release's codes (q − 8), repacked outside the bracket. X is int8 per tensor.
- It applies no scales, so its output is not y, and it has no accuracy.
- The reading, report-only: at an M where N-w4 fails the speed (or energy) rule, taken against the
  rivals at least as accurate as C-nb4, a q4_0 NPU kernel cannot pass it either.
- [v2, W5] That reading holds only for a q4_0 kernel with int8 activations, no more accurate than
  C-nb4 (route (i) below).
  - A kernel with bf16 activations (route (ii)) is not bounded by N-w4.
  - N-bf16 is that route's proxy (INFERRED): the same bf16 MACs and bf16-rounded weights, plus an
    expansion step, with fewer weight bytes.

**Not an arm: an NPU arm on the release's q4_0 codes (U6).** [v2, the gate's request: the scope for
the user.] The whole-array designs sum all of K in int32, and q4_0 has one fp16 scale per 32
weights along K. So no existing kernel can consume the release. Folding the scales per column is
lossy and is not the release. There are two routes.

- **(i) W4A8 with a per-block epilogue.**
  - The design change:
    - the core loops K in k = 32 blocks, one q4_0 block each;
    - it MACs int8 x int4 into an int32 partial for the m x n tile;
    - then an f32 epilogue: convert, times the n per-column scales of that block, times the
      activation scale, and add to an f32 C tile.
  - It also needs:
    - a scale stream beside B (n fp16 per k-block per tile: K·N/16 bytes in all);
    - a fourth ObjectFifo in the IRON design;
    - a host repack: GGUF's K-blocked (j, j + 16) nibbles to the design's pairs along N, with the
      −8 folded into the codes.
  - AIE-ML has no native fp32 vector multiply (SPEC: aie2 config FP32_SUPPORT 0; the API emulates
    it in 5-13 bf16 multiply-unit ops per 16 lanes, accuracy_low to the default safe). Through that
    path the epilogue adds 5-13 ops per 512-MAC instruction, about 3-7x N-i8's core instructions
    per block (INFERRED, one multiply-unit issue per cycle). Operand-aware epilogues would cost
    less, and are not examined:
    - the partial fits int16, and d splits exactly into two bf16 parts: 4 bf16 MACs, about 2.5x;
    - an int16 scale ratio into acc64: 1 op, about 1x, lossy at about 2^-15 of a column's largest
      block scale.
    Route (ii) runs at the bf16 rate, 128-MAC mmul against int8's 256: about 2x N-i8 plus the
    expansion. Neither route is ranked. A compile-only bundle count of each epilogue form, in the
    isa_gate pattern, would rank them in hours, before any build.
  - Its bound is N-w4.
- **(ii) W4A16, the AWQ path AMD's closed Phoenix kernels took.**
  - The core expands q4_0 to bf16: int4 -> int8 unpack is free (MEASURED), then int8 -> bf16 and the
    block scale, amortized over the m rows that reuse each weight.
  - Then it MACs at the bf16 rate.
  - N-bf16 is its proxy on speed and accuracy class.
- **Rough effort** (a rough estimate, not derived): several days to about two weeks for either.
  That covers the core kernel, the IRON design, the host repack and the checks below. Program
  memory is 16,384 B per core.
- **What it would be tested against:**
  1. an ISA and compile gate in the int4 study's isa_gate pattern: the fit in program memory, and
     the bundle count per block-tile, which the repo takes as cycles;
  2. bit-exactness against a numpy model of the same arithmetic in the same order;
  3. rel-L2 against 3c's float64 references (the manifest);
  4. 3c's rules in a 3c window, against C-nb4 (the same arithmetic class) and every more accurate
     rival, on speed and energy;
  5. if it ever runs a whole model, (c)'s KL harness.
- **The evidence 3c gives the user for U6:** N-w4 bounds route (i), and N-bf16 stands in for route
  (ii).

## 4. The sitting (one per M, U7)

**Start.**
- The host-load gate: CLEAR, refuse on PEER.
- xrt-smi: no hardware contexts. The 780M is the only DirectML adapter.
- Every pin (manifest, models, xclbins, tool) matches.
- memory_start: no process at 4 GB or more.
- Everything runs under silicon_probe_record.py --device.

**A window,** per arm and pass (the (b) method, with (c)'s per-token form and (e)'s mechanics):
1. Idle: 60 s with nothing launched, counters running.
2. The reader launches. It opens its session or contexts, builds its buffers, and runs the layer
   once as the output check.
   - The check: rel-L2 per linear against the reference; for N-bf16 also its own bf16 product; for
     the int8 arms the SHA.
   - Each linear's references are loaded, checked and released before READY.
   - The loop's working set is X, W and the outputs in the arm's dtype. For an fp32 arm at M =
     8192 that is about 1.9 GB (X 570 MB, outputs 973 MB, W 377 MB; DERIVED) [v2, W9], plus the
     runtime's own copies.
   - [v2, W4] Loading and checking takes about 15-35 s at M = 8192. The largest step is N-bf16's
     two reference sets, about 3.9 GB read; the float64 products themselves are precomputed. The
     window's length does not depend on it.
   - Then it prints READY.
3. Counters start, then the common go.
4. The reader loops its layer from the go. The window is [go + 20 s, go + 80 s].
   - The warm-up is 20 s by time and runs straight into the window, with no idle gap between them.
     (3b: the short DirectML rows were still ramping after 3 counted warm-ups.)
5. After the window the reader stops, and a 10 s settle follows.

Pass 2 is pass 1 reversed. The pass 1 order is:
- C-fp32@8, C-fp32@16, C-i8@8, C-i8@16, C-nb0@8, C-nb0@16, C-nb4@8, C-nb4@16;
- D-fp16, D-fp32, D-i8, D-nb16;
- N-bf16, N-i8, then N-w4.
- [v2, B2 / U12] A VOID window gets at most one re-run, at the end of the same pass. It is logged
  with its position, and a second VOID stands.

**The numbers.**
- **T (layer ms):** the median wall time of the layer iterations that complete inside the window.
  - An iteration is the whole layer, with everything the arm does in it.
  - For a rival that is one session call.
  - For the NPU it is its sixteen dispatches, four context alternations and the split-K sums.
- **E (J per prompt token, per layer, above idle):** (mean package power over the window's rows −
  the idle's mean) / (M x the layers completed in the window / the window's seconds).
  - Completed layers use fractional overlap, as (e)'s W rate did.
  - E x 34 is the model's weight GEMMs (DERIVED).
- The reader logs every layer iteration's start and completion time, and the log keeps them. That
  answers the gate's note after (e).
- Rows: typeperf at 1 Hz with the package, core, CPU, GPU Engine and memory counters. A 1 s trim
  at each end; at least 50 rows.

**Time (DERIVED).**
- A window is about 175 s: 60 s idle, about 25 s load and check, 20 s warm-up, 60 s window and
  10 s settle.
- 15 windows x 2 passes is about 88 min, plus any U12 re-runs (about 3 min each).
- Two sittings make about 3 h in all, plus the load check (about 15 min).

## 5. Compile-only check (step 4; Q1 is scored against it)

Planning arithmetic, already run with stage 3's constraints() (no chip):
- Every P and F tile passes at q, kv, o and the 2560 piece, at M = 2048 and 8192.
- Unsliced gate/up (N = 10240) is refused at every tile by the 2^20-word C step. 4·m·N ≤ 2^20
  gives m ≤ 25.6, exactly as Llama's 11008 was.
- Unsliced down passes. It is split-K anyway, for the layout (U3).
- None of whole_array's field limits scales with M:
  - the 64-iteration A wrap is N/(4n), at most 10 here;
  - the C step is 4·m·N;
  - the B tile is k·n.
  - M enters only the host runtime sequence and the core's loop count. M = 8192 passes the shape
    rules: M/(4m) is even at m = 32 and 64.
- **[v2, B3] The insts.bin fit is now traced.**
  - All 28 stage-3 builds in build/llm_prefill fit insts.bin = 16 + 2576·M/(8m) B exactly. Their
    rows are committed with the plan (Order, step 3).
  - The largest is 41,232 B: the unsliced S2u builds, bf16 and i8 at M = 2048, K = 4096,
    N = 11008, m = 16. Both ran as GEMMs in stage 3.
  - By the fit, 3c's M = 8192 builds are 41,232 B (i8 P, m = 64) and 82,448 B (bf16 P, m = 32).
    The first equals the largest run so far; the second has not run.
- The buffers at M = 8192 are at most 84 MB (C of a 2560 piece, f32). The largest run so far is
  90 MB.

The check, logged as results/llm/llm_prefill3c_build_desktop2_<date>.log, builds every NPU xclbin
through the aiecc verifier:
- 4 shapes x 2 dtypes x 2 M;
- F only where P is refused;
- N-w4's 4 x 2 under U4.

It records per build: the constraints() values, L1 bytes, insts.bin size and the xclbin SHA.

**The load check** (Order, step 6; pre-sitting):
- It opens the four contexts at once, allocates the M = 8192 buffers, and runs each reader to READY
  and one layer.
- [v2, B5] It tests whether one XRT buffer can serve runs in two contexts.
- If the four contexts cannot be held, 3c STOPS and comes back to the user. There is no silent
  redesign.

## 6. Rules, per M, each labelled

**Valid passes [v2, B2].**
- A window is valid if it is neither VOID nor FAILED (after at most one U12 re-run of a VOID).
- An arm with fewer than two valid passes is INCOMPLETE, and U8 cannot decide from its one value.
- A reader that exits non-zero is FAILED. That window is not re-run: a crash is a defect to report.
- [v2, W10] A failed output check voids that window, like every other VOID. A systematic failure
  will fail its re-run too, and then stands.

**Repeat [v2, W2].**
- An arm's T holds if |p1 − p2| ≤ 0.10 x (p1 + p2)/2, and likewise its E. This is 3b's form: its
  broken row read 4.79 / 47.30 = 10.1%.
- An arm that holds on both takes the mean. Otherwise the arm is INCOMPLETE.

**U8, deciding with INCOMPLETE arms [v2, W1].**
- A rule that needs INCOMPLETE arms is evaluated at every combination of their two pass values,
  with the complete arms at their means.
- It is decided only if every combination gives the same outcome. Otherwise the rule is
  INCOMPLETE.
- Membership in "at least as accurate" uses the worse of an arm's two output-check rel-L2s.
- INCOMPLETE stands. There is no third run without the user.

**Speed.**
- An NPU arm beats a chip if T_rival ≥ 1.10 x T_npu for every arm of that chip with rel-L2 ≤ 1.10
  x the NPU arm's.
- Speed KEEP needs both chips beaten; otherwise speed KILL.
- Labels: T MEASURED, ratios DERIVED.

**Energy.**
- An NPU arm beats a chip if E_rival ≥ 1.10 x E_npu for every at-least-as-accurate arm of that
  chip.
- Energy KEEP needs both chips beaten; otherwise energy KILL.
- (b)'s labels go in the verdict verbatim:
  - above idle, with idle charged to no one;
  - package counters only (the user's rule), so completeness is never shown;
  - DRAM is outside the package, which pulls ratios toward 1;
  - E is DERIVED from measured power and rate.
- (b) showed the 780M and the NPU inside the package counter, and that is cited (U9). [v2, B4] v1's
  "UNDECIDED" exit is deleted: it had no pre-registered trigger.

**Accuracy.**
- An NPU arm wins on accuracy if its rel-L2 x 1.10 < every arm of both chips.
- [v2, W6] This cannot be met by construction, and no reader should take the accuracy row as a
  test that could have passed. C-fp32 and D-fp32 are in the set, W's d·(q − 8) is exact in fp32,
  and the NPU's best arm rounds X and W to bf16.

**The role.**
- At each M, an NPU arm earns a prefill role if speed or energy KEEPs (accuracy cannot, §6).
- The verdict names which, and whether the other is KILL or INCOMPLETE.

**VOID (per window).** Any of these voids it:
- under 50 rows;
- pinning not read back (CPU);
- placement not all DirectML;
- a failed output check;
- under 5 completed layer iterations;
- the memory rule, (e)'s (U10): a measured process's own hard faults over 25/s, or Pages Input/sec
  averaging over 1,000 in the window.

**Report-only (no rule):**
- N-w4's reading (§3);
- the NPU's per-piece T and TOPS;
- the witnesses (§7).

## 7. Witnesses (report-only)

- Each process's CPU %; the package, core-sum and residual power.
- The 780M busy per engine, from its per-instance columns.
- The NPU adapter's busy; the idle's SD; the host-load snapshot; xrt-smi before and after every NPU
  arm.
- The CPU arms' affinities; DirectML placement.
- Once per NPU window, outside the loop: the gate/up concatenation done as 3b's layout would do it,
  timed, so 3b's accounting sits beside 3c's.

## 8. Predictions (written before the plan commit; the rules decide, these do not)

The bases are:
- 3b's rates (MEASURED on Llama shapes);
- (b)'s loop powers (MEASURED on other loops);
- this draft's layer work (DERIVED).

Transferring them to Gemma's GEMMs is INFERRED.

- **Q1, compile:** every P tile passes the aiecc verifier at every shape and both M. insts.bin
  equals the fit (16 + 2576·M/(8m)) at every 3c build, as it does at all 28 of stage 3's.
- **Q2, load:**
  - the NPU arm holds its four contexts and the M = 8192 buffers;
  - one XRT buffer can serve runs in two contexts (low confidence; untested here).
- **Q3, N-i8's T:** 110-135 ms at M = 2048 and 430-520 ms at M = 8192.
  - The basis: the layer's work at 3.4-3.9 TOPS (99-114 and 396-455 ms), plus 4 switches (about
    3 ms), plus the serial split-K sums (8-15 and 30-60 ms, §9 U3).
- **Q4, C-i8 (report-only, cross-sitting) [v2, W7]:** its TOPS in sitting B is at most its TOPS in
  sitting A (3b's direction). D-fp16 follows the same direction.
- **Q5, speed at M = 2048: KILL for both NPU arms.**
  - N-i8 does not reach 1.10x the CPU's best int8 (0.80-1.10x).
  - [v2, W3] N-bf16 loses to D-fp16, if D-fp16's measured rel-L2 is at most 1.10x N-bf16's (Q9
    expects it far below). If it is not, Q5's bf16 half is scored against the remaining rivals.
- **Q6, speed at M = 8192: KILL** (moderate confidence). N-i8 against the faster of C-i8 and C-nb4,
  at the better thread count, lands at 0.9-1.2x, with 1.10 inside that range.
- **Q7, energy: KEEP for N-i8 at both M,** at 1.5x or more fewer J/token than the best rival.
  - The basis: the NPU loop at about 18-26 W above idle, against 40-65 W for 16 CPU threads, a
    lower figure at 8, and 30-45 W for the 780M, at similar times.
  - The ~15 W wake step that every busy arm pays ((b), post hoc) pulls this toward 1.
- **Q8, energy: KEEP for N-bf16 at both M** (lower confidence). The binding rival is D-fp16 or
  D-nb16, at a margin of 1.2-1.6x.
- **Q9, accuracy:**
  - The NPU wins nowhere on it, by construction (§6).
  - N-bf16 reads 2e-3 to 3e-3, and N-i8 1e-2 to 3e-2.
  - [v2, W3] D-fp16 reads 2e-4 to 6e-4, D-fp32 under 1e-5 and C-fp32 under 1e-6. 3b read 3.61e-4,
    1.33e-6 and 3.12e-7 against N-bf16's 2.35e-3, on Llama shapes, random W and K up to 11008.
  - C-nb0 reads under 1e-6, C-nb4 3e-3 to 1e-2, and D-nb16 3e-4 to 6e-4.
  - The int8 basis is 3b's 1.54e-2, on random W ~ N(0, 0.02). Per-column int8 over a q4_0 grid,
    with mixed block scales in each column, may read higher.
  - The int8 control holds at both M.
- **Q10, MatMulNBits at M ≥ 2048** (never timed here):
  - D-nb16 is slower than D-fp16 at both M.
  - C-nb4 lands at 0.7-1.1x C-i8's speed at the same thread count.
- **Q11, N-w4 (report-only):** at M = 8192 it reads 1.1-1.4x faster than N-i8's layer. Its speed
  reading is FAIL at M = 2048 and OPEN at 8192.
- **Q12:** every DirectML arm holds the repeat rule with the 20 s warm-up.
  - The risk is named: 3b's one broken row moved between passes, not sessions (44.90 / 49.69 ms,
    10.1%).

## 9. The user's decisions (2026-09-24, relayed by the gate: "Go with U1-U12")

Every decision took the recommendation, listed first below; the gate's position follows each.

- **U1, shapes: the seven linears as shipped** (recommended), or fused QKV and gate|up. Every arm
  would take the same fusion. The release stores them separate, and 3/3b ran unfused. The gate
  agrees.
- **U2, inputs: the release's blk.16 as W, with random X** (recommended).
  - Real hidden states would need a model forward over 8192 tokens of text, and their activation
    outliers would change the int8 errors.
  - The limit is stated either way. The gate agrees.
- **U3, the NPU's gate/up/down layout.**
  - **(a) Recommended:** the slices are consumed where they lie, and down runs split-K with the
    host sums inside the loop, serially.
    - This is the dataflow a real pipeline would use: gate and up feed an elementwise product per
      slice, and down splits along K to meet them.
    - It needs 4 contexts, not 5.
    - Its host cost is not small. The three adds over [M, 2560] move about 190 MB at M = 2048 and
      750 MB at 8192: about 8-15 ms and 30-60 ms (DERIVED).
    - That is the same order as (b)'s concatenation. So the choice is which host op matches the
      dataflow, and 4 contexts against 5, not cheap against realistic.
  - **(b) Alternative:** 3b's accounting, with gate's and up's outputs concatenated on the host
    (about 25 ms each at M = 2048 and about 100 ms at 8192, DERIVED from 26-29 ms per 90 MB) and
    down unsliced. That needs 5 contexts, at the measured cap.
  - (b) is timed beside (a) in every NPU window (§7). The gate agrees with (a), with the serial
    sums stated (W13).
- **U4, the N-w4 bound: include it, report-only** (recommended), with W5's scope: it bounds route
  (i) only.
  - It needs its builds and a raw-pyxrt host path for its packed B.
  - It is dropped, and this is stated, if its compile or load check fails. Nothing is redesigned.
  - The gate agrees.
- **U5, the q4_0 rival arms: C-nb0, C-nb4 and D-nb16** (recommended all three). The brief named
  C-nb4 and D-nb16. C-nb0 is the only CPU q4_0 arm with fp32-class accuracy. The gate agrees.
- **U6, no NPU arm on the release's q4_0 codes in 3c** (recommended).
  - §3 scopes what a kernel would take: two routes, the design change, a rough effort and its
    tests.
  - 3c's N-w4 and N-bf16 are the evidence for whether to build one. The user asked today why there
    is no int4 kernel for the LLM work: this is where that decision sits.
  - The gate agrees.
- **U7, two sittings, one per M, about 90 min each** (recommended), or one sitting of about 3 h.
  Every rule is per M, and energy is compared only within a sitting. The gate agrees.
- **U8, an INCOMPLETE arm still decides a rule when every combination of its pass values gives the
  same outcome** (recommended; W1's form).
  - This pre-registers what 3b could only say post hoc ("the broken row cannot change the
    outcome").
  - The alternative is 3b's plain INCOMPLETE. The gate agrees, with W1.
- **U9, energy controls: cite (b)'s INSIDE result** (recommended), or re-run (b)'s controls in each
  sitting (about 10 min more). (b)'s finding is about the counter's domain, not a sitting's level.
  The gate agrees, with B4.
- **U10, the memory rule: (e)'s** (recommended). 3/3b had none. (c) lost 3 of 10 arm-passes per
  sitting to a page-in rule set at 100. The gate agrees.
- **U11, the CPU's thread counts: 8 and 16 threads, each its own arm** (recommended). This is 8 CPU
  windows a pass.
  - Speed is then judged against the faster count, and energy against the one with fewer joules.
  - Choosing one count by speed and using it for energy too would hand the NPU an energy win
    against a configuration picked for another axis: (e) v3's artifact.
  - The alternative is a speed-selected count for both rules, about 20 min shorter a sitting. The
    gate agrees.
- **U12 [v2, B2]: a VOID window gets at most one re-run,** at the end of the same pass, logged with
  its position (recommended; the gate's recommendation).
  - A second VOID stands. FAILED windows are not re-run.
  - This is not a third run of a valid arm, and (c) lost 3 of 10 arm-passes per sitting to voids.
  - The alternative is no re-run.
- **U13 does not arise [v2, B5].** v2 adopts the gate's one-session form for the rivals outright.
  The only open point is whether the NPU can share X_attn across two contexts; the load check
  decides it, and v2 states the handicap if it cannot.

## 10. Unverified going in

- MatMulNBits at M > 1, on either chip: never timed here. [v2, W11] Nor is D-nb16's implementation
  known: it may dequantize then GEMM, or run natively.
- The 82 KB insts.bin (bf16 P at M = 8192) and 84 MB buffers.
- Four whole_array contexts held by one process. The cap of 5 was measured with a graph-engine
  xclbin.
- Whether one XRT buffer can serve runs in two contexts.
- ORT's sequential executor running the seven nodes in graph order (INFERRED).
- Whether 20 s of warm-up by time steadies DirectML.
- The GEMM loops' power on each chip. Q7 and Q8 are INFERRED from other loops.
- The 0.748 ms switch. It was measured on a non-GEMM kernel; 3c pays it inside the loop instead of
  adding it.
"""


READINGS = """\
READINGS: the code's choices where v2's text leaves one (not part of v2's text). The gate ruled on the first
plan commit (2c29296): R1, R2 and R6 accepted; R3 accepted with A1; R4 accepted as amended by A1 and A2; R5
accepted with A3; A4 freezes the verdict code. This second plan commit carries A1-A4; the plan text above
is unchanged.
- R1 Repeat, arm-level (v2 §6: "An arm that holds on both takes the mean. Otherwise the arm is
  INCOMPLETE."): an arm with two valid passes whose T or E breaks the 10% rule is INCOMPLETE, and U8
  takes both of its pass values of the metric a rule reads.
- R2 An arm with fewer than two valid passes (v2: "U8 cannot decide from its one value") is unknown,
  and U8's own logic applies: a rule is decided only when no value of the unknown arm could change its
  outcome. A rival that may be at least as accurate could only block the NPU arm on its chip, so a rule
  the other arms already KILL stays KILL, and one they would KEEP is INCOMPLETE. An NPU arm with fewer
  than two valid passes leaves its own rules INCOMPLETE.
- R3 Membership uses the worse of an arm's valid windows' rel-L2s, each its worst linear. A worse error
  only removes an arm from the set, so a known error that already removes an unknown arm decides. A
  VOID window's error is not used. A1 closes the gap this left: a finite garbage pass can no longer stay
  valid and drop its arm from the set (R4).
- R4 The output check (v2 §4 lists what it checks, not when it fails) fails, voiding its window, on:
  - a non-finite output;
  - (A1) a worst-linear rel-L2 above the arm's wiring bound, R6's bounds, fixed before the wiring run:
    1e-4 for fp32 and nb0, 1e-2 for fp16 and nb16, 0.2 for nb4 and int8 (C-fp32, C-i8, C-nb0, C-nb4,
    D-fp16, D-fp32, D-i8, D-nb16, and N-i8 at int8's). The bounds are in PROTOCOL_JSON (wiring_max, and
    void_rel_l2_max per arm), and the verdict applies them to every window whatever the log says;
  - N-bf16 over 1e-4 rel-L2 from its own bf16 product (stage 3's check);
  - any int8 arm's (C-i8@8, C-i8@16, D-i8, N-i8) int32 not identical to the exact int8 product (the
    manifest's, computed in float64, exact since every |sum| < 2^53): the same check for every int8
    arm, rival or NPU, which implies v2's SHA identity;
  - (A2) N-w4's int32 not identical to the exact int8 x (q - 8) product, in float64, so a mis-packed B
    cannot bound the wrong computation. N-w4 still has no accuracy: its output is not y.
  Such a VOID takes U12's one re-run like any other. The int8 control (§2, Q9) is every int8 window's
  SHAs, VOID or not; it holds if all agree.
- R5 Scoring the predictions: "C-i8" and "C-nb4" mean the better of the two thread counts (U11 made
  each two arms); "N-bf16 loses to D-fp16" means it does not beat D-fp16 by the speed rule's 1.10x;
  "the best rival" (Q7, Q8) is the best of the NPU arm's at-least-as-accurate set. A prediction is MISS
  if any part is known false, else NOT SCORED if any part is unknown, else HIT.
  (A3) Q1's "every 3c build" is every build the build log records, N-w4's and any F build included, with
  no carve-out: Q1 is scored from the build log's BUILD_ROW_JSON rows (dt bf16 | i8 | w4, M, K, N, tile
  P | F, m, k, n, verifier_ok, insts_bytes). Every planned P build (the four NPU shapes, both M, bf16, i8
  and w4) must pass the verifier, and every row's insts.bin must equal 16 + 2576.M/(8m). Q2 reads the
  load check's LOADCHECK_SUMMARY_JSON booleans contexts, buffers_8192 and shared_buffer.
- R6 Step 2 ran each model once at M = 64 (its own seed, not the sitting's X) as a wiring check (a
  miswired graph reads rel-L2 near 1): bounds 1e-4 (fp32, nb0), 1e-2 (fp16, nb16), 0.2 (int8, nb4),
  and each int8 model against the exact product. Its M = 64 rel-L2s are a wiring check, own seed, not
  a 3c result: they are never cited as Q9's outcome or as any 3c accuracy figure, and no rule reads
  them. They were seen before this plan commit; Q1-Q12 are v2's text unchanged.
- A4 The rules are the verdict code, and it is frozen here. VERDICT_CODE_SHA256 is sha256 over the
  source of taken, worst, arm_record, rule, evaluate, int8_control, in_range and score, plus what they
  call or read (void_bound, state_of, q1, parse), and the JSON of the constants REPEAT_MAX through
  RERUN_VOID, Q9_RANGES and WIRING_MAX (plus MS, ORDER, NPU_ARMS, MODEL_OF, NPU_SHAPES, INSTS_FIT).
  Every later 3c log prints it, and later code-only commits keep those functions and constants
  byte-identical. The window-state code (R4, U10 and the VOID list) is not written yet: when it lands,
  its HEAD goes to the gate before the load check.
"""


def sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def latest(pattern: str):
    got = sorted(RESULTS.glob(pattern))
    return got[-1] if got else None


def prereg() -> int:
    """The plan commit's log: v2's text with the two agreed edits, the readings, PROTOCOL_JSON, the
    predictions, and step 2's and the insts-fit's logs by hash, with the dropped arms. No chip."""
    print(PREREG, flush=True)
    print(READINGS, flush=True)
    say("PROTOCOL_JSON", protocol())
    say("PREDICTIONS_JSON", [{"id": q, "text": t} for q, t in PREDICTIONS])
    print(f"PREREG_TEXT_SHA256 {hashlib.sha256(PREREG.encode('utf-8')).hexdigest()}", flush=True)
    print(f"PROTOCOL_JSON_SHA256 {hashlib.sha256(json.dumps(protocol()).encode('utf-8')).hexdigest()}", flush=True)
    print(f"VERDICT_CODE_SHA256 {verdict_code_sha()}", flush=True)
    ok = True
    for tag, pat in (("MODELS_LOG", "llm_prefill3c_models_*.log"), ("INSTS_FIT_LOG", "llm_prefill3c_insts_fit_*.log")):
        p = latest(pat)
        if p is None:
            print(f"{tag} MISSING: run its stage first", flush=True)
            ok = False
            continue
        crlf = b"\r\n" in p.read_bytes()
        print(f"{tag} {p.relative_to(ROOT).as_posix()}: sha256 {sha(p)} ({'CRLF' if crlf else 'LF'} working copy), "
              f"{sha_lf(p)} (LF blob)", flush=True)
        text = p.read_text(encoding="utf-8")
        if tag == "MODELS_LOG":
            dropped = next((json.loads(s.split(" ", 1)[1]) for s in text.splitlines()
                            if s.startswith("DROPPED_ARMS_JSON ")), None)
            say("DROPPED_ARMS_JSON", dropped)
            for s in text.splitlines():
                if s.startswith("MODEL_JSON "):
                    print("  " + s, flush=True)
            print("  Its MODEL_CHECK_JSON rel-L2s (M = 64): wiring check, own seed, not a 3c result; never cited as "
                  "Q9's outcome or as any 3c accuracy figure (R6).", flush=True)
            ok &= any(s.startswith("MODELS OK") for s in text.splitlines())
        else:
            fit = next((json.loads(s.split(" ", 1)[1]) for s in text.splitlines() if s.startswith("INSTS_FIT_JSON ")),
                       None)
            say("INSTS_FIT_JSON", fit)
            ok &= bool(fit) and fit["builds"] == fit["exact"] == 28
    print("PREREG", "OK" if ok else "INCOMPLETE", flush=True)
    return 0 if ok else 2


# ---------------------------------------------------------------- selftest (no chip, no GPU)

def selftest() -> int:
    import tempfile
    import onnxruntime as ort
    fails = []

    def expect(what, got, want):
        print(f"  {what}: {got}" + ("" if got == want else f"  (expected {want})"))
        if got != want:
            fails.append(what)

    print("Graphs (tiny shapes, the same builder, the CPU EP):")
    rng = np.random.default_rng(7)
    tiny = [("q", "attn_q", 64, 32, "x_attn"), ("k", "attn_k", 64, 32, "x_attn"), ("o", "attn_output", 32, 64, "x_o"),
            ("down", "ffn_down", 96, 64, "x_down")]
    lin = {}
    for name, _, K, N, _ in tiny:
        codes = rng.integers(0, 16, (N, K // 32, 32)).astype(np.uint8)
        d = (rng.standard_normal(N * K // 32) * 0.01).astype(np.float16).view(np.uint16)
        lin[name] = (codes, d)
    xs = {x: rng.standard_normal((5, K), dtype=np.float32) for x, K in (("x_attn", 64), ("x_o", 32), ("x_down", 96))}
    outs = {}
    for kind in ("fp32", "fp16", "nb0", "nb4", "nb16", "i8_u8", "i8_s8"):
        sess = ort.InferenceSession(layer_model(kind, lin, tiny), providers=["CPUExecutionProvider"])
        expect(f"{kind}: outputs in graph order", [o.name for o in sess.get_outputs()], [f"y_{n}" for n, *_ in tiny])
        try:
            ys = sess.run(None, feed_for(kind, xs))
        except Exception as ex:                                        # noqa: BLE001
            print(f"  {kind}: not runnable on this ORT's CPU EP ({type(ex).__name__}); its graph is built")
            continue
        outs[kind] = ys
        e = max(check_outputs(kind, ys, xs, lin, tiny).values())
        bound = {"fp32": 1e-6, "fp16": 5e-3, "nb0": 1e-5, "nb4": 3e-2, "nb16": 5e-3, "i8_u8": 3e-2, "i8_s8": 3e-2}[kind]
        expect(f"{kind}: worst rel-L2 {e:.2e} < {bound:g}", bool(e < bound), True)
    for kind in ("fp32", "nb0", "i8_u8", "i8_s8"):
        expect(f"{kind} ran", kind in outs, True)
    if "i8_u8" in outs and "i8_s8" in outs:
        expect("u8 x s8 with zero point 128 equals s8 x s8",
               all(np.array_equal(a, b) for a, b in zip(outs["i8_u8"], outs["i8_s8"])), True)
        xq, _ = quant_x(xs["x_attn"])
        wq, _ = quant_w(dense(*lin["q"]))
        expect("int32 equals the exact product (float64 of int8)",
               bool(np.array_equal(outs["i8_u8"][0], (xq.astype(np.float64) @ wq.astype(np.float64)).astype(np.int32))),
               True)
    w = dense(*lin["q"])
    import gemma_compress as gc
    expect("W = d.(q - 8) exactly, [K, N]", bool(np.array_equal(
        w[:32, 0], gc.f16_to_f32(lin["q"][1][:1]) * (lin["q"][0][0, 0].astype(np.float32) - 8))), True)

    print("Rules (synthetic records):")

    def rec(M, arm, p, T, E, err, state="OK", pos=None, rerun=False):
        return {"M": M, "arm": arm, "pass": p, "T": T, "E": E, "err": None if err is None else {"q": err},
                "state": state, "position": pos if pos is not None else ORDER.index(arm) + (0 if p == 1 else 100),
                "rerun": rerun}

    base = {"C-fp32@8": (400, 8.0, 3e-7), "C-fp32@16": (420, 9.0, 3e-7), "C-i8@8": (110, 2.6, 1.5e-2),
            "C-i8@16": (100, 2.9, 1.5e-2), "C-nb0@8": (420, 8.0, 3e-7), "C-nb0@16": (430, 9.0, 3e-7),
            "C-nb4@8": (120, 2.7, 6e-3), "C-nb4@16": (110, 3.0, 6e-3), "D-fp16": (130, 2.3, 3.6e-4),
            "D-fp32": (500, 7.0, 1.3e-6), "D-i8": (600, 9.0, 1.5e-2), "D-nb16": (160, 2.8, 4e-4),
            "N-bf16": (150, 1.5, 2.4e-3), "N-i8": (115, 1.1, 1.6e-2), "N-w4": (95, 0.9, None)}

    def recs_for(M, override=None):
        out = []
        for arm, (T, E, err) in base.items():
            o = (override or {}).get(arm)
            for p in (1, 2):
                t, e = o[p - 1] if o else (T, E)
                out.append(rec(M, arm, p, t, e, err))
        return out

    def rule_of(recs, npu="N-i8", r="energy", M=2048):
        return evaluate(recs + recs_for(8192 if M == 2048 else 2048))[M]["rules"][npu][r]["outcome"]

    ev = evaluate(recs_for(2048) + recs_for(8192))
    expect("speed: N-i8 115 x 1.10 > C-i8@16's 100: KILL", ev[2048]["rules"]["N-i8"]["speed"]["outcome"], "KILL")
    expect("energy: N-i8 1.1 x 1.10 < every rival (least D-fp16 2.3): KEEP",
           ev[2048]["rules"]["N-i8"]["energy"]["outcome"], "KEEP")
    expect("energy: N-bf16 1.5 x 1.10 < its set's least (D-fp16 2.3): KEEP",
           ev[2048]["rules"]["N-bf16"]["energy"]["outcome"], "KEEP")
    expect("speed: N-bf16 150 x 1.10 > D-fp16's 130: KILL", ev[2048]["rules"]["N-bf16"]["speed"]["outcome"], "KILL")
    expect("the role at 2048: KEEP (energy)", ev[2048]["role"], "KEEP")
    expect("accuracy: KILL (cannot pass)", ev[2048]["rules"]["N-bf16"]["accuracy"]["outcome"], "KILL")
    expect("N-w4 speed reading: 95 x 1.10 = 104.5 <= every arm as accurate as C-nb4 (C-nb4@16 110 the least): OPEN",
           ev[2048]["w4"]["T"]["reading"], "OPEN")
    expect("... C-i8 (1.5e-2 > 1.1 x 6e-3) is outside that set", ev[2048]["arms"]["C-i8@16"]["err"] > ACC_TIE * 6e-3, True)
    # repeat, arm-level (R1)
    a = arm_record([rec(2048, "D-fp32", 1, 44.90, 1, 1e-6), rec(2048, "D-fp32", 2, 49.69, 1, 1e-6)], "D-fp32")
    expect("repeat: 3b's 44.90 / 49.69 (10.1%) breaks T: the arm is BROKEN", (a["T"]["holds"], a["state"]),
           (False, "BROKEN"))
    a = arm_record([rec(2048, "D-fp32", 1, 45.0, 1, 1e-6), rec(2048, "D-fp32", 2, 49.5, 1, 1e-6)], "D-fp32")
    expect("repeat: 45.0 / 49.5 (9.5%) holds: COMPLETE, the mean", (a["state"], a["values"]["T"]), ("COMPLETE", [47.25]))
    a = arm_record([rec(2048, "D-fp32", 1, 100, 1.0, 1e-6), rec(2048, "D-fp32", 2, 101, 1.3, 1e-6)], "D-fp32")
    expect("R1: T holds but E breaks: BROKEN, and T keeps both pass values", (a["state"], a["values"]["T"]),
           ("BROKEN", [100, 101]))
    # U8
    expect("U8: D-fp16's E broken (2.3 / 2.9), N-i8 beats both values: KEEP",
           rule_of(recs_for(2048, {"D-fp16": [(130, 2.3), (130, 2.9)]})), "KEEP")
    expect("U8: D-fp16's E broken across the line (1.15 / 1.5 against 1.21): INCOMPLETE",
           rule_of(recs_for(2048, {"D-fp16": [(130, 1.15), (130, 1.5)]})), "INCOMPLETE")
    expect("U8: the NPU arm's own E broken (1.0 / 1.3), both under 2.3 / 1.10: KEEP",
           rule_of(recs_for(2048, {"N-i8": [(115, 1.0), (115, 1.3)]})), "KEEP")
    expect("U8: the NPU arm's own E broken (1.0 / 2.2) across 2.3 / 1.10 = 2.09: INCOMPLETE",
           rule_of(recs_for(2048, {"N-i8": [(115, 1.0), (115, 2.2)]})), "INCOMPLETE")
    # R2: an unknown rival
    miss = [r for r in recs_for(2048) if not (r["arm"] == "D-fp16" and r["pass"] == 2)]
    expect("R2: D-fp16 with one valid pass could block an energy KEEP: INCOMPLETE", rule_of(miss), "INCOMPLETE")
    expect("R2: ... a speed KILL the CPU already decides stays KILL", rule_of(miss, r="speed"), "KILL")
    miss = [r for r in recs_for(2048) if not (r["arm"] == "D-i8" and r["pass"] == 2)]
    expect("R2: an unknown rival already outside the set (D-i8 1.5e-2 > 1.1 x 2.4e-3) decides nothing: N-bf16 energy "
           "KEEP", rule_of(miss, "N-bf16"), "KEEP")
    miss = [r for r in recs_for(2048) if not (r["arm"] == "N-i8" and r["pass"] == 1)]
    expect("R2: the NPU arm with one valid pass: INCOMPLETE", rule_of(miss, r="speed"), "INCOMPLETE")
    # U12
    rr = [r for r in recs_for(2048) if not (r["arm"] == "D-fp16" and r["pass"] == 2)]
    rr += [rec(2048, "D-fp16", 2, 999, 99, 0.9, state="VOID", pos=150),
           rec(2048, "D-fp16", 2, 130, 2.3, 3.6e-4, pos=160, rerun=True)]
    ev = evaluate(rr + recs_for(8192))
    expect("U12: VOID then a valid re-run: two valid passes", ev[2048]["arms"]["D-fp16"]["valid"], 2)
    expect("R3: the VOID window's error is not used", ev[2048]["arms"]["D-fp16"]["err"], 3.6e-4)
    rr[-2]["state"] = "FAILED"
    expect("U12: a FAILED window is not replaced", evaluate(rr + recs_for(8192))[2048]["arms"]["D-fp16"]["valid"], 1)
    two = [rec(2048, "D-fp16", 1, 130, 2.3, 1e-4), rec(2048, "D-fp16", 2, 130, 2.3, 9e-3)]
    expect("R3: membership takes the worse of two errors", arm_record(two, "D-fp16")["err"], 9e-3)
    # A1: an OK window above its arm's wiring bound is VOID, and takes U12's re-run
    expect("A1: the bounds by arm", [void_bound(a) for a in ("C-fp32@8", "C-nb4@16", "D-fp16", "D-nb16", "D-i8",
                                                             "N-i8", "N-bf16", "N-w4")],
           [1e-4, 0.2, 1e-2, 1e-2, 0.2, 0.2, None, None])
    garbage = [rec(2048, "D-fp32", 1, 500, 7.0, 1.3e-6), rec(2048, "D-fp32", 2, 480, 6.9, 0.9)]
    a = arm_record(garbage, "D-fp32")
    expect("A1: a finite garbage pass (0.9 > 1e-4) marked OK is VOID: one valid pass, its error unused",
           (a["valid"], a["state"], a["err"]), (1, "MISSING", 1.3e-6))
    a = arm_record(garbage + [rec(2048, "D-fp32", 2, 490, 7.0, 1.4e-6, pos=190, rerun=True)], "D-fp32")
    expect("A1: ... and its U12 re-run is taken", (a["valid"], a["err"]), (2, 1.4e-6))

    print("Q1 over the build rows (A3), and three-valued scoring:")

    def row(dt, M, K, N, tile="P", m=None, ok=True, insts=None):
        m = m or {"bf16": 32, "i8": 64, "w4": 64}[dt]
        return {"dt": dt, "M": M, "K": K, "N": N, "tile": tile, "m": m, "k": 64, "n": 64, "verifier_ok": ok,
                "insts_bytes": INSTS_FIT[0] + INSTS_FIT[1] * M // (8 * m) if insts is None else insts}
    rows = [row(dt, M, K, N) for dt in ("bf16", "i8", "w4") for M in MS for K, N in NPU_SHAPES]
    ev0 = evaluate(recs_for(2048) + recs_for(8192))
    ctl0 = int8_control([])
    expect("Q1: all 24 planned P builds verified and on the fit: HIT", score(ev0, ctl0, rows)["Q1"], "HIT")
    off = rows[:-1] + [dict(rows[-1], insts_bytes=rows[-1]["insts_bytes"] + 16)]
    expect("Q1: one w4 build off the fit: MISS (no carve-out)", score(ev0, ctl0, off)["Q1"], "MISS")
    expect("Q1: an F build off the fit also counts: MISS",
           score(ev0, ctl0, rows + [row("i8", 8192, 2560, 2560, "F", 64, True, 1)])["Q1"], "MISS")
    expect("Q1: a planned P build with no row: NOT SCORED", score(ev0, ctl0, rows[1:])["Q1"], NS)
    expect("Q1: no row for one, another known false: MISS", score(ev0, ctl0, rows[2:] + [row("bf16", 2048, 2560, 2048,
                                                                                         ok=False)])["Q1"], "MISS")
    expect("VERDICT_CODE_SHA256 is stable and 64 hex", (verdict_code_sha() == verdict_code_sha(),
                                                        len(verdict_code_sha())), (True, 64))

    print("Predictions and the verdict end to end (synthetic):")
    recs = recs_for(2048) + recs_for(8192)
    for r in recs:
        if r["arm"] in ("C-i8@8", "C-i8@16", "D-i8", "N-i8"):
            r["i8_sha"] = ["ab"] * 7
    ev = evaluate(recs)
    sc = score(ev, int8_control(recs))
    expect("Q1 and Q2 without build or load logs: NOT SCORED", (sc["Q1"], sc["Q2"]), (NS, NS))
    expect("Q3: N-i8 at 115 ms in both sittings: MISS (8192 wants 430-520)", sc["Q3"], "MISS")
    expect("Q12: every DirectML arm complete: HIT", sc["Q12"], "HIT")
    expect("int8 control identical", int8_control(recs)[2048]["identical"], True)
    next(r for r in recs if r["M"] == 2048 and r["arm"] == "D-i8" and r["pass"] == 2)["i8_sha"] = ["cd"] * 7
    expect("int8 control: one differing window breaks it", int8_control(recs)[2048]["identical"], False)
    bad = recs_for(2048, {"D-i8": [(600, 9.0), (700, 9.0)]}) + recs_for(8192)
    expect("Q12: a broken DirectML arm: MISS", score(evaluate(bad), int8_control(bad))["Q12"], "MISS")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.log"
        p.write_text("\n".join("WINDOW_JSON " + json.dumps(r) for r in recs_for(2048) + recs_for(8192)) + "\n",
                     encoding="utf-8")
        expect("the verdict over a synthetic log: decided at both M (exit 0)", verdict([p]), 0)

    print("Step 4 (builds and inputs, no compile here):")
    import llm_prefill_bench as s3
    same = all(s3.constraints(dict(dt=dt, M=M, K=K, N=N, m=t[0], k=t[1], n=t[2], cs=t[3]))
               == tuple(fields(dt, M, K, N, *t)[:2]) for dt in ("bf16", "i8") for tl, t in NPU_TILES[dt].items()
               for M in MS for K, N in NPU_SHAPES)
    expect("fields() equals stage 3's constraints() for every bf16 and i8 tile, shape and M", same, True)
    ok24 = [(dt, M, K, N) for dt in ("bf16", "i8", "w4") for M in MS for K, N in NPU_SHAPES
            if not fields(dt, M, K, N, *NPU_TILES[dt]["P"])[0]]
    expect("all 24 planned P builds pass the field precheck", len(ok24), 24)
    expect("w4's L1 equals w4a8's l1_estimate (64/128/64 cs1, native): 44,288 B",
           fields("w4", 8192, 2560, 2560, *NPU_TILES["w4"]["P"])[1], 44288)
    expect("unsliced gate/up is refused by the C step (as planned)",
           "C step over 2^20 words" in fields("i8", 2048, 2560, 10240, *NPU_TILES["i8"]["P"])[0], True)
    expect("refusal kinds: a design assert; an aiecc op-verification diagnostic; a tool error is not a refusal",
           [refusal_kind(AssertionError("A must be tileable")),
            refusal_kind(RuntimeError("[aiecc] Compilation failed with exit code 1:\nloc(\"x\"): error: 'aie.dma_bd' op "
                                      "Cannot give more than 3 dimensions")),
            refusal_kind(RuntimeError("[aiecc] Compilation failed with exit code 3221225477:\n")),
            refusal_kind(MemoryError()), refusal_kind(FileNotFoundError("xclbinutil"))],
           ["design assert", "aiecc verifier", None, None, None])
    expect("build dirs", [build_dir("bf16", 8192, 2560, 2560, "P").name, build_dir("w4", 2048, 2560, 1024, "P").name],
           ["bf16_M8192_K2560_N2560_m32k64n128cs1", "w4_M2048_K2560_N1024_m64k128n64cs1_native_unroll2"])
    aq = rng.integers(-127, 128, (8, 96)).astype(np.int8)
    bq = rng.integers(-8, 8, (96, 16)).astype(np.int8)
    expect("exact_int equals int64 numpy", bool(np.array_equal(exact_int(aq, bq),
                                                                aq.astype(np.int64) @ bq.astype(np.int64))), True)
    codes = lin["q"][0]
    b4 = np.ascontiguousarray((codes.reshape(32, 64).astype(np.int16) - 8).astype(np.int8).T)
    expect("N-w4's B = (q - 8) as [K, N]: B * d reproduces W", bool(np.array_equal(
        b4[:32, 0].astype(np.float32) * gc.f16_to_f32(lin["q"][1][:1]), dense(*lin["q"])[:32, 0])), True)
    expect("the plan's hashes are the second plan commit's (PREREG, PROTOCOL_JSON, VERDICT_CODE)",
           (hashlib.sha256(PREREG.encode("utf-8")).hexdigest()[:8],
            hashlib.sha256(json.dumps(protocol()).encode("utf-8")).hexdigest()[:8], verdict_code_sha()[:8]),
           ("f7fa696b", "a997580c", "6e459e56"))

    print("The plan text:")
    expect("the scoped route (i) text is in", "Operand-aware epilogues would cost" in PREREG, True)
    expect("v2's replaced sentence is gone", "The risk to name first" in PREREG, False)
    expect("the role edit is in", "speed or energy KEEPs (accuracy cannot, §6)." in PREREG, True)
    expect("the user's words are quoted", '"Go with U1-U12"' in PREREG, True)
    expect("PROTOCOL_JSON serializes", bool(json.dumps(protocol())), True)
    expect("layer work at 2048: 386.5 GFLOP", round(layer_flops(2048) / 1e9, 1), 386.5)
    expect("weights per layer: 94,371,840", sum(K * N for _, _, K, N, _ in LINEARS), 94_371_840)
    expect("the 3c insts.bin fit at M = 8192: i8 P 41,232 B, bf16 P 82,448 B",
           (INSTS_FIT[0] + INSTS_FIT[1] * 8192 // (8 * 64), INSTS_FIT[0] + INSTS_FIT[1] * 8192 // (8 * 32)),
           (41232, 82448))
    print("SELFTEST", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("models", "placement", "insts-fit", "prereg", "build", "inputs", "verdict",
                                     "selftest"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.mode == "placement":
        return placement(a.args[0])
    if a.mode == "verdict":
        return verdict(a.args)
    return {"models": models, "insts-fit": insts_fit, "prereg": prereg, "build": build, "inputs": inputs,
            "selftest": selftest}[a.mode]()


if __name__ == "__main__":
    sys.exit(main())
