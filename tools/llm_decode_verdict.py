#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The LLM study's Phase 1 pre-registration and its mechanical verdict, from one set of constants.

    python tools/llm_decode_verdict.py --prereg
        prints the pre-registration (question, arms, rules, predictions) and a PREREG_JSON line. It is
        committed before any Phase 1 timing runs.
    python tools/llm_decode_verdict.py --logs results/llm/llm_*_<machine>_<date>.log
        reads the ROW_JSON lines that tools/llm_gemv_bench.py and tools/cpu_mem_bw.py print, and the
        HOST_LOAD_VERDICT lines of the load witnesses, and prints the tables and the verdict.

Nothing here was tuned after seeing a timing: the constants are the ones --prereg printed.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Llama-2-7B's 32 decoder layers, linear weights only: per layer q, k, v, o are 4096 x 4096 (K x N),
# gate and up 4096 x 11008, down 11008 x 4096. lm_head, embedding, attention and the KV cache are
# excluded on every side of the comparison.
LAYERS = 32
SHAPES = {(4096, 4096): 4, (4096, 11008): 2, (11008, 4096): 1}
LINEAR_WEIGHTS = LAYERS * sum(k * n * c for (k, n), c in SHAPES.items())        # 6,476,005,376
BLOCKS = (32, 128)

# The NPU's most favourable case: the fewest bytes (group 128, bf16 scales, no zero points, 4.125
# bits per weight) at the highest NPU DRAM rate this repo has seen (SILICON 1.6, 28.1 GB/s per
# direction in a round trip; the engine's fill transport is 26.8 GB/s). DERIVED, not a hard cap:
# no clean read-only NPU test exists.
NPU_BYTES_MIN = LINEAR_WEIGHTS * (0.5 + 2 / 128)                                   # 3.339 GB
NPU_GBPS_BEST = 28.1
NPU_GBPS_FILL = 26.8
NPU_FLOOR_MS = NPU_BYTES_MIN / (NPU_GBPS_BEST * 1e9) * 1e3                         # 118.8 ms

# Decisive int4 configurations: every one runs all 3 shapes at both blocks.
CONFIGS = {
    "C23a0": {"ep": "cpu", "ort": "1.23", "t1": "fp32", "acc": 0, "threads": 8,
              "text": "CPU, ONNX Runtime 1.23.3 (resnet_env17), fp32 activations, accuracy_level 0 (fp32 compute)"},
    "C23a4": {"ep": "cpu", "ort": "1.23", "t1": "fp32", "acc": 4, "threads": 8,
              "text": "CPU, ONNX Runtime 1.23.3, fp32 activations, accuracy_level 4 (int8 compute)"},
    "C30a0": {"ep": "cpu", "ort": "1.30", "t1": "fp32", "acc": 0, "threads": 8,
              "text": "CPU, ONNX Runtime 1.30.0 (mlir-aie-iron), accuracy_level 0"},
    "C30a4": {"ep": "cpu", "ort": "1.30", "t1": "fp32", "acc": 4, "threads": 8,
              "text": "CPU, ONNX Runtime 1.30.0, accuracy_level 4"},
    "Df32": {"ep": "dml", "ort": "1.23", "t1": "fp32", "acc": 0, "threads": None,
             "text": "Radeon 780M, DirectML EP 1.23.3, fp32 activations and scales, every node on DirectML"},
    "Df16": {"ep": "dml", "ort": "1.23", "t1": "fp16", "acc": 0, "threads": None,
             "text": "Radeon 780M, DirectML EP 1.23.3, fp16 activations and scales, every node on DirectML"},
}
FP32_COMPUTE = ("C23a0", "C30a0", "Df32")      # must reproduce float64 to fp32 rounding
FP32_ERR_MAX = 1e-5                             # rel_l2 above this on an fp32-compute row is a PROBLEM
MIN_RUN_BYTES = 1 << 30
TIMING = {"warmup": 3, "reps": 20, "statistic": "median", "graph_optimization": "ORT_ENABLE_ALL",
          "cpu_intra_op_threads": 8, "execution_mode": "sequential"}
SPEEDUP_ACC4 = 1.2                              # prediction P3's threshold, descriptive only

PREREG = f"""LLM study, Phase 1: CPU and DirectML yardsticks for M = 1 decode, pre-registered before any timing

Question
  Can the NPU alone decode a Llama-2-7B-shaped int4 model faster than the better of the CPU and
  the Radeon 780M on this APU (Ryzen 7 8700G, DDR5-6000 on two channels)? And what error does each
  of those yardsticks carry, so a later NPU arm can be judged on accuracy?

Workload
  y = x W^T at M = 1 through ONNX Runtime's com.microsoft MatMulNBits: uint4 weights, round-to-nearest
  asymmetric per block of K, one zero point per block, blocks of {BLOCKS[0]} and {BLOCKS[1]}. Shapes (K x N):
  4096 x 4096, 4096 x 11008, 11008 x 4096. Each timed run streams >= 1 GiB of distinct weight
  copies (64x the 16 MiB L3), so every row reads DRAM. Weights ~ N(0, 0.02), seed 20260923, generator
  rtn-asym-uint4-v1 (tools/llm_gemv_bench.py). x ~ N(0, 1) rounded to fp16.

Decisive configurations (each: 3 shapes x 2 blocks = 6 rows; 36 rows in all)
""" + "".join(f"  {name:6s} {c['text']}\n" for name, c in CONFIGS.items()) + f"""
Context rows (never decide)
  dense  fp32 MatMul on the CPU (1.23.3) and fp16 MatMul on DirectML, same shapes, the block-32
         weights dequantized: the dense-weight read rate of each chip.
  sweep  C23a4 at 4096 x 11008, block 32, CPU threads 1, 2, 4, 16 (8 is in the matrix).
  read   tools/cpu_mem_bw.py (numpy float32 sums over 4 GiB, threads 1, 2, 4, 8, 16);
         ReduceSum over a 1 GiB initializer through ONNX Runtime: CPU fp32 (8 threads), DirectML
         fp16 and fp32.

Timing
  warmup {TIMING['warmup']}, then {TIMING['reps']} timed session.run calls; the median decides. ms per GEMV =
  median / copies. Session creation, weight upload and CPU prepacking are outside the timed
  region. ONNX Runtime default graph optimizations, sequential execution, CPU intra-op
  threads 8. DirectML rows use session.disable_cpu_ep_fallback, so a session that would put any
  node on the CPU is refused and the row is void (the controls prove the refusal works in this
  build). One sitting, serial, announced to the other sessions beforehand; the NPU idle
  (xrt-smi: no hardware contexts) before and after; a host-load witness per group.

Accuracy
  one GEMV (copy 0) per row against float64 y_ref = x W^T, W the exact dequantized weights: rel_l2 =
  ||y - y_ref|| / ||y_ref|| and max_abs_rel = max|y - y_ref| / max|y_ref|. A configuration's error
  E is its largest rel_l2 over the 3 shapes, per block.

Token time (DERIVED from MEASURED rows)
  T(config, block) = {LAYERS} x (4 t(4096x4096) + 2 t(4096x11008) + 1 t(11008x4096)), t = ms per GEMV.
  Linear layers only: {LINEAR_WEIGHTS:,} weights. T_best = the smallest T over the 12 (config, block)
  pairs.

NPU floor (DERIVED)
  The NPU's most favourable case: {NPU_BYTES_MIN / 1e9:.3f} GB per token (group 128, bf16 scales, no zero
  points) at {NPU_GBPS_BEST} GB/s, the highest NPU DRAM rate this repo has seen: {NPU_FLOOR_MS:.1f} ms. At the
  {NPU_GBPS_FILL} GB/s fill transport it is {NPU_BYTES_MIN / (NPU_GBPS_FILL * 1e9) * 1e3:.1f} ms. No clean read-only NPU test exists.

Rule (speed)
  KILL   if T_best <= {NPU_FLOOR_MS:.1f} ms: NPU-only int4 decode of a 7B model cannot beat the better of
         the CPU and DirectML at the NPU DRAM rates measured so far. It reopens only if a clean NPU
         read test (Phase 3 S1) measures at least BW_needed = {NPU_BYTES_MIN / 1e9:.3f} GB / T_best.
  ALIVE  otherwise: NPU-only decode stays a speed candidate, and S1 must show BW_needed.
  Either way the verdict prints T_best, its configuration, and BW_needed.
Rule (accuracy, for Phase 3)
  Each (config, block) is a point (T, E). An NPU arm earns an accuracy role only if no CPU or
  DirectML point is at least as fast AND at least as accurate (Pareto non-dominance). Phase 1 prints
  the front; it cannot decide this rule alone.
Problems that void the verdict (printed as PROBLEM, verdict INCOMPLETE)
  a missing decisive row; a DirectML row whose session failed; a row streaming < 1 GiB; a
  non-finite output; an fp32-compute row ({', '.join(FP32_COMPUTE)}) with rel_l2 > {FP32_ERR_MAX:g}; a host-load
  witness reading PEER for a timing group.

Written predictions (stated expectations; the rules above decide, these do not)
  P1  numpy read at 8-16 threads: 50-75 GB/s (96 GB/s theoretical at the configured DDR5-6000).
  P2  DirectML ReduceSum fp16: 40-85 GB/s.
  P3  the best CPU int4 configuration reads 25-55 GB/s of int4 bytes; accuracy_level 4 is at least
      {SPEEDUP_ACC4}x faster than accuracy_level 0 at the same block.
  P4  DirectML int4 reads 20-60 GB/s, and its GEMV at a shape takes under half the time of the
      fp16 dense GEMV there. If not, DirectML is expanding the weights before it reads them.
  P5  T_best <= {NPU_FLOOR_MS:.1f} ms, i.e. KILL: NPU-only 7B decode loses on speed.
  P6  fp32-compute rows reach rel_l2 <= 1e-6; accuracy_level 4 about 5e-3; DirectML fp16 about 4e-4.
      So an NPU arm (bf16 or int8 activations) can win on accuracy only where it is faster than
      every fp32-compute configuration.

Unverified by design: attention, the KV cache and lm_head are not timed; one prompt position; one
machine; ONNX Runtime's CPU prepacking may change the bytes the CPU actually reads (T is what
decides, not GB/s); llama.cpp is not a yardstick here (the user's choice); the NPU is not used.
"""


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def cmd_prereg(_args) -> int:
    print(PREREG)
    head = git("rev-parse", "HEAD")
    dirty = [line for line in git("status", "--porcelain").splitlines() if line]
    print(f"git HEAD {head} (+{len(dirty)} uncommitted paths: this log's own commit)")
    record = {"layers": LAYERS, "shapes": [[k, n, c] for (k, n), c in SHAPES.items()], "blocks": list(BLOCKS),
              "linear_weights": LINEAR_WEIGHTS, "npu_bytes_min": NPU_BYTES_MIN, "npu_gbps_best": NPU_GBPS_BEST,
              "npu_gbps_fill": NPU_GBPS_FILL, "npu_floor_ms": NPU_FLOOR_MS, "configs": CONFIGS,
              "fp32_compute": list(FP32_COMPUTE), "fp32_err_max": FP32_ERR_MAX, "min_run_bytes": MIN_RUN_BYTES,
              "timing": TIMING, "git_head": head}
    print("PREREG_JSON " + json.dumps(record, sort_keys=True))
    return 0


def config_of(row: dict, ort: str):
    for name, c in CONFIGS.items():
        if (row["ep"], row["t1"], ort) == (c["ep"], c["t1"], c["ort"]) and row.get("acc") == c["acc"] \
                and row.get("threads") == c["threads"]:
            return name
    return None


def read_logs(paths):
    rows, problems, witnesses = [], [], []
    for p in map(Path, paths):
        text = p.read_text(encoding="utf-8")
        if p.name.startswith("load_"):
            for v in re.findall(r"^HOST_LOAD_VERDICT (\S+)", text, flags=re.M):
                witnesses.append((p.name, v))
            continue
        ort = None
        for line in text.splitlines():
            if line.startswith("HEADER "):
                ver = json.loads(line[7:]).get("onnxruntime")
                ort = ".".join(ver.split(".")[:2]) if ver else None
            elif line.startswith("ROW_JSON "):
                r = json.loads(line[9:])
                r["log"], r["ort"] = p.name, ort
                rows.append(r)
        for m in re.findall(r"^ROW_FAILED (.*)$", text, flags=re.M):
            problems.append(f"{p.name}: row failed: {m}")
    for name, v in witnesses:
        if v == "PEER":
            problems.append(f"{name}: a host-load witness reads PEER")
    return rows, problems, witnesses


def cmd_logs(args) -> int:
    rows, problems, witnesses = read_logs(args.logs)
    gemv = [r for r in rows if r["kind"] == "gemv"]
    t = {}
    for r in gemv:
        if r["arm"] != "nbits":
            continue
        name = config_of(r, r["ort"])
        if name is None:
            continue
        key = (name, r["block"], r["k"], r["n"])
        if key in t:
            problems.append(f"two rows for {key}")
        t[key] = r
        if r["run_bytes"] < MIN_RUN_BYTES:
            problems.append(f"{key}: {r['run_bytes']} bytes per run < 1 GiB")
        if not r["finite"]:
            problems.append(f"{key}: non-finite output")
        if name in FP32_COMPUTE and r["rel_l2"] > FP32_ERR_MAX:
            problems.append(f"{key}: fp32-compute rel_l2 {r['rel_l2']:.2e} > {FP32_ERR_MAX:g}")

    print("Read bandwidth (context)")
    for r in rows:
        if r["kind"] == "cpu_read":
            print(f"  numpy float32 sum, {r['threads']:2d} threads: {r['gbps']:6.2f} GB/s (best {r['gbps_best']:.2f})")
        elif r["kind"] == "readbw":
            who = f"ONNX Runtime {r['ep']} ReduceSum {r['dtype']}" + (f", {r['threads']} threads" if r["threads"] else "")
            print(f"  {who}: {r['gbps']:6.2f} GB/s (best {r['gbps_best']:.2f})")
    print()

    print(f"{'config':6s} {'block':>5s} {'K x N':>12s} {'ms/GEMV':>9s} {'GB/s':>7s} {'rel_l2':>9s} {'max_abs_rel':>11s}")
    for name in CONFIGS:
        for b in BLOCKS:
            for (k, n) in SHAPES:
                r = t.get((name, b, k, n))
                if r is None:
                    problems.append(f"missing decisive row {name} block {b} {k}x{n}")
                    continue
                print(f"{name:6s} {b:5d} {f'{k}x{n}':>12s} {r['ms_per_gemv']:9.4f} {r['gbps']:7.2f} "
                      f"{r['rel_l2']:9.2e} {r['max_abs_rel']:11.2e}")
    print()

    points = {}
    for name in CONFIGS:
        for b in BLOCKS:
            got = [t.get((name, b, k, n)) for (k, n) in SHAPES]
            if any(g is None for g in got):
                continue
            tt = LAYERS * sum(c * g["ms_per_gemv"] for c, g in zip(SHAPES.values(), got))
            tok_bytes = LAYERS * sum(c * g["copy_bytes"] for c, g in zip(SHAPES.values(), got))
            points[(name, b)] = {"T": tt, "E": max(g["rel_l2"] for g in got), "gbps": tok_bytes / tt / 1e6}
    print(f"{'config':6s} {'block':>5s} {'T ms/token':>10s} {'tokens/s':>8s} {'int4 GB/s':>9s} {'E (max rel_l2)':>14s}")
    for (name, b), p in sorted(points.items(), key=lambda kv: kv[1]["T"]):
        print(f"{name:6s} {b:5d} {p['T']:10.1f} {1e3 / p['T']:8.2f} {p['gbps']:9.2f} {p['E']:14.2e}")
    front = [k for k, p in points.items()
             if not any(q["T"] <= p["T"] and q["E"] <= p["E"] and (q["T"] < p["T"] or q["E"] < p["E"])
                        for kk, q in points.items() if kk != k)]
    print("Pareto front (T, E), CPU and DirectML: " +
          "; ".join(f"{n} b{b} {points[(n, b)]['T']:.1f} ms {points[(n, b)]['E']:.1e}"
                    for n, b in sorted(front, key=lambda kb: points[kb]["T"])))
    print()

    dense = [r for r in gemv if r["arm"] == "dense"]
    for r in sorted(dense, key=lambda r: (r["ep"], r["k"], r["n"])):
        print(f"context dense {r['ep']} {r['t1']} {r['k']}x{r['n']}: {r['ms_per_gemv']:.4f} ms/GEMV, "
              f"{r['gbps']:.2f} GB/s, rel_l2 {r['rel_l2']:.2e}")
    sweep = [r for r in gemv if r["arm"] == "nbits" and r["ep"] == "cpu" and r.get("threads") not in (8, None)]
    for r in sorted(sweep, key=lambda r: r["threads"]):
        print(f"context sweep {r['ort']} acc {r['acc']} block {r['block']} {r['k']}x{r['n']}, {r['threads']} threads: "
              f"{r['ms_per_gemv']:.4f} ms/GEMV, {r['gbps']:.2f} GB/s")
    for name, v in witnesses:
        if v != "CLEAR":
            print(f"witness {name}: {v}")
    print()

    if problems:
        for p in problems:
            print(f"PROBLEM {p}")
        print("VERDICT INCOMPLETE: the rows above do not meet the pre-registered setup")
        return 2
    (bn, bb), best = min(points.items(), key=lambda kv: kv[1]["T"])
    need = NPU_BYTES_MIN / (best["T"] / 1e3) / 1e9
    print(f"T_best {best['T']:.1f} ms/token ({bn}, block {bb}); NPU floor {NPU_FLOOR_MS:.1f} ms "
          f"({NPU_BYTES_MIN / 1e9:.3f} GB at {NPU_GBPS_BEST} GB/s); BW_needed {need:.1f} GB/s")
    if best["T"] <= NPU_FLOOR_MS:
        print(f"VERDICT KILL: NPU-only 7B int4 decode cannot beat {bn} at the NPU DRAM rates measured so far "
              f"({NPU_GBPS_FILL} fill, {NPU_GBPS_BEST} best). It reopens only if a clean NPU read test measures "
              f">= {need:.1f} GB/s.")
    else:
        print(f"VERDICT ALIVE: NPU-only decode stays a speed candidate; Phase 3 S1 must show >= {need:.1f} GB/s "
              f"of NPU reads.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prereg", action="store_true")
    mode.add_argument("--logs", nargs="+")
    args = ap.parse_args()
    return cmd_prereg(args) if args.prereg else cmd_logs(args)


if __name__ == "__main__":
    sys.exit(main())
