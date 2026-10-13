#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The W4A8 accuracy gate's pre-registration and its mechanical verdict, from one set of constants.

    python tools/w4a8_verdict.py --prereg [--models A.onnx B.onnx ...]
        prints the pre-registration text and a PREREG_JSON line (constants, git HEAD, onnxruntime
        version, sha256 of every model named). It is committed before any evaluation runs.
    python tools/w4a8_verdict.py --logs results/int4/eval_*.log
        parses mAP@50-95 from each pipelines/yolov8n/5_eval_map.py log and prints the table and the
        verdict under the same constants. The log name gives the row:
        eval_<model>_<arm>_cpu_<unopt|opt>_<machine>_<date>.log

Nothing here was tuned after seeing a mAP: the constants are the ones printed by --prereg.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRIC = "mAP@50-95"
IMAGES = 5000                      # all of COCO val2017
MODELS = ("yolov8n", "yolov8s")
BASELINE = "b8"                    # the shipped W8A8 file, ORT_DISABLE_ALL
DECISION_ARMS = ("e1", "e2")       # the better of the two decides, per model
CONTEXT_ARMS = ("e1h", "u")        # reported, never decide
KILL_LINE = 1.0                    # points of mAP@50-95, absolute, per model: PASS iff delta <= KILL_LINE
NARROW_LINE = 2.0                  # KILL_LINE < delta <= NARROW_LINE adds a descriptive note only
OPT_GAP_CAVEAT = 0.2               # |B8 optimized - B8 unoptimized| >= this is a caveat for the docs
EVAL = {"conf": 0.001, "iou": 0.7, "max_det": 300, "ep": "cpu", "ort_opt": "disable_all"}

ARM_TEXT = {
    "b8": "B8   the shipped W8A8 XINT8 file, unchanged",
    "e1": "E1   W4, one pow2 scale per weight tensor (today's engine lowers it)",
    "e2": "E2   W4, one pow2 scale per 32-output-channel group (needs a compiler change: a shift per packet)",
    "e1h": "E1h  E1 with the stem and the 6 head-output convs kept W8 (diagnostic)",
    "u": "U    W4, one pow2 scale per output channel (context: needs a kernel change)",
}

PREREG = f"""W4A8 accuracy gate, pre-registered before any evaluation (smoke runs included)

Question: does 4-bit weight quantization (W4A8) of YOLOv8n and YOLOv8s lose more than {KILL_LINE} point of
{METRIC} against the shipped W8A8 model on COCO val2017, emulated on the CPU with no NPU?

Metric   {METRIC}, all {IMAGES} val2017 images, pipelines/yolov8n/5_eval_map.py at conf {EVAL['conf']},
         IoU {EVAL['iou']}, max_det {EVAL['max_det']}, per-class NMS, pycocotools.
Setup    --ep cpu --ort-opt disable_all (ORT_DISABLE_ALL, the reference the engine's integer oracle
         matches layer for layer), one pinned --ort-threads for every row, resnet_env17, one sitting,
         serial, no NPU. Never --fresh.
Weights  round-to-nearest only, grid [-7, 7], pow2 scales chosen by Quark's MinMSE on the float
         weights. Activation Q/DQ, biases and every other scale stay as shipped (frozen at W8).
Arms     {ARM_TEXT['b8']}
         {ARM_TEXT['e1']}
         {ARM_TEXT['e2']}
         {ARM_TEXT['e1h']}
         {ARM_TEXT['u']}
         B8o  B8 again with ONNX Runtime's default optimizations (cross-check only)

Delta    per model, B8 minus the better of E1 and E2, from the printed 2-decimal {METRIC} values.
PASS     delta <= {KILL_LINE} for BOTH yolov8n and yolov8s. Accuracy then no longer blocks int4 in the
         engine; the next gates are the packet format (variable-size packets) and the 224 B .text
         left in the core. The headline names the passing arm per model: E1 runs on today's engine,
         E2 needs a compiler change.
KILL     otherwise: "int4-in-engine killed at RTN (pow2 scales, this dialect); GPTQ/AdaRound untested".
Note     if {KILL_LINE} < delta <= {NARROW_LINE}, the log adds "fails narrowly; a recovery method is the untested
         next step". The note does not change the verdict.
Context  E1h, U and B8o never decide. They are reported beside the verdict.

Written prediction: B8o - B8 ~= 0.00. With int8 biases, optimized ONNX Runtime should keep each Conv
in float rather than fuse it to QLinearConv, so both sessions compute the same thing. A gap of 0.00
confirms that reading; a gap of {OPT_GAP_CAVEAT} or more means it was wrong, and the docs' optimized-CPU mAPs
need a caveat.
Stated expectation, not an assumption: per-tensor RTN at 4 bits is expected to lose several points;
per-group less.

Unverified by design: no int4-packed engine lowering exists (E2's "compiler-only" is on paper); no
recovery method (GPTQ, AdaRound, bias correction) runs; no float per-channel-scale arm, so the cost
of pow2 scales is unmeasured; the NPU is not used.
"""


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def cmd_prereg(args) -> int:
    import onnxruntime as ort
    print(PREREG)
    head = git("rev-parse", "HEAD")
    dirty = [line for line in git("status", "--porcelain").splitlines() if line]
    models = {m.replace("\\", "/"): hashlib.sha256(Path(m).read_bytes()).hexdigest() for m in args.models}
    for name, digest in models.items():
        print(f"sha256 {digest}  {name}")
    print(f"git HEAD {head} (+{len(dirty)} uncommitted paths: this log's own commit)")
    print(f"onnxruntime {ort.__version__}")
    print(f"ort-threads {args.threads} for every row")
    record = {"metric": METRIC, "images": IMAGES, "models": list(MODELS), "baseline": BASELINE,
              "decision_arms": list(DECISION_ARMS), "context_arms": list(CONTEXT_ARMS),
              "kill_line": KILL_LINE, "narrow_line": NARROW_LINE, "opt_gap_caveat": OPT_GAP_CAVEAT,
              "eval": EVAL, "rounding": "RTN", "grid": [-7, 7], "git_head": head,
              "onnxruntime": ort.__version__, "ort_threads": args.threads, "sha256": models}
    print("PREREG_JSON " + json.dumps(record, sort_keys=True))
    return 0


LOG_NAME = re.compile(r"eval_(yolov8[ns])_([a-z0-9]+)_cpu_(unopt|opt)_[a-z0-9]+_\d{8}\.log$")


def parse_log(path: Path) -> dict:
    m = LOG_NAME.search(path.name)
    if not m:
        raise SystemExit(f"{path.name}: not eval_<model>_<arm>_cpu_<unopt|opt>_<machine>_<date>.log")
    text = path.read_text(encoding="utf-8")

    def one(pattern):
        found = re.findall(pattern, text, flags=re.M)
        if len(found) != 1:
            raise SystemExit(f"{path.name}: expected one match of {pattern!r}, found {len(found)}")
        return found[0]
    return {"model": m.group(1), "arm": m.group(2), "opt": m.group(3), "log": path.name,
            "map": float(one(r"^mAP@50-95\s+([0-9.]+)$")),
            "images": int(one(r"^=== \S+ \| CPU \| \d+px \| (\d+) images ===$")),
            "ort_opt": one(r"^ort-opt : (\S+)"), "threads": one(r"ort-threads : (.+)$").strip(),
            "ort": one(r"^onnxruntime (\S+)$"), "sha256": one(r"^model sha256 ([0-9a-f]{64})$"),
            "skipped": len(re.findall(r"^\s*skip unreadable", text, flags=re.M))}


def cmd_logs(args) -> int:
    rows = [parse_log(Path(p)) for p in args.logs]
    problems = []
    table = {}
    for r in rows:
        key = (r["model"], r["arm"], r["opt"])
        if key in table:
            problems.append(f"two logs for {key}")
        table[key] = r
        if r["images"] != IMAGES or r["skipped"]:
            problems.append(f"{r['log']}: {r['images']} images, {r['skipped']} skipped")
        want = "disable_all" if r["opt"] == "unopt" else "default"
        if r["ort_opt"] != want:
            problems.append(f"{r['log']}: ort-opt {r['ort_opt']}, the name says {want}")
    for field in ("threads", "ort"):
        seen = {r[field] for r in rows}
        if len(seen) > 1:
            problems.append(f"{field} differs between rows: {sorted(seen)}")

    print(f"{'model':8s} {'arm':4s} {'ORT':6s} {METRIC:>9s} {'delta vs B8':>11s}  log")
    for model in MODELS:
        base = table.get((model, BASELINE, "unopt"))
        for arm in (BASELINE,) + DECISION_ARMS + CONTEXT_ARMS:
            for opt in ("unopt", "opt"):
                r = table.get((model, arm, opt))
                if r is None:
                    continue
                d = "" if base is None or r is base else f"{base['map'] - r['map']:+.2f}"
                print(f"{model:8s} {arm.upper():4s} {opt:6s} {r['map']:>9.2f} {d:>11s}  {r['log']}")
    print()

    verdicts = {}
    for model in MODELS:
        base = table.get((model, BASELINE, "unopt"))
        arms = {a: table.get((model, a, "unopt")) for a in DECISION_ARMS}
        if base is None or any(v is None for v in arms.values()):
            problems.append(f"{model}: missing B8 or a decision arm (unoptimized)")
            continue
        deltas = {a: round(base["map"] - arms[a]["map"], 2) for a in DECISION_ARMS}
        best = min(DECISION_ARMS, key=lambda a: (deltas[a], DECISION_ARMS.index(a)))
        passing = [a.upper() for a in DECISION_ARMS if deltas[a] <= KILL_LINE]
        verdicts[model] = {"delta": deltas[best], "best": best, "passing": passing, "deltas": deltas}
        line = (f"{model}: B8 {base['map']:.2f}; " +
                "; ".join(f"{a.upper()} {arms[a]['map']:.2f} (delta {deltas[a]:.2f})" for a in DECISION_ARMS) +
                f"; decision arm {best.upper()}, delta {deltas[best]:.2f} "
                f"{'<=' if deltas[best] <= KILL_LINE else '>'} {KILL_LINE}; passing arms: "
                f"{', '.join(passing) if passing else 'none'}")
        print(line)
        if KILL_LINE < deltas[best] <= NARROW_LINE:
            print(f"{model}: fails narrowly; a recovery method is the untested next step")
        for arm in CONTEXT_ARMS:
            r = table.get((model, arm, "unopt"))
            if r is not None:
                print(f"{model}: context {arm.upper()} {r['map']:.2f} (delta {base['map'] - r['map']:.2f}; does not decide)")
        o = table.get((model, BASELINE, "opt"))
        if o is not None:
            gap = round(o["map"] - base["map"], 2)
            print(f"{model}: B8o - B8 = {gap:+.2f} (predicted ~0.00; "
                  f"{'caveat for the optimized-CPU mAPs in the docs' if abs(gap) >= OPT_GAP_CAVEAT else 'prediction holds'})")
    print()
    if problems:
        for p in problems:
            print(f"PROBLEM {p}")
        print("VERDICT INCOMPLETE: the rows above do not meet the pre-registered setup")
        return 2
    if all(v["delta"] <= KILL_LINE for v in verdicts.values()):
        print("VERDICT PASS: accuracy no longer blocks int4 in the engine at RTN; " +
              "; ".join(f"{m} passes with {', '.join(v['passing'])}" for m, v in verdicts.items()) +
              ". E1 runs on today's engine; E2 needs a compiler change. Next gates: variable-size packets, "
              "the 224 B .text.")
    else:
        print("VERDICT KILL: int4-in-engine killed at RTN (pow2 scales, this dialect); GPTQ/AdaRound untested. " +
              "; ".join(f"{m} delta {v['delta']:.2f} ({v['best'].upper()})" for m, v in verdicts.items()))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prereg", action="store_true")
    mode.add_argument("--logs", nargs="+")
    ap.add_argument("--models", nargs="*", default=[], help="with --prereg: models to fingerprint")
    ap.add_argument("--threads", type=int, default=None, help="with --prereg: the pinned --ort-threads")
    args = ap.parse_args()
    if args.prereg and not args.threads:
        ap.error("--prereg needs --threads")
    return cmd_prereg(args) if args.prereg else cmd_logs(args)


if __name__ == "__main__":
    sys.exit(main())
