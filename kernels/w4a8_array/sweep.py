# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run whole_array_w4a8.py over arms, one fresh process per run, into one JSONL, and tabulate.

Each arm is `arm:mode:c_single_buffer`. Arms rotate by one place per repetition, so no arm
always runs first or last. xrt-smi's partition report is recorded at the start and at the
end: a foreign hardware context on the device while this runs is contention, not a
finding. Every run draws the same A and B, so every arm must return the same C; the table
checks that (C's sha256) as well as each run's own bit-exact check against numpy.

GOPS = 2MKN over the NPU-bracket average (whole_array.py's own "NPU time": the runtime's
timer around kernel.wait()), the convention of every whole_array figure in this repo.

Usage (ironenv, Desktop 2):
    python kernels/w4a8_array/sweep.py --out results/aie/w4a8_array_raw.jsonl --tag main
    python kernels/w4a8_array/sweep.py --arms native:unroll2:1,upstream:default:1 --reps 1
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_DESIGN = os.path.join(_HERE, "whole_array_w4a8.py")
_XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"

DEFAULT_ARMS = ",".join(
    [
        "upstream:default:1",  # whole_array as built for the 4852.06 GOPS row -- the anchor
        "i8:default:1",        # upstream's kernel re-typed (the one-core control)
        "i8:unroll2:1",        # the best int8 core schedule
        "unpack:unroll2:1",    # B bytes halved, kernel ~= i8:default per call (3,749 vs 3,735)
        "unpack:no-unroll:1",  # B bytes halved, kernel ~= i8:unroll2 per call (3,509 vs 3,559)
        "native:default:1",
        "native:unroll2:1",
        "native:unroll2:0",    # C double-buffered: fits only because B is half the bytes
    ]
)


def _rel(arg: str) -> str:
    """Repo-relative, so a committed row carries no local profile or worktree path; a path
    outside the repo keeps only its file name."""
    try:
        rel = os.path.relpath(arg, _ROOT) if os.path.isabs(arg) or os.path.exists(arg) else arg
    except ValueError:
        return f"<outside-repo>/{os.path.basename(arg)}"
    return f"<outside-repo>/{os.path.basename(arg)}" if rel.startswith("..") else rel


def xrt_contexts() -> str:
    try:
        out = subprocess.run([_XRT_SMI, "examine", "-r", "aie-partitions"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        return f"xrt-smi failed: {e}"
    lines = [ln.strip() for ln in out.splitlines() if "context" in ln.lower()]
    return " | ".join(lines) or out.strip()[-200:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="JSONL to append rows to")
    ap.add_argument("--arms", default=DEFAULT_ARMS, help="comma-separated arm:mode:c_single_buffer")
    ap.add_argument("--shape", default="2048x2048x2048", help="M x K x N")
    ap.add_argument("--tile", default="64x128x64", help="m x k x n")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--log", help="also append everything printed to this file (UTF-8)")
    args = ap.parse_args(argv)

    def print(*parts, sep=" ", flush=True):  # noqa: A001 -- tee to --log
        text = sep.join(str(p) for p in parts)
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        if args.log:
            with open(args.log, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    arms = [tuple(a.split(":")) for a in args.arms.split(",") if a]
    M, K, N = (int(x) for x in args.shape.lower().split("x"))
    m, k, n = (int(x) for x in args.tile.lower().split("x"))

    def log(row):
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    log({"kind": "start", "host": socket.gethostname(), "t": time.strftime("%Y-%m-%d %H:%M:%S"),
         "argv": [_rel(a) for a in sys.argv], "xrt_smi": xrt_contexts(), "tag": args.tag})
    runs, failures = [], []
    for rep in range(args.reps):
        shift = rep % len(arms)
        for arm, mode, csb in arms[shift:] + arms[:shift]:
            cmd = [sys.executable, _DESIGN, "--dev", "npu", "-M", str(M), "-K", str(K), "-N", str(N),
                   "-m", str(m), "-k", str(k), "-n", str(n), "--n-aie-cols", str(args.cols),
                   "--arm", arm, "--mode", mode, "--c-single-buffer", csb,
                   "--warmup", str(args.warmup), "--iters", str(args.iters)]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.perf_counter() - t0
            rows = [ln for ln in proc.stdout.splitlines() if ln.startswith("W4A8_ROW ")]
            row = json.loads(rows[-1][len("W4A8_ROW "):]) if rows else {}
            verdict = "PASS" if "PASS!" in proc.stdout else ("FAIL" if "FAIL!" in proc.stdout else "-")
            row.update({"kind": "run", "tag": f"{args.tag} rep{rep}", "rep": rep,
                        "host": socket.gethostname(), "t": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "exit": proc.returncode, "wall_s": round(dt, 1), "verdict": verdict})
            if not rows:
                row.update({"arm": arm, "mode": mode, "c_single_buffer": int(csb),
                            "tail": (proc.stdout[-800:] + proc.stderr[-800:])})
            log(row)
            runs.append(row)
            npu = row.get("npu_us")
            print(f"rep{rep} {arm}:{mode}:{csb:<2} exit {proc.returncode} {dt:5.1f}s  {verdict}  "
                  + (f"NPU avg/min/max {npu[0]:.1f}/{npu[1]:.1f}/{npu[2]:.1f} us  "
                     f"{row['gops']:.2f} GOPS  mism {row['mismatches']}  C {row['c_sha256']}"
                     if npu else "no timing row"), flush=True)
            if proc.returncode != 0 or verdict != "PASS":
                failures.append((rep, arm, mode, csb, proc.returncode))
                print(proc.stdout[-1500:], proc.stderr[-1500:], sep="\n", flush=True)
    log({"kind": "end", "host": socket.gethostname(), "t": time.strftime("%Y-%m-%d %H:%M:%S"),
         "xrt_smi": xrt_contexts(), "tag": args.tag, "failures": failures})

    ops = 2.0 * M * K * N
    good = [r for r in runs if r.get("npu_us") and r.get("verdict") == "PASS"]
    hashes = {r["c_sha256"] for r in good}
    print(f"\nSUMMARY  {M}x{K}x{N}  tile {m}/{k}/{n}  cols {args.cols}  iters {args.iters} "
          f"warmup {args.warmup}  reps {args.reps}")
    print(f"  C sha256 across all passing runs: {sorted(hashes)} "
          f"({'one value: every arm computed the same C' if len(hashes) == 1 else 'DIFFERENT'})")
    by_arm: dict = {}
    for r in good:
        by_arm.setdefault((r["arm"], r["mode"], r["c_single_buffer"]), []).append(r)
    anchor = by_arm.get(("upstream", "default", 1))
    anchor_avg = statistics.mean(r["npu_us"][0] for r in anchor) if anchor else None
    print(f"  {'arm':<24} {'B':<5} {'L1 est':>7} {'runs':>4} {'NPU avg us (per run)':<30} "
          f"{'mean':>8} {'GOPS':>8} {'vs upstream':>11}")
    for key in [tuple([a, mo, int(c)]) for a, mo, c in arms]:
        rs = by_arm.get(key)
        name = f"{key[0]}:{key[1]}:{key[2]}"
        if not rs:
            print(f"  {name:<24} no passing run")
            continue
        avgs = [r["npu_us"][0] for r in rs]
        mean = statistics.mean(avgs)
        print(f"  {name:<24} {'int4' if rs[0]['b_packed'] else 'int8':<5} {rs[0]['l1_est']:>7,} "
              f"{len(rs):>4} {' '.join(f'{a:.1f}' for a in avgs):<30} {mean:>8.1f} "
              f"{ops / (1000 * mean):>8.2f} "
              f"{(anchor_avg / mean) if anchor_avg else float('nan'):>10.3f}x")
    print(f"done; {len(failures)} failed runs: {failures}")
    return 0 if not failures and len(hashes) == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
