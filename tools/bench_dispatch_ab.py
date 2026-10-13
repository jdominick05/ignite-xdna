#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Interleaved same-sitting dispatch A/B of graph-engine containers, real and all-NOP, one process per run.

    python tools/bench_dispatch_ab.py --pair sesr=build/a_rb2.ignite,build/a_rb3.ignite \
        --pair yolov8n=build/b_rb2.ignite,build/b_rb3.ignite --rounds 4 --iters 300 --json out.json

Each ``--pair label=control,test`` is a control and a test container of one model. Every round runs every
container once through ``tools/engine_dispatch_floor.py`` (``--iters`` dispatches of the container and of its
all-NOP copy, each its own hardware context), and the order reverses on every second round, so each container
runs early and late. Per pair it prints each round's real and NOP dispatch means, the test-minus-control delta,
and how many rounds the test was lower. Run it under ``scripts/research-lowlevel.sh --npu`` so the sitting has a
contention witness; it checks nothing about the device itself. Written on Desktop 2 (Phoenix), 2026-10-13, for
the retire-cadence landing sitting; what has been run is in its logs.
"""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "tools" / "engine_dispatch_floor.py"


def run_floor(container: Path, iters: int, image: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "floor.json"
        cmd = [sys.executable, str(FLOOR), str(container), "--iters", str(iters), "--json", str(out)]
        if image:
            cmd += ["--image", str(image)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
            raise SystemExit(f"[ab] {container.name} failed; stopping - do not retry in a loop")
        d = json.loads(out.read_text(encoding="utf-8"))
    return {"real_ms": d["real"]["mean_ms"], "nop_ms": d["nop"]["mean_ms"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pair", action="append", required=True, metavar="LABEL=CONTROL,TEST")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--image", type=Path, default=None, help="passed to engine_dispatch_floor.py (its default if unset)")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()

    arms = []                                   # (label, role, path)
    for spec in args.pair:
        label, paths = spec.split("=", 1)
        control, test = (Path(p).resolve() for p in paths.split(","))
        for p in (control, test):
            if not p.exists():
                raise SystemExit(f"container not found: {p}")
        arms += [(label, "control", control), (label, "test", test)]
    results = []
    for rnd in range(1, args.rounds + 1):
        for label, role, path in (arms[::-1] if rnd % 2 == 0 else arms):
            r = run_floor(path, args.iters, args.image)
            r.update(label=label, role=role, container=path.name, round=rnd)
            results.append(r)
            print(f"[ab] round {rnd} {label:10s} {role:7s} {path.name}: real {r['real_ms']:.3f} ms, "
                  f"nop {r['nop_ms']:.3f} ms", flush=True)
    print()
    for label in dict.fromkeys(a[0] for a in arms):
        per = {role: {r["round"]: r for r in results if r["label"] == label and r["role"] == role}
               for role in ("control", "test")}
        for key in ("real_ms", "nop_ms"):
            c = [per["control"][i][key] for i in sorted(per["control"])]
            t = [per["test"][i][key] for i in sorted(per["test"])]
            lower = sum(b < a for a, b in zip(c, t))
            print(f"[summary] {label:10s} {key[:-3]:4s}: control {statistics.fmean(c):.3f} | test "
                  f"{statistics.fmean(t):.3f} | delta {statistics.fmean(t) - statistics.fmean(c):+.3f} ms | "
                  f"test lower in {lower}/{len(c)} rounds | control rounds "
                  + " / ".join(f"{x:.3f}" for x in c) + " | test rounds " + " / ".join(f"{x:.3f}" for x in t))
    args.json.write_text(json.dumps({"results": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
