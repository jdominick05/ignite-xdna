#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare per-layer shim traces of a control and a test container (``tools/engine_shim_trace.py --json`` files).

    python tools/shim_trace_compare.py --control a.json [a2.json ...] --test b.json [b2.json ...] \\
        --onnx models/yolov8s_cut_xint8.onnx --split box [--json out.json]

Every input holds one or more runs; runs are averaged per container. Layers are split into two sets, ``--split box``
being the layers ``compiler/box_fills.apply_box_fills`` converts (computed offline from the ONNX, as the compiler
does) against the rest. Per set it reports, test minus control, in cycles at the trace clock:
- ``cycles``: the layer windows (marker to marker);
- ``lead``: marker to the layer's first traced DMA event (holds a box layer's MemTile configure; not verdict-bearing);
- ``W_bp``: weight-stream backpressure (shim MM2S 1 blocked: the cores' weight buffers are full, weights AHEAD);
- ``A_bp``: activation-stream backpressure (shim MM2S 0 blocked by the MemTile);
- ``O_starve``: output-channel starvation (S2MM 0 has a task, the cores have produced nothing yet);
- ``W_mem`` / ``A_mem``: DDR read starvation of each stream.
It then applies the verdict rule pre-registered for the box-fills sitting (scratch pre-registration, quoted in the
log): on the box set, ``dW = sum dW_bp``, ``dC = sum dcycles``; no loss to attribute when dC < MIN_LOSS_CYCLES,
else H-W (weights late) when dW <= -0.5 dC, H-A (activations late, weights piling up) when dW >= +0.5 dC, else
neither. Offline; opens no hardware context.
Written on Desktop 2, 2026-10-15.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

KEYS = {"cycles": "cycles", "lead": "lead", "W_bp": "DMA_MM2S_1_STREAM_BACKPRESSURE", "A_bp": "DMA_MM2S_0_STREAM_BACKPRESSURE",
        "O_starve": "DMA_S2MM_0_STREAM_STARVATION", "W_mem": "DMA_MM2S_1_MEMORY_STARVATION",
        "A_mem": "DMA_MM2S_0_MEMORY_STARVATION"}
CLOCK_GHZ = 1.80
# Below this the box set holds no loss to attribute: about 3x the box-set layer-time difference between two control
# runs (88,782 cycles, 2026-10-15), 0.17 ms.
MIN_LOSS_CYCLES = 300_000


def load(paths):
    """{key: [per-layer mean over runs]}, layer names, run count, per-run frame cycles."""
    runs, names = [], None
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        names = names or d["layers"]
        if d["layers"] != names:
            raise SystemExit(f"{p}: a different layer list")
        runs += d["runs"]
    per = {k: [] for k in KEYS}
    frames = []
    for r in runs:
        col = sorted(r["shims"])[0]
        layers = r["shims"][col]["layers"]
        frames.append(r["shims"][col]["frame_cycles"])
        for k, ev in KEYS.items():
            per[k].append([L[ev] for L in layers])
    return {k: np.mean(v, axis=0) for k, v in per.items()}, names, len(runs), frames


def box_set(onnx: Path):
    from ignite_xdna.compiler import engine_schedule as sched
    from ignite_xdna.compiler.box_fills import apply_box_fills
    from ignite_xdna.compiler.graph_ir import lower_yolov8n
    ir = lower_yolov8n(onnx)
    scheds, _ = sched.schedule_graph(ir, sched.plan_workspace(ir))
    apply_box_fills(scheds)
    return {k for k, s in enumerate(scheds) if getattr(s, "box", None) is not None}, [s.name for s in scheds]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--control", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--split", choices=("box",), default="box")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    c, names, nc, fc = load(a.control)
    t, names_t, nt, ft = load(a.test)
    if names != names_t:
        raise SystemExit("control and test have different layer lists")
    boxed, sched_names = box_set(a.onnx)
    if len(sched_names) != len(names):
        raise SystemExit(f"the ONNX schedules {len(sched_names)} layers, the traces hold {len(names)}")
    print(f"[compare] control {nc} runs, test {nt} runs; frame (marker span) control "
          f"{np.mean(fc) / CLOCK_GHZ / 1e6:.3f} ms, test {np.mean(ft) / CLOCK_GHZ / 1e6:.3f} ms "
          f"(DERIVED at {CLOCK_GHZ} GHz; per-run control {[round(x / CLOCK_GHZ / 1e6, 3) for x in fc]}, test {[round(x / CLOCK_GHZ / 1e6, 3) for x in ft]})")
    sets = {"box": sorted(boxed), "rest": [k for k in range(len(names)) if k not in boxed]}
    out = {"sets": {}, "layers": []}
    print(f"\n{'set':5} {'layers':>6} " + " ".join(f"{k:>14}" for k in KEYS) + "   (test - control, cycles)")
    for sname, idx in sets.items():
        d = {k: float((t[k][idx] - c[k][idx]).sum()) for k in KEYS}
        base = {k: float(c[k][idx].sum()) for k in KEYS}
        out["sets"][sname] = {"layers": len(idx), "delta": d, "control": base}
        print(f"{sname:5} {len(idx):6d} " + " ".join(f"{d[k]:14,.0f}" for k in KEYS))
        print(f"{'':5} {'ctl':>6} " + " ".join(f"{base[k]:14,.0f}" for k in KEYS))
    print("\nper layer (test - control): k, boxed, " + ", ".join(KEYS) + ", name")
    for k, name in enumerate(names):
        row = {kk: float(t[kk][k] - c[kk][k]) for kk in KEYS}
        out["layers"].append({"layer": k, "boxed": k in boxed, "name": name, "delta": row})
        print(f"  {k:2d} {'B' if k in boxed else '.'} " + " ".join(f"{row[kk]:11,.0f}" for kk in KEYS) + f"  {name}")
    dC, dW = out["sets"]["box"]["delta"]["cycles"], out["sets"]["box"]["delta"]["W_bp"]
    total = sum(out["sets"][s]["delta"]["cycles"] for s in sets)
    share = dC / total if total else float("nan")
    if dC < MIN_LOSS_CYCLES:
        verdict = "no loss located on the box set"
    else:
        verdict = "H-W (weights late)" if dW <= -0.5 * dC else "H-A (activations late)" if dW >= 0.5 * dC else "neither"
    print(f"\n[compare] box set: dC {dC:,.0f} cycles ({dC / CLOCK_GHZ / 1e6:+.3f} ms), dW_bp {dW:,.0f}; "
          f"share of the frame's dC on boxed layers {share:.0%} (total {total / CLOCK_GHZ / 1e6:+.3f} ms)")
    print(f"[compare] rule (dW <= -0.5 dC: H-W; dW >= +0.5 dC: H-A): {verdict}")
    out.update(verdict=verdict, dC_box=dC, dW_box=dW, box_share=share)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
