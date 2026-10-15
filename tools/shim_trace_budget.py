#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-layer shim budget of a traced graph-engine container: where each stream's time goes on the traced column.

    python tools/shim_trace_budget.py --build build/conv_engine/yolov8s_tr_ctl \\
        --trace results/aie/shim_trace_box_yolov8s_desktop2_20261015T1854Z_ctl.json \\
        [--layer-timing a.json [b.json ...]] [--rate-gbps 6.79] [--json out.json]

The shim trace (``tools/engine_shim_trace.py --json``) gives, per layer window, the cycles each shim channel spent
blocked: MM2S 1 (weights, broadcast to the column's four cores) in stream backpressure or DDR starvation, MM2S 0
(activations, to the MemTile) the same, and S2MM 0 (the output join) starved. It records no "moving" event, so this
tool adds the bytes: it disassembles the build's ``insts.bin``, splits it at the per-layer trace markers, and sums
for each layer and channel of the traced column the bytes its task pushes move (BD buffer length x (repeat + 1)).
Bytes over a stream rate give each channel's moving time; what is left of the window is residual:
    residual = window - backpressure - DDR starvation - moving
On a channel with one task per layer (the weights) the residual should be near zero, which checks the rate and the
trace together. On the activations the residual is time the channel had no task to run (or ran slower than the rate).
What the shim cannot see: the MemTile-to-core split of the activations, the core DMA and lock handshakes, the cores'
compute. Weight backpressure means at least ONE of the four cores' weight buffers is full (the slowest gates the
broadcast); it does not say the cores are computing.

``--layer-timing`` joins ``tools/engine_layer_timing.py --nop --json`` files (warm, per-layer real and NOP medians):
per layer, compute share = (real - NOP) / real. The join is refused unless the layer names match.

Every traced number is a context's first, cold dispatch, on logical column 0 only; ms are DERIVED at CLOCK_GHZ (the
trace timer's rate is assumed). The default rate, 6.79 GB/s, is the MEASURED all-four-columns one-stream route of
results/aie/stream_ceiling_phoenix_desktop2_20261013T060213Z.log (27.17 GB/s / 4), rather than the one-column 6.88,
because all four columns stream at once in the engine. Backpressure and starvation are measured cycles; moving time,
and so the residual, are DERIVED from both the rate and the assumed clock. Offline; opens no hardware context.
Written on Desktop 2, 2026-10-15.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from disasm_txn import disassemble_transaction  # noqa: E402

CLOCK_GHZ = 1.80
SHIM_BD0, SHIM_BD_STRIDE, SHIM_BDS = 0x1D000, 0x20, 16
QUEUES = {0x1D204: "o", 0x1D214: "a", 0x1D21C: "w"}   # S2MM 0, MM2S 0, MM2S 1 task queues (S2MM 1 is the trace)
EVENT_GENERATE = 0x34008
EV = {"W_bp": "DMA_MM2S_1_STREAM_BACKPRESSURE", "W_mem": "DMA_MM2S_1_MEMORY_STARVATION",
      "A_bp": "DMA_MM2S_0_STREAM_BACKPRESSURE", "A_mem": "DMA_MM2S_0_MEMORY_STARVATION",
      "O_starve": "DMA_S2MM_0_STREAM_STARVATION"}


def layer_bytes(insts: bytes, col: int = 0):
    """[{channel: (pushes, bytes)}] per layer of one column, split at that column's shim Event_Generate writes."""
    bd = {}
    segs, cur = [], {}
    for op in disassemble_transaction(insts):
        if "addr" not in op:
            continue
        a = op["addr"]
        c, row, reg = (a >> 25) & 0x7F, (a >> 20) & 0x1F, a & 0xFFFFF
        if c != col or row != 0:
            continue
        if op["op"] == "BLOCKWRITE" and SHIM_BD0 <= reg < SHIM_BD0 + SHIM_BDS * SHIM_BD_STRIDE \
                and (reg - SHIM_BD0) % SHIM_BD_STRIDE == 0:
            bd[(reg - SHIM_BD0) // SHIM_BD_STRIDE] = op["words"]
        elif op["op"] == "WRITE" and reg == EVENT_GENERATE:
            segs.append(cur)
            cur = {}
        elif op["op"] == "WRITE" and reg in QUEUES:
            v = op["val"]
            words = bd[v & 0xF]
            n = words[0] * 4 * (((v >> 16) & 0xFF) + 1)
            p, b = cur.get(QUEUES[reg], (0, 0))
            cur[QUEUES[reg]] = (p + 1, b + n)
    segs.append(cur)
    # The setup's start fire and the flush tail leave segments with no task of these channels. Only leading and
    # trailing ones may be empty: an empty one between layers would shift every later layer's bytes by one.
    kept = [k for k, s in enumerate(segs) if s]
    if kept and kept != list(range(kept[0], kept[-1] + 1)):
        raise SystemExit("a layer between two others has no task on this column; the split cannot be aligned")
    return [segs[k] for k in kept]


def load_trace(paths):
    runs, names = [], None
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        names = names or d["layers"]
        if d["layers"] != names:
            raise SystemExit(f"{p}: a different layer list")
        runs += d["runs"]
    keys = ["cycles", "lead"] + list(EV.values())
    per = {k: [] for k in keys}
    for r in runs:
        col = sorted(r["shims"])[0]
        for k in keys:
            per[k].append([L[k] for L in r["shims"][col]["layers"]])
    return {k: np.array(v, dtype=float) for k, v in per.items()}, names, len(runs)


def load_timing(paths, names):
    real, nop = [], []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for arm, out in (("real", real), ("nop", nop)):
            rows = d["arms"][arm]["layers"]
            if [r["name"] for r in rows] != names:
                raise SystemExit(f"{p} {arm}: layer names differ from the trace's")
            out.append([r["net_ms"] for r in rows])
    return np.mean(real, axis=0), np.mean(nop, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", type=Path, required=True, help="the traced container's build dir (holds insts.bin)")
    ap.add_argument("--trace", nargs="+", required=True)
    ap.add_argument("--layer-timing", nargs="*", default=[])
    ap.add_argument("--rate-gbps", type=float, default=27.17 / 4)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    t, names, nruns = load_trace(a.trace)
    lb = layer_bytes((a.build / "insts.bin").read_bytes())
    if len(lb) != len(names):
        raise SystemExit(f"insts.bin splits into {len(lb)} layers with tasks, the trace holds {len(names)}")
    cyc_per_byte = CLOCK_GHZ / a.rate_gbps
    mean = {k: v.mean(axis=0) for k, v in t.items()}
    win = mean["cycles"]
    rows = []
    for k, name in enumerate(names):
        r = {"layer": k, "name": name, "window": float(win[k]), "lead": float(mean["lead"][k])}
        for ch in "wao":
            pushes, nbytes = lb[k].get(ch, (0, 0))
            r[f"{ch}_tasks"], r[f"{ch}_bytes"] = pushes, nbytes
        for key, ev in EV.items():
            r[key] = float(mean[ev][k])
        r["W_move"] = r["w_bytes"] * cyc_per_byte
        r["A_move"] = r["a_bytes"] * cyc_per_byte
        r["W_resid"] = r["window"] - r["W_bp"] - r["W_mem"] - r["W_move"]
        r["A_resid"] = r["window"] - r["A_bp"] - r["A_mem"] - r["A_move"]
        rows.append(r)
    real = nop = None
    if a.layer_timing:
        real, nop = load_timing(a.layer_timing, names)
        for r, re_, no in zip(rows, real, nop):
            r["real_ms"], r["nop_ms"] = float(re_), float(no)
            r["compute_share"] = float((re_ - no) / re_) if re_ > 0 else float("nan")

    def ms(x):
        return x / CLOCK_GHZ / 1e6

    print(f"[budget] {nruns} traced runs (cold first dispatches), logical column 0; rate {a.rate_gbps:.2f} GB/s "
          f"= {1 / cyc_per_byte:.3f} B/cycle at {CLOCK_GHZ} GHz (DERIVED); ms DERIVED at {CLOCK_GHZ} GHz")
    tot = {k: sum(r[k] for r in rows) for k in ("window", "W_bp", "W_mem", "W_move", "W_resid", "A_bp", "A_mem",
                                                 "A_move", "A_resid", "O_starve", "lead")}
    tb = {ch: sum(r[f"{ch}_bytes"] for r in rows) for ch in "wao"}
    tp = {ch: sum(r[f"{ch}_tasks"] for r in rows) for ch in "wao"}
    print(f"[budget] column 0 per frame: weights {tp['w']} tasks {tb['w']:,} B; activations {tp['a']} tasks "
          f"{tb['a']:,} B; output {tp['o']} tasks {tb['o']:,} B")
    print(f"[budget] frame window {tot['window']:,.0f} cycles = {ms(tot['window']):.3f} ms")
    for s in ("W", "A"):
        parts = [f"{p} {tot[f'{s}_{p}']:,.0f} ({tot[f'{s}_{p}'] / tot['window']:.0%})" for p in ("bp", "mem", "move", "resid")]
        print(f"[budget] {'weights    ' if s == 'W' else 'activations'}: " + "; ".join(parts))
    print(f"[budget] output starvation {tot['O_starve']:,.0f} ({tot['O_starve'] / tot['window']:.0%}); "
          f"lead {tot['lead']:,.0f} ({tot['lead'] / tot['window']:.0%})")
    hdr = (f"\n{'k':>2} {'window':>9} {'Wt':>3} {'W_bp':>5} {'W_mem':>5} {'W_mv':>5} {'W_res':>6} {'At':>4} "
           f"{'A_bp':>5} {'A_mem':>5} {'A_mv':>5} {'A_res':>6} {'O_st':>5}")
    if real is not None:
        hdr += f" {'real':>6} {'nop':>6} {'comp':>5}"
    print(hdr + "   (fractions of the layer window; window in cycles; real/nop warm ms)")
    for r in rows:
        w = r["window"]
        line = (f"{r['layer']:2d} {w:9,.0f} {r['w_tasks']:3d} {r['W_bp'] / w:5.0%} {r['W_mem'] / w:5.0%} "
                f"{r['W_move'] / w:5.0%} {r['W_resid'] / w:6.0%} {r['a_tasks']:4d} {r['A_bp'] / w:5.0%} "
                f"{r['A_mem'] / w:5.0%} {r['A_move'] / w:5.0%} {r['A_resid'] / w:6.0%} {r['O_starve'] / w:5.0%}")
        if real is not None:
            line += f" {r['real_ms']:6.3f} {r['nop_ms']:6.3f} {r['compute_share']:5.0%}"
        print(line + f"  {r['name']}")
    out = {"runs": nruns, "rate_gbps": a.rate_gbps, "clock_ghz": CLOCK_GHZ, "totals": tot, "bytes": tb,
           "tasks": tp, "layers": rows}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
