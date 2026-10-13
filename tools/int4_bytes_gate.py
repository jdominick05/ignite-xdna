#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate B of the INT4 study: the most int4 weights could save the graph engine, per frame.

    python tools/int4_bytes_gate.py [--log results/aie/<name>.log]

No device, no hardware context. Rebuilds each model's default schedule (the one
`ignite-compile --engine graph` emits: activation ring off; the CLI exposes no ring)
through `tools/engine_stream_report.py`'s own traversal, then walks every weight packet
streamed in one frame and prices three changes. All the milliseconds are DERIVED:

  trim   send each packet as its used region, not a fixed 9,472 B object:
         128 B header + 128 B bias + the weight layout the header declares
         (k*k*ncin*4*64 B for a conv; the trailing non-zero extent otherwise)
  int4   halve every conv packet's weight layout. A fixed-size packet saves nothing by
         construction, so int4 presupposes variable-size packets; its saving is the same
         with or without the trim.
  int4+  int4 plus 128 B per conv packet (half its header and bias, as if int4 let two
         packets merge); the CEILING adds half the weight DMA tasks' issue time on top
         (engine_stream_report's 4 ops x 145 ns per task) -- generous, not a design

Bytes become milliseconds at the ~26.8 GB/s fill transport docs/BENCHMARKS.md measured
(`engine_stream_report.TRANSPORT_BYTES_PER_S`), as if the frame were entirely bound by
that transport. That is the best case for a byte saving: if the floor is not purely
bandwidth-bound, the real saving is smaller. The denominator is each model's SMALLEST
measured NPU dispatch in the docs (cited per row), which also favours int4.

The gate (pre-registered in the study plan and printed in the log before the numbers):
int4 is killed for the engine if its bound is below 5 % of the measured dispatch for every
model; above 5 % for any model means "not killed by bytes, unresolved". It is asymmetric
because the bound is a best case.

Trust order: the YOLOv8n and YOLOv8s weight bytes per frame must equal commit 5162671's
corrected figures (26,303,744 and 83,618,816 B) and this walk must equal
`engine_stream_report.report`'s total for every model, or no row is printed as a result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ignite_xdna  # noqa: E402
import engine_stream_report as esr  # noqa: E402
from ignite_xdna.compiler import engine_emulator as em  # noqa: E402
from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler.graph_ir import lower_yolov8n  # noqa: E402
from ignite_xdna.compiler.serializer import HEADER_SIZE, IgniteHeader  # noqa: E402

HDR_BIAS = em.W_OFFSET  # 256: 128 B header + 128 B int32 bias
GATE_PCT = 5.0

# name, ONNX, shipped container, smallest measured NPU dispatch (ms), its source
MODELS = [
    ("YOLOv8n", "models/yolov8n_cut_xint8.onnx", "build/yolov8n_full.ignite", 7.161,
     "docs/MODEL_ZOO_BENCHMARKS.md:297 (live_ignition dispatch)"),
    ("YOLOv8s", "models/yolov8s_cut_xint8.onnx", "build/yolov8s.ignite", 16.609,
     "docs/MODEL_ZOO_BENCHMARKS.md:298 (live_ignition dispatch)"),
    ("YOLOv8n-pose", "models/yolov8n-pose_cut_xint8.onnx", "build/yolov8n_pose.ignite", 7.548,
     "docs/MODEL_ZOO_BENCHMARKS.md:388 (NPU dispatch)"),
    ("SESR-M7", "models/sesr_m7_xint8.onnx", "build/sesr_m7.ignite", 4.208,
     "docs/MODEL_ZOO_BENCHMARKS.md:186 (dispatch)"),
    ("resnet50_head", "build/test_resnet50_head.onnx", "build/resnet50_head.ignite", 1.142,
     "docs/BENCHMARKS.md:10125 (NPU dispatch)"),
]
PINNED = {"YOLOv8n": 26_303_744, "YOLOv8s": 83_618_816}  # commit 5162671's body


def pattern_runs(p):
    """(offset, nbytes) of each contiguous innermost run of a DmaPattern."""
    outer, ostr = p.sizes[:-1], p.strides[:-1]
    for idx in itertools.product(*[range(n) for n in outer]):
        yield p.offset + sum(i * s for i, s in zip(idx, ostr)), p.sizes[-1]


def packet_cost(pkt: np.ndarray) -> dict:
    hdr = em.PacketHeader.from_words(pkt[:em.HDR_BYTES].view(np.int32))
    nz = np.flatnonzero(pkt)
    tail = int(nz[-1]) + 1 if nz.size else 0
    if hdr.op == em.OP_CONV:
        layout = hdr.k * hdr.k * hdr.ncin * em.OUT_BLOCKS * 64
        return dict(conv=True, layout=layout, overrun=tail > HDR_BIAS + layout,
                    nonzero_w=int(np.count_nonzero(pkt[HDR_BIAS:HDR_BIAS + layout])))
    payload = max(0, -(-(tail - HDR_BIAS) // 4) * 4) if tail > HDR_BIAS else 0
    return dict(conv=False, layout=payload, overrun=False, nonzero_w=0)


def walk(onnx: Path) -> dict:
    ir = lower_yolov8n(onnx)
    ws = es.plan_workspace(ir)
    scheds, store = es.schedule_graph(ir, ws, activation_ring=0)
    n = dict(sent=0, conv_sent=0, bytes=0, layout=0, conv_layout=0, nonzero_w=0, overruns=0,
             unaligned=0)
    cache = {}
    for s in scheds:
        for prog in s.programs:
            for it in prog:
                if it[0] in esr.RING_ITEMS or esr.KINDS.get(it[0]) != "weight":
                    continue
                runs = [(it[1], it[2])] if it[0] == "w" else \
                    [r for p in esr.patterns(it[1:]) for r in pattern_runs(p)]
                for off, nb in runs:
                    if off % em.W_BYTES or nb % em.W_BYTES:
                        n["unaligned"] += 1
                    for k in range(nb // em.W_BYTES):
                        o = off + k * em.W_BYTES
                        if o not in cache:
                            cache[o] = packet_cost(store.packet_at(o))
                        c = cache[o]
                        n["sent"] += 1
                        n["bytes"] += em.W_BYTES
                        n["layout"] += c["layout"]
                        n["overruns"] += c["overrun"]
                        if c["conv"]:
                            n["conv_sent"] += 1
                            n["conv_layout"] += c["layout"]
                            n["nonzero_w"] += c["nonzero_w"]
    n["store_bytes"] = store.nbytes
    n["w_fills"] = sum(s.w_fills for s in scheds)
    n["unique_packets"] = len(cache)
    return n


def ceiling_ms(int4p_bytes: int, w_tasks: int) -> float:
    """int4 + merged headers, plus half the weight tasks' issue time (if merging halved them),
    priced as engine_stream_report prices task issue."""
    return (int4p_bytes / esr.TRANSPORT_BYTES_PER_S
            + 0.5 * w_tasks * esr.OPS_PER_TASK_ISSUE * esr.SECONDS_PER_OP) * 1e3


def manifest(path: Path) -> dict:
    with open(path, "rb") as f:
        h = IgniteHeader.unpack(f.read(HEADER_SIZE))
        f.seek(h.manifest_offset)
        return json.loads(f.read(h.manifest_size).decode("utf-8").rstrip("\0 \n"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", type=Path, default=None, help="also write the report here (UTF-8)")
    args = ap.parse_args()

    here = Path(ignite_xdna.__file__).resolve()
    if ROOT not in here.parents:
        raise SystemExit(f"ignite_xdna resolved to {here}, not this checkout's src/ -- refusing")
    bw = esr.TRANSPORT_BYTES_PER_S
    ms = lambda b: b / bw * 1e3  # noqa: E731

    out = ["INT4 study, gate B: the most int4 weights could save the graph engine per frame (DERIVED)",
           f"UTC: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
           f"MACHINE: {platform.node()}"]
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out += [f"COMMIT: {commit} (worktree-int4-study)",
            "COMMAND: python tools/int4_bytes_gate.py" + (f" --log {args.log.as_posix()}" if args.log else ""),
            f"ignite_xdna: {here.relative_to(ROOT).as_posix()} (this checkout's src/, asserted)",
            f"SCOPE: no device. Default schedule (activation ring off). Bytes -> ms at {bw / 1e9:.1f} GB/s,",
            "  as if the frame were purely transport-bound: every ms below is a DERIVED best case.",
            "",
            "PRE-REGISTERED GATE (from the study plan, before this run):",
            f"  int4-in-engine is KILLED if its bound is < {GATE_PCT:.0f}% of the smallest measured dispatch for",
            "  EVERY model; >= 5% for any model -> 'not killed by bytes, unresolved' (needs a src/ change).",
            f"  5% = the drift this repo found inseparable from zero (H12, 5.0% floor drift).",
            "  Expected before the run: YOLOv8n ~0.19 ms / ~2.5% with trim; YOLOv8s might exceed 5%.",
            ""]

    rows, ok = [], True
    for name, onnx, cont, disp, src in MODELS:
        r = walk(ROOT / onnx)
        rep = esr.report(ROOT / onnx, 0)["totals"]["weight"]
        ref, r["w_tasks"] = rep["bytes"], rep["tasks"]
        checks = [f"walk == engine_stream_report ({ref:,} B): {r['bytes'] == ref}"]
        good = r["bytes"] == ref and r["unaligned"] == 0 and r["overruns"] == 0
        if name in PINNED:
            checks.append(f"== 5162671's {PINNED[name]:,} B: {r['bytes'] == PINNED[name]}")
            good &= r["bytes"] == PINNED[name]
        try:
            m = manifest(ROOT / cont).get("graph_engine", {})
            mw, mf = m.get("wpackets_bytes"), m.get("weight_fills")
            checks.append(f"shipped {Path(cont).name}: wpackets_bytes {mw} vs {r['store_bytes']}, "
                          f"weight_fills {mf} vs {r['w_fills']}"
                          + (" (match)" if (mw, mf) == (r["store_bytes"], r["w_fills"]) else " (DIFFERS)"))
        except (OSError, ValueError) as e:
            checks.append(f"shipped {cont}: unreadable ({e})")
        ok &= good
        trim = r["bytes"] - (r["sent"] * HDR_BIAS + r["layout"])
        int4 = r["conv_layout"] // 2
        int4p = int4 + 128 * r["conv_sent"]
        nz_floor = r["bytes"] - (r["sent"] * HDR_BIAS + r["nonzero_w"] + (r["layout"] - r["conv_layout"]))
        rows.append((name, disp, src, r, trim, int4, int4p, nz_floor, checks, good))

    for name, disp, src, r, trim, int4, int4p, nz_floor, checks, good in rows:
        out.append(f"{name}  ({r['sent']:,} weight packets sent per frame, {r['conv_sent']:,} conv; "
                   f"{r['unique_packets']:,} unique; dispatch {disp} ms, {src})")
        for c in checks:
            out.append(f"  check: {c}")
        out.append(f"  check: unaligned runs {r['unaligned']}, packets whose data overruns their declared layout {r['overruns']}")
        if not good:
            out.append("  CHECKS FAILED -- no result for this row")
            out.append("")
            continue
        w = r["bytes"]
        out.append(f"  weight bytes streamed / frame     {w:>14,} B  {ms(w):7.3f} ms")
        out.append(f"  of which declared layout (conv)   {r['conv_layout']:>14,} B  fill {r['conv_layout'] / w:6.1%}")
        out.append(f"  of which non-zero weight bytes    {r['nonzero_w']:>14,} B  fill {r['nonzero_w'] / w:6.1%}  (a floor on real weights)")
        for label, b in (("trim to declared layout", trim), ("int4 (variable-size packets)", int4),
                         ("int4 + merged headers", int4p), ("trim + int4", trim + int4),
                         ("trim to non-zero bytes (floor)", nz_floor)):
            out.append(f"  save: {label:<31} {b:>12,} B  {ms(b):7.3f} ms  {ms(b) / disp:6.1%} of dispatch")
        cm = ceiling_ms(int4p, r["w_tasks"])
        out.append(f"  save: int4 CEILING (merged headers + half the {r['w_tasks']} weight tasks' issue time)"
                   f"  {cm:7.3f} ms  {cm / disp:6.1%} of dispatch")
        out.append("")

    if not ok:
        out.append("VERDICT: NONE -- a trust check failed (see CHECKS FAILED above).")
    else:
        pct = {n: ms(i4) / d * 100 for n, d, _, _, _, i4, _, _, _, _ in rows}
        pctp = {n: ceiling_ms(i4p, r["w_tasks"]) / d * 100 for n, d, _, r, _, _, i4p, _, _, _ in rows}
        over = [n for n in pct if pct[n] >= GATE_PCT]
        overp = [n for n in pctp if pctp[n] >= GATE_PCT]
        out.append("int4 bound as % of smallest measured dispatch: " +
                   ", ".join(f"{n} {pct[n]:.1f}%" for n in pct))
        out.append("int4 CEILING as % of the same:                " +
                   ", ".join(f"{n} {pctp[n]:.1f}%" for n in pctp))
        if not over and not overp:
            top = max(pctp, key=pctp.get)
            out.append(f"VERDICT: int4-in-engine KILLED at gate B -- every model's best case is under {GATE_PCT:.0f}%.")
            out.append(f"  Narrowest: {top}'s CEILING at {pctp[top]:.1f}% ({GATE_PCT - pctp[top]:.1f} points under the line);"
                       f" its plain int4 bound is {pct[top]:.1f}%.")
        else:
            out.append(f"VERDICT: NOT KILLED BY BYTES, UNRESOLVED for {sorted(set(over) | set(overp))} "
                       f"(best case >= {GATE_PCT:.0f}%); settling it needs a src/ change (excluded).")
        out.append("In every row int4 presupposes the variable-size packets that the trim alone would use;")
        out.append("compare 'trim to declared layout' (no accuracy cost) against 'int4' for the lever order.")
        out.append("The CEILING prices what int4 could enable beyond halving bytes as two int4 packets in one")
        out.append("9,472 B object and half the weight tasks gone. Not priced: any change to activation traffic,")
        out.append("which int4 weights do not touch, and any accuracy cost, which this gate does not reach.")
    text = "\n".join(out)
    print(text)
    if args.log:
        if args.log.exists():
            raise SystemExit(f"refusing to overwrite existing log {args.log}")
        args.log.write_text(text + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
