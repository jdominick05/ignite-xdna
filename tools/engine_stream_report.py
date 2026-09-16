#!/usr/bin/env python3
"""Per-frame shim DMA traffic of a graph-engine schedule, by kind, without a device.

    python tools/engine_stream_report.py models/sesr_m7_xint8.onnx [--json out.json]

Lowers the model and builds the schedule the compiler emits (``engine_schedule.schedule_graph``
with its defaults), then counts the DMA tasks and bytes of every column-program item: weight
fills from the static packet buffer (``w``/``W``), activation fills (``a``/``A``) and output
drains (``o``). Bytes are what the DMA moves in one frame with repeats expanded, so a stride-0
weight repeat counts every object it sends. Task counts follow
``engine_sequence.program_task_count``.

The milliseconds printed for weight traffic are DERIVED, not measured: bytes at the ~26.8 GB/s
fill transport and four instruction ops per task at ~145 ns, both measured on Device 0 in
docs/BENCHMARKS.md ("Graph engine latency from 19.2 to 7.9 ms glass-to-glass").
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ignite_xdna.compiler import engine_emulator as em  # noqa: E402
from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler.engine_sequence import OPS_PER_TASK_ISSUE, DmaPattern  # noqa: E402
from ignite_xdna.compiler.graph_ir import lower_yolov8n  # noqa: E402

KINDS = {"w": "weight", "W": "weight", "a": "activation", "A": "activation", "o": "drain"}
# Arming the MemTile activation ring, not transport: no DMA task and no DDR bytes, so these are skipped
# rather than counted as an unknown kind.
RING_ITEMS = ("R", "S")
A_BYTES = 6400
TRANSPORT_BYTES_PER_S = 26.8e9
SECONDS_PER_OP = 145e-9


def patterns(obj):
    if isinstance(obj, DmaPattern):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from patterns(x)


def empty() -> dict:
    return {k: {"tasks": 0, "bytes": 0} for k in ("weight", "activation", "drain")}


def report(model: Path, activation_ring: int = 0) -> dict:
    ir = lower_yolov8n(model)
    ws = es.plan_workspace(ir)
    scheds, store = es.schedule_graph(ir, ws, activation_ring=activation_ring)
    totals, layers, unknown = empty(), [], set()
    packets = 0
    for s in scheds:
        per = empty()
        for prog in s.programs:
            for it in prog:
                if it[0] in RING_ITEMS:
                    continue
                kind = KINDS.get(it[0])
                if kind is None:
                    unknown.add(it[0])
                    continue
                per[kind]["tasks"] += len(it[1]) if it[0] == "a" else 1
                # A "w" item names its run by offset and length instead of by a pattern, so its bytes are the
                # explicit field; every other kind carries the patterns its task streams.
                per[kind]["bytes"] += it[2] if it[0] == "w" else sum(p.nbytes for p in patterns(it[1:]))
        for k in per:
            for f in ("tasks", "bytes"):
                totals[k][f] += per[k][f]
        packets += s.packets
        layers.append({"layer": s.name, "rounds": s.rounds, "activation_packets": s.packets,
                       "w_fills": s.w_fills, **per})
    tasks = sum(v["tasks"] for v in totals.values())
    w = totals["weight"]
    derived_ms = w["bytes"] / TRANSPORT_BYTES_PER_S * 1e3 + w["tasks"] * OPS_PER_TASK_ISSUE * SECONDS_PER_OP * 1e3
    return {"model": model.name, "activation_ring": activation_ring,
            "layers": len(scheds), "rounds": sum(s.rounds for s in scheds),
            "static_weight_packet_bytes": store.nbytes, "activation_packets": packets,
            # Without a ring every packet a core sees was fetched for it, so the fills are exactly the packets.
            # With one, a tile is fetched once and replayed for its output groups, so the fills are the smaller
            # number and equality is the wrong test: what must hold is that nothing is fetched twice over.
            "activation_bytes_check": (totals["activation"]["bytes"] <= packets * A_BYTES if activation_ring
                                       else totals["activation"]["bytes"] == packets * A_BYTES),
            "activation_packets_per_fetch": packets * A_BYTES / (totals["activation"]["bytes"] or 1),
            "weight_bytes_whole_packets": w["bytes"] % em.W_BYTES == 0,
            "unknown_item_kinds": sorted(unknown), "dma_tasks": tasks, "totals": totals,
            "derived_weight_traffic_ms": derived_ms, "per_layer": layers}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--activation-ring", type=int, default=0,
                    help="schedule with a MemTile activation ring of this many slots (0 = the per-group "
                         "schedule, which is what a container is built with by default)")
    args = ap.parse_args()
    r = report(args.model, args.activation_ring)
    t = r["totals"]
    print(f"[stream] {r['model']}: {r['layers']} layers, {r['rounds']} rounds, {r['dma_tasks']} DMA tasks per frame")
    for k in ("weight", "activation", "drain"):
        print(f"[stream]   {k:10s} {t[k]['tasks']:6d} tasks {t[k]['bytes']:14,d} B")
    print(f"[stream] checks: activation bytes = packets x {A_BYTES}: {r['activation_bytes_check']} | weight bytes in "
          f"whole {em.W_BYTES}-B packets: {r['weight_bytes_whole_packets']} | unknown item kinds: {r['unknown_item_kinds']}")
    print(f"[stream] DERIVED weight traffic: {r['derived_weight_traffic_ms']:.3f} ms per frame "
          f"({t['weight']['bytes']:,} B at {TRANSPORT_BYTES_PER_S / 1e9:.1f} GB/s + {t['weight']['tasks']} tasks x "
          f"{OPS_PER_TASK_ISSUE} ops x {SECONDS_PER_OP * 1e9:.0f} ns)")
    if args.json:
        args.json.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
