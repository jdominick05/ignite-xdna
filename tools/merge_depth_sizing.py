#!/usr/bin/env python3
"""Size the merge-depth lever: descriptors saved by making a fill contiguous, priced against
the measured floor.

    python tools/merge_depth_sizing.py [--json out.json]

WHY THIS IS THE QUESTION. ``canonical`` folds contiguous dimensions, ``merge_quad`` folds a
quad's four fills into one task with one more dimension, and ``merge_runs`` folds up to 64
consecutive same-shape fills into one more. Both merges REFUSE a pattern that already has
four dimensions, so a fill must be at most two-dimensional after ``canonical`` to get both.

  k3s2 emits (16, 400): two-dimensional, gets both merges, folds 23.6 (yolov8n) to 39.6
  (yolov8s) packets into one DMA task.
  k1 emits (8, 5, 160) at strides (plane_bytes, pitch, 1): the rows fold into the bytes only
  when the pitch is 160, i.e. a 20-pixel-wide map, so on every wider map it stays
  three-dimensional, the quad merge spends the last dimension and merge_runs is refused.
  It folds about 4.7, and it is the largest single kind - 38% of yolov8s's fills.

WHAT IS SIMULATED. Each selected kind's fill is replaced by a CONTIGUOUS run of A_BYTES at
the same offset, and the real scheduler re-merges. That is what the layout change
``tools/fill_layout_sizing.py`` calls "planes packed adjacent" produces: with the plane stride
set to five times the pitch, k1's (8, 5, 160) folds to (40, 160) and then to two dimensions.
Byte counts are unchanged by the substitution, so the saving reported here is descriptors
only, on identical traffic.

WHAT IT COSTS. Packing planes adjacent is byte-free only where a kind's chained windows abut.
``fill_layout_sizing.py`` measured that per shape on yolov8n: k1's windows abut (step 5 rows
>= 5 rows per packet, 1.00x bytes), while k3s1 replicates 8 of 5 stepped rows (1.60x), k1up2
2.00x and pool 3.20x. Those factors are applied below as a transport penalty at the measured
per-column rate. The earlier sizing declined every kind but k1 on those factors, under a cost
model that over-counted transport 3.10x
(results/aie/object_size_repricing_20260920.log); re-priced, the declined half is worth more
than the half proposed.

DERIVED, not measured: no container was built and no hardware context was opened. The
offsets are today's, so this is an upper bound on merging under today's placement rather than
a prediction of a built layout, and it accounts for neither workspace capacity nor the lock
and barrier structure a repack would disturb. The 6,400 B control IS asserted against the DMA
task counts measured on silicon.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ignite_xdna.compiler import engine_emulator as em  # noqa: E402
from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler.engine_sequence import linear  # noqa: E402

import engine_stream_report as esr  # noqa: E402

BASE, BASE_QUAD = es.a_pattern, es.quad_patterns

# results/aie/split_segment_floor_phoenix_20260920.log and object_size_repricing_20260920.log
MEASURED = {"yolov8n": dict(floor_ms=5.510, tasks=2972, shim=35_276_672),
            "yolov8s": dict(floor_ms=13.340, tasks=7143, shim=74_511_488)}
COL_BYTES_PER_S, COLS = 6.898931e9, 4
# bytes a kind costs when its planes are packed adjacent, from tools/fill_layout_sizing.py
REPLICATION = {"k1": 1.00, "res": 1.00, "k3s1": 1.60, "k1up2": 2.00,
               "pool": 3.20, "fused_k3k3": 3.20, "k3s2": 1.00, "k5s1": 1.00}

SCENARIOS = [("control", ()),
             ("k1", ("k1",)),
             ("k1 + res", ("k1", "res")),
             ("k1 + res + k3s1", ("k1", "res", "k3s1")),
             ("every kind", tuple(REPLICATION))]


def install(kinds):
    if not kinds:
        es.a_pattern, es.quad_patterns = BASE, BASE_QUAD
        return
    ks = frozenset(kinds)

    def a_pattern(ws, ir, layer, chunk, y0, x0, group=0):
        p = BASE(ws, ir, layer, chunk, y0, x0, group)
        return linear(p.buffer, p.offset, em.A_BYTES) if chunk.kind in ks else p

    def quad_patterns(ws, ir, layer, chunk, y_quad, x0, group=0, coarse=False):
        qs = BASE_QUAD(ws, ir, layer, chunk, y_quad, x0, group, coarse)
        if chunk.kind not in ks:
            return qs
        return [linear(q.buffer, q.offset, em.A_BYTES) for q in qs]

    es.a_pattern, es.quad_patterns = a_pattern, quad_patterns


def per_kind_packets(model):
    """Packets each kind contributes, tagged at creation. Exact: they sum to the total."""
    install(())
    tally = {}

    def tagged(ws, ir, layer, chunk, y0, x0, group=0):
        p = BASE(ws, ir, layer, chunk, y0, x0, group)
        tally[id(p)] = chunk.kind
        return p
    es.a_pattern = tagged
    es.quad_patterns = lambda ws, ir, layer, chunk, yq, x0, group=0, coarse=False: (
        BASE_QUAD(ws, ir, layer, chunk, yq, x0, group, coarse))
    ir = es.__dict__["GraphIR"] and None
    from ignite_xdna.compiler.graph_ir import lower_yolov8n
    g = lower_yolov8n(model)
    ws = es.plan_workspace(g)
    scheds, _ = es.schedule_graph(g, ws)
    out = {}
    for s in scheds:
        for prog in s.programs:
            for it in prog:
                if it[0] not in ("a", "A"):
                    continue
                pats = list(esr.patterns(it[1:]))
                ks = {tally.get(id(p)) for p in pats}
                k = ks.pop() if len(ks) == 1 else None
                b = sum(p.nbytes for p in pats)
                out[k] = out.get(k, 0) + b // em.A_BYTES
    install(())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    res = {}
    print("[merge] DERIVED over real schedules. No container built, no hardware context opened.")
    for name, m in MEASURED.items():
        model = ROOT / "models" / f"{name}_cut_xint8.onnx"
        install(())
        base = esr.report(model)
        assert base["dma_tasks"] == m["tasks"], (name, base["dma_tasks"], m["tasks"])
        core = sum(v["bytes"] for v in base["totals"].values())
        transfer = m["shim"] / (COLS * COL_BYTES_PER_S) * 1e3
        us = (m["floor_ms"] - transfer) * 1e3 / m["tasks"]
        core_to_shim = core / m["shim"]
        pk = per_kind_packets(model)
        print(f"\n[merge] {name}: control {base['dma_tasks']} tasks matches silicon; "
              f"floor {m['floor_ms']:.3f} = {transfer:.3f} transfer + {m['floor_ms']-transfer:.3f} "
              f"per-task at {us:.4f} us/task; core:shim {core_to_shim:.3f}")
        print(f"[merge] {'scenario':18} {'tasks':>7} {'d tasks':>8} {'d task ms':>10} "
              f"{'d byte ms':>10} {'net ms':>8}")
        rows = []
        for label, kinds in SCENARIOS:
            install(kinds)
            r = esr.report(model)
            d = r["dma_tasks"] - base["dma_tasks"]
            d_task_ms = d * us / 1e3
            extra = sum(pk.get(k, 0) * em.A_BYTES * (REPLICATION[k] - 1.0) for k in kinds)
            d_byte_ms = extra / core_to_shim / (COLS * COL_BYTES_PER_S) * 1e3
            rows.append({"scenario": label, "kinds": list(kinds), "dma_tasks": r["dma_tasks"],
                         "d_tasks": d, "d_task_ms": d_task_ms, "extra_core_bytes": extra,
                         "d_byte_ms": d_byte_ms, "net_ms": d_task_ms + d_byte_ms})
            print(f"[merge] {label:18} {r['dma_tasks']:>7} {d:>+8} {d_task_ms:>+10.3f} "
                  f"{d_byte_ms:>+10.3f} {d_task_ms + d_byte_ms:>+8.3f}")
        install(())
        res[name] = {"control_tasks": base["dma_tasks"], "us_per_task": us,
                     "core_to_shim": core_to_shim, "packets_by_kind": pk, "scenarios": rows}
    if args.json:
        args.json.write_text(json.dumps(res, indent=2, default=str) + "\n",
                             encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())