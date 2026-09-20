#!/usr/bin/env python3
"""What a different activation object size would cost, priced against the measured floor.

    python tools/activation_object_sweep.py [--sizes 7200,8000,8800,9600] [--json out.json]

The engine's activation packet is ``engine_emulator.A_BYTES`` = 6,400 B. This tool builds the
real schedule at other object sizes and counts the DMA tasks and bytes the compiler would
emit, then prices the difference with constants measured on Device 0. It builds no container
and opens no hardware context; every number it prints for a size other than 6,400 is DERIVED
over a real schedule, not measured.

Feasibility. One ``a_pattern`` is ONE strided BD whose extent must equal the object size
exactly, and the packet carries ``ncin`` planes of ``rows_in x cols_in x 8`` bytes. So for
every chunk kind

    ncin * rows_in * cols_in * 8 == A_BYTES,  rows_in >= rows needed,  cols_in >= cols needed,
    ncin <= weight capacity, from k*k*ncin*256 <= W_MAX_BYTES (9,216): k1 <= 36, k3 <= 4, k5 <= 1

``rows_in`` and ``cols_in`` may exceed what a kind needs - the surplus is junk inside the
plane - so the size is not confined to multiples of the current plane sizes. Two conditions
do bind, and together they leave only multiples of 800:

  - k1 reads the output tile's own 5 x 20 pixels, so 800 B per plane with no junk possible.
  - k3s1 is pinned at ncin = 4 by weight capacity, so A_BYTES / 8 must divide by 4.

The ceiling is core data memory: 6,144 B of headroom at activation depth 2
(results/model_zoo/memory_yolov8s.json) puts A_BYTES at 9,472 B or less. 9,600 is listed
anyway, and needs 256 B freed elsewhere on the tile before it could be built.

Packets are not tasks. ``merge_quad`` folds a quad's four fills into one DMA task and then
keeps folding along whatever BD dimensions are left under the 4-D ceiling, so a kind whose
pattern is 2-D merges far deeper than one that is 3-D. Measured on today's schedule, k3s2
folds 23.6 (yolov8n) to 39.6 (yolov8s) packets into one task while every 3-D kind folds
about 4. Every candidate here therefore keeps k3s2 and k5s1 two-dimensional, at one input
block per packet with junk rows; giving k3s2 a plane dimension instead costs more tasks than
the larger object saves.

The control at 6,400 runs the untouched compiler and is asserted against the DMA task counts
measured on silicon (results/aie/split_segment_floor_phoenix_20260920.log). A candidate's
k1up2 source block is generated as rows of 25 pixels rather than today's 20, which the core's
up2 expansion would have to be taught; the control row is run twice, once with the real up2
shape and once with the generated one, and the difference is reported as the bound that
uncertainty puts on every candidate row.
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
from ignite_xdna.compiler.engine_sequence import DmaPattern  # noqa: E402

import engine_stream_report as esr  # noqa: E402

BASE_PATTERN, BASE_QUAD, BASE_GEOM = es.a_pattern, es.quad_patterns, dict(es.CHUNK_GEOMETRY)

# Measured on Device 0, quiet host, 2026-09-20:
#   floor_ms, dma_tasks   results/aie/split_segment_floor_phoenix_20260920.log
#   shim_bytes            tools/shim_channel_audit.py over the same containers
MEASURED = {
    "yolov8n": {"floor_ms": 5.510, "tasks": 2972, "shim_bytes": 35_276_672},
    "yolov8s": {"floor_ms": 13.340, "tasks": 7143, "shim_bytes": 74_511_488},
}
# Achievable DDR passthrough per column, measured
COL_BYTES_PER_S = 6.898931e9
COLS = 4


def shapes(a):
    """DMA pattern of each chunk kind at object size ``a``; each product must equal ``a``."""
    n, r = a // 800, a // 400
    return {"k1": (n, 5, 160), "k1up2": (n, 4, 200), "k3s1": (4, n, 200),
            "k3s2": (r, 400), "k5s1": (r, 400), "pool": (2, r, 200),
            "res": (n, 5, 160), "fused_k3k3": (2, r, 200)}


def geometry(a):
    """CHUNK_GEOMETRY at object size ``a``: (rows_in, cols_in, plane_bytes, ncin, planes read)."""
    n, r = a // 800, a // 400
    return {"k1": (5, 20, 800, n, n), "k1up2": (5, 20, 800, n, n),
            "k3s1": (n, 25, 200 * n, 4, 4), "k3s2": (r, 50, a, 1, 1), "k5s1": (r, 50, a, 1, 1),
            "pool": (r, 25, a // 2, 2, 2), "res": (5, 20, 800, 4, n),
            "fused_k3k3": (r, 25, a // 2, 2, 2)}


def feasible(a):
    """Why ``a`` cannot be built, or None."""
    if a % 32:
        return f"{a} is not a multiple of 32, so k3s1 cannot keep ncin = 4"
    if a % 800:
        return f"{a} is not a multiple of 800, so k1 would read junk columns"
    for k, s in shapes(a).items():
        p = 1
        for d in s:
            p *= d
        if p != a:
            return f"{k} pattern {s} is {p} B, not {a}"
    if shapes(a)["k3s2"][0] < 11:
        return f"k3s2 needs 11 input rows, {a} gives {shapes(a)['k3s2'][0]}"
    return None


def install(a, up2_real=False):
    """Point the compiler at object size ``a``. ``up2_real`` keeps today's 4 x 20 up2 source."""
    if a == 6400 and up2_real:
        em.A_BYTES, es.CHUNK_GEOMETRY = 6400, dict(BASE_GEOM)
        es.a_pattern, es.quad_patterns, esr.A_BYTES = BASE_PATTERN, BASE_QUAD, 6400
        return
    sh = shapes(a)
    em.A_BYTES, es.CHUNK_GEOMETRY, esr.A_BYTES = a, geometry(a), a

    def a_pattern(ws, ir, layer, chunk, y0, x0, group=0):
        s = sh[chunk.kind]
        if chunk.kind == "res":
            seg = layer.residual
            pl = ws.placements[seg.tensor]
            return DmaPattern("ws", pl.offset(seg.block_offset + group * es.OUT_BLOCKS, y0, x0),
                              s, (pl.plane_bytes, pl.pitch, 1))
        seg = layer.input if isinstance(layer, es.PoolLayer) else layer.inputs[chunk.seg_index]
        pl = ws.placements[seg.tensor]
        b0 = seg.block_offset + chunk.block_start
        if chunk.kind == "pool":
            b0 += group * es.OUT_BLOCKS
        oy, ox = {"k1": (y0, x0), "k1up2": (y0 >> 1, x0 >> 1), "k3s1": (y0 - 1, x0 - 1),
                  "k3s2": (2 * y0 - 1, 2 * x0 - 1), "k5s1": (y0 - 2, x0 - 2),
                  "pool": (y0 - 2, x0 - 2), "fused_k3k3": (y0 - 2, x0 - 2)}[chunk.kind]
        st = (pl.plane_bytes, pl.pitch, 1) if len(s) == 3 else (pl.pitch, 1)
        return DmaPattern("ws", pl.offset(b0, oy, ox), s, st)

    def quad_patterns(ws, ir, layer, chunk, y_quad, x0, group=0, coarse=False):
        # The coarse up2 branch spaces the four cores by two SOURCE rows so their packets merge
        # into one task. Dropping it costs 48 (yolov8n) / 224 (yolov8s) tasks that are an
        # artifact of the patch rather than of the object size.
        if coarse and chunk.kind == "k1up2":
            seg = layer.inputs[chunk.seg_index]
            pl = ws.placements[seg.tensor]
            b0 = seg.block_offset + chunk.block_start
            return [DmaPattern("ws", pl.offset(b0, (y_quad >> 1) + 2 * r, x0 >> 1), sh["k1up2"],
                               (pl.plane_bytes, pl.pitch, 1)) for r in range(es.ROWS)]
        return [a_pattern(ws, ir, layer, chunk, y_quad + es.TILE_R * r, x0, group)
                for r in range(es.ROWS)]

    es.a_pattern, es.quad_patterns = a_pattern, quad_patterns


def count(model, a, up2_real=False):
    install(a, up2_real)
    r = esr.report(model)
    t = r["totals"]
    return {"a_bytes": a, "dma_tasks": r["dma_tasks"],
            "act_tasks": t["activation"]["tasks"], "act_bytes": t["activation"]["bytes"],
            "core_bytes": sum(v["bytes"] for v in t.values())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="7200,8000,8800,9600",
                    help="candidate object sizes in bytes, comma separated")
    ap.add_argument("--models", default="yolov8n,yolov8s")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    out = {"measured": MEASURED, "col_bytes_per_s": COL_BYTES_PER_S, "models": {}}

    print("[sweep] DERIVED over real schedules. No container was built at any size but 6,400.")
    print("[sweep] core data memory caps the object at 9,472 B (6,144 B headroom, depth 2)")
    for a in sizes:
        why = feasible(a)
        if why:
            print(f"[sweep] {a} B INFEASIBLE: {why}")
    sizes = [a for a in sizes if not feasible(a)]

    for name in args.models.split(","):
        model = ROOT / "models" / f"{name}_cut_xint8.onnx"
        base = count(model, 6400, up2_real=True)
        m = MEASURED[name]
        assert base["dma_tasks"] == m["tasks"], (name, base["dma_tasks"], m["tasks"])
        up2_bound = count(model, 6400)["dma_tasks"] - base["dma_tasks"]

        transfer_ms = m["shim_bytes"] / (COLS * COL_BYTES_PER_S) * 1e3
        per_task_ms = m["floor_ms"] - transfer_ms
        us_per_task = per_task_ms * 1e3 / m["tasks"]
        core_to_shim = base["core_bytes"] / m["shim_bytes"]

        print(f"\n[sweep] {name}: control {base['dma_tasks']} DMA tasks matches the measured floor "
              f"({m['floor_ms']:.3f} ms = {transfer_ms:.3f} transfer + {per_task_ms:.3f} per-task, "
              f"{us_per_task:.4f} us/task); core:shim bytes {core_to_shim:.3f}; "
              f"up2 shape costs at most {up2_bound} tasks per candidate row")
        print(f"[sweep] {'A':>6} {'tasks':>7} {'d tasks':>8} {'core B':>14} {'d task ms':>10} "
              f"{'d xfer ms':>10} {'net ms':>8}")
        print(f"[sweep] {6400:>6} {base['dma_tasks']:>7} {0:>8} {base['core_bytes']:>14,} "
              f"{0.0:>10.3f} {0.0:>10.3f} {0.0:>8.3f}")
        rows = []
        for a in sizes:
            c = count(model, a)
            d_tasks = c["dma_tasks"] - base["dma_tasks"]
            d_task_ms = d_tasks * us_per_task / 1e3
            d_shim = (c["core_bytes"] - base["core_bytes"]) / core_to_shim
            d_xfer_ms = d_shim / (COLS * COL_BYTES_PER_S) * 1e3
            net = d_task_ms + d_xfer_ms
            best = net - up2_bound * us_per_task / 1e3
            rows.append({**c, "d_tasks": d_tasks, "d_task_ms": d_task_ms,
                         "d_transfer_ms": d_xfer_ms, "net_ms": net, "net_ms_best_case": best})
            print(f"[sweep] {a:>6} {c['dma_tasks']:>7} {d_tasks:>+8} {c['core_bytes']:>14,} "
                  f"{d_task_ms:>+10.3f} {d_xfer_ms:>+10.3f} {net:>+8.3f}"
                  f"   (best case {best:+.3f})")
        out["models"][name] = {"control": base, "us_per_task": us_per_task,
                               "core_to_shim": core_to_shim, "up2_task_bound": up2_bound,
                               "transfer_ms": transfer_ms, "per_task_ms": per_task_ms,
                               "candidates": rows}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())