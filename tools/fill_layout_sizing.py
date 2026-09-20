#!/usr/bin/env python3
"""Price the layouts that would free a descriptor dimension, using the strides the schedule emits.

    python tools/fill_layout_sizing.py models/sesr_m7_xint8.onnx [--rate-gbps 6.898931]
        [--floor-ms 2.629] [--fill-bytes 11929600] [--fill-tasks 710]

Offline: wraps the real merge_runs to capture the patterns the schedule emits, then does arithmetic
on those measured shapes. No device, no hardware context, no source change.

Why a layout at all: `merge_runs` adds one outermost dimension to a chain, so a pattern already
carrying four cannot chain further than `sizes[0]` -- measured at 4 packets per descriptor on both
models, where the hardware allows 64. A free dimension buys chain length, and chain length is the
only lever found that moves the floor in the right direction (the floor is per-task waiting: 18%
wire utilisation, and 0.599 ms of floor moved on byte-identical traffic).

Each shape is reported as emitted, then under two ways to free a dimension:

  lines   make the packet span the whole padded line (row bytes -> pitch bytes), so the row
          dimension and the byte dimension collapse into one contiguous run
  planes  store a packet's planes adjacently in the workspace (plane stride == rows * pitch), so the
          plane dimension collapses -- the same bytes on the wire, only a placement change

For each it prints the chain ceiling that unlocks, the bytes that must then be moved, and the
per-packet footprint against the 64 KB core data memory -- because the two budgets are different and
a layout that wins one can lose the other outright.
"""
import argparse
import datetime
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import engine_sequence as es_seq, graph_ir  # noqa: E402

CORE_RAM = 65536
captured = []


def wrapped(patterns, max_n=None):
    max_n = es_seq.MAX_REPEAT if max_n is None else max_n
    out = es_seq.merge_runs_orig(patterns, max_n)
    captured.extend(es_seq.canonical(p) for p in patterns)
    return out


def report(p, args):
    """Print one four-dimensional shape as emitted and under each layout that frees a dimension."""
    c, pl, r, b = p.sizes
    step, plane_stride, pitch = p.strides[0], p.strides[1], p.strides[2]
    q = step / pitch
    today_pkt = pl * r * b
    print(f"\n[shape] sizes {tuple(p.sizes)} strides {p.strides}: {pl} planes x {r} rows x {b} B "
          f"= {today_pkt:,} B per packet, {c} packets per descriptor, line pitch {pitch:,} B "
          f"(row is {b / pitch * 100:.1f}% of a line), chain step {q:g} lines")
    rows = [("emitted", None)]
    if plane_stride == r * pitch:
        rows.append(("planes already adjacent", None))
    else:
        rows.append(("planes packed adjacent (plane stride %s -> %s B)"
                     % (f"{plane_stride:,}", f"{r * pitch:,}"), 1))
    rows.append((f"lines spanned (row {b} B -> pitch {pitch:,} B)", 2))
    if plane_stride != r * pitch:
        rows.append(("both: planes packed AND lines spanned", 3))
    for name, freed in rows:
        dims = 4 - (freed or 0)
        chain = es_seq.MAX_REPEAT if dims <= 3 else c
        if freed == 1:                       # planes adjacent: a placement change, priced honestly
            # Chained packets step q lines and span r of them. Where q >= r their sources do not
            # overlap, so packing a packet's planes adjacently moves exactly the same bytes.
            # Where q < r, adjacency replicates the overlapping rows: r/q workspace copies, each of
            # which has to be filled over the wire, so the "free" dimension is not free at all.
            repl = r / q if q and q < r else 1.0
            pkt_bytes = today_pkt * repl
            note = ("same bytes on the wire (chained windows abut or are disjoint: step %g >= %d rows)"
                    % (q, r) if repl == 1.0 else
                    "replicates %d of %g stepped rows: %.2fx the bytes, because chained windows "
                    "overlap" % (r, q, repl))
        elif freed == 2:                     # lines spanned: the padding must actually move
            union = (chain - 1) * q + r      # rows covered by `chain` overlapping packets
            pkt_bytes = pl * union * pitch / chain
            note = (f"delivers {pkt_bytes / today_pkt:.2f}x the useful bytes "
                    f"({union / chain:.2f} of {r} rows per packet after dedup)")
        elif freed == 3:
            union = (chain - 1) * q + r
            pkt_bytes = pl * union * pitch / chain
            note = f"both changes; {pkt_bytes / today_pkt:.2f}x the useful bytes"
        else:
            pkt_bytes, note = today_pkt, "baseline"
        per = pl * r * (pitch if freed in (2, 3) else b)
        fits = "fits" if per <= CORE_RAM else f"** {per:,} B EXCEEDS the 64 KB core ** (one packet " \
            "at a time must land in local memory)"
        if freed in (2, 3):
            dedup = f"{r * chain / union:.2f}x dedup"
        else:
            dedup = "-"
        print(f"  [{name}]")
        print(f"    dims {dims}, chain ceiling {chain} packets/descriptor "
              f"({chain / c:.0f}x fewer descriptors than {c}), per-packet footprint {per:,} B "
              f"{fits}, {note}, chain-overlap dedup {dedup}")
        if args.fill_bytes and args.floor_ms:
            infl = pkt_bytes / today_pkt
            t_bytes = args.fill_bytes / (args.rate_gbps * 1e9) * 1e3
            t_task = args.floor_ms - t_bytes
            new_bytes = t_bytes * infl
            new_task = t_task * (c / chain)
            print(f"    FLOOR MODEL (task cost proportional to descriptor count, bytes priced at "
                  f"{args.rate_gbps} GB/s): today {t_bytes:.3f} transfer + {t_task:.3f} per-task "
                  f"= {args.floor_ms:.3f} ms -> {new_bytes:.3f} + {new_task:.3f} = "
                  f"{new_bytes + new_task:.3f} ms  [upper bound on the gain: ignores lock and "
                  f"barrier cost and any workspace-size effect of the layout itself]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--rate-gbps", type=float, default=6.898931,
                    help="measured per-column, per-direction shim rate")
    ap.add_argument("--floor-ms", type=float, default=None,
                    help="this container's measured non-compute floor, for the FLOOR MODEL line")
    ap.add_argument("--fill-bytes", type=int, default=None,
                    help="host->device activation bytes per dispatch, from the shim channel audit")
    args = ap.parse_args()

    print("UTC:", datetime.datetime.now(datetime.timezone.utc).isoformat())
    print("MACHINE:", platform.node(), platform.processor())
    cmd = ["python", "tools/fill_layout_sizing.py", args.model]
    if args.floor_ms is not None:
        cmd += ["--floor-ms", str(args.floor_ms), "--fill-bytes", str(args.fill_bytes)]
    print("COMMAND:", " ".join(cmd))
    print("COMMIT:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    # Relative, so a committed log never carries a developer profile path -- while still proving
    # which checkout the analysis imported (a worktree or the main one), which is why it is here.
    print("PACKAGE:", os.path.relpath(__import__("ignite_xdna").__file__, ROOT))

    es_seq.merge_runs_orig = es_seq.merge_runs
    es.merge_runs = wrapped
    es_seq.merge_runs = wrapped
    ir = graph_ir.lower_yolov8n(args.model)
    ws = es.plan_workspace(ir)
    es.schedule_graph(ir, ws)

    four = [p for p in captured if len(p.sizes) == 4]
    shapes = Counter(tuple(p.sizes) for p in four)
    by_shape = {}
    for p in four:
        by_shape.setdefault(tuple(p.sizes), p)
    print(f"\n[sizing] {args.model}: {len(captured)} patterns offered to merge_runs, "
          f"{len(four)} four-dimensional across {len(shapes)} shapes "
          f"(cross-check against tools/fill_premerge_audit.py's count for this model)")
    for shape, n in shapes.most_common():
        print(f"[count] {shape} x{n}")
        if n >= 20:
            report(by_shape[shape], args)

    # Aggregate sizing, across every four-dimensional shape. Each such pattern is one descriptor
    # chaining c packets today; the byte-free class (chained windows abut: rows == step in lines)
    # can re-chain to MAX_REPEAT at no wire cost, the rest would pay r/q copies to do so.
    free = repl = 0
    free_pkts = repl_pkts = 0
    for p in four:
        c, pl, r, b = p.sizes
        q = p.strides[0] / p.strides[2]
        if q and q >= r:
            free += 1
            free_pkts += c
        else:
            repl += 1
            repl_pkts += c
    # Estimate, and the assumption is named in the print: packets re-chain within a stream, so the
    # byte-free class needs ceil(free_pkts / MAX_REPEAT) descriptors instead of `free`.
    est = -(-free_pkts // es_seq.MAX_REPEAT)
    print(f"\n[sizsum] {len(four)} four-dimensional descriptors today ({free + repl} patterns, "
          f"{free_pkts + repl_pkts} packets)")
    print(f"[sizsum]   byte-free to re-chain (windows abut): {free} = "
          f"{free / len(four) * 100:.0f}% of them, {free_pkts} packets")
    print(f"[sizsum]   would replicate overlapping rows (1.0-3.2x bytes): {repl} = "
          f"{repl / len(four) * 100:.0f}%, {repl_pkts} packets -> not proposed")
    print(f"[sizsum] ESTIMATE assuming byte-free packets re-chain within a stream: "
          f"{free} descriptors -> {est}, i.e. {free - est} of the stream's tasks; every figure "
          f"above is arithmetic on the emitted strides, and no timing was taken")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
