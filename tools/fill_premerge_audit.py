#!/usr/bin/env python3
"""Report what the schedule offers the fill merger and what it returns, offline.

    python tools/fill_premerge_audit.py models/sesr_m7_xint8.onnx

The emitted stream says how many shim tasks a container ends up with; it cannot say whether a
chain of fills was ever *offered* to the merger. This runs the real lowering and scheduling path
with `merge_runs` wrapped, so the gap between adjacency and hardware limit becomes visible:

  * the returned pattern count is the container's host->device activation task count. That
    identity is the check that the wrap sees the schedule the compiler really ran -- measured on
    both models here: SESR M7 returns 710, its stream pushes 710 activation tasks; YOLOv8n returns
    2,146, its stream pushes 2,146.
  * patterns that already carry four dimensions are the ones that stand alone. The shim descriptor
    has 3-D addressing plus an iteration modifier (docs/SILICON.md 1.4) and `DmaPattern` accepts at
    most four sizes, so a merge needs a free dimension. "Nothing adjacent matched" is a different
    verdict from this, and it is not what dominates.

No device, no hardware context, no source change: the module is wrapped in this process only.
"""
import argparse
import datetime
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import engine_sequence as es_seq, graph_ir  # noqa: E402

calls = []


def classify(patterns, max_n):
    """Mirror merge_runs' own walk on the CANONICALISED patterns it actually compares.

    merge_runs canonicalises first, so contiguous rows lose a dimension before it looks at shape
    equality; classifying the raw list would call a run mergeable that the merger breaks on a
    canonical shape difference, and would overstate how much it declined.
    """
    patterns = [es_seq.canonical(p) for p in patterns]
    runs, i = [], 0
    while i < len(patterns):
        p0 = patterns[i]
        j, delta, stop = i + 1, None, None
        if len(p0.sizes) > 3:
            stop = "already uses four dimensions"
        else:
            while j < len(patterns) and j - i < max_n:
                p = patterns[j]
                if p.buffer != p0.buffer or p.sizes != p0.sizes or p.strides != p0.strides:
                    stop = "shape or buffer changed"
                    break
                d = p.offset - patterns[j - 1].offset
                if delta is None:
                    if d <= 0:
                        stop = "non-positive step"
                        break
                    if d % 4:
                        stop = "step not word aligned"
                        break
                    delta = d
                elif d != delta:
                    stop = "step changed"
                    break
                j += 1
            else:
                if j - i >= max_n and j < len(patterns):
                    stop = "hit the 64-per-descriptor cap"
        runs.append((j - i, delta, p0.nbytes, stop))
        i = j if j > i else i + 1
    return runs


def wrapped(patterns, max_n=None):
    max_n = es_seq.MAX_REPEAT if max_n is None else max_n
    out = es_seq.merge_runs_orig(patterns, max_n)
    calls.append((list(patterns), list(out), classify(patterns, max_n),
                  [es_seq.canonical(p) for p in patterns]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--tasks", type=int, default=None,
                    help="host->device activation tasks in this container's emitted stream, to check "
                         "the returned-pattern identity (see the module docstring)")
    args = ap.parse_args()

    print("UTC:", datetime.datetime.now(datetime.timezone.utc).isoformat())
    print("MACHINE:", platform.node(), platform.processor())
    print("COMMAND: python tools/fill_premerge_audit.py", args.model)
    print("COMMIT:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())

    es_seq.merge_runs_orig = es_seq.merge_runs
    es.merge_runs = wrapped
    es_seq.merge_runs = wrapped
    ir = graph_ir.lower_yolov8n(args.model)
    ws = es.plan_workspace(ir)
    es.schedule_graph(ir, ws)

    inp = sum(len(a) for a, _, _, _ in calls)
    outp = sum(len(b) for _, b, _, _ in calls)
    print(f"\n[premerge] {args.model}: {len(ir.layers)} layers, {len(calls)} merge_runs calls")
    print(f"[premerge] offered {inp} patterns, returned {outp} -- {inp - outp} fewer, "
          f"{(inp - outp) / max(1, inp) * 100:.0f}% collapsed")
    if args.tasks is not None:
        ok = isclose(outp, args.tasks, rel_tol=0, abs_tol=0)
        print(f"[identity] returned {outp} vs {args.tasks} activation tasks in the emitted stream: "
              f"{'MATCH -- the wrap saw the schedule the compiler ran' if ok else 'MISMATCH'}")

    why = defaultdict(lambda: [0, 0])
    for _, _, runs, _ in calls:
        for n, delta, nbytes, stop in runs:
            key = ("collapsed whole" if stop is None and n > 1 else
                   "lone: nothing adjacent matched" if n == 1 and stop is None else stop)
            why[key][0] += 1
            why[key][1] += n
    for k, (r, n) in sorted(why.items(), key=lambda kv: -kv[1][1]):
        print(f"[runs ] {k:32s}: {r:5d} runs, {n:5d} patterns")

    step = defaultdict(int)
    for _, _, runs, _ in calls:
        for n, delta, nbytes, stop in runs:
            if n > 1 and delta:
                tag = ("overlap (step < payload)" if delta < nbytes else
                       "contiguous (step == payload)" if delta == nbytes else "gap (step > payload)")
                step[tag] += n
    for k, n in sorted(step.items(), key=lambda kv: -kv[1]):
        print(f"[step ] {k}: {n} patterns offered together")

    dims = defaultdict(int)
    dims_raw = defaultdict(int)
    for a, _, _, c in calls:
        for p in a:
            dims_raw[len(p.sizes)] += 1
        for p in c:
            dims[len(p.sizes)] += 1
    print("[dims ] as offered (raw):", ", ".join(f"{k} x{v}" for k, v in sorted(dims_raw.items())))
    print("[dims ] as merge_runs compares them (canonicalised):",
          ", ".join(f"{k} x{v}" for k, v in sorted(dims.items())),
          "-- a pattern with four canonical dimensions is one the merger will not touch")

    # What those four dimensions are spent on. The innermost size is a byte count and its stride is
    # 1, so `sizes[3]` is one row of the packet and `strides[2]` is the pitch between rows; if the
    # pitch exceeds the row, a whole dimension is paying for padding, and the chain ceiling is then
    # sizes[0] - the packets one descriptor can carry - instead of the hardware's 64.
    four = [p for _, _, _, c in calls for p in c if len(p.sizes) == 4]
    shapes = Counter(tuple(p.sizes) for p in four)
    by_shape = {}
    for p in four:
        by_shape.setdefault(tuple(p.sizes), p)
    print(f"[dims4] {len(four)} four-dimensional patterns; "
          f"{sum(1 for p in four if 1 in p.sizes)} carry a degenerate size-1 dimension (a slot the "
          f"merger could use); extents {sorted({p.nbytes for p in four})}")
    for shape, n in shapes.most_common(6):
        p = by_shape[shape]
        row, pitch = shape[3], p.strides[2]
        print(f"[dims4]   {str(shape):16s} x{n:5d} {p.nbytes:>7,} B: row {row} B, pitch {pitch:,} B "
              f"({pitch / row:.2f}x the row), chain step {p.strides[0]:,} B = "
              f"{p.strides[0] / pitch:g} pitches, {shape[0]} packets per descriptor")
    if four:
        contig = sum(1 for p in four if p.strides[2] == p.sizes[3])
        print(f"[dims4] rows already contiguous (pitch == row bytes) in {contig} of {len(four)}; "
              f"every pattern above pays a dimension for row padding, which is why none can chain "
              f"past {max(p.sizes[0] for p in four)} packets")

    declined = sum(n for _, _, runs, _ in calls for n, d, nb, s in runs if s and n > 1)
    print(f"[out  ] {declined} patterns were in a multi-pattern run that the merger declined; "
          f"the four-dimension rule and the 64 cap are the limits it enforces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
