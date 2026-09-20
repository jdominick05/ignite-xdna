#!/usr/bin/env python3
"""Read graph-engine transaction streams and report how each uses the shim's DMA channels.

    python tools/shim_channel_audit.py --arm "SESR M7 rb=2=build/conv_engine/sesr_m7/insts.bin:2.650" \
        --arm "SESR M7 ring=build/conv_engine/sesr_ring2/insts.bin:5.952" [--measured-gbps 6.898931]

Offline: it opens no device and no hardware context, and reads only the streams the compiler
emitted. It answers a question a dispatch floor cannot, because a floor says how long the
non-compute part of a dispatch took and not whether the wire was busy while it did: how many
shim channels does this container actually program, with how many tasks and how many bytes, and
what fraction of a measured channel rate is that.

``--arm LABEL=PATH[:FLOOR_MS]`` is repeatable; the floor is the measured non-compute floor of one
dispatch for that container, from a device sitting, and is what turns bytes into utilisation.
Omit it for an arm whose floor was never measured, and the tool reports structure only rather than
inventing a denominator.

The stream format is scheduler.py's, decoded by tools/disasm_txn.py. An op's target packs as
(col << 25) | (row << 20) | reg, so the register is masked out before comparing. Shim buffer
descriptors occupy 0x1D000..0x1D200, 8 words apart, so descriptor id = (reg - 0x1D000) / 32, and
word 0 of a descriptor is its length in 32-bit words (docs/SILICON.md 1.3). A task is a WRITE of
that descriptor's id to the channel's start-queue register -- the shim's analogue of the mem-tile
layout engine_sequence.py records (stride 8, S2MM first, START_QUEUE in the word above): S2MM0
0x1D204, S2MM1 0x1D20C, MM2S0 0x1D214, MM2S1 0x1D21C. The pushed value carries the descriptor in
its low 4 bits, the arrival/completion lock above bit 16, and a flag at bit 31. Descriptors are
reused once retired, so a task's bytes come from the descriptor slot as it stands at that point in
the stream. Any push whose descriptor was never written is an error, not a gap: a partial decode
would silently understate an arm's traffic.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.disasm_txn import disassemble_transaction  # noqa: E402

BD_LO, BD_HI = 0x1D000, 0x1D200
PUSH = {0x1D204: "S2MM0", 0x1D20C: "S2MM1", 0x1D214: "MM2S0", 0x1D21C: "MM2S1"}
CHANNELS = ("MM2S0", "MM2S1", "S2MM0", "S2MM1")


def audit(stream_path: Path):
    """``(tasks, bd_writes, tokens, n_ops)``; a task is ``(column, channel, bytes)`` in stream order."""
    ops = disassemble_transaction(stream_path.read_bytes())
    slots, tasks, tokens = defaultdict(dict), [], 0
    bd_writes = 0
    for op in ops:
        if "addr" not in op:
            tokens += op["op"] == "TCT"
            continue
        reg, col = op["addr"] & 0xFFFFF, op["addr"] >> 25
        if op["op"] == "BLOCKWRITE" and BD_LO <= reg < BD_HI:
            slots[col][(reg - BD_LO) // 32] = op["words"][0] * 4
            bd_writes += 1
        elif op["op"] == "WRITE" and reg in PUSH:
            nbytes = slots[col].get(op["val"] & 0xF)
            if nbytes is None:
                raise ValueError(f"{stream_path.name}: push {len(tasks) + 1} on column {col} names "
                                 f"descriptor {op['val'] & 0xF}, which was never written")
            tasks.append((col, PUSH[reg], nbytes))
    return tasks, bd_writes, tokens, len(ops)


def parse_arm(spec: str):
    """``LABEL=PATH[:FLOOR_MS]``. The label may not contain ``=`` -- the spec splits on the first one."""
    label, sep, rest = spec.partition("=")
    path, _, floor = rest.partition(":")
    if not sep or not path:
        raise SystemExit(f"--arm {spec!r} is not LABEL=PATH[:FLOOR_MS] (and LABEL may not contain '=')")
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--arm {spec!r}: no such stream {p}")
    return label, p, (float(floor) if floor else None)


def summarise(label: str, path: Path, floor_ms, measured_gbps, report):
    tasks, bd_writes, tokens, nops = audit(path)
    cols = sorted({c for c, _, _ in tasks})
    by = defaultdict(lambda: [0, 0])
    for col, chan, n in tasks:
        by[(col, chan)][0] += 1
        by[(col, chan)][1] += n
    in_bytes = sum(n for _, c, n in tasks if c.startswith("MM2S"))
    out_bytes = sum(n for _, c, n in tasks if c.startswith("S2MM"))
    total = in_bytes + out_bytes
    sizes = sorted((n for _, _, n in tasks), reverse=True)
    hist = defaultdict(int)
    for _, _, n in tasks:
        hist[n] += 1
    idle = [f"col{c}:{ch}" for c in cols for ch in CHANNELS if (c, ch) not in by]
    print(f"\n[arm] {label}: {path}")
    print(f"[arm] sha256 {hashlib.sha256(path.read_bytes()).hexdigest()} ops={nops} "
          f"BDwrites={bd_writes} tasks={len(tasks)} tokens={tokens} columns={cols}")
    for col in cols:
        print(f"      col{col} " + " ".join(f"{ch}:{by[(col, ch)][0]:4d}t/{by[(col, ch)][1]:>10,d}B"
                                            for ch in CHANNELS if (col, ch) in by))
    print(f"      host->device {in_bytes:,} B over "
          f"{sum(1 for _, c, _ in tasks if c.startswith('MM2S'))} tasks ({in_bytes / total * 100:.0f}%"
          f" of bytes) | device->host {out_bytes:,} B over "
          f"{sum(1 for _, c, _ in tasks if c.startswith('S2MM'))} tasks | total {total:,} B")
    print(f"      avg {total / len(tasks):,.0f} B/task, median {sizes[len(sizes) // 2]:,} B/task, "
          f"biggest {', '.join(f'{n:,}B x{k}' for n, k in sorted(hist.items(), reverse=True)[:5])}")
    print(f"      idle shim channels: {', '.join(idle) if idle else 'none'}")
    util = per_col = wire = None
    if floor_ms:
        per_col = total / len(cols) / (floor_ms * 1e-3) / 1e9
        line = f"      DERIVED at a {floor_ms:.3f} ms floor: {per_col:.2f} GB/s per column"
        if measured_gbps:
            util = per_col / measured_gbps
            wire = total / len(cols) / (measured_gbps * 1e9) * 1e3
            line += (f" = {util * 100:.0f}% of the measured {measured_gbps:.6f} GB/s; these bytes "
                     f"need {wire:.3f} ms per column, so {floor_ms - wire:.3f} ms of the floor is "
                     f"not transfer time")
        print(line + ".")
    report.append({"label": label, "stream": str(path), "tasks": len(tasks), "tokens": tokens,
                   "columns": cols, "bytes_in": in_bytes, "bytes_out": out_bytes,
                   "bytes_total": total, "median_bytes": sizes[len(sizes) // 2],
                   "biggest_bds": {str(n): k for n, k in sorted(hist.items(), reverse=True)[:8]},
                   "per_channel": {f"{c}/{ch}": {"tasks": v[0], "bytes": v[1]}
                                   for (c, ch), v in sorted(by.items())},
                   "idle_channels": idle, "floor_ms": floor_ms,
                   "gbps_per_column": round(per_col, 4) if per_col else None,
                   "utilisation": round(util, 4) if util else None,
                   "transfer_ms_per_column": round(wire, 4) if wire else None})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=PATH[:FLOOR_MS]",
                    help="a stream to audit; repeatable. The floor is that container's measured "
                         "non-compute dispatch floor in ms, from a device sitting.")
    ap.add_argument("--measured-gbps", type=float, default=None, metavar="GBPS",
                    help="measured per-column, per-direction shim rate to compare against "
                         "(e.g. 6.898931, one column full duplex)")
    ap.add_argument("--json", default=None, help="write the same numbers here as JSON")
    args = ap.parse_args()
    if not args.arm:
        ap.error("need at least one --arm LABEL=PATH[:FLOOR_MS]")
    report = []
    for spec in args.arm:
        label, path, floor = parse_arm(spec)
        summarise(label, path, floor, args.measured_gbps, report)
    print("\n[compare] per arm: tasks, completion tokens, bytes, median task, per-column rate, "
          "utilisation against the comparison rate, and the floor time that is not transfer")
    for r in report:
        util = f"{r['utilisation'] * 100:.0f}%" if r["utilisation"] is not None else "-"
        gap = (f"{r['floor_ms'] - r['transfer_ms_per_column']:.3f}ms"
               if r["floor_ms"] and r["transfer_ms_per_column"] else "-")
        gpc = f"{r['gbps_per_column']:.2f}" if r["gbps_per_column"] else "-"
        print(f"[compare] {r['label']:32s} tasks={r['tasks']:5d} tokens={r['tokens']:5d} "
              f"bytes={r['bytes_total']:>11,d} median={r['median_bytes']:>6,d} "
              f"GB/s/col={gpc:>5} utilised={util:>4} not-transfer={gap}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
