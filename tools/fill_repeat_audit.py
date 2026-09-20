#!/usr/bin/env python3
"""Measure where repeated activation fills fall in a dispatch: near, or across the stream.

    python tools/fill_repeat_audit.py build/conv_engine/yolov8n_full/insts.bin [--packet-bytes 6400]

Offline: it decodes the container's emitted transaction stream and opens no device or hardware
context. ``tools/fill_merge_audit.py`` asks whether small fills could have been one repeated
descriptor; this asks a different question about the fills that *are* repeated -- the same
descriptor shape at the same address, sent again inside the same dispatch. Whether that is cheap to
remove depends entirely on how far apart the two sends are: a near repeat is the same window
re-fetched for the next round of the same layer, which a larger window or a retained packet would
cover, while a far repeat crosses into other layers and is only removable by keeping the data
on-chip across a layer boundary.

A descriptor is 8 words; word 0 is the length in 32-bit words, word 1 the address in 32-bit words,
and words 2-7 the address generator's configuration (docs/SILICON.md 1.3-1.4), so a repeat is two
tasks whose control words and address all match. Only host-to-device tasks are considered: a drain
that writes the same address twice is a different question.
"""
import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tools.disasm_txn import disassemble_transaction  # noqa: E402

PUSH = {0x1D204: "S2MM0", 0x1D20C: "S2MM1", 0x1D214: "MM2S0", 0x1D21C: "MM2S1"}
BUCKETS = (("near, <=4 tasks", 4), ("same layer block, <=16", 16), ("far, >16", 1 << 30))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream", nargs="?", default="build/conv_engine/sesr_m7/insts.bin")
    ap.add_argument("--packet-bytes", type=int, default=6400)
    args = ap.parse_args()

    ops = disassemble_transaction(open(args.stream, "rb").read())
    desc, seq = {}, []
    for op in ops:
        if "addr" not in op:
            continue
        reg, col = op["addr"] & 0xFFFFF, op["addr"] >> 25
        if op["op"] == "BLOCKWRITE" and 0x1D000 <= reg < 0x1D200:
            desc[(col, (reg - 0x1D000) // 32)] = list(op["words"])
        elif op["op"] == "WRITE" and reg in PUSH:
            seq.append((col, PUSH[reg], desc[(col, op["val"] & 0xF)]))

    fills = [(i, t) for i, t in enumerate(seq)
             if t[1].startswith("MM2S") and t[2][0] * 4 == args.packet_bytes]
    seen, gaps = {}, []
    for i, (col, chan, d) in fills:
        key = (col, chan, tuple(d[2:8]), d[1])
        if key in seen:
            gaps.append(i - seen[key])
        seen[key] = i

    print(f"[repeat] {args.stream} sha256 {hashlib.sha256(open(args.stream, 'rb').read()).hexdigest()}")
    print(f"[repeat] {len(fills)} host->device tasks carrying a {args.packet_bytes:,} B payload, "
          f"{len(seen)} distinct (column, channel, control words, address), "
          f"{len(gaps)} sent more than once ({len(gaps) / len(fills) * 100:.0f}% of these tasks)")
    hist = defaultdict(int)
    for g in gaps:
        for name, hi in BUCKETS:
            if g <= hi:
                hist[name] += 1
                break
    for name, _ in BUCKETS:
        if hist[name]:
            print(f"[repeat]   {name:24s}: {hist[name]:4d} "
                  f"({hist[name] / max(1, len(gaps)) * 100:.0f}% of repeats)")
    if gaps:
        near = sum(1 for g in gaps if g <= 16)
        print(f"[repeat] mean gap {sum(gaps) / len(gaps):.1f} tasks, median "
              f"{sorted(gaps)[len(gaps) // 2]} tasks; {near} of {len(gaps)} repeats are within 16 "
              f"tasks of their first send, {len(gaps) - near} are further")
    print("[repeat] DERIVED reading: a near repeat is coverable by a wider window or by holding "
          "the packet across rounds; a far repeat needs the data retained across a layer boundary, "
          "which is the transport change, not a packing one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
