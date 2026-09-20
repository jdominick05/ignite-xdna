"""Classify every small activation fill in a stream by whether merging could collapse it.

A shim descriptor is 8 words: word 0 is the length in 32-bit words and word 1 the address in
32-bit words (docs/SILICON.md 1.3-1.4); words 2-7 carry the address generator's per-dimension
strides and sizes plus the lock pair. Grouping tasks by everything except length and address finds
the tasks that are the same shape of transfer at different offsets. Within each group the source
addresses are cut into maximal constant-step runs, and each run is classified by its step against
the payload:

  overlap    step < payload    the windows cover overlapping source ranges (this tool does NOT
                               decide whether the repeat dimension could express such a chain)
  contiguous step == payload   a plain larger memcpy
  gap        step > payload    a regular chain, repeatable while the step fits the 20-bit field

WHAT AN ARTIFACT CANNOT SAY. These counts show descriptors that were pushed, never what the merger
was offered. ``tools/fill_premerge_audit.py`` wraps the real ``merge_runs`` and measures that
instead, and its verdict supersedes any inference drawn here about a missed merge: on both models
tested the merger already collapses every adjacent chain it is offered, and the residue is the
descriptor's four-dimension limit. Read it before quoting a saving from this tool. What this file
does establish is the redundancy: bytes delivered against the union of the windows read, which is
the over-read the packets exist to pay for.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tools.disasm_txn import disassemble_transaction  # noqa: E402

PUSH = {0x1D204: "S2MM0", 0x1D20C: "S2MM1", 0x1D214: "MM2S0", 0x1D21C: "MM2S1"}
MAX_STEP_BYTES = (2 ** 20 - 1) * 4      # shim BD step field, 20 bits of 32-bit words
MAX_REPEAT = 64                          # engine_sequence.MAX_REPEAT on Phoenix


def classify(stream: Path, small: int):
    ops = disassemble_transaction(stream.read_bytes())
    desc, queue = {}, defaultdict(list)
    for op in ops:
        if "addr" not in op:
            continue
        reg, col = op["addr"] & 0xFFFFF, op["addr"] >> 25
        if op["op"] == "BLOCKWRITE" and 0x1D000 <= reg < 0x1D200:
            desc[(col, (reg - 0x1D000) // 32)] = list(op["words"])
        elif op["op"] == "WRITE" and reg in PUSH:
            queue[(col, PUSH[reg])].append(desc[(col, op["val"] & 0xF)])

    groups = defaultdict(list)
    for (col, chan), ds in queue.items():
        for d in ds:
            if d[0] * 4 == small:
                groups[(col, chan, tuple(d[2:8]))].append(d[1] * 4)

    runs = []
    for key, addrs in groups.items():
        uniq = sorted(set(addrs))
        i = 0
        while i < len(uniq):
            if i + 1 >= len(uniq):
                runs.append((key, 1, None))
                break
            j = i + 1
            while j + 1 < len(uniq) and uniq[j + 1] - uniq[j] == uniq[i + 1] - uniq[i]:
                j += 1
            runs.append((key, j - i + 1, uniq[i + 1] - uniq[i]))
            i = j + 1
    dupes = sum(len(v) - len(set(v)) for v in groups.values())
    return runs, groups, dupes, sum(len(ds) for ds in queue.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream", nargs="?", default="build/conv_engine/sesr_m7/insts.bin")
    ap.add_argument("--packet-bytes", type=int, default=6400,
                    help="the payload size to classify (default: the engine's activation packet)")
    args = ap.parse_args()

    runs, groups, dupes, total = classify(Path(args.stream), args.packet_bytes)
    pk = args.packet_bytes
    print(f"[merge] {args.stream}: {pk:,} B packets grouped by identical control words")
    print(f"[merge] {len(groups)} groups, {sum(len(set(v)) for v in groups.values())} distinct "
          f"addresses, {dupes} duplicate addresses (the same window fetched twice), "
          f"{len(runs)} maximal constant-step runs")

    cls = defaultdict(lambda: {"runs": 0, "packets": 0, "bds": 0, "delivered": 0, "union": 0})
    for _, n, step in runs:
        if step is None:
            k, bds = "lone (no partner in its group)", n
        elif step < pk:
            k, bds = "step < payload: OVERLAPPING source ranges", n
        elif step == pk:
            k, bds = "contiguous: a larger memcpy", (n + MAX_REPEAT - 1) // MAX_REPEAT
        elif step > MAX_STEP_BYTES:
            k, bds = f"gap too wide for the 20-bit step field (max {MAX_STEP_BYTES:,} B)", n
        else:
            k, bds = "regular gap: repeatable", (n + MAX_REPEAT - 1) // MAX_REPEAT
        c = cls[k]
        c["runs"] += 1
        c["packets"] += n
        c["bds"] += bds
        c["delivered"] += n * pk
        c["union"] = c.get("union", 0)
        if step is None:
            c["union"] += pk
        elif step < pk:
            c["union"] += pk                      # a lone window's union contribution
            if n > 1:
                c["union"] += (n - 1) * step
        else:
            c["union"] += n * pk
    lens = defaultdict(int)
    for _, n, _ in runs:
        lens[n] += 1
    for n, r in sorted(lens.items()):
        print(f"[runs ] length {n:2d}: {r:3d} runs" + ("   (2 points make no claim of regularity)"
                                                       if n == 2 else ""))
    tot_pk = tot_bd = 0
    for k, c in sorted(cls.items(), key=lambda kv: -kv[1]["packets"]):
        tot_pk += c["packets"]
        tot_bd += c["bds"]
        red = f", delivers {c['delivered'] / c['union']:.2f}x the address range it reads" \
            if c["union"] and c["delivered"] > c["union"] else ""
        print(f"[cls  ] {k}: {c['runs']} runs, {c['packets']} packets, "
              f"{c['bds']} descriptors if merged{red}")
    other = total - tot_pk
    print(f"[merge] this size class: {tot_pk} packets would become {tot_bd} descriptors; "
          f"the stream's other {other} tasks are unaffected, so the whole-stream floor could not "
          f"drop below {other + tot_bd} tasks from {total} by merging alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
