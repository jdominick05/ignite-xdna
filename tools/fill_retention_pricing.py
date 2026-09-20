#!/usr/bin/env python3
"""Price cross-layer activation retention against the shipped transport, chains included.

    python tools/fill_retention_pricing.py          # defaults are the logged SESR M7 figures

Offline arithmetic on logged measurements -- no device, no hardware context, no source change. It
exists because [the layout sizing](fill_layout_sizing.py) showed the MemTile activation ring multiplies
the class of descriptors that a freed dimension could collapse, and the question "would a
dimension-free retention design be affordable?" then needs a price, not a hope.

Every input is a measured figure, printed with its source so a reader can redo the arithmetic:
  * default rb=2: 1,007 shim descriptors, 13,181,568 B, floor 2.629 ms, compute 1.682 ms
    (results/aie/shim_channel_utilisation_sesr_yolov8n_desktop2_20260919.log and
    results/aie/retire_batch_cadence_sweep_phoenix_20260919T0251Z.log)
  * ring of 2 slots: 4,732 descriptors, 39,781,248 B, floor 5.952 ms, compute 0.358 ms
    (same shim-channel log)
  * four-dimensional (chain-capped-at-4) pattern classes from fill_layout_sizing.py, run with and
    without --ring 2
  * 6.898931 GB/s per column per direction, one column full duplex
    (results/aie/silicon_stream_width_desktop2_20260919.log)

The model decomposes a floor into transfer + per-task, so the per-task unit is a residue, and it
prices a hypothetical by scaling descriptor count at the arm's own unit. Two honest limits follow
from that and are printed rather than hidden: the unit differs by task TYPE (the ring's own unit is
0.953 us against the shipped container's 2.136 us, so task cost does NOT scale across designs -- an
assumption tools/fill_layout_lever_math.py makes and which this tool does not reuse), and the whole
thing is calibrated on exactly the two arms it then interpolates. A derived costing of this same
lever once predicted YOLOv8s would win by 0.748 ms and it lost by 3.40 ms.
"""
import argparse
import datetime
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate-gbps", type=float, default=6.898931)
    ap.add_argument("--cols", type=int, default=4)
    a = ap.add_argument
    a("arm", nargs="*", default=[], metavar="NAME:TASKS:BYTES:FLOOR:COMPUTE:FOURD:PACKETS:ABUT",
        help="an arm to price; defaults to the two logged SESR M7 arms")
    args = ap.parse_args()

    default = [
        ("shipped rb=2", 1007, 13181568, 2.629, 1.682, 338, 1352, 169),
        ("ring, 2 slots", 4732, 39781248, 5.952, 0.358, 1352, 5408, 169),
    ]
    arms = []
    for spec in args.arm:
        name, tasks, byts, floor, comp, fourd, pkts, abut = spec.split(":")
        arms.append((name, int(tasks), int(byts), float(floor), float(comp), int(fourd),
                     int(pkts), int(abut)))
    if not arms:
        arms = list(default)
        print("[arms] using the logged SESR M7 defaults; pass NAME:TASKS:BYTES:FLOOR:COMPUTE:"
              "FOURD:PACKETS:ABUT to price another pair")

    print("UTC:", datetime.datetime.now(datetime.timezone.utc).isoformat())
    print("MACHINE:", platform.node(), platform.processor())
    print("COMMAND: python tools/fill_retention_pricing.py", " ".join(args.arm))
    print("COMMIT:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    dirty = subprocess.check_output(["git", "status", "--short", "--", "src", "tools"], text=True).strip()
    if dirty:
        print("SOURCE: working tree modified (the tool reads logs, not src/):")
        for line in dirty.splitlines():
            print("       ", line)

    rows = []
    for name, tasks, byts, floor, comp, fourd, pkts, abut in arms:
        transfer = byts / args.cols / (args.rate_gbps * 1e9) * 1e3
        unit = (floor - transfer) / tasks
        print(f"\n[arm] {name}: {tasks} descriptors, {byts:,} B, floor {floor:.3f} ms")
        print(f"  transfer {transfer:.3f} + per-task {floor - transfer:.3f} -> "
              f"{unit * 1e3:.3f} us/descriptor; dispatch {floor + comp:.3f} ms "
              f"(compute {comp:.3f})")
        print(f"  chain-capped class: {fourd} descriptors / {pkts} packets = "
              f"{pkts / fourd:g} per descriptor; {abut} abut ({abut / fourd * 100:.0f}%) so a freed "
              f"dimension is byte-free only for those")
        rows.append((name, tasks, byts, floor, comp, fourd, pkts, unit))

    base = rows[0]
    print(f"\n[units] the per-descriptor unit differs by arm: " +
          ", ".join(f"{n} {u * 1e3:.3f} us" for n, _, _, _, _, _, _, u in rows))
    print("  so per-task cost does not scale across task TYPES, and a model that assumed it would "
          "under-price a design whose tasks are individually cheap")

    ring = next((r for r in rows if "ring" in r[0]), None)
    if ring and len(rows) > 1:
        _, rtasks, rbytes, rfloor, rcomp, rfour, rpkts, runit = ring
        rtransfer = rbytes / args.cols / (args.rate_gbps * 1e9) * 1e3
        print(f"\n[ideal] collapse the ring's chain-capped class as far as the hardware allows "
              f"({rtasks} descriptors, {rfour} of them capped, {rpkts} packets)")
        for chain in (4, 8, 16, 32, 64):
            collapsed = -(-rpkts // chain)
            tasks = rtasks - rfour + collapsed
            floor = rtransfer + tasks * runit
            disp = floor + rcomp
            delta = disp - (base[3] + base[4])
            print(f"  chain {chain:2d}: descriptors {tasks:5d}, floor {floor:.3f}, dispatch "
                  f"{disp:.3f} ms -> " +
                  ("BEATS the shipped container" if delta < 0 else
                   f"still {delta:.3f} ms slower than the shipped container"))
        print(f"\n[why] retention bought compute {base[4]:.3f} -> {rcomp:.3f} ms "
              f"(-{base[4] - rcomp:.3f}) and cost floor {base[3]:.3f} -> {rfloor:.3f} ms "
              f"(+{rfloor - base[3]:.3f}) on {rtasks / base[1]:.1f}x the descriptors and "
              f"{rbytes / base[2]:.1f}x the bytes")
        print(f"[why] deleting the ring's ENTIRE capped class -- not collapsing it, deleting it -- "
              f"still leaves {rtasks - rfour} descriptors at {runit * 1e3:.3f} us = "
              f"{(rtasks - rfour) * runit:.3f} ms against a compute win of "
              f"{base[4] - rcomp:.3f} ms")
    if ring:
        print("\n[verdict] on these arms, chain length is not what retention costs: the best the "
              "freed dimension can do is collapse the capped class, and that still leaves the "
              "retention design slower than the shipped one. Reading, not measurement: what would "
              "have to change is the number of serves -- retain at a granularity coarse enough that "
              "the serves are fewer than the fills they replace, rather than chaining the serves "
              "better.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
