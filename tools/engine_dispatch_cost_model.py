"""What governs a whole-frame dispatch: the activation packet, or the instruction stream?

WHY. The engine's standing against AMD inverts with model size - YOLOv8n 1.34x faster, YOLOv8x
1.47x slower - and the `ptr` port showed the convolution loop is not the binding term: a 21.7%
saving in hot-loop bundles moved a YOLOv8x frame by 6%. This composes what is already measured on
this machine into a model of where the rest goes, so the question is narrowed before an
instrument is pointed at it.

Everything here is DERIVED. Packet counts and instruction-stream sizes are read out of built
containers; dispatch times are supplied from measured A/B logs. No new measurement is taken, and
a derived cost is not a latency prediction - this repo has the MemTile activation ring as the
standing reminder, a design that cut DDR traffic 38.3% and ran 20% slower.

TWO CANDIDATES.
  A  the activation packet is the unit of cost:
     dispatch = 0.25 ms + packets x (145 ns instruction op + 6,400 B at 26.8 GB/s)
     No fitted parameter; every constant is already established
     (docs/BENCHMARKS.md, docs/SILICON.md, compiler/engine_emulator.py).
  B  the instruction stream is the unit:
     dispatch = 0.25 ms + instruction_bytes x k
     One constant, k, which the fit reports rather than assumes. B is what
     docs/BENCHMARKS.md:7891-7893 concluded from a different direction - appending harmless BD
     writes cost 145 ns per op, a transport probe moved 78.7 MB in 2.94 ms whether as 52 or 772
     tasks, and "the per-round schedule was paying for ops and for columns waiting on each other,
     not for bytes". If B holds across the family, A's fit is downstream of it: more activation
     packets means more BD writes means a longer instruction stream.

The discriminator is whether k is constant across models. A term that is really the cost has a
flat per-unit price; one that merely correlates does not.

    bash scripts/research-lowlevel.sh --log results/aie/engine_dispatch_cost_model_<date>.log \\
        --checks-only -- bash scripts/research-iron.sh tools/engine_dispatch_cost_model.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXED_MS = 0.25          # fixed per dispatch, docs/BENCHMARKS.md:7891
OP_NS = 145.0            # per instruction-stream op, same line
BW = 26.8e9              # transport, docs/SILICON.md
A_BYTES = 6400           # int8 activation packet, compiler/engine_emulator.py

# Containers built for the ptr port gate, and the dispatch of their ptr arm in the same-sitting
# interleaved A/Bs. The control arm is the container; the ptr arm is the kernel now committed.
DEFAULT = [
    ("yolov8n", "build/ptr_gate/ab_yolov8n_ctl.ignite", 6.964),
    ("yolov8s", "build/ptr_gate/ab_yolov8s_ctl.ignite", 15.840),
    ("yolov8x", "build/ptr_gate/x_split_ctl.ignite", 114.989),
]


def manifest_of(path: Path) -> dict:
    from ignite_xdna.compiler.serializer import IgniteModelReader
    r = IgniteModelReader(str(path))
    m = getattr(r, "manifest", None)
    if m is None:
        for attr in ("read_manifest", "manifest_dict", "load_manifest"):
            if hasattr(r, attr):
                m = getattr(r, attr)()
                break
    if m is None:
        raise SystemExit(f"{path}: cannot reach the manifest")
    return m.get("graph_engine", m)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=None, metavar="NAME=CONTAINER=MS",
                    help="repeatable; overrides the built-in set")
    args = ap.parse_args()

    arms = DEFAULT
    if args.arm:
        arms = []
        for spec in args.arm:
            name, container, ms = spec.rsplit("=", 2)
            arms.append((name, container, float(ms)))

    rows = []
    for name, container, disp in arms:
        p = ROOT / container
        if not p.exists():
            raise SystemExit(f"missing {p}")
        ge = manifest_of(p)
        rows.append(dict(model=name, dispatch_ms=disp,
                         apkts=int(ge["activation_packets"]),
                         insts_bytes=int(ge["insts_bytes"]),
                         rounds=int(ge["rounds"]),
                         wpackets_bytes=int(ge["wpackets_bytes"]),
                         workspace_bytes=int(ge["workspace_bytes"])))

    per_pkt_ns = OP_NS + A_BYTES / BW * 1e9
    for r in rows:
        a = FIXED_MS + r["apkts"] * per_pkt_ns / 1e6
        r["model_a_ms"] = a
        r["model_a_explained"] = a / r["dispatch_ms"]
        r["ns_per_instruction_byte"] = (r["dispatch_ms"] - FIXED_MS) * 1e6 / r["insts_bytes"]
        r["bytes_per_op"] = OP_NS / r["ns_per_instruction_byte"]
        r["activation_bytes"] = r["apkts"] * A_BYTES
        # workspace_bytes is peak allocation under reuse, not the sum of tensor bytes, so this is
        # a LOWER BOUND on how often an activation tile is re-read.
        r["refetch_lower_bound"] = r["activation_bytes"] / r["workspace_bytes"]

    ks = [r["ns_per_instruction_byte"] for r in rows]
    k = sum(ks) / len(ks)
    for r in rows:
        b = FIXED_MS + r["insts_bytes"] * k / 1e6
        r["model_b_ms"] = b
        r["model_b_explained"] = b / r["dispatch_ms"]

    for r in rows:
        print("ENGINE_DISPATCH_COST " + json.dumps(r, sort_keys=True), flush=True)
    print("ENGINE_DISPATCH_COST_FIT " + json.dumps({
        "model_b_k_ns_per_instruction_byte": k,
        "model_b_k_spread": max(ks) / min(ks),
        "model_a_per_packet_ns": per_pkt_ns,
        "frame_time_range": max(r["dispatch_ms"] for r in rows)
                            / min(r["dispatch_ms"] for r in rows),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
