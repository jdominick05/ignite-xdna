"""What would this model cost in bf16, before the bf16 front-end exists to measure it?

WHY NOW. #139's front-end is the largest build left in the bf16 arc, and bf16's whole case is an
accuracy one: 0.5029 person IoU against AMD's best measured int8 of 0.4468, +12.6% relative. That
case only survives if the frame is something a user would accept. The int8 dense path already loses
3.15x to AMD, and its host and transport alone exceed AMD's entire frame, so it is worth knowing the
shape of the bf16 answer BEFORE building the thing that would measure it.

EVERYTHING HERE IS DERIVED, and this repo keeps a standing reminder of what that is worth: the
MemTile activation ring cut DDR traffic 38.3% by derivation and ran 20% SLOWER on silicon. A bracket
from a cost model is a reason to look, never a result.

THE METHOD. Take a BUILT int8 container and its MEASURED dispatch, calibrate the instruction-stream
cost model against that container rather than against another family's fit, then re-count the packets
under bf16 geometry and re-apply it.

  weights     a 9,216 B payload holds 9,216 int8 weights or 4,608 bf16       -> packets x2
  activations a packet holds 6,400 elements either way (6,400 B / 12,800 B)  -> packets x1, bytes x2
  outputs     3,200 int8 elements or 1,600 bf16                              -> packets x2

WHY A BRACKET AND NOT A NUMBER. The manifest records `activation_packets` as one figure and does not
separate tiles read from tiles written, so the output doubling cannot be counted exactly. The two
ends are therefore chosen to contain the truth rather than to be likely:

  LOW   only the weight packets add instruction stream. A buffer descriptor is the same descriptor
        whatever the element width - only the byte count inside it changes - so if no extra BD is
        issued on the activation side, this is the floor.
  HIGH  the whole packet mix doubles, which is what happens if every activation packet in the count
        is a written tile.

NOT MODELLED, and each of these pushes the same way: the bf16 kernel's own per-pass cost (it runs
35.6% above the fixed-shape kernel at k1, and 39 of MODNet-Cut's 71 convolutions are 1x1), and any
change in how many rounds the schedule needs. #132's 1,311 droppable all-zero weight packets are not
modelled either, and that one would help BOTH widths.

    bash scripts/research-lowlevel.sh --log results/dense/<name>.log --checks-only -- \\
        bash scripts/research-iron.sh tools/bf16_dispatch_estimate.py \\
            --container build/modnet_cut_adaround_dense.ignite \\
            --dispatch-ms 41.697 --host-ms 22.496 --transfer-ms 29.944 \\
            --pre-ms 2.089 --post-ms 1.366 --rival-ms 31.117 --rival-name amd
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ignite_xdna.compiler import engine_emulator as i8

FIXED_MS = 0.25          # fixed per dispatch, docs/BENCHMARKS.md:7891

# bf16 packet geometry. design.py owns these, but importing it pulls in IRON for no gain here, so
# they are restated with the arithmetic that fixes them and checked against the emulator below.
BF16_W_PAYLOAD_BYTES = i8.W_MAX_BYTES        # same 9,216 B payload, holding half as many values
BF16_A_ELEMS = 6400                          # design.py A_BYTES 12,800 / 2
BF16_O_ELEMS = 1600                          # design.py O_BYTES 3,200 / 2


def emit(tag, **payload) -> None:
    print(tag, json.dumps(payload, sort_keys=True), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", type=Path, required=True, help="a BUILT int8 container")
    ap.add_argument("--dispatch-ms", type=float, required=True,
                    help="that container's MEASURED npu_ms, which calibrates k")
    ap.add_argument("--host-ms", type=float, default=0.0)
    ap.add_argument("--transfer-ms", type=float, default=0.0)
    ap.add_argument("--pre-ms", type=float, default=0.0)
    ap.add_argument("--post-ms", type=float, default=0.0)
    ap.add_argument("--rival-ms", type=float, default=0.0, help="a measured rival frame to divide by")
    ap.add_argument("--rival-name", default="rival")
    args = ap.parse_args()

    from ignite_xdna.compiler.serializer import IgniteModelReader
    ge = IgniteModelReader(str(ROOT / args.container)).manifest["graph_engine"]

    insts = int(ge["insts_bytes"])
    apkts = int(ge["activation_packets"])
    wpkt_bytes = int(ge["wpackets_bytes"])
    wpkts, rem = divmod(wpkt_bytes, i8.W_BYTES)
    if rem:
        raise SystemExit(f"wpackets_bytes {wpkt_bytes} is not a whole number of {i8.W_BYTES} B packets")

    # The element ratios, derived rather than asserted, so a geometry change cannot pass silently.
    w_ratio = i8.W_MAX_BYTES / (BF16_W_PAYLOAD_BYTES / 2)      # int8 weights per packet / bf16
    a_ratio = i8.A_BYTES / BF16_A_ELEMS                        # int8 elements per packet / bf16
    o_ratio = i8.O_BYTES / BF16_O_ELEMS
    emit("GEOMETRY", int8_weights_per_packet=i8.W_MAX_BYTES, bf16_weights_per_packet=BF16_W_PAYLOAD_BYTES // 2,
         int8_activation_elems=i8.A_BYTES, bf16_activation_elems=BF16_A_ELEMS,
         int8_output_elems=i8.O_BYTES, bf16_output_elems=BF16_O_ELEMS,
         weight_packet_ratio=w_ratio, activation_packet_ratio=a_ratio, output_packet_ratio=o_ratio)

    k_ns_per_byte = (args.dispatch_ms - FIXED_MS) * 1e6 / insts
    emit("INT8_MEASURED", container=str(args.container), insts_bytes=insts,
         activation_packets=apkts, weight_packets=wpkts, wpackets_bytes=wpkt_bytes,
         rounds=int(ge["rounds"]), workspace_bytes=int(ge["workspace_bytes"]),
         dispatch_ms=args.dispatch_ms, k_ns_per_instruction_byte=round(k_ns_per_byte, 4))

    total = apkts + wpkts
    low_factor = (apkts + wpkts * w_ratio) / total    # weight side only
    high_factor = 2.0                                 # the whole mix
    out = {}
    for name, factor in (("low", low_factor), ("high", high_factor)):
        dispatch = FIXED_MS + insts * factor * k_ns_per_byte / 1e6
        transfer = args.transfer_ms * 2.0             # every tile and weight doubles in bytes
        frame = args.pre_ms + dispatch + args.host_ms + transfer + args.post_ms
        out[name] = dict(insts_factor=round(factor, 4), dispatch_ms=round(dispatch, 3),
                         transfer_ms=round(transfer, 3), host_ms=args.host_ms,
                         frame_ms=round(frame, 3))
        if args.rival_ms:
            out[name][f"vs_{args.rival_name}"] = round(frame / args.rival_ms, 3)
    emit("BF16_ESTIMATE", derived=True, **out)
    emit("CAVEAT",
         note="DERIVED, not measured. The MemTile activation ring cut DDR traffic 38.3% by "
              "derivation and ran 20% slower on silicon; treat this as a reason to look.",
         not_modelled=["the bf16 kernel's own per-pass cost (35.6% above the fixed-shape kernel at "
                       "k1, and 39 of 71 convolutions here are 1x1)",
                       "any change in the number of schedule rounds",
                       "#132's 1,311 droppable all-zero weight packets, which would help both widths"],
         bracket_reason="the manifest does not separate activation tiles read from tiles written, so "
                        "the output doubling cannot be counted exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
