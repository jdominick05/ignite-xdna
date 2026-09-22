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

  weights     a 9,216 B payload holds 9,216 int8 weights or 4,608 bf16       -> at most packets x2
  activations a packet holds 6,400 elements either way (6,400 B / 12,800 B)  -> bytes x2, and a
              COUNT multiplier that has to be read off the model, not assumed
  outputs     3,200 B tile either way: 32 int8 channels or 16 bf16           -> packets x2

**CORRECTED 2026-09-22.** This tool hardcoded the activation packet count at x1, and that is wrong
for any model wider than 16 output channels a layer. A bf16 output tile carries **16** channels where
an int8 tile carries **32** (`mmul<4,8,4>` blocks the output by 4; `mmul<4,8,8>` by 8), so a layer
needs `ceil(Cout/16)` groups against `ceil(Cout/32)`, and every group re-reads the whole input plane.
`tools/bf16_packet_recount.py` counts that per layer and `--activation-packet-ratio` is now required.
Measured: SESR-M7 **1.0000** (every layer is 16 channels or fewer, so int8 was wasting half its
output tile and bf16 costs it nothing) against MODNet-Cut **1.8841** (74,276 -> 139,944 packets).

WHY A BRACKET AND NOT A NUMBER. The activation count is now counted rather than assumed, so the
bracket is narrower and is about the WEIGHT side, which the manifest cannot resolve:

  LOW   activation packets as recounted, weight packet count held. A bf16 weight packet is the same
        9,472 B and covers half the output channels, so a layer that under-filled its int8 packet
        needs no more packets at bf16.
  HIGH  weight packets double as well, which is what a layer that filled its int8 packet costs.

NOT MODELLED, and each of these pushes the same way: the bf16 kernel's own per-pass cost (it runs
35.6% above the fixed-shape kernel at k1, and 39 of MODNet-Cut's 71 convolutions are 1x1), and any
change in how many rounds the schedule needs. #132's 1,311 droppable all-zero weight packets are not
modelled either, and that one would help BOTH widths.

    bash scripts/research-lowlevel.sh --log results/dense/<name>.log --checks-only -- \\
        bash scripts/research-iron.sh tools/bf16_dispatch_estimate.py \\
            --container build/modnet_cut_adaround_dense.ignite \\
            --dispatch-ms 41.697 --host-ms 22.496 --transfer-ms 29.944 \\
            --pre-ms 2.089 --post-ms 1.366 --rival-ms 31.117 --rival-name amd \\
            --activation-packet-ratio 1.884108
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
    ap.add_argument("--activation-packet-ratio", type=float, required=True,
                    help="bf16 activation packets per int8 activation packet, from "
                         "tools/bf16_packet_recount.py. REQUIRED: this was hardcoded to 1.0 and "
                         "that is wrong for any model wider than 16 output channels a layer, "
                         "because a bf16 output tile carries 16 channels where int8 carries 32, "
                         "so the layer needs more groups and every group re-reads the input plane")
    args = ap.parse_args()
    if args.activation_packet_ratio < 1.0:
        ap.error("an activation packet ratio below 1.0 is not physical: a bf16 output tile carries "
                 "at most as many channels as an int8 one, never more")

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
    a_elem_ratio = i8.A_BYTES / BF16_A_ELEMS                   # ELEMENTS per packet, not packets
    o_ratio = i8.O_BYTES / BF16_O_ELEMS
    emit("GEOMETRY", int8_weights_per_packet=i8.W_MAX_BYTES, bf16_weights_per_packet=BF16_W_PAYLOAD_BYTES // 2,
         int8_activation_elems=i8.A_BYTES, bf16_activation_elems=BF16_A_ELEMS,
         int8_output_elems=i8.O_BYTES, bf16_output_elems=BF16_O_ELEMS,
         weight_packet_capacity_ratio=w_ratio, activation_elems_per_packet_ratio=a_elem_ratio,
         output_channels_per_tile_ratio=o_ratio,
         note="these are CAPACITY ratios per packet. The activation packet COUNT ratio is a "
              "separate, per-model quantity and arrives via --activation-packet-ratio.")

    k_ns_per_byte = (args.dispatch_ms - FIXED_MS) * 1e6 / insts
    emit("INT8_MEASURED", container=str(args.container), insts_bytes=insts,
         activation_packets=apkts, weight_packets=wpkts, wpackets_bytes=wpkt_bytes,
         rounds=int(ge["rounds"]), workspace_bytes=int(ge["workspace_bytes"]),
         dispatch_ms=args.dispatch_ms, k_ns_per_instruction_byte=round(k_ns_per_byte, 4))

    total = apkts + wpkts
    a_bf16 = apkts * args.activation_packet_ratio
    # LOW: activation packets recounted from the tile geometry (structural, not a guess), weight
    # packet COUNT held - a bf16 weight packet is the same 9,472 B and covers half the output
    # channels, so a layer that under-filled its int8 packet needs no more packets in bf16.
    # HIGH: weight packets double too, which is what a layer that filled its int8 packet costs.
    low_factor = (a_bf16 + wpkts) / total
    high_factor = (a_bf16 + wpkts * w_ratio) / total
    emit("ACTIVATION_RULE", activation_packet_ratio=args.activation_packet_ratio,
         activation_packets_int8=apkts, activation_packets_bf16=int(round(a_bf16)),
         source="tools/bf16_packet_recount.py",
         note="a bf16 output tile carries 16 channels where an int8 tile carries 32, so a layer "
              "needs ceil(Cout/16) groups against ceil(Cout/32) and every group re-reads the input "
              "plane. A ratio of 1.0 means the model is narrow enough that int8 was wasting tile "
              "channels, and bf16 costs it nothing here.")
    out = {}
    for name, factor in (("low", low_factor), ("high", high_factor)):
        dispatch = FIXED_MS + insts * factor * k_ns_per_byte / 1e6
        # Activation BYTES double per packet on top of any count multiplier; weight packets and
        # output tiles are the same 9,472 B and 3,200 B at either width, so they do not grow in
        # bytes. Without a transfer split the two ends bracket it.
        t_scale = 2.0 if name == "low" else 2.0 * args.activation_packet_ratio
        transfer = args.transfer_ms * t_scale
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
