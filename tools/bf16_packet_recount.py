"""Re-count a built int8 container's packets under bf16 TILE geometry, per layer.

WHY THIS EXISTS, AND WHAT IT CORRECTS. `tools/bf16_dispatch_estimate.py` scales a container's packet
mix by fixed element ratios: weights x2 because a 9,216 B payload holds 9,216 int8 values or 4,608
bf16, activations x1 in count with double bytes, outputs x2. The activation rule is the one that does
not survive contact with the tile geometry.

**A bf16 output tile carries 16 channels where an int8 tile carries 32.** int8's `O_BYTES = 3200` is
5 x 20 x 32 at one byte; bf16's `O_ELEMS = 1600` is 5 x 20 x **16** at two bytes - the same 3,200 B
object, half the channels, because `mmul<4,8,4>` blocks the output by 4 where `mmul<4,8,8>` blocks it
by 8. So a layer needs `ceil(Cout / 16)` groups in bf16 against `ceil(Cout / 32)` in int8, and every
group re-reads the whole input plane. When that ratio is 2 the activation packet COUNT doubles on top
of its bytes doubling, and the old estimator's "packets x1" understates the frame badly.

BUT THE RATIO IS NOT ALWAYS 2, AND THAT IS THE POINT OF COUNTING RATHER THAN ASSUMING. A layer
narrower than 16 output channels needs one group at either width, so bf16 costs it nothing - and a
layer narrower than 32 channels is one where INT8 is wasting half of its output tile. Only layers
above 16 channels pay. The ratio has to be read off the model, per layer, which is what this does.

    bash scripts/research-lowlevel.sh --log results/aie/<name>.log --checks-only -- \\
        bash scripts/research-iron.sh tools/bf16_packet_recount.py \\
            --container build/sesr_m7.ignite --model models/sesr_m7_xint8.onnx

Offline. No device, no timing claim. Counts and ratios only; feed them to the dispatch estimator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

INT8_TILE_CHANNELS = 32      # engine_emulator O_BYTES 3,200 = 5 x 20 x 32, one byte each
BF16_TILE_CHANNELS = 16      # design.py O_ELEMS 1,600 = 5 x 20 x 16, two bytes each


def emit(tag, **payload) -> None:
    print(tag, json.dumps(payload, sort_keys=True), flush=True)


def conv_out_channels(graph):
    """{conv node name: (Cout, Cin, k)} - the weight may arrive through a DequantizeLinear."""
    init = {t.name: t for t in graph.initializer}
    produced_by = {o: n for n in graph.node for o in n.output}
    out = {}
    for n in graph.node:
        if n.op_type != "Conv":
            continue
        src = init.get(n.input[1])
        if src is None:                                   # walk back through DQ to the int8 tensor
            p = produced_by.get(n.input[1])
            src = init.get(p.input[0]) if p is not None else None
        if src is None:
            continue
        d = list(src.dims)
        out[n.name] = (int(d[0]), int(d[1]), int(d[2]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", type=Path, required=True, help="a BUILT int8 container")
    ap.add_argument("--model", type=Path, required=True,
                    help="the ONNX that container was built from, for per-layer Cout")
    args = ap.parse_args()

    from ignite_xdna.compiler.serializer import IgniteModelReader
    path = args.container if args.container.is_absolute() else ROOT / args.container
    ge = IgniteModelReader(str(path)).manifest["graph_engine"]
    mpath = args.model if args.model.is_absolute() else ROOT / args.model
    shapes = conv_out_channels(onnx.load(str(mpath)).graph)

    emit("RECOUNT_ENV", container=args.container.as_posix(), model=args.model.as_posix(),
         int8_tile_channels=INT8_TILE_CHANNELS, bf16_tile_channels=BF16_TILE_CHANNELS,
         insts_bytes=int(ge["insts_bytes"]), rounds=int(ge["rounds"]),
         activation_packets=int(ge["activation_packets"]))

    a_i8 = a_bf = r_i8 = r_bf = 0
    unmatched = []
    for L in ge["layers"]:
        name, pk, rd = L["name"], int(L["packets"]), int(L["rounds"])
        if name not in shapes:
            unmatched.append(name)
            a_i8 += pk
            a_bf += pk
            r_i8 += rd
            r_bf += rd
            continue
        cout, cin, k = shapes[name]
        gi = -(-cout // INT8_TILE_CHANNELS)
        gb = -(-cout // BF16_TILE_CHANNELS)
        ratio = gb / gi
        a_i8 += pk
        a_bf += pk * ratio
        r_i8 += rd
        r_bf += rd * ratio
        emit("LAYER", index=int(L["index"]), name=name, cout=cout, cin=cin, k=k,
             groups_int8=gi, groups_bf16=gb, group_ratio=ratio,
             packets_int8=pk, packets_bf16=int(pk * ratio),
             rounds_int8=rd, rounds_bf16=int(rd * ratio),
             int8_tile_wasted_channels=gi * INT8_TILE_CHANNELS - cout)

    a_ratio = a_bf / a_i8 if a_i8 else 1.0
    emit("ACTIVATION_RECOUNT", packets_int8=a_i8, packets_bf16=int(round(a_bf)),
         packet_count_ratio=round(a_ratio, 6), rounds_int8=r_i8, rounds_bf16=int(round(r_bf)),
         unmatched_layers=unmatched,
         note="a bf16 activation packet also carries twice the BYTES of an int8 one, so activation "
              "DDR traffic scales as this ratio TIMES two; the instruction stream scales as this "
              "ratio alone, because a buffer descriptor is one descriptor whatever it carries")
    emit("VERDICT",
         activation_packet_count_ratio=round(a_ratio, 6),
         old_estimator_assumed=1.0,
         understated=bool(a_ratio > 1.0),
         reading=("bf16 costs this model nothing in activation packet count: every layer is at or "
                  "below 16 output channels, so one group serves it at either width - and int8 is "
                  "wasting tile channels here" if a_ratio == 1.0 else
                  "bf16 needs more groups than int8 on this model, so activation packets multiply "
                  "on top of their bytes doubling and the old bracket understates the frame"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
