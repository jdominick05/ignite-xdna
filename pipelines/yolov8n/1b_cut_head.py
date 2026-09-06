"""
YOLO step 1b: cut the DFL / anchor-decode tail off yolov8n.onnx.

Why: on XDNA1 the VitisAI EP took ZERO nodes from the full quantized graph
(confirmed from yolocachekey/vitisai_ep_report.json - no NPU entry in
deviceStat at all). The last 18 nodes of a yolov8 export - Reshape / Concat /
Softmax / Transpose / Slice / Sub / Add / Div / Mul / Sigmoid - are pure
arithmetic that costs under a millisecond in numpy. Keeping them in the ONNX
graph buys nothing and gives the partitioner something to trip over.

This rewrites the model so its outputs are the six raw detection convs
(npu.yolo.HEAD_OUTS). npu.yolo_decode.decode_heads reproduces the removed tail
in numpy; the anchor grid and stride vector there were verified equal to the
constants ultralytics bakes into the graph.

    conda activate resnet_env
    python pipelines/yolov8n/1b_cut_head.py

Then quantize with 3b_quantize_cut.py - and with no subgraph exclusion, since
there is no float tail left to exclude.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS
from npu.yolo import HEAD_OUTS

# Tail ops that must be gone afterwards. Concat is NOT one of them: the
# backbone C2f blocks and SPPF use it legitimately.
TAIL_OPS = {"Reshape", "Softmax", "Transpose", "Slice", "Div", "Sub"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(MODELS / "yolov8n.onnx"))
    ap.add_argument("--out", dest="dst", default=str(MODELS / "yolov8n_cut.onnx"))
    ap.add_argument("--input-name", default="images")
    args = ap.parse_args()

    src = onnx.load(args.src)
    produced = {o for n in src.graph.node for o in n.output}
    missing = [o for o in HEAD_OUTS if o not in produced]
    if missing:
        print("ERROR: these tensors are not in the graph:")
        for m in missing:
            print("   ", m)
        print("\nConv nodes under /model.22/:")
        for n in src.graph.node:
            if n.op_type == "Conv" and "/model.22/cv" in n.name:
                print("   ", n.name, "->", n.output[0])
        raise SystemExit(1)

    onnx.utils.extract_model(args.src, args.dst, [args.input_name], HEAD_OUTS)

    dst = onnx.load(args.dst)
    onnx.checker.check_model(dst)
    print(f"{args.src}: {len(src.graph.node)} nodes")
    print(f"{args.dst}: {len(dst.graph.node)} nodes")
    print("ops:", Counter(n.op_type for n in dst.graph.node).most_common())
    print("input :", dst.graph.input[0].name,
          [d.dim_value for d in dst.graph.input[0].type.tensor_type.shape.dim])
    for o in dst.graph.output:
        print("output:", o.name,
              [d.dim_value for d in o.type.tensor_type.shape.dim])

    if [o.name for o in dst.graph.output] != HEAD_OUTS:
        raise SystemExit("output order is not HEAD_OUTS order - decode would be wrong")

    leftover = sorted({n.op_type for n in dst.graph.node} & TAIL_OPS)
    if leftover:
        print(f"\nWARNING: decode-tail ops still present: {leftover}")
    else:
        print("\nClean: no Reshape/Softmax/Transpose/Slice/Div/Sub left. "
              "Conv/Sigmoid/Mul/Concat/MaxPool/Resize/Split/Add only.")


if __name__ == "__main__":
    main()
