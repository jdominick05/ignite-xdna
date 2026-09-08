"""
YOLOv6n step 1b: cut the anchor-decode tail off yolov6n.onnx.

Same rationale as pipelines/yolov8n/1b_cut_head.py: the decode tail (Sigmoid,
Concat, Split, Sub/Add/Div/Mul arithmetic for dist2bbox, Expand for the
anchor grid) is cheap numpy, and giving the VitisAI EP a float tail after
the quantized conv trunk risks it fragmenting the graph rather than placing
one clean subgraph. yolov6n's tail is shorter than yolov8n's -- no DFL
Softmax at all, since configs/yolov6n.py ships reg_max=0 / use_dfl=False.

This rewrites the model so its outputs are the six raw detection convs
(npu.yolov6.HEAD_OUTS). npu.yolov6_decode.decode_heads reproduces the
removed tail in numpy.

    conda activate resnet_env
    python pipelines/yolov6n/1b_cut_head.py

Then quantize with 3b_quantize_cut.py.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS
from npu.yolov6 import HEAD_OUTS

# Tail ops that must be gone afterwards. Concat is NOT one of them (Rep-PAN
# feature concatenation is legitimate), and neither is Sigmoid/Mul: the
# head's own stem/cls_conv/reg_conv layers use SiLU (ConvBNSiLU), which
# exports as Sigmoid+Mul with no native ONNX SiLU op -- same as yolov8n's
# backbone SiLU activations surviving its cut. Only the final decode
# Sigmoid (applied to cls_preds right before the removed concat) is gone.
TAIL_OPS = {"Split", "Sub", "Div", "Expand", "Unsqueeze", "Cast"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(MODELS / "yolov6n.onnx"))
    ap.add_argument("--out", dest="dst", default=str(MODELS / "yolov6n_cut.onnx"))
    ap.add_argument("--input-name", default="images")
    args = ap.parse_args()

    src = onnx.load(args.src)
    produced = {o for n in src.graph.node for o in n.output}
    missing = [o for o in HEAD_OUTS if o not in produced]
    if missing:
        print("ERROR: these tensors are not in the graph:")
        for m in missing:
            print("   ", m)
        print("\nConv nodes under /detect/:")
        for n in src.graph.node:
            if n.op_type == "Conv" and "/detect/" in n.name:
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
        print("\nClean: no decode-tail ops left. "
              "Conv/ConvTranspose/Relu/Mul/Concat/MaxPool only.")


if __name__ == "__main__":
    main()
