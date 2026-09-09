"""
YOLOv11 step 1b: cut the DFL / anchor-decode tail off yolo11n.onnx.

Why: the last 18+ nodes of a YOLOv11 export - Reshape / Concat / Softmax /
Transpose / Slice / Sub / Add / Div / Mul / Sigmoid in /model.23/ - are pure
arithmetic that costs under a millisecond in numpy. Keeping them in the ONNX
graph gives the VitisAI EP partitioner something to trip over and causes fallbacks.

This rewrites the model so its outputs are the six raw detection convs
(npu.yolov11.HEAD_OUTS). npu.yolov11.decode_heads reproduces the removed tail
in numpy.

    conda activate resnet_env
    python pipelines/yolov11/1b_cut_head.py

Then quantize with 3b_quantize_cut.py.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS
from npu.yolov11 import HEAD_OUTS

# Tail ops in /model.23/ that must be gone after cutting.
HEAD_TAIL_OPS = {"Reshape", "Softmax", "Transpose", "Slice", "Div", "Sub"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(MODELS / "yolo11n.onnx"))
    ap.add_argument("--out", dest="dst", default=str(MODELS / "yolo11n_cut.onnx"))
    ap.add_argument("--input-name", default="images")
    args = ap.parse_args()

    src = onnx.load(args.src)
    produced = {o for n in src.graph.node for o in n.output}
    missing = [o for o in HEAD_OUTS if o not in produced]
    if missing:
        print("ERROR: these tensors are not in the graph:")
        for m in missing:
            print("   ", m)
        print("\nConv nodes under /model.23/:")
        for n in src.graph.node:
            if n.op_type == "Conv" and "/model.23/cv" in n.name:
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

    head_leftover = sorted({n.op_type for n in dst.graph.node
                            if "/model.23/" in n.name and n.op_type in HEAD_TAIL_OPS})
    if head_leftover:
        print(f"\nWARNING: detect-head decode-tail ops still present in /model.23/: {head_leftover}")
    else:
        print("\nClean: no decode tail ops remaining in detect head (/model.23/).")


if __name__ == "__main__":
    main()
