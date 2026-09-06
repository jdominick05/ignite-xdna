"""
YOLO-pose step 1b: cut the DFL / anchor-decode / kpt-decode tail off
yolov8n-pose.onnx.

Why: the detect pipeline (pipelines/yolov8n/1b_cut_head.py) found the VitisAI
EP takes ZERO nodes from a full quantized yolov8 graph because of the
Reshape/Softmax/Transpose/Slice/Sub/Div arithmetic tail -- the pose head ends
in the same op types, so the same cut is needed here before it's even worth
trying the full graph on the NPU.

This rewrites the model so its outputs are the nine raw head convs
(npu.yolo_pose.HEAD_OUTS). npu.yolo_pose_decode.decode_heads reproduces the
removed tail in numpy, verified bit-close (max abs diff 1.2e-4, float32 noise)
against the ONNX graph's own output0 on a real image.

    conda activate resnet_env
    python pipelines/yolov8n-pose/1b_cut_head.py

Then quantize with 3b_quantize_cut.py.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS
from npu.yolo_pose import HEAD_OUTS

TAIL_OPS = {"Reshape", "Softmax", "Transpose", "Slice", "Div", "Sub"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(MODELS / "yolov8n-pose.onnx"))
    ap.add_argument("--out", dest="dst", default=str(MODELS / "yolov8n-pose_cut.onnx"))
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
        print("\nClean: no Reshape/Softmax/Transpose/Slice/Div/Sub left.")


if __name__ == "__main__":
    main()
