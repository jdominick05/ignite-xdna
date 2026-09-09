"""
YOLO-World v2 step 1b: cut the contrastive text projection and DFL tail off yolov8s-worldv2.onnx.

Why:
  1. The contrastive text projection (5D Einsum + Exp + Mul + Add in /model.22/cv4)
     couples the text embeddings to the ONNX graph. Decoupling this allows dynamic
     open-vocabulary prompt changes on host CPU without touching or recompiling the NPU graph.
  2. The DFL bounding box decode tail (Softmax, Reshape, Slice, Sub, Add, Div)
     is pure elementwise arithmetic running in <1ms in NumPy. Keeping it in the ONNX
     graph causes VitisAI EP graph fracture and host fallbacks.

This rewrites the model so its outputs are the six raw backbone/head convs
(npu.yolow.HEAD_OUTS):
  - 3 box distribution tensors (64ch) at strides 8/16/32
  - 3 visual feature tensors (512ch) at strides 8/16/32

    conda activate resnet_env
    python pipelines/yolow/1b_cut_head.py

Then quantize with 3b_quantize_cut.py.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS
from npu.yolow import HEAD_OUTS

# Tail ops in /model.22/ that must be gone after cutting.
HEAD_TAIL_OPS = {"Softmax", "Slice", "Div", "Sub"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(MODELS / "yolov8s-worldv2.onnx"))
    ap.add_argument("--out", dest="dst", default=None)
    ap.add_argument("--input-name", default="images")
    args = ap.parse_args()

    default_dst = str(Path(args.src).with_name(Path(args.src).stem + "_cut.onnx"))
    dst_path = args.dst or default_dst

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

    onnx.utils.extract_model(args.src, dst_path, [args.input_name], HEAD_OUTS)

    dst = onnx.load(dst_path)
    onnx.checker.check_model(dst)
    print(f"{args.src}: {len(src.graph.node)} nodes")
    print(f"{dst_path}: {len(dst.graph.node)} nodes")
    print("ops:", Counter(n.op_type for n in dst.graph.node).most_common())
    print("input :", dst.graph.input[0].name,
          [d.dim_value for d in dst.graph.input[0].type.tensor_type.shape.dim])
    for o in dst.graph.output:
        print("output:", o.name,
              [d.dim_value for d in o.type.tensor_type.shape.dim])

    if [o.name for o in dst.graph.output] != HEAD_OUTS:
        raise SystemExit("output order is not HEAD_OUTS order - decode would be wrong")

    head_leftover = sorted({n.op_type for n in dst.graph.node
                            if "/model.22/" in n.name and n.op_type in HEAD_TAIL_OPS})
    if head_leftover:
        print(f"\nWARNING: detect-head decode-tail ops still present in /model.22/: {head_leftover}")
    else:
        print("\nClean: no decode tail ops remaining in detect head (/model.22/).")

    print(f"\nNext: python pipelines/yolow/3b_quantize_cut.py --in {dst_path}")


if __name__ == "__main__":
    main()
