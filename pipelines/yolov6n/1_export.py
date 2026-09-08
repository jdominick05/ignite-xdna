"""
YOLOv6n step 1: export yolov6n (Meituan RepVGG backbone) to ONNX for Ryzen AI,
then verify the graph.

    conda activate resnet_env
    python pipelines/yolov6n/1_export.py

Weights: download the official release once --
    curl -L -o models/yolov6n.pt https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt

Needs the official Meituan YOLOv6 repo checked out next to this one (not
vendored here -- same convention as kernels/'s mlir-aie dependency):
    git clone --depth 1 https://github.com/meituan/YOLOv6.git ~/YOLOv6
Override the location with YOLOV6_DIR if it lives somewhere else.

torch.load on the .pt is a pickle load -- confirmed this is Meituan's own
official GitHub release asset (github.com/meituan/YOLOv6/releases/download/...),
not a third-party mirror, before running it.

Unlike yolov8n's export, this graph never contains a Softmax: yolov6n ships
with use_dfl=False (reg_max=0, see configs/yolov6n.py), so the box head is 4
raw ltrb channels straight out of a Conv, no DFL weighted-sum decode. The
export graph still bakes in Sigmoid + the anchor-decode arithmetic
(dist2bbox, stride multiply, concat) after the raw per-level conv outputs --
that's what 1b_cut_head.py removes, same idea as yolov8n's cut, simpler tail.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import onnx
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path
from npu.paths import MODELS

import os
YOLOV6_DIR = os.environ.get("YOLOV6_DIR", os.path.join(os.path.expanduser("~"), "YOLOv6"))
sys.path.insert(0, YOLOV6_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(MODELS / "yolov6n.pt"))
    ap.add_argument("--out", default=str(MODELS / "yolov6n.onnx"))
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()

    from yolov6.layers.common import RepVGGBlock
    from yolov6.utils.checkpoint import load_checkpoint

    model = load_checkpoint(args.weights, map_location="cpu", inplace=True, fuse=True)
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()  # collapse the multi-branch train-time
                                       # graph into one 3x3 conv -- this IS
                                       # the "RepVGG backbone" hypothesis from
                                       # RESEARCH.md Category C, applied here
    model.eval()

    img = torch.zeros(1, 3, args.size, args.size)
    with torch.no_grad():
        y = model(img)  # dry run -- also confirms it doesn't crash pre-export
    # Model.forward branches on torch.onnx.is_in_onnx_export(): outside a
    # real export it returns [decoded_tensor, featmaps], not just the tensor.
    y0 = y[0] if isinstance(y, (list, tuple)) else y
    print("dry-run output shape:", tuple(y0.shape))

    torch.onnx.export(
        model, img, args.out,
        opset_version=17,          # LOCKED: project invariant, see CLAUDE.md
        input_names=["images"],
        output_names=["outputs"],
        dynamic_axes=None,         # LOCKED: static batch 1 only
        do_constant_folding=True,
    )
    print("exported", args.out)

    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    g = m.graph
    opsets = [(o.domain or "ai.onnx", o.version) for o in m.opset_import]
    shape = [d.dim_value or d.dim_param for d in g.input[0].type.tensor_type.shape.dim]
    print("opset :", opsets)
    print("input :", g.input[0].name, shape)
    print("output:", g.output[0].name,
          [d.dim_value or d.dim_param for d in g.output[0].type.tensor_type.shape.dim])
    print("ops   :", Counter(n.op_type for n in g.node).most_common())

    assert any(v == 17 for _, v in opsets), "opset is not 17"
    assert shape[0] == 1, "batch is not static 1"
    assert shape[2] == shape[3] == args.size, \
        f"exported input is {shape[2]}x{shape[3]}, asked for {args.size}"

    softmax_nodes = [n.name for n in g.node if n.op_type == "Softmax"]
    if softmax_nodes:
        print(f"\nUNEXPECTED: {len(softmax_nodes)} Softmax node(s) present "
              f"-- reg_max/use_dfl config may differ from configs/yolov6n.py's "
              f"defaults. Check before quantizing.")
    else:
        print("\nOK: no Softmax in the graph (use_dfl=False, reg_max=0, as "
              "configs/yolov6n.py sets by default) -- ready for 1b_cut_head.py")


if __name__ == "__main__":
    main()
