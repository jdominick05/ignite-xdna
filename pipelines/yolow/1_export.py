"""
YOLO-World v2 step 1: export yolov8s-worldv2 to ONNX for Ryzen AI, then verify the graph.

    python pipelines/yolow/1_export.py                  # stock yolov8s-worldv2, 640
    python pipelines/yolow/1_export.py --no-text-attn   # ablate C2fAttn text attention blocks

Run in the env that has torch + ultralytics (resnet_env).
"""
import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS
from npu.yolow import COCO_TXT_FEATS_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(MODELS / "yolov8s-worldv2.pt"))
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--no-text-attn", action="store_true",
                    help="ablate the C2fAttn text cross-attention blocks (layers 12, 15, 18, 21) "
                         "by replacing 5D Einsum/ReduceMax with static learned-bias channel scaling")
    ap.add_argument("--out", default=None,
                    help="target export path (defaults to models/yolov8s-worldv2.onnx or "
                         "models/yolov8s-worldv2_no_attn.onnx)")
    args = ap.parse_args()

    default_name = "yolov8s-worldv2_no_attn.onnx" if args.no_text_attn else "yolov8s-worldv2.onnx"
    target_out = args.out or str(MODELS / default_name)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    coco_classes = list(model.names.values())
    model.set_classes(coco_classes)

    # Save normalized COCO text embeddings for decoupled NumPy contrastive scoring
    txt_feats = model.model.txt_feats
    txt_feats_norm = F.normalize(txt_feats, dim=-1).squeeze(0).cpu().numpy()
    np.save(COCO_TXT_FEATS_PATH, txt_feats_norm)
    print(f"saved {txt_feats_norm.shape} normalized COCO text embeddings to {COCO_TXT_FEATS_PATH}")

    if args.no_text_attn:
        print("Ablating C2fAttn text attention blocks at layers 12, 15, 18, 21...")
        m = model.model
        for i in [12, 15, 18, 21]:
            layer = m.model[i]
            ab = layer.attn

            def make_forward(block):
                scale_ch = (block.bias.sigmoid() * block.scale).repeat_interleave(block.hc).view(1, -1, 1, 1)
                def forward(x, guide):
                    return block.proj_conv(x) * scale_ch
                return forward

            ab.forward = make_forward(ab)

    ul_path = str(Path(args.weights).with_suffix(".onnx"))
    stash = None
    if os.path.abspath(target_out) != os.path.abspath(ul_path) and os.path.exists(ul_path):
        stash = ul_path + ".displaced"
        os.replace(ul_path, stash)
        print(f"stashed the existing {ul_path} while exporting")

    try:
        path = model.export(format="onnx", opset=17, imgsz=args.size,
                            simplify=True, dynamic=False, batch=1)
        print("exported", path)
        if os.path.abspath(target_out) != os.path.abspath(path):
            os.makedirs(os.path.dirname(os.path.abspath(target_out)), exist_ok=True)
            shutil.move(path, target_out)
            path = target_out
            print("moved to ", path)
    finally:
        if stash and os.path.exists(stash):
            os.replace(stash, ul_path)
            print(f"restored {ul_path}")

    m = onnx.load(path)
    onnx.checker.check_model(m)
    g = m.graph
    opsets = [(o.domain or "ai.onnx", o.version) for o in m.opset_import]
    shape = [d.dim_value or d.dim_param for d in g.input[0].type.tensor_type.shape.dim]
    print("opset :", opsets)
    print("input :", g.input[0].name, shape)
    print("output:", g.output[0].name,
          [d.dim_value or d.dim_param for d in g.output[0].type.tensor_type.shape.dim])
    print("ops   :", Counter(n.op_type for n in g.node).most_common())
    print("\nNext: python pipelines/yolow/1b_cut_head.py --in " + path)


if __name__ == "__main__":
    main()
