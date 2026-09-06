"""
YOLO-pose step 1: export yolov8n-pose to ONNX for Ryzen AI, then verify the graph.

    python pipelines/yolov8n-pose/1_export.py            # yolov8n-pose, 640
    python pipelines/yolov8n-pose/1_export.py --size 416

Run in the env that has torch + ultralytics (resnet_env).

Verified against the exported graph (see docs/DECISIONS.md): the pose head
(model.22, class Pose) adds a third conv branch, cv4, alongside detect's cv2
(box) and cv3 (cls, here 1 channel for "person" only) -- 51 raw channels
(17 keypoints x x/y/visibility) per anchor. Output is (1, 56, N), not (1, 84, N).
"""
import argparse
import os
import shutil
from collections import Counter

import onnx

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(MODELS / "yolov8n-pose.pt"))
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--out", default=None,
                    help="move the export here (see pipelines/yolov8n/1_export.py "
                         "for why this matters for a resolution sweep)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)  # downloads if missing

    ul_path = str(Path(args.weights).with_suffix(".onnx"))
    stash = None
    if args.out and os.path.abspath(args.out) != os.path.abspath(ul_path) \
            and os.path.exists(ul_path):
        stash = ul_path + ".displaced"
        os.replace(ul_path, stash)
        print(f"stashed the existing {ul_path} while exporting")

    try:
        path = model.export(format="onnx", opset=17, imgsz=args.size,
                            simplify=True, dynamic=False, batch=1)
        print("exported", path)
        if args.out and os.path.abspath(args.out) != os.path.abspath(path):
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            shutil.move(path, args.out)
            path = args.out
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

    assert any(v == 17 for _, v in opsets), "opset is not 17"
    assert shape[0] == 1, "batch is not static 1"
    assert shape[2] == shape[3] == args.size, \
        f"exported input is {shape[2]}x{shape[3]}, asked for {args.size}"

    from npu.yolo_pose import HEAD_OUTS
    produced = {o for n in g.node for o in n.output}
    for n in HEAD_OUTS:
        print(f"  {'OK ' if n in produced else 'MISSING'} {n}")
    if not all(n in produced for n in HEAD_OUTS):
        print("\nHead node names differ from expected. Pose head convs are:")
        for n in g.node:
            if "/model.22/" in n.name and n.op_type == "Conv":
                print(f"    {n.op_type:9s} {n.name} -> {n.output[0]}")
        print("Paste that list and we'll pick the right cv2/cv3/cv4 outputs.")
    else:
        print("\nOK - ready for 1b_cut_head.py")


if __name__ == "__main__":
    main()
