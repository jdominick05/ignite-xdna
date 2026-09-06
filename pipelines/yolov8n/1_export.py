"""
YOLO step 1: export yolov8n to ONNX for Ryzen AI, then verify the graph.

    python pipelines/yolov8n/1_export.py            # yolov8n, 640
    python pipelines/yolov8n/1_export.py --size 416 # smaller input = faster on NPU

Run in the env that has torch + ultralytics (resnet_env).
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
    ap.add_argument("--weights", default=str(MODELS / "yolov8n.pt"))
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--out", default=None,
                    help="move the export here. Ultralytics names the .onnx "
                         "after the .pt regardless of --size, so exporting the "
                         "same weights at a second resolution would silently "
                         "overwrite the first. A resolution sweep needs this.")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)  # downloads if missing

    # Ultralytics always writes <weights>.onnx, whatever --size says, and then
    # --out moves it. Without this dance, exporting yolov8n at 512 would both
    # overwrite models/yolov8n.onnx AND move it away, deleting the 640 export
    # that yolo-bench.sh uses as its float CPU baseline. Stash anything already
    # sitting on that path and put it back afterwards.
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
        # Restore even if the export raised, so a failed run cannot eat the
        # model that was already there.
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

    # The post-processing subgraph we keep in float for quantization.
    # These names come from an onnxslim'd ultralytics export; verify they exist.
    head_start = ["/model.22/Concat", "/model.22/Concat_1"]
    head_end = ["/model.22/Concat_3"]
    names = {n.name for n in g.node}
    for n in head_start + head_end:
        print(f"  {'OK ' if n in names else 'MISSING'} {n}")
    if not all(n in names for n in head_start + head_end):
        print("\nHead node names differ from expected. Detect head nodes are:")
        for n in g.node:
            if "/model.22/" in n.name and n.op_type in ("Concat", "Sigmoid", "Softmax"):
                print(f"    {n.op_type:9s} {n.name}")
        print("Paste that list and we'll pick the right start/end nodes.")
    else:
        print("\nOK - ready for y2 (COCO data) and y3 (quantize)")


if __name__ == "__main__":
    main()
