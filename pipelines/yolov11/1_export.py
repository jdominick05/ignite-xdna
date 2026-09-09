"""
YOLOv11 step 1: export yolo11n to ONNX for Ryzen AI, then verify the graph.

    python pipelines/yolov11/1_export.py              # stock yolo11n, 640
    python pipelines/yolov11/1_export.py --no-c2psa   # ablate C2PSA attention block

Run in the env that has torch + ultralytics (resnet_env).
"""
import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(MODELS / "yolo11n.pt"))
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--no-c2psa", action="store_true",
                    help="ablate the C2PSA self-attention block (layer 10) with Identity "
                         "for attention vs pure-CNN control characterization")
    ap.add_argument("--out", default=None,
                    help="target export path (defaults to models/yolo11n.onnx or "
                         "models/yolo11n_no_c2psa.onnx)")
    args = ap.parse_args()

    default_name = "yolo11n_no_c2psa.onnx" if args.no_c2psa else "yolo11n.onnx"
    target_out = args.out or str(MODELS / default_name)

    from ultralytics import YOLO
    model = YOLO(args.weights)

    if args.no_c2psa:
        print("Ablating C2PSA attention block at layer 10...")
        ident = torch.nn.Identity()
        ident.f = model.model.model[10].f
        ident.i = model.model.model[10].i
        ident.type = "torch.nn.Identity"
        model.model.model[10] = ident

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

    assert any(v == 17 for _, v in opsets), "opset is not 17"
    assert shape[0] == 1, "batch is not static 1"
    assert shape[2] == shape[3] == args.size, \
        f"exported input is {shape[2]}x{shape[3]}, asked for {args.size}"

    # Verify detect head exists
    cv_nodes = [n.name for n in g.node if "/model.23/" in n.name and n.op_type == "Conv"]
    print(f"Detect head Conv nodes (/model.23/): {len(cv_nodes)}")
    assert len(cv_nodes) >= 6, f"Expected at least 6 detect head Conv nodes, found {len(cv_nodes)}"
    print(f"\nOK - ready for step 1b (cut head) -> {path}")


if __name__ == "__main__":
    main()
