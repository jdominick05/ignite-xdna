"""
MiDaS step 1: Export MiDaS v2.1 Small monocular depth estimation to ONNX
for Ryzen AI, optimize graph structure, and emit head-cut variants.

Exports:
  models/midas_small_fp32.onnx          -- Stock PyTorch export (opset 17, static 1x3x256x256)
  models/midas_small_cut.onnx           -- Head-cut variant without final Squeeze, raw (1,1,256,256)
  models/midas_small_nearest_cut.onnx   -- NPU-optimized nearest-neighbor resize variant (single DPU subgraph)
  models/preprocess_config_midas.json   -- Preprocessing config metadata

Run in resnet_env (has torch + onnx + onnxslim):
    python pipelines/midas/1_export.py
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import onnx
import onnxslim
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.midas import INPUT_SIZE
from npu.paths import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=INPUT_SIZE, help="Input square resolution (default: 256)")
    ap.add_argument("--out-dir", default=str(MODELS), help="Directory to save exported ONNX models")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MiDaS_small from torch.hub (isl-org/MiDaS)...")
    model = torch.hub.load("isl-org/MiDaS", "MiDaS_small", pretrained=True)
    model.eval()

    dummy = torch.randn(1, 3, args.size, args.size)
    stock_path = out_dir / "midas_small_fp32.onnx"

    print(f"Exporting stock FP32 model to {stock_path} (opset 17, batch=1, static {args.size}x{args.size})...")
    torch.onnx.export(
        model,
        dummy,
        str(stock_path),
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["depth"],
        dynamic_axes=None,
    )

    print("Simplifying exported graph with onnxslim...")
    slim_model = onnxslim.slim(str(stock_path))

    # The output node before the final Squeeze is the raw (1, 1, H, W) depth map
    output_conv_node = None
    for node in reversed(slim_model.graph.node):
        if node.op_type == "Relu" and "output_conv" in node.name:
            output_conv_node = node
            break

    if output_conv_node is None:
        raise SystemExit("Could not identify output_conv Relu node before Squeeze in slim model")

    raw_output_name = output_conv_node.output[0]
    print(f"Cutting head at raw depth output: {raw_output_name}")

    cut_path = out_dir / "midas_small_cut.onnx"
    temp_slim = out_dir / "temp_midas_slim.onnx"
    onnx.save(slim_model, str(temp_slim))
    onnx.utils.extract_model(str(temp_slim), str(cut_path), ["image"], [raw_output_name])
    if temp_slim.is_file():
        temp_slim.unlink()

    cut_model = onnx.load(str(cut_path))
    onnx.checker.check_model(cut_model)
    print(f"Wrote {cut_path}: {len(cut_model.graph.node)} nodes")
    print("  Ops:", Counter(n.op_type for n in cut_model.graph.node).most_common())

    # Build the NPU-optimized nearest-neighbor variant
    # VitisAI EP rejects bilinear Resize with align_corners to CPU (causing 5 DPU subgraphs),
    # whereas nearest-neighbor Resize compiles natively to AIE in a single monolithic subgraph.
    nearest_path = out_dir / "midas_small_nearest_cut.onnx"
    nearest_model = onnx.load(str(cut_path))
    for n in nearest_model.graph.node:
        if n.op_type == "Resize":
            attrs = {a.name: a for a in n.attribute}
            if "mode" in attrs:
                attrs["mode"].s = b"nearest"
            if "coordinate_transformation_mode" in attrs:
                attrs["coordinate_transformation_mode"].s = b"asymmetric"
            if "nearest_mode" in attrs:
                attrs["nearest_mode"].s = b"floor"
            else:
                n.attribute.append(onnx.helper.make_attribute("nearest_mode", "floor"))

    onnx.save(nearest_model, str(nearest_path))
    onnx.checker.check_model(nearest_model)
    print(f"Wrote {nearest_path} (nearest-neighbor resize for single DPU subgraph offload)")

    cfg = {
        "architecture": "midas_v21_small",
        "input_name": "image",
        "input_size": [1, 3, args.size, args.size],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "output_name": raw_output_name,
        "output_shape": [1, 1, args.size, args.size],
    }
    cfg_path = out_dir / "preprocess_config_midas.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote preprocessing config: {cfg_path}")


if __name__ == "__main__":
    main()
