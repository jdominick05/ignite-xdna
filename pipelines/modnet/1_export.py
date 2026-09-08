"""
Step 1: Export MODNet portrait matting network to ONNX for Ryzen AI (XDNA1).

Produces:
  models/modnet/modnet_fp32.onnx           - FP32 graph, opset 17, static shapes
  models/modnet/preprocess_config.json     - Preprocessing parameters

Run in resnet_env:
    python pipelines/modnet/1_export.py
    python pipelines/modnet/1_export.py --size 256 --out models/modnet/modnet_r256_fp32.onnx
    python pipelines/modnet/1_export.py --ckpt models/modnet/modnet_webcam_portrait_matting.ckpt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.modnet.model import MODNet

DEFAULT_CKPT = "models/modnet/modnet_photographic_portrait_matting.ckpt"
DEFAULT_OUT = "models/modnet/modnet_fp32.onnx"
DEFAULT_CFG = "models/modnet/preprocess_config.json"


def main():
    ap = argparse.ArgumentParser(description="Export MODNet to static ONNX opset 17.")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help=f"Path to .ckpt file (default: {DEFAULT_CKPT})")
    ap.add_argument("--size", type=int, default=512, help="Input side H=W (default: 512)")
    ap.add_argument("--batch", type=int, default=1, help="Static batch size (default: 1)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"Output ONNX path (default: {DEFAULT_OUT})")
    ap.add_argument("--cfg-out", default=DEFAULT_CFG, help=f"Output config path (default: {DEFAULT_CFG})")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.cfg_out)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1_export] Initializing MODNet...")
    model = MODNet(in_channels=3, hr_channels=32)
    print(f"[1_export] Loading checkpoint from {args.ckpt}...")
    model.load_ckpt(args.ckpt)
    model.eval()

    dummy_input = torch.randn(args.batch, 3, args.size, args.size, dtype=torch.float32)

    print(f"[1_export] Exporting to ONNX (opset {args.opset}, shape [{args.batch}, 3, {args.size}, {args.size}])...")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # Strictly static for VitisAI EP
    )

    print(f"[1_export] Export complete: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")

    cfg = {
        "model_name": "modnet",
        "input_name": "input",
        "output_name": "output",
        "input_size": [args.batch, 3, args.size, args.size],
        "height": args.size,
        "width": args.size,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "rescale": 1.0 / 255.0,
        "interpolation": "bilinear",
        "opset": args.opset,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[1_export] Preprocessing config written: {cfg_path}")


if __name__ == "__main__":
    main()
