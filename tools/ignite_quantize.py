#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
tools/ignite_quantize.py

Standalone Python Post-Training Quantization (PTQ) CLI for AMD Phoenix XDNA1 (AIE2).
Eliminates external Docker and Vitis-AI dependencies by implementing Cross-Layer
Equalization (CLE) and Adaptive Rounding (AdaRound) in native Python/NumPy.

Produces quantized ONNX and scale artifacts that feed directly into tools/ignite_compile.py.

Usage:
  python tools/ignite_quantize.py \\
    --model models/yolov8n.onnx \\
    --calib-data data/coco128/ \\
    --output build/yolov8n_quant.onnx \\
    --scales build/scales.json
"""

import argparse
import os
from pathlib import Path
import sys
import time

# Ensure repository root and src directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ignite_xdna.quantization import PTQEngine, QuantizationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Python PTQ Engine (CLE + AdaRound) for AMD Phoenix XDNA1"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to input float ONNX model (e.g. models/yolov8n.onnx or models/yolov8n_cut.onnx)",
    )
    parser.add_argument(
        "--calib-data",
        type=str,
        default="data/coco128/",
        help="Path to calibration dataset directory containing JPEG images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="build/yolov8n_quant.onnx",
        help="Path to write the quantized ONNX model",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="build/scales.json",
        help="Path to write the JSON quantization scales metadata",
    )
    parser.add_argument(
        "--cle",
        dest="cle",
        action="store_true",
        default=True,
        help="Enable Cross-Layer Equalization (CLE) (default: True)",
    )
    parser.add_argument(
        "--no-cle",
        dest="cle",
        action="store_false",
        help="Disable Cross-Layer Equalization",
    )
    parser.add_argument(
        "--adaround",
        dest="adaround",
        action="store_true",
        default=True,
        help="Enable AdaRound optimization (default: True)",
    )
    parser.add_argument(
        "--no-adaround",
        dest="adaround",
        action="store_false",
        help="Disable AdaRound optimization",
    )
    parser.add_argument(
        "--num-calib",
        type=int,
        default=128,
        help="Number of calibration images to ingest (default: 128)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of AdaRound optimization iterations per layer (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="AdaRound learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Directly invoke tools/ignite_compile.py to produce a deployable .ignite container",
    )
    parser.add_argument(
        "--ignite-output",
        type=str,
        default="build/yolov8n.ignite",
        help="Output path for .ignite container if --compile is specified",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("================================================================================")
    print("      IGNITE-XDNA: Standalone Python Post-Training Quantization Engine          ")
    print("      Transforms: Cross-Layer Equalization (CLE) + Adaptive Rounding (AdaRound) ")
    print("      Platform: AMD Phoenix XDNA1 (AIE2) Architecture                          ")
    print("================================================================================\n")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[!] ERROR: Input model file not found: {model_path}")
        sys.exit(1)

    calib_dir = Path(args.calib_data)
    if args.adaround and not calib_dir.exists():
        print(f"[!] ERROR: Calibration data directory not found: {calib_dir}")
        sys.exit(1)

    config = QuantizationConfig(
        use_cle=args.cle,
        use_adaround=args.adaround,
        num_calib=args.num_calib,
        adaround_iterations=args.iterations,
        adaround_lr=args.lr,
    )

    engine = PTQEngine(config=config)
    summary = engine.quantize(
        model_path=model_path,
        calib_data_dir=calib_dir,
        output_onnx_path=args.output,
        output_scales_path=args.scales,
    )

    print("\n--------------------------------------------------------------------------------")
    print("                       QUANTIZATION SUMMARY                                     ")
    print("--------------------------------------------------------------------------------")
    print(f"  Model Output:           {summary['model_path']}")
    print(f"  Scales Output:          {summary['scales_path']}")
    print(f"  CLE Pairs Equalized:    {summary['cle_pairs']}")
    print(f"  CLE Biases Absorbed:    {summary['cle_biases_absorbed']}")
    print(f"  AdaRound Layers:        {summary['adaround_layers']}")
    print(f"  Total Runtime:          {summary['elapsed_seconds']}s")
    print("--------------------------------------------------------------------------------\n")

    if args.compile:
        print("[*] Invoking tools/ignite_compile.py to generate binary container...")
        from ignite_xdna.compiler.cli import compile_model
        compile_model(
            input_path=summary['model_path'],
            output_path=args.ignite_output,
            quant_scales_path=summary['scales_path'],
        )
        print(f"[+] Successfully compiled binary container: {args.ignite_output}")


if __name__ == "__main__":
    main()
