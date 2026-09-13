#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
tools/ignite_compile.py

One-Click Bare-Metal Model Compiler CLI for AMD Phoenix XDNA1 NPU.
Ingests an arbitrary YOLOv8n ONNX model, validates operator compatibility,
partitions the DAG into monolithic MemTile stages (0 CPU fallback partitions),
synthesizes 64-byte aligned CDO transaction streams, and serializes everything
into a single deployable .ignite binary container.

Usage:
  python tools/ignite_compile.py --input models/yolov8n_cut_xint8.onnx --output build/yolov8n.ignite --verify-silicon
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ignite_xdna.compiler.cli import (
    main,
    compile_model,
    validate_onnx_compatibility,
    verify_on_silicon,
    parse_args,
    SUPPORTED_OPERATORS,
)


if __name__ == "__main__":
    main()
