#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
tools/generate_fused_mlir.py - Re-export wrapper for ignite_xdna.compiler.generate_fused_mlir.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ignite_xdna.compiler.generate_fused_mlir import generate_fused_mlir

if __name__ == "__main__":
    generate_fused_mlir()
