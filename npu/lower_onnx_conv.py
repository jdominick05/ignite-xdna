#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
npu/lower_onnx_conv.py - Backwards-compatibility re-export shim.
Forwards to ignite_xdna.compiler.lower_onnx_conv.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ignite_xdna.compiler.lower_onnx_conv import (
    extract_conv_subgraph,
    pack_weights_aie2_vector_layout,
    pack_bias_aie2_vector_layout,
    emit_layer_init_binary,
    emit_layer_exec_binary,
    emit_layer_transaction_binary,
    emit_fused_2layer_transaction_binary,
    prepare_image_activations,
    unblock_aie2_egress,
    run_exact_fixed_point_reference,
    run_ort_cpu_reference,
    build_fused_2layer_onnx_subgraph,
    run_fused_2layer_ort_cpu_reference,
    run_fused_2layer_fixed_point_reference,
    execute_fused_2layer_on_silicon,
    execute_layer_on_silicon,
    calculate_parity,
    lower_and_execute_conv,
    build_arg_parser,
)

__all__ = [
    "extract_conv_subgraph",
    "pack_weights_aie2_vector_layout",
    "pack_bias_aie2_vector_layout",
    "emit_layer_init_binary",
    "emit_layer_exec_binary",
    "emit_layer_transaction_binary",
    "emit_fused_2layer_transaction_binary",
    "prepare_image_activations",
    "unblock_aie2_egress",
    "run_exact_fixed_point_reference",
    "run_ort_cpu_reference",
    "build_fused_2layer_onnx_subgraph",
    "run_fused_2layer_ort_cpu_reference",
    "run_fused_2layer_fixed_point_reference",
    "execute_fused_2layer_on_silicon",
    "execute_layer_on_silicon",
    "calculate_parity",
    "lower_and_execute_conv",
    "build_arg_parser",
]

if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    lower_and_execute_conv(args)
