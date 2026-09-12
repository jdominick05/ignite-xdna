# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
ignite_xdna.compiler: Compiler lowering, ONNX subgraph extraction, CDO transaction generation.
"""

from .lower_onnx_conv import (
    extract_conv_subgraph,
    pack_weights_aie2_vector_layout,
    pack_bias_aie2_vector_layout,
    emit_layer_init_binary,
    emit_layer_exec_binary,
    emit_layer_transaction_binary,
    emit_fused_2layer_transaction_binary,
    prepare_image_activations,
    unblock_aie2_egress,
    execute_layer_on_silicon,
    execute_fused_2layer_on_silicon,
    lower_and_execute_conv,
)
from .generate_fused_mlir import generate_fused_mlir

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
    "execute_layer_on_silicon",
    "execute_fused_2layer_on_silicon",
    "lower_and_execute_conv",
    "generate_fused_mlir",
]
