# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
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
    run_exact_fixed_point_reference,
    run_fused_2layer_fixed_point_reference,
    run_fused_2layer_ort_cpu_reference,
)
from .generate_fused_mlir import generate_fused_mlir
from .partitioner import (
    GraphPartitioner,
    ConvLayerMeta,
    NpuFusedPartition,
    CpuFallbackPartition,
    PartitionedGraph,
    build_synthetic_multi_layer_conv_model,
)
from .topology import TopologyFinding, audit_model, enforce
from .scheduler import (
    MemTileMultiPassScheduler,
    SchedulePlan,
    PassDescriptor,
    emit_multi_layer_transaction_bundle,
    emit_multi_stage_transaction_bundle,
    emit_unified_monolithic_transaction_bundle,
    chain_stage_transaction_streams,
    run_n_layer_fixed_point_reference,
    run_n_layer_ort_cpu_reference,
    execute_multi_layer_on_silicon,
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
    "execute_layer_on_silicon",
    "execute_fused_2layer_on_silicon",
    "lower_and_execute_conv",
    "run_exact_fixed_point_reference",
    "run_fused_2layer_fixed_point_reference",
    "run_fused_2layer_ort_cpu_reference",
    "generate_fused_mlir",
    "GraphPartitioner",
    "ConvLayerMeta",
    "NpuFusedPartition",
    "CpuFallbackPartition",
    "PartitionedGraph",
    "build_synthetic_multi_layer_conv_model",
    "TopologyFinding",
    "audit_model",
    "enforce",
    "MemTileMultiPassScheduler",
    "SchedulePlan",
    "PassDescriptor",
    "emit_multi_layer_transaction_bundle",
    "emit_multi_stage_transaction_bundle",
    "emit_unified_monolithic_transaction_bundle",
    "chain_stage_transaction_streams",
    "run_n_layer_fixed_point_reference",
    "run_n_layer_ort_cpu_reference",
    "execute_multi_layer_on_silicon",
]
