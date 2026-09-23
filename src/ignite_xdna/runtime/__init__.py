# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ignite_xdna.runtime: Hardware runtime, PyXRT driver, ERT ring scheduler, and parity verification.
"""

from .driver import XrtSiliconHarness, setup_xrt_environment
from .parity import (
    srs_s8_s32,
    Im2ColGoldenReference,
    calculate_numerical_parity,
    unblock_aie2_egress,
)
from .ring_scheduler import (
    BufferSet,
    profile_hardware_execution,
    profile_pipelined_hardware_execution,
)
from .profiler import (
    HardwareEventProfiler,
    HardwareTimestamps,
    PartitionProfileRecord,
    map_yolo_node_to_stage,
    CAT_HOST_DISPATCH,
    CAT_DDR_BOUNCE,
    CAT_SPATIAL_STEM,
    CAT_HIGH_CHANNEL,
    CAT_CPU_FALLBACK,
)
from .session import InferenceSession, RunHandle
from .splice import DeviceBuffer, KernelSplicer, KernelStage, SpliceResult, TensorSpec

__all__ = [
    "XrtSiliconHarness",
    "setup_xrt_environment",
    "srs_s8_s32",
    "Im2ColGoldenReference",
    "calculate_numerical_parity",
    "unblock_aie2_egress",
    "BufferSet",
    "profile_hardware_execution",
    "profile_pipelined_hardware_execution",
    "HardwareEventProfiler",
    "HardwareTimestamps",
    "PartitionProfileRecord",
    "map_yolo_node_to_stage",
    "CAT_HOST_DISPATCH",
    "CAT_DDR_BOUNCE",
    "CAT_SPATIAL_STEM",
    "CAT_HIGH_CHANNEL",
    "CAT_CPU_FALLBACK",
    "InferenceSession",
    "RunHandle",
    "DeviceBuffer",
    "KernelSplicer",
    "KernelStage",
    "SpliceResult",
    "TensorSpec",
]
