# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
ignite_xdna.runtime: Hardware runtime, PyXRT driver, ERT ring scheduler, and parity verification.
"""

from .driver import XrtSiliconHarness, setup_xrt_environment
from .parity import (
    srs_s8_s32,
    Im2ColGoldenReference,
    calculate_numerical_parity,
)
from .ring_scheduler import (
    BufferSet,
    profile_hardware_execution,
    profile_pipelined_hardware_execution,
)

__all__ = [
    "XrtSiliconHarness",
    "setup_xrt_environment",
    "srs_s8_s32",
    "Im2ColGoldenReference",
    "calculate_numerical_parity",
    "BufferSet",
    "profile_hardware_execution",
    "profile_pipelined_hardware_execution",
]
