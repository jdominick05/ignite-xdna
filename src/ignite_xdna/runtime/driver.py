#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/runtime/driver.py
PyXRT runtime driver abstractions and buffer object (BO) lifecycle management on AMD Phoenix AIE2.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Any


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def setup_xrt_environment():
    """Register Windows XRT SDK search paths and DLL directories."""
    dll_paths = [
        r'C:\Xilinx\XRT\xrt_sdk\xrt\lib',
        r'C:\Windows\System32',
        r'C:\Windows\System32\AMD',
    ]
    for p in dll_paths:
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
    sdk_python = r'C:\Xilinx\XRT\xrt_sdk\xrt\python'
    if os.path.isdir(sdk_python) and sdk_python not in sys.path:
        sys.path.insert(0, sdk_python)


class XrtSiliconHarness:
    """
    Manages physical AMD Phoenix XDNA1 NPU device context, XCLBIN registration,
    and asynchronous / synchronous ERT command submission via pyxrt.
    """
    def __init__(self, device_idx: int = 0):
        setup_xrt_environment()
        import pyxrt
        self.pyxrt = pyxrt
        self.device_idx = device_idx
        self.dev = pyxrt.device(device_idx)
        self.context = None
        self.kernel = None
        self.xclbin = None

    def load_xclbin(self, xclbin_path: str, kernel_name: str = "MLIR_AIE"):
        """Load and register XCLBIN on Phoenix silicon."""
        if not os.path.isabs(xclbin_path):
            xclbin_path = str(get_repo_root() / xclbin_path)
        if not os.path.exists(xclbin_path):
            raise FileNotFoundError(f"XCLBIN not found: {xclbin_path}")
        self.kernel = None
        self.context = None
        self.xclbin = self.pyxrt.xclbin(xclbin_path)
        uuid = self.dev.register_xclbin(self.xclbin)
        self.context = self.pyxrt.hw_context(self.dev, uuid)
        self.kernel = self.pyxrt.kernel(self.context, kernel_name)
        return self.kernel

    def create_instruction_bo(self, bin_path: str) -> Tuple[Any, int]:
        """Create cacheable instruction buffer for compiled NPU transaction stream."""
        if not os.path.isabs(bin_path):
            bin_path = str(get_repo_root() / bin_path)
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Instruction binary not found: {bin_path}")
        with open(bin_path, "rb") as f:
            insts_bytes = f.read()
        bo_instr = self.pyxrt.bo(self.dev, len(insts_bytes), self.pyxrt.bo.cacheable, self.kernel.group_id(1))
        bo_instr.write(insts_bytes, 0)
        bo_instr.sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return bo_instr, len(insts_bytes)

    def create_host_bo(self, size_bytes: int, group_id: int):
        """Create host-visible shared memory buffer object."""
        return self.pyxrt.bo(self.dev, size_bytes, self.pyxrt.bo.host_only, self.kernel.group_id(group_id))

    def create_double_buffered_pair(self, size_bytes: int, group_id: int):
        """Allocate a ping-pong pair of host-visible shared memory buffer objects."""
        bo_0 = self.create_host_bo(size_bytes, group_id)
        bo_1 = self.create_host_bo(size_bytes, group_id)
        return bo_0, bo_1

    def dispatch_kernel(self, bo_instr, num_instr_bytes: int, *bos, timeout_ms: int = 2000):
        """
        Submit ERT command to ring buffer and await completion.
        Opcode 3 corresponds to standard MLIR_AIE opcode dispatch.
        """
        run = self.kernel(3, bo_instr, num_instr_bytes, *bos)
        state = run.wait(timeout_ms)
        return run, state
