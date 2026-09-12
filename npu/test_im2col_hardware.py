#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
npu/test_im2col_hardware.py - Backwards-compatibility re-export shim.
Forwards to ignite_xdna.runtime.test_im2col_hardware.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ignite_xdna.runtime.driver import setup_xrt_environment, XrtSiliconHarness
from ignite_xdna.runtime.parity import srs_s8_s32, Im2ColGoldenReference, calculate_numerical_parity
from ignite_xdna.runtime.ring_scheduler import BufferSet, profile_hardware_execution, profile_pipelined_hardware_execution
from ignite_xdna.runtime.test_im2col_hardware import run_hardware_im2col_harness

__all__ = [
    "setup_xrt_environment",
    "srs_s8_s32",
    "Im2ColGoldenReference",
    "XrtSiliconHarness",
    "calculate_numerical_parity",
    "profile_hardware_execution",
    "BufferSet",
    "profile_pipelined_hardware_execution",
    "run_hardware_im2col_harness",
]

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="AMD Phoenix AIE2 Hardware im2col Execution Harness")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark timed iterations for synchronous baseline")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for synchronous baseline")
    parser.add_argument("--pipe-iters", type=int, default=500, help="Benchmark timed iterations for double-buffered pipeline")
    parser.add_argument("--pipe-warmup", type=int, default=50, help="Warmup iterations for double-buffered pipeline")
    parser.add_argument("--log-file", type=str, default="results/aie/hardware_im2col_pipelining.log", help="Path to write execution log")
    parser.add_argument("--fused-2layer", action="store_true", default=False, help="Execute 2-layer MemTile L2 activation ping-pong fused pipeline on Phoenix silicon")
    parser.add_argument("--test-20core", action="store_true", default=False, help="Include experimental 20-core array stress test")
    args = parser.parse_args()

    if args.fused_2layer:
        try:
            from ignite_xdna.compiler.lower_onnx_conv import lower_and_execute_conv
        except (ImportError, ModuleNotFoundError):
            from npu.lower_onnx_conv import lower_and_execute_conv

        fused_args = argparse.Namespace(
            model=str(_repo_root / "models" / "yolov8n_cut_xint8.onnx"),
            node_name=None,
            conv_index=None,
            base_txn=str(_repo_root / "build" / "im2col_4d_16core_clean.bin"),
            xclbin=str(_repo_root / "build" / "im2col_4d_16core.xclbin"),
            image=str(_repo_root / "data" / "bisenetv2_calib" / "000000000139.jpg"),
            iters=args.pipe_iters,
            warmup=args.pipe_warmup,
            device_idx=0,
            log_dir=str(_repo_root / "results" / "aie"),
            log_name="hardware_fused_layer_verification.log",
            fused_2layer=True
        )
        lower_and_execute_conv(fused_args)
        sys.stdout.flush()
        os._exit(0)

    run_hardware_im2col_harness(
        iters=args.iters,
        warmup=args.warmup,
        pipe_iters=args.pipe_iters,
        pipe_warmup=args.pipe_warmup,
        log_path=args.log_file,
        test_20core=args.test_20core,
    )
    sys.stdout.flush()
