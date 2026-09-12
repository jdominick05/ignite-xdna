#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/runtime/ring_scheduler.py
Asynchronous double-buffered ERT ring-buffer scheduler and hardware latency profiling.
"""

import time
from typing import Dict, Any, List, Tuple
import numpy as np
from .driver import XrtSiliconHarness


class BufferSet:
    """Encapsulates a set of host-device shared buffers for one pipeline slot."""
    def __init__(self, inputs: list, outputs: list, bos: list):
        self.inputs = inputs    # list of (bo, data_bytes)
        self.outputs = outputs  # list of bo
        self.bos = bos          # list of bo passed to kernel execution


def profile_hardware_execution(
    harness: XrtSiliconHarness,
    bo_instr,
    ninstr: int,
    *bos,
    num_ops: int = 0,
    num_cores: int = 1,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> Dict[str, Any]:
    """
    Benchmark hardware kernel execution latency over warm iterations.
    Calculates Effective TOPS and Core ALU Issue Density against 1.80 GHz tile clock.
    """
    # Warmup loop
    for _ in range(warmup_iters):
        run = harness.kernel(3, bo_instr, ninstr, *bos)
        state = run.wait(2000)
        if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            raise RuntimeError(f"Warmup dispatch failed with state: {state}")

    # Monotonic timing loop
    latencies_us = []
    for _ in range(bench_iters):
        t0 = time.perf_counter_ns()
        run = harness.kernel(3, bo_instr, ninstr, *bos)
        state = run.wait(2000)
        t1 = time.perf_counter_ns()
        if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            raise RuntimeError(f"Benchmark dispatch failed with state: {state}")
        latencies_us.append((t1 - t0) / 1e3)

    latencies_us = np.array(latencies_us)
    mean_us = float(np.mean(latencies_us))
    median_us = float(np.median(latencies_us))
    min_us = float(np.min(latencies_us))
    max_us = float(np.max(latencies_us))
    p95_us = float(np.percentile(latencies_us, 95))
    p99_us = float(np.percentile(latencies_us, 99))

    latency_sec = mean_us * 1e-6
    effective_tops = (num_ops / latency_sec) / 1e12 if latency_sec > 0 else 0.0

    peak_tops = num_cores * (1.80e9 * 256) / 1e12
    alu_issue_density_pct = (effective_tops / peak_tops) * 100.0 if peak_tops > 0 else 0.0

    return {
        "iters": bench_iters,
        "mean_us": mean_us,
        "median_us": median_us,
        "min_us": min_us,
        "max_us": max_us,
        "p95_us": p95_us,
        "p99_us": p99_us,
        "effective_tops": effective_tops,
        "peak_tops": peak_tops,
        "alu_issue_density_pct": alu_issue_density_pct,
    }


def profile_pipelined_hardware_execution(
    harness: XrtSiliconHarness,
    bo_instr,
    ninstr: int,
    ping_set: BufferSet,
    pong_set: BufferSet,
    num_ops: int = 0,
    num_cores: int = 1,
    warmup_iters: int = 50,
    bench_iters: int = 500,
) -> Dict[str, Any]:
    """
    Benchmark asynchronous double-buffered ring-buffer pipelined execution.
    Overlaps host memory DMA transfers (sync to/from device) and ERT ring-buffer
    command enqueueing with ongoing physical NPU kernel execution.
    """
    sets = [ping_set, pong_set]

    # Warmup pipeline
    for bo, data in sets[0].inputs:
        bo.write(data, 0)
        bo.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    run_prev = harness.kernel(3, bo_instr, ninstr, *sets[0].bos)

    for i in range(1, warmup_iters):
        curr_idx = i % 2
        prev_idx = (i - 1) % 2
        curr_set = sets[curr_idx]
        prev_set = sets[prev_idx]

        for bo, data in curr_set.inputs:
            bo.write(data, 0)
            bo.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run_curr = harness.kernel(3, bo_instr, ninstr, *curr_set.bos)

        state = run_prev.wait(2000)
        if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            raise RuntimeError(f"Warmup pipelined dispatch failed with state: {state}")
        for bo in prev_set.outputs:
            bo.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        run_prev = run_curr

    state = run_prev.wait(2000)
    for bo in sets[(warmup_iters - 1) % 2].outputs:
        bo.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    # Monotonic timing benchmark over bench_iters
    latencies_us = []
    t0_wall = time.perf_counter_ns()

    # Prologue: Frame 0
    for bo, data in sets[0].inputs:
        bo.write(data, 0)
        bo.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    run_prev = harness.kernel(3, bo_instr, ninstr, *sets[0].bos)

    for i in range(1, bench_iters):
        t_start = time.perf_counter_ns()
        curr_idx = i % 2
        prev_idx = (i - 1) % 2
        curr_set = sets[curr_idx]
        prev_set = sets[prev_idx]

        for bo, data in curr_set.inputs:
            bo.write(data, 0)
            bo.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run_curr = harness.kernel(3, bo_instr, ninstr, *curr_set.bos)

        state = run_prev.wait(2000)
        if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            raise RuntimeError(f"Pipelined dispatch failed with state: {state}")
        for bo in prev_set.outputs:
            bo.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        run_prev = run_curr
        t_end = time.perf_counter_ns()
        latencies_us.append((t_end - t_start) / 1e3)

    # Epilogue: drain last frame
    state = run_prev.wait(2000)
    if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
        raise RuntimeError(f"Pipelined dispatch drain failed with state: {state}")
    for bo in sets[(bench_iters - 1) % 2].outputs:
        bo.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    t1_wall = time.perf_counter_ns()

    wall_total_us = (t1_wall - t0_wall) / 1e3
    effective_per_iter_us = wall_total_us / bench_iters
    fps = bench_iters / ((t1_wall - t0_wall) * 1e-9)
    effective_tops = (num_ops * fps) / 1e12

    peak_tops = num_cores * (1.80e9 * 256) / 1e12
    alu_issue_density_pct = (effective_tops / peak_tops) * 100.0 if peak_tops > 0 else 0.0

    l_arr = np.array(latencies_us) if latencies_us else np.array([effective_per_iter_us])
    return {
        "iters": bench_iters,
        "wall_total_us": wall_total_us,
        "effective_per_iter_us": effective_per_iter_us,
        "fps": fps,
        "mean_step_us": float(np.mean(l_arr)),
        "median_step_us": float(np.median(l_arr)),
        "min_step_us": float(np.min(l_arr)),
        "max_step_us": float(np.max(l_arr)),
        "p95_step_us": float(np.percentile(l_arr, 95)),
        "p99_step_us": float(np.percentile(l_arr, 99)),
        "effective_tops": effective_tops,
        "peak_tops": peak_tops,
        "alu_issue_density_pct": alu_issue_density_pct,
    }
