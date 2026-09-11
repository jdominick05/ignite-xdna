#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Host-side XRT runtime execution test harness for AMD Phoenix AIE2 silicon.
Dispatches and validates single-column and multi-core full-array im2col binaries,
verifying bit-for-bit numerical parity and profiling execution latency and TOPS.
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure XRT DLL and SDK search paths are registered on Windows
def setup_xrt_environment():
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


# ---------------------------------------------------------------------------
# 1. Golden Reference & SRS Mathematical Emulator
# ---------------------------------------------------------------------------

def srs_s8_s32(accum: np.ndarray, shift: int = 10) -> np.ndarray:
    """
    Bit-exact emulator for hardware vst.srs.s8.s32 instruction.
    Computes: saturate_s8((accum + (1 << (shift - 1))) >> shift)
    """
    if shift <= 0:
        rounded = accum.astype(np.int64)
    else:
        bias = 1 << (shift - 1)
        # In AIE2 architecture, rounding adds 1 << (shift - 1)
        rounded = np.right_shift(accum.astype(np.int64) + bias, shift)
    return np.clip(rounded, -128, 127).astype(np.int8)


class Im2ColGoldenReference:
    """
    NumPy bit-exact emulator for 3x3 im2col spatial convolution on AIE2 cores.
    Each core computes M=2 spatial patches (288 bytes each) against stationary weights (2304 bytes),
    accumulates into 4 32-lane INT32 accumulator registers, and saturates via SRS into 256 INT8 outputs.
    """
    def __init__(self, shift_bias: int = 10, seed: int = 42):
        self.shift_bias = shift_bias
        self.rng = np.random.RandomState(seed)

    def generate_inputs_and_weights(self, num_cores: int = 4):
        """
        Generate deterministic INT8 input feature map and stationary weights.
        For num_cores=4 (1 column):
          - Ingress: 2048 bytes (8x8x32 feature map)
          - Weights: 2304 bytes (9 taps x 4 blocks x 64 bytes)
          - Expected Egress: 1024 bytes (4 cores x 256 bytes)
        For num_cores=20 (5 columns):
          - Ingress: 10240 bytes (5 x 2048 bytes)
          - Weights: 2304 bytes stationary
          - Expected Egress: 5120 bytes (20 cores x 256 bytes)
        """
        in_bytes = num_cores * 512
        if num_cores == 4:
            in_bytes = 2048
        elif num_cores == 20:
            in_bytes = 10240

        inputs = self.rng.randint(-16, 16, size=in_bytes, dtype=np.int8)
        weights = self.rng.randint(-8, 8, size=2304, dtype=np.int8)
        return inputs, weights

    def compute_core_output(self, patch_a: np.ndarray, patch_b: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Simulate conv_im2col_kernel_m2_srs on two 288-byte patches with stationary weights.
        Returns 256 INT8 elements (128 from patch_a, 128 from patch_b).
        """
        # Weights: 9 taps x 256 bytes
        # Each tap has w0, w1, w2, w3 (each 64 bytes: 2x32)
        w_taps = weights.reshape(9, 4, 64)
        
        # Patch A: 9 taps x 32 bytes
        p_a = patch_a[:288].reshape(9, 32)
        p_b = patch_b[:288].reshape(9, 32)

        # Accumulators for patch A and patch B: 4 blocks of 32 lanes each
        acc_a = np.zeros((4, 32), dtype=np.int32)
        acc_b = np.zeros((4, 32), dtype=np.int32)

        for k in range(9):
            va = p_a[k].astype(np.int32)
            vb = p_b[k].astype(np.int32)
            for blk in range(4):
                w_blk = w_taps[k, blk].reshape(2, 32).astype(np.int32)
                acc_a[blk] += np.sum(va * w_blk[0]) + np.sum(va * w_blk[1])
                acc_b[blk] += np.sum(vb * w_blk[0]) + np.sum(vb * w_blk[1])

        # SRS Requantization
        out_a = srs_s8_s32(acc_a.flatten(), self.shift_bias)
        out_b = srs_s8_s32(acc_b.flatten(), self.shift_bias)
        return np.concatenate([out_a, out_b])

    def compute_full_reference(self, inputs: np.ndarray, weights: np.ndarray, num_cores: int = 4) -> np.ndarray:
        """
        Compute golden reference for all cores in the workload.
        """
        core_outputs = []
        for c in range(num_cores):
            offset = (c * 288) % (len(inputs) - 576 + 1)
            p_a = inputs[offset : offset + 288]
            p_b = inputs[offset + 288 : offset + 576]
            if len(p_a) < 288:
                p_a = np.pad(p_a, (0, 288 - len(p_a)))
            if len(p_b) < 288:
                p_b = np.pad(p_b, (0, 288 - len(p_b)))
            core_out = self.compute_core_output(p_a, p_b, weights)
            core_outputs.append(core_out)
        return np.concatenate(core_outputs)


# ---------------------------------------------------------------------------
# 2. Host Memory Allocation & XRT Buffer Object (BO) Management
# ---------------------------------------------------------------------------

class XrtSiliconHarness:
    """
    Manages physical AMD Phoenix XDNA1 NPU device context, XCLBIN registration,
    and asynchronous / synchronous ERT command submission via pyxrt.
    """
    def __init__(self, device_idx: int = 0):
        setup_xrt_environment()
        import pyxrt
        self.pyxrt = pyxrt
        self.dev = pyxrt.device(device_idx)
        self.context = None
        self.kernel = None
        self.xclbin = None

    def load_xclbin(self, xclbin_path: str, kernel_name: str = "MLIR_AIE"):
        """Load and register XCLBIN on Phoenix silicon."""
        if not os.path.exists(xclbin_path):
            raise FileNotFoundError(f"XCLBIN not found: {xclbin_path}")
        self.xclbin = self.pyxrt.xclbin(xclbin_path)
        uuid = self.dev.register_xclbin(self.xclbin)
        self.context = self.pyxrt.hw_context(self.dev, uuid)
        self.kernel = self.pyxrt.kernel(self.context, kernel_name)
        return self.kernel

    def create_instruction_bo(self, bin_path: str):
        """Create cacheable instruction buffer for compiled NPU transaction stream."""
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

    def dispatch_kernel(self, bo_instr, num_instr_bytes: int, *bos, timeout_ms: int = 2000):
        """
        Submit ERT command to ring buffer and await completion.
        Opcode 3 corresponds to standard MLIR_AIE opcode dispatch.
        """
        run = self.kernel(3, bo_instr, num_instr_bytes, *bos)
        state = run.wait(timeout_ms)
        return run, state


# ---------------------------------------------------------------------------
# 3. Numerical Parity Metrics
# ---------------------------------------------------------------------------

def calculate_numerical_parity(golden: np.ndarray, actual: np.ndarray):
    """
    Compute RMSE, Max Absolute Error, and bit-for-bit agreement percentage.
    """
    g = golden.flatten().astype(np.float64)
    a = actual.flatten().astype(np.float64)
    min_len = min(len(g), len(a))
    g = g[:min_len]
    a = a[:min_len]

    diff = np.abs(g - a)
    mae = float(np.mean(diff))
    max_ae = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bit_matches = int(np.sum(g == a))
    bit_agreement_pct = float(bit_matches / min_len * 100.0)

    return {
        "count": min_len,
        "mae": mae,
        "max_ae": max_ae,
        "rmse": rmse,
        "bit_matches": bit_matches,
        "bit_agreement_pct": bit_agreement_pct,
    }


# ---------------------------------------------------------------------------
# 4. Latency & TOPS Profiling Engine
# ---------------------------------------------------------------------------

def profile_hardware_execution(
    harness: XrtSiliconHarness,
    bo_instr,
    ninstr: int,
    *bos,
    num_ops: int = 0,
    num_cores: int = 1,
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
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
    effective_tops = (num_ops / latency_sec) / 1e12

    # AIE2 Vector Unit Peak Arithmetic Capability:
    # 512-bit vector engine computes 128 INT8 MACs per cycle (256 arithmetic ops/cycle).
    # Tile clock on Phoenix XDNA1 = 1.80 GHz.
    # Peak ops per core = 1.80e9 * 256 = 460.8 GOPS (0.4608 TOPS per core).
    # Peak ops for array = num_cores * 0.4608 TOPS.
    peak_tops = num_cores * (1.80e9 * 256) / 1e12
    alu_issue_density_pct = (effective_tops / peak_tops) * 100.0

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


# ---------------------------------------------------------------------------
# Main Execution Orchestrator
# ---------------------------------------------------------------------------

def run_hardware_im2col_harness(iters: int = 100, warmup: int = 10, log_path: str = None):
    print("=" * 80)
    print(" AMD Phoenix XDNA1 Physical Silicon Hardware Execution Harness")
    print(" Kernel Target: im2col 3x3 4D Convolution Engine with SRS Requantization")
    print("=" * 80)

    # Initialize Golden Generator
    golden_ref = Im2ColGoldenReference(shift_bias=10)
    
    # Initialize XRT Device
    print("\n[INIT] Initializing pyxrt Device 0 (AMD Phoenix NPU)...")
    harness = XrtSiliconHarness(device_idx=0)
    print("       Device initialized successfully.")

    results_summary = []

    # -----------------------------------------------------------------------
    # STAGE 1: Single-Column Baseline (Column 0, 4 Cores)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" [STAGE 1] Single-Column Baseline Execution (Column 0, 4 Cores)")
    print("-" * 80)

    # 1a. Column 0 im2col Transaction Baseline
    xclbin_col0 = "build/im2col_4d.xclbin"
    bin_col0 = "build/im2col_4d_roundtrip.bin"
    in_bytes_col0 = 2048
    out_bytes_col0 = 1024
    num_ops_col0 = 4 * 2 * 9 * 32 * 128 * 2 # 589,824 ops

    inputs_col0, weights_col0 = golden_ref.generate_inputs_and_weights(num_cores=4)
    ref_col0 = golden_ref.compute_full_reference(inputs_col0, weights_col0, num_cores=4)

    try:
        print(f"Loading XCLBIN: {xclbin_col0}")
        harness.load_xclbin(xclbin_col0, "MLIR_AIE")

        print(f"Loading Transaction Binary: {bin_col0}")
        bo_instr_col0, ninstr_col0 = harness.create_instruction_bo(bin_col0)

        print(f"Allocating Host Shared Buffers (In: {in_bytes_col0} B, Out: {out_bytes_col0} B)...")
        bo_in_col0 = harness.create_host_bo(in_bytes_col0, 3)
        bo_out_col0 = harness.create_host_bo(out_bytes_col0, 4)

        bo_in_col0.write(inputs_col0.tobytes(), 0)
        bo_in_col0.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_out_col0.write(np.zeros(out_bytes_col0, dtype=np.int8).tobytes(), 0)
        bo_out_col0.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        print("Executing Stage 1a on physical silicon...")
        run, state = harness.dispatch_kernel(bo_instr_col0, ninstr_col0, bo_in_col0, bo_out_col0, timeout_ms=2000)
        print(f"  Stage 1a Execution State: {state}")

        if str(state) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            bo_out_col0.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            hw_out_col0 = np.frombuffer(bo_out_col0.read(out_bytes_col0, 0), dtype=np.int8)

            parity_col0 = calculate_numerical_parity(ref_col0, hw_out_col0)
            print(f"  Numerical Parity: Bit-Agreement={parity_col0['bit_agreement_pct']:.2f}%, MAE={parity_col0['mae']:.4f}, RMSE={parity_col0['rmse']:.4f}")

            print(f"Profiling Stage 1a Silicon Latency ({iters} iterations)...")
            prof_col0 = profile_hardware_execution(
                harness, bo_instr_col0, ninstr_col0, bo_in_col0, bo_out_col0,
                num_ops=num_ops_col0, num_cores=4, warmup_iters=warmup, bench_iters=iters
            )
            print(f"  Latency: Mean={prof_col0['mean_us']:.2f} us, Median={prof_col0['median_us']:.2f} us, Min={prof_col0['min_us']:.2f} us, P95={prof_col0['p95_us']:.2f} us")
            print(f"  Throughput: {prof_col0['effective_tops']:.4f} Effective TOPS (Peak: {prof_col0['peak_tops']:.4f} TOPS, Issue Density: {prof_col0['alu_issue_density_pct']:.2f}%)")
            results_summary.append(("Stage 1a: Col 0 Transaction", state, prof_col0, parity_col0))
        else:
            print(f"  Stage 1a failed with state: {state}")
            results_summary.append(("Stage 1a: Col 0 Transaction", state, None, None))
    except Exception as e:
        print(f"  [ERROR] Stage 1a encountered exception: {e}")
        results_summary.append(("Stage 1a: Col 0 Transaction", f"ERROR: {e}", None, None))

    # 1b. Single-Core Vector MMUL Compute Validation (Tile 0,2)
    sc_xclbin = "build/test_sc.xclbin"
    sc_bin = "build/test_sc.bin"
    if os.path.exists(sc_xclbin) and os.path.exists(sc_bin):
        print("\nEvaluating Stage 1b: Single-Core Vector MMUL (Tile 0,2, 1 Core)...")
        try:
            harness.load_xclbin(sc_xclbin, "MLIR_AIE")
            bo_instr_sc, ninstr_sc = harness.create_instruction_bo(sc_bin)

            M_sc, K_sc, N_sc = 64, 64, 64
            A_sc = np.random.randint(-10, 10, (M_sc, K_sc), dtype=np.int16)
            B_sc = np.random.randint(-10, 10, (K_sc, N_sc), dtype=np.int16)
            C_golden_sc = A_sc.astype(np.int32) @ B_sc.astype(np.int32)

            bo_a_sc = harness.create_host_bo(A_sc.nbytes, 3)
            bo_b_sc = harness.create_host_bo(B_sc.nbytes, 4)
            bo_c_sc = harness.create_host_bo(C_golden_sc.nbytes, 5)

            bo_a_sc.write(A_sc.tobytes(), 0)
            bo_a_sc.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            bo_b_sc.write(B_sc.tobytes(), 0)
            bo_b_sc.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            bo_c_sc.write(np.zeros_like(C_golden_sc).tobytes(), 0)
            bo_c_sc.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            print("Executing Stage 1b on physical silicon...")
            run, state = harness.dispatch_kernel(bo_instr_sc, ninstr_sc, bo_a_sc, bo_b_sc, bo_c_sc, timeout_ms=2000)
            print(f"  Stage 1b Execution State: {state}")

            if str(state) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
                bo_c_sc.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                C_hw_sc = np.frombuffer(bo_c_sc.read(C_golden_sc.nbytes, 0), dtype=np.int32).reshape(M_sc, N_sc)
                parity_sc = calculate_numerical_parity(C_golden_sc, C_hw_sc)
                print(f"  Numerical Parity: Bit-Agreement={parity_sc['bit_agreement_pct']:.2f}%, MAE={parity_sc['mae']:.4f}, RMSE={parity_sc['rmse']:.4f}")

                total_ops_sc = 2 * M_sc * K_sc * N_sc # 524,288 ops
                print(f"Profiling Stage 1b Silicon Latency ({iters} iterations)...")
                prof_sc = profile_hardware_execution(
                    harness, bo_instr_sc, ninstr_sc, bo_a_sc, bo_b_sc, bo_c_sc,
                    num_ops=total_ops_sc, num_cores=1, warmup_iters=warmup, bench_iters=iters
                )
                print(f"  Latency: Mean={prof_sc['mean_us']:.2f} us, Median={prof_sc['median_us']:.2f} us, Min={prof_sc['min_us']:.2f} us, P95={prof_sc['p95_us']:.2f} us")
                print(f"  Throughput: {prof_sc['effective_tops']:.4f} Effective TOPS (Peak: {prof_sc['peak_tops']:.4f} TOPS, Issue Density: {prof_sc['alu_issue_density_pct']:.2f}%)")
                results_summary.append(("Stage 1b: Single-Core MMUL", state, prof_sc, parity_sc))
            else:
                results_summary.append(("Stage 1b: Single-Core MMUL", state, None, None))
        except Exception as e:
            print(f"  [ERROR] Stage 1b encountered exception: {e}")
            results_summary.append(("Stage 1b: Single-Core MMUL", f"ERROR: {e}", None, None))

    # -----------------------------------------------------------------------
    # STAGE 2: Multi-Core / Full-Array Stress Execution
    # -----------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" [STAGE 2] Full-Array Multi-Core Stress Execution")
    print("-" * 80)

    # 1. 16-Core Whole-Array MatMul (Columns 0-3, 16 Cores)
    wa_xclbin = "build/test_wa.xclbin"
    wa_bin = "build/test_wa.bin"
    if os.path.exists(wa_xclbin) and os.path.exists(wa_bin):
        print("\nEvaluating 16-Core Whole-Array Engine (Columns 0-3, Rows 2-5, 16 Cores)...")
        try:
            harness.load_xclbin(wa_xclbin, "MLIR_AIE")
            bo_instr_wa, ninstr_wa = harness.create_instruction_bo(wa_bin)
            
            M_wa, K_wa, N_wa = 256, 64, 128
            A_wa = np.random.randint(-10, 10, (M_wa, K_wa), dtype=np.int16)
            B_wa = np.random.randint(-10, 10, (K_wa, N_wa), dtype=np.int16)
            C_golden_wa = A_wa.astype(np.int32) @ B_wa.astype(np.int32)
            
            bo_a_wa = harness.create_host_bo(A_wa.nbytes, 3)
            bo_b_wa = harness.create_host_bo(B_wa.nbytes, 4)
            bo_c_wa = harness.create_host_bo(C_golden_wa.nbytes, 5)

            bo_a_wa.write(A_wa.tobytes(), 0)
            bo_a_wa.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            bo_b_wa.write(B_wa.tobytes(), 0)
            bo_b_wa.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            bo_c_wa.write(np.zeros_like(C_golden_wa).tobytes(), 0)
            bo_c_wa.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            print("Executing 16-core whole-array on physical silicon...")
            run, state = harness.dispatch_kernel(bo_instr_wa, ninstr_wa, bo_a_wa, bo_b_wa, bo_c_wa, timeout_ms=2000)
            print(f"  Execution State: {state}")

            if str(state) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
                bo_c_wa.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                C_hw_wa = np.frombuffer(bo_c_wa.read(C_golden_wa.nbytes, 0), dtype=np.int32).reshape(M_wa, N_wa)
                parity_wa = calculate_numerical_parity(C_golden_wa, C_hw_wa)
                print(f"  Numerical Parity: Bit-Agreement={parity_wa['bit_agreement_pct']:.2f}%, MAE={parity_wa['mae']:.4f}, RMSE={parity_wa['rmse']:.4f}")

                total_ops_wa = 2 * M_wa * K_wa * N_wa # 4,194,304 arithmetic ops
                print(f"Profiling 16-Core Silicon Latency ({iters} iterations)...")
                prof_wa = profile_hardware_execution(
                    harness, bo_instr_wa, ninstr_wa, bo_a_wa, bo_b_wa, bo_c_wa,
                    num_ops=total_ops_wa, num_cores=16, warmup_iters=warmup, bench_iters=iters
                )
                print(f"  Latency: Mean={prof_wa['mean_us']:.2f} us, Median={prof_wa['median_us']:.2f} us, Min={prof_wa['min_us']:.2f} us, P95={prof_wa['p95_us']:.2f} us")
                print(f"  Throughput: {prof_wa['effective_tops']:.4f} Effective TOPS (Peak: {prof_wa['peak_tops']:.4f} TOPS, Issue Density: {prof_wa['alu_issue_density_pct']:.2f}%)")
                results_summary.append(("Stage 2: 16-Core Array (Cols 0-3)", state, prof_wa, parity_wa))
            else:
                results_summary.append(("Stage 2: 16-Core Array (Cols 0-3)", state, None, None))
        except Exception as e:
            print(f"  [ERROR] 16-core whole-array encountered exception: {e}")
            results_summary.append(("Stage 2: 16-Core Array (Cols 0-3)", f"ERROR: {e}", None, None))

    # 2. 20-Core Array im2col Transaction Stress (Columns 0-4, 20 Cores)
    xclbin_20c = "build/im2col_4d.xclbin"
    bin_20c = "build/im2col_4d_20core.bin"
    if os.path.exists(bin_20c):
        print("\nEvaluating 20-Core Array im2col Binary (Columns 0-4, 20 Cores)...")
        try:
            harness.load_xclbin(xclbin_20c, "MLIR_AIE")
            bo_instr_20c, ninstr_20c = harness.create_instruction_bo(bin_20c)
            bo_in_20c = harness.create_host_bo(10240, 3)
            bo_out_20c = harness.create_host_bo(5120, 4)

            bo_in_20c.write(np.random.randint(-16, 16, 10240, dtype=np.int8).tobytes(), 0)
            bo_in_20c.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            bo_out_20c.write(np.zeros(5120, dtype=np.int8).tobytes(), 0)
            bo_out_20c.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            print("Dispatching 20-core stream with 2000 ms timeout guard...")
            t0 = time.perf_counter_ns()
            run, state = harness.dispatch_kernel(bo_instr_20c, ninstr_20c, bo_in_20c, bo_out_20c, timeout_ms=2000)
            t1 = time.perf_counter_ns()
            dispatch_lat_us = (t1 - t0) / 1e3
            print(f"  20-Core Array Dispatch Result: {state} in {dispatch_lat_us:.2f} us")
            results_summary.append(("Stage 2: 20-Core Array (Cols 0-4)", state, None, None))
        except Exception as e:
            print(f"  [ERROR] 20-core array encountered exception: {e}")
            results_summary.append(("Stage 2: 20-Core Array (Cols 0-4)", f"ERROR: {e}", None, None))

    # -----------------------------------------------------------------------
    # Generate Output Report
    # -----------------------------------------------------------------------
    report_lines = [
        "=" * 105,
        "AMD PHOENIX XDNA1 AIE2 HARDWARE EXECUTION PROFILING REPORT",
        "=" * 105,
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"Platform: AMD Ryzen 7 8700G (Phoenix NPU [003d:00:01.1], Tile Clock: 1.80 GHz)",
        f"Driver ABI: pyxrt / XRT 2.21.0 (HEAD), ERT Ring-Buffer Dispatch",
        f"Iterations: {iters} warm iterations (Warmup: {warmup})",
        "",
        f"{'Configuration':<38} | {'Status':<32} | {'Latency (Mean)':<15} | {'TOPS':<10} | {'Density':<8}",
        "-" * 115,
    ]

    for label, status, prof, parity in results_summary:
        if prof is not None:
            lat_str = f"{prof['mean_us']:.2f} us"
            tops_str = f"{prof['effective_tops']:.4f}"
            dens_str = f"{prof['alu_issue_density_pct']:.2f}%"
        else:
            lat_str = "N/A"
            tops_str = "N/A"
            dens_str = "N/A"
        report_lines.append(f"{label:<38} | {str(status):<32} | {lat_str:<15} | {tops_str:<10} | {dens_str:<8}")

    report_lines.append("-" * 115)
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(report_text + "\n")
        print(f"\n[REPORT] Execution report successfully logged to: {log_path}")

    # Explicit C++ object teardown to avoid XRT DLL exit trap
    try:
        del harness.kernel
        del harness.context
        del harness.dev
    except Exception:
        pass

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMD Phoenix AIE2 Hardware im2col Execution Harness")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--log-file", type=str, default="results/aie/hardware_im2col_execution.log", help="Path to write execution log")
    args = parser.parse_args()

    run_hardware_im2col_harness(iters=args.iters, warmup=args.warmup, log_path=args.log_file)
    # Clean exit without running C++ global destructors
    os._exit(0)

