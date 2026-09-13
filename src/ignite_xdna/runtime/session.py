#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/runtime/session.py
High-level InferenceSession runtime API for AMD Phoenix XDNA1 NPU.
Provides an ergonomic, production-grade interface for model loading,
double-buffered ring scheduling, and physical AIE2 silicon execution.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np

from .driver import XrtSiliconHarness, setup_xrt_environment, get_repo_root
from .parity import unblock_aie2_egress
from .ring_scheduler import BufferSet, profile_pipelined_hardware_execution


class RunHandle:
    """
    Asynchronous execution handle representing an in-flight ERT ring command.
    Enables zero-copy pipelining where input marshalling, DMA transfer,
    and output unblocking overlap with physical AIE2 vector execution.
    """
    def __init__(
        self,
        run: Any,
        bo_out: Any,
        out_bytes: int,
        num_cores: int,
        unswizzle: bool,
        pyxrt_mod: Any,
    ):
        self._run = run
        self._bo_out = bo_out
        self._out_bytes = out_bytes
        self._num_cores = num_cores
        self._unswizzle = unswizzle
        self._pyxrt = pyxrt_mod
        self._completed = False
        self._result: Optional[np.ndarray] = None

    def wait(self, timeout_ms: int = 2000) -> np.ndarray:
        """Wait for hardware kernel execution to complete and return output tensor."""
        if self._completed and self._result is not None:
            return self._result

        state = self._run.wait(timeout_ms)
        if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            raise RuntimeError(f"Asynchronous hardware execution failed with state: {state}")

        self._bo_out.sync(self._pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw_bytes = np.frombuffer(self._bo_out.read(self._out_bytes, 0), dtype=np.int8).copy()

        if self._unswizzle:
            self._result = unblock_aie2_egress(raw_bytes, num_cores=self._num_cores)
        else:
            self._result = raw_bytes

        self._completed = True
        self._run = None
        return self._result

    def is_done(self) -> bool:
        """Check if hardware execution has completed without blocking."""
        if self._completed or self._run is None:
            return True
        try:
            return str(self._run.state()) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED"
        except Exception:
            return False

    def result(self, timeout_ms: int = 2000) -> np.ndarray:
        """Alias for wait() adhering to concurrent.futures.Future interface."""
        return self.wait(timeout_ms=timeout_ms)


class InferenceSession:
    """
    High-level bare-metal inference session for AMD Phoenix XDNA1 NPU.

    Coordinates:
      1. Hardware acquisition & firmware registration (Phoenix NPU [003d:00:01.1]).
      2. One-time parameter programming (latching stationary weights & MemTile locks).
      3. Double-buffered ERT ring-buffer memory pool allocation.
      4. Tensor ingress & egress layout marshalling (unswizzling AIE2 register layout).
      5. Synchronous, asynchronous, and benchmark execution interfaces.
      6. Clean context-managed hardware resource release.
    """

    def __init__(
        self,
        model_path_or_bundle: Optional[Union[str, Path, Dict[str, str], Tuple[str, str]]] = None,
        device_index: int = 0,
        ring_depth: int = 2,
        enable_fusion: bool = False,
        xclbin_path: Optional[Union[str, Path]] = None,
        num_cores: int = 16,
        in_bytes: Optional[int] = None,
        out_bytes: Optional[int] = None,
        scale_x: Optional[float] = None,
        node_name: Optional[str] = None,
    ):
        self.device_index = device_index
        self.ring_depth = max(1, ring_depth)
        self.enable_fusion = enable_fusion
        self.num_cores = num_cores
        self.scale_x = scale_x
        self.node_name = node_name
        self._closed = False
        self._repo_root = get_repo_root()
        self.partitioned_graph: Optional[Any] = None

        # 1. Resolve buffer dimensions
        if in_bytes is not None:
            self.in_bytes = in_bytes
        else:
            self.in_bytes = 8192 if self.num_cores == 16 else (self.num_cores * 512 if self.num_cores == 4 else 2048)

        if out_bytes is not None:
            self.out_bytes = out_bytes
        else:
            self.out_bytes = 4096 if self.num_cores == 16 else (self.num_cores * 256)

        # 2. Resolve XCLBIN and transaction binaries
        self.init_txn_path, self.exec_txn_path = self._resolve_transaction_binaries(model_path_or_bundle)
        self.xclbin_path = self._resolve_xclbin(xclbin_path)

        # 3. Hardware Initialization
        setup_xrt_environment()
        self.harness = XrtSiliconHarness(device_idx=self.device_index)
        self.harness.load_xclbin(str(self.xclbin_path), "MLIR_AIE")

        # 4. Pre-allocate execution instruction buffer
        self.bo_instr_exec, self.ninstr_exec = self.harness.create_instruction_bo(str(self.exec_txn_path))

        # 5. Allocate Double-Buffered Host Memory Pool
        self.buffers: List[Dict[str, Any]] = []
        for _ in range(self.ring_depth):
            bo_in = self.harness.create_host_bo(self.in_bytes, 3)
            bo_out = self.harness.create_host_bo(self.out_bytes, 4)
            self.buffers.append({
                "bo_in": bo_in,
                "bo_out": bo_out,
                "in_flight_run": None,
            })

        self.bo_in_ping = self.buffers[0]["bo_in"]
        self.bo_in_pong = self.buffers[1]["bo_in"] if self.ring_depth > 1 else self.buffers[0]["bo_in"]
        self.bo_out_ping = self.buffers[0]["bo_out"]
        self.bo_out_pong = self.buffers[1]["bo_out"] if self.ring_depth > 1 else self.buffers[0]["bo_out"]
        self._current_slot = 0

        # 6. One-Time Parameter Programming (binds to session buffers)
        if self.init_txn_path is not None and os.path.exists(self.init_txn_path):
            self._program_stationary_parameters()

    def _resolve_xclbin(self, explicit_path: Optional[Union[str, Path]]) -> Path:
        """Resolves target AIE2 firmware XCLBIN."""
        if explicit_path is not None:
            p = Path(explicit_path)
            if not p.is_absolute():
                p = self._repo_root / p
            if p.exists():
                return p
            raise FileNotFoundError(f"Specified XCLBIN not found: {p}")

        if self.enable_fusion:
            cand = self._repo_root / "build" / "im2col_fused_2layer.xclbin"
            if cand.exists():
                return cand

        cand_16 = self._repo_root / "build" / "im2col_4d_16core.xclbin"
        if cand_16.exists():
            return cand_16

        cand_4 = self._repo_root / "build" / "im2col_4d.xclbin"
        if cand_4.exists():
            return cand_4

        raise FileNotFoundError(f"Could not locate compiled XCLBIN in {self._repo_root / 'build'}")

    def _resolve_transaction_binaries(
        self,
        model_or_bundle: Optional[Union[str, Path, Dict[str, str], Tuple[str, str]]]
    ) -> Tuple[Optional[str], str]:
        """Resolves or compiles one-time init and per-frame exec transaction binaries."""
        if isinstance(model_or_bundle, dict):
            init_p = model_or_bundle.get("init")
            exec_p = model_or_bundle.get("exec")
            if exec_p is None:
                raise ValueError("Transaction bundle dictionary must contain an 'exec' key")
            return (str(self._abs_path(init_p)) if init_p else None, str(self._abs_path(exec_p)))

        # Check if PartitionedGraph object was passed directly
        if hasattr(model_or_bundle, "npu_partitions") and hasattr(model_or_bundle, "partitions"):
            return self._compile_partitioned_graph(model_or_bundle)

        if isinstance(model_or_bundle, (tuple, list)):
            if len(model_or_bundle) >= 2:
                return (str(self._abs_path(model_or_bundle[0])), str(self._abs_path(model_or_bundle[1])))
            elif len(model_or_bundle) == 1:
                return (None, str(self._abs_path(model_or_bundle[0])))

        if isinstance(model_or_bundle, (str, Path)):
            path_str = str(model_or_bundle)
            p = self._abs_path(path_str)

            if path_str.endswith(".onnx"):
                return self._compile_onnx_subgraphs(p)

            if path_str.endswith(".bin"):
                exec_p = str(p)
                init_p = None
                if "exec" in p.stem:
                    stem_init = p.stem.replace("exec_minimal", "init").replace("exec", "init")
                    cand = p.with_name(stem_init + p.suffix)
                    if cand.exists():
                        init_p = str(cand)
                return (init_p, exec_p)

        # Default resolution if None passed
        if self.enable_fusion:
            fused_init = self._repo_root / "build" / "layer_fused_init.bin"
            fused_exec = self._repo_root / "build" / "layer_fused_exec.bin"
            if fused_exec.exists():
                return (str(fused_init) if fused_init.exists() else None, str(fused_exec))
        else:
            conv0_init = self._repo_root / "build" / "layer_conv0_init.bin"
            conv0_exec = self._repo_root / "build" / "layer_conv0_exec.bin"
            if conv0_exec.exists():
                return (str(conv0_init) if conv0_init.exists() else None, str(conv0_exec))

        raise FileNotFoundError(
            f"Could not resolve transaction binaries for input: {model_or_bundle}. "
            "Pass an ONNX model path, transaction bundle dict, or .bin path."
        )

    def _abs_path(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self._repo_root / p)

    def _compile_onnx_subgraphs(self, onnx_path: Path) -> Tuple[Optional[str], str]:
        """Lowers ONNX QDQ subgraphs to CDO transaction binaries via compiler."""
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

        from ignite_xdna.compiler.partitioner import GraphPartitioner
        partitioner = GraphPartitioner(onnx_path)
        pg = partitioner.partition()
        return self._compile_partitioned_graph(pg)

    def _compile_partitioned_graph(self, pg: Any) -> Tuple[Optional[str], str]:
        """Compiles an extracted PartitionedGraph with generalized N-layer scheduler."""
        from ignite_xdna.compiler.scheduler import MemTileMultiPassScheduler, emit_multi_layer_transaction_bundle

        self.partitioned_graph = pg
        npu_parts = pg.npu_partitions
        if not npu_parts:
            raise ValueError(f"No fusible NPU subgraphs found in partitioned graph: {pg.model_name}")

        npu_part = npu_parts[0]
        n_layers = npu_part.num_layers

        base_txn = str(self._repo_root / "build" / "layer_conv0_exec.bin")
        if not os.path.exists(base_txn):
            base_txn = str(self._repo_root / "build" / "im2col_4d_16core.bin")

        out_init = self._repo_root / "build" / f"subgraph_{n_layers}layer_init.bin"
        out_exec = self._repo_root / "build" / f"subgraph_{n_layers}layer_exec.bin"

        scheduler = MemTileMultiPassScheduler(num_cores=self.num_cores)
        plan = scheduler.schedule(npu_part)
        init_p, exec_p = emit_multi_layer_transaction_bundle(
            schedule=plan,
            base_txn_path=base_txn,
            out_init_path=str(out_init),
            out_exec_path=str(out_exec),
        )
        return (str(init_p), str(exec_p))


    def _program_stationary_parameters(self):
        """Dispatches one-time parameter initialization and primes hardware pipeline."""
        bo_init, ninstr_init = self.harness.create_instruction_bo(self.init_txn_path)
        bo_in = self.buffers[0]["bo_in"]
        bo_out = self.buffers[0]["bo_out"]

        bo_in.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_out.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Dispatch init stream
        run_init, state_init = self.harness.dispatch_kernel(
            bo_init, ninstr_init, bo_in, bo_out, timeout_ms=3000
        )
        if str(state_init) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            raise RuntimeError(f"Stationary parameter initialization failed with state: {state_init}")

    def _marshal_ingress(self, input_tensor: Any) -> bytes:
        """
        Validates, quantizes (if float), and packs input data into contiguous bytes
        matching hardware ingress port specifications.
        """
        # Support PyTorch CPU tensors without strict torch dependency
        if hasattr(input_tensor, "detach") and hasattr(input_tensor, "cpu"):
            input_tensor = input_tensor.detach().cpu().numpy()

        arr = np.asarray(input_tensor)

        # Quantize float inputs if scale provided
        if np.issubdtype(arr.dtype, np.floating):
            if self.scale_x is not None and self.scale_x > 0:
                arr = np.clip(np.round(arr / self.scale_x), -128, 127).astype(np.int8)
            else:
                arr = np.clip(np.round(arr), -128, 127).astype(np.int8)
        elif arr.dtype != np.int8:
            arr = arr.astype(np.int8)

        flat = arr.flatten()
        n = len(flat)

        if n == self.in_bytes:
            return flat.tobytes()
        elif n == self.in_bytes // 4:
            # Single-column feature map broadcast/tiled across all 4 columns
            return np.tile(flat, 4).tobytes()
        elif n < self.in_bytes:
            padded = np.zeros(self.in_bytes, dtype=np.int8)
            padded[:n] = flat
            return padded.tobytes()
        else:
            return flat[:self.in_bytes].tobytes()

    def _marshal_egress(self, bo_out: Any, unswizzle: bool = True) -> np.ndarray:
        """Synchronizes device buffer and unswizzles AIE2 output layout."""
        bo_out.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw_bytes = np.frombuffer(bo_out.read(self.out_bytes, 0), dtype=np.int8).copy()

        if unswizzle:
            return unblock_aie2_egress(raw_bytes, num_cores=self.num_cores)
        return raw_bytes

    def run(self, input_tensor: Any, unswizzle: bool = True, timeout_ms: int = 2000) -> np.ndarray:
        """
        Synchronously executes inference on physical AMD Phoenix NPU silicon,
        orchestrating fallback CPU execution for pre/post-subgraph partitions if present.

        Args:
            input_tensor: NumPy array or PyTorch CPU tensor containing input activations.
            unswizzle: When True, unswizzles vector register format into contiguous [pixels, channels].
            timeout_ms: Maximum wait duration before raising hardware timeout.

        Returns:
            NumPy array of INT8 output activations.
        """
        if self._closed:
            raise RuntimeError("Cannot invoke run() on a closed InferenceSession")

        if self.partitioned_graph is not None and self.partitioned_graph.cpu_partitions:
            import onnxruntime as ort
            from ignite_xdna.compiler.partitioner import CpuFallbackPartition, NpuFusedPartition

            cur_data = input_tensor
            for part in self.partitioned_graph.partitions:
                if isinstance(part, CpuFallbackPartition):
                    if part.onnx_model is not None:
                        sess = ort.InferenceSession(part.onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
                        inp_node = sess.get_inputs()[0]
                        inp_name = inp_node.name
                        if hasattr(cur_data, "detach"):
                            cur_data = cur_data.detach().cpu().numpy()
                        if "float" in inp_node.type and cur_data.dtype in (np.int8, np.uint8):
                            cur_data = cur_data.astype(np.float32)
                        elif "int8" in inp_node.type and cur_data.dtype != np.int8:
                            cur_data = cur_data.astype(np.int8)
                        cur_data = sess.run(None, {inp_name: cur_data})[0]
                elif isinstance(part, NpuFusedPartition):
                    cur_data = self._execute_npu_direct(cur_data, unswizzle=unswizzle, timeout_ms=timeout_ms)
            return cur_data

        return self._execute_npu_direct(input_tensor, unswizzle=unswizzle, timeout_ms=timeout_ms)

    def _execute_npu_direct(self, input_tensor: Any, unswizzle: bool = True, timeout_ms: int = 2000) -> np.ndarray:
        """Direct NPU hardware execution without CPU fallback routing."""
        slot = self.buffers[self._current_slot]
        if slot["in_flight_run"] is not None:
            slot["in_flight_run"].wait(timeout_ms)
            slot["in_flight_run"] = None

        data_bytes = self._marshal_ingress(input_tensor)
        slot["bo_in"].write(data_bytes, 0)
        slot["bo_in"].sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # In physical AIE2 double-buffered streaming pipeline, 2 dispatches push
        # input into MemTile ping-pong stage and drain bit-exact egress to host.
        self.harness.dispatch_kernel(self.bo_instr_exec, self.ninstr_exec, slot["bo_in"], slot["bo_out"], timeout_ms=timeout_ms)
        run = self.harness.kernel(3, self.bo_instr_exec, self.ninstr_exec, slot["bo_in"], slot["bo_out"])
        state = run.wait(timeout_ms)
        if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            raise RuntimeError(f"Hardware execution failed with state: {state}")

        out = self._marshal_egress(slot["bo_out"], unswizzle=unswizzle)
        self._current_slot = (self._current_slot + 1) % self.ring_depth
        return out

    def run_async(self, input_tensor: Any, unswizzle: bool = True) -> RunHandle:
        """
        Asynchronously submits an inference request to the ERT ring buffer.

        Args:
            input_tensor: NumPy array or PyTorch CPU tensor containing input activations.
            unswizzle: When True, unswizzles output layout upon completion.

        Returns:
            RunHandle for synchronization and result retrieval.
        """
        if self._closed:
            raise RuntimeError("Cannot invoke run_async() on a closed InferenceSession")

        slot = self.buffers[self._current_slot]
        if slot["in_flight_run"] is not None:
            slot["in_flight_run"].wait()
            slot["in_flight_run"] = None

        data_bytes = self._marshal_ingress(input_tensor)
        slot["bo_in"].write(data_bytes, 0)
        slot["bo_in"].sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # In physical AIE2 double-buffered streaming pipeline, 2 dispatches push
        # input into MemTile ping-pong stage and drain bit-exact egress to host.
        self.harness.dispatch_kernel(self.bo_instr_exec, self.ninstr_exec, slot["bo_in"], slot["bo_out"])
        run = self.harness.kernel(3, self.bo_instr_exec, self.ninstr_exec, slot["bo_in"], slot["bo_out"])
        handle = RunHandle(
            run=run,
            bo_out=slot["bo_out"],
            out_bytes=self.out_bytes,
            num_cores=self.num_cores,
            unswizzle=unswizzle,
            pyxrt_mod=self.harness.pyxrt
        )
        slot["in_flight_run"] = handle
        self._current_slot = (self._current_slot + 1) % self.ring_depth
        return handle

    def benchmark(
        self,
        input_tensor: Optional[Any] = None,
        warmup: int = 50,
        iterations: int = 500
    ) -> Dict[str, Any]:
        """
        Profiles sustained double-buffered pipelined execution over iterations.

        Returns:
            Dictionary containing mean, median, min, p95 latency (μs) and sustained FPS.
        """
        if self._closed:
            raise RuntimeError("Cannot benchmark on a closed InferenceSession")

        if input_tensor is None:
            input_tensor = np.zeros(self.in_bytes, dtype=np.int8)

        data_bytes = self._marshal_ingress(input_tensor)

        ping_set = BufferSet(
            [(self.bo_in_ping, data_bytes)],
            [self.bo_out_ping],
            [self.bo_in_ping, self.bo_out_ping]
        )
        pong_set = BufferSet(
            [(self.bo_in_pong, data_bytes)],
            [self.bo_out_pong],
            [self.bo_in_pong, self.bo_out_pong]
        )

        num_ops = self.num_cores * 2 * 9 * 32 * 8 * 2
        if self.enable_fusion:
            num_ops *= 2

        prof = profile_pipelined_hardware_execution(
            harness=self.harness,
            bo_instr=self.bo_instr_exec,
            ninstr=self.ninstr_exec,
            ping_set=ping_set,
            pong_set=pong_set,
            num_ops=num_ops,
            num_cores=self.num_cores,
            warmup_iters=warmup,
            bench_iters=iterations,
        )

        return {
            "iterations": iterations,
            "mean_us": prof["effective_per_iter_us"],
            "pipelined_mean_us": prof["effective_per_iter_us"],
            "median_us": prof["median_step_us"],
            "min_us": prof["min_step_us"],
            "max_us": prof["max_step_us"],
            "p95_us": prof["p95_step_us"],
            "p99_us": prof["p99_step_us"],
            "fps": prof["fps"],
            "effective_tops": prof["effective_tops"],
            "peak_tops": prof["peak_tops"],
            "alu_issue_density_pct": prof["alu_issue_density_pct"],
            "wall_total_us": prof["wall_total_us"],
        }

    def close(self):
        """Cleanly releases PyXRT buffer handles, sync locks, and hardware contexts."""
        if self._closed:
            return

        for slot in self.buffers:
            if slot.get("in_flight_run") is not None:
                try:
                    slot["in_flight_run"].wait(500)
                except Exception:
                    pass
                slot["in_flight_run"]._run = None
            slot["in_flight_run"] = None
            slot["bo_in"] = None
            slot["bo_out"] = None

        self.buffers.clear()
        self.bo_in_ping = None
        self.bo_in_pong = None
        self.bo_out_ping = None
        self.bo_out_pong = None
        self.bo_instr_exec = None

        if hasattr(self, "harness") and self.harness is not None:
            try:
                self.harness.kernel = None
                self.harness.context = None
                self.harness.dev = None
            except Exception:
                pass

        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
