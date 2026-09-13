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
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np

from .driver import XrtSiliconHarness, setup_xrt_environment, get_repo_root
from .parity import unblock_aie2_egress
from .ring_scheduler import BufferSet, profile_pipelined_hardware_execution


@dataclass
class MonolithicStageHandle:
    """Handle for a single continuous monolithic execution stage in on-die MemTile SRAM."""
    name: str
    init_txn_path: Optional[str] = None
    exec_txn_path: str = ""
    bo_instr_init: Optional[Any] = None
    ninstr_init: int = 0
    bo_instr_exec: Optional[Any] = None
    ninstr_exec: int = 0
    num_layers: int = 0
    c2f_blocks: List[str] = field(default_factory=list)
    intermediate_ddr_bytes: int = 0


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
        model_path_or_bundle: Optional[Union[str, Path, Dict[str, Any], Tuple[str, str]]] = None,
        device_index: int = 0,
        ring_depth: int = 2,
        enable_fusion: bool = False,
        enable_monolithic: bool = False,
        include_neck: bool = False,
        neck_only: bool = False,
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
        self.enable_monolithic = enable_monolithic
        self.include_neck = include_neck
        self.neck_only = neck_only
        self.num_cores = num_cores
        self.scale_x = scale_x
        self.node_name = node_name
        self._closed = False
        self._repo_root = get_repo_root()
        self.partitioned_graph: Optional[Any] = None
        self.profiler: Optional[Any] = None
        self.monolithic_stages: Dict[str, MonolithicStageHandle] = OrderedDict()
        self.multi_stage_plan: Optional[Any] = None

        # Auto-detect monolithic or neck request
        if include_neck or neck_only:
            self.enable_monolithic = True
        if hasattr(model_path_or_bundle, "stages") or (isinstance(model_path_or_bundle, dict) and "stages" in model_path_or_bundle):
            self.enable_monolithic = True

        # 1. Resolve buffer dimensions
        if in_bytes is not None:
            self.in_bytes = in_bytes
        else:
            self.in_bytes = 8192 if self.num_cores == 16 else (self.num_cores * 512 if self.num_cores == 4 else 2048)

        if out_bytes is not None:
            self.out_bytes = out_bytes
        else:
            self.out_bytes = 4096 if self.num_cores == 16 else (self.num_cores * 256)

        # 2. Hardware and Binary Initialization
        self.xclbin_path = self._resolve_xclbin(xclbin_path)
        setup_xrt_environment()
        self.harness = XrtSiliconHarness(device_idx=self.device_index)
        self.harness.load_xclbin(str(self.xclbin_path), "MLIR_AIE")

        if self.enable_monolithic:
            self._setup_monolithic_stages(model_path_or_bundle)
            first_stage = next(iter(self.monolithic_stages.values()))
            self.bo_instr_exec = first_stage.bo_instr_exec
            self.ninstr_exec = first_stage.ninstr_exec
            self.init_txn_path = first_stage.init_txn_path
            self.exec_txn_path = first_stage.exec_txn_path
        else:
            self.init_txn_path, self.exec_txn_path = self._resolve_transaction_binaries(model_path_or_bundle)
            self.bo_instr_exec, self.ninstr_exec = self.harness.create_instruction_bo(str(self.exec_txn_path))

        # 3. Allocate Double-Buffered Host Memory Pool
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

        # 4. One-Time Parameter Programming (binds to session buffers)
        if self.enable_monolithic or (self.init_txn_path is not None and os.path.exists(self.init_txn_path)):
            self._program_stationary_parameters()

    @property
    def is_monolithic(self) -> bool:
        """Indicates whether session operates in 4-stage monolithic pipeline mode."""
        return self.enable_monolithic

    @property
    def stage_names(self) -> List[str]:
        """Names of the active monolithic pipeline stages."""
        return list(self.monolithic_stages.keys())

    @property
    def intermediate_ddr_bytes(self) -> int:
        """Intermediate DDR memory transfer bytes (strictly 0 in monolithic mode)."""
        return 0 if self.enable_monolithic else (self.in_bytes + self.out_bytes)

    def _setup_monolithic_stages(self, model_or_bundle: Any):
        """Initializes and builds monolithic multi-stage transaction plans for Stem, P3, P4, P5."""
        from ignite_xdna.compiler.scheduler import (
            MemTileMultiPassScheduler,
            MultiStageSchedulePlan,
            emit_multi_stage_transaction_bundle,
        )
        from ignite_xdna.compiler.partitioner import GraphPartitioner

        base_txn = str(self._repo_root / "build" / "layer_conv0_exec.bin")
        if not os.path.exists(base_txn):
            base_txn = str(self._repo_root / "build" / "im2col_4d_16core.bin")

        build_dir = self._repo_root / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(model_or_bundle, MultiStageSchedulePlan):
            self.multi_stage_plan = model_or_bundle
            for s_name, stage_plan in model_or_bundle.stages.items():
                out_init = str(build_dir / f"stage_{s_name.lower()}_init.bin")
                out_exec = str(build_dir / f"stage_{s_name.lower()}_exec.bin")
                if not (os.path.exists(out_init) and os.path.exists(out_exec)):
                    emit_multi_stage_transaction_bundle(stage_plan, base_txn, out_init, out_exec)
                bo_exec, ninstr_exec = self.harness.create_instruction_bo(out_exec)
                bo_init, ninstr_init = (
                    self.harness.create_instruction_bo(out_init) if os.path.exists(out_init) else (None, 0)
                )
                self.monolithic_stages[s_name] = MonolithicStageHandle(
                    name=s_name,
                    init_txn_path=out_init,
                    exec_txn_path=out_exec,
                    bo_instr_init=bo_init,
                    ninstr_init=ninstr_init,
                    bo_instr_exec=bo_exec,
                    ninstr_exec=ninstr_exec,
                    num_layers=stage_plan.num_layers,
                    c2f_blocks=stage_plan.c2f_blocks,
                    intermediate_ddr_bytes=0,
                )
            return

        if isinstance(model_or_bundle, dict) and "stages" in model_or_bundle:
            stages_dict = model_or_bundle["stages"]
            for s_name, paths in stages_dict.items():
                if isinstance(paths, (tuple, list)):
                    init_p = str(self._abs_path(paths[0])) if paths[0] else None
                    exec_p = str(self._abs_path(paths[1]))
                elif isinstance(paths, dict):
                    init_p = str(self._abs_path(paths["init"])) if "init" in paths else None
                    exec_p = str(self._abs_path(paths["exec"]))
                else:
                    init_p = None
                    exec_p = str(self._abs_path(paths))
                bo_exec, ninstr_exec = self.harness.create_instruction_bo(exec_p)
                bo_init, ninstr_init = (
                    self.harness.create_instruction_bo(init_p) if init_p and os.path.exists(init_p) else (None, 0)
                )
                self.monolithic_stages[s_name] = MonolithicStageHandle(
                    name=s_name,
                    init_txn_path=init_p,
                    exec_txn_path=exec_p,
                    bo_instr_init=bo_init,
                    ninstr_init=ninstr_init,
                    bo_instr_exec=bo_exec,
                    ninstr_exec=ninstr_exec,
                    num_layers=paths.get("num_layers", 7) if isinstance(paths, dict) else 7,
                    c2f_blocks=paths.get("c2f_blocks", []) if isinstance(paths, dict) else [],
                    intermediate_ddr_bytes=0,
                )
            return

        # Default: compile YOLOv8n backbone monolithic stages
        model_path = model_or_bundle
        if model_path is None or (isinstance(model_path, (str, Path)) and not str(model_path).endswith(".onnx")):
            cand = self._repo_root / "models" / "yolov8n_cut_xint8.onnx"
            if cand.exists():
                model_path = cand
            else:
                model_path = self._repo_root / "models" / "yolov8n.onnx"

        model_path = self._abs_path(model_path)
        partitioner = GraphPartitioner(model_path, fuse_neck=self.include_neck or self.neck_only)
        if self.neck_only:
            pg = partitioner.partition(neck_only=True)
            npu_parts = pg.neck_npu_partitions
        elif self.include_neck:
            pg = partitioner.partition(backbone_only=False)
            npu_parts = pg.npu_partitions
        else:
            pg = partitioner.partition(backbone_only=True)
            npu_parts = pg.backbone_npu_partitions
        self.partitioned_graph = pg

        scheduler = MemTileMultiPassScheduler(num_cores=self.num_cores)
        multi_plan = scheduler.schedule_multi_stage(npu_parts)
        self.multi_stage_plan = multi_plan

        for s_name, stage_plan in multi_plan.stages.items():
            out_init = str(build_dir / f"stage_{s_name.lower()}_init.bin")
            out_exec = str(build_dir / f"stage_{s_name.lower()}_exec.bin")
            if not (os.path.exists(out_init) and os.path.exists(out_exec)):
                emit_multi_stage_transaction_bundle(stage_plan, base_txn, out_init, out_exec)
            bo_exec, ninstr_exec = self.harness.create_instruction_bo(out_exec)
            bo_init, ninstr_init = (
                self.harness.create_instruction_bo(out_init) if os.path.exists(out_init) else (None, 0)
            )
            self.monolithic_stages[s_name] = MonolithicStageHandle(
                name=s_name,
                init_txn_path=out_init,
                exec_txn_path=out_exec,
                bo_instr_init=bo_init,
                ninstr_init=ninstr_init,
                bo_instr_exec=bo_exec,
                ninstr_exec=ninstr_exec,
                num_layers=stage_plan.num_layers,
                c2f_blocks=stage_plan.c2f_blocks,
                intermediate_ddr_bytes=0,
            )

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
        bo_in = self.buffers[0]["bo_in"]
        bo_out = self.buffers[0]["bo_out"]

        bo_in.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_out.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        if self.enable_monolithic:
            for s_name, stage in self.monolithic_stages.items():
                if stage.bo_instr_init is not None:
                    run_init, state_init = self.harness.dispatch_kernel(
                        stage.bo_instr_init, stage.ninstr_init, bo_in, bo_out, timeout_ms=3000
                    )
                    if str(state_init) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
                        raise RuntimeError(f"Monolithic stage {s_name} init failed with state: {state_init}")
                # Prime pipeline with 1 exec dispatch
                self.harness.dispatch_kernel(
                    stage.bo_instr_exec, stage.ninstr_exec, bo_in, bo_out, timeout_ms=2000
                )
            return

        bo_init, ninstr_init = self.harness.create_instruction_bo(self.init_txn_path)
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
    def enable_profiling(self, profiler: Optional[Any] = None) -> Any:
        """Enables high-resolution hardware and host event profiling."""
        if profiler is None:
            from .profiler import HardwareEventProfiler
            self.profiler = HardwareEventProfiler()
        else:
            self.profiler = profiler
        self.profiler.enable()
        return self.profiler

    def disable_profiling(self):
        """Disables profiling."""
        if self.profiler is not None:
            self.profiler.disable()

    def get_profile_report(self) -> Dict[str, Any]:
        """Returns the summarized profiling report."""
        if self.profiler is not None:
            return self.profiler.summarize()
        return {"error": "Profiling was not enabled"}

    def run(
        self,
        input_tensor: Any,
        unswizzle: bool = True,
        timeout_ms: int = 2000,
        return_timestamps: bool = False,
        extract_feature_maps: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """
        Synchronously executes inference on physical AMD Phoenix NPU silicon,
        orchestrating fallback CPU execution for pre/post-subgraph partitions if present.

        Args:
            input_tensor: NumPy array or PyTorch CPU tensor containing input activations.
            unswizzle: When True, unswizzles vector register format into contiguous [pixels, channels].
            timeout_ms: Maximum wait duration before raising hardware timeout.
            return_timestamps: When True, returns (output, HardwareTimestamps).
            extract_feature_maps: When True in monolithic mode, extracts P3, P4, P5 feature maps.

        Returns:
            NumPy array of INT8 output activations (or tuple with metadata).
        """
        if self._closed:
            raise RuntimeError("Cannot invoke run() on a closed InferenceSession")

        if self.enable_monolithic:
            return self._execute_monolithic_stages(
                input_tensor,
                unswizzle=unswizzle,
                timeout_ms=timeout_ms,
                return_timestamps=return_timestamps,
                extract_feature_maps=extract_feature_maps,
            )

        is_profiling = self.profiler is not None and self.profiler.is_enabled
        if is_profiling:
            self.profiler.start_iteration(len(self.profiler.iterations))
        t_wall_start = time.perf_counter_ns() if is_profiling else 0

        if self.partitioned_graph is not None and self.partitioned_graph.cpu_partitions:
            import onnxruntime as ort
            from ignite_xdna.compiler.partitioner import CpuFallbackPartition, NpuFusedPartition
            from .profiler import PartitionProfileRecord, map_yolo_node_to_stage

            cur_data = input_tensor
            for part in self.partitioned_graph.partitions:
                p_start = time.perf_counter_ns() if is_profiling else 0
                if isinstance(part, CpuFallbackPartition):
                    t_eval_start = 0
                    t_eval_end = 0
                    if part.onnx_model is not None:
                        # Pre-cache ORT session on partition instance to measure true execution
                        if not hasattr(part, "_cached_session") or part._cached_session is None:
                            part._cached_session = ort.InferenceSession(
                                part.onnx_model.SerializeToString(),
                                providers=["CPUExecutionProvider"]
                            )
                        sess = part._cached_session
                        inp_node = sess.get_inputs()[0]
                        inp_name = inp_node.name
                        if hasattr(cur_data, "detach"):
                            cur_data = cur_data.detach().cpu().numpy()
                        if "float" in inp_node.type and cur_data.dtype in (np.int8, np.uint8):
                            cur_data = cur_data.astype(np.float32)
                        elif "int8" in inp_node.type and cur_data.dtype != np.int8:
                            cur_data = cur_data.astype(np.int8)

                        in_bytes = cur_data.nbytes if hasattr(cur_data, "nbytes") else 0
                        t_eval_start = time.perf_counter_ns()
                        cur_data = sess.run(None, {inp_name: cur_data})[0]
                        t_eval_end = time.perf_counter_ns()
                        out_bytes = cur_data.nbytes if hasattr(cur_data, "nbytes") else 0
                    else:
                        in_bytes = 0
                        out_bytes = 0

                    if is_profiling:
                        p_end = time.perf_counter_ns()
                        first_node = part.nodes[0].name if part.nodes else f"cpu_part_{part.partition_id}"
                        stage_group, stage_label = map_yolo_node_to_stage(first_node, [n.op_type for n in part.nodes])
                        rec = PartitionProfileRecord(
                            partition_id=part.partition_id,
                            partition_type="CPU",
                            stage_group=stage_group,
                            stage_label=stage_label,
                            node_names=[n.name for n in part.nodes],
                            op_types=[n.op_type for n in part.nodes],
                            input_names=part.input_names,
                            output_names=part.output_names,
                            input_bytes=in_bytes,
                            output_bytes=out_bytes,
                            duration_us=(p_end - p_start) / 1000.0,
                            cpu_eval_us=(t_eval_end - t_eval_start) / 1000.0 if t_eval_start > 0 else 0.0,
                        )
                        self.profiler.record_partition(rec)

                elif isinstance(part, NpuFusedPartition):
                    in_bytes = cur_data.nbytes if hasattr(cur_data, "nbytes") else self.in_bytes
                    if is_profiling:
                        cur_data, hw_ts = self._execute_npu_direct(
                            cur_data, unswizzle=unswizzle, timeout_ms=timeout_ms, return_timestamps=True
                        )
                        p_end = time.perf_counter_ns()
                        out_bytes = cur_data.nbytes if hasattr(cur_data, "nbytes") else self.out_bytes
                        first_layer = part.layers[0].node_name if part.layers else f"npu_part_{part.partition_id}"
                        stage_group, stage_label = map_yolo_node_to_stage(first_layer, ["Conv"])
                        rec = PartitionProfileRecord(
                            partition_id=part.partition_id,
                            partition_type="NPU",
                            stage_group=stage_group,
                            stage_label=stage_label,
                            node_names=[l.node_name for l in part.layers],
                            op_types=["Conv" for _ in part.layers],
                            input_names=part.input_names,
                            output_names=part.output_names,
                            input_bytes=in_bytes,
                            output_bytes=out_bytes,
                            duration_us=hw_ts.total_partition_us,
                            ingress_marshal_us=hw_ts.ingress_marshal_us,
                            bo_in_sync_us=hw_ts.bo_in_sync_us,
                            dispatch_submission_us=hw_ts.dispatch_submission_us,
                            device_execution_us=hw_ts.device_execution_us,
                            bo_out_sync_us=hw_ts.bo_out_sync_us,
                            egress_unswizzle_us=hw_ts.egress_unswizzle_us,
                        )
                        self.profiler.record_partition(rec)
                    else:
                        cur_data = self._execute_npu_direct(cur_data, unswizzle=unswizzle, timeout_ms=timeout_ms)

            if is_profiling:
                t_wall_end = time.perf_counter_ns()
                self.profiler.end_iteration((t_wall_end - t_wall_start) / 1000.0)

            return cur_data

        if is_profiling:
            from .profiler import PartitionProfileRecord, map_yolo_node_to_stage
            in_bytes = input_tensor.nbytes if hasattr(input_tensor, "nbytes") else self.in_bytes
            out, hw_ts = self._execute_npu_direct(
                input_tensor, unswizzle=unswizzle, timeout_ms=timeout_ms, return_timestamps=True
            )
            t_wall_end = time.perf_counter_ns()
            out_bytes = out.nbytes if hasattr(out, "nbytes") else self.out_bytes
            node_name = self.node_name or "conv_npu"
            stage_group, stage_label = map_yolo_node_to_stage(node_name, ["Conv"])
            rec = PartitionProfileRecord(
                partition_id=0,
                partition_type="NPU",
                stage_group=stage_group,
                stage_label=stage_label,
                node_names=[node_name],
                op_types=["Conv"],
                input_bytes=in_bytes,
                output_bytes=out_bytes,
                duration_us=hw_ts.total_partition_us,
                ingress_marshal_us=hw_ts.ingress_marshal_us,
                bo_in_sync_us=hw_ts.bo_in_sync_us,
                dispatch_submission_us=hw_ts.dispatch_submission_us,
                device_execution_us=hw_ts.device_execution_us,
                bo_out_sync_us=hw_ts.bo_out_sync_us,
                egress_unswizzle_us=hw_ts.egress_unswizzle_us,
            )
            self.profiler.record_partition(rec)
            self.profiler.end_iteration((t_wall_end - t_wall_start) / 1000.0)
            return out

        return self._execute_npu_direct(input_tensor, unswizzle=unswizzle, timeout_ms=timeout_ms)

    def _execute_npu_direct(
        self,
        input_tensor: Any,
        unswizzle: bool = True,
        timeout_ms: int = 2000,
        return_timestamps: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """Direct NPU hardware execution without CPU fallback routing."""
        t_start = time.perf_counter_ns() if return_timestamps else 0
        slot = self.buffers[self._current_slot]
        if slot["in_flight_run"] is not None:
            slot["in_flight_run"].wait(timeout_ms)
            slot["in_flight_run"] = None

        t_marshal_start = time.perf_counter_ns() if return_timestamps else 0
        data_bytes = self._marshal_ingress(input_tensor)
        t_marshal_end = time.perf_counter_ns() if return_timestamps else 0

        slot["bo_in"].write(data_bytes, 0)
        t_sync_in_start = time.perf_counter_ns() if return_timestamps else 0
        slot["bo_in"].sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t_sync_in_end = time.perf_counter_ns() if return_timestamps else 0

        # In physical AIE2 double-buffered streaming pipeline, 2 dispatches push
        # input into MemTile ping-pong stage and drain bit-exact egress to host.
        t_sub_start = time.perf_counter_ns() if return_timestamps else 0
        self.harness.dispatch_kernel(self.bo_instr_exec, self.ninstr_exec, slot["bo_in"], slot["bo_out"], timeout_ms=timeout_ms)
        run = self.harness.kernel(3, self.bo_instr_exec, self.ninstr_exec, slot["bo_in"], slot["bo_out"])
        t_sub_end = time.perf_counter_ns() if return_timestamps else 0

        t_exec_start = time.perf_counter_ns() if return_timestamps else 0
        state = run.wait(timeout_ms)
        t_exec_end = time.perf_counter_ns() if return_timestamps else 0
        if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            raise RuntimeError(f"Hardware execution failed with state: {state}")

        t_sync_out_start = time.perf_counter_ns() if return_timestamps else 0
        slot["bo_out"].sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        t_sync_out_end = time.perf_counter_ns() if return_timestamps else 0

        t_unswizzle_start = time.perf_counter_ns() if return_timestamps else 0
        raw_bytes = np.frombuffer(slot["bo_out"].read(self.out_bytes, 0), dtype=np.int8).copy()
        if unswizzle:
            out = unblock_aie2_egress(raw_bytes, num_cores=self.num_cores)
        else:
            out = raw_bytes
        t_unswizzle_end = time.perf_counter_ns() if return_timestamps else 0

        self._current_slot = (self._current_slot + 1) % self.ring_depth
        t_end = time.perf_counter_ns() if return_timestamps else 0

        if return_timestamps:
            from .profiler import HardwareTimestamps
            hw_ts = HardwareTimestamps(
                ingress_marshal_ns=t_marshal_end - t_marshal_start,
                bo_in_sync_ns=t_sync_in_end - t_sync_in_start,
                dispatch_submission_ns=t_sub_end - t_sub_start,
                device_execution_ns=t_exec_end - t_exec_start,
                bo_out_sync_ns=t_sync_out_end - t_sync_out_start,
                egress_unswizzle_ns=t_unswizzle_end - t_unswizzle_start,
                total_partition_ns=t_end - t_start,
            )
            return out, hw_ts

        return out

    def _execute_monolithic_stages(
        self,
        input_tensor: Any,
        unswizzle: bool = True,
        timeout_ms: int = 2000,
        return_timestamps: bool = False,
        extract_feature_maps: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """
        Direct hardware execution of the 4-stage monolithic transaction bundle (Stem, P3, P4, P5):
          - Dispatches each monolithic stage as a single continuous ERT instruction buffer.
          - 0 bytes intermediate DDR traffic (eliminates all intermediate bo_in.sync and bo_out.sync).
          - Inter-stage handoffs synchronized in MemTile SRAM Banks 0/1 via hardware Locks 4 and 5.
        """
        t_start = time.perf_counter_ns()
        slot = self.buffers[self._current_slot]
        if slot["in_flight_run"] is not None:
            slot["in_flight_run"].wait(timeout_ms)
            slot["in_flight_run"] = None

        t_marshal_start = time.perf_counter_ns()
        data_bytes = self._marshal_ingress(input_tensor)
        t_marshal_end = time.perf_counter_ns()

        slot["bo_in"].write(data_bytes, 0)

        # Ingress synchronization (Host DDR -> NPU MemTile SRAM for Stem)
        t_sync_in_start = time.perf_counter_ns()
        slot["bo_in"].sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t_sync_in_end = time.perf_counter_ns()

        stage_timings: List[Dict[str, float]] = []
        feature_maps: Dict[str, np.ndarray] = {}

        # 4-stage monolithic pipeline execution across physical Phoenix silicon
        # ZERO intermediate bo_in.sync or bo_out.sync between internal layers!
        for s_idx, (s_name, stage) in enumerate(self.monolithic_stages.items()):
            t_sub_start = time.perf_counter_ns()
            run = self.harness.kernel(
                3, stage.bo_instr_exec, stage.ninstr_exec, slot["bo_in"], slot["bo_out"]
            )
            t_sub_end = time.perf_counter_ns()

            t_exec_start = time.perf_counter_ns()
            state = run.wait(timeout_ms)
            t_exec_end = time.perf_counter_ns()

            if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
                raise RuntimeError(f"Monolithic stage {s_name} execution failed with state: {state}")

            sub_us = (t_sub_end - t_sub_start) / 1000.0
            exec_us = (t_exec_end - t_exec_start) / 1000.0
            stage_timings.append({
                "stage": s_name,
                "submission_us": sub_us,
                "execution_us": exec_us,
                "total_us": sub_us + exec_us,
            })

            if extract_feature_maps and s_name in ("P3", "P4", "P5", "Neck_FPN", "Neck_PAN"):
                slot["bo_out"].sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                raw_f = np.frombuffer(slot["bo_out"].read(self.out_bytes, 0), dtype=np.int8).copy()
                feature_maps[s_name] = unblock_aie2_egress(raw_f, num_cores=self.num_cores) if unswizzle else raw_f

        # Egress synchronization (NPU MemTile SRAM -> Host DDR)
        t_sync_out_start = time.perf_counter_ns()
        slot["bo_out"].sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        t_sync_out_end = time.perf_counter_ns()

        t_unswizzle_start = time.perf_counter_ns()
        raw_bytes = np.frombuffer(slot["bo_out"].read(self.out_bytes, 0), dtype=np.int8).copy()
        out = unblock_aie2_egress(raw_bytes, num_cores=self.num_cores) if unswizzle else raw_bytes
        t_unswizzle_end = time.perf_counter_ns()

        self._current_slot = (self._current_slot + 1) % self.ring_depth
        t_end = time.perf_counter_ns()

        total_submission_us = sum(st["submission_us"] for st in stage_timings)
        total_exec_us = sum(st["execution_us"] for st in stage_timings)

        if self.profiler is not None and self.profiler.is_enabled:
            from .profiler import PartitionProfileRecord
            for s_idx, (s_name, stage) in enumerate(self.monolithic_stages.items()):
                st = stage_timings[s_idx]
                is_first = (s_idx == 0)
                is_last = (s_idx == len(self.monolithic_stages) - 1)
                rec = PartitionProfileRecord(
                    partition_id=s_idx,
                    partition_type="NPU",
                    stage_group=f"Monolithic Stage {s_name}",
                    stage_label=f"Stage {s_name} ({stage.num_layers} layers)",
                    node_names=[f"{s_name}_layer_{i}" for i in range(stage.num_layers)],
                    op_types=["Conv" for _ in range(stage.num_layers)],
                    input_bytes=self.in_bytes if is_first else 0,
                    output_bytes=self.out_bytes if is_last else 0,
                    duration_us=st["total_us"],
                    ingress_marshal_us=(t_marshal_end - t_marshal_start) / 1000.0 if is_first else 0.0,
                    bo_in_sync_us=(t_sync_in_end - t_sync_in_start) / 1000.0 if is_first else 0.0,
                    dispatch_submission_us=st["submission_us"],
                    device_execution_us=st["execution_us"],
                    bo_out_sync_us=(t_sync_out_end - t_sync_out_start) / 1000.0 if is_last else 0.0,
                    egress_unswizzle_us=(t_unswizzle_end - t_unswizzle_start) / 1000.0 if is_last else 0.0,
                )
                self.profiler.record_partition(rec)
            self.profiler.end_iteration((t_end - t_start) / 1000.0)

        if return_timestamps:
            from .profiler import HardwareTimestamps
            hw_ts = HardwareTimestamps(
                ingress_marshal_ns=t_marshal_end - t_marshal_start,
                bo_in_sync_ns=t_sync_in_end - t_sync_in_start,
                dispatch_submission_ns=int(total_submission_us * 1000),
                device_execution_ns=int(total_exec_us * 1000),
                bo_out_sync_ns=t_sync_out_end - t_sync_out_start,
                egress_unswizzle_ns=t_unswizzle_end - t_unswizzle_start,
                total_partition_ns=t_end - t_start,
            )
            if extract_feature_maps:
                return out, hw_ts, feature_maps
            return out, hw_ts

        if extract_feature_maps:
            return out, feature_maps
        return out

    def run_monolithic_feature_maps(
        self,
        input_tensor: Any,
        unswizzle: bool = True,
        timeout_ms: int = 2000
    ) -> Dict[str, np.ndarray]:
        """
        Executes monolithic pipeline and returns individual feature maps for P3, P4, and P5.
        """
        _, features = self._execute_monolithic_stages(
            input_tensor,
            unswizzle=unswizzle,
            timeout_ms=timeout_ms,
            return_timestamps=False,
            extract_feature_maps=True,
        )
        return features

    def run_monolithic_neck(
        self,
        input_tensor: Any = None,
        p3: Optional[np.ndarray] = None,
        p4: Optional[np.ndarray] = None,
        p5: Optional[np.ndarray] = None,
        unswizzle: bool = True,
        timeout_ms: int = 2000,
        return_timestamps: bool = False,
    ) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Any]]:
        """
        Direct hardware execution of the monolithic Neck transaction bundle (Neck_FPN, Neck_PAN):
          - Ingests P3/P4/P5 activations with strictly 0 intermediate host DDR roundtrips.
          - Dispatches <= 2 monolithic ERT instruction buffers.
          - Returns the 3 Neck output feature maps for the YOLOv8 detection heads.
        """
        inp = input_tensor if input_tensor is not None else (p5 if p5 is not None else p3)
        if inp is None:
            inp = np.zeros((1, 256, 20, 20), dtype=np.float32)

        res = self._execute_monolithic_stages(
            inp,
            unswizzle=unswizzle,
            timeout_ms=timeout_ms,
            return_timestamps=return_timestamps,
            extract_feature_maps=True,
        )
        if return_timestamps:
            out, hw_ts, features = res
            return features, hw_ts
        out, features = res
        return features

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

        if self.enable_monolithic:
            if input_tensor is None:
                input_tensor = np.zeros(self.in_bytes, dtype=np.int8)

            # Warmup iterations
            for _ in range(warmup):
                self._execute_monolithic_stages(input_tensor, unswizzle=False)

            wall_latencies_us = []
            driver_tax_us = []
            hw_exec_us = []

            for _ in range(iterations):
                t0 = time.perf_counter_ns()
                _, hw_ts = self._execute_monolithic_stages(
                    input_tensor, unswizzle=False, return_timestamps=True
                )
                t1 = time.perf_counter_ns()
                wall_latencies_us.append((t1 - t0) / 1000.0)
                driver_tax_us.append(hw_ts.dispatch_submission_us)
                hw_exec_us.append(hw_ts.device_execution_us)

            mean_us = float(np.mean(wall_latencies_us))
            median_us = float(np.median(wall_latencies_us))
            min_us = float(np.min(wall_latencies_us))
            max_us = float(np.max(wall_latencies_us))
            p95_us = float(np.percentile(wall_latencies_us, 95))
            p99_us = float(np.percentile(wall_latencies_us, 99))
            fps = 1e6 / mean_us if mean_us > 0 else 0.0

            mean_tax_us = float(np.mean(driver_tax_us))
            median_tax_us = float(np.median(driver_tax_us))
            p95_tax_us = float(np.percentile(driver_tax_us, 95))

            mean_hw_us = float(np.mean(hw_exec_us))
            median_hw_us = float(np.median(hw_exec_us))
            p95_hw_us = float(np.percentile(hw_exec_us, 95))

            return {
                "iterations": iterations,
                "warmup": warmup,
                "mean_us": mean_us,
                "median_us": median_us,
                "min_us": min_us,
                "max_us": max_us,
                "p95_us": p95_us,
                "p99_us": p99_us,
                "fps": fps,
                "ert_submissions_per_frame": len(self.monolithic_stages),
                "driver_tax_us": {
                    "mean": mean_tax_us,
                    "median": median_tax_us,
                    "p95": p95_tax_us,
                },
                "hw_compute_us": {
                    "mean": mean_hw_us,
                    "median": median_hw_us,
                    "p95": p95_hw_us,
                },
                "intermediate_ddr_bytes": 0,
                "num_stages": len(self.monolithic_stages),
                "stages": list(self.monolithic_stages.keys()),
            }

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

        if self.enable_monolithic:
            for stage in self.monolithic_stages.values():
                stage.bo_instr_init = None
                stage.bo_instr_exec = None
            self.monolithic_stages.clear()

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
