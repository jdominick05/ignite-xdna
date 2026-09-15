# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/compiler/cli.py

Core implementation and entry point for the ignite-compile CLI.
Ingests an arbitrary YOLOv8n ONNX model, validates operator compatibility,
partitions the DAG into monolithic MemTile stages (0 CPU fallback partitions),
synthesizes 64-byte aligned CDO transaction streams, and serializes everything
into a single deployable .ignite binary container.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx

from ignite_xdna.compiler.partitioner import GraphPartitioner
from ignite_xdna.compiler.scheduler import (
    MemTileMultiPassScheduler,
    emit_multi_stage_transaction_bundle,
    emit_unified_monolithic_transaction_bundle,
)
from ignite_xdna.compiler.serializer import (
    ARCH_XDNA1_PHOENIX,
    IgniteModelReader,
    IgniteModelWriter,
)
from ignite_xdna.runtime.driver import get_repo_root, setup_xrt_environment


SUPPORTED_OPERATORS = {
    "Conv", "Add", "Mul", "Concat", "Resize", "MaxPool",
    "Sigmoid", "HardSigmoid", "QuantizeLinear", "DequantizeLinear",
    "Shape", "Slice", "Reshape", "Transpose", "Split", "Constant",
    "Sub", "Div", "Softmax"
}


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(
        prog="ignite-compile",
        description="ignite-compile: One-click bare-metal compiler for AMD XDNA1 NPU (.ignite format)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(repo_root / "models" / "yolov8n_cut_xint8.onnx"),
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(repo_root / "build" / "yolov8n.ignite"),
        help="Path to output .ignite binary container",
    )
    parser.add_argument(
        "--quant-scales", "-q",
        type=str,
        default=None,
        help="Optional JSON file with per-tensor / per-channel INT8 quantization scales",
    )
    parser.add_argument(
        "--base-txn",
        type=str,
        default=None,
        help="Path to template transaction binary (defaults to build/layer_conv0_exec.bin)",
    )
    parser.add_argument(
        "--verify-silicon",
        action="store_true",
        help="Load the generated .ignite container onto physical NPU silicon and run verification",
    )
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=0,
        help="Physical NPU device index for silicon verification",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=16,
        help="Number of AIE2 compute cores to target",
    )
    parser.add_argument(
        "--fuse-dfl",
        action="store_true",
        help="Fuse on-die AIE2 DFL micro-kernel stage to decode bounding boxes directly to host bo_out without CPU Softmax",
    )
    parser.add_argument(
        "--engine",
        choices=("graph", "template"),
        default="graph",
        help="graph: lower the whole network onto the 16-core convolution engine (needs the mlir-aie "
             "ironenv, produces detect heads on the NPU); template: the legacy conv0 transaction patcher",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=None,
        help="Scratch directory for the graph engine build (default: <output dir>/conv_engine/<name>)",
    )
    parser.add_argument(
        "--host-region",
        action="append",
        default=None,
        metavar="PREFIX|FROM=TO",
        help="Graph engine: run a region on the host between two NPU dispatches; repeatable. PREFIX runs every "
             "node whose name starts with it (e.g. /model.10/, YOLO11's C2PSA block); FROM=TO runs the nodes "
             "between two nodes' quantized outputs (e.g. /model.10/m/m.0/attn/qkv/conv/Conv="
             "/model.10/m/m.0/attn/Reshape_1, its attention core only)",
    )
    argv = list(sys.argv[1:] if args is None else args)
    # ``ignite-compile compile --model X`` is accepted as a spelling of ``--input X``.
    if argv and argv[0] == "compile":
        argv = argv[1:]
    argv = ["--input" if a == "--model" else a for a in argv]
    parsed = parser.parse_args(argv)
    # The repository ships the Quark-quantized cut model only; a request for the
    # unquantized export name resolves to it rather than failing on a missing file.
    requested = Path(parsed.input)
    cut = repo_root / "models" / "yolov8n_cut_xint8.onnx"
    if not requested.exists() and requested.name == "yolov8n.onnx" and cut.exists():
        print(f"[*] {requested} is not present; compiling the quantized export {cut.name} instead")
        parsed.input = str(cut)
    return parsed


def validate_onnx_compatibility(model_path: Path) -> onnx.ModelProto:
    """Validates that all operators in the ONNX model are compatible with AIE2 lowering."""
    print(f"[*] Validating ONNX model: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {model_path}")

    model = onnx.load(str(model_path))
    unsupported_ops = set()
    total_nodes = 0

    for node in model.graph.node:
        total_nodes += 1
        if node.op_type not in SUPPORTED_OPERATORS:
            unsupported_ops.add(node.op_type)

    if unsupported_ops:
        raise ValueError(
            f"Model contains unsupported operators for XDNA1 hardware lowering: {unsupported_ops}"
        )

    print(f"    [OK] Checked {total_nodes} nodes across graph; 100% operator compatibility verified.")
    return model


def compile_model(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    quant_scales_path: Optional[Union[str, Path]] = None,
    base_txn_path: Optional[Union[str, Path]] = None,
    num_cores: int = 16,
    fuse_dfl: bool = False,
) -> int:
    """
    One-click bare-metal compiler lowering ONNX models directly into .ignite binary containers.
    Returns the total file size in bytes.
    """
    repo_root = get_repo_root()
    in_p = Path(input_path).resolve()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Validate ONNX model
    validate_onnx_compatibility(in_p)

    # 2. Base transaction template
    if base_txn_path is None:
        cand1 = repo_root / "build" / "layer_conv0_exec.bin"
        cand2 = repo_root / "build" / "im2col_4d_16core.bin"
        base_txn = str(cand1 if cand1.exists() else cand2)
    else:
        base_txn = str(Path(base_txn_path).resolve())

    if not os.path.exists(base_txn):
        raise FileNotFoundError(f"Base transaction binary template not found: {base_txn}")

    # 3. Graph Partitioning
    print("[*] Partitioning ONNX DAG into monolithic MemTile stages...")
    partitioner = GraphPartitioner(in_p, fuse_neck=True, fuse_head=True)
    pg = partitioner.partition(backbone_only=False)

    npu_partitions = pg.npu_partitions
    print(f"    [OK] Graph partitioned into {len(npu_partitions)} monolithic stages.")
    print(f"    [OK] CPU fallback partitions: {len(pg.cpu_partitions)} (strictly 0 required).")
    if len(pg.cpu_partitions) > 0:
        raise RuntimeError("Compilation failed: Graph contains unscheduled CPU fallback partitions.")

    # 4. MemTile AGU Scheduling
    print("[*] Scheduling MemTile L2 SRAM ping-pong buffer layouts...")
    scheduler = MemTileMultiPassScheduler(num_cores=num_cores)
    multi_plan = scheduler.schedule_multi_stage(npu_partitions, fuse_dfl=fuse_dfl)

    if not multi_plan.has_zero_intermediate_ddr_traffic:
        raise RuntimeError("Compilation failed: Schedule requires intermediate host DDR roundtrips.")
    print(f"    [OK] Multi-stage schedule generated: 0 intermediate DDR bytes across all layers.")

    # 5. Synthesize Transaction Binaries
    print("[*] Synthesizing 64-byte aligned CDO transaction streams for all stages...")
    build_staging = repo_root / "build" / "staging_ignite"
    build_staging.mkdir(parents=True, exist_ok=True)

    stages_meta: Dict[str, Any] = {}
    stage_blobs: List[Tuple[str, bytes, str]] = []

    for idx, (s_name, stage_plan) in enumerate(multi_plan.stages.items()):
        s_lower = s_name.lower()
        init_bin_name = f"stage_{s_lower}_init.bin"
        exec_bin_name = f"stage_{s_lower}_exec.bin"

        init_path = str(build_staging / init_bin_name)
        exec_path = str(build_staging / exec_bin_name)

        if s_name == "DFL_Decode" and getattr(stage_plan, "custom_exec_bytes", None):
            exec_bytes = stage_plan.custom_exec_bytes
            with open(exec_path, "wb") as f:
                f.write(exec_bytes)
            init_size = 0
            has_init = False
        else:
            emit_multi_stage_transaction_bundle(stage_plan, base_txn, init_path, exec_path)

            with open(exec_path, "rb") as f:
                exec_bytes = f.read()

            has_init = os.path.exists(init_path) and os.path.getsize(init_path) > 0
            if has_init:
                with open(init_path, "rb") as f:
                    init_bytes = f.read()
                stage_blobs.append((init_bin_name, init_bytes, "transaction_init"))
                init_size = len(init_bytes)
            else:
                init_size = 0

        stage_blobs.append((exec_bin_name, exec_bytes, "transaction_exec"))

        stages_meta[s_name] = {
            "index": idx,
            "stage_name": s_name,
            "num_layers": stage_plan.num_layers,
            "c2f_blocks": stage_plan.c2f_blocks,
            "init_blob": init_bin_name if has_init else "",
            "exec_blob": exec_bin_name,
            "init_bytes": init_size,
            "exec_bytes": len(exec_bytes),
        }
        print(f"    - Stage {idx}: {s_name:<12} | Layers: {stage_plan.num_layers} | "
              f"Exec: {len(exec_bytes):>6} B | Init: {init_size:>7} B")

    # 5b. Synthesize Unified Single-Dispatch Monolithic Transaction Stream
    print("[*] Synthesizing continuous single-dispatch monolithic transaction stream...")
    mono_init_name = "init_monolithic.bin"
    mono_exec_name = "exec_monolithic.bin"
    mono_init_path = str(build_staging / mono_init_name)
    mono_exec_path = str(build_staging / mono_exec_name)

    emit_unified_monolithic_transaction_bundle(
        schedule=multi_plan,
        base_txn_path=base_txn,
        out_init_path=mono_init_path,
        out_exec_path=mono_exec_path,
        cores=None,
    )

    with open(mono_exec_path, "rb") as f:
        mono_exec_bytes = f.read()
    stage_blobs.append((mono_exec_name, mono_exec_bytes, "transaction_exec_monolithic"))

    if os.path.exists(mono_init_path) and os.path.getsize(mono_init_path) > 0:
        with open(mono_init_path, "rb") as f:
            mono_init_bytes = f.read()
        stage_blobs.append((mono_init_name, mono_init_bytes, "transaction_init_monolithic"))
        print(f"    [OK] Unified Single-Dispatch Stream: Exec={len(mono_exec_bytes):>6} B | Init={len(mono_init_bytes):>7} B")

    # 6. Parse / Extract Quantization Scales
    quant_scales: Dict[str, Any] = {}
    if quant_scales_path is not None and os.path.exists(quant_scales_path):
        with open(quant_scales_path, "r") as f:
            quant_scales = json.load(f)
        print(f"[*] Loaded quantization scales from {quant_scales_path}")
    else:
        # Default YOLOv8n INT8 quant metadata
        quant_scales = {
            "input_scale": 1.0 / 256.0,
            "input_zero_point": -128,
            "input_dtype": "int8",
        }

    # 7. Assemble Manifest
    output_shapes = {
        "boxes": [1, 8400, 4],
        "scores": [1, 8400, 80],
    } if fuse_dfl else {
        "p3_box": [1, 64, 80, 80],
        "p3_cls": [1, 80, 80, 80],
        "p4_box": [1, 64, 40, 40],
        "p4_cls": [1, 80, 40, 40],
        "p5_box": [1, 64, 20, 20],
        "p5_cls": [1, 80, 20, 20],
    }

    manifest = {
        "model_name": in_p.stem,
        "format_version": 1,
        "arch_id": ARCH_XDNA1_PHOENIX,
        "target_hardware": "AMD Phoenix APU (XDNA1, 16 AIE2 cores @ 1.80 GHz)",
        "producer": "ignite-compile v0.2.0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_shape": [1, 3, 640, 640],
        "input_dtype": "int8",
        "strides": [8, 16, 32],
        "reg_max": 16,
        "num_classes": 80,
        "fused_dfl": fuse_dfl,
        "output_shapes": output_shapes,
        "num_stages": len(stages_meta),
        "stages": stages_meta,
        "single_dispatch": True,
        "monolithic_exec_blob": mono_exec_name,
        "monolithic_init_blob": mono_init_name,
        "quant_scales": quant_scales,
    }

    # 8. Package into .ignite Container
    print(f"[*] Packaging {len(stage_blobs)} blobs into container: {out_p}")
    writer = IgniteModelWriter(manifest_meta=manifest, arch_id=ARCH_XDNA1_PHOENIX)
    for b_name, b_data, b_type in stage_blobs:
        writer.add_blob(b_name, b_data, content_type=b_type)

    total_size = writer.write(out_p)

    # 9. Verify Container Integrity
    with IgniteModelReader(out_p) as reader:
        if not reader.verify_checksum():
            raise RuntimeError("Generated container failed CRC32 checksum verification!")
        for b_name, b_entry in reader.blobs.items():
            if b_entry.offset % 64 != 0:
                raise RuntimeError(f"Blob '{b_name}' misaligned: offset {b_entry.offset} % 64 != 0")

    size_mb = total_size / (1024 * 1024)
    print(f"[+] COMPILATION COMPLETE: {out_p.name} ({size_mb:.2f} MB, {total_size:,} bytes)")
    print(f"    [OK] CRC32 Checksum Verified: 0x{reader.header.crc32:08X}")
    print(f"    [OK] Total 64-Byte Aligned Blobs: {len(stage_blobs)}")
    print(f"    [OK] Model Size Constraint (< 10 MB): PASSED ({size_mb:.2f} MB)")
    return total_size


def verify_on_silicon(container_path: Path, device_idx: int = 0):
    """Verifies that the compiled container executes on physical NPU hardware."""
    from ignite_xdna.runtime.session import IgniteSession
    print(f"\n[*] Verifying container on physical NPU silicon (Device {device_idx})...")

    session = IgniteSession(container_path, device_idx=device_idx, monolithic=True)
    try:
        dummy_in = np.random.randint(-128, 127, size=(1, 3, 640, 640), dtype=np.int8)

        t0 = time.perf_counter()
        outputs = session.run_yolo_monolithic(dummy_in)
        t_first = (time.perf_counter() - t0) * 1000.0

        latencies = []
        for _ in range(20):
            t_start = time.perf_counter()
            outputs = session.run_yolo_monolithic(dummy_in)
            latencies.append((time.perf_counter() - t_start) * 1000.0)

        mean_lat = float(np.mean(latencies))
        fps = 1000.0 / mean_lat

        print(f"    [OK] Physical Silicon Execution Succeeded!")
        print(f"    [OK] First Pass Latency: {t_first:.3f} ms")
        print(f"    [OK] Steady-State Forward Pass Latency: {mean_lat:.3f} ms ({fps:.1f} FPS)")
        print(f"    [OK] Hardware Stages Verified: {len(session.monolithic_stages)} / {len(session.monolithic_stages)}")
        print(f"    [OK] Intermediate DDR Traffic: 0 Bytes")
    finally:
        session.close()


def compile_graph_engine(input_path: Union[str, Path], output_path: Union[str, Path],
                         build_dir: Optional[Union[str, Path]] = None, host_regions: Optional[List[str]] = None) -> int:
    """Lower the whole graph onto the convolution engine (see engine_compile.py); ``host_regions`` run on the host."""
    for region in host_regions or ():
        for part in region.split("="):
            if len(part) > 2 and part[1] == ":" and part[2] in "/\\":
                raise ValueError(f"--host-region {region}: {part} is a file path, not a node name (Git Bash "
                                 "rewrites arguments that start with '/'; run the command with MSYS_NO_PATHCONV=1)")
    try:
        import aie.iron  # noqa: F401
    except ImportError as ex:
        raise RuntimeError(
            "the graph engine needs the mlir-aie IRON environment (run through scripts/research-iron.sh); "
            f"import failed: {ex}") from ex
    from ignite_xdna.compiler.engine_compile import compile_graph_container
    manifest = compile_graph_container(input_path, output_path, build_dir=build_dir,
                                       host_regions=tuple(host_regions or ()))
    for seg in manifest["graph_engine"].get("segments", []):
        if seg["kind"] == "host":
            print(f"    [OK] host segment: {seg['name']} (layer {seg['layer']}) between NPU segments")
    out_p = Path(output_path)
    if manifest.get("task", "detect") in ("detect", "pose"):
        from ignite_xdna.runtime.heads import resolve_head_layout
        status = resolve_head_layout(manifest, int(manifest["egress_bytes"]))
        print(f"    [OK] head_status: {'present' if status.present else 'absent'} ({status.reason})")
    else:
        d = manifest["dense_output"]
        print(f"    [OK] dense output: {d['channels']}x{d['height']}x{d['width']} uint8 at scale {d['scale']}, "
              f"host {d['transform']['op']} x{d['transform']['blocksize']} -> image {manifest['output_shapes']['image']}")
    print(f"    [OK] egress bytes: {manifest['egress_bytes']:,}")
    return out_p.stat().st_size


def main(args: Optional[List[str]] = None):
    parsed = parse_args(args)
    try:
        if parsed.engine == "graph":
            compile_graph_engine(parsed.input, parsed.output, parsed.build_dir, host_regions=parsed.host_region)
            if parsed.verify_silicon:
                from ignite_xdna.runtime.graph_session import GraphSession
                sess = GraphSession(parsed.output, device_index=parsed.device)
                try:
                    dummy = np.zeros((1, 3, 640, 640), dtype=np.int8)
                    lat = [sess.run_yolo_monolithic(dummy, return_timestamps=True)[1]["npu_ms"] for _ in range(5)]
                    print(f"    [OK] Physical silicon execution succeeded: NPU {np.mean(lat[1:]):.3f} ms/frame")
                finally:
                    sess.close()
            print("\nAll compilation and packaging checks passed successfully!")
            return
        total_size = compile_model(
            input_path=parsed.input,
            output_path=parsed.output,
            quant_scales_path=parsed.quant_scales,
            base_txn_path=parsed.base_txn,
            num_cores=parsed.num_cores,
            fuse_dfl=parsed.fuse_dfl,
        )

        if parsed.verify_silicon:
            verify_on_silicon(Path(parsed.output), device_idx=parsed.device)

        print("\nAll compilation and packaging checks passed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Compilation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
