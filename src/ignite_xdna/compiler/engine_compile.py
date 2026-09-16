"""Compile a YOLOv8n QDQ ONNX model into a graph-engine ``.ignite`` container.

The container carries everything the runtime needs to execute the whole
network on the NPU without the CPU oracle:

* ``engine.xclbin``   the 16-core convolution engine (kernels/aie2/conv_engine),
* ``insts.bin``       one instruction stream for all layers (layer barriers inside); a graph with
                      host layers instead carries ``insts_<i>.bin`` per NPU segment and
                      ``host_<j>.onnx`` per host layer, listed in ``graph_engine.segments`` in
                      execution order (dispatch a segment, run the host layer on the workspace,
                      dispatch the next),
* ``wpackets.bin``    the static weight packets,
* manifest ``graph_engine``: workspace size, tensor placements (so the runtime
  rebuilds the halo fill and finds the input and head tensors), per-layer
  statistics, and the ``head_layout`` contract of ``runtime/heads.py`` for the
  int8 NCHW egress the runtime assembles from the six head tensors.

Building needs the mlir-aie IRON environment (``scripts/research-iron.sh``);
planning, scheduling and emulation do not.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ignite_xdna.compiler import engine_emulator as em
from ignite_xdna.compiler import engine_schedule as es
from ignite_xdna.compiler.graph_ir import ConvLayer, GraphIR, HostLayer, lower_yolov8n
from ignite_xdna.compiler.serializer import ARCH_XDNA1_PHOENIX, IgniteModelReader, IgniteModelWriter

ENGINE_NAME = "conv_engine_v1"
HEAD_NAMES = ("p3_box", "p4_box", "p5_box", "p3_cls", "p4_cls", "p5_cls")
# YOLOv8-pose keeps the detect heads with one class (person) and adds a keypoint branch (cv4): 17 COCO
# keypoints x (x, y, visibility) per anchor.
POSE_HEAD_NAMES = HEAD_NAMES + ("p3_kpt", "p4_kpt", "p5_kpt")
POSE_KPT_SHAPE = (17, 3)
_HEAD_OUTPUT = re.compile(r"/cv([234])\.([012])/")
_HEAD_BRANCH = {"2": "box", "3": "cls", "4": "kpt"}


def head_name_for(onnx_output: str) -> str:
    """Map an ONNX head output name (/model.22/cv2.0/... -> p3_box, cv3.1 -> p4_cls, cv4.2 -> p5_kpt)."""
    m = _HEAD_OUTPUT.search(onnx_output)
    if m is None:
        raise ValueError(f"{onnx_output} is not a YOLOv8 head output (/cv2.*, /cv3.* or /cv4.*)")
    return f"p{3 + int(m.group(2))}_{_HEAD_BRANCH[m.group(1)]}"


def graph_task(ir: GraphIR) -> str:
    """The runtime contract a lowered graph's outputs fit: YOLO detect heads, YOLO pose heads (one person class
    plus 51 keypoint channels per level) or a dense upscaled image."""
    if len(ir.outputs) == 1 and ir.output_transforms.get(ir.outputs[0][0], {}).get("op") == "depth_to_space":
        return "super_resolution"
    if len(ir.outputs) == len(HEAD_NAMES):
        return "detect"
    if len(ir.outputs) == len(POSE_HEAD_NAMES) and all(_HEAD_OUTPUT.search(o) for o, _ in ir.outputs):
        heads = {head_name_for(o): ir.tensors[t] for o, t in ir.outputs}
        kpt_channels = POSE_KPT_SHAPE[0] * POSE_KPT_SHAPE[1]
        if (sorted(heads) == sorted(POSE_HEAD_NAMES)
                and all(heads[f"p{i}_cls"].channels == 1 and heads[f"p{i}_kpt"].channels == kpt_channels
                        for i in (3, 4, 5))):
            return "pose"
    raise ValueError(f"no runtime contract for graph outputs {[o for o, _ in ir.outputs]}")


# SESR's preprocessing (npu/sesr.py): the float input is the RGB pixel minus 128. It is not in the graph.
SR_INPUT_NORMALIZATION = {"mean": 128.0, "divisor": 1.0}


def plan_segments(ir: GraphIR, scheds: List[es.LayerSchedule]) -> List[Dict[str, Any]]:
    """Cut the scheduled layers at host layers, in execution order.

    An NPU segment is a half-open layer range with its DMA task count (the counts
    ``engine_sequence.split_instruction_stream`` cuts the lowered stream by); a host segment names its
    layer, input and output tensors. Blob names are assigned in order.
    """
    from ignite_xdna.compiler.engine_sequence import program_task_count
    segments: List[Dict[str, Any]] = []
    start, tasks = None, 0
    for s in scheds:
        L = ir.layers[s.layer_index]
        if isinstance(L, HostLayer):
            if start is not None:
                segments.append({"kind": "npu", "layers": [start, s.layer_index], "tasks": tasks})
            segments.append({"kind": "host", "layer": s.layer_index, "name": L.name, "input": L.input.tensor,
                             "output": L.output, "op_types": dict(L.op_types)})
            start, tasks = None, 0
        else:
            if start is None:
                start = s.layer_index
            tasks += sum(program_task_count(p) for p in s.programs)
    if start is not None:
        segments.append({"kind": "npu", "layers": [start, scheds[-1].layer_index + 1], "tasks": tasks})
    n_npu = n_host = 0
    for seg in segments:
        if seg["kind"] == "npu":
            seg["blob"] = f"insts_{n_npu}.bin"
            n_npu += 1
        else:
            seg["blob"] = f"host_{n_host}.onnx"
            n_host += 1
    return segments


def build_manifest(ir: GraphIR, ws: es.Workspace, scheds: List[es.LayerSchedule], store: es.PacketStore,
                   model_name: str, insts_bytes: int, xclbin_sha: str, kernel_sha: str, compile_s: float) -> Dict[str, Any]:
    placements = {}
    for name, p in ws.placements.items():
        placements[name] = {"base": p.base, "halo": p.halo, "halo_value": p.halo_value, "height": p.height,
                            "width": p.width, "blocks": p.blocks, "planes": p.planes, "channels": ir.tensors[name].channels,
                            "scale": ir.tensors[name].scale, "zero_point": ir.tensors[name].zero_point}
    task = graph_task(ir)
    t_in = ir.tensors[ir.input]
    layers = [{"index": s.layer_index, "name": s.name, "output": ir.layers[s.layer_index].output,
               "rounds": s.rounds, "packets": s.packets, "w_fills": s.w_fills,
               **({"kind": "host"} if isinstance(ir.layers[s.layer_index], HostLayer) else {})} for s in scheds]
    graph_engine = {
        "kernel_sha256": kernel_sha,
        "xclbin_sha256": xclbin_sha,
        "workspace_bytes": ws.nbytes,
        "input_tensor": ir.input,
        "placements": placements,
        "layers": layers,
        "rounds": sum(s.rounds for s in scheds),
        "activation_packets": sum(s.packets for s in scheds),
        "weight_fills": sum(s.w_fills for s in scheds),
        "wpackets_bytes": store.nbytes,
        "insts_bytes": insts_bytes,
        "compile_seconds": round(compile_s, 1),
        "tile": {"rows": es.TILE_R, "cols": es.TILE_C, "a_bytes": em.A_BYTES, "w_bytes": em.W_BYTES,
                 "o_bytes": em.O_BYTES},
    }
    manifest: Dict[str, Any] = {
        "model_name": model_name,
        "format_version": 1,
        "arch_id": ARCH_XDNA1_PHOENIX,
        "target_hardware": "AMD Phoenix APU (XDNA1, 16 AIE2 cores)",
        "producer": "ignite-compile v0.4.0 (graph engine)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": ENGINE_NAME,
        "task": task,
        "input_shape": [1, t_in.channels, t_in.height, t_in.width],
        "input_dtype": "int8" if task in ("detect", "pose") else "uint8",
        "single_dispatch": True,
        "quant_scales": {"input_scale": t_in.scale, "input_zero_point": t_in.zero_point, "input_dtype": "uint8"},
        "graph_engine": graph_engine,
        "num_stages": len(scheds),
        "stages": {},
    }
    if task in ("detect", "pose"):
        heads = {}
        offset = 0
        output_shapes = {}
        head_layout = {}
        for onnx_name, tensor in ir.outputs:
            hn = head_name_for(onnx_name)
            t = ir.tensors[tensor]
            nbytes = t.channels * t.height * t.width
            heads[hn] = {"tensor": tensor, "onnx_output": onnx_name, "channels": t.channels, "height": t.height,
                         "width": t.width, "egress_offset": offset}
            output_shapes[hn] = [1, t.channels, t.height, t.width]
            # The runtime converts uint8 (zero point 128) to int8 by flipping the top
            # bit, so the int8 view has zero point 0 at the same scale.
            head_layout[hn] = {"offset": offset, "scale": t.scale, "zero_point": 0}
            offset += nbytes
        graph_engine["heads"] = heads
        manifest.update({"strides": [8, 16, 32], "reg_max": 16, "num_classes": 80 if task == "detect" else 1,
                         "fused_dfl": False, "output_shapes": output_shapes, "head_layout": head_layout,
                         "egress_bytes": offset})
        if task == "pose":
            manifest["kpt_shape"] = list(POSE_KPT_SHAPE)
    else:
        # Dense egress: the tail tensor is read back as [blocks][H][W][8] uint8 (zero point 128) and the
        # host applies the model's DepthToSpace; the image is clip((q - zp) * scale + mean).
        onnx_name, tensor = ir.outputs[0]
        t = ir.tensors[tensor]
        transform = ir.output_transforms[onnx_name]
        bs = int(transform["blocksize"])
        manifest.update({
            "upscale": bs,
            "input_normalization": dict(SR_INPUT_NORMALIZATION),
            "output_shapes": {"image": [1, t.channels // (bs * bs), t.height * bs, t.width * bs]},
            "dense_output": {"tensor": tensor, "onnx_output": onnx_name, "channels": t.channels, "height": t.height,
                             "width": t.width, "scale": t.scale, "zero_point": t.zero_point,
                             "layout": "blocks_hw8", "transform": transform},
            "egress_bytes": t.channels * t.height * t.width,
        })
    return manifest


def compile_graph_container(onnx_path, output_path, build_dir: Optional[Path] = None, layers: Optional[int] = None,
                            verbose: bool = True, host_regions: Sequence[str] = (),
                            activation_ring: int = 0) -> Dict[str, Any]:
    """Lower, schedule, build the device binaries and write the container. Returns the manifest.

    ``host_regions`` are node-name prefixes or ``FROM=TO`` boundaries run on the host between dispatches
    (``graph_ir.HostLayer``).

    ``activation_ring`` (slots, 0 = off) builds the engine with a hand-written MemTile ring for activations
    instead of the split ObjectFifo, so one shim fetch can serve several output groups of a tile.
    """
    import aie.iron as iron
    from aie.iron.device import NPU1
    from aie.utils.compile.utils import compile_mlir_module
    from ignite_xdna.compiler.engine_sequence import SequenceEmitter
    from kernels.aie2.conv_engine import design as eng

    t0 = time.perf_counter()
    onnx_path = Path(onnx_path)
    out_p = Path(output_path)
    build_dir = Path(build_dir or out_p.parent / "conv_engine" / out_p.stem).resolve()
    ir = lower_yolov8n(onnx_path, host_regions=host_regions)
    ws = es.plan_workspace(ir)
    scheds, store = es.schedule_graph(ir, ws, activation_ring=activation_ring)
    if layers is not None:
        scheds = scheds[:layers]
    if verbose:
        n_host = sum(isinstance(L, HostLayer) for L in ir.layers)
        print(f"[*] Lowered {len(ir.layers)} layers ({n_host} on the host); workspace {ws.nbytes / 1e6:.1f} MB; "
              f"{sum(s.rounds for s in scheds)} rounds, {store.nbytes / 1e6:.2f} MB static packets")

    # yolov8s streams 29.5 MB of weight packets, past the 16 MB default packet extent.
    ws_extent, wp_extent = eng.ddr_extents(ws.nbytes, store.nbytes)

    def body(ws_arg, wp_arg):
        emitter = SequenceEmitter(ws_arg, wp_arg, ws_extent, wp_extent,
                                  {c: eng.fifo_names(c) for c in range(eng.COLS)})
        for s in scheds:
            emitter.run_column_programs(s.programs, bd_budget=14)

    iron.set_current_device(NPU1())
    program = eng.build_program(iron.get_current_device(), body, ws_bytes=ws_extent, wp_bytes=wp_extent,
                                a_ring=activation_ring)
    module = program.resolve_program()
    work = build_dir / "design.prj"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    (build_dir / "design.mlir").write_text(str(module), encoding="utf-8")
    xclbin_path = build_dir / "engine.xclbin"
    insts_path = build_dir / "insts.bin"
    compile_mlir_module(module, insts_path=insts_path, xclbin_path=xclbin_path, work_dir=work,
                        device=iron.get_current_device())
    xclbin = xclbin_path.read_bytes()
    insts = insts_path.read_bytes()
    kernel_sha = hashlib.sha256(eng.KERNEL_SOURCE.read_bytes()).hexdigest()
    manifest = build_manifest(ir, ws, scheds, store, onnx_path.stem, len(insts),
                              hashlib.sha256(xclbin).hexdigest(), kernel_sha, time.perf_counter() - t0)
    manifest["graph_engine"]["ddr_extents_bytes"] = {"workspace": ws_extent, "packets": wp_extent}
    segments = plan_segments(ir, scheds)
    hosts = [seg for seg in segments if seg["kind"] == "host"]
    blobs = [("engine.xclbin", xclbin, "xclbin")]
    if hosts:
        # One stream per NPU segment: the emitter retires every task at each layer barrier, so the lowered
        # stream cuts cleanly before and after a host layer (which issues no tasks).
        from ignite_xdna.compiler.engine_sequence import split_instruction_stream
        npu = [seg for seg in segments if seg["kind"] == "npu"]
        pieces = split_instruction_stream(insts, [seg["tasks"] for seg in npu])
        for seg, piece in zip(npu, pieces):
            seg["insts_bytes"] = len(piece)
            blobs.append((seg["blob"], piece, "npu_instructions"))
        for seg in hosts:
            onnx_bytes = ir.layers[seg["layer"]].onnx_bytes
            seg["onnx_sha256"] = hashlib.sha256(onnx_bytes).hexdigest()
            blobs.append((seg["blob"], onnx_bytes, "onnx_host_model"))
        manifest["single_dispatch"] = False
        manifest["graph_engine"]["segments"] = segments
    else:
        blobs.append(("insts.bin", insts, "npu_instructions"))
    blobs.append(("wpackets.bin", store.blob().tobytes(), "weight_packets"))
    writer = IgniteModelWriter(manifest_meta=manifest, arch_id=ARCH_XDNA1_PHOENIX)
    for name, data, content_type in blobs:
        writer.add_blob(name, data, content_type=content_type)
    total = writer.write(out_p)
    with IgniteModelReader(out_p) as reader:
        if not reader.verify_checksum():
            raise RuntimeError("container checksum verification failed")
    if verbose:
        print(f"[+] {out_p} written: {total:,} bytes in {time.perf_counter() - t0:.1f} s "
              f"(xclbin {len(xclbin):,} B, insts {len(insts):,} B, packets {store.nbytes:,} B)")
    return manifest
