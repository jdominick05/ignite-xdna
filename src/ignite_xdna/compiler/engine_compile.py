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
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ignite_xdna.compiler import engine_schedule as es
from ignite_xdna.compiler.graph_ir import ConvLayer, GraphIR, HostLayer, lower_yolov8n
from ignite_xdna.compiler.serializer import (ARCH_XDNA1_PHOENIX, ELEM_BF16, ELEM_INT, ENGINE_CONV_BF16,
                                             ENGINE_CONV_INT8, IgniteModelReader, IgniteModelWriter)

ENGINE_NAME = ENGINE_CONV_INT8


@dataclass(frozen=True)
class EngineProfile:
    """Everything about a compile that differs between the int8 and the bf16 engine.

    One bundle rather than six parameters, because these are NOT independent. A design module, its
    emulator and its scheduler agree on a packet header layout and an opcode table; mixing them
    across engines produces a container whose header words mean something other than what wrote
    them, and the symptom is a wrong name or a false refusal rather than a crash.

    Modules are named rather than held, because importing a design pulls IRON in and the compile
    is the only caller that wants that cost. importlib caches, so resolving one twice is free.
    """

    name: str                       # what the container declares in manifest["engine"]
    elem: str                       # ELEM_INT writes no placement key at all; absent means integer
    design_module: str
    emulator_module: str
    schedule_module: Optional[str]
    object_file: str = "engine.o"
    #: Whether this design's ``build_program`` takes the dispatch gate and the two transport
    #: experiments (``ops``, ``a_ring``, ``w_buf``). The bf16 design takes none of them.
    transport_options: bool = True

    @property
    def emulator(self):
        return importlib.import_module(self.emulator_module)

    @property
    def schedule(self):
        if self.schedule_module is None:
            raise NotImplementedError(
                f"{self.name} has no scheduler yet: the packer that builds its weight and "
                f"activation packets is the fork tracked as the bf16 schedule, and a container "
                f"cannot be emitted without it. The rest of this profile is wired and tested.")
        return importlib.import_module(self.schedule_module)

    def load_design(self):
        return importlib.import_module(self.design_module)


INT8_PROFILE = EngineProfile(
    name=ENGINE_CONV_INT8, elem=ELEM_INT,
    design_module="kernels.aie2.conv_engine.design",
    emulator_module="ignite_xdna.compiler.engine_emulator",
    schedule_module="ignite_xdna.compiler.engine_schedule",
)

BF16_PROFILE = EngineProfile(
    name=ENGINE_CONV_BF16, elem=ELEM_BF16,
    design_module="kernels.bf16_conv.design",
    emulator_module="ignite_xdna.compiler.engine_bf16_emulator",
    schedule_module=None,
    object_file="engine_bf16.o",
    transport_options=False,
)

ENGINE_PROFILES = {"int8": INT8_PROFILE, "bf16": BF16_PROFILE}


def profile_for_manifest(manifest: Dict[str, Any]) -> EngineProfile:
    """The profile a built container was produced with, so a reader walks it with its own tables."""
    name = (manifest or {}).get("engine")
    for profile in ENGINE_PROFILES.values():
        if profile.name == name:
            return profile
    raise ValueError(f"no engine profile for {name!r}; known: "
                     f"{[p.name for p in ENGINE_PROFILES.values()]}")


HEAD_NAMES = ("p3_box", "p4_box", "p5_box", "p3_cls", "p4_cls", "p5_cls")
# YOLOv8-pose keeps the detect heads with one class (person) and adds a keypoint branch (cv4): 17 COCO
# keypoints x (x, y, visibility) per anchor.
POSE_HEAD_NAMES = HEAD_NAMES + ("p3_kpt", "p4_kpt", "p5_kpt")
POSE_KPT_SHAPE = (17, 3)
# An end-to-end, NMS-free detector carries two head branches and exports the one2one one: YOLO26's
# inference head is /model.23/one2one_cv2.0/..., where YOLOv8's is /model.22/cv2.0/.... The branch
# and level semantics are identical, so the prefix is optional here. It is spelled out rather than
# matched as a general \w+_ so an unexpected third naming scheme still fails loudly.
_HEAD_OUTPUT = re.compile(r"/(?:one2one_)?cv([234])\.([012])/")
_HEAD_BRANCH = {"2": "box", "3": "cls", "4": "kpt"}


def head_name_for(onnx_output: str) -> str:
    """Map an ONNX head output name (/model.22/cv2.0/... -> p3_box, cv3.1 -> p4_cls, cv4.2 -> p5_kpt).

    ``one2one_cv2.0`` maps to the same ``p3_box``: it is an NMS-free detector's inference branch.
    """
    m = _HEAD_OUTPUT.search(onnx_output)
    if m is None:
        raise ValueError(f"{onnx_output} is not a detect head output "
                         f"(/cv2.*, /cv3.*, /cv4.* or the one2one_ form of those)")
    return f"p{3 + int(m.group(2))}_{_HEAD_BRANCH[m.group(1)]}"


def graph_task(ir: GraphIR) -> str:
    """The runtime contract a lowered graph's outputs fit: YOLO detect heads, YOLO pose heads (one person class
    plus 51 keypoint channels per level), classification logits, a dense upscaled image, or the segmentation and
    matting maps a dense lowering declares for itself.

    A lowering that knows its own contract says so, and is believed. Everything else is inferred from the
    outputs, and for a single output the test is the rank the MODEL declares, not the tensor's placement.
    `resnet50_head` carries its logits in a 1000x20x20 workspace tensor but its graph output is `(1, 1000)`:
    a vector, read at one pixel. A segmentation map's graph output is `(1, 19, 512, 512)`. So a rank-4 single
    output is a map, and calling it classification is how a segmentation graph would silently compile to a
    container declaring `num_classes` and no dense output - refuse it and make the caller name the task.
    """
    declared = getattr(ir, "task", "") or ""
    if declared:
        return declared
    if len(ir.outputs) == 1 and ir.output_transforms.get(ir.outputs[0][0], {}).get("op") == "depth_to_space":
        return "super_resolution"
    if len(ir.outputs) == 1:
        onnx_name, tensor = ir.outputs[0]
        t = ir.tensors[tensor]
        if t.shape is not None and len(t.shape) <= 2:
            return "classify"
        shape = list(t.shape) if t.shape is not None else [1, t.channels, t.height, t.width]
        raise ValueError(
            f"the single output {onnx_name} has shape {shape}, which is a map rather than a vector of logits. "
            "A dense graph must name its task explicitly (segment or matte); inferring classify from one "
            "output would emit num_classes and no dense output for it.")
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

# Tasks whose output is a map the size of the input rather than a head, a vector or an upscaled image.
DENSE_TASKS = ("segment", "matte")


def plan_segments(ir: GraphIR, scheds: List[es.LayerSchedule],
                  split_layers: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    """Cut the scheduled layers at host layers and explicit split points, in execution order.

    An NPU segment is a half-open layer range with its DMA task count (the counts
    ``engine_sequence.split_instruction_stream`` cuts the lowered stream by); a host segment names its
    layer, input and output tensors. Blob names are assigned in order.
    """
    from ignite_xdna.compiler.engine_sequence import program_task_count
    segments: List[Dict[str, Any]] = []
    splits = set(split_layers or ())
    start, tasks = None, 0
    for s in scheds:
        L = ir.layers[s.layer_index]
        if isinstance(L, HostLayer):
            if start is not None and tasks > 0:
                segments.append({"kind": "npu", "layers": [start, s.layer_index], "tasks": tasks})
            seg = {"kind": "host", "layer": s.layer_index, "name": L.name, "input": L.input.tensor,
                   "output": L.output, "op_types": dict(L.op_types)}
            if L.inputs:  # a Concat view over several tensors; a single-tensor input keeps the older manifest form
                seg["inputs"] = [{"tensor": g.tensor, "block_offset": g.block_offset, "blocks": g.blocks}
                                 for g in L.inputs]
                seg["in_channels"] = L.in_channels
            if L.named_inputs and L.named_outputs:
                # A host region that names its boundaries may take several tensors and produce several, and may
                # keep a value on the CPU between two regions. The older single-in single-out keys stay beside
                # these so a runtime that does not know version 2 still describes the segment correctly.
                from ignite_xdna.compiler.dense_regions import boundary_metadata
                seg["boundary_version"] = 2
                seg["input_bindings"] = [boundary_metadata(ir.tensors[t], name=n)
                                         for n, t in L.named_inputs.items()]
                seg["output_bindings"] = [boundary_metadata(ir.tensors[t], name=n)
                                          for n, t in L.named_outputs.items()]
            segments.append(seg)
            start, tasks = None, 0
        else:
            if s.layer_index in splits and start is not None and tasks > 0:
                segments.append({"kind": "npu", "layers": [start, s.layer_index], "tasks": tasks})
                start = None
                tasks = 0
            if start is None:
                start = s.layer_index
            tasks += sum(program_task_count(p) for p in s.programs)
    if start is not None and tasks > 0:
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


def check_kernel_covers_packets(ge: dict, wpackets: bytes,
                                profile: EngineProfile = INT8_PROFILE) -> None:
    """Refuse a container whose packets reach a dispatch case its kernel was built without.

    The compile-time dispatch gate (kernels/aie2/conv_engine/design.py) compiles out the cases
    a container does not use, deriving the set from these same packets - so ignite-compile
    cannot produce a mismatch. A hand-edited container, or one paired with another build's
    xclbin, could. The core's failure mode there is `default:`: it emits nothing and raises
    nothing, so this is checked rather than left to read as an unexplained byte mismatch.

    Containers built before the gate carry no kernel_ops and are not checked.

    ``profile`` supplies the opcode NAME table, and it has to: the stride survives a change of
    engine by coincidence - W_BYTES is 9,472 and HDR_BYTES 128 in both, and H_OP is word 0 in
    both - but the names do not. Opcode 2 is MAXPOOL to the int8 emulator and RESIDUAL to the
    bf16 one, so reading a bf16 blob through the int8 table refuses a container for reaching a
    case it never reaches. Use ``profile_for_manifest`` when the container says which it is.
    """
    ops = ge.get("kernel_ops")
    if not ops:
        return
    emu = profile.emulator
    a = np.frombuffer(wpackets, dtype=np.uint8)
    n = a.size // emu.W_BYTES
    words = a[:n * emu.W_BYTES].reshape(n, emu.W_BYTES)[:, :emu.HDR_BYTES].copy().view(np.int32)
    present = {emu.OP_NAMES.get(int(o), f"?{o}") for o in np.unique(words[:, emu.H_OP])}
    missing = sorted(present - set(ops) - {"NOP"})
    if missing:
        raise ValueError(
            f"the packets reach {', '.join(missing)} but the kernel was built with only "
            f"{', '.join(ops)}. The core would fall through to `default:` and emit nothing. "
            f"Rebuild the container.")


def build_manifest(ir: GraphIR, ws: es.Workspace, scheds: List[es.LayerSchedule], store: es.PacketStore,
                   model_name: str, insts_bytes: int, xclbin_sha: str, kernel_sha: str, compile_s: float,
                   *, kernel_ops: Optional[List[str]] = None,
                   kernel_object_sha256: Optional[str] = None,
                   profile: EngineProfile = INT8_PROFILE) -> Dict[str, Any]:
    emu = profile.emulator
    placements = {}
    for name, p in ws.placements.items():
        placements[name] = {"base": p.base, "halo": p.halo, "halo_value": p.halo_value, "height": p.height,
                            "width": p.width, "blocks": p.blocks, "planes": p.planes, "band_rows": p.band_rows,
                            "channels": ir.tensors[name].channels, "dtype": p.dtype,
                            # Whether this tensor ever reaches the device. A host-stored tensor is planned a
                            # workspace slot like any other but is handed between host regions in memory, so
                            # the runtime has to be told which it is: writing one to the workspace instead
                            # leaves the region that wanted it reading an address nothing filled.
                            "storage": ir.tensors[name].storage,
                            "scale": ir.tensors[name].scale, "zero_point": ir.tensors[name].zero_point,
                            # Absent means integer, so only a non-integer engine writes the key
                            # and no container built before bf16 changes by a single byte. The
                            # storage "dtype" above stays a spelling numpy accepts, because
                            # np.dtype("bf16") raises and np.dtype("bfloat16") depends on
                            # whether ml_dtypes was imported first.
                            **({"elem": profile.elem} if profile.elem != ELEM_INT else {})}
    task = graph_task(ir)
    t_in = ir.tensors[ir.input]
    layers = [{"index": s.layer_index, "name": s.name, "output": ir.layers[s.layer_index].output,
               "rounds": s.rounds, "packets": s.packets, "w_fills": s.w_fills,
               **({"kind": "host"} if isinstance(ir.layers[s.layer_index], HostLayer) else {})} for s in scheds]
    graph_engine = {
        "kernel_sha256": kernel_sha,
        # kernel_sha256 hashes the SOURCE. The dispatch gate compiles cases out, so two
        # containers can share a source and not a kernel: the OBJECT hash is what identifies
        # what ran. kernel_ops is the opcode set THIS CONTAINER'S PACKETS CARRY - what the gate
        # was derived from, not a read-back of which cases the compiler kept. The two agree by
        # construction for the gateable cases; CONV, MAXPOOL and RESIDUAL are compiled in
        # whether or not a packet uses them, and OP_NOP is served by `default:`. A check
        # against kernel_ops is therefore conservative, which is the safe direction to err.
        "kernel_ops": kernel_ops,
        "kernel_object_sha256": kernel_object_sha256,
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
        # a_bytes is 6,400 at int8 and 12,800 at bf16; o_bytes is the same 3,200 B object either
        # way, carrying 32 channels at one byte or 16 at two.
        "tile": {"rows": emu.TILE_ROWS, "cols": emu.TILE_COLS, "a_bytes": emu.A_BYTES,
                 "w_bytes": emu.W_BYTES, "o_bytes": emu.O_BYTES},
    }
    tensor_placement_abi = {
        "abi_version": 1,
        "workspace_bytes": ws.nbytes,
        "input_tensor": ir.input,
        "input_placement": placements[ir.input],
        "output_tensors": [tensor for _, tensor in ir.outputs],
        "boundary_tensors": {
            ir.layers[i].output: placements[ir.layers[i].output]
            for i in range(len(ir.layers))
            if ir.layers[i].output in placements
        },
    }
    graph_engine["tensor_placement_abi"] = tensor_placement_abi
    manifest: Dict[str, Any] = {
        "model_name": model_name,
        "format_version": 1,
        "arch_id": ARCH_XDNA1_PHOENIX,
        "target_hardware": "AMD Phoenix APU (XDNA1, 16 AIE2 cores)",
        "producer": "ignite-compile v0.4.0 (graph engine)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": profile.name,
        "task": task,
        "input_shape": [1, t_in.channels, t_in.height, t_in.width],
        # The graph's own name for the input, which a host region feeds by name; it is not always the
        # workspace tensor's key.
        "input_name": getattr(t_in, "name", None) or ir.input,
        # A dense container takes the model's own input, which for both segmentation recipes is float32 at
        # scale 1.0 - the quantization is a QuantizeLinear inside the graph, not something the caller applies.
        "input_dtype": "int8" if task in ("detect", "pose")
        else (t_in.dtype or "uint8") if task in DENSE_TASKS else "uint8",
        "single_dispatch": True,
        "quant_scales": {"input_scale": t_in.scale, "input_zero_point": t_in.zero_point, "input_dtype": "uint8"},
        "tensor_placement_abi": tensor_placement_abi,
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
        # reg_max is DERIVED from the box head, not assumed. A DFL head carries 4 * reg_max
        # channels (YOLOv8: 64, so 16 bins per side); a detector that dropped DFL regresses the
        # four distances directly and lands on reg_max 1 (YOLO26). A decoder reads this to know
        # whether to do the softmax-weighted bin reduction at all, so hardcoding 16 silently
        # told every consumer to decode a 4-channel head as if it had bins.
        box_channels = heads["p3_box"]["channels"]
        if box_channels % 4:
            raise ValueError(f"box head has {box_channels} channels, not a multiple of 4")
        manifest.update({"strides": [8, 16, 32], "reg_max": box_channels // 4,
                         "num_classes": 80 if task == "detect" else 1,
                         "fused_dfl": False, "output_shapes": output_shapes, "head_layout": head_layout,
                         "egress_bytes": offset})
        if task == "pose":
            manifest["kpt_shape"] = list(POSE_KPT_SHAPE)
    elif task == "classify":
        onnx_name, tensor = ir.outputs[0]
        t = ir.tensors[tensor]
        manifest.update({
            "num_classes": t.channels,
            "output_shapes": {"logits": [1, t.channels]},
            "classification": {
                "tensor": tensor,
                "onnx_output": onnx_name,
                "channels": t.channels,
                "height": t.height,
                "width": t.width,
                "scale": t.scale,
                "zero_point": t.zero_point,
                "layout": "blocks_hw8",
                "pixel_index": [0, 0],
            },
            "egress_bytes": t.blocks * t.height * t.width * 8,
        })
    elif task in DENSE_TASKS:
        # Segmentation and matting read one map back through the same named-boundary metadata the host regions
        # use, so the runtime needs no second description of it: storage says whether it ever reached the
        # device, and dtype says whether the host still has to dequantize it.
        from ignite_xdna.compiler.dense_regions import boundary_metadata
        onnx_name, tensor = ir.outputs[0]
        t = ir.tensors[tensor]
        manifest.update({
            "output_shapes": {"map": [1, t.channels, t.height, t.width]},
            "dense_output": {**boundary_metadata(t), "onnx_output": onnx_name, "channels": t.channels,
                             "height": t.height, "width": t.width},
            "egress_bytes": t.channels * t.height * t.width * np.dtype(t.dtype or "uint8").itemsize,
        })
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


def check_weight_buffer_addresses(mlir_path: Path, expected: int, columns: int) -> None:
    """Refuse a build whose allocator did not put every weight buffer where the instruction stream points.

    The stream writes whole MemTile descriptors, whose word 1 is an absolute address, so it assumes where
    the buffer is. The buffer cannot be pinned - ``Buffer(address=...)`` breaks the bank-aware allocator on
    a buffer this large - so the assumption is checked against the addresses the build assigned. A wrong
    one does not fail loudly on the device: the descriptors point at other bytes and the dispatch hangs.
    """
    found = re.findall(r'aie\.buffer\([^)]*\)\s*\{address = (\d+) : i32,[^}]*sym_name = "(w\d+_buf)"',
                       Path(mlir_path).read_text(encoding="utf-8"))
    if len(found) != columns:
        raise RuntimeError(f"expected {columns} weight buffers in {mlir_path}, found {len(found)}")
    wrong = [f"{name} at {int(addr):#x}" for addr, name in found if int(addr) != expected]
    if wrong:
        raise RuntimeError(f"weight buffers allocated away from {expected:#x}, where the instruction stream "
                           f"points: {', '.join(wrong)}")


def compile_graph_container(onnx_path, output_path, build_dir: Optional[Path] = None, layers: Optional[int] = None,
                            verbose: bool = True, host_regions: Sequence[str] = (),
                            activation_ring: int = 0, weight_buffer: bool = False,
                            silu_sigmoid: bool = False, workspace_reuse: bool = True,
                            retire_batch: Optional[int] = None,
                            split_layers: Optional[Sequence[int]] = None,
                            decouple_weights: bool = False,
                            task: Optional[str] = None,
                            dense_recipe: Optional[str] = None,
                            engine: Any = "int8",
                            **kwargs) -> Dict[str, Any]:
    """Lower, schedule, build the device binaries and write the container. Returns the manifest.

    ``host_regions`` are node-name prefixes or ``FROM=TO`` boundaries run on the host between dispatches
    (``graph_ir.HostLayer``).

    ``silu_sigmoid`` computes every SiLU with the core's four-line sigmoid epilogue instead of Quark's HardSigmoid
    form (``silu_sigmoid.py``); the manifest records it as ``graph_engine.silu``.

    ``activation_ring`` (slots, 0 = off) builds the engine with a hand-written MemTile ring for activations
    instead of the split ObjectFifo, so one shim fetch can serve several output groups of a tile.

    ``workspace_reuse`` (on by default) lets ``plan_workspace`` hand a freed slot to the next tensor whose
    geometry matches, which is most of the workspace saving. Off, every tensor owns a slot, so a
    post-dispatch readback can see every layer -- the mode ``tools/verify_engine_container.py`` needs to
    check a lowering layer by layer. Recorded in the manifest when off.

    ``retire_batch`` is how many shim tasks the emitter lets pile up on one DMA channel before it retires a
    group (``engine_sequence.run_column_programs``). Bigger retires fewer completion tokens; it also moves
    the await onto a later, slower-to-complete task. None derives it from the schedule's thinnest channel,
    which is the default and the measured-best cadence on both a thin (SESR M7) and a wide (YOLOv8n) stream.
    The effective value is recorded in the manifest, so a container declares the schedule it was built with.
    """
    if host_regions and (activation_ring or weight_buffer):
        # ``split_instruction_stream`` cuts a host-segment stream by counting WRITE ops as task pushes, and
        # both MemTile structures emit configuration WRITEs that belong to no shim task, so the cut would land
        # on the wrong op. Both structures are measured slower than the default and are off by default, so
        # this refuses the combination rather than reconciling the accounting for a path worth nothing.
        raise ValueError("host_regions cannot be combined with activation_ring or weight_buffer: their "
                         "configuration writes break the instruction-stream split between segments")
    # Resolve the engine BEFORE importing IRON. A profile with no scheduler refuses here, and
    # there is no reason to pay for the toolchain to find that out.
    profile = engine if isinstance(engine, EngineProfile) else ENGINE_PROFILES[engine]
    emu = profile.emulator
    sched = profile.schedule
    if not profile.transport_options and (activation_ring or weight_buffer):
        raise ValueError(
            f"the {profile.name} design has no activation ring and no resident weight buffer: "
            f"both are int8-engine transport experiments, and both measured slower than the "
            f"schedule they were meant to beat")

    import aie.iron as iron
    from aie.iron.device import NPU1
    from aie.utils.compile.utils import compile_mlir_module
    from ignite_xdna.compiler.engine_sequence import SequenceEmitter

    eng = profile.load_design()

    t0 = time.perf_counter()
    onnx_path = Path(onnx_path)
    out_p = Path(output_path)
    build_dir = Path(build_dir or out_p.parent / "conv_engine" / out_p.stem).resolve()
    if dense_recipe:
        # A dense recipe finds its own host regions - every region the core cannot take becomes a named host
        # layer - so naming them by hand as well would cut the graph twice in different places.
        if host_regions:
            raise ValueError("dense_recipe picks its own host regions; --host-region cannot be combined with it")
        from ignite_xdna.compiler.dense_regions import lower_dense
        ir = lower_dense(onnx_path, dense_recipe, task=task)
    else:
        ir = lower_yolov8n(onnx_path, host_regions=host_regions, silu_sigmoid=silu_sigmoid)
        if task:
            ir.task = task
    ws = sched.plan_workspace(ir, reuse=workspace_reuse)
    scheds, store = sched.schedule_graph(ir, ws, activation_ring=activation_ring,
                                         weight_buffer=weight_buffer)
    if layers is not None:
        scheds = scheds[:layers]
    if verbose:
        n_host = sum(isinstance(L, HostLayer) for L in ir.layers)
        print(f"[*] Lowered {len(ir.layers)} layers ({n_host} on the host); workspace {ws.nbytes / 1e6:.1f} MB; "
              f"{sum(s.rounds for s in scheds)} rounds, {store.nbytes / 1e6:.2f} MB static packets")

    # yolov8s streams 29.5 MB of weight packets, past the 16 MB default packet extent.
    ws_extent, wp_extent = eng.ddr_extents(ws.nbytes, store.nbytes)

    cadence: List[Dict[str, int]] = []

    def body(ws_arg, wp_arg):
        emitter = SequenceEmitter(ws_arg, wp_arg, ws_extent, wp_extent,
                                  {c: eng.fifo_names(c) for c in range(eng.COLS)},
                                  a_ring=activation_ring)
        for s in scheds:
            emitter.run_column_programs(s.programs, bd_budget=14, retire_batch=retire_batch)
        cadence.append({"retire_batch": emitter.last_retire_batch,
                        "thinnest_channel_tasks": emitter.thinnest_channel_tasks})

    # Compile out the dispatch groups this container's packets never reach. The set is read
    # from the stored packets, so it cannot disagree with what the cores will be handed.
    # The table is checked against the emulator's opcode numbers here rather than in design.py,
    # because a drift between the two would gate out a case a container does reach and the core
    # would fall through to `default:` and silently emit nothing.
    # A design may have nothing to gate: the bf16 kernel implements CONV and RESIDUAL and no
    # more, so it exposes no GATEABLE_OPS and there is no drift to check.
    gateable = getattr(eng, "GATEABLE_OPS", None)
    if gateable is not None:
        assert gateable == {"FUSED_CONV": emu.OP_FUSED_CONV, "MUL": emu.OP_MUL,
                            "SCALE": emu.OP_SCALE, "POOL": emu.OP_POOL}, \
            "design.GATEABLE_OPS has drifted from the emulator's opcode numbers"
    kernel_ops = store.opcodes()
    iron.set_current_device(NPU1())
    build_kwargs = (dict(a_ring=activation_ring, w_buf=weight_buffer, ops=kernel_ops)
                    if profile.transport_options else {})
    program = eng.build_program(iron.get_current_device(), body, ws_bytes=ws_extent, wp_bytes=wp_extent,
                                **build_kwargs)
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
    if weight_buffer:
        check_weight_buffer_addresses(work / "input_with_addresses.mlir", eng.WBUF_ADDRESS, eng.COLS)
    xclbin = xclbin_path.read_bytes()
    insts = insts_path.read_bytes()
    kernel_sha = hashlib.sha256(eng.KERNEL_SOURCE.read_bytes()).hexdigest()
    # Not Optional in practice: a None here is an identity field that does not identify, and
    # it is the field that replaces kernel_sha256 now the gate has made that ambiguous. An
    # inline=True build would emit engine.ll instead, and that wants a deliberate change here
    # rather than a silent null in every container it produces.
    kernel_obj = work / profile.object_file
    if not kernel_obj.exists():
        raise RuntimeError(
            f"no linked kernel object at {kernel_obj}: the manifest's kernel_object_sha256 is "
            f"what says which kernel a container got, and the dispatch gate means the source "
            f"hash no longer does")
    kernel_obj_sha = hashlib.sha256(kernel_obj.read_bytes()).hexdigest()
    manifest = build_manifest(ir, ws, scheds, store, onnx_path.stem, len(insts),
                              hashlib.sha256(xclbin).hexdigest(), kernel_sha, time.perf_counter() - t0,
                              kernel_ops=sorted(emu.OP_NAMES[o] for o in kernel_ops),
                              kernel_object_sha256=kernel_obj_sha, profile=profile)
    manifest["graph_engine"]["ddr_extents_bytes"] = {"workspace": ws_extent, "packets": wp_extent}
    # Which ONNX this container was compiled from. The compile caches in this repo key on a NAME rather
    # than a model hash, so "is this container the model I think it is" has needed an external answer
    # more than once; recording it here lets a harness refuse a stale pairing instead of measuring it.
    manifest["source_model_sha256"] = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    if silu_sigmoid:   # absent means Quark's HardSigmoid form, as every earlier container
        manifest["graph_engine"]["silu"] = "sigmoid4"
    if not workspace_reuse:   # absent means reuse on: co-tenant tensors share a workspace slot
        manifest["graph_engine"]["workspace_reuse"] = False
    if cadence:
        # What the emitter retired on, per shim DMA channel: derived from the schedule's thinnest channel
        # unless --retire-batch forced it. Dispatch time is a property of this number (SESR M7: 4.31 ms at 2,
        # 5.25 ms at 4), so a container states the cadence it was built with. The minimum over the layers is
        # the binding one -- a layer whose channels are all deeper is not the stall.
        ge = manifest["graph_engine"]
        ge["retire_batch"] = min(c["retire_batch"] for c in cadence)
        ge["retire_batch_thinnest_channel"] = min(c["thinnest_channel_tasks"] for c in cadence)
        if retire_batch is not None:
            ge["retire_batch_forced"] = retire_batch
    segments = plan_segments(ir, scheds, split_layers=split_layers)
    hosts = [seg for seg in segments if seg["kind"] == "host"]
    blobs = [("engine.xclbin", xclbin, "xclbin")]
    if len(segments) > 1:
        # One stream per NPU segment: the emitter retires every task at each layer barrier, so the lowered
        # stream cuts cleanly at segment boundaries.
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
    if decouple_weights:
        weights_bytes = store.blob().tobytes()
        weights_sha = hashlib.sha256(weights_bytes).hexdigest()
        weights_file = out_p.with_suffix(".weights")
        weights_file.write_bytes(weights_bytes)
        manifest["graph_engine"]["decoupled_weights"] = True
        manifest["graph_engine"]["weights_file"] = weights_file.name
        manifest["graph_engine"]["weights_sha256"] = weights_sha
        manifest["graph_engine"]["weights_bytes"] = len(weights_bytes)
    else:
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
