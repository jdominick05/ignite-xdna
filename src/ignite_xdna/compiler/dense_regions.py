# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit BiSeNetV2/MODNet-Cut recipes at original Q/DQ boundaries.

Only Conv(+Relu, or a Clip the lowering proves is a ReLU) regions accepted by the
existing integer lowering are native.
All other operations retain their original ONNX definitions in named host regions.
This policy is deliberately limited to the two dense model families; it is not a
general ONNX fallback partitioner. No interpolation or network branch is rewritten.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Sequence

import numpy as np
import onnx

from .graph_ir import ConvLayer, GraphIR, HostLayer, Segment, TensorInfo, _Graph, lower_yolov8n

RECIPES = {"bisenetv2": "segment", "modnet_cut": "matte"}


def tensor_info(g: _Graph, name: str, producer: str = "") -> TensorInfo:
    shape = tuple(g.shape(name))
    if len(shape) != 4 or shape[0] != 1 or any(d <= 0 for d in shape):
        raise ValueError(f"dense boundary {name}: expected static NCHW batch one, got {shape}")
    vi = next(v for v in [*g.g.input, *g.g.output, *g.g.value_info] if v.name == name)
    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)).name
    if dtype not in ("uint8", "float32"):
        raise ValueError(f"dense boundary {name}: unsupported dtype {dtype}")
    src = g.by_output.get(name)
    scale, zp = g.scale_zp(src) if src is not None and src.op_type == "QuantizeLinear" else (1., 0)
    return TensorInfo(name, *shape[1:], scale, zp, producer, dtype, shape)


def boundary_metadata(t: TensorInfo, name: str | None = None) -> dict:
    return {"name": name or t.name, "tensor": t.name,
            "shape": list(t.shape or (1, t.channels, t.height, t.width)),
            "dtype": t.dtype, "layout": "NCHW", "storage": t.storage,
            "storage_layout": "NCHW" if t.storage == "host" else "blocks_hw8",
            "scale": t.scale if t.dtype == "uint8" else None,
            "zero_point": t.zero_point if t.dtype == "uint8" else None}


def lower_dense(model_or_path, recipe: str, task: str | None = None) -> GraphIR:
    if recipe not in RECIPES:
        raise ValueError(f"unknown dense recipe {recipe!r}")
    if task and task != RECIPES[recipe]:
        raise ValueError(f"{recipe} recipe requires task {RECIPES[recipe]}")
    model = onnx.load(str(model_or_path)) if not isinstance(model_or_path, onnx.ModelProto) else model_or_path
    model = onnx.shape_inference.infer_shapes(model)
    g = _Graph(model)
    if len(g.g.input) != 1 or len(g.g.output) != 1:
        raise ValueError("dense recipes require one image input and one spatial output")
    inp, out = g.g.input[0].name, g.g.output[0].name
    inp_info, out_info = tensor_info(g, inp, "input"), tensor_info(g, out)
    expected_channels = 19 if recipe == "bisenetv2" else 1
    if inp_info.channels != 3 or out_info.channels != expected_channels:
        raise ValueError(f"{recipe}: expected RGB input and {expected_channels} output channels")
    extractor = onnx.utils.Extractor(model)
    order = {o: i for i, n in enumerate(g.g.node) for o in n.output}
    boundaries = {n.output[0] for n in g.g.node if n.op_type == "QuantizeLinear"
                  and not g.is_constant(n.output[0])} | {inp, out}
    units, visited = [], {inp}

    def visit(target):
        if target in visited:
            return
        deps, seen = set(), set()

        def walk(name):
            if name != target and name in boundaries:
                deps.add(name)
                return
            if not name or g.is_constant(name) or name in seen:
                return
            seen.add(name)
            node = g.by_output.get(name)
            if node is None:
                raise ValueError(f"unresolved dense boundary {name}")
            for arg in node.input:
                walk(arg)

        walk(target)
        inputs = sorted(deps, key=lambda x: (order.get(x, -1), x))
        for dep in inputs:
            visit(dep)
        sub = extractor.extract_model(inputs, [target])
        ops = [n for n in sub.graph.node if n.op_type not in ("Constant", "QuantizeLinear", "DequantizeLinear")]
        native = None
        # Small feature maps and every non-convolution operator stay on the CPU.
        info = tensor_info(g, target)
        if (len(inputs) == 1 and info.dtype == "uint8" and min(info.height, info.width) >= 20
                and sum(n.op_type == "Conv" for n in ops) == 1
                and all(n.op_type in ("Conv", "Relu", "Clip") for n in ops)):
            dq = next((c for c in g.consumers.get(target, []) if c.op_type == "DequantizeLinear"), None)
            if dq is not None:
                try:
                    local = lower_yolov8n(extractor.extract_model(inputs, [dq.output[0]]))
                    if len(local.layers) == 1 and isinstance(local.layers[0], ConvLayer):
                        from . import engine_schedule as es
                        # Acceptance includes packet and geometry constraints, not just ONNX op names.
                        ws = es.plan_workspace(local)
                        es.schedule_graph(local, ws)
                        native = local.layers[0]
                except (ValueError, KeyError, NotImplementedError):
                    pass
        units.append((target, inputs, native))
        visited.add(target)

    visit(out)
    tensors = {inp: inp_info}
    layers = []
    pending = []

    def flush():
        if not pending:
            return
        produced = {u[0] for u in pending}
        inputs = list(dict.fromkeys(i for _, ins, _ in pending for i in ins if i not in produced))
        # Export only values consumed outside this region (or the graph output).
        needed = {i for target, ins, _ in units if target not in produced for i in ins} | {out}
        outputs = [u[0] for u in pending if u[0] in needed]
        sub = extractor.extract_model(inputs, outputs)
        name = f"{recipe}/host_{len(layers):03d}"
        host = HostLayer(name, len(layers), Segment(inputs[0], 0, tensors[inputs[0]].blocks),
                         outputs[0], sub.SerializeToString(), dict(Counter(n.op_type for n in sub.graph.node)),
                         named_inputs={n: n for n in inputs}, named_outputs={n: n for n in outputs})
        layers.append(host)
        for n in outputs:
            tensors[n] = tensor_info(g, n, name)
        pending.clear()

    for target, inputs, native in units:
        if native is None:
            pending.append((target, inputs, native))
        else:
            flush()
            native = replace(native, index=len(layers))
            layers.append(native)
            tensors[target] = tensor_info(g, target, native.name)
    flush()
    if not any(isinstance(layer, ConvLayer) for layer in layers):
        raise ValueError(f"{recipe}: no supported NPU convolution regions")
    device_tensors = {s.tensor for layer in layers if isinstance(layer, ConvLayer) for s in layer.inputs}
    device_tensors.update(layer.output for layer in layers if isinstance(layer, ConvLayer))
    for name, info in tensors.items():
        if name not in device_tensors:
            info.storage = "host"
    return GraphIR(tensors, layers, inp, [(out, out)], task=RECIPES[recipe], recipe=recipe)
