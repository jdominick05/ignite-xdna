"""Whole-tensor reference execution of a ``GraphIR`` and ONNX Runtime comparison.

``run_direct`` evaluates every layer on full uint8 tensors with exact integer
arithmetic (the same rules as the core program, without tiling), which is the
fast oracle the packet-level emulation is checked against. ``ort_intermediates``
runs the original QDQ model in ONNX Runtime and returns the same uint8 tensors
so the integer plan can be validated against the model's float semantics.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import numpy as np

from ignite_xdna.compiler import engine_emulator as em
from ignite_xdna.compiler.graph_ir import ConvLayer, GraphIR, HostLayer, PoolLayer, Segment, ZP


def quantize_input(image_chw_float: np.ndarray, scale: float, zp: int = ZP) -> np.ndarray:
    """uint8 image tensor exactly as the model's input QuantizeLinear produces it."""
    x = np.asarray(image_chw_float, dtype=np.float32)
    q = np.round(x.astype(np.float64) / scale).astype(np.int64) + zp
    return np.clip(q, 0, 255).astype(np.uint8)


def gather_input(tensors: Dict[str, np.ndarray], segments: List[Segment], height: int, width: int) -> np.ndarray:
    parts = []
    for s in segments:
        src = tensors[s.tensor]
        chans = src[s.block_offset * 8:(s.block_offset + s.blocks) * 8]
        if s.up2:
            chans = np.repeat(np.repeat(chans, 2, axis=1), 2, axis=2)
        if chans.shape[1:] != (height, width):
            raise ValueError(f"segment {s.tensor} is {chans.shape[1:]}, expected {(height, width)}")
        parts.append(chans)
    return np.concatenate(parts, axis=0)


def conv_direct(layer: ConvLayer, x: np.ndarray) -> np.ndarray:
    """uint8 [Cin][H][W] -> uint8 [Cout][Ho][Wo] with the engine's exact integer rules."""
    cin, h, w = x.shape
    k, s, p = layer.k, layer.stride, layer.pad
    xp = np.full((cin, h + 2 * p, w + 2 * p), ZP, dtype=np.uint8)
    xp[:, p:p + h, p:p + w] = x
    ho = (h + 2 * p - k) // s + 1
    wo = (w + 2 * p - k) // s + 1
    wts = layer.weights.astype(np.float64)   # exact: |acc| < 2^53
    cout = wts.shape[0]
    acc = np.zeros((cout, ho, wo), dtype=np.float64)
    for ky in range(k):
        for kx in range(k):
            patch = xp[:, ky:ky + s * ho:s, kx:kx + s * wo:s].astype(np.float64)   # [cin][ho][wo]
            acc += np.tensordot(wts[:, :, ky, kx], patch, axes=([1], [0]))
    acc = acc.astype(np.int64) + layer.bias_acc()[:, None, None]
    q = em.sat_u8(em.rne_shift(acc, layer.shift_out))
    if layer.hswish is not None:   # HardSwish, or ReLU expressed as the same epilogue
        q = layer.hswish.table[q]
    if layer.sigmoid is not None:  # SiLU through the sigmoid epilogue
        q = layer.sigmoid.table[q]
    return q


def maxpool5_direct(x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    xp = np.zeros((c, h + 4, w + 4), dtype=np.uint8)
    xp[:, 2:2 + h, 2:2 + w] = x
    out = np.zeros_like(x)
    for dy in range(5):
        for dx in range(5):
            out = np.maximum(out, xp[:, dy:dy + h, dx:dx + w])
    return out


_HOST_SESSIONS: Dict[tuple, Any] = {}


def host_session(onnx_bytes: bytes, optimize: bool = False):
    """ONNX Runtime CPU session for a host layer's extracted model, cached by content.

    ``optimize`` False disables graph optimizations, as ``ort_intermediates`` does.
    """
    import onnxruntime as ort
    key = (hashlib.sha256(onnx_bytes).hexdigest(), bool(optimize))
    sess = _HOST_SESSIONS.get(key)
    if sess is None:
        so = ort.SessionOptions()
        if not optimize:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(onnx_bytes, so, providers=["CPUExecutionProvider"])
        _HOST_SESSIONS[key] = sess
    return sess


def run_host_layer(layer: HostLayer, x: np.ndarray, optimize: bool = False) -> np.ndarray:
    """uint8 [C][H][W] -> uint8 [C'][H'][W'] through the host layer's extracted model."""
    sess = host_session(layer.onnx_bytes, optimize)
    y = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(x[None], dtype=np.uint8)})[0]
    return np.asarray(y, dtype=np.uint8)[0]


def run_direct(ir: GraphIR, input_q: np.ndarray, stop_after: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Evaluate the graph on uint8 tensors; returns {tensor name: uint8 [C][H][W]}."""
    tensors: Dict[str, np.ndarray] = {ir.input: input_q}
    for L in ir.layers:
        if stop_after is not None and L.index > stop_after:
            break
        t = ir.tensors[L.output]
        if isinstance(L, ConvLayer):
            src = ir.tensors[L.inputs[0].tensor]
            in_h = src.height * (2 if L.inputs[0].up2 else 1)
            in_w = src.width * (2 if L.inputs[0].up2 else 1)
            x = gather_input(tensors, L.inputs, in_h, in_w)
            y = conv_direct(L, x)
            if L.residual is not None:
                r = gather_input(tensors, [L.residual], t.height, t.width)
                y = em.residual_combine(y, r, L.residual_shift, L.residual_lsh_main, L.residual_lsh_res)
                if L.post_hswish is not None:
                    y = em.hswish_epilogue(y, L.post_hswish.params)
            tensors[L.output] = y
        elif isinstance(L, HostLayer):
            segs = L.input_segments()
            src = ir.tensors[segs[0].tensor]
            x = gather_input(tensors, segs, src.height, src.width)[:L.in_channels or src.channels]
            tensors[L.output] = run_host_layer(L, x)
        else:
            x = gather_input(tensors, [L.input], t.height, t.width)
            tensors[L.output] = maxpool5_direct(x)
    return tensors


def ort_intermediates(model_path, image_nchw_float: np.ndarray, names: List[str]) -> Dict[str, np.ndarray]:
    """Run the QDQ model (a path or a ModelProto, e.g. ``silu_sigmoid.reference_model``) in ONNX Runtime exposing the
    requested uint8 tensors as outputs."""
    import copy

    import onnx
    import onnxruntime as ort
    model = copy.deepcopy(model_path) if isinstance(model_path, onnx.ModelProto) else onnx.load(str(model_path))
    existing = {o.name for o in model.graph.output}
    for n in names:
        if n not in existing:
            model.graph.output.append(onnx.helper.make_tensor_value_info(n, onnx.TensorProto.UINT8, None))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    outs = sess.run(names, {sess.get_inputs()[0].name: image_nchw_float.astype(np.float32)})
    return {n: np.asarray(o)[0] for n, o in zip(names, outs)}


def ort_fp32_heads(model_path, image_nchw_float: np.ndarray) -> Dict[str, np.ndarray]:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, {sess.get_inputs()[0].name: image_nchw_float.astype(np.float32)})
    return {n: np.asarray(o)[0] for n, o in zip(names, outs)}
