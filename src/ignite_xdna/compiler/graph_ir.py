"""YOLOv8n QDQ ONNX -> engine layer IR.

The Quark-quantized model stores every activation as uint8 with zero point
128 and a power-of-two scale, weights as per-tensor int8, and each SiLU as a
HardSigmoid chain that is a pure function of one uint8. This module walks the
graph once and produces:

* ``TensorInfo`` for every physical uint8 tensor the engine stores in the
  workspace (the image, every activated conv output, every residual sum, every
  SPPF pool), and
* ``ConvLayer`` / ``PoolLayer`` records in execution order whose inputs are
  ``Segment`` views (tensor, channel block range, optional 2x upsampling), so
  Split, Concat and Resize never materialise, and
* ``HostLayer`` records for regions named by ``host_regions`` (YOLO11's C2PSA
  attention block, or only its attention core): the region is extracted as a
  uint8 -> uint8 ONNX model and run on the host between two dispatches.

Exactness: the integer HardSwish constants are fitted against a float32
re-evaluation of the ONNX chain for all 256 inputs; ``HardSwishFit.max_error``
records the largest remaining deviation in output LSBs (0 when exact).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from ignite_xdna.compiler.engine_emulator import HardSwishParams, hswish_epilogue, rne_shift, sat_u8

ZP = 128
HS_ALPHA = np.float32(0.1666666716337204)
HS_BETA = np.float32(0.5)


def _log2_exact(scale: float) -> int:
    e = math.log2(scale)
    if abs(e - round(e)) > 1e-9:
        raise ValueError(f"scale {scale} is not a power of two")
    return int(round(e))


@dataclass
class TensorInfo:
    name: str                 # uint8 (QuantizeLinear output) tensor name
    channels: int
    height: int
    width: int
    scale: float
    zero_point: int
    producer: str = ""       # layer name or "input"

    @property
    def blocks(self) -> int:
        return (self.channels + 7) // 8


@dataclass
class Segment:
    tensor: str
    block_offset: int
    blocks: int
    up2: bool = False


@dataclass
class HardSwishFit:
    params: HardSwishParams
    table: np.ndarray        # exact float-chain table, uint8[256]
    max_error: int


@dataclass
class ConvLayer:
    name: str
    index: int
    inputs: List[Segment]
    in_scale: float
    k: int
    stride: int
    pad: int
    weights: np.ndarray      # int8 [Cout][Cin][k][k]
    bias_q: np.ndarray       # int8 [Cout]
    bias_scale: float
    weight_scale: float
    conv_scale: float        # scale of the conv output QuantizeLinear (s1)
    output: str              # physical output tensor name
    act: Optional[str] = None            # "hswish" or None
    hswish: Optional[HardSwishFit] = None
    residual: Optional[Segment] = None   # added after the activation
    residual_shift: int = 0              # log2(out_scale / finer operand scale): the rounding shift
    residual_lsh_main: int = 0           # log2(act_scale / finer operand scale)
    residual_lsh_res: Optional[int] = None  # log2(residual scale / finer operand scale); None: residual_shift
    act_scale: Optional[float] = None    # scale after the activation (s2)

    @property
    def cin(self) -> int:
        return int(self.weights.shape[1])

    @property
    def cout(self) -> int:
        return int(self.weights.shape[0])

    @property
    def in_blocks(self) -> int:
        return sum(s.blocks for s in self.inputs)

    @property
    def shift_out(self) -> int:
        return -_log2_exact(self.in_scale) - _log2_exact(self.weight_scale) + _log2_exact(self.conv_scale)

    def bias_acc(self) -> np.ndarray:
        """int32 accumulator bias: quantized bias rescaled, minus 128 * sum(w), plus 128 << shift."""
        ratio = self.bias_scale / (self.in_scale * self.weight_scale)
        e = _log2_exact(ratio)
        if e < 0:
            raise ValueError(f"{self.name}: bias scale ratio {ratio} is fractional")
        b = self.bias_q.astype(np.int64) << e
        wsum = self.weights.astype(np.int64).sum(axis=(1, 2, 3))
        return (b - ZP * wsum + (ZP << self.shift_out)).astype(np.int64)


@dataclass
class PoolLayer:
    name: str
    index: int
    input: Segment
    output: str
    k: int = 5
    pad: int = 2


@dataclass
class HostLayer:
    """A graph region the engine does not lower, run on the host between two dispatches.

    ``onnx_bytes`` is the region extracted by ``onnx.utils.Extractor`` between its input and output
    QuantizeLinear tensors, so it maps the uint8 input tensor to the uint8 output tensor exactly as the
    original graph does. The input is one whole physical tensor; the output is a physical tensor like
    a conv output.
    """
    name: str                 # the region spec: a node-name prefix ("/model.10/") or "FROM=TO"
    index: int
    input: Segment
    output: str
    onnx_bytes: bytes
    op_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class GraphIR:
    tensors: Dict[str, TensorInfo]
    layers: List[object]          # ConvLayer | PoolLayer | HostLayer in execution order
    input: str
    outputs: List[Tuple[str, str]]  # (onnx float output name, physical tensor name)
    adjacency: List[List[str]] = field(default_factory=list)  # tensor groups that must be contiguous
    # onnx output name -> host-side transform of the read-back tensor, e.g. {"op": "depth_to_space", ...}
    output_transforms: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# HardSwish table + integer fit
# ----------------------------------------------------------------------------

def hardswish_float_table(s1: float, s2: float, k: float = 1.0001220703125, hs_scale: float = 1 / 128) -> np.ndarray:
    """Evaluate the ONNX chain Q(s1)->DQ->HardSigmoid->Mul(k)->Q(1/128)->DQ->Mul(x)->Q(s2) in float32."""
    q1 = np.arange(256, dtype=np.int64)
    x = ((q1 - ZP).astype(np.float32) * np.float32(s1)).astype(np.float32)
    h = np.maximum(np.float32(0), np.minimum(np.float32(1), (HS_ALPHA * x).astype(np.float32) + HS_BETA)).astype(np.float32)
    h2 = (h * np.float32(k)).astype(np.float32)
    qh = sat_u8(np.round(h2.astype(np.float64) / hs_scale).astype(np.int64) + ZP)  # np.round = half to even
    h3 = ((qh.astype(np.int64) - ZP).astype(np.float32) * np.float32(hs_scale)).astype(np.float32)
    y = (x * h3).astype(np.float32)
    q2 = sat_u8(np.round(y.astype(np.float64) / s2).astype(np.int64) + ZP)
    return q2


def fit_hardswish(s1: float, s2: float, k: float = 1.0001220703125) -> HardSwishFit:
    """Find integer epilogue constants reproducing ``hardswish_float_table`` exactly when possible."""
    table = hardswish_float_table(s1, s2, k)
    q1 = np.arange(256, dtype=np.int64).astype(np.uint8)
    e1, e2 = -_log2_exact(s1), -_log2_exact(s2)
    ysh = 7 + e1 - e2
    if ysh < 0:
        raise ValueError(f"HardSwish output scale {s2} finer than the engine supports for input scale {s1}")
    best = None
    for q in range(7, 15):
        for sh1 in range(8, 21):
            a1 = int(round(float(HS_ALPHA) * s1 * (1 << (sh1 + q))))
            if a1 >= 1 << 15:
                continue
            b1 = 1 << (sh1 + q - 1)
            for sh2 in range(q, q + 8):
                k2 = int(round(k * 128 * (1 << (sh2 - q))))
                if k2 >= 1 << 15:
                    continue
                params = HardSwishParams(a1=a1, b1=b1, s1=sh1, qmax=1 << q, k2=k2, s2=sh2, ysh=ysh)
                got = hswish_epilogue(q1, params)
                err = int(np.max(np.abs(got.astype(np.int64) - table.astype(np.int64))))
                if best is None or err < best.max_error:
                    best = HardSwishFit(params, table, err)
                if err == 0:
                    return best
    return best


def relu_epilogue() -> HardSwishFit:
    """ReLU at a zero-point-128 quantization, max(q, 128), as constants of the HardSwish epilogue.

    t = q - 128; hs = clip(t * 64, 0, 64) is 64 for t >= 1 and 0 otherwise; y = (t * hs) >> 6 = t or 0.
    Exact for all 256 inputs, so ReLU layers need no second activation path in the core.
    """
    params = HardSwishParams(a1=64, b1=0, s1=0, qmax=64, k2=1, s2=0, ysh=6)
    table = np.maximum(np.arange(256, dtype=np.int64), ZP).astype(np.uint8)
    got = hswish_epilogue(np.arange(256, dtype=np.int64).astype(np.uint8), params)
    return HardSwishFit(params, table, int(np.max(np.abs(got.astype(np.int64) - table.astype(np.int64)))))


# ----------------------------------------------------------------------------
# ONNX walk
# ----------------------------------------------------------------------------

class _Graph:
    def __init__(self, model: onnx.ModelProto):
        self.g = model.graph
        self.inits = {t.name: numpy_helper.to_array(t) for t in self.g.initializer}
        for n in self.g.node:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        self.inits[n.output[0]] = numpy_helper.to_array(a.t)
        self.by_output = {o: n for n in self.g.node for o in n.output}
        self._const_memo: Dict[str, bool] = {}
        self.consumers: Dict[str, List[onnx.NodeProto]] = {}
        for n in self.g.node:
            for i in n.input:
                self.consumers.setdefault(i, []).append(n)
        self.value_shapes = {}
        for vi in list(self.g.value_info) + list(self.g.input) + list(self.g.output):
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            self.value_shapes[vi.name] = dims

    def const(self, name):
        return self.inits.get(name)

    def is_constant(self, name: str, _memo: Optional[Dict[str, bool]] = None) -> bool:
        """True for an initializer or Constant output, or a tensor every input of whose producer is constant.

        YOLO-World's text guide is such a tensor: built once from Constant nodes inside ``/model.12/attn/`` and read
        by the other three attention blocks, so it belongs to no region's activation boundary."""
        memo = self._const_memo if _memo is None else _memo
        if name in memo:
            return memo[name]
        if name in self.inits:
            memo[name] = True
            return True
        node = self.by_output.get(name)
        memo[name] = False  # a cycle cannot make a tensor constant
        result = node is not None and all(not i or self.is_constant(i, memo) for i in node.input)
        memo[name] = result
        return result

    def scale_zp(self, qdq_node) -> Tuple[float, int]:
        s = float(self.inits[qdq_node.input[1]].flatten()[0])
        z = int(self.inits[qdq_node.input[2]].flatten()[0]) if len(qdq_node.input) > 2 else 0
        return s, z

    def q_source(self, float_name: str):
        """For a float tensor produced by DequantizeLinear, return (uint8 name, scale, zp)."""
        n = self.by_output.get(float_name)
        if n is None or n.op_type != "DequantizeLinear":
            raise ValueError(f"{float_name} is not the output of a DequantizeLinear")
        s, z = self.scale_zp(n)
        return n.input[0], s, z

    def q_sink(self, float_name: str):
        """The QuantizeLinear consuming a float tensor: (uint8 name, scale, zp)."""
        qs = [c for c in self.consumers.get(float_name, []) if c.op_type == "QuantizeLinear"]
        if len(qs) != 1:
            raise ValueError(f"{float_name} has {len(qs)} QuantizeLinear consumers")
        s, z = self.scale_zp(qs[0])
        return qs[0].output[0], s, z

    def float_consumers(self, float_name: str):
        return [c for c in self.consumers.get(float_name, []) if c.op_type != "QuantizeLinear"]

    def dq_of(self, q_name: str) -> str:
        dqs = [c for c in self.consumers.get(q_name, []) if c.op_type == "DequantizeLinear"]
        if len(dqs) != 1:
            raise ValueError(f"{q_name} has {len(dqs)} DequantizeLinear consumers")
        return dqs[0].output[0]

    def shape(self, name: str):
        if name in self.value_shapes:
            return self.value_shapes[name]
        raise KeyError(name)


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def _boundary_tensor(G: _Graph, spec: str, token: str) -> str:
    """The uint8 tensor one side of a ``FROM=TO`` host region names: a QuantizeLinear output tensor, or the tensor a
    node's output is quantized to."""
    src = G.by_output.get(token)
    if src is not None and src.op_type == "QuantizeLinear":
        return token
    node = next((n for n in G.g.node if n.name == token), None)
    if node is None:
        raise ValueError(f"host region {spec}: {token} is neither a uint8 tensor nor a node name")
    if node.op_type == "QuantizeLinear":
        return node.output[0]
    return G.q_sink(node.output[0])[0]


def _boundary_region(G: _Graph, spec: str) -> Dict[str, Any]:
    """The nodes on a path from ``FROM``'s uint8 tensor to ``TO``'s (see ``_host_region``)."""
    first, _, last = spec.partition("=")
    q_in, q_out = _boundary_tensor(G, spec, first.strip()), _boundary_tensor(G, spec, last.strip())
    below = set()                       # nodes computed from q_in
    stack = [q_in]
    while stack:
        for c in G.consumers.get(stack.pop(), []):
            if c.name not in below:
                below.add(c.name)
                stack.extend(c.output)
    above = set()                       # nodes q_out is computed from
    stack = [q_out]
    while stack:
        n = G.by_output.get(stack.pop())
        if n is not None and n.name not in above:
            above.add(n.name)
            stack.extend(i for i in n.input if i)
    names = below & above
    if not names:
        raise ValueError(f"host region {spec}: no node lies between {q_in} and {q_out}")
    nodes = [n for n in G.g.node if n.name in names]
    produced = {o for n in nodes for o in n.output}
    for n in nodes:
        for i in n.input:
            if not i or i == q_in or i in produced or i in G.inits:
                continue
            src = G.by_output.get(i)
            if src is not None and src.op_type == "DequantizeLinear" and src.input[0] in G.inits:
                continue  # a weight or scale constant
            raise ValueError(f"host region {spec}: {n.name} reads {i} from outside the region")
    graph_outputs = {o.name for o in G.g.output}
    for o in sorted(produced - {q_out}):
        if o in graph_outputs:
            raise ValueError(f"host region {spec}: {o} inside the region is a graph output")
        for c in G.consumers.get(o, []):
            if c.name in names:
                continue
            # A channel view may read a dequantized tensor of the region directly, or through its own
            # DequantizeLinear; the lowering resolves the Reshape or refuses it.
            view = c.op_type == "Reshape" and G.by_output[o].op_type == "DequantizeLinear"
            view = view or (c.op_type == "DequantizeLinear" and
                            all(f.op_type == "Reshape" for f in G.consumers.get(c.output[0], [])))
            if not view:
                raise ValueError(f"host region {spec}: {c.op_type} {c.name} outside the region reads {o}")
    dqs = [c for c in G.consumers.get(q_out, []) if c.op_type == "DequantizeLinear"]
    if not dqs:
        raise ValueError(f"host region {spec}: nothing dequantizes its output {q_out}")
    order = {n.name: i for i, n in enumerate(G.g.node)}
    return {"spec": spec, "names": names, "q_in": q_in, "q_out": q_out, "dq_out": dqs[0].output[0],
            "q_node": G.by_output[q_out], "builder": min(names, key=order.__getitem__)}


def _host_region(G: _Graph, spec: str) -> Dict[str, Any]:
    """The nodes a host region names, with exactly one uint8 activation input and one uint8 output.

    ``spec`` is a node-name prefix (``"/model.10/"``: every node named ``prefix*``) or ``"FROM=TO"``: the nodes on a
    path from the uint8 output of ``FROM`` to the uint8 output of ``TO``, each a node name or a QuantizeLinear output
    tensor (YOLO11's attention core, from its qkv convolution to the reshape after the second MatMul).

    Prefix regions: inputs read from initializers, Constant nodes or DequantizeLinear of an initializer (weights) are
    part of the region's constants. The output is the float tensor of a DequantizeLinear inside the region that nodes
    outside it consume; its QuantizeLinear input is the uint8 output tensor. The layer is built at that
    DequantizeLinear.

    Boundary regions: a tensor inside the region may also be read outside it through DequantizeLinear -> Reshape,
    which the lowering resolves as a channel view of a stored tensor (YOLO11's ``v``, read by the ``pe``
    convolution). The layer is built at the region's first node, so engine layers that combine its output with
    such a view come after it.
    """
    if "=" in spec:
        return _boundary_region(G, spec)
    prefix = spec
    nodes = [n for n in G.g.node if n.name.startswith(prefix)]
    if not nodes:
        raise ValueError(f"host region {prefix}: no node names start with it")
    names = {n.name for n in nodes}
    produced = {o for n in nodes for o in n.output}
    graph_outputs = {o.name for o in G.g.output}
    acts = set()
    for n in nodes:
        for i in n.input:
            if not i or i in produced or i in G.inits:
                continue
            src = G.by_output.get(i)
            if src is None:
                raise ValueError(f"host region {prefix}: {n.name} reads graph input {i}")
            if src.op_type == "DequantizeLinear" and src.input[0] in G.inits:
                continue  # a weight or bias constant
            if G.is_constant(i):
                continue  # derived from constants only, possibly by another region's nodes; extraction copies them
            if src.op_type != "DequantizeLinear":
                raise ValueError(f"host region {prefix}: {n.name} reads {i} from {src.op_type} {src.name} outside it")
            acts.add(src.input[0])
    outs = sorted({o for n in nodes for o in n.output
                   if o in graph_outputs or (not G.is_constant(o)
                                             and any(c.name not in names for c in G.consumers.get(o, [])))})
    if len(acts) != 1 or len(outs) != 1:
        raise ValueError(f"host region {prefix}: needs one activation input and one output, "
                         f"found inputs {sorted(acts)} and outputs {outs}")
    dq_out = outs[0]
    dq = G.by_output[dq_out]
    if dq_out in graph_outputs or dq.op_type != "DequantizeLinear":
        raise ValueError(f"host region {prefix}: output {dq_out} is not a DequantizeLinear inside the graph")
    q_node = G.by_output.get(dq.input[0])
    if q_node is None or q_node.op_type != "QuantizeLinear" or q_node.name not in names:
        raise ValueError(f"host region {prefix}: {dq_out} does not dequantize a QuantizeLinear of the region")
    return {"spec": prefix, "names": names, "q_in": next(iter(acts)), "q_out": dq.input[0], "dq_out": dq_out,
            "q_node": q_node, "builder": dq.name}


def lower_yolov8n(model_or_path, host_regions: Sequence[str] = ()) -> GraphIR:
    """Lower a QDQ graph to engine layers. ``host_regions`` name regions the engine does not lower, as node-name
    prefixes (``"/model.10/"``) or ``"FROM=TO"`` boundaries (see ``_host_region``); each becomes one ``HostLayer``."""
    model = onnx.load(str(model_or_path)) if not isinstance(model_or_path, onnx.ModelProto) else model_or_path
    model = onnx.shape_inference.infer_shapes(model)
    G = _Graph(model)
    regions = [_host_region(G, prefix) for prefix in host_regions]
    region_of = {name: r for r in regions for name in r["names"]}
    if len(region_of) != sum(len(r["names"]) for r in regions):
        raise ValueError(f"host regions {list(host_regions)} overlap")
    tensors: Dict[str, TensorInfo] = {}
    views: Dict[str, List[Segment]] = {}   # virtual uint8 tensors -> segments
    layers: List[object] = []
    adjacency: List[List[str]] = []

    def resolve(q_name: str) -> List[Segment]:
        if q_name in tensors:
            return [Segment(q_name, 0, tensors[q_name].blocks)]
        if q_name in views:
            return list(views[q_name])
        raise KeyError(f"unknown uint8 tensor {q_name}")

    def dims_chw(float_name: str) -> Tuple[int, int, int]:
        d = G.shape(float_name)
        if len(d) != 4 or d[0] != 1:
            raise ValueError(f"{float_name}: unexpected shape {d}")
        return int(d[1]), int(d[2]), int(d[3])

    # Graph input: images (float) -> QuantizeLinear -> DequantizeLinear
    inp = G.g.input[0].name
    q_in, s_in, z_in = G.q_sink(inp)
    c, h, w = dims_chw(inp)
    tensors[q_in] = TensorInfo(q_in, c, h, w, s_in, z_in, producer="input")

    def add_view(q_name, segments):
        if q_name in views or q_name in tensors:
            raise ValueError(f"duplicate tensor {q_name}")
        views[q_name] = segments

    def channel_view(node, so: float, zo: int) -> List[Segment]:
        """Segments of a Reshape that only moves whole channels: its Reshape/Slice chain back to a stored tensor, at
        one quantization, ends [1, 8k, H, W] with every output channel one input channel in pixel order."""
        chain, f = [node], node.input[0]
        while True:
            q, s, z = G.q_source(f)
            if abs(s - so) > 1e-12 or z != zo:
                raise ValueError(f"{node.name}: channel view requantizes {q} ({s}, {z}) -> ({so}, {zo})")
            if q in tensors or q in views:
                break
            qn = G.by_output.get(q)
            src = G.by_output.get(qn.input[0]) if qn is not None and qn.op_type == "QuantizeLinear" else None
            if src is None or src.op_type not in ("Reshape", "Slice"):
                raise ValueError(f"{node.name}: {q} is not a reshape or slice of a stored tensor")
            chain.append(src)
            f = src.input[0]
        base = resolve(q)
        c, h, w = dims_chw(f)
        # Label every element of the stored tensor with channel * H * W + pixel and replay the chain.
        lab = (np.arange(c, dtype=np.int64)[:, None] * (h * w) + np.arange(h * w)[None, :]).reshape(1, c, h, w)
        for op in reversed(chain):
            if op.op_type == "Reshape":
                if int(_attr(op, "allowzero", 0)):
                    raise ValueError(f"{op.name}: allowzero reshape")
                shape = [int(v) for v in G.const(op.input[1]).flatten()]
                lab = lab.reshape([lab.shape[i] if d == 0 else d for i, d in enumerate(shape)])
            else:
                starts, ends = G.const(op.input[1]).flatten(), G.const(op.input[2]).flatten()
                axes = G.const(op.input[3]).flatten() if len(op.input) > 3 and op.input[3] else range(len(starts))
                steps = G.const(op.input[4]).flatten() if len(op.input) > 4 and op.input[4] else [1] * len(starts)
                index = [slice(None)] * lab.ndim
                for a, st, en, sp in zip(axes, starts, ends, steps):
                    index[int(a) % lab.ndim] = slice(int(st), int(en), int(sp))
                lab = lab[tuple(index)]
        if lab.ndim != 4 or lab.shape[0] != 1 or tuple(lab.shape[2:]) != (h, w) or lab.shape[1] % 8:
            raise ValueError(f"{node.name}: view shape {list(lab.shape)} is not [1, 8k, {h}, {w}]")
        src_ch = lab[0, :, 0, 0] // (h * w)
        if not np.array_equal(lab[0], src_ch[:, None, None] * (h * w) + np.arange(h * w).reshape(h, w)):
            raise ValueError(f"{node.name}: moves pixels, not whole channels")
        base_blocks = [(sg.tensor, sg.block_offset + b, sg.up2) for sg in base for b in range(sg.blocks)]
        segs: List[Segment] = []
        for k in range(0, len(src_ch), 8):
            ch = src_ch[k:k + 8]
            if ch[0] % 8 or not np.array_equal(ch, np.arange(ch[0], ch[0] + 8)):
                raise ValueError(f"{node.name}: channels {ch.tolist()} are not one 8-channel block")
            t_name, blk, up2 = base_blocks[int(ch[0]) // 8]
            if segs and segs[-1].tensor == t_name and segs[-1].up2 == up2 and \
                    segs[-1].block_offset + segs[-1].blocks == blk:
                segs[-1] = Segment(t_name, segs[-1].block_offset, segs[-1].blocks + 1, up2)
            else:
                segs.append(Segment(t_name, blk, 1, up2))
        return segs

    def check_same_scale(q_a, q_b, what):
        sa = tensors[q_a].scale if q_a in tensors else None
        if sa is None:
            return
        if abs(sa - q_b) > 1e-12:
            raise ValueError(f"{what}: scale {sa} != {q_b}")

    scales: Dict[str, float] = {q_in: s_in}
    transforms: Dict[str, Tuple[str, Dict[str, Any]]] = {}  # host-side output transforms (DepthToSpace)

    for node in G.g.node:
        region = region_of.get(node.name)
        if region is not None:
            # The region's nodes are not lowered. Its layer is built at the DequantizeLinear that exposes the
            # output: the input precedes that node and every consumer of the output follows it.
            if node.name == region["builder"]:
                # Own names: q_in, c, h, w and friends belong to the enclosing walk (q_in is the graph input).
                r_in, r_out, r_qnode = region["q_in"], region["q_out"], region["q_node"]
                if r_in not in tensors:
                    raise ValueError(f"host region {region['spec']}: input {r_in} is not a physical tensor")
                r_scale, r_zp = G.scale_zp(r_qnode)
                if r_zp != ZP:
                    raise ValueError(f"host region {region['spec']}: output zero point {r_zp}")
                _log2_exact(r_scale)
                r_c, r_h, r_w = dims_chw(region["dq_out"])
                sub = onnx.utils.Extractor(model).extract_model([r_in], [r_out])
                sub_ins, sub_outs = list(sub.graph.input), list(sub.graph.output)
                if [i.name for i in sub_ins] != [r_in] or [o.name for o in sub_outs] != [r_out] or \
                        sub_ins[0].type.tensor_type.elem_type != onnx.TensorProto.UINT8 or \
                        sub_outs[0].type.tensor_type.elem_type != onnx.TensorProto.UINT8:
                    raise ValueError(f"host region {region['spec']}: extracted model is not uint8 {r_in} -> {r_out}")
                host = HostLayer(name=region["spec"], index=len(layers), input=Segment(r_in, 0, tensors[r_in].blocks),
                                 output=r_out, onnx_bytes=sub.SerializeToString(),
                                 op_types=dict(Counter(n.op_type for n in sub.graph.node)))
                tensors[r_out] = TensorInfo(r_out, r_c, r_h, r_w, r_scale, ZP, producer=host.name)
                scales[r_out] = r_scale
                layers.append(host)
            continue
        if node.op_type in ("Constant", "QuantizeLinear", "DequantizeLinear", "HardSigmoid", "Mul", "Relu"):
            continue
        if node.op_type == "Conv":
            x_q, sx, zx = G.q_source(node.input[0])
            w_q, sw, zw = G.q_source(node.input[1])
            y_f = node.output[0]
            cout, oh, ow = dims_chw(y_f)
            if len(node.input) > 2 and node.input[2]:
                b_q, sb, zb = G.q_source(node.input[2])
                bias = G.const(b_q)
            else:  # bias-free conv (SESR): a zero bias at the product scale
                sb, zb, bias = sx * sw, 0, np.zeros(cout, dtype=np.int8)
            if zx != ZP or zw != 0 or zb != 0:
                raise ValueError(f"{node.name}: unsupported zero points x={zx} w={zw} b={zb}")
            weights = G.const(w_q)
            kernel = [int(v) for v in _attr(node, "kernel_shape")]
            strides = [int(v) for v in _attr(node, "strides", [1, 1])]
            if len(set(kernel)) != 1 or len(set(strides)) != 1:
                raise ValueError(f"{node.name}: non-square kernel {kernel} or strides {strides}")
            k, stride = kernel[0], strides[0]
            pads = _attr(node, "pads", [0, 0, 0, 0])
            if any(int(p) != int(pads[0]) for p in pads):
                raise ValueError(f"{node.name}: asymmetric pads {pads}")
            dilations = [int(v) for v in _attr(node, "dilations", [1, 1])]
            if any(d != 1 for d in dilations):
                raise ValueError(f"{node.name}: dilations {dilations} are not supported")
            group = int(_attr(node, "group", 1))
            c_in = dims_chw(node.input[0])[0]
            if group != 1:
                if not (group == c_in == cout and weights.shape[1] == 1):
                    raise ValueError(f"{node.name}: group {group} convolution ({c_in} -> {cout} channels) is not "
                                     f"supported; only depthwise (group == Cin == Cout)")
                # Depthwise as the dense convolution whose off-diagonal taps are zero: every
                # off-diagonal product is 0 in the integer accumulator, so the output is exact and
                # the per-channel weight sum in ConvLayer.bias_acc is unchanged.
                dense = np.zeros((cout, cout) + tuple(weights.shape[2:]), dtype=weights.dtype)
                dense[np.arange(cout), np.arange(cout)] = weights[:, 0]
                weights = dense
            elif weights.shape[1] != c_in:
                raise ValueError(f"{node.name}: weights read {weights.shape[1]} input channels, the input has {c_in}")
            y_cons = G.consumers.get(y_f, [])
            relu = len(y_cons) == 1 and y_cons[0].op_type == "Relu"
            # Conv -> Relu -> QuantizeLinear: the ReLU clamps at the zero point of that one
            # quantization, so the conv output is quantized at the ReLU's scale and the
            # activation is max(q, 128).
            conv_q, s1, z1 = G.q_sink(y_cons[0].output[0] if relu else y_f)
            if z1 != ZP:
                raise ValueError(f"{node.name}: output zero point {z1}")
            layer = ConvLayer(name=node.name, index=len(layers), inputs=resolve(x_q), in_scale=sx, k=k,
                              stride=stride, pad=int(pads[0]), weights=weights.astype(np.int8),
                              bias_q=bias.astype(np.int8), bias_scale=sb, weight_scale=sw, conv_scale=s1,
                              output=conv_q)
            # Activation chain?
            conv_f = G.dq_of(conv_q)
            cons = G.float_consumers(conv_f)
            kinds = sorted(c.op_type for c in cons)
            if relu:
                layer.act = "relu"
                layer.act_scale = s1
                layer.hswish = relu_epilogue()
                out_name, out_scale = conv_q, s1
            elif kinds == ["HardSigmoid", "Mul"]:
                hs_node = [c for c in cons if c.op_type == "HardSigmoid"][0]
                mul_act = [c for c in cons if c.op_type == "Mul"][0]
                alpha = float(_attr(hs_node, "alpha", 0.2))
                if abs(alpha - float(HS_ALPHA)) > 1e-6:
                    raise ValueError(f"{node.name}: HardSigmoid alpha {alpha}")
                scale_mul = G.consumers[hs_node.output[0]][0]
                k_const = G.const(scale_mul.input[1])
                k_val = float(np.asarray(k_const).flatten()[0])
                hs_q, hs_scale, hs_zp = G.q_sink(scale_mul.output[0])
                if abs(hs_scale - 1 / 128) > 1e-12 or hs_zp != ZP:
                    raise ValueError(f"{node.name}: HardSigmoid quantization {hs_scale}/{hs_zp}")
                act_q, s2, z2 = G.q_sink(mul_act.output[0])
                layer.act = "hswish"
                layer.act_scale = s2
                layer.hswish = fit_hardswish(s1, s2, k_val)
                layer.output = act_q
                out_name = act_q
                out_scale = s2
            elif (not cons and conv_f in [o.name for o in G.g.output]) or \
                    (cons and {c.op_type for c in cons if c.name not in region_of} <= {"Conv", "Add", "DepthToSpace"}):
                # No activation (a head, SESR's head/tail convs, or YOLO11's qkv read by a host region).
                out_name, out_scale = conv_q, s1
            else:
                raise ValueError(f"{node.name}: unexpected consumers {kinds}")
            # Residual add directly after the activation? It attaches to the operand computed last:
            # the other operand must already exist (SESR's head conv feeds the long skip Add that
            # body.6 completes, so the Add belongs to body.6).
            act_f = G.dq_of(out_name)
            act_cons = G.float_consumers(act_f)

            def ready(name: str) -> bool:
                if name == act_f:
                    return True
                q = G.q_source(name)[0]
                return q in tensors or q in views

            adds = [c for c in act_cons if c.op_type == "Add" and all(ready(i) for i in c.input)]
            if adds:
                if len(act_cons) != 1:
                    raise ValueError(f"{node.name}: activation feeds more than the residual Add")
                add = adds[0]
                other = [i for i in add.input if i != act_f][0]
                res_q, s_res, _ = G.q_source(other)
                add_q, s_add, _ = G.q_sink(add.output[0])
                segs = resolve(res_q)
                if len(segs) != 1:
                    raise ValueError(f"{add.name}: residual must be one segment")
                layer.residual = segs[0]
                # Float semantics with power-of-two scales, exact in integers at the finest of the three
                # scales: y = rne((t_main << lsh_main) + (t_res << lsh_res), shift). An output finer than both
                # operands (YOLO11's attention Add) shifts both left and rounds nothing.
                s_min = min(out_scale, s_res, s_add)
                layer.residual_lsh_main = _log2_exact(out_scale / s_min)
                layer.residual_lsh_res = _log2_exact(s_res / s_min)
                layer.residual_shift = _log2_exact(s_add / s_min)
                layer.output = add_q
                out_name, out_scale = add_q, s_add
                if layer.residual_shift < 0:
                    raise ValueError(f"{add.name}: output scale {s_add} is finer than both operands")
            tensors[out_name] = TensorInfo(out_name, cout, oh, ow, out_scale, ZP, producer=layer.name)
            scales[out_name] = out_scale
            layers.append(layer)
        elif node.op_type == "Slice":
            x_q, sx, _ = G.q_source(node.input[0])
            starts = int(G.const(node.input[1]).flatten()[0])
            ends = int(G.const(node.input[2]).flatten()[0])
            axes = int(G.const(node.input[3]).flatten()[0])
            if axes != 1 or starts % 8 or ends % 8:
                raise ValueError(f"{node.name}: unsupported slice axis {axes} [{starts}, {ends})")
            out_q, so, _ = G.q_sink(node.output[0])
            if abs(so - sx) > 1e-12:
                raise ValueError(f"{node.name}: slice rescales {sx} -> {so}")
            segs = resolve(x_q)
            if len(segs) != 1:
                raise ValueError(f"{node.name}: slice of a multi-segment view")
            base = segs[0]
            add_view(out_q, [Segment(base.tensor, base.block_offset + starts // 8, (ends - starts) // 8, base.up2)])
            scales[out_q] = so
        elif node.op_type == "Concat":
            if int(_attr(node, "axis")) != 1:
                raise ValueError(f"{node.name}: concat axis")
            segs: List[Segment] = []
            for i in node.input:
                x_q, sx, _ = G.q_source(i)
                out_q_probe = None
                segs.extend(resolve(x_q))
                scales.setdefault(x_q, sx)
            out_q, so, _ = G.q_sink(node.output[0])
            for i in node.input:
                x_q, sx, _ = G.q_source(i)
                if abs(sx - so) > 1e-12:
                    raise ValueError(f"{node.name}: concat rescales {x_q} {sx} -> {so}")
            # Merge segments that are contiguous blocks of one tensor.
            merged: List[Segment] = []
            for s in segs:
                if merged and merged[-1].tensor == s.tensor and not s.up2 and not merged[-1].up2 \
                        and merged[-1].block_offset + merged[-1].blocks == s.block_offset:
                    merged[-1] = Segment(s.tensor, merged[-1].block_offset, merged[-1].blocks + s.blocks)
                else:
                    merged.append(s)
            add_view(out_q, merged)
            scales[out_q] = so
        elif node.op_type == "Resize":
            x_q, sx, _ = G.q_source(node.input[0])
            out_q, so, _ = G.q_sink(node.output[0])
            mode = _attr(node, "mode", b"nearest")
            if (mode if isinstance(mode, str) else mode.decode()) != "nearest":
                raise ValueError(f"{node.name}: resize mode {mode}")
            if abs(sx - so) > 1e-12:
                raise ValueError(f"{node.name}: resize rescales")
            segs = resolve(x_q)
            add_view(out_q, [Segment(s.tensor, s.block_offset, s.blocks, up2=True) for s in segs])
            scales[out_q] = so
        elif node.op_type == "MaxPool":
            x_q, sx, _ = G.q_source(node.input[0])
            out_q, so, _ = G.q_sink(node.output[0])
            k = int(_attr(node, "kernel_shape")[0])
            pads = _attr(node, "pads")
            if k != 5 or any(int(p) != 2 for p in pads) or abs(sx - so) > 1e-12:
                raise ValueError(f"{node.name}: unsupported maxpool")
            segs = resolve(x_q)
            if len(segs) != 1:
                raise ValueError(f"{node.name}: pool input must be one segment")
            c, h, w = dims_chw(node.output[0])
            tensors[out_q] = TensorInfo(out_q, c, h, w, so, ZP, producer=node.name)
            scales[out_q] = so
            layers.append(PoolLayer(name=node.name, index=len(layers), input=segs[0], output=out_q))
        elif node.op_type == "DepthToSpace":
            # A pure channel-to-space permutation at one quantization: the host applies it to
            # the read-back tensor (SESR's pixel shuffle), the engine never computes it.
            x_q, sx, _ = G.q_source(node.input[0])
            out_q, so, zo = G.q_sink(node.output[0])
            if abs(sx - so) > 1e-12 or zo != ZP or x_q not in tensors:
                raise ValueError(f"{node.name}: depth-to-space must read a physical tensor at its own scale")
            mode = _attr(node, "mode", b"DCR")
            transforms[out_q] = (x_q, {"op": "depth_to_space", "blocksize": int(_attr(node, "blocksize")),
                                       "mode": mode.decode() if isinstance(mode, bytes) else str(mode)})
        elif node.op_type == "Reshape":
            # Outside host regions a reshape must be a channel view (YOLO11's attention v: qkv channels 64-127,
            # then 192-255, read by the pe convolution).
            out_q, so, zo = G.q_sink(node.output[0])
            add_view(out_q, channel_view(node, so, zo))
            scales[out_q] = so
        elif node.op_type == "Add":
            continue  # handled with the producing conv
        else:
            raise ValueError(f"unsupported op {node.op_type} ({node.name})")

    for node in G.g.node:
        if node.op_type == "Add" and node.name not in region_of and G.q_sink(node.output[0])[0] not in tensors:
            raise ValueError(f"{node.name}: no conv output absorbed this Add")
    for r in regions:
        if r["q_out"] not in tensors:
            raise ValueError(f"host region {r['spec']}: its output was never built")

    outputs = []
    output_transforms: Dict[str, Dict[str, Any]] = {}
    for o in G.g.output:
        q, s, _ = G.q_source(o.name)
        if q in transforms:
            q, output_transforms[o.name] = transforms[q]
        if q not in tensors:
            raise ValueError(f"graph output {o.name} is not a physical tensor")
        outputs.append((o.name, q))

    # Concat inputs that span several physical tensors must be contiguous in the workspace.
    for segs in views.values():
        pass
    for layer in layers:
        if isinstance(layer, ConvLayer):
            for s in layer.inputs:
                pass
    adjacency = _adjacency_groups(views, tensors)
    return GraphIR(tensors=tensors, layers=layers, input=q_in, outputs=outputs, adjacency=adjacency,
                   output_transforms=output_transforms)


def _adjacency_groups(views, tensors) -> List[List[str]]:
    """Physical tensors that a Concat lists back to back become one contiguous allocation group.

    Blocks of a single tensor were already merged; here consecutive *different*
    tensors of equal geometry and scale are grouped so the consumer can read them
    through one segment with a uniform plane stride.
    """
    groups: List[List[str]] = []
    for name, segs in views.items():
        run: List[str] = []
        prev = None
        for s in segs:
            t = tensors[s.tensor]
            if prev is not None and not s.up2 and s.block_offset == 0 and s.blocks == t.blocks \
                    and (prev.height, prev.width, prev.scale) == (t.height, t.width, t.scale) \
                    and run and run[-1] == prev.name:
                run.append(t.name)
            else:
                if len(run) > 1:
                    groups.append(run)
                run = [t.name] if (s.block_offset == 0 and s.blocks == t.blocks and not s.up2) else []
            prev = t
        if len(run) > 1:
            groups.append(run)
    # Deduplicate while preserving order.
    seen, out = set(), []
    for g in groups:
        key = tuple(g)
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


def summarize(ir: GraphIR) -> str:
    lines = []
    for L in ir.layers:
        if isinstance(L, ConvLayer):
            t = ir.tensors[L.output]
            segs = ", ".join(f"{s.tensor.split('/')[-1][:24]}[{s.block_offset}:{s.block_offset + s.blocks}]"
                             f"{'x2' if s.up2 else ''}" for s in L.inputs)
            hs = f" hswish(err={L.hswish.max_error})" if L.hswish else ""
            res = f" +res>>{L.residual_shift}" if L.residual else ""
            lines.append(f"{L.index:2d} conv {L.name:38s} {L.cin:3d}->{L.cout:3d} k{L.k} s{L.stride} "
                         f"{t.height}x{t.width} shift={L.shift_out}{hs}{res} in=[{segs}]")
        elif isinstance(L, HostLayer):
            t = ir.tensors[L.output]
            ops = ", ".join(f"{k} {v}" for k, v in sorted(L.op_types.items()))
            lines.append(f"{L.index:2d} host {L.name:38s} {t.channels:3d} {t.height}x{t.width} ({ops})")
        else:
            t = ir.tensors[L.output]
            lines.append(f"{L.index:2d} pool {L.name:38s} {t.channels:3d} {t.height}x{t.width}")
    return "\n".join(lines)
