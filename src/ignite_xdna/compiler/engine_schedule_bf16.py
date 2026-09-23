"""Packets, workspace and DMA schedule for the bf16 convolution engine, carrying W8A16 work.

A FORK OF ``engine_schedule``, NOT A PARAMETERISATION OF IT. ``engine_schedule.py`` stays untouched,
so every shipping int8 container is provably unaffected by anything here. What is imported from it
is geometry and nothing else - ``Placement``, ``Workspace``, ``Chunk``, ``CHUNK_GEOMETRY``,
``layer_chunks``, ``tile_origins``, ``column_rounds``, ``LayerSchedule`` - none of which reads the
zero point, the int8 group width, a byte count or the int8 emulator. Everything that does is forked.

WHAT CARRIES OVER AND WHAT DOUBLES. A bf16 activation object is 6,400 ELEMENTS, the same count as
int8's 6,400 bytes, and a weight packet's payload is ``k*k*ncin*128`` bf16 against int8's
``k*k*ncin*256`` bytes, inside the same 9,472 B object. So the chunk geometry - rows, columns, plane
elements and input blocks a packet takes - is identical, and ``CHUNK_GEOMETRY`` is used as it stands.
Its third field is named ``plane_bytes`` because at int8 the two coincide; here it is an ELEMENT count
and goes into ``H_PLANE_ELEMS`` unchanged. What differs: every DMA byte count doubles, and one output
group is 16 channels (two 8-channel blocks) where int8's is 32.

WHAT IS CARRIED. W8A16: the IR's int8 weights times their power-of-two ``weight_scale``, which bf16
represents exactly (seven magnitude bits against eight), and activations in REAL units, so a layer is
``y = sum(w * s_w * x) + b_q * s_b`` with no requantization anywhere. ``in_scale``, ``conv_scale``,
``shift_out``, ``bias_acc()`` and the residual shift fields are int8 arithmetic and are ignored.

WHAT IS REFUSED, loudly, because the bf16 core cannot do it or its transport does not exist:
pools, fused convolutions and upsampled (``k1up2``) inputs - the kernel has no MAXPOOL, FUSED_CONV or
F_UP2; HardSwish, sigmoid and any activation after a residual add - its epilogue is ReLU and ReLU6;
host layers - a host step on uint8 QDQ tensors means nothing at bf16; the activation ring and the
resident weight buffer - not in the bf16 design. The int8 per-round schedule is not forked at all:
it is a bisection path, and it builds untrimmed packets (below).

0 x NaN = NaN. int8 is immune to garbage it multiplies by a zero weight; bf16 is not. A packet that
multiplies over-read or never-written memory by zero weights returns NaN if those bits are a NaN or
an infinity. ``trim_ncin`` is what stops a packet from multiplying the planes it only over-reads, so
here it is not an option: every packet is trimmed, and a chunk with no real input channel is refused.
The channels a packet does multiply by zero - the tail of a partially real block, like the image's
channels 3..7 - must hold finite values; a producer layer computes exact zeros there, and the runtime
must write zeros into the graph input's.

A LIMIT IN THE IR, not fixed here. The lowering records ``act = "relu"`` both for a Relu and for a
Clip(0, 6) whose upper bound int8 saturation already performs. At W8A16 nothing saturates, so a
ReLU6 model would lose its bound. SESR uses Relu nodes; the oracle comparison
(``tools/w8a16_oracle.py``) is what would expose it on a model that does not.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ignite_xdna.compiler import engine_bf16_emulator as em
from ignite_xdna.compiler.engine_schedule import (Chunk, LayerSchedule, Placement, Workspace, _segment_channel_base,
                                                  column_rounds, layer_chunks, tile_origins)
from ignite_xdna.compiler.engine_sequence import (COLS, MAX_REPEAT, ROWS, DmaPattern, canonical, linear, merge_quad,
                                                  merge_runs)
from ignite_xdna.compiler.graph_ir import ConvLayer, FusedConvLayer, GraphIR, HostLayer, PoolLayer

TILE_R, TILE_C = em.TILE_ROWS, em.TILE_COLS
OUT_BLOCKS_8 = em.OUT_BLOCKS_8          # 8-channel blocks one output group emits
GROUP_CHANNELS = OUT_BLOCKS_8 * 8       # 16, against int8's 32
SUPPORTED_KINDS = ("k1", "k3s1", "k3s2", "k5s1", "res")
BIAS_BYTES = em.BIAS_ELEMS * 2
PAYLOAD_BYTES = em.W_BYTES - em.HDR_BYTES - BIAS_BYTES

# The workspace's STORAGE dtype - what numpy sees. "bf16" is the semantic tag the manifest carries
# beside it as ``elem``; it never reaches np.dtype(), where "bf16" raises and "bfloat16" resolves only
# if ml_dtypes happened to be imported first.
STORAGE_DTYPE = "uint16"
ITEM = np.dtype(STORAGE_DTYPE).itemsize
# A shim buffer descriptor's step field is 20 bits of 32-bit words: 4 MiB (docs/SILICON.md). bf16
# doubles every plane, so it reaches this at a smaller map than int8 does - a 640x640 plane is 6.6 MB.
MAX_STRIDE_BYTES = (1 << 20) * 4


# ----------------------------------------------------------------------------
# Admission
# ----------------------------------------------------------------------------

def check_ir(ir: GraphIR) -> None:
    """Refuse a graph the bf16 engine cannot run, naming the layer and the reason."""
    for L in ir.layers:
        if isinstance(L, HostLayer):
            raise ValueError(f"{L.name}: a host layer runs a uint8 QDQ region on the CPU; the bf16 engine "
                             f"has no host step")
        if isinstance(L, PoolLayer):
            raise ValueError(f"{L.name}: max pool; the bf16 kernel has no MAXPOOL")
        if isinstance(L, FusedConvLayer):
            raise ValueError(f"{L.name}: fused convolution; the bf16 kernel has no FUSED_CONV")
        if not isinstance(L, ConvLayer):
            raise ValueError(f"{L.name}: {type(L).__name__} is not a layer the bf16 engine runs")
        if L.act not in (None, "relu"):
            raise ValueError(f"{L.name}: activation {L.act!r}; the bf16 epilogue is ReLU or none")
        if L.sigmoid is not None or L.post_hswish is not None:
            raise ValueError(f"{L.name}: sigmoid or post-residual activation; the bf16 epilogue is ReLU")
        if any(s.up2 for s in L.inputs):
            raise ValueError(f"{L.name}: upsampled input; the bf16 kernel has no F_UP2")
        kinds = {c.kind for c in layer_chunks(ir, L)}
        if not kinds <= set(SUPPORTED_KINDS):
            raise ValueError(f"{L.name}: chunk kinds {sorted(kinds - set(SUPPORTED_KINDS))} have no bf16 packet")


# ----------------------------------------------------------------------------
# Weight packets
# ----------------------------------------------------------------------------

def emits(header: np.ndarray) -> bool:
    """Does this packet write an output object?

    The two opcodes choose their destination differently: a convolution writes ``out`` only under
    F_EMIT, a residual always retires somewhere and picks ``scratch`` only under F_HOLD.
    """
    op, flags = int(header[em.H_OP]), int(header[em.H_FLAGS])
    if op == em.OP_RESIDUAL:
        return not (flags & em.F_HOLD)
    if op == em.OP_CONV:
        return bool(flags & em.F_EMIT)
    return False


def check_counts(header: np.ndarray) -> None:
    """The aliasing contract ``kernels/bf16_conv/design._core_fn`` says no schedule enforced.

    The core runs ``count_out`` passes with a real output object and then ``count_acc`` passes with
    ``scratch`` passed as the output pointer too. A packet counted in ``count_acc`` that emits would
    write its tile over the held tile a later OP_RESIDUAL reads - wrong without looking wrong - and
    one counted in ``count_out`` that does not emit would release an output object unwritten. One
    header serves every pass of its object, so exactly one of the two counts may be non-zero.
    """
    n_out, n_acc = int(header[em.H_COUNT_OUT]), int(header[em.H_COUNT_ACC])
    if (n_out > 0) == (n_acc > 0):
        raise ValueError(f"count_out {n_out} and count_acc {n_acc}: exactly one must be non-zero")
    if n_acc and emits(header):
        raise ValueError(f"an emitting packet counted in count_acc={n_acc}: in the accumulate-only loop "
                         f"the output pointer IS scratch, so it would overwrite the held tile")
    if n_out and not emits(header):
        raise ValueError(f"a non-emitting packet counted in count_out={n_out}: the core would release "
                         f"an output object unwritten")


def pack_w(header: np.ndarray, bias: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """One 9,472 B weight packet: 128 B header, 128 B replicated bias, then bf16 weights.

    ``bias`` and ``wts`` are float32; they are rounded to bf16 here, on the bits, with no
    ``ml_dtypes`` - importing it makes ``np.dtype("bfloat16")`` resolve, and whether it had been
    imported first is exactly the import-order hazard this compiler avoids.
    """
    header = np.asarray(header, np.int32)
    check_counts(header)
    pkt = np.zeros(em.W_BYTES, np.uint8)
    pkt[:em.HDR_BYTES] = header.view(np.uint8)
    b = np.asarray(bias, np.float32).reshape(-1)
    if b.size != em.BIAS_ELEMS:
        raise ValueError(f"bias has {b.size} values, the packet carries {em.BIAS_ELEMS}")
    pkt[em.HDR_BYTES:em.HDR_BYTES + BIAS_BYTES] = em.bf16_bits(em.to_bf16(b)).view(np.uint8)
    wb = em.bf16_bits(em.to_bf16(np.asarray(wts, np.float32).reshape(-1))).view(np.uint8)
    if wb.size > PAYLOAD_BYTES:
        raise ValueError(f"weights {wb.size} B exceed the {PAYLOAD_BYTES} B payload")
    pkt[em.HDR_BYTES + BIAS_BYTES:em.HDR_BYTES + BIAS_BYTES + wb.size] = wb
    return pkt


def unpack_w(pkt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(header int32[32], bias float32[64], weights float32) - what ``run_packet`` takes.

    The weight region is sliced to the size the header implies, ``k*k*ncin*NCO*32`` bf16, which is
    what the core walks; the packet itself is a fixed 9,472 B.
    """
    pkt = np.ascontiguousarray(pkt, dtype=np.uint8)
    header = pkt[:em.HDR_BYTES].view(np.int32).copy()
    bias = em.from_bf16_bits(pkt[em.HDR_BYTES:em.HDR_BYTES + BIAS_BYTES].view(np.uint16))
    k, ncin = int(header[em.H_K]), int(header[em.H_NCIN])
    n = k * k * ncin * em.NCO * 32 if int(header[em.H_OP]) == em.OP_CONV else 0
    w0 = em.HDR_BYTES + BIAS_BYTES
    wts = em.from_bf16_bits(pkt[w0:w0 + 2 * n].view(np.uint16))
    return header, bias, wts


def dequantized_weights(layer: ConvLayer) -> np.ndarray:
    """float32 [Cout][Cin][k][k]: the IR's int8 weights times their scale, refused unless bf16-exact.

    With a power-of-two scale an int8 weight is exactly representable in bf16, so the packet carries
    the int8 model's weights with no rounding at all. A scale that breaks that would change the
    weights silently; it is refused rather than rounded.
    """
    w = layer.weights.astype(np.float64) * float(layer.weight_scale)
    w32 = w.astype(np.float32)
    if not (np.array_equal(w32.astype(np.float64), w) and np.array_equal(em.to_bf16(w32), w32)):
        raise ValueError(f"{layer.name}: weights * weight_scale ({layer.weight_scale}) is not exact in bf16; "
                         f"W8A16 carries the int8 weights unrounded, which needs a power-of-two scale")
    return w32


def dequantized_bias(layer: ConvLayer) -> np.ndarray:
    """float32 [Cout]: ``bias_q * bias_scale``. Rounded to bf16 by ``pack_w``, which is the one rounding."""
    return (layer.bias_q.astype(np.float64) * float(layer.bias_scale)).astype(np.float32)


def conv_packet(layer: ConvLayer, group: int, chunk: Chunk, count_out: int, count_acc: int) -> np.ndarray:
    """The weight packet of one chunk of one 16-channel output group.

    Weights are ``[tap][cin_block][NCO][8][4]``: element ``(c, b, kk, n)`` multiplies input channel
    ``8c + kk`` into output channel ``4b + n`` of the group, the order the core's walk and
    ``mmul<4, 8, 4>`` read them. The bias is pre-replicated to the accumulator's 4x4, element
    ``m*4 + n`` being channel ``n`` - bf16 has no 4-element load.

    Always trimmed: the header advertises only the input blocks holding real channels, because the
    core multiplies every block it is told about and an over-read block multiplied by zero weights
    is NaN whenever its bits are (see the module docstring).
    """
    seg = layer.inputs[chunk.seg_index]
    taps = layer.k * layer.k
    cbase = _segment_channel_base(layer, chunk.seg_index) + chunk.block_start * 8
    cin_avail = min(chunk.ncin * 8, seg.blocks * 8 - chunk.block_start * 8, layer.cin - cbase)
    if cin_avail <= 0:
        raise ValueError(f"{layer.name}: chunk {chunk.index} holds no real input channel; at bf16 a packet "
                         f"that multiplies only over-read memory by zero weights can return NaN")
    ncin = min(chunk.ncin, -(-cin_avail // 8))
    co0 = group * GROUP_CHANNELS
    cout_avail = min(GROUP_CHANNELS, layer.cout - co0)
    if cout_avail <= 0:
        raise ValueError(f"{layer.name}: output group {group} starts past its {layer.cout} channels")

    w = np.zeros((taps, ncin, em.NCO, 8, 4), np.float32)
    block = dequantized_weights(layer)[co0:co0 + cout_avail, cbase:cbase + cin_avail]   # [co][ci][ky][kx]
    wt = block.transpose(2, 3, 1, 0).reshape(taps, cin_avail, cout_avail)              # [tap][ci][co]
    ci = np.arange(cin_avail)[:, None]
    co = np.arange(cout_avail)[None, :]
    w[:, ci // 8, co // 4, ci % 8, co % 4] = wt

    per_ch = np.zeros(GROUP_CHANNELS, np.float32)
    per_ch[:cout_avail] = dequantized_bias(layer)[co0:co0 + cout_avail]
    bias = np.repeat(per_ch.reshape(em.NCO, 1, 4), 4, axis=1).reshape(-1)

    flags = em.F_LOAD_PSUM if chunk.index > 0 else 0
    if chunk.last:
        flags |= em.F_HOLD if layer.residual is not None else em.F_EMIT
        if layer.act == "relu":
            flags |= em.F_RELU
    # chunk.plane_bytes is an ELEMENT count: CHUNK_GEOMETRY was written at one byte an element, and
    # H_PLANE_ELEMS wants elements. Doubling it here would be the bug, not the fix.
    header = em.make_header(op=em.OP_CONV, k=layer.k, stride=layer.stride, ncin=ncin, flags=flags,
                            count_out=count_out, count_acc=count_acc, rows_in=chunk.rows_in,
                            cols_in=chunk.cols_in, plane_elems=chunk.plane_bytes)
    return pack_w(header, bias, w)


def residual_packet(layer: ConvLayer) -> np.ndarray:
    """OP_RESIDUAL: add this round's residual tile to the tile the layer's last chunk held, and emit.

    No activation after the add: ``check_ir`` refuses one. The core ignores k, ncin and the plane
    fields for this opcode; they are given harmless values rather than left zero.
    """
    header = em.make_header(op=em.OP_RESIDUAL, k=1, stride=1, ncin=1, flags=em.F_EMIT, count_out=1,
                            count_acc=0, rows_in=TILE_R, cols_in=TILE_C)
    return pack_w(header, np.zeros(em.BIAS_ELEMS, np.float32), np.zeros(em.NCO * 32, np.float32))


def chunk_packet(layer: ConvLayer, group: int, chunk: Chunk) -> np.ndarray:
    """One round's packet for ``chunk``, counted for exactly one object - the unit a multi-chunk round uses."""
    if chunk.kind == "res":
        return residual_packet(layer)
    emitting = chunk.last and layer.residual is None
    return conv_packet(layer, group, chunk, count_out=1 if emitting else 0, count_acc=0 if emitting else 1)


def output_groups(blocks: int) -> int:
    """16-channel output groups a tensor of ``blocks`` 8-channel blocks needs."""
    return -(-blocks // OUT_BLOCKS_8)


def layer_packets(ir: GraphIR, layer: ConvLayer) -> List[List[np.ndarray]]:
    """Per output group, the chunk packets of one round, in chunk order."""
    chunks = layer_chunks(ir, layer)
    return [[chunk_packet(layer, g, ch) for ch in chunks]
            for g in range(output_groups(ir.tensors[layer.output].blocks))]


# ----------------------------------------------------------------------------
# Workspace
# ----------------------------------------------------------------------------

class Bf16Workspace(Workspace):
    """The DDR workspace of a bf16 container. THE ARRAY HOLDS 16-BIT PATTERNS.

    Float conversion happens only at the boundary, through ``write_values`` / ``read_values``.
    ``write_tensor`` refuses anything but uint16, because the inherited one would cast float32 values
    into uint16 by truncation without a word of complaint - a plausible wrong answer. Compare what
    ``read_tensor`` returns against ``bf16_bits(reference)``, never as floats: -0.0 and +0.0 compare
    equal and are different bytes.

    ``read_tensor`` is inherited and already correct at two bytes an element: it slices by
    ``plane_bytes`` (itemsize-aware through ``Placement.pitch``) and views as the placement dtype.
    """

    def halo_fill(self, background: int = 0) -> np.ndarray:
        """Workspace image with every halo ring set to bf16 +0.0 and everything else ``background``.

        ``background`` is a 16-bit pattern. 0 is what a freshly allocated buffer holds; a NaN pattern
        (0x7FC0) is how a test proves that no multiply-accumulate reads memory nothing wrote.
        """
        ws = np.full(self.nbytes // ITEM, background, dtype=np.uint16)
        for p in self.placements.values():
            if p.halo_value != 0:
                raise ValueError(f"{p.name}: halo value {p.halo_value}; bf16 pads with +0.0")
            if p.halo == 0:
                continue
            for b in range(p.planes):
                lo = (p.base + b * p.plane_bytes) // ITEM
                plane = ws[lo:lo + p.plane_bytes // ITEM].reshape(p.height + 2 * p.halo, p.width + 2 * p.halo, 8)
                plane[:p.halo], plane[-p.halo:] = 0, 0
                plane[:, :p.halo], plane[:, -p.halo:] = 0, 0
        return ws.view(np.uint8)

    def write_tensor(self, ws: np.ndarray, name: str, chw: np.ndarray) -> None:
        if np.asarray(chw).dtype != np.uint16:
            raise TypeError(f"{name}: the workspace holds bf16 bit patterns; pass bf16_bits(values), "
                            f"not {np.asarray(chw).dtype}")
        super().write_tensor(ws, name, chw)

    def write_values(self, ws: np.ndarray, name: str, chw: np.ndarray) -> None:
        """Round float values to bf16 and store their patterns."""
        self.write_tensor(ws, name, em.bf16_bits(em.to_bf16(np.asarray(chw, np.float32))))

    def read_values(self, ws: np.ndarray, name: str) -> np.ndarray:
        """float32 ``[blocks * 8][H][W]`` of the bf16 values a tensor holds."""
        return em.from_bf16_bits(self.read_tensor(ws, name))


def input_plane(placement: Dict, values: np.ndarray) -> np.ndarray:
    """The graph input's plane as a container expects it staged: ``[H + 2h][W + 2h][8]`` uint16 patterns.

    ``placement`` is the manifest's dict, what a session reads; the layout comes from this module's
    ``Bf16Workspace``, what the DMA patterns were built against. So a session's staging is checked
    against the compiler rather than against a second copy of its own arithmetic. The halo ring is
    +0.0, ``values`` go in as bf16 patterns, and the block's lanes past them are +0.0, because the head
    convolution multiplies them by zero weights and 0 x NaN is NaN. One block only: an image is one.
    """
    if placement.get("dtype") != STORAGE_DTYPE or int(placement.get("band_rows") or 0):
        raise ValueError(f"not a bf16 placement: dtype {placement.get('dtype')!r}, "
                         f"band_rows {placement.get('band_rows')!r}")
    if int(placement["blocks"]) != 1:
        raise ValueError(f"the input spans {placement['blocks']} channel blocks; an image is one")
    p = Placement(name="input", base=0, halo=int(placement["halo"]), height=int(placement["height"]),
                  width=int(placement["width"]), blocks=1, planes=int(placement["planes"]),
                  halo_value=int(placement["halo_value"]), dtype=STORAGE_DTYPE, band_rows=0)
    ws = Bf16Workspace(placements={"input": p}, nbytes=p.planes * p.plane_bytes, input="input")
    arr = ws.halo_fill()          # refuses a halo value other than +0.0
    ws.write_values(arr, "input", values)
    return arr[:p.plane_bytes].view(np.uint16).reshape(p.height + 2 * p.halo, p.width + 2 * p.halo, 8)


def _halo_widths(ir: GraphIR) -> Dict[str, int]:
    """Per tensor, the widest padding any convolution reading it needs. The value is always +0.0."""
    halo = {name: 0 for name in ir.tensors}
    for L in ir.layers:
        if L.k > 1:
            for s in L.inputs:
                halo[s.tensor] = max(halo[s.tensor], L.pad)
    return halo


def _planes(ir: GraphIR, name: str) -> int:
    """Real blocks plus the junk planes a 2-block output group writes past them (int8 pads to 4)."""
    t = ir.tensors[name]
    return t.blocks + ((-t.blocks) % OUT_BLOCKS_8 if t.producer != "input" else 0)


def plan_workspace(ir: GraphIR, slack_bytes: int = 64, reuse: bool = True) -> Bf16Workspace:
    """Place every tensor in one workspace, ``[plane][H + 2h][W + 2h][8]`` bf16, halos at +0.0.

    The int8 planner's logic, liveness-based slot reuse included, with three differences: junk planes
    pad to two blocks, every placement is ``uint16`` with a +0.0 halo, and no tensor is band-packed
    (``band_rows`` is a performance layout, worth about 0.1 ms on yolov8n, and not carried into the
    fork). The workspace is then grown to cover the furthest byte any activation packet over-reads.
    """
    check_ir(ir)
    halo = _halo_widths(ir)

    def placement(name: str, base: int, planes: int) -> Placement:
        t = ir.tensors[name]
        return Placement(name=name, base=base, halo=halo[name], height=t.height, width=t.width, blocks=t.blocks,
                         planes=planes, producer=t.producer, halo_value=0, dtype=STORAGE_DTYPE, band_rows=0)

    def align(n: int) -> int:
        return (n + 63) // 64 * 64

    placements: Dict[str, Placement] = {}
    outputs = [L.output for L in ir.layers]
    if not reuse:
        cursor = 0
        for name in [ir.input] + outputs:
            placements[name] = placement(name, cursor, _planes(ir, name))
            cursor = align(cursor + placements[name].nbytes)
    else:
        first_use: Dict[str, int] = {}
        last_use: Dict[str, int] = {}
        for idx, L in enumerate(ir.layers):
            first_use.setdefault(L.output, idx)
            last_use[L.output] = max(last_use.get(L.output, idx), idx)
            for t in [s.tensor for s in L.inputs] + ([L.residual.tensor] if L.residual else []):
                last_use[t] = max(last_use.get(t, 0), idx)
        for _, t in ir.outputs:
            last_use[t] = len(ir.layers) + 1
        placements[ir.input] = placement(ir.input, 0, _planes(ir, ir.input))
        cursor = align(placements[ir.input].nbytes)
        # Identical geometry means identical pitch and interior offsets, and drains write only the
        # interior, so a slot handed to a later tensor keeps its halo ring intact.
        by_geom = defaultdict(list)
        for name in outputs:
            t = ir.tensors[name]
            by_geom[(t.height, t.width, halo[name])].append((first_use[name], last_use[name], _planes(ir, name), name))
        for key, tlist in by_geom.items():
            slots: List[dict] = []
            for f, l, planes, name in sorted(tlist, key=lambda e: e[0]):
                slot = next((s for s in slots if s["last_use"] < f), None)
                if slot is None:
                    slots.append({"last_use": l, "max_planes": planes, "tensors": [name]})
                else:
                    slot["last_use"] = l
                    slot["max_planes"] = max(slot["max_planes"], planes)
                    slot["tensors"].append(name)
            for slot in slots:
                for name in slot["tensors"]:
                    placements[name] = placement(name, cursor, slot["max_planes"])
                cursor = align(cursor + placements[slot["tensors"][0]].nbytes)
    ws = Bf16Workspace(placements=placements, nbytes=cursor, input=ir.input)
    max_read = cursor
    for L in ir.layers:
        t = ir.tensors[L.output]
        for chunk in layer_chunks(ir, L):
            for y0 in {0, t.height - TILE_R}:
                for x0 in {0, t.width - TILE_C}:
                    for g in range(output_groups(t.blocks)):
                        max_read = max(max_read, int(a_pattern(ws, L, chunk, y0, x0, g).indices().max()) + 1)
    ws.nbytes = align(max_read + slack_bytes)
    return ws


# ----------------------------------------------------------------------------
# DMA patterns, in BYTES
# ----------------------------------------------------------------------------

def _pattern(buffer: str, offset: int, sizes: Tuple[int, ...], strides: Tuple[int, ...]) -> DmaPattern:
    p = canonical(DmaPattern(buffer, offset, sizes, strides))
    if any(s > MAX_STRIDE_BYTES for s in p.strides):
        raise ValueError(f"a stride of {max(p.strides):,} B exceeds the shim descriptor's "
                         f"{MAX_STRIDE_BYTES:,} B step field; this map is too large for a bf16 plane")
    return p


def a_pattern(ws: Workspace, layer: ConvLayer, chunk: Chunk, y0: int, x0: int, group: int = 0) -> DmaPattern:
    """The DMA pattern that fills one core's 12,800 B activation packet for output tile (y0, x0).

    Derived from the chunk's own ELEMENT geometry - ``read_blocks`` planes of ``rows_in`` rows of
    ``cols_in`` pixels - times the item size, in one place. The int8 file writes the same shapes as
    byte literals (200, 400, 160), and those are the counts that go silently wrong at bf16: right
    addresses, half the length. Here there is no literal to carry over.

    A residual chunk reads the residual tensor at the OUTPUT tile's origin, starting at this group's
    two blocks; the six blocks after them are over-read to make up the fixed packet, and never used.
    """
    if chunk.read_blocks * chunk.rows_in * chunk.cols_in * 8 != em.A_ELEMS:
        raise ValueError(f"{layer.name}: a {chunk.kind} chunk does not fill one {em.A_ELEMS}-element packet")
    if chunk.kind == "res":
        seg = layer.residual
        b0, oy, ox = seg.block_offset + group * OUT_BLOCKS_8, y0, x0
    else:
        seg = layer.inputs[chunk.seg_index]
        b0 = seg.block_offset + chunk.block_start
        oy, ox = layer.stride * y0 - layer.pad, layer.stride * x0 - layer.pad
    p = ws.placements[seg.tensor]
    return _pattern("ws", p.offset(b0, oy, ox), (chunk.read_blocks, chunk.rows_in, chunk.cols_in * 8 * ITEM),
                    (p.plane_stride, p.pitch, 1))


def o_pattern(ws: Workspace, layer: ConvLayer, group: int, y0: int, x0: int) -> DmaPattern:
    """The drain of one joined 12,800 B output object: four cores' strips of two 8-channel blocks."""
    p = ws.placements[layer.output]
    return DmaPattern("ws", p.offset(group * OUT_BLOCKS_8, y0, x0), (ROWS, OUT_BLOCKS_8, TILE_R, TILE_C * 8 * ITEM),
                      (TILE_R * p.pitch, p.plane_stride, p.pitch, 1))


def quad_patterns(ws: Workspace, layer: ConvLayer, chunk: Chunk, y_quad: int, x0: int,
                  group: int = 0) -> List[DmaPattern]:
    """The four per-core fill patterns of the quad starting at output row ``y_quad``."""
    return [a_pattern(ws, layer, chunk, y_quad + TILE_R * r, x0, group) for r in range(ROWS)]


def run_drain(ws: Workspace, layer: ConvLayer, group: int, run: List[Tuple[int, int]]) -> DmaPattern:
    """One drain for a run of vertically adjacent quads at one tile column; rounds are (y, x0).

    Canonicalised for the reason the int8 one is: a foldable four-dimensional tap is ambiguous to the
    descriptor lowering, whose outermost dimension may be read as the repeat or as addressing.
    """
    q0, x0 = run[0]
    o = o_pattern(ws, layer, group, q0, x0)
    return _pattern("ws", o.offset, (ROWS * len(run),) + tuple(o.sizes[1:]), o.strides)


# ----------------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------------

class PacketStore:
    """Deduplicated static weight packets laid out back to back, each re-checked against the contract."""

    def __init__(self):
        self._offsets: Dict[bytes, int] = {}
        self._chunks: List[np.ndarray] = []
        self.nbytes = 0

    def add(self, pkt: np.ndarray) -> int:
        return self.add_run([pkt])

    def add_run(self, pkts: List[np.ndarray]) -> int:
        """Store packets back to back (one weight task streams them all); dedup by content."""
        for p in pkts:
            if p.size != em.W_BYTES:
                raise ValueError(f"a weight packet is {p.size} B, not {em.W_BYTES}")
            check_counts(p[:em.HDR_BYTES].view(np.int32))
        key = b"".join(p.tobytes() for p in pkts)
        if key in self._offsets:
            return self._offsets[key]
        off = self.nbytes
        self._offsets[key] = off
        for p in pkts:
            self._chunks.append(p)
            self.nbytes += em.W_BYTES
        return off

    def blob(self) -> np.ndarray:
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.uint8)

    def packets_at(self, offset: int, nbytes: int) -> List[np.ndarray]:
        first = offset // em.W_BYTES
        return self._chunks[first:first + nbytes // em.W_BYTES]

    def opcodes(self) -> set:
        """Every opcode the stored packets carry, read out of the packets themselves."""
        return {int(p[:em.HDR_BYTES].view(np.int32)[em.H_OP]) for p in self._chunks}


def schedule_layer(ir: GraphIR, ws: Workspace, layer: ConvLayer, store: PacketStore, weight_repeat: bool = True,
                   balance_columns: bool = True, merge_group_weights: bool = True) -> LayerSchedule:
    """Cut one layer into as few DMA tasks as the transport allows: int8's coarse schedule, at bf16.

    Rounds are taken tile column first, so a column's rounds form runs of vertically adjacent quads
    at one tile column (at most 16, the repeat limit of 64 packets). Per run: one drain, issued
    ahead of its fills and held until they are issued. A single-chunk layer streams every strip of a
    run in one fill task and serves every round of a column from one weight packet, through its
    header's ``count_out``. A multi-chunk layer merges each round's per-chunk fills across the four
    cores where they are regularly spaced, and streams its run of weight packets once per round with
    a stride-0 repeat. Output groups are 16 channels, so a layer has ``ceil(blocks / 2)`` of them.
    """
    t = ir.tensors[layer.output]
    chunks = layer_chunks(ir, layer)
    single = len(chunks) == 1
    n_groups = output_groups(t.blocks)
    quad_rows = TILE_R * ROWS
    ys, xs = tile_origins(t.height, quad_rows), tile_origins(t.width, TILE_C)
    max_quads = MAX_REPEAT // ROWS
    programs: List[List[tuple]] = [[] for _ in range(COLS)]
    n_rounds = n_packets = n_w = 0
    per_col_groups: List[List[tuple]] = [[] for _ in range(COLS)]
    for g in range(n_groups):
        rounds = [(y, x0) for x0 in xs for y in ys]
        per_col = column_rounds(rounds, g, balance=balance_columns)
        for c in range(COLS):
            mine = per_col[c]
            if not mine:
                continue
            runs: List[List[Tuple[int, int]]] = []
            for q, x0 in mine:
                last = runs[-1][-1] if runs else None
                if last is not None and last[1] == x0 and last[0] == q - quad_rows and len(runs[-1]) < max_quads:
                    runs[-1].append((q, x0))
                else:
                    runs.append([(q, x0)])
            run_fills: List[List[List[DmaPattern]]] = []
            for run in runs:
                if single:
                    strips = [p for q, x0 in run for p in quad_patterns(ws, layer, chunks[0], q, x0, g)]
                    run_fills.append([merge_runs(strips)])
                else:
                    per_round = []
                    for q, x0 in run:
                        pats: List[DmaPattern] = []
                        for ch in chunks:
                            quad = quad_patterns(ws, layer, ch, q, x0, g)
                            merged = merge_quad(quad)
                            pats.extend([merged] if merged is not None else quad)
                        per_round.append(merge_runs(pats))
                    run_fills.append(per_round)
                n_rounds += len(run)
                n_packets += ROWS * len(run) * len(chunks)
            per_col_groups[c].append((g, mine, runs, run_fills))

    for c in range(COLS):
        entries = per_col_groups[c]
        if not entries:
            continue
        items_of = [sum(len(f) for per_round in rf for f in per_round) for _, _, _, rf in entries]
        if single:
            pkts = [conv_packet(layer, g, chunks[0], count_out=len(mine), count_acc=0) for g, mine, _, _ in entries]
            if merge_group_weights:
                # The groups' packets back to back: each serves its own group's rounds in order.
                programs[c].append(("w", store.add_run(pkts), len(pkts) * em.W_BYTES, sum(items_of)))
                n_w += 1
            for k, (g, mine, runs, run_fills) in enumerate(entries):
                if not merge_group_weights:
                    programs[c].append(("w", store.add(pkts[k]), em.W_BYTES, items_of[k]))
                    n_w += 1
                for run, per_round in zip(runs, run_fills):
                    programs[c].append(("o", run_drain(ws, layer, g, run), len(per_round[0])))
                    programs[c].extend(("A", f) for f in per_round[0])
            continue
        runs_pkts = [[chunk_packet(layer, g, ch) for ch in chunks] for g, _, _, _ in entries]
        run_len = len(runs_pkts[0]) * em.W_BYTES
        rounds_col = [len(mine) for _, mine, _, _ in entries]
        # Only the repeat (outermost) dimension of a DMA task may have stride 0, so the groups' runs
        # share one task where every group has a single round in this column: the runs back to back,
        # repeated with a positive stride. Columns with several rounds per group keep one repeated
        # task per group.
        merged_w = (merge_group_weights and weight_repeat and 1 < len(entries) <= MAX_REPEAT
                    and all(n == 1 for n in rounds_col))
        if merged_w:
            off = store.add_run([p for run in runs_pkts for p in run])
            programs[c].append(("W", DmaPattern("wp", off, (len(entries), 1, 1, run_len), (run_len, 0, 0, 1)),
                                sum(items_of)))
            n_w += 1
        for k, (g, mine, runs, run_fills) in enumerate(entries):
            run_offset = None if merged_w else store.add_run(runs_pkts[k])
            if weight_repeat and not merged_w:
                rounds_items = [len(f) for per_round in run_fills for f in per_round]
                for start in range(0, len(rounds_items), MAX_REPEAT):
                    part = rounds_items[start:start + MAX_REPEAT]
                    pat = (DmaPattern("wp", run_offset, (len(part), 1, 1, run_len), (0, 0, 0, 1)) if len(part) > 1
                           else linear("wp", run_offset, run_len))
                    programs[c].append(("W", pat, sum(part)))
                    n_w += 1
            for run, per_round in zip(runs, run_fills):
                programs[c].append(("o", run_drain(ws, layer, g, run), sum(len(f) for f in per_round)))
                for fills in per_round:
                    if not weight_repeat:
                        programs[c].append(("w", run_offset, run_len, len(fills)))
                        n_w += 1
                    programs[c].extend(("A", f) for f in fills)
    return LayerSchedule(layer.index, layer.name, programs, n_rounds, n_packets, n_w)


def schedule_graph(ir: GraphIR, ws: Workspace, activation_ring: int = 0, weight_buffer: bool = False,
                   **flags) -> Tuple[List[LayerSchedule], PacketStore]:
    """Every layer's schedule and the packet store they share - the call ``engine_compile`` makes.

    ``activation_ring`` and ``weight_buffer`` are accepted because the compile passes them, and
    refused because the bf16 design has neither. ``flags`` are ``schedule_layer``'s, for bisection.
    """
    if activation_ring or weight_buffer:
        raise ValueError("the bf16 design has no activation ring and no resident weight buffer")
    check_ir(ir)
    store = PacketStore()
    scheds = [schedule_layer(ir, ws, L, store, **flags) for L in ir.layers]
    return scheds, store


# ----------------------------------------------------------------------------
# Emulating a schedule on a workspace array
# ----------------------------------------------------------------------------

def emulate_layer(sched: LayerSchedule, store: PacketStore, ws_arr: np.ndarray,
                  wp_arr: Optional[np.ndarray] = None, mac_model: Optional[str] = None) -> None:
    """Replay the column programs through the bf16 core emulator, updating ``ws_arr`` in place.

    Streams are replayed as the hardware sees them: fills and weight tasks append to per-column
    FIFOs, every four activation packets form one object (one packet per core), and a drain writes
    its objects once the cores have emitted them, whether it was issued before or after its fills.

    Each core keeps its own fp32 psum and its ``scratch`` across the layer, as the device buffers
    do. A weight object serves its ``count_out`` objects with a real output object and then its
    ``count_acc`` objects with ``scratch`` passed as the output pointer as well - exactly what
    ``kernels/bf16_conv/design._core_fn`` does. That aliasing is modelled rather than assumed away,
    so a packet that broke the count contract would corrupt the held tile here as it would there.
    """
    blob = wp_arr
    for c, items in enumerate(sched.programs):
        psum = [np.zeros(em.PSUM_FLOATS, np.float32) for _ in range(ROWS)]
        scratch = [np.zeros(em.SCRATCH_ELEMS, np.float32) for _ in range(ROWS)]
        w_queue: List[np.ndarray] = []
        cur = None
        used = n_out = n_acc = 0
        pending: List[np.ndarray] = []
        drains: List[DmaPattern] = []
        packets: List[np.ndarray] = []

        def flush_drains():
            while drains:
                n_obj = drains[0].nbytes // (ROWS * em.O_BYTES)
                if len(pending) < n_obj:
                    return
                drains.pop(0).write(ws_arr, np.concatenate(pending[:n_obj]))
                del pending[:n_obj]

        def consume(stream):
            nonlocal cur, used, n_out, n_acc
            packets.extend(stream)
            while len(packets) >= ROWS:
                obj = packets[:ROWS]
                del packets[:ROWS]
                if cur is None or used == n_out + n_acc:
                    if not w_queue:
                        raise AssertionError(f"{sched.name}: activation object without a weight object")
                    cur = unpack_w(w_queue.pop(0))
                    n_out, n_acc, used = int(cur[0][em.H_COUNT_OUT]), int(cur[0][em.H_COUNT_ACC]), 0
                hdr, bias, wts = cur
                real_out = used < n_out
                used += 1
                outs = []
                for r, pkt in enumerate(obj):
                    act = em.from_bf16_bits(np.ascontiguousarray(pkt).view(np.uint16))
                    if real_out:
                        o = np.zeros(em.OUT_ELEMS, np.float32)
                        em.run_packet(hdr, act, wts, bias, psum[r], o, scratch[r], core_row=r, mac_model=mac_model)
                        outs.append(em.bf16_bits(o).view(np.uint8))
                    else:
                        em.run_packet(hdr, act, wts, bias, psum[r], scratch[r], scratch[r], core_row=r,
                                      mac_model=mac_model)
                if real_out:
                    pending.append(np.concatenate(outs))
                    flush_drains()

        for it in items:
            kind = it[0]
            if kind == "w":
                if blob is None:
                    w_queue.extend(store.packets_at(it[1], it[2]))
                else:
                    w_queue.extend(blob[it[1] + k:it[1] + k + em.W_BYTES] for k in range(0, it[2], em.W_BYTES))
            elif kind == "W":
                if blob is None:
                    blob = store.blob()
                data = it[1].read(blob)
                w_queue.extend(data[k:k + em.W_BYTES] for k in range(0, data.size, em.W_BYTES))
            elif kind in ("a", "A"):
                stream = []
                for pat in (it[1] if kind == "a" else [it[1]]):
                    data = pat.read(ws_arr)
                    if data.size % em.A_BYTES:
                        raise AssertionError(f"{sched.name}: fill of {data.size} bytes is not whole packets")
                    stream.extend(data[k:k + em.A_BYTES] for k in range(0, data.size, em.A_BYTES))
                consume(stream)
            elif kind == "o":
                drains.append(it[1])
                flush_drains()
            else:
                raise AssertionError(f"{sched.name}: item {kind!r} has no meaning on the bf16 engine")
        if packets:
            raise AssertionError(f"{sched.name}: {len(packets)} activation packets do not form whole objects")
        if pending or drains:
            raise AssertionError(f"{sched.name}: {len(pending)} emitted objects, {len(drains)} drains left")
        if w_queue or (cur is not None and used != n_out + n_acc):
            raise AssertionError(f"{sched.name}: weight objects left over ({len(w_queue)}, "
                                 f"{n_out + n_acc - used} uses of the last)")
