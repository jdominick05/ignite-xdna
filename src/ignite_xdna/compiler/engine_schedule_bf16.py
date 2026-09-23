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

from typing import List, Tuple

import numpy as np

from ignite_xdna.compiler import engine_bf16_emulator as em
from ignite_xdna.compiler.engine_schedule import Chunk, _segment_channel_base, layer_chunks
from ignite_xdna.compiler.graph_ir import ConvLayer, FusedConvLayer, GraphIR, HostLayer, PoolLayer

TILE_R, TILE_C = em.TILE_ROWS, em.TILE_COLS
OUT_BLOCKS_8 = em.OUT_BLOCKS_8          # 8-channel blocks one output group emits
GROUP_CHANNELS = OUT_BLOCKS_8 * 8       # 16, against int8's 32
SUPPORTED_KINDS = ("k1", "k3s1", "k3s2", "k5s1", "res")
BIAS_BYTES = em.BIAS_ELEMS * 2
PAYLOAD_BYTES = em.W_BYTES - em.HDR_BYTES - BIAS_BYTES


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
