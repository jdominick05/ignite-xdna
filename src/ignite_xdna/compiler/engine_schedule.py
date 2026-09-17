"""Workspace planning, tiling and packet scheduling for the convolution engine.

* ``plan_workspace`` places every physical tensor of a ``GraphIR`` in one DDR
  workspace in the channel-blocked layout ``[block][H + 2h][W + 2h][8]`` with
  a halo ring of ``h`` pixels (1 for 3x3 consumers, 2 for the SPPF pools) that
  the runtime fills once with the zero point, plus junk planes that absorb the
  output blocks a 4-block tile writes beyond the tensor's real channels.
* ``schedule_layer`` cuts a layer into 5x20-pixel tiles, groups four vertically
  adjacent strips into a round (one packet per core of a column), splits the
  input channels into chunks that fit one 6,400-byte activation packet, builds
  the weight packets and returns per-column item lists for
  ``engine_sequence.SequenceEmitter``.
* ``emulate_layer`` replays those item lists with the NumPy core emulator on
  a workspace array, so the DMA descriptors, the packet headers and the core
  arithmetic are all exercised before anything touches silicon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ignite_xdna.compiler import engine_emulator as em
from ignite_xdna.compiler.engine_sequence import COLS, MAX_REPEAT, ROWS, DmaPattern, linear, merge_quad, merge_runs
from ignite_xdna.compiler.graph_ir import ConvLayer, GraphIR, HostLayer, PoolLayer, Segment, TensorInfo, ZP

TILE_R, TILE_C = em.TILE_ROWS, em.TILE_COLS
OUT_BLOCKS = em.OUT_BLOCKS


@dataclass
class Placement:
    name: str
    base: int
    halo: int
    height: int
    width: int
    blocks: int          # real channel blocks
    planes: int          # blocks + junk planes
    producer: str = ""
    halo_value: int = ZP  # 128 (zero point) for conv consumers, 0 (-inf) for max-pool consumers

    @property
    def pitch(self) -> int:
        return (self.width + 2 * self.halo) * 8

    @property
    def plane_bytes(self) -> int:
        return (self.height + 2 * self.halo) * self.pitch

    @property
    def nbytes(self) -> int:
        return self.planes * self.plane_bytes

    def offset(self, block: int, y: int, x: int) -> int:
        """Byte offset of tensor pixel (y, x) of ``block``; y/x may reach into the halo."""
        return self.base + block * self.plane_bytes + (y + self.halo) * self.pitch + (x + self.halo) * 8


@dataclass
class Workspace:
    placements: Dict[str, Placement]
    nbytes: int
    input: str

    def halo_fill(self) -> np.ndarray:
        """Workspace image with every halo ring set to the zero point (128)."""
        ws = np.zeros(self.nbytes, dtype=np.uint8)
        for p in self.placements.values():
            if p.halo == 0:
                continue
            for b in range(p.planes):
                plane = ws[p.base + b * p.plane_bytes:p.base + (b + 1) * p.plane_bytes].reshape(
                    p.height + 2 * p.halo, p.width + 2 * p.halo, 8)
                plane[:p.halo, :, :] = p.halo_value
                plane[-p.halo:, :, :] = p.halo_value
                plane[:, :p.halo, :] = p.halo_value
                plane[:, -p.halo:, :] = p.halo_value
        return ws

    def read_tensor(self, ws: np.ndarray, name: str) -> np.ndarray:
        """Return the real interior of a tensor as uint8 [C][H][W]."""
        p = self.placements[name]
        planes = ws[p.base:p.base + p.blocks * p.plane_bytes].reshape(
            p.blocks, p.height + 2 * p.halo, p.width + 2 * p.halo, 8)
        interior = planes[:, p.halo:p.halo + p.height, p.halo:p.halo + p.width, :]
        chw = np.transpose(interior, (0, 3, 1, 2)).reshape(p.blocks * 8, p.height, p.width)
        return chw

    def write_tensor(self, ws: np.ndarray, name: str, chw: np.ndarray) -> None:
        p = self.placements[name]
        c = chw.shape[0]
        padded = np.full((p.blocks * 8, p.height, p.width), ZP, dtype=np.uint8)
        padded[:c] = chw
        blocked = np.transpose(padded.reshape(p.blocks, 8, p.height, p.width), (0, 2, 3, 1))
        planes = ws[p.base:p.base + p.blocks * p.plane_bytes].reshape(
            p.blocks, p.height + 2 * p.halo, p.width + 2 * p.halo, 8)
        planes[:, p.halo:p.halo + p.height, p.halo:p.halo + p.width, :] = blocked


def _halos(ir: GraphIR) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Per-tensor halo width and halo value: zero point for 3x3 convs, 0 for max pools."""
    halo = {name: 0 for name in ir.tensors}
    value = {name: ZP for name in ir.tensors}
    pooled = set()
    for L in ir.layers:
        if isinstance(L, ConvLayer) and L.k > 1:
            for s in L.inputs:
                halo[s.tensor] = max(halo[s.tensor], L.pad)   # 1 for 3x3, 2 for SESR's 5x5
        elif isinstance(L, PoolLayer):
            halo[L.input.tensor] = max(halo[L.input.tensor], 2)
            pooled.add(L.input.tensor)
    for name in pooled:
        if any(isinstance(L, ConvLayer) and L.k > 1 and any(s.tensor == name for s in L.inputs) for L in ir.layers):
            raise ValueError(f"{name} feeds both a 3x3 conv and a max pool; halo values conflict")
        value[name] = 0
    return halo, value


def plan_workspace(ir: GraphIR, slack_bytes: int = 64) -> Workspace:
    halo, halo_value = _halos(ir)
    placements: Dict[str, Placement] = {}
    cursor = 0
    order = [ir.input] + [L.output for L in ir.layers]
    for name in order:
        t = ir.tensors[name]
        engine_written = t.producer != "input"
        junk = (-t.blocks) % OUT_BLOCKS if engine_written else 0
        p = Placement(name=name, base=cursor, halo=halo[name], height=t.height, width=t.width,
                      blocks=t.blocks, planes=t.blocks + junk, producer=t.producer, halo_value=halo_value[name])
        placements[name] = p
        cursor = (cursor + p.nbytes + 63) // 64 * 64
    ws = Workspace(placements=placements, nbytes=cursor, input=ir.input)
    # Over-read slack: the largest byte any activation packet reads past the end.
    max_read = cursor
    for L in ir.layers:
        for chunk in layer_chunks(ir, L):
            for y0, x0 in ((ir.tensors[L.output].height - TILE_R, ir.tensors[L.output].width - TILE_C),):
                pat = a_pattern(ws, ir, L, chunk, y0, x0)
                max_read = max(max_read, int(pat.indices().max()) + 1)
    ws.nbytes = (max_read + slack_bytes + 63) // 64 * 64
    return ws


# ----------------------------------------------------------------------------
# Chunks and DMA patterns
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    kind: str            # "k1", "k1up2", "k3s1", "k3s2", "pool", "res"
    seg_index: int       # index into layer.inputs (res: residual)
    block_start: int     # first block inside the segment
    ncin: int            # blocks the header advertises
    read_blocks: int     # blocks the DMA reads (10 for k1up2, ncin otherwise)
    rows_in: int
    cols_in: int
    plane_bytes: int
    index: int           # chunk ordinal within the layer
    last: bool           # last conv chunk (retires the accumulators)


def conv_chunk_kind(layer: ConvLayer, seg: Segment) -> str:
    geometry = f"{layer.k}x{layer.k} stride-{layer.stride} pad-{layer.pad}"
    if layer.k == 1:
        # The k1 access pattern reads the output tile's own pixels: no stride, no padding.
        if layer.stride != 1 or layer.pad != 0:
            raise ValueError(f"{layer.name}: no packet geometry for a {geometry} conv")
        return "k1up2" if seg.up2 else "k1"
    if layer.k == 5 and layer.stride == 1 and layer.pad == 2 and not seg.up2:
        return "k5s1"
    if layer.k != 3 or layer.pad != 1 or layer.stride not in (1, 2) or seg.up2:
        raise ValueError(f"{layer.name}: no packet geometry for a {geometry} conv")
    return "k3s2" if layer.stride == 2 else "k3s1"


# (rows_in, cols_in, plane_bytes, header ncin, blocks the DMA reads)
CHUNK_GEOMETRY = {
    "k1": (5, 20, 800, 8, 8),
    "k1up2": (5, 20, 800, 8, 10),
    "k3s1": (8, 25, 1600, 4, 4),
    "k3s2": (16, 50, 6400, 1, 1),
    # 5x5 stride 1 reads 9 rows x 24 pixels of one block; 25 taps x 256 weights fill 6,400 of the
    # 9,216 weight bytes, so one input block per packet, read through the k3s2 plane layout.
    "k5s1": (16, 50, 6400, 1, 1),
    "pool": (16, 25, 3200, 2, 2),
    "res": (5, 20, 800, 4, 8),
}


def layer_chunks(ir: GraphIR, layer) -> List[Chunk]:
    chunks: List[Chunk] = []
    if isinstance(layer, HostLayer):
        return chunks  # computed on the host; the engine reads no packets for it
    if isinstance(layer, PoolLayer):
        rows_in, cols_in, pb, ncin, rb = CHUNK_GEOMETRY["pool"]
        for i in range(2):  # blocks 0-1 held, blocks 2-3 emitted
            chunks.append(Chunk("pool", 0, 2 * i, ncin, rb, rows_in, cols_in, pb, i, i == 1))
        return chunks
    idx = 0
    specs = []
    for si, seg in enumerate(layer.inputs):
        kind = conv_chunk_kind(layer, seg)
        rows_in, cols_in, pb, ncin, rb = CHUNK_GEOMETRY[kind]
        for start in range(0, seg.blocks, ncin):
            specs.append((kind, si, start, ncin, rb, rows_in, cols_in, pb))
    for i, (kind, si, start, ncin, rb, rows_in, cols_in, pb) in enumerate(specs):
        chunks.append(Chunk(kind, si, start, ncin, rb, rows_in, cols_in, pb, i, i == len(specs) - 1))
    if layer.residual is not None:
        rows_in, cols_in, pb, ncin, rb = CHUNK_GEOMETRY["res"]
        chunks.append(Chunk("res", -1, 0, ncin, rb, rows_in, cols_in, pb, len(specs), False))
    return chunks


def a_pattern(ws: Workspace, ir: GraphIR, layer, chunk: Chunk, y0: int, x0: int, group: int = 0) -> DmaPattern:
    """DMA pattern that fills one core's 6,400-byte packet for output tile (y0, x0)."""
    if chunk.kind == "res":
        seg = layer.residual
        p = ws.placements[seg.tensor]
        # Four real blocks plus four over-read blocks make the fixed 6,400-byte packet.
        return DmaPattern("ws", p.offset(seg.block_offset + group * OUT_BLOCKS, y0, x0),
                          (8, TILE_R, TILE_C * 8), (p.plane_bytes, p.pitch, 1))
    seg = layer.input if isinstance(layer, PoolLayer) else layer.inputs[chunk.seg_index]
    p = ws.placements[seg.tensor]
    b0 = seg.block_offset + chunk.block_start
    if chunk.kind == "pool":
        b0 += group * OUT_BLOCKS  # a pool round produces the same four blocks it reads
    if chunk.kind == "k1":
        return DmaPattern("ws", p.offset(b0, y0, x0), (8, 5, 160), (p.plane_bytes, p.pitch, 1))
    if chunk.kind == "k1up2":
        return DmaPattern("ws", p.offset(b0, y0 >> 1, x0 >> 1), (10, 4, 160), (p.plane_bytes, p.pitch, 1))
    if chunk.kind == "k3s1":
        return DmaPattern("ws", p.offset(b0, y0 - 1, x0 - 1), (4, 8, 200), (p.plane_bytes, p.pitch, 1))
    if chunk.kind == "k3s2":
        return DmaPattern("ws", p.offset(b0, 2 * y0 - 1, 2 * x0 - 1), (16, 400), (p.pitch, 1))
    if chunk.kind == "k5s1":
        return DmaPattern("ws", p.offset(b0, y0 - 2, x0 - 2), (16, 400), (p.pitch, 1))
    if chunk.kind == "pool":
        return DmaPattern("ws", p.offset(b0, y0 - 2, x0 - 2), (2, 16, 200), (p.plane_bytes, p.pitch, 1))
    raise ValueError(chunk.kind)


def o_pattern(ws: Workspace, layer, group: int, y0: int, x0: int) -> DmaPattern:
    """DMA pattern that drains one joined 12,800-byte object: four strips of four blocks."""
    p = ws.placements[layer.output]
    return DmaPattern("ws", p.offset(group * OUT_BLOCKS, y0, x0), (4, OUT_BLOCKS, TILE_R, TILE_C * 8),
                      (TILE_R * p.pitch, p.plane_bytes, p.pitch, 1))


# Coarse schedule: core r of a quad expands source rows (y_quad / 2) + 2r .. + 3 with
# phase r, so the four cores' upsampling packets are spaced by two source rows and
# merge into one task ((5r + i) >> 1 - 2r == (i + r) >> 1 for output row i).
UP2_PHASES_COARSE = (0, 1, 2, 3)
UP2_PHASES_STRIP = (0, 1, 0, 1)


def quad_patterns(ws: Workspace, ir: GraphIR, layer, chunk: Chunk, y_quad: int, x0: int, group: int = 0,
                  coarse: bool = False) -> List[DmaPattern]:
    """The four per-core fill patterns of the quad starting at output row ``y_quad``."""
    if coarse and chunk.kind == "k1up2":
        seg = layer.inputs[chunk.seg_index]
        p = ws.placements[seg.tensor]
        b0 = seg.block_offset + chunk.block_start
        return [DmaPattern("ws", p.offset(b0, (y_quad >> 1) + 2 * r, x0 >> 1), (10, 4, 160), (p.plane_bytes, p.pitch, 1))
                for r in range(ROWS)]
    return [a_pattern(ws, ir, layer, chunk, y_quad + TILE_R * r, x0, group) for r in range(ROWS)]


def tile_origins(extent: int, step: int) -> List[int]:
    """Tile origins covering ``extent`` pixels with tiles of ``step``: every multiple of ``step`` that
    fits, plus ``extent - step`` when ``step`` does not divide ``extent``. The last tile then overlaps
    its neighbour (SESR's 256-pixel maps: ..., 220, 236) and computes the shared pixels twice from
    the same inputs, so both writes carry the same bytes. YOLO maps divide evenly and are unchanged."""
    if extent < step:
        raise ValueError(f"a {extent}-pixel map is smaller than one {step}-pixel tile")
    origins = list(range(0, extent - step + 1, step))
    if origins[-1] + step < extent:
        origins.append(extent - step)
    return origins


def run_drain(ws: Workspace, layer, group: int, run: List[Tuple[int, int]]) -> DmaPattern:
    """One drain for a run of vertically adjacent quads (at most 16) at one tile column; rounds are (y, x0)."""
    q0, x0 = run[0]
    o = o_pattern(ws, layer, group, q0, x0)
    return DmaPattern("ws", o.offset, (ROWS * len(run),) + tuple(o.sizes[1:]), o.strides)


# ----------------------------------------------------------------------------
# Weight packets
# ----------------------------------------------------------------------------

def _segment_channel_base(layer: ConvLayer, seg_index: int) -> int:
    return sum(s.blocks * 8 for s in layer.inputs[:seg_index])


def conv_packet(layer: ConvLayer, group: int, chunk: Chunk, count_out: int, count_acc: int,
                phases: Tuple[int, int, int, int] = (0, 1, 0, 1), trim_ncin: bool = False) -> np.ndarray:
    """Weight packet of one chunk. ``trim_ncin`` advertises only the blocks that hold real
    input channels, so the core skips the junk blocks the fixed-size packet over-reads.
    Output blocks stay four: skipping junk output blocks needs a run-time guard inside the
    core's block loops, which costs more than it saves (docs/DECISIONS.md)."""
    seg = layer.inputs[chunk.seg_index]
    taps = layer.k * layer.k
    cbase = _segment_channel_base(layer, chunk.seg_index) + chunk.block_start * 8
    # Channels beyond the segment's real blocks or beyond the ONNX Cin (the
    # 3-channel image is stored in one 8-channel block) get zero weights.
    cin_avail = min(chunk.ncin * 8, seg.blocks * 8 - chunk.block_start * 8, layer.cin - cbase)
    ncin = max(1, min(chunk.ncin, -(-cin_avail // 8))) if trim_ncin else chunk.ncin
    w = np.zeros((taps, ncin, OUT_BLOCKS, 8, 8), dtype=np.int8)
    co0 = group * OUT_BLOCKS * 8
    cout_avail = min(OUT_BLOCKS * 8, layer.cout - co0)
    if cin_avail > 0 and cout_avail > 0:
        block = layer.weights[co0:co0 + cout_avail, cbase:cbase + cin_avail]  # [co][ci][ky][kx]
        wt = np.transpose(block, (2, 3, 1, 0)).reshape(taps, cin_avail, cout_avail)  # [tap][ci][co]
        for ci in range(cin_avail):
            for co in range(cout_avail):
                w[:, ci // 8, co // 8, ci % 8, co % 8] = wt[:, ci, co]
    bias = np.zeros(32, dtype=np.int64)
    bacc = layer.bias_acc()
    if bacc.min() < -(1 << 31) or bacc.max() >= (1 << 31):   # an int32 bias can leave the accumulator's range
        raise ValueError(f"{layer.name}: accumulator bias {bacc.min()}..{bacc.max()} does not fit int32")
    bias[:cout_avail] = bacc[co0:co0 + cout_avail]
    flags = 0
    if chunk.index > 0:
        flags |= em.F_LOAD_PSUM
    if chunk.last:
        flags |= em.F_HOLD if layer.residual is not None else em.F_EMIT
        if layer.hswish is not None:   # HardSwish, or ReLU through the same epilogue
            flags |= em.F_HSWISH
        if layer.sigmoid is not None:  # SiLU through the sigmoid epilogue, applied to the finished tile
            flags |= em.F_SIGMOID
    if seg.up2:
        flags |= em.F_UP2
    last_sig = chunk.last and layer.sigmoid is not None
    hdr = em.PacketHeader(op=em.OP_CONV, k=layer.k, stride=layer.stride, ncin=ncin, nco=OUT_BLOCKS,
                          flags=flags, shift_out=layer.shift_out,
                          hs=layer.hswish.params if layer.hswish else None,
                          sig=layer.sigmoid.params if last_sig else None, count_out=count_out,
                          count_acc=count_acc, phases=phases, rows_in=chunk.rows_in,
                          cols_in=chunk.cols_in, plane_bytes=chunk.plane_bytes)
    return em.pack_w_packet(hdr, bias.astype(np.int32), w)


def residual_packet(layer: ConvLayer) -> np.ndarray:
    flags, lsh_m, lsh_r = em.F_EMIT, 0, 0
    if layer.residual_lsh_main or (layer.residual_lsh_res is not None
                                   and layer.residual_lsh_res != layer.residual_shift):
        flags |= em.F_RES_SHIFTS
        lsh_m, lsh_r = layer.residual_lsh_main, layer.residual_lsh_res
    hs = None
    if layer.post_hswish is not None:  # activation after the add, at this packet's constants
        flags |= em.F_HSWISH
        hs = layer.post_hswish.params
    hdr = em.PacketHeader(op=em.OP_RESIDUAL, ncin=4, nco=OUT_BLOCKS, flags=flags, rsh=layer.residual_shift,
                          rlsh_m=lsh_m, rlsh_r=lsh_r, count_out=1, hs=hs)
    return em.pack_w_packet(hdr, np.zeros(32, np.int32), None)


def pool_packet(chunk: Chunk) -> np.ndarray:
    flags = (em.F_EMIT | em.F_LOAD_PSUM) if chunk.last else em.F_HOLD
    hdr = em.PacketHeader(op=em.OP_MAXPOOL, ncin=chunk.ncin, nco=OUT_BLOCKS, flags=flags,
                          count_out=1 if chunk.last else 0, count_acc=0 if chunk.last else 1,
                          rows_in=chunk.rows_in, cols_in=chunk.cols_in, plane_bytes=chunk.plane_bytes)
    return em.pack_w_packet(hdr, np.zeros(32, np.int32), None)


class PacketStore:
    """Deduplicated static weight packets laid out back to back."""

    def __init__(self):
        self._offsets: Dict[bytes, int] = {}
        self._chunks: List[np.ndarray] = []
        self.nbytes = 0

    def add(self, pkt: np.ndarray) -> int:
        return self.add_run([pkt])

    def add_run(self, pkts: List[np.ndarray]) -> int:
        """Store packets back to back (one weight task streams them all); dedup by content."""
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

    def packet_at(self, offset: int) -> np.ndarray:
        return self._chunks[offset // em.W_BYTES]

    def packets_at(self, offset: int, nbytes: int) -> List[np.ndarray]:
        first = offset // em.W_BYTES
        return self._chunks[first:first + nbytes // em.W_BYTES]


# ----------------------------------------------------------------------------
# Layer schedule
# ----------------------------------------------------------------------------

@dataclass
class LayerSchedule:
    layer_index: int
    name: str
    programs: List[List[tuple]]       # per column item lists
    rounds: int
    packets: int                      # activation packets (all cores)
    w_fills: int
    # Set when the MemTile activation ring serves this layer: the geometry the instruction stream arms it with.
    ring: Optional["RingPlan"] = None


def round_packets(layer, group: int, chunks: List[Chunk], coarse: bool = False,
                  trim_ncin: bool = False) -> List[np.ndarray]:
    """The weight packets one round of a multi-chunk layer consumes, in chunk order."""
    run = []
    for ch in chunks:
        if ch.kind == "res":
            run.append(residual_packet(layer))
        elif ch.kind == "pool":
            run.append(pool_packet(ch))
        else:
            emits = ch.last and layer.residual is None
            phases = UP2_PHASES_COARSE if coarse and ch.kind == "k1up2" else UP2_PHASES_STRIP
            run.append(conv_packet(layer, group, ch, count_out=1 if emits else 0, count_acc=0 if emits else 1,
                                   phases=phases, trim_ncin=trim_ncin))
    return run


def column_rounds(rounds: List[Tuple[int, int]], group: int, balance: bool = True) -> List[List[Tuple[int, int]]]:
    """Assign one output group's rounds to the four columns.

    Rounds are sliced contiguously (tile column first), so vertically adjacent
    quads share a column. A group with fewer rounds than columns (every 20x20
    map has one round per group) is rotated by its group index with ``balance``,
    so a layer's groups spread over all columns instead of queuing on column 0.
    """
    n_r = len(rounds)
    if balance and n_r < COLS:
        per_col: List[List[Tuple[int, int]]] = [[] for _ in range(COLS)]
        for i, rnd in enumerate(rounds):
            per_col[(group * n_r + i) % COLS].append(rnd)
        return per_col
    return [rounds[c * n_r // COLS:(c + 1) * n_r // COLS] for c in range(COLS)]


# ----------------------------------------------------------------------------
# MemTile activation ring
# ----------------------------------------------------------------------------

MEMTILE_RING_BYTES = 447488    # a column's free MemTile SRAM once the split and join buffers are counted

# The resident weight buffer's limits, mirrored from kernels/aie2/conv_engine/design.py, which compiles
# them. They are restated rather than imported because importing that module pulls in aie.dialects, which
# exists only in ironenv, and the offline schedule gate is worth being runnable without it; the gate reads
# design.py's source and asserts the two agree, so they cannot drift silently.
WBUF_BYTES = 303104
WBUF_ARMS_MAX = 4
WBUF_OBJECTS = WBUF_BYTES // em.W_BYTES


def _wbuf_pieces(n_objects: int) -> Tuple[int, int]:
    """``(objects per piece, pieces)`` for streaming ``n_objects`` weight objects through the buffer.

    A piece must DIVIDE the run exactly: the fill and serve are one fixed-length descriptor each, executed
    ``pieces`` times by the queue push, so a remainder would make the last execution short and desynchronise
    every later object. And ``pieces - 1`` is that push's repeat count, which the verifier caps.
    """
    for piece in range(min(n_objects, WBUF_OBJECTS), 0, -1):
        if n_objects % piece == 0 and n_objects // piece <= MAX_REPEAT:
            return piece, n_objects // piece
    raise ValueError(f"cannot stream {n_objects} weight objects through a {WBUF_OBJECTS}-object buffer "
                     f"in at most {MAX_REPEAT} pieces")


@dataclass
class RingPlan:
    """How one layer uses the MemTile activation ring.

    Every layer with packet geometry goes through the ring, because the ring replaces the split ObjectFifo
    rather than sitting beside it. A MemTile has six DMA channels in each direction; the ring needs one S2MM
    and four MM2S, and the output join already holds four S2MM and one MM2S. There is no second delivery path
    to fall back to, so a layer that cannot amortise is still served by the ring and simply fetches once per
    group, which moves exactly the bytes the per-group schedule moved.

    ``chunks`` is the tile's chunk count and ``capacity`` how wide one pass of the ring is. A tile wider than
    the window is filled and served in several passes, which is what yolov8s's two 32-chunk layers need, and
    that width is the window narrowed to a divisor of the tile so every pass of a layer has the same shape
    (see ``_pass_width``). ``serves`` is how many times a fetched pass is replayed: the number of the tile's output groups the
    column owns when every chunk is group-invariant, and 1 otherwise. ``a_pattern`` offsets by the output
    group for the "res" and "pool" kinds, so those packets differ per group and cannot be replayed; every conv
    kind is group-invariant. ``replicas`` is how many columns hold a copy of the same tile: a layer with fewer
    rounds than columns would idle three of them if its tile were pinned to one, so it is replicated instead
    and each column takes its share of the groups.
    """
    chunks: int
    capacity: int
    serves: int
    replicas: int

    @property
    def passes(self) -> int:
        return -(-self.chunks // self.capacity)


def _pass_width(chunks: int, ring_slots: int) -> int:
    """How wide one pass of the ring is: the window, narrowed to a divisor of the tile.

    The descriptors that walk one pass form a cycle, and a cycle's length is that width. A cycle can only be
    relinked while its channel is quiesced, which a layer boundary guarantees and a tile boundary does not, so
    every pass of a layer has to be the same width. A window that does not divide the tile breaks that: 32
    chunks in passes of six ends with a pass of two, and on yolov8s every one of the 66 ring layers had a
    column arming two widths.

    Narrowing the window to the largest divisor that fits removes the remainder without padding anything - 8
    and 16 and 32 become 4, 9 becomes 3. The alternative was to pad each short pass out to the full window, but
    a padding slot cannot be a hole: the cycle walks every slot, so each must be filled and served or its
    ``space`` is never released, and the cores take their trip count from the weight packet header rather than
    from packets arriving, so every padding slot needs a real fill and a NOP weight packet. Measured over both
    models that came to 5.9-6.6 ms a frame, more than the whole ring saves.

    Narrowing costs 12 DMA tasks on yolov8n and 96 on yolov8s and not one byte, because a pass merges its
    fills over fewer chunks; against what arming per layer rather than per tile is worth, 0.2% and 1.1%.
    """
    if chunks <= ring_slots:
        return chunks
    for width in range(ring_slots, 0, -1):
        if chunks % width == 0:
            return width
    return 1


def ring_plan(ir: GraphIR, layer, ring_slots: int) -> Optional[RingPlan]:
    """The layer's ring plan, or None if it keeps the per-group schedule.

    A layer qualifies when it has more than one output group for a fetch to amortise and a tile that fits the
    ring, and when every one of its chunks is group-invariant. ``a_pattern`` offsets by the output group for the
    "res" and "pool" kinds, so those packets differ per group; every conv kind is group-invariant.

    A layer that mixes the two is left on the per-group schedule. Every activation packet reaches a core through
    the MemTile, so a mixed layer would have to interleave ring-served packets with per-group fetches in the
    exact chunk order its weight packets encode, and that order cannot simply be rearranged: ``ch.last`` retires
    the accumulators, so a residual moved after it changes the result. Modelled over the measured streams, the
    restriction costs 0.18 ms of yolov8s's dispatch floor (7.69 ms against 7.51) and 0.01 ms of yolov8n's.
    """
    if ring_slots <= 0 or isinstance(layer, HostLayer):
        return None
    try:
        chunks = layer_chunks(ir, layer)
    except (ValueError, KeyError, AttributeError):   # an op with no packet geometry
        return None
    if not chunks:
        return None
    if ring_slots * ROWS * em.A_BYTES > MEMTILE_RING_BYTES:
        raise ValueError(f"a {ring_slots}-slot ring needs more than the MemTile's {MEMTILE_RING_BYTES} B")
    t = ir.tensors[layer.output]
    groups = (t.blocks + OUT_BLOCKS - 1) // OUT_BLOCKS
    invariant = all(c.kind not in ("res", "pool") for c in chunks)
    quad_rows = TILE_R * ROWS
    tiles = len(tile_origins(t.height, quad_rows)) * len(tile_origins(t.width, TILE_C))
    # A tile that does not fit one pass cannot be replayed either: the next group would have to re-fetch the
    # pass before it, so a multi-pass layer fetches per group like the per-group schedule and saves nothing.
    if invariant and groups > 1 and len(chunks) <= ring_slots:
        replicas = min(groups, max(1, -(-COLS // tiles)))
        serves = -(-groups // replicas)
    else:
        replicas, serves = 1, 1
    return RingPlan(chunks=len(chunks), capacity=_pass_width(len(chunks), ring_slots),
                    serves=serves, replicas=replicas)


def schedule_layer_ring(ir: GraphIR, ws: Workspace, layer, store: PacketStore, plan: RingPlan,
                        trim_ncin: bool = True) -> LayerSchedule:
    """One layer's tile-outer schedule: fetch a tile once, replay it for every output group its column owns.

    ``schedule_layer_coarse`` loops output groups outermost and re-issues the layer's whole input tile inside
    each one. Here the loop is inverted. The layer's (tile, group) work is ordered tile first and sliced
    contiguously across the columns, so a column that owns several groups of one tile fills it once, and a tile
    whose groups are split across columns is filled once per column (``RingPlan.replicas``). Every chunk is
    group-invariant - ``ring_plan`` admits no other layer - so the whole tile is replayed and nothing of it is
    re-fetched.

    Items per tile: ``R`` arms the arrival for the tile's window, the fills follow, then each group's drain and
    weight run, and ``S`` replays the window once per group.

    ``R`` carries a third element saying whether the emitter may await this column's outstanding drains before
    re-arming. The ring's five descriptors and its lock pair are shared by every tile, so a re-arm must not
    overtake the serves still reading them, and a MemTile serve cannot report its own completion to the shim.
    A drain is the shim-side proof that those serves are done - the cores had to consume their slices to emit
    the objects it moves - and it is a completion token the emitter already awaits. Only a tile or a group's
    first pass may use it; a later pass shares the group's single drain, which cannot complete until that last
    pass has been served, so awaiting it between passes would deadlock the dispatch.

    The per-group schedule issues a drain ahead of its fills and holds it, because its fills feed the cores
    directly and a core must have a drain queued before it can emit. A ring fill lands in the MemTile instead,
    and nothing computes until ``S``, so the fills go first and the drains and weights need no hold at all.
    That also keeps the drain channel retirable: ``served`` clears only the oldest held task of a channel per
    activation item, and ``retire_channel`` refuses to walk past a held one, so G drains sharing one set of
    fills cannot each hold on them.

    The drain is what this costs. A tile's groups are no longer consecutive for one group, so a drain covers
    one quad and one group instead of a run of up to 16. Measured over the streams that is about 3,500 more
    instruction ops on yolov8s, against the 9,901 the ring removes, and slightly net-negative on yolov8n; both
    are second order against the bytes.
    """
    t = ir.tensors[layer.output]
    chunks = layer_chunks(ir, layer)
    n_groups = (t.blocks + OUT_BLOCKS - 1) // OUT_BLOCKS
    quad_rows = TILE_R * ROWS
    ys, xs = tile_origins(t.height, quad_rows), tile_origins(t.width, TILE_C)
    rounds = [(y, x0) for x0 in xs for y in ys]
    pairs = [(i, g) for i in range(len(rounds)) for g in range(n_groups)]
    programs: List[List[tuple]] = [[] for _ in range(COLS)]
    n_rounds = n_packets = n_w = 0
    for c in range(COLS):
        mine = pairs[c * len(pairs) // COLS:(c + 1) * len(pairs) // COLS]
        if not mine:
            continue
        # Consecutive pairs of one tile: this column fills that tile once and replays it for those groups.
        tiles: List[Tuple[int, List[int]]] = []
        for tile_index, g in mine:
            if tiles and tiles[-1][0] == tile_index:
                tiles[-1][1].append(g)
            else:
                tiles.append((tile_index, [g]))
        for tile_index, group_list in tiles:
            y, x0 = rounds[tile_index]

            def pass_fills(part, group):
                # ``a_pattern`` offsets a "res" or "pool" chunk by ``group * OUT_BLOCKS``, so a layer that is
                # fetched once per group must name that group here. Only a replayed tile may pass 0, and only
                # because every chunk of such a layer is group-invariant.
                pats: List[DmaPattern] = []
                for ch in part:
                    quad = quad_patterns(ws, ir, layer, ch, y, x0, group, coarse=True)
                    merged = merge_quad(quad)
                    pats.extend([merged] if merged is not None else quad)
                return merge_runs(pats)

            def group_work(g):
                pkts = round_packets(layer, g, chunks, coarse=True, trim_ncin=trim_ncin)
                programs[c].append(("o", run_drain(ws, layer, g, [(y, x0)]), 0))
                programs[c].append(("w", store.add_run(pkts), len(pkts) * em.W_BYTES, 0))

            if plan.serves > 1:
                # One fetch of the whole tile, replayed for every group this column owns. Every drain
                # outstanding here belongs to the previous tile, whose "S" was issued before it, so each of
                # them can complete and their completion orders this re-arm behind that tile's serves.
                programs[c].append(("R", len(chunks), True, len(group_list)))
                programs[c].extend(("A", f) for f in pass_fills(chunks, 0))
                for g in group_list:
                    group_work(g)
                    n_w += 1
                    n_packets += ROWS * len(chunks)
                programs[c].append(("S", len(group_list)))
            else:
                # No replay to amortise, so the tile is fetched per group, in passes the ring can hold. The
                # drain and the weights follow the first pass's fills, as they do above: a ring fill lands in
                # the MemTile and nothing computes until the "S" that follows it.
                for g in group_list:
                    for p in range(0, len(chunks), plan.capacity):
                        part = chunks[p:p + plan.capacity]
                        # Only the pass that begins a group may wait. This group's drain is issued with that
                        # first pass and completes only once every later pass has been served, so a pass that
                        # waited on it would wait for work its own "S" has not issued yet.
                        programs[c].append(("R", len(part), p == 0, 1))
                        programs[c].extend(("A", f) for f in pass_fills(part, g))
                        if p == 0:
                            group_work(g)
                            n_w += 1
                            n_packets += ROWS * len(chunks)
                        programs[c].append(("S", 1))
            n_rounds += 1
    return LayerSchedule(layer.index, layer.name, programs, n_rounds, n_packets, n_w, ring=plan)


def schedule_layer_coarse(ir: GraphIR, ws: Workspace, layer, store: PacketStore,
                          weight_repeat: bool = True, trim_ncin: bool = True,
                          balance_columns: bool = True, merge_group_weights: bool = True,
                          weight_buffer: bool = False) -> LayerSchedule:
    """Cut one layer into as few DMA tasks as the transport allows.

    Rounds are taken tile column first, so a column's rounds form runs of
    vertically adjacent quads at one tile column (at most 16, the repeat limit
    of 64 packets). Per run: one drain, issued ahead of its fills and held
    until they are issued; for a single-chunk layer one fill task streams every
    strip of the run; for a multi-chunk layer each round's per-chunk fills are
    merged across chunks where they are regularly spaced (stride-2 chunks) and
    across the four cores. A multi-chunk layer's weight packets are one run per
    round, streamed as that run repeated once per round (``weight_repeat``) by
    one task per column.
    """
    t = ir.tensors[layer.output]
    chunks = layer_chunks(ir, layer)
    single = len(chunks) == 1
    n_groups = (t.blocks + OUT_BLOCKS - 1) // OUT_BLOCKS
    quad_rows = TILE_R * ROWS
    ys, xs = tile_origins(t.height, quad_rows), tile_origins(t.width, TILE_C)
    max_quads = MAX_REPEAT // ROWS
    programs: List[List[tuple]] = [[] for _ in range(COLS)]
    n_rounds = n_packets = n_w = 0
    # Gather every column's groups first, so one task can stream a column's weight
    # packets for all of its groups ahead of their drains and fills.
    per_col_groups: List[List[tuple]] = [[] for _ in range(COLS)]
    for g in range(n_groups):
        # A round is (y, x0): the quad's first output row and the tile column, in pixels.
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
            # run_fills[run][round] = merged fill patterns of that round (single chunk: one entry per run)
            run_fills: List[List[List[DmaPattern]]] = []
            for run in runs:
                if single:
                    strips = [p for q, x0 in run
                              for p in quad_patterns(ws, ir, layer, chunks[0], q, x0, g, coarse=True)]
                    run_fills.append([merge_runs(strips)])
                else:
                    per_round = []
                    for q, x0 in run:
                        pats: List[DmaPattern] = []
                        for ch in chunks:
                            quad = quad_patterns(ws, ir, layer, ch, q, x0, g, coarse=True)
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
            pkts = [pool_packet(chunks[0]) if isinstance(layer, PoolLayer)
                    else conv_packet(layer, g, chunks[0], count_out=len(mine), count_acc=0, trim_ncin=trim_ncin)
                    for g, mine, _, _ in entries]
            if weight_buffer:
                # A single-chunk layer amortises through the packet header (count_out), not through a
                # stride-0 repeat, so it replays nothing and has nothing to recover. It still has to be
                # ROUTED, because with the buffer on there is no shim-to-core weight flow left - so it is
                # streamed THROUGH the buffer rather than held in it.
                piece, pieces = _wbuf_pieces(len(pkts))
                programs[c].append(("T", piece * em.W_BYTES // 4, pieces))
                programs[c].append(("w", store.add_run(pkts), len(pkts) * em.W_BYTES, sum(items_of)))
                n_w += 1
            elif merge_group_weights:
                # The groups' packets back to back: each serves its group's packets in order.
                programs[c].append(("w", store.add_run(pkts), len(pkts) * em.W_BYTES, sum(items_of)))
                n_w += 1
            for k, (g, mine, runs, run_fills) in enumerate(entries):
                if not merge_group_weights and not weight_buffer:
                    programs[c].append(("w", store.add(pkts[k]), em.W_BYTES, items_of[k]))
                    n_w += 1
                for run, per_round in zip(runs, run_fills):
                    programs[c].append(("o", run_drain(ws, layer, g, run), len(per_round[0])))
                    programs[c].extend(("A", f) for f in per_round[0])
            continue
        runs_pkts = [round_packets(layer, g, chunks, coarse=True, trim_ncin=trim_ncin) for g, _, _, _ in entries]
        run_len = len(runs_pkts[0]) * em.W_BYTES
        rounds_col = [len(mine) for _, mine, _, _ in entries]
        if weight_buffer:
            # The stride-0 repeat below is the re-send, and this is where it stops being the shim's job:
            # one contiguous fetch of every group's run into the MemTile, then one ARM per group whose
            # serve replays that group's slice for its own rounds. Arms are all armed at the layer
            # boundary and sequenced by the start queue, so no descriptor is rewritten while it is live.
            off = store.add_run([p for run in runs_pkts for p in run])
            arms = tuple((k * run_len // 4, run_len // 4, sum(len(per_round) for per_round in rf))
                         for k, (_, _, _, rf) in enumerate(entries))
            held = (all(r > 1 for _, _, r in arms) and len(arms) <= WBUF_ARMS_MAX
                    and sum(run for _, run, _ in arms) * 4 <= WBUF_BYTES)
            if held:
                programs[c].append(("B", arms))
            else:
                # Nothing to recover, or too big to hold. Layers whose groups each send their run once
                # are exactly the ones the sizing measurement said never need holding - and the largest of
                # them concatenate 1,212,416 B across eight groups, four times the buffer and twice the
                # four descriptors a start queue holds. Stream those through instead.
                piece, pieces = _wbuf_pieces(len(entries) * len(runs_pkts[0]))
                programs[c].append(("T", piece * em.W_BYTES // 4, pieces))
            programs[c].append(("w", off, len(entries) * run_len, sum(items_of)))
            n_w += 1
        # Only the repeat (outermost) dimension of a DMA task may have stride 0, so the
        # groups' runs share one task where every group has a single round in this column
        # (20x20 and 40x40 maps): the groups' runs back to back, repeated with a positive
        # stride. Columns with several rounds per group keep one repeated task per group.
        merged_w = (merge_group_weights and weight_repeat and not weight_buffer
                    and 1 < len(entries) <= MAX_REPEAT and all(n == 1 for n in rounds_col))
        if merged_w:
            off = store.add_run([p for run in runs_pkts for p in run])
            programs[c].append(("W", DmaPattern("wp", off, (len(entries), 1, 1, run_len), (run_len, 0, 0, 1)),
                                sum(items_of)))
            n_w += 1
        for k, (g, mine, runs, run_fills) in enumerate(entries):
            run_offset = None if (merged_w or weight_buffer) else store.add_run(runs_pkts[k])
            if weight_repeat and not merged_w and not weight_buffer:
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


def schedule_layer(ir: GraphIR, ws: Workspace, layer, store: PacketStore, merge_fills: bool = True,
                   pair_drains: bool = True, weight_runs: bool = True, coarse: bool = True,
                   weight_repeat: bool = True, trim_ncin: bool = True, balance_columns: bool = True,
                   merge_group_weights: bool = True, activation_ring: int = 0,
                   weight_buffer: bool = False) -> LayerSchedule:
    """Cut one layer into rounds and per-column item lists.

    ``coarse`` (the default) is ``schedule_layer_coarse``. Without it, the
    per-round schedule applies: ``merge_fills`` (one 4-D task per round for the
    four cores), ``weight_runs`` (a round's weight packets streamed by one task)
    and ``pair_drains`` (two vertically adjacent rounds drained by one task) are
    each silicon-verified bit-exact on all 66 layers; together they take the
    frame from 38.5 to 11.7 ms (docs/BENCHMARKS.md). The flags exist for bisection.
    """
    if isinstance(layer, HostLayer):
        # No DMA traffic: the runtime ends the dispatch before this layer and starts the next one after it.
        return LayerSchedule(layer.index, layer.name, [[] for _ in range(COLS)], 0, 0, 0)
    if coarse:
        # `ring_plan` returns None when the ring is off, so a flag-off build takes exactly the path it did.
        # With the ring on it returns a plan for every layer that moves packets: the ring replaces the split
        # ObjectFifo, so a layer left on the per-group schedule would fetch activations nothing ever serves.
        plan = ring_plan(ir, layer, activation_ring)
        if plan is not None:
            return schedule_layer_ring(ir, ws, layer, store, plan, trim_ncin=trim_ncin)
        return schedule_layer_coarse(ir, ws, layer, store, weight_repeat=weight_repeat, trim_ncin=trim_ncin,
                                     balance_columns=balance_columns, merge_group_weights=merge_group_weights,
                                     weight_buffer=weight_buffer)
    t = ir.tensors[layer.output]
    chunks = layer_chunks(ir, layer)
    single = len(chunks) == 1
    n_groups = (t.blocks + OUT_BLOCKS - 1) // OUT_BLOCKS
    quads = t.height // (TILE_R * ROWS)
    if t.height % (TILE_R * ROWS) or t.width % TILE_C:
        raise ValueError(f"{layer.name}: {t.height}x{t.width} is not tileable into 4x5 rows and 20 cols")
    programs: List[List[tuple]] = [[] for _ in range(COLS)]
    n_rounds = 0
    n_packets = 0
    n_w = 0
    p_out = ws.placements[layer.output]
    for g in range(n_groups):
        # Column-tile major, quads inner: consecutive rounds of a column are
        # vertically adjacent quads, so two rounds' outputs drain as one task.
        rounds = [(q, x0) for x0 in range(0, t.width, TILE_C) for q in range(quads)]
        n_r = len(rounds)
        per_col = [rounds[c * n_r // COLS:(c + 1) * n_r // COLS] for c in range(COLS)]
        for c in range(COLS):
            mine = per_col[c]
            if not mine:
                continue
            if single:
                pkt = (pool_packet(chunks[0]) if isinstance(layer, PoolLayer)
                       else conv_packet(layer, g, chunks[0], count_out=len(mine), count_acc=0))
                programs[c].append(("w", store.add(pkt), em.W_BYTES, len(mine)))
                n_w += 1
            else:
                run = round_packets(layer, g, chunks)
                run_offset = store.add_run(run)
                chunk_offsets = [run_offset + i * em.W_BYTES for i in range(len(run))]
            pending_drain: List[DmaPattern] = []
            for ri, (q, x0) in enumerate(mine):
                y_base = q * TILE_R * ROWS
                if not single and weight_runs:
                    # One task streams every chunk's weight packet of this round.
                    programs[c].append(("w", run_offset, len(chunks) * em.W_BYTES, len(chunks)))
                    n_w += 1
                for ci, ch in enumerate(chunks):
                    if not single and not weight_runs:
                        programs[c].append(("w", chunk_offsets[ci], em.W_BYTES, 1))
                        n_w += 1
                    pats = [a_pattern(ws, ir, layer, ch, y_base + TILE_R * r, x0, g) for r in range(ROWS)]
                    merged = merge_quad(pats) if merge_fills else None
                    programs[c].append(("A", merged) if merged is not None else ("a", pats))
                    n_packets += ROWS
                drain = o_pattern(ws, layer, g, y_base, x0)
                nxt = mine[ri + 1] if ri + 1 < len(mine) else None
                adjacent_next = pair_drains and nxt is not None and nxt[1] == x0 and nxt[0] == q + 1
                if adjacent_next and not pending_drain:
                    pending_drain.append(drain)   # drained together with the next round
                else:
                    if pending_drain:
                        first = pending_drain.pop()
                        drain = DmaPattern("ws", first.offset, (2 * ROWS,) + first.sizes[1:], first.strides)
                    programs[c].append(("o", drain))
                n_rounds += 1
            if pending_drain:
                raise AssertionError("unpaired pending drain")
    return LayerSchedule(layer.index, layer.name, programs, n_rounds, n_packets, n_w)


def schedule_graph(ir: GraphIR, ws: Workspace, merge_fills: bool = True, pair_drains: bool = True,
                   weight_runs: bool = True, coarse: bool = True, weight_repeat: bool = True,
                   trim_ncin: bool = True, balance_columns: bool = True,
                   merge_group_weights: bool = True,
                   activation_ring: int = 0,
                   weight_buffer: bool = False) -> Tuple[List[LayerSchedule], PacketStore]:
    store = PacketStore()
    scheds = [schedule_layer(ir, ws, L, store, merge_fills=merge_fills, pair_drains=pair_drains,
                             weight_runs=weight_runs, coarse=coarse, weight_repeat=weight_repeat,
                             trim_ncin=trim_ncin, balance_columns=balance_columns,
                             merge_group_weights=merge_group_weights, activation_ring=activation_ring,
                             weight_buffer=weight_buffer)
              for L in ir.layers]
    return scheds, store


# ----------------------------------------------------------------------------
# Packet-level emulation of a schedule on a workspace array
# ----------------------------------------------------------------------------

def emulate_layer(sched: LayerSchedule, store: PacketStore, ws_arr: np.ndarray, wp_arr: Optional[np.ndarray] = None) -> None:
    """Replay the column programs with the core emulator, updating ``ws_arr`` in place.

    Streams are replayed as the hardware sees them: fills and weight tasks
    append packets to per-column FIFOs, every four activation packets form one
    object (one packet per core), and a drain writes its objects once the cores
    have emitted them, whether it was issued before or after its fills.
    """
    blob = wp_arr
    for c, items in enumerate(sched.programs):
        states = [em.CoreState() for _ in range(ROWS)]
        w_queue: List[np.ndarray] = []     # weight objects delivered, in FIFO order
        cur_w = None
        remaining = 0                      # activation objects the current weight object still serves
        pending: List[np.ndarray] = []     # emitted output objects
        drains: List[DmaPattern] = []      # drains waiting for their objects
        packets: List[np.ndarray] = []     # activation packets not yet grouped into an object
        tile: Optional[List[np.ndarray]] = None   # a ring tile being collected, replayed by its "S" item

        def flush_drains():
            while drains:
                n_obj = drains[0].nbytes // (ROWS * em.O_BYTES)
                if len(pending) < n_obj:
                    return
                drains.pop(0).write(ws_arr, np.concatenate(pending[:n_obj]))
                del pending[:n_obj]

        def consume(stream):
            """Feed activation packets through the cores, pairing every four of them with a weight object."""
            nonlocal cur_w, remaining
            packets.extend(stream)
            while len(packets) >= ROWS:
                obj = packets[:ROWS]
                del packets[:ROWS]
                if remaining == 0:
                    if not w_queue:
                        raise AssertionError(f"{sched.name}: activation packet without a weight object")
                    cur_w = w_queue.pop(0)
                    hdr = em.unpack_w_packet(cur_w)[0]
                    remaining = hdr.count_out + hdr.count_acc
                remaining -= 1
                outs = [em.run_packet(cur_w, pkt, states[r], r) for r, pkt in enumerate(obj)]
                if all(o is not None for o in outs):
                    pending.append(np.concatenate(outs))
                elif any(o is not None for o in outs):
                    raise AssertionError("cores disagree on emission")
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
                # Inside a ring tile the fill only lands in the MemTile; the cores see it once per "S" replay.
                if tile is None:
                    consume(stream)
                else:
                    tile.extend(stream)
            elif kind == "R":
                tile = []
            elif kind == "S":
                if tile is None:
                    raise AssertionError(f"{sched.name}: a ring replay with no tile filled")
                for _ in range(it[1]):
                    consume(list(tile))
                tile = None
            elif kind == "o":
                drains.append(it[1])
                flush_drains()
        if packets:
            raise AssertionError(f"{sched.name}: {len(packets)} activation packets do not form whole objects")
        if pending or drains:
            raise AssertionError(f"{sched.name}: {len(pending)} emitted objects, {len(drains)} drains left")
        if w_queue or remaining:
            raise AssertionError(f"{sched.name}: weight objects left over ({len(w_queue)}, {remaining})")


def emulate_host_layer(ir: GraphIR, ws: Workspace, layer: HostLayer, ws_arr: np.ndarray, optimize: bool = False) -> None:
    """The host step between two dispatches on a workspace array: read the input tensor, run the layer's
    extracted model and write the output tensor's interior (its halo ring is left as planned)."""
    from ignite_xdna.compiler.graph_reference import run_host_layer
    segs = layer.input_segments()
    parts = [ws.read_tensor(ws_arr, s.tensor)[s.block_offset * 8:(s.block_offset + s.blocks) * 8] for s in segs]
    x = np.concatenate(parts, axis=0)[:layer.in_channels or ir.tensors[segs[0].tensor].channels]
    ws.write_tensor(ws_arr, layer.output, run_host_layer(layer, x, optimize=optimize))
