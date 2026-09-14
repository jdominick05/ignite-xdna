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
from ignite_xdna.compiler.engine_sequence import COLS, ROWS, DmaPattern, merge_quad
from ignite_xdna.compiler.graph_ir import ConvLayer, GraphIR, PoolLayer, Segment, TensorInfo, ZP

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
        if isinstance(L, ConvLayer) and L.k == 3:
            for s in L.inputs:
                halo[s.tensor] = max(halo[s.tensor], 1)
        elif isinstance(L, PoolLayer):
            halo[L.input.tensor] = max(halo[L.input.tensor], 2)
            pooled.add(L.input.tensor)
    for name in pooled:
        if any(isinstance(L, ConvLayer) and L.k == 3 and any(s.tensor == name for s in L.inputs) for L in ir.layers):
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
    if layer.k == 1:
        return "k1up2" if seg.up2 else "k1"
    return "k3s2" if layer.stride == 2 else "k3s1"


CHUNK_GEOMETRY = {
    "k1": (5, 20, 800, 8, 8),
    "k1up2": (5, 20, 800, 8, 10),
    "k3s1": (8, 25, 1600, 4, 4),
    "k3s2": (16, 50, 6400, 1, 1),
    "pool": (16, 25, 3200, 2, 2),
    "res": (5, 20, 800, 4, 8),
}


def layer_chunks(ir: GraphIR, layer) -> List[Chunk]:
    chunks: List[Chunk] = []
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
    if chunk.kind == "pool":
        return DmaPattern("ws", p.offset(b0, y0 - 2, x0 - 2), (2, 16, 200), (p.plane_bytes, p.pitch, 1))
    raise ValueError(chunk.kind)


def o_pattern(ws: Workspace, layer, group: int, y0: int, x0: int) -> DmaPattern:
    """DMA pattern that drains one joined 12,800-byte object: four strips of four blocks."""
    p = ws.placements[layer.output]
    return DmaPattern("ws", p.offset(group * OUT_BLOCKS, y0, x0), (4, OUT_BLOCKS, TILE_R, TILE_C * 8),
                      (TILE_R * p.pitch, p.plane_bytes, p.pitch, 1))


# ----------------------------------------------------------------------------
# Weight packets
# ----------------------------------------------------------------------------

def _segment_channel_base(layer: ConvLayer, seg_index: int) -> int:
    return sum(s.blocks * 8 for s in layer.inputs[:seg_index])


def conv_packet(layer: ConvLayer, group: int, chunk: Chunk, count_out: int, count_acc: int) -> np.ndarray:
    seg = layer.inputs[chunk.seg_index]
    taps = layer.k * layer.k
    w = np.zeros((taps, chunk.ncin, OUT_BLOCKS, 8, 8), dtype=np.int8)
    cbase = _segment_channel_base(layer, chunk.seg_index) + chunk.block_start * 8
    # Channels beyond the segment's real blocks or beyond the ONNX Cin (the
    # 3-channel image is stored in one 8-channel block) get zero weights.
    cin_avail = min(chunk.ncin * 8, seg.blocks * 8 - chunk.block_start * 8, layer.cin - cbase)
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
    bias[:cout_avail] = bacc[co0:co0 + cout_avail]
    flags = 0
    if chunk.index > 0:
        flags |= em.F_LOAD_PSUM
    if chunk.last:
        flags |= em.F_HOLD if layer.residual is not None else em.F_EMIT
        if layer.act == "hswish":
            flags |= em.F_HSWISH
    if seg.up2:
        flags |= em.F_UP2
    hdr = em.PacketHeader(op=em.OP_CONV, k=layer.k, stride=layer.stride, ncin=chunk.ncin, nco=OUT_BLOCKS,
                          flags=flags, shift_out=layer.shift_out,
                          hs=layer.hswish.params if layer.hswish else None, count_out=count_out,
                          count_acc=count_acc, phases=(0, 1, 0, 1), rows_in=chunk.rows_in,
                          cols_in=chunk.cols_in, plane_bytes=chunk.plane_bytes)
    return em.pack_w_packet(hdr, bias.astype(np.int32), w)


def residual_packet(layer: ConvLayer) -> np.ndarray:
    hdr = em.PacketHeader(op=em.OP_RESIDUAL, ncin=4, nco=OUT_BLOCKS, flags=em.F_EMIT, rsh=layer.residual_shift,
                          count_out=1)
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


def schedule_layer(ir: GraphIR, ws: Workspace, layer, store: PacketStore, merge_fills: bool = True,
                   pair_drains: bool = True, weight_runs: bool = True) -> LayerSchedule:
    """Cut one layer into rounds and per-column item lists.

    ``merge_fills`` (one 4-D task per round for the four cores), ``weight_runs``
    (a round's weight packets streamed by one task) and ``pair_drains`` (two
    vertically adjacent rounds drained by one task) are each silicon-verified
    bit-exact on all 66 layers; together they take the frame from 38.5 to 11.7 ms
    (docs/BENCHMARKS.md). The flags exist for bisection.
    """
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
                run = []
                for ch in chunks:
                    if ch.kind == "res":
                        run.append(residual_packet(layer))
                    elif ch.kind == "pool":
                        run.append(pool_packet(ch))
                    else:
                        emits = ch.last and layer.residual is None
                        run.append(conv_packet(layer, g, ch, count_out=1 if emits else 0, count_acc=0 if emits else 1))
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
                   weight_runs: bool = True) -> Tuple[List[LayerSchedule], PacketStore]:
    store = PacketStore()
    scheds = [schedule_layer(ir, ws, L, store, merge_fills=merge_fills, pair_drains=pair_drains,
                             weight_runs=weight_runs) for L in ir.layers]
    return scheds, store


# ----------------------------------------------------------------------------
# Packet-level emulation of a schedule on a workspace array
# ----------------------------------------------------------------------------

def emulate_layer(sched: LayerSchedule, store: PacketStore, ws_arr: np.ndarray, wp_arr: Optional[np.ndarray] = None) -> None:
    """Replay the column programs with the core emulator, updating ``ws_arr`` in place."""
    for c, items in enumerate(sched.programs):
        states = [em.CoreState() for _ in range(ROWS)]
        w_queue: List[np.ndarray] = []     # weight objects delivered, in FIFO order
        cur_w = None
        remaining = 0                      # A packets the current weight object still serves
        pending: List[np.ndarray] = []
        for it in items:
            if it[0] == "w":
                if wp_arr is None:
                    w_queue.extend(store.packets_at(it[1], it[2]))
                else:
                    w_queue.extend(wp_arr[it[1] + k:it[1] + k + em.W_BYTES]
                                   for k in range(0, it[2], em.W_BYTES))
            elif it[0] in ("a", "A"):
                if remaining == 0:
                    if not w_queue:
                        raise AssertionError(f"{sched.name}: activation packet without a weight object")
                    cur_w = w_queue.pop(0)
                    hdr = em.unpack_w_packet(cur_w)[0]
                    remaining = hdr.count_out + hdr.count_acc
                remaining -= 1
                if it[0] == "A":
                    data = it[1].read(ws_arr)
                    packets = [data[r * em.A_BYTES:(r + 1) * em.A_BYTES] for r in range(ROWS)]
                else:
                    packets = [pat.read(ws_arr) for pat in it[1]]
                outs = []
                for r, pkt in enumerate(packets):
                    outs.append(em.run_packet(cur_w, pkt, states[r], r))
                if all(o is not None for o in outs):
                    pending.append(np.concatenate(outs))
                elif any(o is not None for o in outs):
                    raise AssertionError("cores disagree on emission")
            elif it[0] == "o":
                n_obj = it[1].nbytes // (ROWS * em.O_BYTES)
                if len(pending) < n_obj:
                    raise AssertionError(f"{sched.name}: drain of {n_obj} objects with {len(pending)} emitted")
                it[1].write(ws_arr, np.concatenate(pending[:n_obj]))
                del pending[:n_obj]
        if pending:
            raise AssertionError(f"{sched.name}: {len(pending)} emitted objects never drained")
        if w_queue or remaining:
            raise AssertionError(f"{sched.name}: weight objects left over ({len(w_queue)}, {remaining})")
