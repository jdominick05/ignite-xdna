"""IRON transport for the Phoenix packet-driven convolution engine.

Sixteen persistent workers on columns 0..3, rows 2..5 run the same core
program (engine.cc). Per column: one weight ObjectFifo broadcast from the shim
to the four cores, one activation ObjectFifo split at the MemTile into four
6,400-byte per-core packets, and one output ObjectFifo joined at the MemTile
from four 3,200-byte objects. The runtime sequence is supplied by the caller
so the same transport serves the engine build and every per-layer instruction
stream.
"""
from pathlib import Path

import numpy as np

from aie.dialects import arith, memref
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.ir import IndexType
from aie.iron import (Acquire, Bd, BdIteration, Buffer, DmaChannel, Flow, Lock, ObjectFifo, Program, Release,
                      Runtime, TileDma, Worker)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config

COLS = 4
ROWS = 4
CORES = COLS * ROWS

W_BYTES = 9472          # 128 header + 128 bias + 9,216 weights
A_BYTES = 6400
O_BYTES = 3200
PSUM_BYTES = 16000      # 4 x 800 int32 partial sums + 3,200-byte held tile
W_WORDS = W_BYTES // 4
PSUM_WORDS = PSUM_BYTES // 4

H_COUNT_OUT = 15
H_COUNT_ACC = 16

# Declared DDR argument extents. The buffers passed at runtime may be smaller
# or larger; the verifier only checks that every BD stays inside these.
WS_BYTES = 48 * 1024 * 1024
WP_BYTES = 16 * 1024 * 1024

KERNEL_SOURCE = Path(__file__).with_name("engine.cc")

w_ty = np.ndarray[(W_WORDS,), np.dtype[np.int32]]
a_ty = np.ndarray[(A_BYTES,), np.dtype[np.uint8]]
a_col_ty = np.ndarray[(ROWS * A_BYTES,), np.dtype[np.uint8]]
o_ty = np.ndarray[(O_BYTES,), np.dtype[np.uint8]]
o_col_ty = np.ndarray[(ROWS * O_BYTES,), np.dtype[np.uint8]]
psum_ty = np.ndarray[(PSUM_WORDS,), np.dtype[np.int32]]
ws_ty = np.ndarray[(WS_BYTES,), np.dtype[np.uint8]]
wp_ty = np.ndarray[(WP_BYTES,), np.dtype[np.uint8]]


def fifo_names(col):
    return {"w": f"w{col}", "a": f"a{col}", "o": f"o{col}"}


def _core_fn(w_in, a_in, o_out, engine, psum, scratch_out, row):
    w = w_in.acquire(1)
    idx_ty = IndexType.get()
    n_out = memref.load(w, [arith.constant(idx_ty, H_COUNT_OUT)])
    n_acc = memref.load(w, [arith.constant(idx_ty, H_COUNT_ACC)])
    n_out = arith.index_cast(idx_ty, n_out)
    n_acc = arith.index_cast(idx_ty, n_acc)
    # Packets that emit an output object (count_out), then accumulate-only packets.
    for _ in range_(n_out):
        a = a_in.acquire(1)
        o = o_out.acquire(1)
        engine(w, a, o, psum, row)
        a_in.release(1)
        o_out.release(1)
    for _ in range_(n_acc):
        a = a_in.acquire(1)
        engine(w, a, scratch_out, psum, row)
        a_in.release(1)
    w_in.release(1)


def _core_fn_ring(w_in, a_buf, a_cons, a_prod, o_out, engine, psum, scratch, row):
    """The same core program, taking activations from the MemTile ring through raw locks.

    The ring holds a tile's chunk packets so one shim fetch serves every output group of that tile. A core still
    sees one packet per chunk per group and emits one output object per group, so ``engine.cc`` and the packet
    headers are unchanged; only where the packet came from differs. One core buffer, not a pair: the ring
    supplies the pipelining, and alternating buffers inside ``range_`` would need the trip count's parity.
    """
    w = w_in.acquire(1)
    idx_ty = IndexType.get()
    n_out = memref.load(w, [arith.constant(idx_ty, H_COUNT_OUT)])
    n_acc = memref.load(w, [arith.constant(idx_ty, H_COUNT_ACC)])
    n_out = arith.index_cast(idx_ty, n_out)
    n_acc = arith.index_cast(idx_ty, n_acc)
    for _ in range_(n_out):
        a_cons.acquire(1)
        o = o_out.acquire(1)
        engine(w, a_buf, o, psum, row)
        a_prod.release(1)
        o_out.release(1)
    for _ in range_(n_acc):
        a_cons.acquire(1)
        engine(w, a_buf, scratch, psum, row)
        a_prod.release(1)
    w_in.release(1)


def ddr_extents(workspace_bytes, packet_bytes):
    """Declared (workspace, packet) DDR extents: the defaults, raised to the next MiB for larger models."""
    mib = 1 << 20
    return (max(WS_BYTES, -(-int(workspace_bytes) // mib) * mib),
            max(WP_BYTES, -(-int(packet_bytes) // mib) * mib))


# The ring is split into windows, so the next tile's fetch lands while this tile's groups are still being
# served. Its buffer descriptors are placeholders the instruction stream rewrites per layer, so their ids are
# pinned and the lowering assigns the output join's BDs around them.
RING_WINDOWS = 2
RING_BD_FILL = 0                              # one arrival BD per window, on the (even) S2MM channel 0
RING_BD_SERVE = RING_BD_FILL + RING_WINDOWS   # then one serve BD per core per window
RING_BD_ODD = 24                              # a MemTile's odd channels may only reach ids 24 and above


def _ring_bd(channel, window):
    """The pinned id of a serve BD, inside the half of the MemTile's 48 that its channel can reach.

    ``AIETargetModel::isBdChannelAccessible`` splits a MemTile's buffer descriptors by channel parity: an even
    channel may use only ids below 24, an odd channel only 24 and above. No other tile type is split. Ids are a
    per-tile resource, so the two halves are packed independently, after the arrival's.
    """
    base = RING_BD_ODD if channel % 2 else RING_BD_SERVE
    return base + (channel // 2) * RING_WINDOWS + window


def _activation_ring(col, name, slots):
    """One column's activations through a hand-written MemTile ring instead of a split ObjectFifo.

    The shim writes a tile's chunk packets into one window of the ring; each core's MM2S channel walks that
    window with a single BD, so one shim fetch serves every output group of the tile instead of one. The shim
    allocation keeps the fifo's name, so the runtime sequence's fill tasks bind unchanged.

    How many slots a window holds is the layer's chunk count, and how many times it is replayed the layer's
    output-group count, so both are written per layer from the instruction stream: ``aiex.npu.writebd`` and
    ``aiex.npu.push_queue`` reach any tile with a DMA engine, and ignite-xdna regenerates ``insts.bin`` for
    every container, so neither needs a new xclbin.

    The serve acquires no ring lock. One shared counter would overflow the 63 a lock register holds (C
    arrivals x G serves reaches 256), and per-slot locks would need one BD per slot on each of the four serve
    channels, past the MemTile's 48. Ordering comes from the instruction stream instead: ``arrived`` is a peek
    lock the serve acquires and releases by the same count - net zero, so every replay sees the same tokens
    rather than consuming them - which the runtime clears when it recycles the window, and a tile's fills wait
    on the drain of the tile two back.

    Returns ``(handles, parts)``: one ``(buffer, cons lock, prod lock)`` per core, and the flows, locks and DMA
    programs to register on the Runtime.
    """
    if slots % RING_WINDOWS:
        raise ValueError(f"a {slots}-slot ring does not divide into {RING_WINDOWS} windows")
    shim = Tile(col, 0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col, 1, tile_type=AIETileType.MemTile)
    slot_bytes = ROWS * A_BYTES
    window = slots // RING_WINDOWS
    ring_ty = np.ndarray[(slots * slot_bytes,), np.dtype[np.uint8]]
    ring = Buffer(ring_ty, name=f"{name}_ring", tile=mem)
    # One pair of locks per window, so a tile landing in one window cannot disturb the tile being served from
    # the other. A buffer descriptor that touches a lock at all must both acquire and release one, so the
    # arrival takes a slot from ``space`` and hands it to ``arrived`` rather than releasing alone; the runtime
    # restores both when it recycles the window, which is the re-arm the aie2p DMA model wants regardless.
    arrived = [Lock(mem, init=0, name=f"{name}_arrived{w}") for w in range(RING_WINDOWS)]
    space = [Lock(mem, init=window, name=f"{name}_space{w}") for w in range(RING_WINDOWS)]
    # The arrival walks its window's slots, one token per slot; the layer's chunk count replaces the iteration
    # size at runtime.
    fills = [Bd(ring, offset=w * window * slot_bytes, length=slot_bytes, bd_id=RING_BD_FILL + w,
                acquires=[Acquire(space[w], 1)], releases=[Release(arrived[w], 1)],
                iteration=BdIteration(size=window, stride=slot_bytes))
             for w in range(RING_WINDOWS)]
    # A serve sends one core's 6,400-byte slice from every slot of the window. Its acquire and release counts
    # are that slot count, written per layer and equal to each other.
    serves = [DmaChannel(DMAChannelDir.MM2S, r,
                         [Bd(ring, offset=w * window * slot_bytes + r * A_BYTES, length=A_BYTES,
                             bd_id=_ring_bd(r, w),
                             acquires=[Acquire(arrived[w], window)], releases=[Release(arrived[w], window)],
                             iteration=BdIteration(size=window, stride=slot_bytes))
                          for w in range(RING_WINDOWS)])
              for r in range(ROWS)]
    flows = [Flow(shim, mem, src_channel=0, dst_channel=0, shim_symbol=name)]
    locks = arrived + space
    tile_dmas = [TileDma(mem, [DmaChannel(DMAChannelDir.S2MM, 0, fills), *serves])]
    handles = []
    for r in range(ROWS):
        core = Tile(col, r + 2, tile_type=AIETileType.CoreTile)
        buf = Buffer(a_ty, name=f"{name}_{r}_buf", tile=core)
        c_prod = Lock(core, init=1, name=f"{name}_{r}_prod")
        c_cons = Lock(core, init=0, name=f"{name}_{r}_cons")
        tile_dmas.append(TileDma(core, [DmaChannel(DMAChannelDir.S2MM, 0,
                                                   [Bd(buf, acquires=[Acquire(c_prod, 1)],
                                                       releases=[Release(c_cons, 1)])])]))
        locks += [c_prod, c_cons]
        flows.append(Flow(mem, core, src_channel=r, dst_channel=0))
        handles.append((buf, c_cons, c_prod))
    return handles, (flows, locks, tile_dmas)


def build_program(device, sequence_body, w_depth=2, ws_bytes=WS_BYTES, wp_bytes=WP_BYTES, a_ring: int = 0):
    """Return an IRON Program for the engine with ``sequence_body(ws, wp)``.

    The body emits raw shim DMA tasks against the FIFO allocation symbols
    ``fifo_names(c)`` (see ``ignite_xdna.compiler.engine_sequence``). ``ws_bytes`` and
    ``wp_bytes`` are the declared DDR extents every task must stay inside (``ddr_extents``).
    """
    engine = ExternalFunction(
        "engine",
        source_file=str(KERNEL_SOURCE),
        arg_types=[w_ty, a_ty, o_ty, psum_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
        compile_flags=["-O2"],
    )
    workers = []
    w_prods, a_prods, o_conses = [], [], []
    rings = []                       # (flows, locks, tile dmas) of each column's ring, registered below
    for c in range(COLS):
        names = fifo_names(c)
        w_of = ObjectFifo(w_ty, name=names["w"], depth=w_depth)
        if a_ring:
            a_split, ring_parts = _activation_ring(c, names["a"], a_ring)
            rings.append(ring_parts)
        else:
            a_col = ObjectFifo(a_col_ty, name=names["a"], depth=2)
            a_split = a_col.cons().split(
                [r * A_BYTES for r in range(ROWS)],
                tile=Tile(c, 1),
                depths=[2] * ROWS,
                obj_types=[a_ty] * ROWS,
                names=[f"a{c}_{r}" for r in range(ROWS)],
            )
        o_col = ObjectFifo(o_col_ty, name=names["o"], depth=2)
        o_join = o_col.prod().join(
            [r * O_BYTES for r in range(ROWS)],
            tile=Tile(c, 1),
            depths=[2] * ROWS,
            obj_types=[o_ty] * ROWS,
            names=[f"o{c}_{r}" for r in range(ROWS)],
        )
        for r in range(ROWS):
            psum = Buffer(psum_ty, name=f"psum_{c}_{r}")
            scratch = Buffer(o_ty, name=f"scratch_{c}_{r}")
            if a_ring:
                buf, a_cons, a_prod = a_split[r]
                fn, args = _core_fn_ring, [w_of.cons(), buf, a_cons, a_prod, o_join[r].prod(),
                                           engine, psum, scratch, r]
            else:
                fn, args = _core_fn, [w_of.cons(), a_split[r].cons(), o_join[r].prod(),
                                      engine, psum, scratch, r]
            workers.append(
                Worker(
                    fn,
                    args,
                    tile=Tile(c, r + 2),
                    # engine() frames total ~0.7 KB; 2 KB leaves margin (an overflow hangs the core)
                    stack_size=0x800,
                )
            )
        w_prods.append(w_of.prod(tile=Tile(c, 0)))
        if not a_ring:
            a_prods.append(a_col.prod(tile=Tile(c, 0)))
        o_conses.append(o_col.cons(tile=Tile(c, 0)))

    def sequence(ws, wp, w_p, a_p, o_c):
        # The handles only register the shim endpoints; transfers are emitted by name.
        sequence_body(ws, wp)

    ws_t = np.ndarray[(int(ws_bytes),), np.dtype[np.uint8]]
    wp_t = np.ndarray[(int(wp_bytes),), np.dtype[np.uint8]]
    rt = Runtime(sequence, [ws_t, wp_t, w_prods, a_prods, o_conses])
    for flows, locks, tile_dmas in rings:
        for flow in flows:
            rt.add_flow(flow)
        for lock in locks:
            rt.add_lock(lock)
        for tile_dma in tile_dmas:
            rt.add_tile_dma(tile_dma)
    return Program(device, rt, workers=workers)
