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
from aie.iron import (Acquire, Bd, Buffer, DmaChannel, Flow, Lock, ObjectFifo, Program, Release,
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


# One buffer descriptor per slot, on each of five chains: the arrival, and a serve per core. That is what a
# lock per slot costs, because a descriptor's lock id is a fixed field and no descriptor can release a
# different lock on each execution.
#
# Six slots is the most the MemTile's descriptors allow. It has 48, the output join already holds 16 of them,
# and ``AIETargetModel::isBdChannelAccessible`` splits the rest by channel parity - an even channel reaches only
# ids below 24, an odd channel only 24 and above, and no other tile type is split. Five chains of six is 30, and
# the only arrangement that fits puts two chains in the even half and three in the odd: the arrival on an odd
# S2MM channel, and the serves split across MM2S 0 and 2 (even) and 1 and 3 (odd). Leaving the arrival on an
# even channel caps the window at four.
#
# The ids are pinned because the lowering assigns the output join's descriptors around them, and a descriptor no
# ``aie.dma_start`` reaches is dropped from the id allocator and its id handed to another fifo. The join needs
# twelve ids in the even half and four in the odd, so the ring takes 0-11 and 24-41 and leaves 12-23 and 42-47.
RING_SLOTS_MAX = 6
RING_FILL_CHANNEL = 5           # an odd S2MM channel; the join holds S2MM 1-4 and MM2S 4
RING_BD_SERVE_EVEN = 0          # MM2S 0 and 2
RING_BD_FILL = 24               # the arrival, in the odd half
RING_BD_SERVE_ODD = 30          # MM2S 1 and 3
# Two locks per slot, so the accounting balances itself and nothing has to be rewritten between tiles.
RING_LOCK_ARRIVED = 0           # one per slot
RING_LOCK_SPACE = 16            # one per slot, clear of the arrivals at any window size


def _ring_fill_bd(slot, slots):
    """The pinned id of the arrival descriptor that writes one slot."""
    return RING_BD_FILL + slot


def _ring_bd(channel, slot, slots):
    """The pinned id of a serve descriptor, in the half of the MemTile's 48 its channel can reach.

    Ids are a per-tile resource and the two parity halves are packed independently: the even channels (MM2S 0
    and 2) take ids from 0 upwards, the odd ones (MM2S 1 and 3) from 30, above the arrival's own six.
    """
    base = RING_BD_SERVE_ODD if channel % 2 else RING_BD_SERVE_EVEN
    return base + (channel // 2) * slots + slot


def _activation_ring(col, name, slots):
    """One column's activations through a hand-written MemTile ring instead of a split ObjectFifo.

    The shim writes a tile's chunk packets into one window of the ring; each core's MM2S channel walks that
    window with a single BD, so one shim fetch serves every output group of the tile instead of one. The shim
    allocation keeps the fifo's name, so the runtime sequence's fill tasks bind unchanged.

    How many slots a window holds is the layer's chunk count, and how many times it is replayed the layer's
    output-group count, so both are written per layer from the instruction stream: ``aiex.npu.writebd`` and
    ``aiex.npu.push_queue`` reach any tile with a DMA engine, and ignite-xdna regenerates ``insts.bin`` for
    every container, so neither needs a new xclbin.

    **A serve acquires one token of ``arrived`` and releases none**, and that shape is forced from both sides.

    Acquiring the window's whole count makes ``arrived`` a mutex: a serve holds it for the length of its
    transfer, and a transfer ends only when its core has taken the bytes, so a core that ran ahead and filled
    the output join blocks, its serve never releases, and the other three channels can never acquire. Measured
    on silicon: the first replayed layer emitted exactly two objects from core 0 - the output join's depth is
    two - one from core 1 and none from cores 2 and 3, then deadlocked. Acquiring a smaller count only narrows
    that window, because a single-chunk layer leaves one token and one blocked serve still owns it.

    Dropping the lock instead is worse, and was also measured: a shim fill task's completion token says the
    shim pushed its bytes, not that this tile's S2MM wrote them into the ring, so nothing off the tile can
    stand in for ``arrived``. Every layer came back wrong that way, including single-chunk ones that had been
    byte-exact - the serves were reading slots as they were being written.

    So the arrival hands out one token per core per replay: its release count is ``ROWS * serves``, the only
    field of it the instruction stream rewrites per layer, and each serve takes one per slice and returns none.
    Production ``slots * ROWS * serves`` matches consumption exactly, a blocked serve holds only the single
    token it took, and the count stays inside the 7-bit release field - four cores by at most eight replays is
    32. Nothing releases ``space``; the runtime restores both locks when it recycles the window.

    "Returns none" is spelled as a release of zero, not as no release at all: a descriptor that touches a lock
    must carry both a ``use_lock`` acquire and a ``use_lock`` release or the lowering rejects it, which is the
    same rule that makes the arrival hand its slot to ``arrived`` rather than releasing ``space`` alone.

    Returns ``(handles, parts)``: one ``(buffer, cons lock, prod lock)`` per core, and the flows, locks and DMA
    programs to register on the Runtime.
    """
    if not 1 <= slots <= RING_SLOTS_MAX:
        raise ValueError(f"a {slots}-slot ring needs {5 * slots} of the MemTile's 48 buffer descriptors, and "
                         f"the output join already holds 16: at most {RING_SLOTS_MAX} slots fit")
    shim = Tile(col, 0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col, 1, tile_type=AIETileType.MemTile)
    slot_bytes = ROWS * A_BYTES
    ring_ty = np.ndarray[(slots * slot_bytes,), np.dtype[np.uint8]]
    ring = Buffer(ring_ty, name=f"{name}_ring", tile=mem)
    # Two locks per slot, and the pair balances itself over a tile so nothing has to be rewritten between
    # tiles. The arrival takes ``N`` from a slot's ``space`` and hands ``N`` to its ``arrived``; each of the N
    # serves of that slot takes one ``arrived`` and gives one ``space`` back, so both locks end a tile exactly
    # where they started. ``N`` is ROWS * serves - one token per core per replay - which is at most 32 and so
    # inside the 63 a MemTile lock value holds. A single counter cannot do this: shared across the window it
    # reaches slots * ROWS * serves, which is 128 by yolov8n's layer 13 and 512 on yolov8s.
    #
    # ``N`` is the only thing the instruction stream writes, and it writes it once per layer rather than once
    # per tile: the acquire value and the release value share word 7 of a descriptor, and ``space`` is set at
    # the same layer boundary, where every task has been retired and the channel is quiescent.
    arrived = [Lock(mem, lock_id=RING_LOCK_ARRIVED + i, init=0, name=f"{name}_arrived{i}")
               for i in range(slots)]
    space = [Lock(mem, lock_id=RING_LOCK_SPACE + i, init=ROWS, name=f"{name}_space{i}")
             for i in range(slots)]
    # One arrival descriptor per slot, chained into a cycle. A cycle is one task that never completes, so the
    # hardware never clears its Valid_BD and never advances an iteration counter, and nothing has to be pushed
    # to make it run: the channel free-runs from the configuration CDO's ``aie.dma_start`` and the locks alone
    # sequence it. That is how the output join's own eight-descriptor cycle already works - it takes no runtime
    # push at all - and it is what makes an arm per tile unnecessary.
    fills = [Bd(ring, offset=i * slot_bytes, length=slot_bytes, bd_id=_ring_fill_bd(i, slots),
                acquires=[Acquire(space[i], ROWS)], releases=[Release(arrived[i], ROWS)],
                next=(i + 1) % slots)
             for i in range(slots)]
    # A serve sends one core's 6,400-byte slice from every slot of the window, taking one arrival token for it
    # and giving nothing back - see above for why acquiring the whole count deadlocks the four channels against
    # each other. The release is written explicitly as zero rather than omitted: a descriptor that touches a
    # lock must carry both ops or the lowering refuses it ("buffer descriptor with a lock must have both
    # use_lock(acquire) and use_lock(release)"), and releasing 1 instead would make the pair net zero, leaving
    # ``arrived`` permanently high so a serve would only ever prove that some slot had landed rather than the
    # one it is about to read. These counts never vary, so the instruction stream leaves these descriptors
    # alone.
    serves = [DmaChannel(DMAChannelDir.MM2S, r,
                         [Bd(ring, offset=i * slot_bytes + r * A_BYTES, length=A_BYTES,
                             bd_id=_ring_bd(r, i, slots),
                             acquires=[Acquire(arrived[i], 1)], releases=[Release(space[i], 1)],
                             next=(i + 1) % slots)
                          for i in range(slots)])
              for r in range(ROWS)]
    flows = [Flow(shim, mem, src_channel=0, dst_channel=RING_FILL_CHANNEL, shim_symbol=name)]
    locks = arrived + space
    tile_dmas = [TileDma(mem, [DmaChannel(DMAChannelDir.S2MM, RING_FILL_CHANNEL, fills), *serves])]
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
