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
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, WireBundle
from aie.ir import IndexType, IntegerType
from aie.iron import (Acquire, Bd, BdIteration, Buffer, DmaChannel, Flow, Lock, ObjectFifo, PacketFlow, Program,
                      Release, Runtime, TileDma, Worker)
from aie.iron.dataflow.flow import PacketDest
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

# Dispatch groups engine.cc can compile out, and the opcode each one serves. Together they are
# 5,408 B of a 15,184 B object, and no container ignite-compile produces reaches any of them
# (docs/BENCHMARKS.md). The numbers must agree with engine.cc's enum and the emulator's; the
# compile path asserts that against ignite_xdna.compiler.engine_emulator rather than trusting it.
GATEABLE_OPS = {"FUSED_CONV": 4, "MUL": 5, "SCALE": 6, "POOL": 7}


def gate_flags(ops):
    """-D flags compiling out every gateable dispatch group `ops` does not use.

    `ops` is the set of opcode NUMBERS a container's packets carry (PacketStore.opcodes()).
    None gates nothing, which is what every direct caller gets: the census, the synthetic
    silicon sequence and the offline tests all build the whole kernel.

    Safe against the cache hazard that cost this repo a sitting: ExternalFunction's
    _content_digest hashes the source TEXT and its sorted compile_flags, so two gate sets
    cannot share an object. (The @iron.jit key that hashes only path+mtime is not on this
    path - the engine builds through build_program, not through a jit'd generator.)
    """
    if ops is None:
        return []
    return [f"-DENGINE_OP_{name}=0" for name, op in sorted(GATEABLE_OPS.items())
            if op not in ops]

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


def _core_fn_wbuf(w_buf, w_cons, w_prod, a_in, o_out, engine, psum, scratch_out, row):
    """The same core program, taking WEIGHTS from the MemTile buffer through raw locks.

    Today the shim re-sends a layer's whole weight run once per round with a stride-0 repeat, so a layer's
    weight traffic is its run times its tile count. The MemTile holds that run instead: the shim fetches it
    once per layer and one MM2S replays it per round, broadcast to all four cores - weights are the same
    bytes for every core, so one channel does what the activation split needed four for.

    A core still sees one weight object per chunk per round and the packet headers are unchanged, so
    ``engine.cc`` is untouched; only where the object came from differs.

    One core buffer, not a pair, for the same reason the ring uses one: alternating buffers inside ``range_``
    would need the trip count's parity. Here it also removes a correctness hazard rather than only saving
    space. With a double-buffered weight fifo the MM2S runs ahead into the spare slot whenever the core is
    the slower party, so the last round of a layer can leave a prefetched object sitting in a core - which
    the NEXT layer's first acquire would consume as its own weights. With a single buffer the serve can
    never be more than one object ahead, and the per-layer barrier (``run_column_programs`` retires every
    channel of every column, and an output drain completes only once all four cores produced that round -
    which requires them to have consumed that round's weights) makes delivery and consumption exact.
    """
    w_cons.acquire(1)
    idx_ty = IndexType.get()
    # An ObjectFifo acquire hands back an MLIR Value, which is what the other two core functions load the
    # header out of. A raw Buffer is an IRON wrapper instead, and ``memref.load`` wants the Value beneath
    # it - "Operand 0 of operation memref.load must be a Value". ExternalFunction calls DO accept the
    # wrapper, so ``engine`` below still takes ``w_buf`` itself, exactly as the ring passes ``a_buf``.
    # Unwrapped by getattr rather than ``.op`` so this keeps working if IRON ever passes a Value directly,
    # which is the same idiom the sequence emitter uses for its runtime-data handles.
    w_mem = getattr(w_buf, "op", w_buf)
    n_out = memref.load(w_mem, [arith.constant(idx_ty, H_COUNT_OUT)])
    n_acc = memref.load(w_mem, [arith.constant(idx_ty, H_COUNT_ACC)])
    n_out = arith.index_cast(idx_ty, n_out)
    n_acc = arith.index_cast(idx_ty, n_acc)
    for _ in range_(n_out):
        a = a_in.acquire(1)
        o = o_out.acquire(1)
        engine(w_buf, a, o, psum, row)
        a_in.release(1)
        o_out.release(1)
    for _ in range_(n_acc):
        a = a_in.acquire(1)
        engine(w_buf, a, scratch_out, psum, row)
        a_in.release(1)
    w_prod.release(1)


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


# The resident weight buffer. Today the shim re-sends a layer's whole weight run once per round with a
# stride-0 repeat, so a replayed layer's weight traffic is its run times its round count. Measured from the
# emitted patterns: 27,279,360 B a frame on yolov8s (1.018 ms at 26.8 GB/s) across the 19 layers that
# replay, and 9,888,768 B (0.369 ms) across yolov8n's 14. Only genuine stride-0 replays count - where
# ``merge_group_weights`` concatenates distinct groups' runs the bytes are already fetched once.
#
# 303,104 B is the largest resident set AMONG THOSE LAYERS. The largest over all layers is 1,212,416 B, but
# every layer that big replays exactly once and so never needs holding; sizing from one of those would size
# the buffer for a layer that will never use it. The MemTile's largest contiguous free run is about
# 314,880 B - an ``aie.buffer`` cannot straddle the banks the split and join already occupy - so this fits
# with 11,776 B to spare.
WBUF_BYTES = 303104
WBUF_FILL_CHANNEL = 5           # an odd S2MM; the split holds S2MM 0 and the join S2MM 1-4
WBUF_SERVE_CHANNEL = 5          # an odd MM2S; the split holds MM2S 0-3 and the join MM2S 4
# Ids clear of BOTH allocations so a build may carry the ring and this together: the join takes MemTile
# locks 0-7 and the split 8-15, while the ring takes descriptors 0-11 and 24-41 and locks 0-5 and 16-21,
# leaving descriptors 42-47. Both channels here are odd, and ``isBdChannelAccessible`` lets an odd channel
# reach only ids 24 and above.
# Up to four ARMS. A layer's columns can carry several output groups, each with its own weight run, so the
# buffer's contents change inside a layer - measured, four arms at most on yolov8s and three on yolov8n,
# and only one of yolov8s's nineteen replaying layers needs just one (worth 0.042 ms of the 1.018).
#
# Each arm therefore gets its OWN serve descriptor, and every one of them is armed at the layer boundary;
# the hardware start queue then sequences them, because a channel runs queued tasks in order. The
# alternative - rewriting one descriptor's offset between groups - is a rewrite while the channel is live,
# which is precisely the activation ring's narrowing transition and cost four protocol revisions and two
# silicon hangs. A MemTile start queue holds exactly four entries, and the fill is on a different channel
# from the serves, so four arms fit with nothing to spare.
WBUF_ARMS_MAX = 4
WBUF_BD_FILL = 42
WBUF_BD_SERVE = 43              # 43..46, one per arm; 47 stays free
WBUF_LOCK_SPACE = 32
WBUF_LOCK_READY = 33
# The buffer's L1 address as the instruction stream ASSUMES it, because the stream writes whole descriptors and
# a descriptor's word 1 holds an absolute address rather than an offset into its buffer. It is checked, not
# pinned: ``Buffer(address=0)`` fails to compile - the bank-aware allocator then places an activation buffer at
# 0x10000 inside it, and the sequential fallback fails too - while unpinned, the sequential fallback puts the
# buffer at 0. ``engine_compile`` reads the assigned address back after every build and refuses a mismatch,
# so a changed allocation is a build error rather than a descriptor pointing at the wrong bytes.
WBUF_ADDRESS = 0


def _weight_buffer(col, name):
    """One column's weights through a MemTile buffer instead of a shim broadcast.

    The shim fetches a layer's weight run once into the MemTile; one MM2S then replays it per round to all
    four cores. Weights are the SAME bytes for every core, so a single channel broadcasts where the
    activation split needed four - and the cores see the same objects in the same order, so ``engine.cc``
    and the packet headers are untouched.

    The lock discipline is the ring's, which is already proven on silicon, with one substitution: a fill
    acquires ``space`` and releases ``ready``, both by the layer's replay count, and each serve takes one
    ``ready`` and returns one ``space``, so both locks end a layer exactly where they started. The ring had
    to range-check its ``ROWS * serves`` against the 63 a MemTile lock value holds; here the count is the
    replay, whose measured maximum across both models is 16, so it never comes close.

    A net-zero serve - acquiring and releasing the same lock - was considered and is WRONG: it would prove
    only that the lock had been touched, not that the bytes this serve is about to read had landed, which is
    the second failure the ring measured ("every layer came back wrong that way, including single-chunk ones
    that had been byte-exact").

    Lengths and lock counts here are placeholders. Both vary per layer and are written by the instruction
    stream at the layer boundary, where ``run_column_programs`` has retired every channel of every column -
    the same point, and for the same reason, that the ring reconfigures at.
    """
    shim = Tile(col, 0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col, 1, tile_type=AIETileType.MemTile)
    buf_ty = np.ndarray[(WBUF_BYTES,), np.dtype[np.uint8]]
    wbuf = Buffer(buf_ty, name=f"{name}_buf", tile=mem)
    space = Lock(mem, lock_id=WBUF_LOCK_SPACE, init=1, name=f"{name}_space")
    ready = Lock(mem, lock_id=WBUF_LOCK_READY, init=0, name=f"{name}_ready")
    fill = Bd(wbuf, offset=0, length=W_BYTES, bd_id=WBUF_BD_FILL,
              acquires=[Acquire(space, 1)], releases=[Release(ready, 1)])
    # One serve per arm. Offsets and lengths are placeholders: a layer writes its own, and a layer that
    # uses fewer arms simply never pushes the rest.
    #
    # They are chained into a CYCLE purely so the lowering can reach them. IRON's ``Bd.next`` defaults to
    # "self", and a descriptor no ``aie.dma_start`` reaches is dropped from the id allocator and its id
    # handed to another fifo - so four self-linked serves compile to ONE, and pushing 44..46 would push
    # descriptors that do not exist. Measured: the first build emitted `bd_id = 43, next_bd_id = 43` and
    # nothing else, and a single layer hung because arms 1-3 never served.
    #
    # The cycle is only a compile-time device. ``_configure_wbuf`` clears Use_Next_BD on each descriptor at
    # runtime, so every serve completes and retires on its own and the queue sequences the arms.
    serves = [Bd(wbuf, offset=k * W_BYTES, length=W_BYTES, bd_id=WBUF_BD_SERVE + k,
                 acquires=[Acquire(ready, 1)], releases=[Release(space, 1)],
                 next=(k + 1) % WBUF_ARMS_MAX)
              for k in range(WBUF_ARMS_MAX)]
    flows = [Flow(shim, mem, src_channel=1, dst_channel=WBUF_FILL_CHANNEL, shim_symbol=name)]
    locks = [space, ready]
    tile_dmas = [TileDma(mem, [DmaChannel(DMAChannelDir.S2MM, WBUF_FILL_CHANNEL, [fill]),
                               DmaChannel(DMAChannelDir.MM2S, WBUF_SERVE_CHANNEL, serves)])]
    handles = []
    for r in range(ROWS):
        core = Tile(col, r + 2, tile_type=AIETileType.CoreTile)
        cbuf = Buffer(w_ty, name=f"{name}_{r}_buf", tile=core)
        c_prod = Lock(core, init=1, name=f"{name}_{r}_prod")
        c_cons = Lock(core, init=0, name=f"{name}_{r}_cons")
        # One core buffer, not the fifo's pair: see ``_core_fn_wbuf`` for why double buffering here would
        # let a prefetched object outlive a layer boundary and be consumed as the next layer's weights.
        tile_dmas.append(TileDma(core, [DmaChannel(DMAChannelDir.S2MM, 1,
                                                   [Bd(cbuf, acquires=[Acquire(c_prod, 1)],
                                                       releases=[Release(c_cons, 1)])])]))
        locks += [c_prod, c_cons]
        flows.append(Flow(mem, core, src_channel=WBUF_SERVE_CHANNEL, dst_channel=1))
        handles.append((cbuf, c_cons, c_prod))
    return handles, (flows, locks, tile_dmas)


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


# Block 5 stream sharing (compiler/stream_share.py): one activation quad in four rides each column's weight stream.
# The schedule side (which quad goes where, and the order that keeps both streams deadlock-free) is
# compiler/stream_share.py; these are the pieces the transport pins, imported there and by the emitter.
SHARE_PERIOD = 4                # quads per period of the MemTile chains
SHARE_B_POSITIONS = (3,)        # positions of each period carried by stream 1
SHARE_A_SLOTS = 6               # buffer A: two periods of its three positions
SHARE_B_SLOTS = 8               # buffer B: every stream-1 quad one weight run waits behind
SHARE_W_DEPTH = 2               # core-side weight objects: a weight task completes this far ahead of consumption
SHARE_A_IN_CHANNEL = 0          # MemTile S2MM from shim stream 0 (circuit)
SHARE_B_IN_CHANNEL = 5          # MemTile S2MM from shim stream 1 (packet); odd, clear of the join's S2MM 1-4
# Descriptor ids, pinned for the reason the ring's are: the lowering assigns the output join's twelve even and
# four odd ids around them. Even channels reach ids 0-23 and odd ones 24-47 (``isBdChannelAccessible``): A-in and
# MM2S 0 and 2 take 0-11, B-in and MM2S 1 and 3 take 24-35, which is the ring's footprint exactly.
SHARE_BD_A_IN = 0               # 0-3, one per core slice
SHARE_BD_B_IN = 24              # 24-27
SHARE_BD_SERVE_EVEN = 4         # MM2S 0: 4-7, MM2S 2: 8-11
SHARE_BD_SERVE_ODD = 28         # MM2S 1: 28-31, MM2S 3: 32-35
SHARE_LOCK_BASE = 40            # 16 MemTile locks: A space, A full, B space, B full, one per core; clear of the
                                # join (0-7), the split's old 8-15, the ring's 0-5/16-21 and the buffer's 32-33
SHARE_PKT_W, SHARE_PKT_B = 1, 2  # packet ids per column: + 8 * col (weights fan out to the cores, quads to B)


def share_packet_ids(col):
    """(weight packet id, buffer-B packet id) of one column's packet-routed weight stream."""
    return SHARE_PKT_W + 8 * col, SHARE_PKT_B + 8 * col


def _share_serve_bd(channel, k):
    base = SHARE_BD_SERVE_ODD if channel % 2 else SHARE_BD_SERVE_EVEN
    return base + (channel // 2) * SHARE_PERIOD + k


def _core_fn_shared(w_bufs, w_full, w_space, a_bufs, a_full, a_space, parity, o_out, engine, psum, scratch_out,
                    row):
    """The core program with both inputs on raw double buffers, the buffer chosen by a parity word.

    The core-side DMA alternates two descriptors (one per buffer) on a counting lock pair, exactly as an
    ObjectFifo of depth 2 lowers; the ObjectFifo's own core side knows which buffer is next by unrolling the
    loop by its depth, which a ``range_`` of runtime trip count cannot do. So the parity lives in a two-word
    buffer instead - weights in word 0, activations in word 1 - flipped after every object consumed. The DMA
    and the core advance one object at a time forever, so they stay in step across layers and frames with no
    reset. ``engine.cc`` and the packets are unchanged.
    """
    idx_ty = IndexType.get()
    i32 = IntegerType.get_signless(32)
    par = getattr(parity, "op", parity)
    c0, c1 = arith.constant(idx_ty, 0), arith.constant(idx_ty, 1)
    zero, one = arith.constant(i32, 0), arith.constant(i32, 1)

    def pick(word, bufs):
        p = memref.load(par, [word])
        b = arith.select(arith.cmpi(arith.CmpIPredicate.eq, p, zero),
                         getattr(bufs[0], "op", bufs[0]), getattr(bufs[1], "op", bufs[1]))
        memref.store(arith.xori(p, one), par, [word])
        return b

    w_full.acquire(1)
    w = pick(c0, w_bufs)
    n_out = arith.index_cast(idx_ty, memref.load(w, [arith.constant(idx_ty, H_COUNT_OUT)]))
    n_acc = arith.index_cast(idx_ty, memref.load(w, [arith.constant(idx_ty, H_COUNT_ACC)]))
    for _ in range_(n_out):
        a_full.acquire(1)
        a = pick(c1, a_bufs)
        o = o_out.acquire(1)
        engine(w, a, o, psum, row)
        a_space.release(1)
        o_out.release(1)
    for _ in range_(n_acc):
        a_full.acquire(1)
        a = pick(c1, a_bufs)
        engine(w, a, scratch_out, psum, row)
        a_space.release(1)
    w_space.release(1)


def _stream_shared(col, names):
    """One column's activation and weight inputs for stream sharing.

    Shim stream 0 carries three quads of every four, circuit-routed into MemTile buffer A. Shim stream 1 is
    packet-routed: weight objects (packet W) fan out to the four cores' S2MM 1, and the fourth quad of every
    period (packet B) lands in buffer B. Each input channel writes core k's 6,400-byte slice of a quad with
    descriptor k, so every core has its own lock pair per buffer; one ``BdIteration`` per descriptor walks the
    buffer's slots, so depth costs no descriptors. Core k's MM2S reads its slice of positions 0, 1, 2 from A and
    of position 3 from B - the cores see quads in consumption order, whichever stream brought them.

    This is ``tools/stream_share_probe.py``'s ``build_qf`` (q = 1), moved to channels the output join leaves
    free: B arrives on S2MM 5, not 1.

    Returns ``(handles, parts)``: per core ``(w_bufs, w_full, w_space, a_bufs, a_full, a_space, parity)``, and
    the flows, locks and DMA programs to register on the Runtime.
    """
    shim = Tile(col, 0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col, 1, tile_type=AIETileType.MemTile)
    cores = [Tile(col, r + 2, tile_type=AIETileType.CoreTile) for r in range(ROWS)]
    quad = ROWS * A_BYTES
    u8 = np.dtype[np.uint8]
    n_b = len(SHARE_B_POSITIONS)
    n_a = SHARE_PERIOD - n_b
    assert SHARE_A_SLOTS % n_a == 0 and SHARE_B_SLOTS % n_b == 0
    bufs = {"A": (Buffer(np.ndarray[(SHARE_A_SLOTS * quad,), u8], name=f"{names['a']}_A", tile=mem), SHARE_A_SLOTS),
            "B": (Buffer(np.ndarray[(SHARE_B_SLOTS * quad,), u8], name=f"{names['a']}_B", tile=mem), SHARE_B_SLOTS)}
    lid = iter(range(SHARE_LOCK_BASE, SHARE_LOCK_BASE + 4 * ROWS))
    space = {b: [Lock(mem, lock_id=next(lid), init=d, name=f"{names['a']}_{b}_space{k}") for k in range(ROWS)]
             for b, (_, d) in bufs.items()}
    full = {b: [Lock(mem, lock_id=next(lid), init=0, name=f"{names['a']}_{b}_full{k}") for k in range(ROWS)]
            for b in bufs}
    locks = [lk for grid in (space, full) for row in grid.values() for lk in row]
    chans = []
    for b, ch, bd0 in (("A", SHARE_A_IN_CHANNEL, SHARE_BD_A_IN), ("B", SHARE_B_IN_CHANNEL, SHARE_BD_B_IN)):
        buf, d = bufs[b]
        chans.append(DmaChannel(DMAChannelDir.S2MM, ch, [
            Bd(buf, offset=k * A_BYTES, length=A_BYTES, bd_id=bd0 + k, acquires=[Acquire(space[b][k])],
               releases=[Release(full[b][k])], iteration=BdIteration(size=d, stride=quad), next=(k + 1) % ROWS)
            for k in range(ROWS)]))
    flows = []
    for k in range(ROWS):
        chain, ia, ib = [], 0, 0
        for p in range(SHARE_PERIOD):
            if p in SHARE_B_POSITIONS:
                buf, d = bufs["B"]
                chain.append(Bd(buf, offset=ib * quad + k * A_BYTES, length=A_BYTES, bd_id=_share_serve_bd(k, p),
                                acquires=[Acquire(full["B"][k])], releases=[Release(space["B"][k])],
                                iteration=BdIteration(size=d // n_b, stride=n_b * quad), next=(p + 1) % SHARE_PERIOD))
                ib += 1
            else:
                buf, d = bufs["A"]
                chain.append(Bd(buf, offset=ia * quad + k * A_BYTES, length=A_BYTES, bd_id=_share_serve_bd(k, p),
                                acquires=[Acquire(full["A"][k])], releases=[Release(space["A"][k])],
                                iteration=BdIteration(size=d // n_a, stride=n_a * quad), next=(p + 1) % SHARE_PERIOD))
                ia += 1
        chans.append(DmaChannel(DMAChannelDir.MM2S, k, chain))
        flows.append(Flow(mem, cores[k], src_channel=k, dst_channel=0))
    tile_dmas = [TileDma(mem, chans)]
    pkt_w, pkt_b = share_packet_ids(col)
    flows.append(Flow(shim, mem, src_channel=0, dst_channel=SHARE_A_IN_CHANNEL, shim_symbol=names["a"]))
    flows.append(PacketFlow(pkt_id=pkt_w, src=shim, dst=cores[0], src_port=WireBundle.DMA, src_channel=1,
                            dst_port=WireBundle.DMA, dst_channel=1, shim_symbol=names["w"],
                            extra_dsts=[PacketDest(core, WireBundle.DMA, 1) for core in cores[1:]]))
    flows.append(PacketFlow(pkt_id=pkt_b, src=shim, dst=mem, src_port=WireBundle.DMA, src_channel=1,
                            dst_port=WireBundle.DMA, dst_channel=SHARE_B_IN_CHANNEL))
    handles = []
    for r, core in enumerate(cores):
        n = f"{names['a']}_{r}"
        a_bufs = [Buffer(a_ty, name=f"{n}_a{s}", tile=core) for s in range(2)]
        w_bufs = [Buffer(w_ty, name=f"{n}_w{s}", tile=core) for s in range(2)]
        parity = Buffer(np.ndarray[(2,), np.dtype[np.int32]], name=f"{n}_parity", tile=core,
                        initial_value=np.zeros(2, dtype=np.int32))
        a_space, a_full = Lock(core, init=2, name=f"{n}_a_space"), Lock(core, init=0, name=f"{n}_a_full")
        w_space, w_full = Lock(core, init=SHARE_W_DEPTH, name=f"{n}_w_space"), Lock(core, init=0, name=f"{n}_w_full")
        locks += [a_space, a_full, w_space, w_full]
        achain = [Bd(a_bufs[s], acquires=[Acquire(a_space)], releases=[Release(a_full)], next=1 - s)
                  for s in range(2)]
        wchain = [Bd(w_bufs[s], acquires=[Acquire(w_space)], releases=[Release(w_full)], next=1 - s)
                  for s in range(SHARE_W_DEPTH)]
        tile_dmas.append(TileDma(core, [DmaChannel(DMAChannelDir.S2MM, 0, achain),
                                        DmaChannel(DMAChannelDir.S2MM, 1, wchain)]))
        handles.append((w_bufs, w_full, w_space, a_bufs, a_full, a_space, parity))
    return handles, (flows, locks, tile_dmas)


def build_program(device, sequence_body, w_depth=2, ws_bytes=WS_BYTES, wp_bytes=WP_BYTES, a_ring: int = 0,
                  w_buf: bool = False, ops=None, a_depth: int = 2, o_depth: int = 2, stream_share: bool = False):
    """Return an IRON Program for the engine with ``sequence_body(ws, wp)``.

    The body emits raw shim DMA tasks against the FIFO allocation symbols
    ``fifo_names(c)`` (see ``ignite_xdna.compiler.engine_sequence``). ``ws_bytes`` and
    ``wp_bytes`` are the declared DDR extents every task must stay inside (``ddr_extents``).

    ``w_depth``, ``a_depth`` and ``o_depth`` are the CORE-side object counts of the weight, activation and
    output fifos (the MemTile-side split and join buffers stay at depth 2). 2 is double-buffered and the
    default; 1 frees core data memory (9,472 / 6,400 / 3,200 B per core) at the cost of overlap between
    transfer and compute. Depth 1 is an experiment (Block 5: what single-buffering costs), not a shipped mode.
    """
    engine = ExternalFunction(
        "engine",
        source_file=str(KERNEL_SOURCE),
        arg_types=[w_ty, a_ty, o_ty, psum_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
        compile_flags=["-O2"] + gate_flags(ops),
    )
    if a_ring and w_buf:
        # Both would hand the core raw locks for BOTH operands, which needs a fourth core function; and
        # measuring them together would make neither attributable, which is the mistake that left the
        # ring's costing uninterpretable. The ring is off by default and slower, so this stays unbuilt
        # until there is a reason to pay for it.
        raise ValueError("the activation ring and the resident weight buffer cannot be built together yet")
    if stream_share and (a_ring or w_buf or (w_depth, a_depth) != (2, 2)):
        raise ValueError("stream sharing replaces both input fifos; it takes no ring, weight buffer or core depths")
    workers = []
    w_prods, a_prods, o_conses = [], [], []
    rings = []                       # (flows, locks, tile dmas) of each column's ring, registered below
    wbufs = []                       # the same, for each column's resident weight buffer
    shares = []                      # the same, for each column's stream-shared inputs
    for c in range(COLS):
        names = fifo_names(c)
        if stream_share:
            share_handles, share_parts = _stream_shared(c, names)
            shares.append(share_parts)
        elif w_buf:
            w_split, wbuf_parts = _weight_buffer(c, names["w"])
            wbufs.append(wbuf_parts)
        else:
            w_of = ObjectFifo(w_ty, name=names["w"], depth=w_depth)
        if a_ring:
            a_split, ring_parts = _activation_ring(c, names["a"], a_ring)
            rings.append(ring_parts)
        elif not stream_share:
            a_col = ObjectFifo(a_col_ty, name=names["a"], depth=2)
            a_split = a_col.cons().split(
                [r * A_BYTES for r in range(ROWS)],
                tile=Tile(c, 1),
                depths=[a_depth] * ROWS,
                obj_types=[a_ty] * ROWS,
                names=[f"a{c}_{r}" for r in range(ROWS)],
            )
        o_col = ObjectFifo(o_col_ty, name=names["o"], depth=2)
        o_join = o_col.prod().join(
            [r * O_BYTES for r in range(ROWS)],
            tile=Tile(c, 1),
            depths=[o_depth] * ROWS,
            obj_types=[o_ty] * ROWS,
            names=[f"o{c}_{r}" for r in range(ROWS)],
        )
        for r in range(ROWS):
            psum = Buffer(psum_ty, name=f"psum_{c}_{r}")
            scratch = Buffer(o_ty, name=f"scratch_{c}_{r}")
            if stream_share:
                fn, args = _core_fn_shared, [*share_handles[r], o_join[r].prod(), engine, psum, scratch, r]
            elif a_ring:
                buf, a_cons, a_prod = a_split[r]
                fn, args = _core_fn_ring, [w_of.cons(), buf, a_cons, a_prod, o_join[r].prod(),
                                           engine, psum, scratch, r]
            elif w_buf:
                wbuf_r, w_cons, w_prod = w_split[r]
                fn, args = _core_fn_wbuf, [wbuf_r, w_cons, w_prod, a_split[r].cons(), o_join[r].prod(),
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
        if not (w_buf or stream_share):
            # With the buffer the shim's weight stream lands in the MemTile, not in a fifo whose producer
            # is a shim endpoint, so there is no handle to register here - the flow carries the symbol.
            w_prods.append(w_of.prod(tile=Tile(c, 0)))
        if not (a_ring or stream_share):
            a_prods.append(a_col.prod(tile=Tile(c, 0)))
        o_conses.append(o_col.cons(tile=Tile(c, 0)))

    def sequence(ws, wp, w_p, a_p, o_c):
        # The handles only register the shim endpoints; transfers are emitted by name.
        sequence_body(ws, wp)

    ws_t = np.ndarray[(int(ws_bytes),), np.dtype[np.uint8]]
    wp_t = np.ndarray[(int(wp_bytes),), np.dtype[np.uint8]]
    rt = Runtime(sequence, [ws_t, wp_t, w_prods, a_prods, o_conses])
    for flows, locks, tile_dmas in rings + wbufs + shares:
        for flow in flows:
            rt.add_flow(flow)
        for lock in locks:
            rt.add_lock(lock)
        for tile_dma in tile_dmas:
            rt.add_tile_dma(tile_dma)
    return Program(device, rt, workers=workers)
