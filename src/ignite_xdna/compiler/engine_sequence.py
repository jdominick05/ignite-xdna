"""Runtime-sequence emission for the convolution engine.

A column program is a list of items executed by one column's shim DMAs:

* ``("w", offset, length, serves)`` stream ``length`` bytes of weight packets from
                           the static packet buffer (``serves`` activation items
                           must be issued before the task may be awaited),
* ``("W", pattern, serves)`` the same for a patterned weight stream (a run of
                           packets repeated once per round),
* ``("a", [p0, p1, p2, p3])`` fill the activation FIFO with one 6,400-byte packet
                           per core (four DMA tasks, one per pattern),
* ``("A", pattern)``       one task for any number of packets that are regularly
                           spaced (``merge_quad``, ``merge_runs``), and
* ``("o", pattern[, serves])`` drain joined output objects; with ``serves`` the
                           drain is issued ahead of the fills that feed it and is
                           held until those ``serves`` activation items are issued,
* ``("R", slots, barrier, serves)`` arm the MemTile activation ring for a tile of
                           ``slots`` chunks that will be replayed ``serves`` times;
                           with ``barrier`` the column's outstanding drains are
                           awaited first, which orders the re-arm behind the serves
                           of the tile before it, and
* ``("S", serves)``        replay the armed tile once per output group it feeds.

A task has three hardware access dimensions plus a repeat dimension (the
outermost size, at most 64 on Phoenix), so one task can move up to 64
regularly spaced packets or objects.

Items are grouped into rounds (everything up to and including an output
drain). The emitter keeps two rounds in flight per column: after issuing round
``i`` it awaits the drain of round ``i - 1`` and frees that round's tasks, so at
most twelve of a shim tile's sixteen buffer descriptors are ever live. Weight
tasks that span several rounds are freed after the last drain they serve.

``DmaPattern`` is the single description of a transfer shared by the DMA and
by the NumPy emulation (``read``/``write`` move exactly the bytes the shim
would, in stream order).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

COLS = 4
ROWS = 4


@dataclass(frozen=True)
class DmaPattern:
    """Up to four-dimensional byte access pattern over a DDR buffer.

    ``sizes`` are outermost first; the innermost stride must be 1 and every
    size/stride/offset a multiple of four bytes except the innermost size,
    which must be a multiple of four bytes as well (32-bit DMA words).
    """
    buffer: str                 # "ws" (workspace) or "wp" (static packets)
    offset: int
    sizes: Tuple[int, ...]
    strides: Tuple[int, ...]

    def __post_init__(self):
        if len(self.sizes) != len(self.strides) or not 1 <= len(self.sizes) <= 4:
            raise ValueError("a pattern has one to four dimensions")
        if self.strides[-1] != 1:
            raise ValueError("innermost stride must be 1")
        if self.offset % 4 or self.sizes[-1] % 4 or any(s % 4 for s in self.strides[:-1]):
            raise ValueError(f"pattern is not 32-bit aligned: {self}")

    @property
    def nbytes(self) -> int:
        return prod(self.sizes)

    def padded(self) -> Tuple[List[int], List[int]]:
        sizes = [1] * (4 - len(self.sizes)) + list(self.sizes)
        strides = [0] * (4 - len(self.strides)) + list(self.strides)
        return sizes, strides

    def indices(self) -> np.ndarray:
        """Byte offsets in stream order."""
        sizes, strides = self.padded()
        idx = np.zeros(1, dtype=np.int64)
        for size, stride in zip(sizes, strides):
            idx = (idx[:, None] + (np.arange(size, dtype=np.int64) * stride)[None, :]).reshape(-1)
        return idx + self.offset

    def read(self, buf: np.ndarray) -> np.ndarray:
        return np.asarray(buf, dtype=np.uint8)[self.indices()]

    def write(self, buf: np.ndarray, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.uint8).reshape(-1)
        if data.size != self.nbytes:
            raise ValueError(f"pattern moves {self.nbytes} bytes, got {data.size}")
        buf[self.indices()] = data

    def tap(self, total_bytes: int):
        from aie.helpers.taplib.tap import TensorAccessPattern
        sizes, strides = self.padded()
        return TensorAccessPattern((1, total_bytes), self.offset, sizes, strides)


def linear(buffer: str, offset: int, length: int) -> DmaPattern:
    return DmaPattern(buffer, offset, (length,), (1,))


def canonical(p: DmaPattern) -> DmaPattern:
    """The same access with unit dimensions dropped and contiguous dimensions folded.

    Dimension ``i`` folds into ``i + 1`` when its stride equals the inner extent
    (``strides[i] == sizes[i + 1] * strides[i + 1]``), so the bytes and their order are
    unchanged: a tile row of a 20-column tensor without a halo, ``[5 rows][160 B]`` at
    a 160-byte pitch, becomes one 800-byte run. The compiler folds such patterns too;
    folding them first lets patterns with more dimensions merge (``merge_runs``).
    """
    sizes, strides = list(p.sizes), list(p.strides)
    changed = True
    while changed and len(sizes) > 1:
        changed = False
        for i in range(len(sizes) - 2, -1, -1):
            if sizes[i] == 1 or strides[i] == sizes[i + 1] * strides[i + 1]:
                sizes[i + 1] *= sizes[i]
                del sizes[i], strides[i]
                changed = True
                break
    if tuple(sizes) == tuple(p.sizes):
        return p
    return DmaPattern(p.buffer, p.offset, tuple(sizes), tuple(strides))


def merge_quad(patterns: Sequence[DmaPattern]) -> Optional[DmaPattern]:
    """One 4-D pattern covering four per-core patterns spaced by a constant byte delta, or None."""
    if len(patterns) != 4:
        return None
    patterns = [canonical(p) for p in patterns]
    p0 = patterns[0]
    if any(p.buffer != p0.buffer or p.sizes != p0.sizes or p.strides != p0.strides for p in patterns[1:]):
        return None
    if len(p0.sizes) > 3:
        return None
    delta = patterns[1].offset - p0.offset
    if delta <= 0 or delta % 4 or any(patterns[i].offset - p0.offset != i * delta for i in range(4)):
        return None
    return DmaPattern(p0.buffer, p0.offset, (4,) + tuple(p0.sizes), (delta,) + tuple(p0.strides))


MAX_REPEAT = 64  # the verifier's range for the repeat (outermost) dimension


def merge_runs(patterns: Sequence[DmaPattern], max_n: int = MAX_REPEAT) -> List[DmaPattern]:
    """Merge consecutive patterns of one shape spaced by a constant byte delta.

    Each maximal run (at most ``max_n`` long) of patterns with at most three
    dimensions becomes one pattern with an extra outermost dimension; patterns
    that already have four dimensions, or that break the spacing, stand alone.
    The stream order of the result is the order of ``patterns``; inputs are
    ``canonical`` first, so contiguous rows do not use up a dimension.
    """
    patterns = [canonical(p) for p in patterns]
    out: List[DmaPattern] = []
    i = 0
    while i < len(patterns):
        p0 = patterns[i]
        j = i + 1
        delta = None
        if len(p0.sizes) <= 3:
            while j < len(patterns) and j - i < max_n:
                p = patterns[j]
                if p.buffer != p0.buffer or p.sizes != p0.sizes or p.strides != p0.strides:
                    break
                d = p.offset - patterns[j - 1].offset
                if delta is None:
                    if d <= 0 or d % 4:
                        break
                    delta = d
                elif d != delta:
                    break
                j += 1
        if j - i == 1:
            out.append(p0)
        else:
            out.append(DmaPattern(p0.buffer, p0.offset, (j - i,) + tuple(p0.sizes), (delta,) + tuple(p0.strides)))
        i = j
    return out


def program_task_count(items: Sequence[tuple]) -> int:
    """Number of shim DMA tasks a column program issues."""
    n = 0
    for it in items:
        n += len(it[1]) if it[0] == "a" else 1
    return n


OPS_PER_TASK_ISSUE = 4  # BLOCKWRITE (BD), DDR_PATCH, MASKWRITE, WRITE (queue push); a TCT per await

# ---------------------------------------------------------------------------
# Arming the MemTile activation ring
#
# The ring's descriptors are compiled by kernels/aie2/conv_engine/design.py, which pins their ids and its lock
# ids; the instruction stream only has to say how many slots of a window a layer fills and how many times the
# window is replayed. Everything else - the buffer address, the offsets, the lengths, the stride - is already
# in the descriptor, so a layer changes one field.
#
# A MemTile buffer descriptor is eight 32-bit words at MEMTILE_BD_BASE + 0x20 * bd_id. Confirmed by lowering
# `aiex.npu.writebd` twice with different fields and reading the words it emits:
#   word 0  transfer length, in 32-bit words     word 1  buffer offset
#   word 6  (iteration_size << 17) | iteration_stride     word 7  bit 31 = valid
# `iteration_size` is the hardware's 6-bit Iteration_Wrap and is stored off by one, so a tile of C slots writes
# C - 1. One maskwrite sets it and leaves the compiled stride and address alone.
#   word 7  bit 31 valid, bits 24-30 lock_rel_val, 16-23 lock_rel_id, 15 lock_acq_enable, 8-14 lock_acq_val,
#           0-7 lock_acq_id
# The arrival takes a slot from `space` and hands `ROWS * serves` tokens to `arrived`, one per core per replay,
# and that release count is the only lock field the instruction stream rewrites. A serve acquires one of those
# and releases none, so its word-7 lock fields never vary and are left exactly as compiled. An acquire is
# stored NEGATED - AcquireGreaterEqual N is -N in the 7-bit field - and a release positive; the values the
# configuration CDO wrote confirm it, an acquire of 8 appearing as 0x78. The release field has no enable bit,
# unlike the acquire, so a descriptor that must release nothing needs an explicit zero there, not an omission.
#
# Word 1 carries Next_BD in bits 25:20 and Use_Next_BD in bit 19, above the buffer address in bits 18:0. Every
# compiled ring descriptor self-links, because IRON's ``Bd.next`` defaults to "self" - the streaming pattern an
# objectFIFO wants. A chain with Use_Next_BD set is one task that never completes, so its channel never
# advances to the next entry of its queue and a runtime push only stacks behind it. Clearing that one bit is
# what makes a pushed descriptor run once and retire.
#
# Iteration is an address modifier, not a repeat count: execution k adds ((current + k) % wrap) * stepsize, and
# one lock acquire and one release fire per execution. How many executions happen comes from the queue push,
# whose repeat count register holds one less than the number asked for.
RING_BD_STRIDE = 0x20
RING_BD_NEXT_WORD = 0x04
RING_BD_ITERATION_WORD = 0x18
RING_BD_LOCK_WORD = 0x1C
RING_USE_NEXT_BD = 1 << 19
# Two descriptor fields are consumed by running it: Valid_BD is cleared when a task completes, and
# Iteration_Current advances as the BD executes. Both must be restored before the descriptor is pushed again.
RING_VALID_BD = 1 << 31
RING_ITERATION_SHIFT = 17
RING_ITERATION_MASK = 0x3F << RING_ITERATION_SHIFT
RING_ITERATION_CURRENT_SHIFT = 23
RING_ITERATION_CURRENT_MASK = 0x3F << RING_ITERATION_CURRENT_SHIFT
RING_LOCK_REL_SHIFT = 24
RING_LOCK_MASK = 0x7F
RING_LOCK_REL_VALUE_MASK = RING_LOCK_MASK << RING_LOCK_REL_SHIFT
# The acquire value sits in the same word and is stored NEGATED in seven bits: AcquireGreaterEqual N is -N,
# which is why the configuration CDO writes an acquire of 8 as 0x78. Both values live in word 7, so a layer
# that changes how many tokens a slot carries rewrites them in one masked write.
RING_LOCK_ACQ_SHIFT = 8
RING_LOCK_ACQ_VALUE_MASK = RING_LOCK_MASK << RING_LOCK_ACQ_SHIFT
# Next_BD sits above the buffer address in word 1, with Use_Next_BD one bit below it. Relinking a cycle to a
# shorter one is a write to these two fields and nothing else.
RING_NEXT_BD_SHIFT = 20
RING_NEXT_BD_MASK = 0x3F << RING_NEXT_BD_SHIFT

# The head of a free-running chain already holds its acquire.
#
# A cyclic chain is never idle: one descriptor is always the channel's current one, and a descriptor takes its
# lock when it becomes current rather than when stream bytes arrive for it. AMD's programming guide says so -
# locks are "acquired before the transfer starts and released after it completes" - and the S2MM status register
# corroborates it structurally by carrying two different stall states, Stalled_Lock_Acq and
# Stalled_Stream_Starvation, which a channel acquiring only on data arrival would not need. For a producer lock
# at its replenished resting value, which is exactly what ``space`` is here, the acquire SUCCEEDED: the head has
# already drawn its slot down and is parked on the stream.
#
# So crediting every slot alike hands the DMA tokens it is still holding, and the arrival runs a lap ahead of
# the serves still reading that slot. Measured: a one-layer container built with this False returned layer 0
# with 45,459 of 1,638,400 bytes wrong - computed, nearly right, sporadically overwritten - where per-tile
# arming had been byte-exact.
#
# Only ``space`` is affected, and the asymmetry is the tell: ``space`` rests high so a parked acquire succeeds,
# while ``arrived`` rests at zero so a parked serve blocks without taking anything. Per-tile arming never met
# this because a completed task leaves the channel idle with no current descriptor holding anything.
RING_HEAD_PARKED = True
MEMTILE_ROW = 1
# A MemTile's DMA channel control registers, confirmed against both builds' configuration CDO: six channels
# per direction at a stride of eight, S2MM first, each with its START_QUEUE in the word above. Bit 1 of the
# control register is the channel reset; an aie2p channel has no enable bit at all.
MEMTILE_S2MM_CTRL = 0xA0600
MEMTILE_MM2S_CTRL = 0xA0630
MEMTILE_CHANNEL_STRIDE = 8
MEMTILE_CHANNEL_RESET = 1 << 1


def split_instruction_stream(insts: bytes, tasks_per_segment: Sequence[int]) -> List[bytes]:
    """Cut one lowered instruction stream into per-segment streams.

    The emitter retires every task of a segment (a layer) before the next one
    starts, so a segment's ops are ``OPS_PER_TASK_ISSUE`` per task plus its
    TCT waits, and the ops are position independent. Segment boundaries are
    found by counting task issues (WRITE ops); each piece gets a fresh 16-byte
    header (the original major/minor words, its op count and its byte size).
    A task without a completion token is a MASKWRITE short (its issue is then
    three ops), which this count-by-push method tolerates.
    """
    import struct
    from ignite_xdna.compiler.scheduler import TXN_HEADER_BYTES, parse_transaction_stream
    ops = parse_transaction_stream(insts)
    major, minor, num_ops, size = struct.unpack("<4I", insts[:TXN_HEADER_BYTES])
    pushes = sum(1 for o in ops if o["op"] == "WRITE")
    if pushes != sum(tasks_per_segment):
        raise ValueError(f"stream has {pushes} task pushes, schedule implies {sum(tasks_per_segment)}")
    pieces: List[bytes] = []
    cursor = 0
    for n_tasks in tasks_per_segment:
        # Consume ops until this segment's last push and the TCT waits that follow it.
        seen = 0
        end_idx = cursor
        while end_idx < len(ops) and seen < n_tasks:
            if ops[end_idx]["op"] == "WRITE":
                seen += 1
            end_idx += 1
        while end_idx < len(ops) and ops[end_idx]["op"] == "TCT":
            end_idx += 1
        first, last = ops[cursor], ops[end_idx - 1]
        start, end = first["offset"], last["offset"] + last["size"]
        body = insts[start:end]
        pieces.append(struct.pack("<4I", major, minor, end_idx - cursor, TXN_HEADER_BYTES + len(body)) + body)
        cursor = end_idx
    if cursor != len(ops):
        raise ValueError(f"{len(ops) - cursor} ops left after the last segment")
    return pieces


def split_rounds(items: Sequence[tuple]) -> List[List[tuple]]:
    """Group a column program into rounds ending at each output drain."""
    rounds, cur = [], []
    for it in items:
        cur.append(it)
        if it[0] == "o":
            rounds.append(cur)
            cur = []
    if cur:
        rounds.append(cur)
    return rounds


class SequenceEmitter:
    """Emit raw shim DMA tasks inside an IRON runtime sequence body."""

    def __init__(self, ws, wp, ws_bytes: int, wp_bytes: int, fifo_names: Dict[int, Dict[str, str]],
                 a_ring: int = 0):
        from aie.dialects.aiex import (dma_await_task, dma_free_task, dma_start_task, npu_maskwrite32,
                                       npu_push_queue, npu_write32, shim_dma_single_bd_task)
        from aie.dialects._aie_enum_gen import DMAChannelDir
        self._maskwrite = npu_maskwrite32
        self._write32 = npu_write32
        self._push_queue = npu_push_queue
        self._dir = DMAChannelDir
        # The ring's descriptor and lock ids are pinned by the design that compiles them, and read from it here
        # rather than restated, so the two cannot drift: a maskwrite to the wrong descriptor is silent.
        from kernels.aie2.conv_engine.design import (ROWS as RING_ROWS, RING_FILL_CHANNEL, RING_LOCK_ARRIVED,
                                                     RING_LOCK_SPACE, _ring_bd, _ring_fill_bd)
        from .scheduler import MEMTILE_BD_BASE, memtile_lock_reg
        self._ring_rows = RING_ROWS
        # How many slots the design was compiled with, which fixes the descriptor ids: a layer's pass may be
        # narrower than this, but never wider.
        self._ring_slots = a_ring
        self._ring_fill_channel = RING_FILL_CHANNEL
        self._ring_fill = _ring_fill_bd
        self._ring_arrived = RING_LOCK_ARRIVED
        self._ring_space = RING_LOCK_SPACE
        self._ring_serve = _ring_bd
        self._ring_bd_base = MEMTILE_BD_BASE
        self._ring_lock_reg = memtile_lock_reg
        self._ws, self._wp = ws, wp
        self._bytes = {"ws": ws_bytes, "wp": wp_bytes}
        self._names = fifo_names
        self._single = shim_dma_single_bd_task
        self._start = dma_start_task
        self._await = dma_await_task
        self._free = dma_free_task

    def _mem(self, pattern: DmaPattern):
        mem = self._ws if pattern.buffer == "ws" else self._wp
        # IRON hands the sequence body RuntimeData wrappers; the BD wants the MLIR value.
        return getattr(mem, "op", mem)

    def transfer(self, alloc: str, pattern: DmaPattern, token: bool):
        task = self._single(alloc, self._mem(pattern), tap=pattern.tap(self._bytes[pattern.buffer]),
                            issue_token=token)
        self._start(task)
        return task

    def _ring_chains(self, slots: int):
        """The ring's five descriptor chains, each a slot-ordered list of ids: the arrival, then a serve per core.

        Nothing resets these channels any more. They are started once by the configuration CDO and left running:
        a chain whose descriptors link back to their own head is one task that never completes, so the hardware
        never clears a Valid_BD, never advances an iteration counter and never empties a start queue. That is
        exactly how the output join's own eight-descriptor cycle runs - it takes no runtime push at all - and it
        is what lets a layer configure the ring once instead of arming it once per tile.
        """
        yield [self._ring_fill(i, slots) for i in range(slots)]
        for r in range(self._ring_rows):
            yield [self._ring_serve(r, i, slots) for i in range(slots)]

    def _configure_ring(self, col: int, width: int, serves: int, slots: int) -> None:
        """Point one column's ring at a layer's geometry - once per layer, not once per tile.

        Arming used to be per tile, and it had to be: a pushed task consumes its descriptor, because the
        hardware clears ``Valid_BD`` when the task completes and advances ``Iteration_Current`` as it runs, so
        every tile rewrote all five descriptors and pushed them again. Nineteen register writes a tile, 1,104
        tiles a frame on yolov8n and 1,853 on yolov8s: measured against the flag-off container, the ring's
        instruction stream carried 45,521 ops to its 12,258, and that gap was larger than every byte the ring
        saved.

        Nothing is pushed here and nothing is reset. The descriptors form cycles, so each chain is one task
        that never completes, never clears a Valid_BD and never empties a start queue - it free-runs from the
        configuration CDO's ``aie.dma_start`` with the locks alone sequencing it, exactly as the output join's
        own eight-descriptor cycle already does without a single runtime push.

        What is left varies per layer and not per tile:

        * **How many tokens a slot carries.** The arrival hands a slot ``ROWS * serves`` - one per core per
          replay - and each of those serves takes one back, so both of a slot's locks end a tile where they
          started and no tile has to restore them. The acquire and release values share word 7, and the
          acquire is stored negated, so one masked write sets both.
        * **``space``**, which must start at the same count - except at the head of the cycle, which never
          starts there. See ``RING_HEAD_PARKED``: a free-running chain always has a current descriptor, and
          that descriptor has already drawn its slot's ``space`` down and is parked waiting for bytes. It is
          written here and nowhere else, because a layer boundary is the one point where every drain has been
          awaited (``run_column_programs`` retires every channel of every column before returning, and this
          runs once per layer), so the cores have consumed every slice the serves delivered and nothing but
          that one parked descriptor is live.
        * **The width of each cycle**, when the layer's pass is narrower than the compiled window. Only the
          wrap matters, so each chain is relinked slot by slot with the last pointing back at the head.
        """
        tokens = self._ring_rows * serves
        if tokens > RING_LOCK_MASK:
            raise ValueError(f"a tile replayed {serves} times needs {tokens} tokens per slot, past the "
                             f"{RING_LOCK_MASK} a 7-bit lock field holds")
        for chain, ids in enumerate(self._ring_chains(slots)):
            for i in range(width):
                bd = self._ring_bd_base + RING_BD_STRIDE * ids[i]
                # Close the cycle at this layer's width: the last descriptor of the pass points back at the
                # first, every other at its successor.
                self._maskwrite(bd + RING_BD_NEXT_WORD,
                                RING_USE_NEXT_BD | (ids[(i + 1) % width] << RING_NEXT_BD_SHIFT),
                                RING_USE_NEXT_BD | RING_NEXT_BD_MASK, column=col, row=MEMTILE_ROW)
            if chain:
                continue
            # Only the arrival's counts move with the layer; a serve always takes one and gives one back.
            for i in range(width):
                bd = self._ring_bd_base + RING_BD_STRIDE * ids[i]
                self._maskwrite(bd + RING_BD_LOCK_WORD,
                                (tokens << RING_LOCK_REL_SHIFT) | ((-tokens & RING_LOCK_MASK)
                                                                   << RING_LOCK_ACQ_SHIFT),
                                RING_LOCK_REL_VALUE_MASK | RING_LOCK_ACQ_VALUE_MASK,
                                column=col, row=MEMTILE_ROW)
                # The head of the cycle is parked holding this slot's space, so it starts from nothing: its own
                # serves credit the slot back up to ``tokens`` once its fill lands. Every pass is the full
                # width - the window is narrowed to a divisor of the tile - so the chain returns to ids[0]
                # after every tile, and the parked slot is always this one.
                credit = 0 if (RING_HEAD_PARKED and i == 0) else tokens
                self._write32(self._ring_lock_reg(self._ring_space + i), credit,
                              column=col, row=MEMTILE_ROW)

    def run_column_programs(self, programs: Sequence[Sequence[tuple]], bd_budget: int = 14,
                            queue_depth: int = 4, retire_batch: int = 2) -> None:
        """Issue every column's items interleaved within two hardware limits per shim.

        Every task carries a completion token. Before a new task is configured,
        the oldest outstanding tasks are awaited and freed until (a) at most
        ``bd_budget`` buffer descriptors are live on the shim tile and (b) at
        most ``queue_depth`` tasks are pending on the target DMA channel (the
        start queue of a shim channel holds four entries; pushing a fifth
        while activation fills wait on the cores stalls the stream, which is
        how 16-chunk rounds hung). Awaiting a fill blocks until the MemTile
        accepted the packet, which is the backpressure the cores need; a
        round's drain is always issued before any later round's fills, so no
        core ever waits for an output object that has no drain queued.
        """
        # Queue entries: [task, channel, hold, token]. Every ``dma_await_task``
        # consumes exactly one completion token of its channel, so a task that
        # carries a token must be awaited exactly once, in issue order; tasks
        # without a token are freed on the strength of a later awaited task on
        # the same channel (a channel completes its tasks in order). Tokens go to
        # every ``retire_batch``-th task of a channel and to its last task in
        # this program, so no token is ever left unconsumed at the layer barrier.
        # A weight task only completes once the cores consumed (all but the last
        # of) its objects, and a drain issued ahead of its fills only completes
        # once those fills were consumed, so neither may be awaited before the
        # activation items it serves are issued: ``hold`` counts them.
        queues: Dict[int, List[list]] = {c: [] for c in range(len(programs))}
        totals: Dict[int, Dict[str, int]] = {}
        for c, prog in enumerate(programs):
            t = {"w": 0, "a": 0, "o": 0}
            for it in prog:
                if it[0] in ("w", "W"):
                    t["w"] += 1
                elif it[0] == "a":
                    t["a"] += len(it[1])
                elif it[0] == "A":
                    t["a"] += 1
                elif it[0] == "o":
                    t["o"] += 1
            totals[c] = t
        issued: Dict[int, Dict[str, int]] = {c: {"w": 0, "a": 0, "o": 0} for c in range(len(programs))}

        def retire_channel(c: int, channel: str) -> bool:
            """Retire the oldest token group of one channel (its tasks up to and
            including the first tokened one); False if a held task blocks it."""
            group = []
            for i, (_, ch, hold, token) in enumerate(queues[c]):
                if ch != channel:
                    continue
                if hold:
                    return False
                group.append(i)
                if token:
                    break
            if not group or not queues[c][group[-1]][3]:
                return False
            self._await(queues[c][group[-1]][0])
            for i in reversed(group):
                self._free(queues[c].pop(i)[0])
            return True

        def ensure(c: int, channel: str) -> None:
            guard = 0
            while len(queues[c]) >= bd_budget:
                if not any(retire_channel(c, ch) for ch in ("w", "a", "o")):
                    raise RuntimeError("every live task is held or lacks an awaitable token")
                guard += 1
                if guard > 64:
                    raise RuntimeError("could not free buffer descriptors")
            while sum(1 for _, ch, _, _ in queues[c] if ch == channel) >= queue_depth:
                if not retire_channel(c, channel):
                    raise RuntimeError(f"channel {channel} queue is full of held tasks")

        def push(c: int, channel: str, pattern: DmaPattern, hold: int = 0, force_token: bool = False) -> None:
            ensure(c, channel)
            issued[c][channel] += 1
            n = issued[c][channel]
            token = force_token or n == totals[c][channel] or n % retire_batch == 0
            queues[c].append([self.transfer(self._names[c][channel], pattern, token=token), channel, hold, token])

        def served(c: int) -> None:
            # One activation item releases one unit of the oldest held weight task
            # and of the oldest held drain.
            for channel in ("w", "o"):
                for e in queues[c]:
                    if e[1] == channel and e[2] > 0:
                        e[2] -= 1
                        break

        # The shape each column's ring was configured with, for as long as this call - and this is called once
        # per layer, so a column configures its ring once and every later tile of that layer finds it done.
        # There is no barrier and nothing to re-arm: the descriptors free-run as cycles, so a later tile has
        # nothing to overtake.
        configured: Dict[int, tuple] = {}

        def issue(c: int, item: tuple) -> None:
            if item[0] == "w":
                push(c, "w", linear("wp", item[1], item[2]), hold=item[3] if len(item) > 3 else 0)
            elif item[0] == "W":
                push(c, "w", item[1], hold=item[2])
            elif item[0] == "a":
                served(c)
                for p in item[1]:
                    push(c, "a", p)
            elif item[0] == "A":  # one 4-D task fills all four cores' packets
                served(c)
                push(c, "a", item[1])
            elif item[0] == "o":
                push(c, "o", item[1], hold=item[2] if len(item) > 2 else 0)
            elif item[0] == "R":
                # The first tile of a layer configures this column's ring; the rest of the layer's tiles find
                # it configured and cost nothing at all. Every tile of a layer and column carries the same
                # shape - the window is narrowed to a divisor of the tile so the cycles never have to be
                # relinked mid-layer, and a column's replay count is a property of the layer - but a schedule
                # that broke that would corrupt the tile in flight silently, so it is checked rather than
                # assumed.
                shape = (item[1], item[3] if len(item) > 3 else 1)
                if c not in configured:
                    configured[c] = shape
                    self._configure_ring(c, shape[0], shape[1], self._ring_slots)
                elif configured[c] != shape:
                    raise ValueError(f"column {c} arms {shape} after {configured[c]} inside one layer; the "
                                     f"ring is configured once per layer and cannot change shape mid-layer")
            elif item[0] == "S":
                pass                  # the serves free-run on their locks; nothing is pushed for them
            else:
                raise ValueError(item[0])

        n = max((len(p) for p in programs), default=0)
        for i in range(n):
            for c, prog in enumerate(programs):
                if i < len(prog):
                    issue(c, prog[i])
        for c in range(len(programs)):
            for e in queues[c]:
                e[2] = 0  # every item has been issued
            while queues[c]:
                if not retire_channel(c, queues[c][0][1]):
                    raise RuntimeError("layer barrier: a channel's newest task carries no token")
