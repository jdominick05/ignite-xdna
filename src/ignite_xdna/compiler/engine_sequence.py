"""Runtime-sequence emission for the convolution engine.

A column program is a list of items executed by one column's shim DMAs:

* ``("w", offset, length, serves)`` stream ``length`` bytes of weight packets from
                           the static packet buffer (``serves`` activation items
                           consume them, one object each),
* ``("a", [p0, p1, p2, p3])`` fill the activation FIFO with one 6,400-byte packet
                           per core (four DMA tasks, one per pattern),
* ``("A", pattern)``       the same as one 4-D task when the four packets are
                           regularly spaced (``merge_quad``), and
* ``("o", pattern)``       drain one 12,800-byte joined output object.

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


def merge_quad(patterns: Sequence[DmaPattern]) -> Optional[DmaPattern]:
    """One 4-D pattern covering four per-core patterns spaced by a constant byte delta, or None."""
    if len(patterns) != 4:
        return None
    p0 = patterns[0]
    if any(p.buffer != p0.buffer or p.sizes != p0.sizes or p.strides != p0.strides for p in patterns[1:]):
        return None
    if len(p0.sizes) > 3:
        return None
    delta = patterns[1].offset - p0.offset
    if delta <= 0 or delta % 4 or any(patterns[i].offset - p0.offset != i * delta for i in range(4)):
        return None
    return DmaPattern(p0.buffer, p0.offset, (4,) + tuple(p0.sizes), (delta,) + tuple(p0.strides))


def program_task_count(items: Sequence[tuple]) -> int:
    """Number of shim DMA tasks a column program issues."""
    n = 0
    for it in items:
        n += len(it[1]) if it[0] == "a" else 1
    return n


OPS_PER_TASK_ISSUE = 4  # BLOCKWRITE (BD), DDR_PATCH, MASKWRITE, WRITE (queue push); a TCT per await


def split_instruction_stream(insts: bytes, tasks_per_segment: Sequence[int]) -> List[bytes]:
    """Cut one lowered instruction stream into per-segment streams.

    The emitter awaits every task of a segment (a layer) before the next one
    starts, so a segment's ops are ``OPS_PER_TASK_ISSUE`` per task plus its
    TCT waits, and the ops are position independent. Segment boundaries are
    found by counting task issues (WRITE ops); each piece gets a fresh 16-byte
    header (the original major/minor words, its op count and its byte size).
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

    def __init__(self, ws, wp, ws_bytes: int, wp_bytes: int, fifo_names: Dict[int, Dict[str, str]]):
        from aie.dialects.aiex import dma_await_task, dma_free_task, dma_start_task, shim_dma_single_bd_task
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

    def run_column_programs(self, programs: Sequence[Sequence[tuple]], bd_budget: int = 14,
                            queue_depth: int = 4, retire_batch: int = 1) -> None:
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
        # Queue entries: [task, channel, hold]. A weight task streams one object per
        # activation item it serves and only completes once the cores consumed all
        # but the last, so it must not be awaited before those items are issued:
        # ``hold`` counts the activation items still to be issued for it.
        queues: Dict[int, List[list]] = {c: [] for c in range(len(programs))}

        def retire_channel(c: int, channel: str, count: int) -> None:
            """Retire the ``count`` oldest retirable tasks of one channel: await the
            newest of them (tasks on a channel complete in order) and free them all."""
            idx = [i for i, (_, ch, hold) in enumerate(queues[c]) if ch == channel and hold == 0][:count]
            if not idx:
                return
            self._await(queues[c][idx[-1]][0])
            for i in reversed(idx):
                task, _, _ = queues[c].pop(i)
                self._free(task)

        def ensure(c: int, channel: str) -> None:
            guard = 0
            while len(queues[c]) >= bd_budget:
                retirable = [e for e in queues[c] if e[2] == 0]
                if not retirable:
                    raise RuntimeError("every live task is held by unissued activation items")
                retire_channel(c, retirable[0][1], retire_batch)
                guard += 1
                if guard > 64:
                    raise RuntimeError("could not free buffer descriptors")
            while sum(1 for _, ch, _ in queues[c] if ch == channel) >= queue_depth:
                if not any(ch == channel and hold == 0 for _, ch, hold in queues[c]):
                    raise RuntimeError(f"channel {channel} queue is full of held tasks")
                retire_channel(c, channel, retire_batch)

        def push(c: int, channel: str, pattern: DmaPattern, hold: int = 0) -> None:
            ensure(c, channel)
            queues[c].append([self.transfer(self._names[c][channel], pattern, token=True), channel, hold])

        def served(c: int) -> None:
            for e in queues[c]:
                if e[1] == "w" and e[2] > 0:
                    e[2] -= 1
                    return

        def issue(c: int, item: tuple) -> None:
            if item[0] == "w":
                push(c, "w", linear("wp", item[1], item[2]), hold=item[3] if len(item) > 3 else 0)
            elif item[0] == "a":
                served(c)
                for p in item[1]:
                    push(c, "a", p)
            elif item[0] == "A":  # one 4-D task fills all four cores' packets
                served(c)
                push(c, "a", item[1])
            elif item[0] == "o":
                push(c, "o", item[1])
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
                retire_channel(c, queues[c][0][1], len(queues[c]))
