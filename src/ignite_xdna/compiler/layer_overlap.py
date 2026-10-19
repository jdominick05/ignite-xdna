# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Workspace conflicts between shim tasks of different layers, for issuing across a layer boundary.

Without it, ``SequenceEmitter.run_column_programs`` ends every layer with a barrier: every task of every column is
awaited before the next layer's first task is pushed. That is what keeps a fill from reading bytes a drain of the
layer before has not written yet, and a drain from overwriting bytes a fill still has to read (the workspace reuses
regions). It also idles the cores at every boundary: the measured tail of a layer, from its last activation task to
the next layer's first, is about 21,000 cycles with no compute behind it (docs/BENCHMARKS.md, the tail sitting).

With ``overlap_layers`` the barrier goes and each fill or drain instead waits for exactly the earlier-layer tasks
whose workspace bytes it conflicts with (read after write, write after read, write after write). This module is that
bookkeeping. Every pattern's offset, run length and stride is a multiple of 8 bytes on the engine's layouts (checked
here, not assumed), so it tracks 8-byte cells and is exact, not conservative.

Per cell and per column it keeps the last drain (writer) and the last fill (reader) that touched it, as a task
sequence number. A column's channel completes its tasks in order, so the last one on a column stands for every
earlier one on that column and channel. ``deps`` returns, per (column, channel), the newest task that must be
complete before a new task may be pushed; ``record`` enters the new task. Weight tasks read the static packet buffer
and never conflict.
"""
from typing import Dict, List, Tuple

import numpy as np

GRAIN = 8
COLS = 4


def pattern_cells(pattern) -> np.ndarray:
    """The 8-byte workspace cells a pattern touches (with repeats; order irrelevant)."""
    sizes, strides = pattern.padded()
    if pattern.offset % GRAIN or sizes[-1] % GRAIN or any(st % GRAIN for sz, st in zip(sizes[:-1], strides[:-1])
                                                          if sz > 1):
        raise ValueError(f"pattern not on the {GRAIN}-byte grain: {pattern}")
    starts = np.full(1, pattern.offset, dtype=np.int64)
    for sz, st in zip(sizes[:-1], strides[:-1]):
        starts = (starts[:, None] + np.arange(sz, dtype=np.int64)[None, :] * st).reshape(-1)
    run = np.arange(sizes[-1] // GRAIN, dtype=np.int64)
    return ((starts // GRAIN)[:, None] + run[None, :]).reshape(-1)


def workspace_patterns(scheds):
    """Every workspace pattern the schedules' fills and drains touch (weight items read the packet buffer)."""
    for s in scheds:
        for items in s.programs:
            for item in items:
                if item[0] == "a":
                    yield from item[1]
                elif item[0] in ("A", "X", "o"):
                    yield item[1]


def off_grain(scheds):
    """The first workspace pattern not on the 8-byte grain, or None. The tracker refuses such a pattern, so the
    compiler checks this before choosing overlap by default rather than failing mid-emission."""
    for p in workspace_patterns(scheds):
        try:
            pattern_cells(p)
        except ValueError:
            return p
    return None


def overlap_blockers(scheds, ir_has_host: bool, transport: bool, modes) -> list:
    """Why a build cannot overlap layers, as a list of reasons (empty means it can).

    ``modes`` maps each excluded build mode's name to whether it is on. Each such mode writes MemTile state at a
    layer boundary, or ends a dispatch there, on the assumption that the column is idle; a host layer ends the NPU
    stream at a boundary; an engine design without transport options (the bf16 one) has not been proved."""
    out = [name for name, on in modes.items() if on]
    if ir_has_host:
        out.append("host layers")
    if not transport:
        out.append("an engine design without transport options")
    if not out and off_grain(scheds) is not None:
        out.append(f"a workspace pattern off the {GRAIN}-byte grain")
    return out


class ConflictTracker:
    """Last writer and last reader per 8-byte workspace cell and column, as task sequence numbers."""

    def __init__(self, nbytes: int, cols: int = COLS):
        n = -(-int(nbytes) // GRAIN)
        self.writer = np.full((cols, n), -1, dtype=np.int32)
        self.reader = np.full((cols, n), -1, dtype=np.int32)
        self.layer_of: List[int] = []
        self._layer = None
        self._latest: Dict[Tuple[int, str], int] = {}
        self._before: Dict[Tuple[int, str], int] = {}   # newest seq per (column, channel) of the earlier layers

    def deps(self, kind: str, pattern, layer: int) -> Dict[Tuple[int, str], int]:
        """{(column, channel): newest seq that must be complete} before a task of ``kind`` ("a" fill or "o" drain)
        of ``layer`` touching ``pattern`` may be pushed.

        Conflicts inside one layer are not dependencies: the barrier design already lets them race, and they are
        over-reads - a tile's last fill reads padding rows past its input, which can land in the layer's own
        output - whose values no core uses (every container is exact). But a same-layer entry can mask an older,
        earlier-layer task on the same cell and column, so then the dependency falls back to every task the
        earlier layers queued on that column and channel."""
        if layer != self._layer:
            self._roll(layer)
        cells = pattern_cells(pattern)
        out: Dict[Tuple[int, str], int] = {}
        sources = [("o", self.writer)] + ([("a", self.reader)] if kind == "o" else [])
        for channel, table in sources:
            for c in range(table.shape[0]):
                seq = int(table[c, cells].max())
                if seq < 0:
                    continue
                if self.layer_of[seq] == layer:
                    seq = self._before.get((c, channel), -1)
                    if seq < 0:
                        continue
                out[(c, channel)] = max(out.get((c, channel), -1), seq)
        return out

    def _roll(self, layer: int) -> None:
        if self._layer is not None and layer < self._layer:
            raise ValueError(f"layer {layer} after layer {self._layer}")
        self._before = dict(self._latest)
        self._layer = layer

    def record(self, kind: str, pattern, col: int, seq: int, layer: int) -> None:
        if seq != len(self.layer_of):
            raise ValueError(f"task sequence {seq} out of order (expected {len(self.layer_of)})")
        if layer != self._layer:
            self._roll(layer)
        self.layer_of.append(layer)
        self._latest[(col, kind)] = seq
        if pattern is None:
            return
        (self.writer if kind == "o" else self.reader)[col, pattern_cells(pattern)] = seq
