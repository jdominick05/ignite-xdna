# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Box fills: the shim moves one tile box, and the MemTile cuts it into the four cores' objects.

This is AMD's split of roles (docs/BENCHMARKS.md, "Block 5: AMD's split of roles, built as a probe"). Today a
one-quad activation fill is a 4-D shim pattern with the core index on its repeat, e.g. ``(4, 4, 8, 200)`` at
``(3280, 53792, 656, 1)``: core j takes 4 planes x rows 5j..5j+7. That leaves no dimension for a second quad, so
every such quad is its own shim task. The union of the four slices is one 3-D box (here 4 planes x 23 rows x
200 B), which the shim can move in one pattern - and boxes that follow each other merge by the repeat. The
MemTile then cuts core j's object out of the box with a 3-D read from ``step * W * j``.

A layer qualifies only when EVERY activation fill it issues is a box of one geometry, because the MemTile's
descriptors are configured once per layer (``SequenceEmitter._configure_box``). Every other layer runs through
the same hand-written MemTile path cut as plain quads, so a container built with box fills uses the hand-written
path for all of its layers.

Fills merge only when they are adjacent items of a column program; a weight or drain item in between ends the run.
Moving fills past those items would change what the MemTile must hold and is not done here.

Every converted fill is checked here, at compile time, to deliver byte for byte the same per-core objects in the
same order as the fill it replaces, so a geometry this module gets wrong is a compile error, not a wrong frame.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ignite_xdna.compiler import engine_emulator as em
from ignite_xdna.compiler.engine_sequence import MAX_REPEAT, DmaPattern, canonical, merge_runs

ROWS = 4
QUAD_BYTES = ROWS * em.A_BYTES
# Boxes merged into one shim task at most. The probe found no gain past two boxes per task at this traffic
# (k2 3.079 / k8 3.133 / k64 3.095 ms), so this only bounds how far one task may run ahead of the cores.
DEFAULT_CAP = 8


@dataclass(frozen=True)
class BoxGeometry:
    """One layer's box: ``planes`` x ``span`` rows x ``width`` bytes in the MemTile, plane-major.

    Core j's object is rows ``step * j .. step * j + rows - 1`` of every plane.
    """
    planes: int
    span: int
    width: int
    step: int
    rows: int

    @property
    def box_bytes(self) -> int:
        return self.planes * self.span * self.width

    def core_read(self, j: int) -> Tuple[int, Tuple[int, int, int], Tuple[int, int, int]]:
        """(offset, sizes, strides) in bytes of core j's object inside one box, outermost first."""
        return (self.step * self.width * j, (self.planes, self.rows, self.width),
                (self.span * self.width, self.width, 1))

    def last_byte_end(self, j: int) -> int:
        """One past the last box byte core j reads."""
        off, (p, r, w), (sp, sr, _) = self.core_read(j)
        return off + (p - 1) * sp + (r - 1) * sr + w

    def cut(self, box: np.ndarray) -> List[np.ndarray]:
        """The four core objects of one box, as the MemTile's serves read them."""
        out = []
        for j in range(ROWS):
            off, sizes, strides = self.core_read(j)
            idx = DmaPattern("ws", off, sizes, strides).indices()
            out.append(box[idx])
        return out


# A plain quad as a box: core j's 6,400 B at 6,400 j. A box-fills container cuts every unconverted layer this way.
# design.BOX_PLAIN must equal it; engine_compile checks the two before a build.
PLAIN = BoxGeometry(1, ROWS * 32, em.A_BYTES // 32, 32, 32)


def geometry_of(p: DmaPattern) -> Optional[BoxGeometry]:
    """The box geometry of a one-quad fill whose core index sits on its outer dimension, else None."""
    c = canonical(p)
    if len(c.sizes) != 4 or c.nbytes != QUAD_BYTES or c.sizes[0] != ROWS:
        return None
    _, planes, rows, width = c.sizes
    s_core, s_plane, s_row, unit = c.strides
    if unit != 1 or s_core % s_row or s_core <= 0:
        return None
    step = s_core // s_row
    # Planes may overlap in DDR (a pooling window's do): the box then reads those rows once per plane, and the
    # MemTile holds each plane's copy. ``_check`` is what decides whether the cut is right.
    return BoxGeometry(planes, (ROWS - 1) * step + rows, width, step, rows)


def box_pattern(p: DmaPattern, g: BoxGeometry) -> DmaPattern:
    """The 3-D DDR pattern of a fill's box, in the MemTile's plane-major order."""
    c = canonical(p)
    return DmaPattern(c.buffer, c.offset, (g.planes, g.span, g.width), (c.strides[1], c.strides[2], 1))


def _check(p: DmaPattern, g: BoxGeometry) -> None:
    """Refuse a conversion unless the cut box delivers exactly the fill's bytes, in its order."""
    want = canonical(p).indices()
    box = box_pattern(p, g).indices()
    got = np.concatenate(g.cut(box))
    if not np.array_equal(got, want):
        raise AssertionError(f"box {g} of fill {p} does not reproduce its per-core objects")


def _fills(item: tuple) -> Optional[List[DmaPattern]]:
    if item[0] == "a":
        return list(item[1])
    if item[0] == "A":
        return [item[1]]
    return None


def layer_geometry(programs: Sequence[Sequence[tuple]]) -> Optional[BoxGeometry]:
    """The single box geometry every activation fill of a layer shares, or None."""
    geom = None
    for prog in programs:
        for it in prog:
            if it[0] in ("R", "S", "P", "X"):
                return None              # ring, stream sharing or already boxed: not this transform's input
            fills = _fills(it)
            if fills is None:
                continue
            for p in fills:
                g = geometry_of(p)
                if g is None or (geom is not None and g != geom):
                    return None
                geom = g
    return geom


def box_programs(programs: Sequence[Sequence[tuple]], g: BoxGeometry,
                 cap: int = DEFAULT_CAP) -> List[List[tuple]]:
    """Rewrite one layer's column programs: each run of adjacent fills becomes ``("X", pattern, boxes)`` items.

    ``pattern`` is a box, or ``boxes`` of them merged by an outer (repeat) dimension. A fill that cannot merge
    with a neighbour stays a single box.
    """
    if not 1 <= cap <= MAX_REPEAT:
        raise ValueError(f"cap {cap} outside 1..{MAX_REPEAT}")
    out = []
    for prog in programs:
        new: List[tuple] = []
        run: List[DmaPattern] = []

        def flush():
            for m in merge_runs(run, max_n=cap):
                new.append(("X", m, m.sizes[0] if len(m.sizes) == 4 else 1))
            run.clear()

        for it in prog:
            fills = _fills(it)
            if fills is None:
                flush()
                new.append(it)
                continue
            for p in fills:
                _check(p, g)
                run.append(box_pattern(p, g))
        flush()
        out.append(new)
    return out


def apply_box_fills(scheds, cap: int = DEFAULT_CAP) -> int:
    """Convert every qualifying layer in place (``sched.box`` set, programs rewritten); returns how many."""
    n = 0
    for s in scheds:
        g = layer_geometry(s.programs)
        if g is None:
            continue
        s.programs = box_programs(s.programs, g, cap)
        s.box = g
        n += 1
    return n


def box_packets(item: tuple, g: BoxGeometry, ws_arr: np.ndarray) -> List[np.ndarray]:
    """The activation packets an ``("X", pattern, boxes)`` item delivers, in core-consumption order."""
    data = item[1].read(ws_arr)
    if data.size != item[2] * g.box_bytes:
        raise AssertionError(f"box item moves {data.size} B, expected {item[2]} x {g.box_bytes}")
    stream = []
    for b in range(item[2]):
        stream.extend(g.cut(data[b * g.box_bytes:(b + 1) * g.box_bytes]))
    return stream
