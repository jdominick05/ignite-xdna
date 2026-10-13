# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Block 5 stream sharing: carry one activation quad in four on each column's weight stream.

The engine's activations reach a column on one shim MM2S stream (~6.9 GB/s), while the column's second stream
carries only weights (docs/BENCHMARKS.md, the stream ceiling). Here the quad at position 3 of every 4 in a
column's consumption order travels on the weight stream instead, packet-routed into a second MemTile buffer (B);
the other three keep stream 0 into buffer A. Every core's MemTile MM2S chain reads positions 0, 1, 2 from A and
3 from B, so each core still sees its activation objects in exactly the order the schedule lists them - the cores,
the packets and ``engine.cc`` are unchanged (tools/stream_share_probe.py ``build_qf`` measured this transport:
0.818 of today's transport time on four columns).

This module rewrites a schedule's column programs for that transport, and only that:

- **Positions.** Every quad (4 x 6,400 B, one object per core) gets its index in its column's consumption order,
  counted across the whole frame. Position p goes on stream 1 when ``p % PERIOD == STREAM1_POSITION``.
- **Splitting.** A fill of k quads becomes per-quad patterns, which are merged back (``merge_runs``) into the
  fewest tasks per stream: runs of consecutive stream-0 quads, and stream-1 quads spaced a period apart.
  A 1-quad fill whose core index sits on the task repeat (the 4-D form most fills take) cannot merge with
  anything, and is not asked to.
- **Order.** Stream 1 must never hold a quad behind a weight run that waits for that quad (a weight run of a
  round completes only after the cores consume it, chunk by chunk, against that round's quads). So each
  segment's stream-1 quads (``"P"`` items) are issued before the weight item that opens the segment. Weight runs
  replayed across rounds would put many rounds' weights ahead of later rounds' quads, so the schedule must be
  built with ``weight_repeat=False`` and ``merge_group_weights=False`` (one weight task per round, per group).
- **Holds.** The emitter may not await a weight task or a drain before the activation tasks it serves are
  issued (``hold``). Holds are recomputed from the quads each item served in the original program: a stream-1
  task moved ahead of its weight item no longer counts toward that item's hold.
- **Frame padding.** The MemTile chains free-run across dispatches, so each column's quad count per frame must
  be a multiple of PERIOD. A column that is short gets a NOP weight object and up to three throwaway quads at
  the end of the frame: the cores consume them and emit nothing.

``check_stream_share`` re-derives every invariant above from the rewritten programs and the original ones; the
compiler calls it on every build. Nothing here touches the hardware.
"""
from dataclasses import dataclass
from math import prod
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ignite_xdna.compiler import engine_emulator as em
from ignite_xdna.compiler.engine_sequence import DmaPattern, canonical, linear, merge_runs

QUAD = 4 * em.A_BYTES          # one activation object for each of a column's four cores
PERIOD = 4
STREAM1_POSITION = 3           # q = 1: one quad in four on the weight stream
B_DEPTH = 8                    # MemTile buffer B slots; every stream-1 quad of one segment must fit
CORE_W_DEPTH = 2               # weight objects a core accepts ahead of consuming them
PAD_SOURCE_OFFSET = 0          # throwaway pad quads read the workspace's first QUAD bytes; their values are unused


def split_quads(p: DmaPattern) -> List[DmaPattern]:
    """Per-quad patterns of a fill, in stream order (the concatenation of their bytes is the fill's).

    The fill is cut at its outermost dimension that can be: dimension i splits into (f, g) when the bytes inside
    one step of it divide a quad, and g of those steps make exactly one quad.
    """
    if p.nbytes % QUAD:
        raise ValueError(f"fill of {p.nbytes} B is not whole quads")
    k = p.nbytes // QUAD
    if k == 1:
        return [p]
    sizes, strides = list(p.sizes), list(p.strides)
    for i in range(len(sizes)):
        inner = prod(sizes[i + 1:])
        if QUAD % inner or sizes[i] % (QUAD // inner):
            continue
        g = QUAD // inner
        f = sizes[i] // g
        outer = list(np.ndindex(*sizes[:i])) if i else [()]
        quads = []
        for idx in outer:
            base = p.offset + sum(a * s for a, s in zip(idx, strides[:i]))
            for fi in range(f):
                quads.append(DmaPattern(p.buffer, base + fi * g * strides[i], (g,) + tuple(sizes[i + 1:]),
                                        (strides[i],) + tuple(strides[i + 1:])))
        assert len(quads) == k
        return [canonical(q) for q in quads]
    raise ValueError(f"cannot cut {p} into quads")


@dataclass
class Quad:
    pattern: DmaPattern
    position: int              # index in the column's consumption order, across the frame
    source: int                # index of the original program item it came from


def _segments(prog: Sequence[tuple]) -> List[Tuple[int, int]]:
    """(start, end) item ranges of maximal runs of consecutive 'A' items."""
    out, start = [], None
    for i, it in enumerate(prog):
        if it[0] == "A":
            if start is None:
                start = i
        elif start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(prog)))
    return out


def _served_items(prog: Sequence[tuple]) -> Dict[int, List[int]]:
    """For each weight item and drain, the indices of the 'A' items it serves (its original ``hold`` many,
    counted from the items after it - the emitter's definition)."""
    a_after = []
    for i, it in enumerate(prog):
        if it[0] in ("w", "W", "o"):
            n = it[3] if it[0] == "w" else it[2]
            nxt = [j for j in range(i + 1, len(prog)) if prog[j][0] == "A"][:n]
            if len(nxt) != n:
                raise ValueError(f"item {i} serves {n} activation items, only {len(nxt)} follow it")
            a_after.append((i, nxt))
    return dict(a_after)


def nop_packet(n_objects: int) -> np.ndarray:
    """A weight object whose op is NOP: the cores consume ``n_objects`` activation objects with it, emit nothing."""
    hdr = em.PacketHeader(op=em.OP_NOP, flags=0, count_out=0, count_acc=n_objects)
    return em.pack_w_packet(hdr, np.zeros(32, dtype=np.int32), None)


class WontFit(ValueError):
    """A replayed weight task serves more stream-1 quads than buffer B holds: the layer needs per-round weights."""


def _weight_objects(it: tuple) -> int:
    return (it[2] if it[0] == "w" else it[1].nbytes) // em.W_BYTES


def share_layer(s, pos: List[int], b_depth: int = B_DEPTH):
    """One layer's column programs rewritten for stream sharing; ``pos`` (per-column next position) advances.

    A weight task of n objects completes only once the cores have consumed n - CORE_W_DEPTH of them, so when
    n > CORE_W_DEPTH ('w' runs, and every 'W' replay spanning several rounds) it is preceded by every stream-1
    quad it serves that is not issued yet - no weight task then waits on the stream for a quad queued behind it.
    Those quads wait in buffer B for the weights behind them, so ``WontFit`` is raised (``pos`` untouched) when
    there are more than ``b_depth``. Every other stream-1 quad is issued at the start of its segment.
    """
    import copy
    if isinstance(s.programs, list) and not any(s.programs):
        if getattr(s, "n_packets", 0):
            raise ValueError(f"{s.name}: a layer with packets and no programs")
    s2 = copy.copy(s)
    s2.programs = []
    new_pos = list(pos)
    for c, prog in enumerate(s.programs):
        if any(it[0] not in ("A", "w", "W", "o") for it in prog):
            raise ValueError(f"{s.name}: stream sharing handles A, w, W and o items only, got "
                             f"{sorted({it[0] for it in prog})}")
        served = _served_items(prog)
        quads_of: Dict[int, List[Quad]] = {}
        for i, it in enumerate(prog):
            if it[0] == "A":
                qs = []
                for qp in split_quads(it[1]):
                    qs.append(Quad(qp, new_pos[c], i))
                    new_pos[c] += 1
                quads_of[i] = qs
        seg_of_item = {i: (a, b) for a, b in _segments(prog) for i in range(a, b)}
        done = set()                                  # A items whose stream-1 quads are issued

        def s1(items):
            return [q for i in items if i not in done for q in quads_of[i] if q.position % PERIOD == STREAM1_POSITION]

        new: List[tuple] = []
        for i, it in enumerate(prog):
            if it[0] in ("w", "W") and _weight_objects(it) > CORE_W_DEPTH:
                ahead = s1(served[i])
                if len(ahead) > b_depth:
                    raise WontFit(f"{s.name} col {c}: a weight item serves {len(ahead)} stream-1 quads")
                new.extend(_merge(ahead, "P"))
                done.update(served[i])
                new.append((it[0], it[1], it[2] if it[0] == "w" else None, i))
            elif it[0] in ("w", "W"):
                new.append((it[0], it[1], it[2] if it[0] == "w" else None, i))
            elif it[0] == "o":
                new.append(("o", it[1], None, i))
            else:
                if i not in done:
                    seg = [j for j in range(*seg_of_item[i]) if j not in done]
                    new.extend(_merge(s1(seg), "P"))
                    done.update(seg)
                qs = [q for q in quads_of[i] if q.position % PERIOD != STREAM1_POSITION]
                new.extend(_merge_consecutive(qs))
        s2.programs.append(_set_holds(new, served, quads_of))
    pos[:] = new_pos
    return s2


def pad_frame(out, store, pos: List[int]) -> None:
    """Pad each column to whole periods at the end of the frame: the MemTile chains free-run across dispatches."""
    for c in range(len(pos)):
        short = (-pos[c]) % PERIOD
        if not short:
            continue
        last = out[-1].programs[c]
        off = store.add(nop_packet(short))
        pad = []
        for _ in range(short):
            q = Quad(linear("ws", PAD_SOURCE_OFFSET, QUAD), pos[c], -1)
            pos[c] += 1
            pad.append(("P" if q.position % PERIOD == STREAM1_POSITION else "A", q.pattern, [q.position]))
        last.extend(x for x in pad if x[0] == "P")
        last.append(("w", off, em.W_BYTES, sum(1 for x in pad if x[0] == "A")))
        last.extend(x for x in pad if x[0] == "A")


def stream_share_programs(scheds, store, first_positions: Sequence[int] = (0, 0, 0, 0)):
    """Rewrite every layer of a frame (no host layers); returns the new schedules and the final positions."""
    pos = list(first_positions)
    out = [share_layer(s, pos) for s in scheds]
    pad_frame(out, store, pos)
    return out, pos


def schedule_stream_shared(ir, ws, layers=None, **flags):
    """Schedule a graph for stream sharing: each layer keeps the shipped weight replays where every replay's
    stream-1 quads fit in buffer B, and takes per-round weights otherwise. Returns (original schedules - the
    ones the transform rewrote, for ``check_stream_share`` - shared schedules, store, per-layer mode).

    ``layers`` keeps a prefix of the graph (the compiler's ``layers``); the period padding goes after its last
    layer, which is the frame's end for that container."""
    from ignite_xdna.compiler import engine_schedule as es
    from ignite_xdna.compiler.graph_ir import HostLayer
    store = es.PacketStore()
    pos = [0] * es.COLS
    orig, out, modes = [], [], []
    for L in (ir.layers if layers is None else ir.layers[:layers]):
        if isinstance(L, HostLayer):
            raise ValueError(f"{L.name}: a host layer splits the dispatch; stream sharing pads only the frame's end")
        # The trial goes into a throwaway store: a layer that falls back would otherwise leave its replay's
        # concatenated runs in the container, unused (49.4 MB of YOLOv8s packets against 29.5 shipped).
        try:
            share_layer(es.schedule_layer(ir, ws, L, es.PacketStore(), **flags), list(pos))
            fits = True
        except WontFit:
            fits = False
        if fits:
            s = es.schedule_layer(ir, ws, L, store, **flags)
            sh, mode = share_layer(s, pos), "replay"
        else:
            s = es.schedule_layer(ir, ws, L, store, **dict(flags, weight_repeat=False, merge_group_weights=False))
            sh, mode = share_layer(s, pos), "per-round"
        orig.append(s)
        out.append(sh)
        modes.append(mode)
    pad_frame(out, store, pos)
    return orig, out, store, modes


def _merge(qs: List[Quad], kind: str) -> List[tuple]:
    """Merge quads of one stream into the fewest tasks (merge_runs), each carrying its quads' positions."""
    if not qs:
        return []
    out = []
    i = 0
    pats = merge_runs([q.pattern for q in qs])
    for p in pats:
        n = p.nbytes // QUAD
        out.append((kind, p, [q.position for q in qs[i:i + n]]))
        i += n
    assert i == len(qs)
    return out


def _merge_consecutive(qs: List[Quad]) -> List[tuple]:
    """Stream-0 quads of one fill: merge only runs of consecutive positions (a gap is a stream-1 quad)."""
    out, run = [], []
    for q in qs:
        if run and q.position != run[-1].position + 1:
            out.extend(_merge(run, "A"))
            run = []
        run.append(q)
    out.extend(_merge(run, "A"))
    return out


def _set_holds(new: List[tuple], served: Dict[int, List[int]], quads_of: Dict[int, List[Quad]]) -> List[tuple]:
    """Final items with emitter holds: a drain, or a weight item of more than CORE_W_DEPTH objects, holds for the
    activation tasks AFTER it that carry any quad it served in the original program.

    A weight item of at most CORE_W_DEPTH objects holds for nothing: it completes once its objects reach the
    cores, which needs only the EARLIER weight objects consumed - and every quad those serve is issued before it
    (``check_stream_share`` proves that). Holding it anyway would block the weight channel: the "P" quads queued
    behind it on that channel fill the shim's 4-deep start queue, and nothing in front of them may be retired."""
    final = []
    for k, it in enumerate(new):
        if it[0] in ("w", "W", "o"):
            src = it[-1]
            want = {q.position for i in served.get(src, []) for q in quads_of[i]}
            hold = sum(1 for x in new[k + 1:] if x[0] in ("A", "P") and want.intersection(x[2]))
            if it[0] in ("w", "W") and _weight_objects(it) <= CORE_W_DEPTH:
                hold = 0
            if it[0] == "w":
                final.append(("w", it[1], it[2], hold))
            elif it[0] == "W":
                final.append(("W", it[1], hold))
            else:
                final.append(("o", it[1], hold))
        else:
            final.append(it)
    return final




def check_stream_share(original, shared, b_depth: int = B_DEPTH) -> Dict[str, int]:
    """Re-derive the transport's invariants from the original and rewritten programs; raises on any violation.

    Per column, across the frame:
    (1) the quads' bytes in position order equal the original fills' bytes in program order, plus fewer than
        PERIOD pad quads, and the frame's quad count is a multiple of PERIOD;
    (2) each stream issues its quads in increasing position, and every quad is on the stream its position says;
    (3) NO DEADLOCK ON STREAM 1: a weight task of n objects completes only once the cores consumed n - 2 of them
        (CORE_W_DEPTH), so when n > CORE_W_DEPTH every stream-1 quad it serves must be issued before it;
    (4) BUFFER B: every stream-1 quad a weight item serves fits in B alongside the others it serves, i.e. at most
        ``b_depth`` of them - a weight task's own quads must never wait for B space behind each other;
    (5) holds: every drain's, and every weight item's of more than CORE_W_DEPTH objects, hold counts exactly the
        tasks after it carrying quads it serves; a weight item of at most CORE_W_DEPTH objects holds 0, and every
        quad served by an earlier weight item is issued before it (so it completes without later tasks).
    """
    n_tasks = n_p = 0
    cols = len(original[0].programs)
    for c in range(cols):
        # Positions from the ORIGINAL programs, independently of the transform.
        orig_quads, served_pos = [], []          # served_pos: per original w/o item, (kind, objects, positions)
        pos = 0
        for s in original:
            prog = s.programs[c]
            first_pos = {}
            for i, it in enumerate(prog):
                if it[0] == "A":
                    first_pos[i] = pos
                    qs = split_quads(it[1])
                    orig_quads.extend(qs)
                    pos += len(qs)
            for i, idx in _served_items(prog).items():
                ps = [first_pos[j] + k for j in idx for k in range(prog[j][1].nbytes // QUAD)]
                objs = _weight_objects(prog[i]) if prog[i][0] in ("w", "W") else 0
                served_pos.append((prog[i][0], objs, set(ps)))
        new_q, last = [], {"A": -1, "P": -1}
        k = 0
        issued, owed = set(), set()          # positions issued so far; positions served by weight items so far
        for s in shared:
            prog = s.programs[c]
            for idx, it in enumerate(prog):
                if it[0] in ("A", "P"):
                    n_tasks += 1
                    n_p += it[0] == "P"
                    qs = split_quads(it[1])
                    if len(qs) != len(it[2]):
                        raise AssertionError(f"{s.name} col {c}: task carries {len(qs)} quads, {len(it[2])} positions")
                    for qp, p in zip(qs, it[2]):
                        if p <= last[it[0]]:
                            raise AssertionError(f"{s.name} col {c}: {it[0]} position {p} after {last[it[0]]}")
                        last[it[0]] = p
                        issued.add(p)
                        if (p % PERIOD == STREAM1_POSITION) != (it[0] == "P"):
                            raise AssertionError(f"{s.name} col {c}: quad {p} on the wrong stream")
                        new_q.append((p, qp))
                elif it[0] in ("w", "W", "o"):
                    if k >= len(served_pos):
                        if it[0] == "w" and s is shared[-1]:
                            continue                     # the frame pad's NOP object
                        raise AssertionError(f"{s.name} col {c}: more weight/drain items than the original")
                    kind, objs, want = served_pos[k]
                    k += 1
                    if kind != it[0]:
                        raise AssertionError(f"{s.name} col {c}: item order changed ({kind} -> {it[0]})")
                    after = [x for x in prog[idx + 1:] if x[0] in ("A", "P") and want.intersection(x[2])]
                    hold = it[3] if it[0] == "w" else it[2]
                    is_w = it[0] in ("w", "W")
                    small = is_w and objs <= CORE_W_DEPTH
                    if small and owed - issued:
                        raise AssertionError(f"{s.name} col {c}: an unheld weight item ahead of "
                                             f"{len(owed - issued)} quads an earlier weight item serves")
                    if is_w:
                        owed |= want
                    if hold != (0 if small else len(after)):
                        raise AssertionError(f"{s.name} col {c}: hold {hold}, {len(after)} served tasks follow")
                    s1 = {p for p in want if p % PERIOD == STREAM1_POSITION}
                    late = s1.intersection(p for x in after if x[0] == "P" for p in x[2])
                    if is_w and objs > CORE_W_DEPTH and late:
                        raise AssertionError(f"{s.name} col {c}: {len(late)} stream-1 quads behind a "
                                             f"{objs}-object weight task that needs them")
                    # Quads issued AHEAD of their weight wait in B until the weight arrives behind them: they
                    # must all fit, or the stream stalls on B space in front of the weight they wait for.
                    # Quads behind their weight only make the stream wait for consumption - no cycle.
                    if is_w and len(s1 - late) > b_depth:
                        raise AssertionError(f"{s.name} col {c}: {len(s1 - late)} stream-1 quads ahead of their "
                                             f"weight item, buffer B holds {b_depth}")
                else:
                    raise AssertionError(f"{s.name} col {c}: unexpected item {it[0]}")
        if k != len(served_pos):
            raise AssertionError(f"col {c}: {len(served_pos) - k} original weight/drain items missing")
        new_q.sort(key=lambda t: t[0])
        orig_bytes = orig_quads
        pad = len(new_q) - len(orig_bytes)
        if not 0 <= pad < PERIOD or len(new_q) % PERIOD:
            raise AssertionError(f"col {c}: {len(new_q)} quads against {len(orig_bytes)} original")
        for (p, qp), op in zip(new_q, orig_bytes):
            if not np.array_equal(qp.indices(), op.indices()) or qp.buffer != op.buffer:
                raise AssertionError(f"col {c}: quad {p} reads different bytes from the original")
    return {"activation_tasks": n_tasks, "stream1_tasks": n_p}
