"""What does one shim task cost the instruction stream, measured directly?

Every Block 5 sizing priced an extra shim task at 0.36 us: 4 instruction ops at the 0.09 us per op that the
engine's per-layer fit (engine_layer_timing) attributes to ops. That is a fit coefficient, not a measurement.
This moves the same bytes through the engine's activation path - shim MM2S -> MemTile split -> the four cores of
each column, 6,400 B per core object, no compute - on all four columns, issued as tasks of k quads each (a quad
is one object for each of the column's four cores). The tasks are retired the way the engine's emitter retires
them: a completion token on every second task and on the channel's last, at most 4 tasks pending per channel,
the oldest token group awaited and freed before a fifth is configured. So the instruction stream per task is
the engine's (configure + patch + push, and an await every second task).

    python tools/task_issue_probe.py build                (in the IRON env: bash scripts/research-iron.sh ...)
    python tools/task_issue_probe.py sit --rounds 3 --iters 30 --json <out>

Grid: k in 1, 2, 4, 16, 64 quads per task x Q in 512, 1024 quads per column. Fit over all points:
    dispatch ms = a + b * MB + c * tasks           (tasks = 4 columns x Q / k)
c is the per-task cost; b the per-byte cost at this traffic. A fifth point field, abytes (contig:64:1024:4:800),
cuts the same shim tasks into smaller core objects, so issue is fixed and only the object count changes. Each measurement runs in its own child process,
the grid rotates across rounds. Written on Desktop 2 (Phoenix), 2026-10-13; what has been run is in its logs.

2026-10-15: AMD's split of roles. o4 is the engine's overlapping per-core cut done on the shim; box moves the union
of the four slices once and has the MemTile cut it. ObjectFifo's split cannot express box (it sizes each child BD
by the gap between offsets: 1,000 / 1,000 / 1,000 / 15,400 B against a 6,400 B pattern, and that build times out),
so o4raw / boxraw are written as raw MLIR: per-buffer producer locks taken 4 at a time, one consumer lock per core,
and a MemTile S2MM chain that releases core j once the last byte it reads has landed. --check makes the cores copy
what they receive back to DDR for a byte-exact comparison.
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

A_BYTES = 6400
QUAD = 4 * A_BYTES
COLS = 4
KS = (1, 2, 4, 16, 64)
QS = (512, 1024)
QUEUE_DEPTH = 4
RETIRE_BATCH = 2
BUILD = ROOT / "build" / "task_issue_probe"


# How one task of k quads addresses DDR. Each entry gives (bytes of DDR one quad spans, sizes, strides) for a task of
# k quads; the outermost size is the queue-push repeat count (shim_dma_single_bd_task), the other three are the BD's
# addressing dimensions. Every shape moves exactly k x 25,600 B per task.
#   contig  the original grid: one contiguous run (the BD folds it), k quads by repeat.
#   t3      one quad as a strided 3-D pattern, 16 planes x 8 rows x 200 B at a 2,048 / 256 B pitch; k quads by
#           repeat. k=1 has no repeat at all.
#   e4      the engine's own shape (e.g. YOLOv8s's (4, 4, 8, 200) fills): 4 per-core slices on the repeat, each a
#           3-D 4 x 8 x 200 B pattern. A second quad would need a fifth dimension, so only k=1 exists.
#   o4      the engine's dominant YOLOv8s fill (3x3 conv, e.g. (4, 4, 8, 200) at (3280, 53792, 656, 1)): core j
#           takes 4 planes x rows 5j..5j+7, so neighbouring cores' slices overlap by a 3-row halo. Cut on the shim,
#           one quad per task (k=1 only), as the engine does today.
#   box     the same quad as the union of its core slices, one 3-D box (4 planes x 23 rows x 200 B, 18,400 B: the
#           halo crosses the shim once), k boxes by repeat. The MemTile cuts the four per-core objects out of it
#           (split offsets + dims_to_stream), so each core receives exactly what o4 delivers.
FOOT = 32768
FOOT_O = 4 * 24 * 256            # o4/box quad footprint: 4 planes of 24 rows at a 256 B pitch
BOX_ROWS = 3 * 5 + 8             # rows 0..22: four 8-row windows at a 5-row step
BOX_BYTES = 4 * BOX_ROWS * 200
SHAPES = {
    "contig": lambda k: (QUAD, [k, 1, 64, 400], [QUAD, 0, 400, 1]),
    "t3": lambda k: (FOOT, [k, 16, 8, 200], [FOOT, 2048, 256, 1]),
    "e4": lambda k: (FOOT, [4, 4, 8, 200], [8192, 2048, 256, 1]),
    "o4": lambda k: (FOOT_O, [4, 4, 8, 200], [5 * 256, 24 * 256, 256, 1]),
    "box": lambda k: (FOOT_O, [k, 4, BOX_ROWS, 200], [FOOT_O, 24 * 256, 256, 1]),
}
SHAPES["o4raw"] = SHAPES["o4"]
SHAPES["boxraw"] = SHAPES["box"]
# Bytes the shim moves per quad; every shape but box moves the four 6,400 B objects themselves.
QUAD_DDR = {"box": BOX_BYTES, "boxraw": BOX_BYTES}
# Raw MemTile layouts: (five S2MM chunks as (offset, len); chunk i >= 1 releases core i - 1, so its end must be at
# or past that core's last byte; per-core MM2S as (offset, sizes, strides)). Core j's last byte in the box is
# 3 x 4,600 + 1,000 j + 1,599 (ends 15,400 / 16,400 / 17,400 / 18,400); in the o4 quad it is 6,400 (j + 1).
RAW = {
    "o4raw": ([(0, 3200), (3200, 3200), (6400, 6400), (12800, 6400), (19200, 6400)],
              [(j * A_BYTES, None, None) for j in range(4)]),
    "boxraw": ([(0, 15400), (15400, 1000), (16400, 1000), (17400, 500), (17900, 500)],
               [(j * 1000, [4, 8, 200], [BOX_ROWS * 200, 200, 1]) for j in range(4)]),
}


# MemTile S2MM channels of the check build's join (core j -> slice j). The input takes channel 1: the MemTile's BD
# pool is halved by channel parity (24 each), and the 10-BD input chain beside the 8-BD join drain overflows one half.
JOIN_S2MM = (0, 2, 3, 4)


def raw_module(k: int, q: int, shape: str, cols: int, check: bool) -> str:
    """The o4raw / boxraw design as MLIR text: same shim traffic and retire order as build_one, MemTile and cores
    by hand. Every lock protocol here is depth 2: two buffers per object, producer lock init 2 (MemTile: init 4 per
    buffer, taken 4 at a time, since four cores read one buffer)."""
    foot, sizes, strides = SHAPES[shape](k)
    quad_ddr = QUAD_DDR.get(shape, QUAD)
    chunks, reads = RAW[shape]
    col_bytes = q * foot
    total = cols * col_bytes
    ntasks = q // k
    L = ["module {", "  aie.device(npu1) {"]
    for c in range(cols):
        L += [f"    %s{c} = aie.tile({c}, 0)", f"    %m{c} = aie.tile({c}, 1)"]
        L += [f"    %t{c}_{j} = aie.tile({c}, {2 + j})" for j in range(4)]
    for c in range(cols):
        m = f"%m{c}"
        L += [f"    %mb{c}_{b} = aie.buffer({m}) {{sym_name = \"mb{c}_{b}\"}} : memref<{quad_ddr}xui8>" for b in (0, 1)]
        L += [f"    %mp{c}_{b} = aie.lock({m}, {b}) {{init = 4 : i32, sym_name = \"mp{c}_{b}\"}}" for b in (0, 1)]
        L += [f"    %mc{c}_{j} = aie.lock({m}, {2 + j}) {{init = 0 : i32, sym_name = \"mc{c}_{j}\"}}" for j in range(4)]
        L.append(f"    %mz{c} = aie.lock({m}, 14) {{init = 0 : i32, sym_name = \"mz{c}\"}}")
        if check:
            L += [f"    %ob{c}_{b} = aie.buffer({m}) {{sym_name = \"ob{c}_{b}\"}} : memref<{QUAD}xui8>" for b in (0, 1)]
            L += [f"    %op{c}_{j} = aie.lock({m}, {6 + 2 * j}) {{init = 2 : i32, sym_name = \"op{c}_{j}\"}}"
                  for j in range(4)]
            L += [f"    %oc{c}_{j} = aie.lock({m}, {7 + 2 * j}) {{init = 0 : i32, sym_name = \"oc{c}_{j}\"}}"
                  for j in range(4)]
        for j in range(4):
            t = f"%t{c}_{j}"
            L += [f"    %cb{c}_{j}_{b} = aie.buffer({t}) {{sym_name = \"cb{c}_{j}_{b}\"}} : memref<{A_BYTES}xui8>"
                  for b in (0, 1)]
            L += [f"    %cp{c}_{j} = aie.lock({t}, 0) {{init = 2 : i32, sym_name = \"cp{c}_{j}\"}}",
                  f"    %cc{c}_{j} = aie.lock({t}, 1) {{init = 0 : i32, sym_name = \"cc{c}_{j}\"}}"]
            if check:
                L += [f"    %xb{c}_{j}_{b} = aie.buffer({t}) {{sym_name = \"xb{c}_{j}_{b}\"}} : memref<{A_BYTES}xui8>"
                      for b in (0, 1)]
                L += [f"    %xp{c}_{j} = aie.lock({t}, 2) {{init = 2 : i32, sym_name = \"xp{c}_{j}\"}}",
                      f"    %xc{c}_{j} = aie.lock({t}, 3) {{init = 0 : i32, sym_name = \"xc{c}_{j}\"}}"]
        L.append(f"    aie.flow(%s{c}, DMA : 0, {m}, DMA : 1)")
        L += [f"    aie.flow({m}, DMA : {j}, %t{c}_{j}, DMA : 0)" for j in range(4)]
        if check:
            L += [f"    aie.flow(%t{c}_{j}, DMA : 0, {m}, DMA : {JOIN_S2MM[j]})" for j in range(4)]
            L.append(f"    aie.flow({m}, DMA : 4, %s{c}, DMA : 0)")
        L.append(f"    aie.shim_dma_allocation @a{c}(%s{c}, MM2S, 0)")
        if check:
            L.append(f"    aie.shim_dma_allocation @o{c}(%s{c}, S2MM, 0)")

        # MemTile DMA: one S2MM chain in, four MM2S chains out (and, for check, the join back).
        R = [f"    %md{c} = aie.memtile_dma({m}) {{", "      %c1 = arith.constant 1 : i32",
             "      %c4 = arith.constant 4 : i32", "      %c0 = arith.constant 0 : i32"]
        blocks = []   # (start line, [bd blocks]) per channel; each bd block is (label, lines)

        def chain(direction, ch, bds):
            blocks.append((direction, ch, bds))

        s2mm = []
        for b in (0, 1):
            for i, (off, ln) in enumerate(chunks):
                # A BD with a lock must both acquire and release (the CDO generator's rule, and it rejects a zero
                # acquire). So the first chunk takes the buffer's 4 producer tokens and puts 4 on a relay lock;
                # chunk i >= 1 takes one relay token and releases core i - 1.
                if i == 0:
                    locks = (f"aie.use_lock(%mp{c}_{b}, AcquireGreaterEqual, %c4)", f"aie.use_lock(%mz{c}, Release, %c4)")
                else:
                    locks = (f"aie.use_lock(%mz{c}, AcquireGreaterEqual, %c1)",
                             f"aie.use_lock(%mc{c}_{i - 1}, Release, %c1)")
                s2mm.append([locks[0], f"aie.dma_bd(%mb{c}_{b} : memref<{quad_ddr}xui8> offset = {off} len = {ln})",
                             locks[1]])
        chain("S2MM", 1, s2mm)
        for j, (off, sz, st) in enumerate(reads):
            dims = f" sizes = {sz} strides = {st}" if sz else ""
            chain("MM2S", j, [[f"aie.use_lock(%mc{c}_{j}, AcquireGreaterEqual, %c1)",
                               f"aie.dma_bd(%mb{c}_{b} : memref<{quad_ddr}xui8> offset = {off} len = {A_BYTES}{dims})",
                               f"aie.use_lock(%mp{c}_{b}, Release, %c1)"] for b in (0, 1)])
        if check:
            for j in range(4):
                chain("S2MM", JOIN_S2MM[j], [[f"aie.use_lock(%op{c}_{j}, AcquireGreaterEqual, %c1)",
                                       f"aie.dma_bd(%ob{c}_{b} : memref<{QUAD}xui8> offset = {j * A_BYTES} len = {A_BYTES})",
                                       f"aie.use_lock(%oc{c}_{j}, Release, %c1)"] for b in (0, 1)])
            chain("MM2S", 4, [[f"aie.use_lock(%oc{c}_{j}, AcquireGreaterEqual, %c1)",
                               f"aie.dma_bd(%ob{c}_{b} : memref<{QUAD}xui8> offset = {j * A_BYTES} len = {A_BYTES})",
                               f"aie.use_lock(%op{c}_{j}, Release, %c1)"] for b in (0, 1) for j in range(4)])
        R += dma_region(f"md{c}", blocks)
        L += R

        for j in range(4):
            t, tag = f"%t{c}_{j}", f"{c}_{j}"
            blocks = [("S2MM", 0, [[f"aie.use_lock(%cp{tag}, AcquireGreaterEqual, %c1)",
                                    f"aie.dma_bd(%cb{tag}_{b} : memref<{A_BYTES}xui8> offset = 0 len = {A_BYTES})",
                                    f"aie.use_lock(%cc{tag}, Release, %c1)"] for b in (0, 1)])]
            if check:
                blocks.append(("MM2S", 0, [[f"aie.use_lock(%xc{tag}, AcquireGreaterEqual, %c1)",
                                            f"aie.dma_bd(%xb{tag}_{b} : memref<{A_BYTES}xui8> offset = 0 len = {A_BYTES})",
                                            f"aie.use_lock(%xp{tag}, Release, %c1)"] for b in (0, 1)]))
            L += [f"    %cm{tag} = aie.mem({t}) {{", "      %c1 = arith.constant 1 : i32"]
            L += dma_region(f"cm{tag}", blocks)
            body = []
            for b in (0, 1):
                body.append(f"        aie.use_lock(%cc{tag}, AcquireGreaterEqual, %c1_i32)")
                if check:
                    body += [f"        aie.use_lock(%xp{tag}, AcquireGreaterEqual, %c1_i32)",
                             f"        scf.for %x = %c0 to %cn step %c1 {{",
                             f"          %v = memref.load %cb{tag}_{b}[%x] : memref<{A_BYTES}xui8>",
                             f"          memref.store %v, %xb{tag}_{b}[%x] : memref<{A_BYTES}xui8>",
                             "        }",
                             f"        aie.use_lock(%xc{tag}, Release, %c1_i32)"]
                body.append(f"        aie.use_lock(%cp{tag}, Release, %c1_i32)")
            L += [f"    %core{tag} = aie.core({t}) {{", "      %c0 = arith.constant 0 : index",
                  "      %c1 = arith.constant 1 : index", f"      %cn = arith.constant {A_BYTES} : index",
                  "      %cmax = arith.constant 9223372036854775807 : index",
                  "      %c1_i32 = arith.constant 1 : i32",
                  "      scf.for %i = %c0 to %cmax step %c1 {"] + body + ["      }", "      aie.end", "    }"]

    # Runtime sequence: the same tasks, tokens and retire order as build_one.
    S = [f"    aie.runtime_sequence(%in: memref<{total}xui8>, %out: memref<{cols * q * QUAD if check else 64}xui8>) {{"]
    n_id = [0]

    def new():
        n_id[0] += 1
        return f"%k{n_id[0]}"

    drains = []
    if check:
        for c in range(cols):
            v = new()
            S += [f"      {v} = aiex.dma_configure_task_for @o{c} {{",
                  f"        aie.dma_bd(%out : memref<{cols * q * QUAD}xui8> offset = {c * q * QUAD} len = {QUAD} "
                  f"sizes = [{q}, 1, 32, 800] strides = [{QUAD}, 0, 800, 1])",
                  "        aie.end", f"      }} {{issue_token = true, repeat_count = {q - 1} : i32}}",
                  f"      aiex.dma_start_task({v})"]
            drains.append(v)
    queues = {c: [] for c in range(cols)}

    def retire(c):
        qu = queues[c]
        i = next(i for i, (_, tok) in enumerate(qu) if tok)
        S.append(f"      aiex.dma_await_task({qu[i][0]})")
        S.extend(f"      aiex.dma_free_task({t})" for t, _ in qu[:i + 1])
        del qu[:i + 1]

    for n in range(ntasks):
        for c in range(cols):
            if len(queues[c]) >= QUEUE_DEPTH:
                retire(c)
            token = (n + 1) % RETIRE_BATCH == 0 or n == ntasks - 1
            v = new()
            # The outermost size is the queue-push repeat, not part of the BD length (the verifier's rule).
            attrs = (["issue_token = true"] if token else []) + ([f"repeat_count = {sizes[0] - 1} : i32"]
                                                                 if sizes[0] > 1 else [])
            S += [f"      {v} = aiex.dma_configure_task_for @a{c} {{",
                  f"        aie.dma_bd(%in : memref<{total}xui8> offset = {c * col_bytes + n * k * foot} "
                  f"len = {int(np.prod(sizes[1:]))} sizes = {list(sizes)} strides = {list(strides)})",
                  "        aie.end", "      }" + (f" {{{', '.join(attrs)}}}" if attrs else ""),
                  f"      aiex.dma_start_task({v})"]
            queues[c].append([v, token])
    for c in range(cols):
        while queues[c]:
            retire(c)
    S += [f"      aiex.dma_await_task({v})" for v in drains]
    S.append("    }")
    return "\n".join(L + S + ["  }", "}"]) + "\n"


def dma_region(name, blocks):
    """A DMA region (aie.mem / aie.memtile_dma body) from channel chains of BD blocks; each chain loops."""
    out, label = [], [0]

    def lab():
        label[0] += 1
        return f"^{name}_{label[0]}"

    for direction, ch, bds in blocks:
        first = lab()
        rest = [lab() for _ in bds[1:]]
        labels = [first] + rest
        nxt = lab()
        out.append(f"      %d{name}_{direction}{ch} = aie.dma_start({direction}, {ch}, {first}, {nxt})")
        for i, lines in enumerate(bds):
            out.append(f"    {labels[i]}:")
            out += [f"      {ln}" for ln in lines]
            out.append(f"      aie.next_bd {labels[(i + 1) % len(labels)]}")
        out.append(f"    {nxt}:")
    out += ["      aie.end", "    }"]
    return out


def expected_objects(data: np.ndarray, col_base: int, q: int) -> np.ndarray:
    """What the four cores of one column must receive over q quads of the o4/box layout, in core-object order."""
    sizes, strides = [4, 4, 8, 200], [5 * 256, 24 * 256, 256, 1]
    idx = np.zeros(sizes, dtype=np.int64)
    for d, (n, s) in enumerate(zip(sizes, strides)):
        shape = [1] * 4
        shape[d] = n
        idx = idx + (np.arange(n) * s).reshape(shape)
    idx = idx.reshape(-1)
    return np.concatenate([data[col_base + n * FOOT_O + idx] for n in range(q)])


def name_of(k, q, shape="contig", cols=COLS, abytes=A_BYTES, check=False):
    base = f"k{k}_q{q}" if shape == "contig" else f"{shape}_k{k}_q{q}"
    base = base if cols == COLS else f"{base}_c{cols}"
    base = base if abytes == A_BYTES else f"{base}_a{abytes}"
    return f"{base}_chk" if check else base


def build_one(k: int, q: int, shape: str = "contig", cols: int = COLS, abytes: int = A_BYTES,
              check: bool = False) -> Path:
    """One variant on columns 0..cols-1; ``cols`` < 4 asks whether the small-task cost is per channel (it then
    reappears in full on one column) or one issue path shared by the array (it then hides under transport).

    ``abytes`` is the core object size. The shim tasks stay byte-for-byte the same (a quad is still 25,600 B of
    DDR), so the instruction stream and issue count are fixed; only the MemTile split and the core fifos cut the
    same stream into more, smaller objects. It asks whether delivering an object costs anything per object on the
    lock-cycled MemTile -> core path, with issue held constant.

    ``check`` builds the same delivery with cores that copy every object they receive into an output fifo, joined
    back through the MemTile to DDR, so ``time_one`` can compare what the cores saw byte for byte (o4/box only)."""
    assert QUAD % (4 * abytes) == 0, "a quad must hold a whole number of 4-core object groups"
    import aie.iron as iron
    from aie.dialects.aiex import dma_await_task, dma_free_task, dma_start_task, shim_dma_single_bd_task
    from aie.helpers.dialects.func import func
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU1, Tile
    from aie.utils.compile.utils import compile_mlir_module

    assert q % k == 0 and k <= 64
    assert shape not in ("e4", "o4", "o4raw") or k == 1, "the engine's 4-D quad leaves no dimension for a second quad"
    assert not check or shape in ("box", "o4", "boxraw", "o4raw"), "check covers the o4/box layout only"
    assert shape != "box" or abytes == A_BYTES
    foot, sizes, strides = SHAPES[shape](k)
    quad_ddr = QUAD_DDR.get(shape, QUAD)
    out_dir = BUILD / name_of(k, q, shape, cols, abytes, check)
    work = out_dir / "design.prj"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    iron.set_current_device(NPU1())
    if shape in RAW:
        assert abytes == A_BYTES
        text = raw_module(k, q, shape, cols, check)
        (out_dir / "design.mlir").write_text(text, encoding="utf-8")
        compile_mlir_module(text, insts_path=out_dir / "insts.bin", xclbin_path=out_dir / "probe.xclbin",
                            work_dir=work, device=iron.get_current_device())
        total = cols * q * foot
        (out_dir / "meta.json").write_text(json.dumps({
            "k": k, "q": q, "shape": shape, "cols": cols, "sizes": list(sizes), "strides": list(strides),
            "tasks": cols * (q // k), "abytes": abytes, "core_objects": cols * q * 4, "total_bytes": total,
            "moved_bytes": cols * q * quad_ddr, "check": check,
            "insts_bytes": (out_dir / "insts.bin").stat().st_size}), encoding="utf-8")
        return out_dir
    obj_ty = np.ndarray[(abytes,), np.dtype[np.uint8]]
    quad_ty = np.ndarray[(quad_ddr,), np.dtype[np.uint8]]   # the MemTile object: one object for each core
    out_quad_ty = np.ndarray[(QUAD,), np.dtype[np.uint8]]
    if shape == "box":
        # Core j's 6,400 B object out of the box (4 planes x 23 rows x 200 B): rows 5j..5j+7 of every plane.
        split_offsets = [j * 5 * 200 for j in range(4)]
        split_dims = [[(4, BOX_ROWS * 200), (8, 200), (200, 1)]] * 4
    else:
        split_offsets = [j * abytes for j in range(4)]
        split_dims = None

    if check:
        @func
        def copy_obj(src: obj_ty, dst: obj_ty, n: np.int32):
            for x in range_(n):
                dst[x] = src[x]

        def core_fn(i, o, kern):
            a = i.acquire(1)
            b = o.acquire(1)
            kern(a, b, abytes)
            i.release(1)
            o.release(1)
    else:
        def core_fn(i):
            i.acquire(1)
            i.release(1)

    workers, shim_ends, names, out_names = [], [], [], []
    for c in range(cols):
        name = f"a{c}"
        f = ObjectFifo(quad_ty, name=name, depth=2)
        parts = f.cons().split(split_offsets, tile=Tile(c, 1), depths=[2] * 4, obj_types=[obj_ty] * 4,
                               names=[f"{name}_{j}" for j in range(4)], dims_to_stream=split_dims)
        if check:
            fo = ObjectFifo(out_quad_ty, name=f"o{c}", depth=2)
            outs = fo.prod().join([j * abytes for j in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                  obj_types=[obj_ty] * 4, names=[f"o{c}_{j}" for j in range(4)])
        for j, part in enumerate(parts):
            args = [part.cons(), outs[j].prod(), copy_obj] if check else [part.cons()]
            workers.append(Worker(core_fn, args, tile=Tile(c, 2 + j)))
        shim_ends.append(f.prod(tile=Tile(c, 0)))
        names.append(name)
        if check:
            shim_ends.append(fo.cons(tile=Tile(c, 0)))
            out_names.append(f"o{c}")
    col_bytes = q * foot             # DDR one column's quads span; each moves quad_ddr of it
    total = cols * col_bytes
    moved = cols * q * quad_ddr
    ntasks = q // k

    def sequence(inp, outp, *handles):
        inp = getattr(inp, "op", inp)
        outp = getattr(outp, "op", outp)
        drains = []
        for c, oname in enumerate(out_names):
            # One drain per column, started before its fills: q quads of 32 x 800 B by repeat.
            t = shim_dma_single_bd_task(oname, outp, offset=c * q * QUAD, sizes=[q, 1, 32, 800],
                                        strides=[QUAD, 0, 800, 1], issue_token=True)
            dma_start_task(t)
            drains.append(t)
        queues = {c: [] for c in range(cols)}   # [task, token]

        def retire(c):
            qu = queues[c]
            i = next(i for i, (_, tok) in enumerate(qu) if tok)
            dma_await_task(qu[i][0])
            for t, _ in qu[:i + 1]:
                dma_free_task(t)
            del qu[:i + 1]

        # Columns interleaved task by task, as the engine's emitter interleaves items.
        for n in range(ntasks):
            for c in range(cols):
                if len(queues[c]) >= QUEUE_DEPTH:
                    retire(c)
                token = (n + 1) % RETIRE_BATCH == 0 or n == ntasks - 1
                t = shim_dma_single_bd_task(names[c], inp, offset=c * col_bytes + n * k * foot,
                                            sizes=list(sizes), strides=list(strides), issue_token=token)
                dma_start_task(t)
                queues[c].append([t, token])
        for c in range(cols):
            while queues[c]:
                retire(c)
        for t in drains:
            dma_await_task(t)

    in_t = np.ndarray[(total,), np.dtype[np.uint8]]
    out_t = np.ndarray[(cols * q * QUAD if check else 64,), np.dtype[np.uint8]]
    rt = Runtime(sequence, [in_t, out_t] + shim_ends)
    module = Program(iron.get_current_device(), rt, workers=workers).resolve_program()
    (out_dir / "design.mlir").write_text(str(module), encoding="utf-8")
    compile_mlir_module(module, insts_path=out_dir / "insts.bin", xclbin_path=out_dir / "probe.xclbin",
                        work_dir=work, device=iron.get_current_device())
    insts = (out_dir / "insts.bin").stat().st_size
    (out_dir / "meta.json").write_text(json.dumps({"k": k, "q": q, "shape": shape, "cols": cols, "sizes": list(sizes),
                                                   "strides": list(strides), "tasks": cols * ntasks,
                                                   "abytes": abytes, "core_objects": cols * q * QUAD // abytes,
                                                   "total_bytes": total, "moved_bytes": moved, "check": check,
                                                   "insts_bytes": insts}), encoding="utf-8")
    return out_dir


def time_one(k: int, q: int, iters: int, warmup: int, timeout_ms: int, shape: str = "contig", cols: int = COLS,
             abytes: int = A_BYTES, check: bool = False) -> dict:
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    d = BUILD / name_of(k, q, shape, cols, abytes, check)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    total = meta["total_bytes"]
    moved = meta.get("moved_bytes", total)   # builds before the shape axis moved their whole buffer
    out_bytes = cols * q * QUAD if check else 64
    # A check needs data with no period: the timing grid's arange repeats every 256 B, the o4/box row pitch.
    data = (np.random.default_rng(1234).integers(0, 256, total, dtype=np.uint8) if check
            else np.arange(total, dtype=np.uint32).astype(np.uint8))
    exact = None
    with XrtSiliconHarness(0) as h:
        pyxrt = h.pyxrt
        h.load_xclbin(str(d / "probe.xclbin"), "MLIR_AIE")
        bo_instr, n_instr = h.create_instruction_bo_from_bytes((d / "insts.bin").read_bytes())
        bo_in = h.create_host_bo(total, 3)
        bo_out = h.create_host_bo(out_bytes, 4)
        bo_in.write(data, 0)
        bo_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = pyxrt.run(h.kernel)
        for i, arg in enumerate((3, bo_instr, n_instr, bo_in, bo_out)):
            run.set_arg(i, arg)
        done = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
        times = []
        for i in range(warmup + iters):
            t0 = time.perf_counter()
            run.start()
            state = run.wait(timeout_ms)
            dt = (time.perf_counter() - t0) * 1e3
            if state != done:
                raise RuntimeError(f"k{k} q{q}: dispatch {i} ended in state {state}")
            if i >= warmup:
                times.append(dt)
        if check:
            bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            got = np.frombuffer(bytes(bo_out.read(out_bytes, 0)), dtype=np.uint8)
            want = np.concatenate([expected_objects(data, c * (total // cols), q) for c in range(cols)])
            exact = {"bytes": int(want.size), "mismatched": int((got != want).sum())}
        del run, bo_in, bo_out, bo_instr
    return {"k": k, "q": q, "shape": shape, "cols": cols, "tasks": meta["tasks"], "mb": moved / 1e6,
            "check": check, "exact": exact,
            "abytes": meta.get("abytes", A_BYTES), "core_objects": meta.get("core_objects", moved // A_BYTES),
            "insts_bytes": meta["insts_bytes"],
            "mean_ms": statistics.fmean(times), "median_ms": statistics.median(times), "min_ms": min(times),
            "times_ms": times}


def fit(rows):
    """Least squares: mean_ms = a + b * mb + c * tasks."""
    X = np.array([[1.0, r["mb"], r["tasks"]] for r in rows])
    y = np.array([r["mean_ms"] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef, r2, y - pred


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    points_help = ("comma list of shape:k:q[:cols[:abytes]] points (cols defaults to 4, abytes to 6400) (shapes: " + ", ".join(SHAPES) + "); replaces the --ks x --qs "
                   "grid of contiguous tasks, and skips the plane fit")
    b = sub.add_parser("build")
    b.add_argument("--ks", default=",".join(map(str, KS)))
    b.add_argument("--qs", default=",".join(map(str, QS)))
    b.add_argument("--points", default=None, help=points_help)
    check_help = "copying cores: every object the cores receive comes back to DDR and is compared exactly (o4/box)"
    b.add_argument("--check", action="store_true", help=check_help)
    t = sub.add_parser("time", help="one measurement; used by 'sit' as a child process")
    t.add_argument("--check", action="store_true", help=check_help)
    t.add_argument("k", type=int)
    t.add_argument("q", type=int)
    t.add_argument("--shape", default="contig", choices=sorted(SHAPES))
    t.add_argument("--cols", type=int, default=COLS)
    t.add_argument("--iters", type=int, default=30)
    t.add_argument("--warmup", type=int, default=3)
    t.add_argument("--timeout-ms", type=int, default=5000)
    t.add_argument("--abytes", type=int, default=A_BYTES)
    s = sub.add_parser("sit")
    s.add_argument("--ks", default=",".join(map(str, KS)))
    s.add_argument("--qs", default=",".join(map(str, QS)))
    s.add_argument("--points", default=None, help=points_help)
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--iters", type=int, default=30)
    s.add_argument("--json", required=True)
    s.add_argument("--check", action="store_true", help=check_help)
    args = ap.parse_args()

    if args.cmd == "time":
        print(json.dumps(time_one(args.k, args.q, args.iters, args.warmup, args.timeout_ms, args.shape, args.cols,
                                  args.abytes, args.check)))
        return
    if args.points:
        combos = []
        for p in args.points.split(","):
            shape, k, q, *rest = p.split(":")
            if shape not in SHAPES:
                raise SystemExit(f"unknown shape {shape!r}")
            combos.append((shape, int(k), int(q), int(rest[0]) if rest else COLS,
                           int(rest[1]) if len(rest) > 1 else A_BYTES))
    else:
        combos = [("contig", k, q, COLS, A_BYTES) for q in map(int, args.qs.split(","))
                  for k in map(int, args.ks.split(","))]
    if args.cmd == "build":
        for shape, k, q, cols, ab in combos:
            t0 = time.perf_counter()
            d = build_one(k, q, shape, cols, ab, args.check)
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            print(f"[build] {d.name}: {meta['tasks']} tasks, insts {meta['insts_bytes']:,} B "
                  f"in {time.perf_counter() - t0:.1f} s", flush=True)
        return

    results = []
    for rnd in range(args.rounds):
        order = combos[rnd % len(combos):] + combos[:rnd % len(combos)]
        if rnd % 2:
            order = order[::-1]
        for shape, k, q, cols, ab in order:
            proc = subprocess.run([sys.executable, __file__, "time", str(k), str(q), "--shape", shape, "--cols", str(cols),
                                   "--abytes", str(ab), "--iters", str(args.iters)] + (["--check"] if args.check else []),
                                  capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
                raise SystemExit(f"[sit] {shape} k{k} q{q} failed in round {rnd}; stopping - do not retry in a loop")
            rec = json.loads(proc.stdout.strip().splitlines()[-1])
            rec["round"] = rnd
            results.append(rec)
            if rec.get("exact") and rec["exact"]["mismatched"]:
                raise SystemExit(f"[sit] {shape} k{k} q{q}: {rec['exact']['mismatched']} of {rec['exact']['bytes']} "
                                 f"bytes differ from the expected core objects; its time does not count")
            print(f"[sit] round {rnd} {shape:6s} k{k:<3} q{q:<5} c{cols} a{ab} {rec['tasks']:5d} tasks {rec['core_objects']:7d} objects {rec['mb']:6.1f} MB: "
                  f"mean {rec['mean_ms']:.3f} ms, min {rec['min_ms']:.3f}", flush=True)
    if args.points:
        by = {}
        for r in results:
            by.setdefault((r["shape"], r["k"], r["q"], r.get("cols", COLS), r.get("abytes", A_BYTES)), []).append(r["mean_ms"])
        for (shape, k, q, cols, ab), v in by.items():
            print(f"[point] {shape:6s} k{k:<3} q{q:<5} c{cols} a{ab} mean of rounds {statistics.fmean(v):.3f} ms "
                  f"(round spread {(max(v) - min(v)) * 1e3:.0f} us)")
        Path(args.json).write_text(json.dumps({"results": results}, indent=1), encoding="utf-8")
        return
    coef, r2, resid = fit(results)
    a, b, c = coef
    print(f"\n[fit] dispatch ms = {a:.4f} + {b:.5f} * MB + {c * 1e3:.4f} us * tasks   (R2 {r2:.4f}, "
          f"{len(results)} points, max |residual| {np.abs(resid).max() * 1e3:.1f} us)")
    print(f"[fit] per task {c * 1e3:.3f} us; per-byte rate {1 / b:.2f} GB/s across the array")
    Path(args.json).write_text(json.dumps({"results": results, "fit": {"a_ms": a, "b_ms_per_mb": b,
                                                                       "c_ms_per_task": c, "r2": r2}},
                                          indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
