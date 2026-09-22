"""IRON transport for the Phoenix packet-driven bf16 convolution engine.

Sixteen persistent workers on columns 0..3, rows 2..5 run the same core program
(engine_bf16.cc). Per column: one weight ObjectFifo broadcast from the shim to the four
cores, one activation ObjectFifo split at the MemTile into four 12,800-byte per-core
packets, and one output ObjectFifo joined at the MemTile from four 3,200-byte objects.
The runtime sequence is supplied by the caller, exactly as in the int8 transport.

Forked from ``kernels/aie2/conv_engine/design.py`` rather than parameterised from it.
Packet element types are compile-time in IRON - they are baked into every ObjectFifo, every
buffer descriptor and the kernel's own signature - so a bf16 engine needs its own xclbin,
and a design that serves both would be a type switch at every line that matters. What IS
shared is the layer above: ``engine_schedule`` plans placement and packing against a
geometry object, so only this transport is duplicated.

Deltas from the int8 design, each of which is a correctness trap if copied across:

* **Offsets into a split or a join are counted in ELEMENTS, not bytes.** int8 cannot tell
  the difference, because its column type is ``uint8`` and the two coincide; that is
  exactly why the int8 file reads ``r * A_BYTES`` and is right anyway. IRON hands these
  straight to ``aie.objectfifo.link``, and ``DMABDOp::getOffsetInBytes()`` multiplies by
  the buffer's element width, so a bf16 fork that copied ``r * A_BYTES`` would start core 2
  at the end of the column buffer and core 3 past it.
* **``H_COUNT_OUT`` / ``H_COUNT_ACC`` are 5 and 6, not 15 and 16.** The bf16 header is a
  different layout, not a subset. ``_core_fn`` drives its run-time trip counts from these
  words, so the wrong index does not fail to compile - it runs the wrong number of packets.
* **The kernel takes six arguments.** ``scratch`` is a real buffer the core reads and
  writes (the held tile for ``F_HOLD`` and ``OP_RESIDUAL``), where the int8 kernel takes
  five and the int8 design's ``scratch`` is only a dummy output sink. See ``_core_fn``.
* **psum is 6,400 B, not 16,000.** It holds 1,600 float32 partial sums and nothing else;
  the int8 buffer carries a 3,200-byte held tile in its tail, which is what the separate
  ``scratch`` argument replaces. That move is what makes the L1 budget close.
* **There is nothing to gate.** The int8 kernel compiles out four dispatch groups no
  container reaches; bf16 implements exactly OP_CONV, OP_RESIDUAL and OP_NOP, so there is
  no ``GATEABLE_OPS`` here and no ``ops`` parameter to ``build_program``.

The activation ring and the resident weight buffer are deliberately absent. Both were built
and measured on the int8 engine and both were slower (docs/BENCHMARKS.md); forking dead
levers would double the surface with no measurement behind it.
"""
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects import arith, memref
from aie.ir import IndexType
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config

COLS = 4
ROWS = 4
CORES = COLS * ROWS

# Byte sizes of one packet, and the element counts the IRON types are declared in. Both are
# spelled out because the two differ for every buffer here except the weight packet, and the
# byte figure is the one the L1 budget is argued in.
W_BYTES = 9472          # 128 header + 128 bias + 9,216 weight payload (4,608 bf16)
A_BYTES = 12800
O_BYTES = 3200
PSUM_BYTES = 6400       # 1,600 float32 partial sums; the held tile lives in `scratch`

W_WORDS = W_BYTES // 4          # the header is int32 words, so the packet is typed int32
A_ELEMS = A_BYTES // 2
O_ELEMS = O_BYTES // 2
PSUM_FLOATS = PSUM_BYTES // 4
SCRATCH_ELEMS = O_ELEMS         # a held tile IS an emitted tile, so it has the output's type

# Header words the core's run-time loops are driven from. bf16's header is its own layout:
# these are 15 and 16 in the int8 design and 5 and 6 here.
H_COUNT_OUT = 5
H_COUNT_ACC = 6

# Declared DDR argument extents. The buffers passed at runtime may be smaller or larger; the
# verifier only checks that every BD stays inside these.
WS_BYTES = 48 * 1024 * 1024
WP_BYTES = 16 * 1024 * 1024

KERNEL_SOURCE = Path(__file__).with_name("engine_bf16.cc")

w_ty = np.ndarray[(W_WORDS,), np.dtype[np.int32]]
a_ty = np.ndarray[(A_ELEMS,), np.dtype[bfloat16]]
a_col_ty = np.ndarray[(ROWS * A_ELEMS,), np.dtype[bfloat16]]
o_ty = np.ndarray[(O_ELEMS,), np.dtype[bfloat16]]
o_col_ty = np.ndarray[(ROWS * O_ELEMS,), np.dtype[bfloat16]]
psum_ty = np.ndarray[(PSUM_FLOATS,), np.dtype[np.float32]]


def fifo_names(col):
    return {"w": f"w{col}", "a": f"a{col}", "o": f"o{col}"}


def _core_fn(w_in, a_in, o_out, engine, psum, scratch, row):
    """One core's program: `count_out` emitting packets, then `count_acc` accumulating ones.

    ``scratch`` is passed BOTH as the kernel's output pointer and as its scratch pointer in
    the accumulate-only loop, and that aliasing is deliberate. A second 3,200-byte buffer
    would not fit - the budget closes with 2,944 B to spare - and it is not needed, because
    of the three things a packet can do to memory, none of them is a write through the
    aliased output pointer:

      * neither F_EMIT nor F_HOLD: the core writes partial sums to psum and nothing else,
      * F_HOLD: the core writes the activated tile to `scratch`, which is where it belongs,
      * F_EMIT: the core writes to `out` - and such a packet belongs in the FIRST loop,
        which holds a real output object.

    That last line is a contract with the packer, and it is a sharper one than int8's. There,
    `scratch` was a dead sink and an emitting packet mis-counted into the accumulate loop
    wrote to nowhere. Here it would write the emitted tile straight over the held tile that a
    later OP_RESIDUAL is going to read, and the result would be wrong without being obviously
    wrong. Any packet counted in `count_acc` must therefore be non-emitting, and the
    schedule asserts that rather than assuming it.
    """
    w = w_in.acquire(1)
    idx_ty = IndexType.get()
    n_out = memref.load(w, [arith.constant(idx_ty, H_COUNT_OUT)])
    n_acc = memref.load(w, [arith.constant(idx_ty, H_COUNT_ACC)])
    n_out = arith.index_cast(idx_ty, n_out)
    n_acc = arith.index_cast(idx_ty, n_acc)
    for _ in range_(n_out):
        a = a_in.acquire(1)
        o = o_out.acquire(1)
        engine(w, a, o, psum, scratch, row)
        a_in.release(1)
        o_out.release(1)
    for _ in range_(n_acc):
        a = a_in.acquire(1)
        engine(w, a, scratch, psum, scratch, row)
        a_in.release(1)
    w_in.release(1)


def ddr_extents(workspace_bytes, packet_bytes):
    """Declared (workspace, packet) DDR extents: the defaults, raised to the next MiB for larger models."""
    mib = 1 << 20
    return (max(WS_BYTES, -(-int(workspace_bytes) // mib) * mib),
            max(WP_BYTES, -(-int(packet_bytes) // mib) * mib))


def build_program(device, sequence_body, w_depth=2, ws_bytes=WS_BYTES, wp_bytes=WP_BYTES):
    """Return an IRON Program for the bf16 engine with ``sequence_body(ws, wp)``.

    The body emits raw shim DMA tasks against the FIFO allocation symbols ``fifo_names(c)``.
    ``ws_bytes`` and ``wp_bytes`` are the declared DDR extents every task must stay inside.

    ``ExternalFunction`` is spelled exactly as ``kernels/bf16_conv/engine_bf16.py`` spells
    it - same source, same include dirs, no compile flags - so that the object linked here
    is byte-for-byte the object the single-core harness took to silicon. ExternalFunction's
    content digest covers the source text and the sorted compile flags, so adding a flag
    here (the int8 design passes ``-O2``) would quietly produce a DIFFERENT kernel from the
    one the 13-case byte-exactness result is evidence about.
    """
    engine = ExternalFunction(
        "engine_bf16",
        source_file=str(KERNEL_SOURCE),
        arg_types=[w_ty, a_ty, o_ty, psum_ty, o_ty, np.int32],
        object_file_name="engine_bf16.o",
        include_dirs=[config.cxx_header_path()],
    )
    workers = []
    w_prods, a_prods, o_conses = [], [], []
    for c in range(COLS):
        names = fifo_names(c)
        w_of = ObjectFifo(w_ty, name=names["w"], depth=w_depth)
        a_col = ObjectFifo(a_col_ty, name=names["a"], depth=2)
        # ELEMENTS, not bytes - see the module docstring.
        a_split = a_col.cons().split(
            [r * A_ELEMS for r in range(ROWS)],
            tile=Tile(c, 1),
            depths=[2] * ROWS,
            obj_types=[a_ty] * ROWS,
            names=[f"a{c}_{r}" for r in range(ROWS)],
        )
        o_col = ObjectFifo(o_col_ty, name=names["o"], depth=2)
        o_join = o_col.prod().join(
            [r * O_ELEMS for r in range(ROWS)],
            tile=Tile(c, 1),
            depths=[2] * ROWS,
            obj_types=[o_ty] * ROWS,
            names=[f"o{c}_{r}" for r in range(ROWS)],
        )
        for r in range(ROWS):
            psum = Buffer(psum_ty, name=f"psum_{c}_{r}")
            scratch = Buffer(o_ty, name=f"scratch_{c}_{r}")
            workers.append(
                Worker(
                    _core_fn,
                    [w_of.cons(), a_split[r].cons(), o_join[r].prod(), engine, psum, scratch, r],
                    tile=Tile(c, r + 2),
                    # Peano's default 1 KB stack grows UPWARD into the tile buffers and
                    # corrupts them silently; 2 KB is what both engines use.
                    stack_size=0x800,
                )
            )
        w_prods.append(w_of.prod(tile=Tile(c, 0)))
        a_prods.append(a_col.prod(tile=Tile(c, 0)))
        o_conses.append(o_col.cons(tile=Tile(c, 0)))

    def sequence(ws, wp, w_p, a_p, o_c):
        # The handles only register the shim endpoints; transfers are emitted by name.
        sequence_body(ws, wp)

    ws_t = np.ndarray[(int(ws_bytes),), np.dtype[np.uint8]]
    wp_t = np.ndarray[(int(wp_bytes),), np.dtype[np.uint8]]
    rt = Runtime(sequence, [ws_t, wp_t, w_prods, a_prods, o_conses])
    return Program(device, rt, workers=workers)
