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
from aie.ir import IndexType
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
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


def build_program(device, sequence_body, w_depth=1):
    """Return an IRON Program for the engine with ``sequence_body(ws, wp)``.

    The body emits raw shim DMA tasks against the FIFO allocation symbols
    ``fifo_names(c)`` (see ``ignite_xdna.compiler.engine_sequence``).
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
    for c in range(COLS):
        names = fifo_names(c)
        w_of = ObjectFifo(w_ty, name=names["w"], depth=w_depth)
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
            workers.append(
                Worker(
                    _core_fn,
                    [w_of.cons(), a_split[r].cons(), o_join[r].prod(), engine, psum, scratch, r],
                    tile=Tile(c, r + 2),
                    stack_size=0x400,
                )
            )
        w_prods.append(w_of.prod(tile=Tile(c, 0)))
        a_prods.append(a_col.prod(tile=Tile(c, 0)))
        o_conses.append(o_col.cons(tile=Tile(c, 0)))

    def sequence(ws, wp, w_p, a_p, o_c):
        # The handles only register the shim endpoints; transfers are emitted by name.
        sequence_body(ws, wp)

    rt = Runtime(sequence, [ws_ty, wp_ty, w_prods, a_prods, o_conses])
    return Program(device, rt, workers=workers)
