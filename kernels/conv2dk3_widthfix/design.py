# design.py -*- Python -*-
"""Isolated single-worker conv2dk3 (3x3 int8 conv) test design.

WHY THIS EXISTS
---------------
Not upstream mlir-aie code -- lives in this repo. `docs/DECISIONS.md` and
`kernels/README.md` (2026-09-07) record that `conv2dk3_i8_vector`/`_ui8_vector` in
mlir-aie's `aie_kernels/aie2/conv2dk3.cc` hardcode `const int iw = 32;` and ignore
their own `input_width` argument in ~20 pointer-stride computations, per the kernel
author's own TODO ("results are wrong. ???"). `bottleneck.py` never exercises a width
other than 32 anyway -- its Tile(0,4) skip-add core L1 budget rejects every wider
shape at `aiecc` compile time, before this bug could ever run.

This design isolates JUST the conv2dk3 worker -- no conv1, no skip-add, no channel
split across two workers -- so it can compile and run at widths bottleneck.py cannot
reach, and so a width-32 fix can be verified (or refuted) without the unrelated L1
ceiling in the way. One worker, full 64 channels in and out, pinned to Tile(0,3) (the
same tile bottleneck.py uses for its 3x3 stage, to reuse its documented placer-safe
slot -- see bottleneck.py's own docstring on the auto-placer limitation).

USAGE (mlir-aie ironenv -- see kernels/README.md for PATH/PYTHONPATH setup)
    python3 test_width.py --shapes 32x32,32x64,32x96
"""

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.device import Tile

CHANNELS = 64


@iron.jit
def conv3x3_only(
    a_in: In,
    w_in: In,
    b_out: Out,
    *,
    tensor_w: CompileTime[int] = 32,
    tensor_h: CompileTime[int] = 32,
    scale: CompileTime[int] = 11,
):
    channels = CHANNELS
    act_ty = np.ndarray[(tensor_w * tensor_h * channels,), np.dtype[np.uint8]]
    wts_ty = np.ndarray[(3 * 3 * channels * channels,), np.dtype[np.int8]]

    row_in_ty = np.ndarray[(tensor_w, 1, channels), np.dtype[np.uint8]]
    row_out_ty = np.ndarray[(tensor_w, 1, channels), np.dtype[np.uint8]]

    conv3 = kernels.conv2dk3(
        input_width=tensor_w,
        input_channels=channels,
        output_channels=channels,
        weight_output_channels=channels,
        act_dtype=np.uint8,
    )

    of_act = ObjectFifo(row_in_ty, name="act_in")
    of_wts = ObjectFifo(wts_ty, depth=1, name="wts_in")
    of_out = ObjectFifo(row_out_ty, name="act_out")

    def conv3x3_fn(of_wts, of_act_in, of_act_out, conv3x3):
        elem_wts = of_wts.acquire(1)

        # top row (zero border above)
        elems_in = of_act_in.acquire(2)
        elem_out = of_act_out.acquire(1)
        conv3x3(
            elems_in[0], elems_in[0], elems_in[1], elem_wts, elem_out,
            tensor_w, channels, channels, 3, 3, 0, scale, 0,
        )
        of_act_out.release(1)

        # middle rows
        for _ in range_(tensor_h - 2):
            elems_in = of_act_in.acquire(3)
            elem_out = of_act_out.acquire(1)
            conv3x3(
                elems_in[0], elems_in[1], elems_in[2], elem_wts, elem_out,
                tensor_w, channels, channels, 3, 3, 1, scale, 0,
            )
            of_act_in.release(1)
            of_act_out.release(1)

        # bottom row (zero border below)
        elems_in = of_act_in.acquire(2)
        elem_out = of_act_out.acquire(1)
        conv3x3(
            elems_in[0], elems_in[1], elems_in[1], elem_wts, elem_out,
            tensor_w, channels, channels, 3, 3, 2, scale, 0,
        )
        of_act_in.release(2)
        of_act_out.release(1)
        of_wts.release(1)

    workers = [
        Worker(
            conv3x3_fn,
            fn_args=[of_wts.cons(), of_act.cons(4), of_out.prod(), conv3],
            tile=Tile(0, 3),
        )
    ]

    def sequence(inp, w, out, act_prod, wts_prod, out_cons):
        act_prod.fill(inp)
        wts_prod.fill(w)
        out_cons.drain(out, wait=True)

    rt = Runtime(
        sequence,
        [act_ty, wts_ty, act_ty, of_act.prod(), of_wts.prod(), of_out.cons()],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()
