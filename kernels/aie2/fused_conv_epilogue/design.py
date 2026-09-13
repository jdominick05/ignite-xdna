"""Explicit 4-column / 16-core transport for the fused epilogue fixture."""
from pathlib import Path

import numpy as np
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.utils import config
from aie.utils.trace.events import CoreEvent, PortEvent
from aie.dialects.aie import WireBundle

PACKET = 11904
OUTPUT = 256
CORES = 16


@iron.jit
def fused_design(x: In, y: Out, *, fused: CompileTime[int],
                 cores: CompileTime[int] = 16,
                 cin: CompileTime[int] = 32,
                 trace_size: CompileTime[int] = 65536):
    assert cores in (1, 16)
    ncols, nrows = (4, 4) if cores == 16 else (1, 1)
    packet = 3264 if cin > 32 else PACKET
    panels = cin // 8 if cin > 32 else 1
    depth = 2 if cin > 32 else 1
    activation_bytes = 960 if cin > 32 else 2304
    input_ty = np.ndarray[(cores * packet * panels,), np.dtype[np.int8]]
    output_ty = np.ndarray[(cores * OUTPUT,), np.dtype[np.int8]]
    packet_ty = np.ndarray[(packet,), np.dtype[np.int8]]
    result_ty = np.ndarray[(OUTPUT,), np.dtype[np.int8]]
    activation_ty = np.ndarray[(activation_bytes,), np.dtype[np.int8]]
    column_in_ty = np.ndarray[(nrows * packet,), np.dtype[np.int8]]
    column_out_ty = np.ndarray[(nrows * OUTPUT,), np.dtype[np.int8]]
    name = 'conv2d_fused' if fused else 'conv2d_raw'
    fn = ExternalFunction(name, object_file_name=name + '.o',
            source_file=str(Path(__file__).with_name('conv2d_fused.cc')),
            arg_types=[packet_ty, result_ty, activation_ty], include_dirs=[config.cxx_header_path()],
            compile_flags=['-O3', '-DCHUNKED_OUTPUT=1',
                           f'-DCIN={cin}', f'-DFUSED={fused}', f'-DKERNEL_NAME={name}'])

    def core(ih, oh, compute, activation_bank):
        a = ih.acquire(1)
        b = oh.acquire(1)
        compute(a, b, activation_bank)
        ih.release(1)
        oh.release(1)

    workers, inputs, outputs, intaps, outtaps = [], [], [], [], []
    for col in range(ncols):
        fi = ObjectFifo(column_in_ty, name=f'input_c{col}', depth=depth)
        fo = ObjectFifo(column_out_ty, name=f'output_c{col}', depth=1)
        ci = fi.cons().split([r * packet for r in range(nrows)], tile=Tile(col, 1),
                             depths=[depth] * nrows, obj_types=[packet_ty] * nrows)
        co = fo.prod().join([r * OUTPUT for r in range(nrows)], tile=Tile(col, 1),
                            depths=[1] * nrows, obj_types=[result_ty] * nrows)
        for row in range(nrows):
            bank = Buffer(activation_ty, name=f'activations_{col}_{row}', address=0x4000)
            workers.append(Worker(core, [ci[row].cons(), co[row].prod(), fn, bank],
                tile=Tile(col, row + 2), trace=1 if trace_size and col == row == 0 else None))
        inputs.append(fi.prod(tile=Tile(col, 0)))
        outputs.append(fo.cons(tile=Tile(col, 0)))
        intaps.append(TensorAccessPattern((cores * packet * panels,), col * nrows * packet * panels,
                                         [1, 1, 1, nrows * packet * panels], [0, 0, 0, 1]))
        outtaps.append(TensorAccessPattern((cores * OUTPUT,), col * nrows * OUTPUT,
                                          [1, 1, 1, nrows * OUTPUT], [0, 0, 0, 1]))

    def sequence(a, b, ih, oh):
        for col in range(ncols):
            ih[col].fill(a, intaps[col])
            oh[col].drain(b, outtaps[col], wait=True)

    program = Program(iron.get_current_device(),
                      Runtime(sequence, [input_ty, output_ty, inputs, outputs]), workers=workers)
    if trace_size:
        program.enable_trace(trace_size=trace_size, workers=[workers[0]],
            coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1,
                             CoreEvent.MEMORY_STALL,
                             PortEvent(CoreEvent.PORT_RUNNING_0, WireBundle.DMA, 0, master=False)])
    return program.resolve_program()
