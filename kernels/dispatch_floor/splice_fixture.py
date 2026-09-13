"""Compile synthetic native bf16 Conv2D graph stages for test_kernel_splice.

This is a fixed 1x1 depthwise Conv ABI fixture, not a general ONNX lowerer.
The GroupNorm stage is the existing kernels/groupnorm_bf16 implementation.
"""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.utils import config
from ml_dtypes import bfloat16


@iron.jit
def conv2d(x: In, y: Out, *, L: CompileTime[int], chunk: CompileTime[int],
           weight: CompileTime[float], bias: CompileTime[float]):
    if L <= 0 or chunk < 16 or chunk % 16 or L % chunk:
        raise ValueError("L must be a positive multiple of a vector-aligned chunk")
    tensor_ty = np.ndarray[(32 * L,), np.dtype[bfloat16]]
    chunk_ty = np.ndarray[(chunk,), np.dtype[bfloat16]]
    kernel = ExternalFunction(
        "splice_conv", arg_types=[chunk_ty, chunk_ty],
        source_file=str(Path(__file__).with_name("splice_conv.cc")),
        object_file_name="splice_conv.o", include_dirs=[config.cxx_header_path()],
        compile_flags=[f"-DCHUNK={chunk}", f"-DWEIGHT={weight}", f"-DBIAS={bias}"],
    )

    def core_fn(fi, fo, compute):
        for _ in range_(4 * L // chunk):
            a = fi.acquire(1)
            b = fo.acquire(1)
            compute(a, b)
            fi.release(1)
            fo.release(1)

    workers, inputs, outputs, taps = [], [], [], []
    for k in range(8):
        col, row = k // 2, 2 + k % 2
        fi, fo = ObjectFifo(chunk_ty, name=f"in{k}"), ObjectFifo(chunk_ty, name=f"out{k}")
        workers.append(Worker(core_fn, fn_args=[fi.cons(), fo.prod(), kernel], tile=Tile(col, row)))
        inputs.append(fi.prod(tile=Tile(col, 0)))
        outputs.append(fo.cons(tile=Tile(col, 0)))
        taps.append(TensorAccessPattern((32 * L,), k * 4 * L, [1, 1, 1, 4 * L], [0, 0, 0, 1]))

    def sequence(a, b, ih, oh):
        for k in range(8):
            ih[k].fill(a, taps[k])
            oh[k].drain(b, taps[k], wait=True)

    return Program(iron.get_current_device(),
                   Runtime(sequence, [tensor_ty, tensor_ty, inputs, outputs]),
                   workers=workers).resolve_program()


def compile_fixtures(directory: Path, L: int, chunk: int):
    from kernels.groupnorm_bf16.groupnorm import groupnorm32
    from aie.iron.device import NPU1

    iron.set_current_device(NPU1())
    artifacts = {}
    for name, design in (
        ("conv_up", conv2d.compilable.specialize(L=L, chunk=chunk, weight=0.5, bias=0.25)),
        ("groupnorm", groupnorm32.compilable.specialize(L=L, chunk=chunk)),
        ("conv_down", conv2d.compilable.specialize(L=L, chunk=chunk, weight=2.0, bias=1.0)),
    ):
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        xclbin, insts = target / "design.xclbin", target / "insts.bin"
        # Explicit output paths: no cache-directory guesses or stale name keys.
        print(f"COMPILE_BEGIN stage={name} L={L} chunk={chunk}", flush=True)
        artifacts[name] = design.compile(xclbin_path=xclbin, inst_path=insts)
        print(f"COMPILE_END stage={name}", flush=True)
    return artifacts
