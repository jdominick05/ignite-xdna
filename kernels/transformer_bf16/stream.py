"""Bounded one-core packet transport for native drafter kernel qualification."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import gc
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

from .common import IRON, PEANO, INSTALL, STUDY, command, save, sha


def harness(function, packet, output, blocks, state_words=0):
    ni, no = blocks*packet, blocks*output
    state = f'    %state = aie.buffer(%t) {{sym_name = "carry_state"}} : memref<{state_words}xf32>\n' if state_words else ""
    extra_type = f", memref<{state_words}xf32>" if state_words else ""
    extra_arg = ", %state" if state_words else ""
    return f'''module {{
  aie.device(npu1) {{
    %s = aie.tile(0, 0)
    %t = aie.tile(0, 2)
{state}    func.func private @{function}(memref<{packet}xi32>, memref<{output}xi32>{extra_type})
    aie.objectfifo @inF(%s, {{%t}}, 1 : i32) : !aie.objectfifo<memref<{packet}xi32>>
    aie.objectfifo @outF(%t, {{%s}}, 1 : i32) : !aie.objectfifo<memref<{output}xi32>>
    %core = aie.core(%t) {{
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %forever = arith.constant 4294967295 : index
      scf.for %it = %zero to %forever step %one {{
        %is = aie.objectfifo.acquire @inF(Consume, 1) : !aie.objectfifosubview<memref<{packet}xi32>>
        %a = aie.objectfifo.subview.access %is[0] : !aie.objectfifosubview<memref<{packet}xi32>> -> memref<{packet}xi32>
        %os = aie.objectfifo.acquire @outF(Produce, 1) : !aie.objectfifosubview<memref<{output}xi32>>
        %o = aie.objectfifo.subview.access %os[0] : !aie.objectfifosubview<memref<{output}xi32>> -> memref<{output}xi32>
        func.call @{function}(%a, %o{extra_arg}) : (memref<{packet}xi32>, memref<{output}xi32>{extra_type}) -> ()
        aie.objectfifo.release @outF(Produce, 1)
        aie.objectfifo.release @inF(Consume, 1)
      }}
      aie.end
    }} {{link_with = "kernel.o", stack_size = 4096 : i32}}
    aie.runtime_sequence(%a: memref<{ni}xi32>, %o: memref<{no}xi32>) {{
      aiex.npu.dma_memcpy_nd(%a[0,0,0,0][1,1,{ni//16},16][0,0,16,1]) {{id = 0 : i64,metadata = @inF}} : memref<{ni}xi32>
      aiex.npu.dma_memcpy_nd(%o[0,0,0,0][1,1,{no//16},16][0,0,16,1]) {{id = 1 : i64,metadata = @outF,issue_token = true}} : memref<{no}xi32>
      aiex.npu.dma_wait {{symbol = @outF}}
    }}
  }}
}}
'''


def build(folder, source, function, packet, output, blocks, sources, contract, state_words=0, optimization="-O3"):
    folder=Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    if packet % 16 or output % 16 or blocks < 1:
        raise ValueError("Invalid stream dimensions")
    if optimization not in ("-O3","-Os","-Oz"):
        raise ValueError("Unsupported kernel optimization level")
    clang = PEANO / "bin/clang++.exe"
    command([clang, optimization, "-std=c++20", "--target=aie2-none-unknown-elf", "-DNDEBUG",
             "-fstack-size-section", "-D__AIE_API_AIE_ADF_HPP__", f"-I{INSTALL/'include'}",
             "-Wno-attributes", "-c", source, "-o", folder / "kernel.o"], folder)
    if command([PEANO / "bin/llvm-nm.exe", "-u", folder / "kernel.o"], folder).strip():
        raise ValueError("Unresolved kernel helpers")
    (folder / "harness.mlir").write_text(harness(function,packet,output,blocks,state_words), encoding="utf-8")
    command([IRON / "Scripts/aiecc.exe", folder / "harness.mlir", f"--peano={PEANO}",
             "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-xclbin",
             "--xclbin-name=final.xclbin", "--dump-intermediates", f"--tmpdir={folder/'prj'}"], folder)
    elfs = list((folder / "prj").rglob("*.elf"))
    if len(elfs) != 1:
        raise ValueError("Expected one core image")
    sections = command([PEANO / "bin/llvm-size.exe", "-A", elfs[0]], folder)
    code_bytes = sum(int(x) for x in re.findall(r"^\.text\S*\s+(\d+)\s+",sections,re.M))
    frames = command([PEANO / "bin/llvm-readobj.exe", "--stack-sizes",elfs[0]],folder)
    frames = [int(x,16) for x in re.findall(r"Size: (0x[0-9a-fA-F]+)",frames)]
    if not 0 < code_bytes <= 16384 or not frames or sum(frames)+256 > 4096:
        raise ValueError("Code or stack bound exceeds reserved memory")
    assembly = command([PEANO / "bin/llvm-objdump.exe", "-dr",elfs[0]],folder)
    (folder / "kernel.asm").write_text(assembly,encoding="utf-8")
    if re.search(r"\b__(?:add|sub|mul|div|eq|ne|lt|le|gt|ge|cmp)sf[23]\b",assembly):
        raise ValueError("Scalar software floating-point helper")
    linker = (folder / "prj/ldScripts_main_core_0_2.ld.script").read_text()
    spans = sorted((int(a,16),int(a,16)+int(n,16),name) for a,name,n in re.findall(
        r"\. = (0x[0-9A-Fa-f]+);\s*\n\s*(\w+) = \.;\s*\n\. \+= (0x[0-9A-Fa-f]+);",linker))
    if len(spans) != 3 + bool(state_words) or any(a < 0x70000 or b > 0x80000 for a,b,_ in spans):
        raise ValueError("Unexpected SRAM allocation")
    if any(a[1] > b[0] for a,b in zip(spans,spans[1:])):
        raise ValueError("Overlapping stack or buffers")
    if any(sha(path) != expected for path,expected in sources.items()):
        raise ValueError("Source changed during compilation")
    program_files=("prj/main.pdi","prj/kernels_main.json","prj/memTopology_main.json")
    meta = dict(sources=sources, contract=contract, function=function, packet_words=packet,
                output_words=output, blocks=blocks, state_words=state_words, code_bytes=code_bytes, stack_frames=frames,sram_spans=spans,
                artifacts={name:sha(folder/name) for name in ("kernel.o","harness.mlir","insts.bin","final.xclbin")},
                program_artifacts={name:sha(folder/name) for name in program_files},
                compiler_sha256=sha(clang),optimization=optimization)
    save(folder / "build.json",meta)
    print(json.dumps(dict(build=str(folder),code_bytes=code_bytes,stack_frames=frames,blocks=blocks)),flush=True)


def execute(folder, packets):
    meta = json.loads((folder / "build.json").read_text())
    for path,expected in meta["sources"].items():
        if sha(path) != expected: raise ValueError("Source changed after build")
    for name,expected in meta["artifacts"].items():
        if sha(folder/name) != expected: raise ValueError("Build artifact changed")
    if packets.dtype != np.uint32 or packets.shape != (meta["blocks"],meta["packet_words"]):
        raise ValueError("Packet dimensions differ from compiled stream")
    witness = command(["C:/Windows/System32/AMD/xrt-smi.exe","examine","-r","aie-partitions"],folder)
    if "No hardware contexts running" not in witness:
        raise RuntimeError("NPU is busy; did not dispatch")
    sys.path.insert(0,str(STUDY / "src"))
    import ignite_xdna
    if not Path(ignite_xdna.__file__).resolve().is_relative_to(STUDY / 'src'):
        raise RuntimeError('Runtime import resolved outside this checkout')
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    with XrtSiliconHarness() as h:
        ib = bi = bo = None
        try:
            h.load_xclbin(str(folder / "final.xclbin"))
            ib,nbytes = h.create_instruction_bo(str(folder / "insts.bin"))
            bi = h.create_host_bo(packets.nbytes,3)
            output_bytes = meta["blocks"]*meta["output_words"]*4
            bo = h.create_host_bo(output_bytes,4)
            to = h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            fr = h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            bi.write(packets.tobytes(),0); bi.sync(to)
            bo.write(np.full(output_bytes//4,0x7fc98765,np.uint32).tobytes(),0); bo.sync(to)
            start = time.perf_counter()
            handle,status = h.dispatch_kernel(ib,nbytes,bi,bo,timeout_ms=5000)
            elapsed = (time.perf_counter()-start)*1000
            if "COMPLETED" not in str(status):
                raise RuntimeError(f"Dispatch {status}; stop without retry")
            bo.sync(fr)
            result = np.frombuffer(bo.read(output_bytes,0),np.uint32).copy().reshape(meta["blocks"],meta["output_words"])
            bi.sync(fr)
            if not np.array_equal(np.frombuffer(bi.read(packets.nbytes,0),np.uint32),packets.reshape(-1)):
                raise ValueError("Input BO modified")
            return result,dict(dispatch_ms=elapsed,status=str(status),input_unchanged=True,
                               runtime_driver_sha256=sha(STUDY / "src/ignite_xdna/runtime/driver.py"))
        finally:
            handle = bo = bi = ib = None
            gc.collect()
