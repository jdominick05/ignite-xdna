"""Trace cycles for an on-chip MemTile-to-core 32-bit stream, without DDR payloads."""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/silicon_onchip_stream_probe"


def module():
    return '''module { aie.device(npu1) {
    %core = aie.tile(0, 2)
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %ones = aie.buffer(%mem) {address = 65536 : i32, initial_value = dense<1> : tensor<16384xi32>} : memref<16384xi32>
    aie.flow(%mem, DMA : 0, %core, Core : 0)
    aie.memtile_dma(%mem) {
      aie.dma_start("MM2S", 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%ones : memref<16384xi32> offset = 0 len = 16384)
      aie.next_bd ^bd
    ^end:
      aie.end
    }
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    func.func private @onchip_stream(memref<64xi32>, memref<64xi32>) attributes {link_with = "onchip.o"}
    aie.core(%core) {
      %zero = arith.constant 0 : index
      %end = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      scf.for %i = %zero to %end step %one {
        %in = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
        %a = aie.objectfifo.subview.access %in[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        %out = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
        %b = aie.objectfifo.subview.access %out[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        func.call @onchip_stream(%a, %b) : (memref<64xi32>, memref<64xi32>) -> ()
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }
    aie.trace @stream_trace(%core) {
      aie.trace.mode "Event-Time"
      aie.trace.packet type = core
      aie.trace.event <"INSTR_EVENT_0">
      aie.trace.event <"INSTR_EVENT_1">
      aie.trace.start broadcast = 15
      aie.trace.stop broadcast = 14
    }
    aie.runtime_sequence(%x: memref<64xi32>, %y: memref<64xi32>) {
      aie.trace.host_config {buffer_size = 65536 : i32}
      aie.trace.start_config @stream_trace
      %in = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      %out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%y : memref<64xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out)
      aiex.dma_start_task(%in)
      aiex.dma_await_task(%out)
      aiex.dma_free_task(%in)
      aiex.dma_free_task(%out)
    }
  }}'''


def build():
    from aie.utils.compile.utils import compile_cxx_core_function, compile_mlir_module
    from aie.utils import config
    work = BUILD / "design.prj"
    work.mkdir(parents=True, exist_ok=True)
    compile_cxx_core_function(str(ROOT / "kernels/silicon_stream/onchip.cc"), "aie2", str(work / "onchip.o"))
    print(subprocess.check_output([str(Path(config.peano_install_dir()) / "bin/llvm-objdump.exe"), "-d", str(work / "onchip.o")], text=True))
    ir = module()
    (BUILD / "probe.mlir").write_text(ir, encoding="utf-8")
    compile_mlir_module(ir, insts_path=BUILD / "insts.bin", xclbin_path=BUILD / "probe.xclbin", work_dir=work)


def run(words):
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    sys.path.insert(0, str(ROOT / "kernels/clock_probe"))
    from clock_probe import TraceReader
    from aie.utils.trace import TraceConfig
    from aie.utils import config
    print("ONCHIP_STREAM: MemTile buffer -> DMA0 -> core Core0; no DDR payload in the timed interval.")
    for name in ("probe.mlir", "insts.bin", "design.prj/onchip.o"):
        print("SHA256", name, hashlib.sha256((BUILD / name).read_bytes()).hexdigest())
    print(subprocess.check_output([str(Path(config.peano_install_dir()) / "bin/llvm-objdump.exe"), "-d", str(BUILD / "design.prj/onchip.o")], text=True))
    tc = TraceConfig(trace_size=65536, trace_file=str(BUILD / "trace.txt"))
    reader = TraceReader(tc)
    rows = []
    with XrtSiliconHarness(0) as h:
        xrt = h.pyxrt
        h.load_xclbin(str(BUILD / "probe.xclbin"), "MLIR_AIE")
        instr, count = h.create_instruction_bo_from_bytes((BUILD / "insts.bin").read_bytes())
        inp = h.create_host_bo(256, 3)
        out = h.create_host_bo(256, 4)
        trace = h.create_host_bo(65536, 5)
        task = xrt.run(h.kernel)
        for i, val in enumerate((3, instr, count, inp, out, trace)):
            task.set_arg(i, val)
        for words in (words,):
            for trial in range(1):
                params = np.zeros(64, dtype=np.uint32)
                params[0] = words
                inp.write(params, 0)
                out.write(np.zeros(64, dtype=np.uint32), 0)
                trace.write(np.zeros(65536//4, dtype=np.uint32), 0)
                inp.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                trace.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                task.start()
                state = task.wait(3000)
                assert state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED, state
                # Output completion and the trace stream have separate DMA queues.
                time.sleep(.02)
                out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                trace.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                data = np.frombuffer(out.read(256, 0), dtype=np.uint32)
                assert data[0] == 1 and data[1] == words, data[:2]
                tc.write_trace(np.frombuffer(trace.read(65536, 0), dtype=np.uint32))
                t0, t1, pairs, events = reader.stamps()
                assert t0 is not None and t1 > t0
                rec = {"words": words, "bytes": words*4, "cycles": t1-t0, "last_word": int(data[0]), "trial": trial}
                rows.append(rec)
                print("ONCHIP_POINT", json.dumps(rec), flush=True)
        del task, inp, out, trace, instr
    gc.collect()


def suite():
    from silicon_probe_record import witness
    print("Fresh context per trace sample; source disassembly shows 16 scalar stream reads per 16-word hardware-loop iteration.")
    rows = []
    for words in (1 << 18, 1 << 20, 1 << 22):
        for trial in range(3):
            print("PRE_ONCHIP", words, trial, flush=True)
            witness()
            try:
                p = subprocess.run([sys.executable, __file__, "run", "--words", str(words)], capture_output=True, text=True, timeout=30)
                print(p.stdout, p.stderr, flush=True)
            finally:
                print("POST_ONCHIP", words, trial, flush=True)
                witness()
            assert p.returncode == 0, p.returncode
            rec = json.loads(next(s.split(" ", 1)[1] for s in p.stdout.splitlines() if s.startswith("ONCHIP_POINT ")))
            rows.append(rec)
    slope, intercept = np.polyfit([r["words"] for r in rows], [r["cycles"] for r in rows], 1)
    assert abs(slope - 1) < .001
    print("ONCHIP_FIT", json.dumps({"cycles_per_32bit_word": slope, "bytes_per_cycle": 4/slope, "intercept_cycles": intercept}), flush=True)
    print("VERDICT: on-chip single-stream 32-bit word/cycle supported by trace slope; opens independent-stream scaling, closes a wider-per-stream engine assumption.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("build", "run", "suite"))
    ap.add_argument("--words", type=int, default=1 << 18)
    args = ap.parse_args()
    if args.mode == "build":
        build()
    elif args.mode == "run":
        run(args.words)
    else:
        suite()
