"""Direct east/west MemTile DMA access, with distinct writer/reader tiles."""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/silicon_mem_neighbour_probe"
CASES = {"local": (1, 1, 1), "west_read": (0, 1, 0), "east_read": (2, 1, 2),
         "west_write": (1, 0, 0), "east_write": (1, 2, 2)}
N = 4096


def design(writer, reader, memory):
    lines = ["module { aie.device(npu1) {"]
    for c in sorted({writer, reader, memory}):
        lines += [f"%s{c} = aie.tile({c}, 0)", f"%m{c} = aie.tile({c}, 1)"]
    lines += [f'%b = aie.buffer(%m{memory}) {{sym_name = "payload", address = 65536 : i32}} : memref<{N}xi32>',
              f'%free = aie.lock(%m{memory}, 0) {{init = 1 : i32}}',
              f'%ready = aie.lock(%m{memory}, 1) {{init = 0 : i32}}',
              f'aie.flow(%s{writer}, DMA : 0, %m{writer}, DMA : 0)',
              f'aie.flow(%m{reader}, DMA : 0, %s{reader}, DMA : 0)',
              f'aie.shim_dma_allocation @input(%s{writer}, MM2S, 0)',
              f'aie.shim_dma_allocation @output(%s{reader}, S2MM, 0)']
    for c in sorted({writer, reader}):
        channels = []
        if c == writer:
            channels.append(("S2MM", "free", "ready"))
        if c == reader:
            channels.append(("MM2S", "ready", "free"))
        lines += [f"aie.memtile_dma(%m{c}) {{", "%one = arith.constant 1 : i32"]
        for j, (direction, acq, rel) in enumerate(channels):
            if j:
                lines.append(f"^start{j}:")
            dest = f"^start{j+1}" if j + 1 < len(channels) else "^end"
            lines.append(f'aie.dma_start("{direction}", 0, ^bd{j}, {dest})')
        for j, (_, acq, rel) in enumerate(channels):
            lines += [f"^bd{j}:", f"aie.use_lock(%{acq}, AcquireGreaterEqual, %one)",
                      f"aie.dma_bd(%b : memref<{N}xi32> offset = 0 len = {N})",
                      f"aie.use_lock(%{rel}, Release, %one)", f"aie.next_bd ^bd{j}"]
        lines += ["^end:", "aie.end", "}"]
    lines += [f"aie.runtime_sequence(%x : memref<{N}xi32>, %y : memref<{N}xi32>) {{"]
    for sym, buf, token in (("input", "x", ""), ("output", "y", " {issue_token = true}")):
        lines += [f"%{sym} = aiex.dma_configure_task_for @{sym} {{",
                  f"aie.dma_bd(%{buf} : memref<{N}xi32> offset = 0 len = {N})", "aie.end", "}" + token]
    lines += ["aiex.dma_start_task(%output)", "aiex.dma_start_task(%input)",
              "aiex.dma_await_task(%output)", "aiex.dma_free_task(%input)",
              "aiex.dma_free_task(%output)", "}", "}}"]
    return "\n".join(lines)


def build():
    from aie.utils.compile.utils import compile_mlir_module
    for case, (writer, reader, memory) in CASES.items():
        d = BUILD / case
        d.mkdir(parents=True, exist_ok=True)
        (d / "design.prj").mkdir(exist_ok=True)
        module = design(writer, reader, memory)
        (d / "probe.mlir").write_text(module, encoding="utf-8")
        print("BUILD", case, flush=True)
        compile_mlir_module(module, insts_path=d / "insts.bin", xclbin_path=d / "probe.xclbin", work_dir=d / "design.prj")


def run(case, seed):
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    d = BUILD / case
    print("CASE", case, "writer,reader,memory", CASES[case], flush=True)
    for name in ("probe.mlir", "probe.xclbin", "insts.bin"):
        print(name, "SHA256", hashlib.sha256((d / name).read_bytes()).hexdigest())
    with XrtSiliconHarness(0) as h:
        xrt = h.pyxrt
        h.load_xclbin(str(d / "probe.xclbin"), "MLIR_AIE")
        instr, count = h.create_instruction_bo_from_bytes((d / "insts.bin").read_bytes())
        inp, out = h.create_host_bo(N * 4, 3), h.create_host_bo(N * 4, 4)
        task = xrt.run(h.kernel)
        for i, val in enumerate((3, instr, count, inp, out)):
            task.set_arg(i, val)
        for seed in (seed,):
            data = np.random.default_rng(seed).integers(0, 2**32, N, dtype=np.uint32)
            inp.write(data, 0)
            out.write(np.bitwise_not(data), 0)
            inp.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            task.start()
            state = task.wait(3000)
            if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"dispatch failed {state}")
            out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            actual = np.frombuffer(out.read(N * 4, 0), dtype=np.uint32)
            mismatches = np.count_nonzero(actual != data)
            print("seed", seed, "bytes", N * 4, "mismatches", int(mismatches), flush=True)
            assert mismatches == 0
        del task, inp, out, instr
    gc.collect()
    print("VERDICT", case, "PASS", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("build", "run", "suite"))
    ap.add_argument("--case", choices=CASES)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    if a.mode == "build":
        build()
    elif a.mode == "run":
        run(a.case, a.seed)
    else:
        from silicon_probe_record import witness
        print("Scope: one submission per fresh context; repeated submissions timed out in the earlier probe log.")
        for case in CASES:
            for seed in (17, 2901, 7021):
                print("PRE_CASE", case, seed, flush=True)
                witness()
                try:
                    p = subprocess.run([sys.executable, __file__, "run", "--case", case, "--seed", str(seed)], timeout=40)
                finally:
                    print("POST_CASE", case, seed, flush=True)
                    witness()
                if p.returncode:
                    raise SystemExit(p.returncode)
        print("VERDICT: all local/east/west read/write controls PASS; opens cross-column MemTile allocation experiments.")


if __name__ == "__main__":
    main()
