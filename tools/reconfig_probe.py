#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Can one XDNA1 hardware context switch designs from inside its instruction stream?

mlir-aie's ``aiex.configure @design { aiex.run @seq(...) }`` with ``aiecc --expand-load-pdis`` turns each
reconfiguration into register writes in the runtime instruction stream: core program memory, core control, DMA
descriptors, locks and stream switches. Upstream tests it only on npu2 with a full ELF. Here the context is opened
on an ordinary xclbin (design ``add2``, one core: out = in + 2) and the stream reconfigures the core to ``add3`` and
back. Decoded with tools/txn_decode.py, the expanded stream still holds LOADPDI ops (an empty-device reset before
each configure); they name PDIs an xclbin does not contain.

Variants, one process each, a 5 s wait, an idle check after every one:
    A    the add2 context, add2's own sequence: buf[0:4] + 2 (baseline)
    B0   one configure @add2 then run, LOADPDI removed: the same design written over itself
    B    configure @add2, run on [0:4] and [12:16]; configure @add3, run on [4:8] and [12:16]; LOADPDI removed.
         Expected: [0:4] + 2, [4:8] + 3, [8:12] unchanged, [12:16] + 5
    S    configure @add3 once and run once, LOADPDI removed (time of one reconfiguration + call)
    C    B's stream as aiecc emits it, LOADPDI kept (run last: a hang can block new contexts for minutes)

    bash scripts/research-iron.sh tools/reconfig_probe.py build
    bash scripts/research-iron.sh tools/reconfig_probe.py run [--variants A,B0,B,S,C] [--reps 20]

Written on Desktop 2 (Phoenix), 2026-10-19; what has been run is in its logs.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import txn_decode  # noqa: E402

BUILD = ROOT / "build" / "reconfig_probe"
IRON = Path.home() / "mlir-aie" / "ironenv"
PEANO = IRON / "Lib/site-packages/llvm-aie"
XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
DEV = "npu1_1col"
N = 64


def design(name, addend):
    return f"""  aie.device({DEV}) @{name} {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @fin(%t00, {{%t02}}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @fout(%t02, {{%t00}}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.core(%t02) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %k = arith.constant {addend} : i32
      %big = arith.constant 0xFFFFFE : index
      scf.for %n = %c0 to %big step %c1 {{
        %si = aie.objectfifo.acquire @fin(Consume, 1) : !aie.objectfifosubview<memref<4xi32>>
        %so = aie.objectfifo.acquire @fout(Produce, 1) : !aie.objectfifosubview<memref<4xi32>>
        %ei = aie.objectfifo.subview.access %si[0] : !aie.objectfifosubview<memref<4xi32>> -> memref<4xi32>
        %eo = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<4xi32>> -> memref<4xi32>
        scf.for %i = %c0 to %c4 step %c1 {{
          %x = memref.load %ei[%i] : memref<4xi32>
          %y = arith.addi %x, %k : i32
          memref.store %y, %eo[%i] : memref<4xi32>
        }}
        aie.objectfifo.release @fin(Consume, 1)
        aie.objectfifo.release @fout(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence @{name}_seq(%a : memref<4xi32>) {{
      %ti = aiex.dma_configure_task_for @fin {{
        aie.dma_bd(%a : memref<4xi32> offset = 0 len = 4)
        aie.end
      }}
      %to = aiex.dma_configure_task_for @fout {{
        aie.dma_bd(%a : memref<4xi32> offset = 0 len = 4)
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%ti)
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
    }}
  }}
"""


def run_on(dev, seq, start):
    return (f"        %v{start} = memref.subview %arg[{start}] [4] [1] : memref<{N}xi32> to memref<4xi32, strided<[1], "
            f"offset: {start}>>\n"
            f"        %r{start} = memref.reinterpret_cast %v{start} to offset: [0], sizes: [4], strides: [1] : "
            f"memref<4xi32, strided<[1], offset: {start}>> to memref<4xi32>\n"
            f"        aiex.run @{seq} (%r{start}) : (memref<4xi32>)\n")


def main_module(plan):
    """plan: [(design, [offsets])] -> a module whose @main sequence configures each design in turn."""
    body, k = [], 0
    for dev, starts in plan:
        runs = "".join(run_on(dev, f"{dev}_seq", s).replace(f"%v{s}", f"%v{k}_{s}").replace(f"%r{s}", f"%r{k}_{s}")
                       for s in starts)
        body.append(f"      aiex.configure @{dev} {{\n{runs}      }}\n")
        k += 1
    return (f"module {{\n  aie.device({DEV}) @main {{\n    aie.runtime_sequence @sequence(%arg : memref<{N}xi32>) {{\n"
            + "".join(body) + "    }\n  }\n" + design("add2", 2) + design("add3", 3) + "}\n")


PLANS = dict(B0=[("add2", [0])], B=[("add2", [0, 12]), ("add3", [4, 12])], S=[("add3", [0])])


def aiecc(d, mlir, *args):
    d.mkdir(parents=True, exist_ok=True)
    (d / "design.mlir").write_text(mlir, encoding="utf-8")
    r = subprocess.run([str(IRON / "Scripts/aiecc.exe"), "design.mlir", f"--peano={PEANO}", *args, "--tmpdir=prj"],
                       cwd=d, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"aiecc failed in {d}: {(r.stdout + r.stderr)[-3000:]}")


def build():
    base = BUILD / "base"
    aiecc(base, "module {\n" + design("add2", 2) + "}\n", "--get-xclbin", "--xclbin-name=final.xclbin",
          "--get-npu-insts", "--npu-insts-name=insts.bin")
    print(f"BUILD base: {base / 'final.xclbin'}", flush=True)
    for name, plan in PLANS.items():
        d = BUILD / name
        aiecc(d, main_module(plan), "--expand-load-pdis", "--get-npu-insts", "--npu-insts-name={0}_insts.bin")
        raw = (d / "main_sequence_insts.bin").read_bytes()
        hdr, ops = txn_decode.ops(raw)
        keep = [(p, n) for p, n, op, _ in ops if op != 8]
        body = b"".join(raw[p:p + n] for p, n in keep)
        head = bytearray(raw[:16])
        import struct
        struct.pack_into("<II", head, 8, len(keep), 16 + len(body))
        (d / "stripped.bin").write_bytes(bytes(head) + body)
        txn_decode.ops((d / "stripped.bin").read_bytes())   # re-parses: the fixed-up header is consistent
        print(f"BUILD {name}: {len(raw)} B, {len(ops)} ops, {len(ops) - len(keep)} LOADPDI removed", flush=True)
    return 0


def expected(name, x):
    y = x.copy()
    if name == "A" or name == "B0":
        y[0:4] += 2
    elif name in ("B", "C"):
        y[0:4] += 2
        y[4:8] += 3
        y[12:16] += 5
    elif name == "S":
        y[0:4] += 3
    return y


def one(name, reps):
    sys.path.insert(0, str(ROOT / "src"))
    from ignite_xdna.runtime.driver import setup_xrt_environment
    setup_xrt_environment()
    import pyxrt
    stream = {"A": BUILD / "base/insts.bin", "C": BUILD / "B/main_sequence_insts.bin"}.get(
        name, BUILD / name / "stripped.bin")
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(str(BUILD / "base/final.xclbin"))
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    to, fr = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    words = stream.read_bytes()
    ib = pyxrt.bo(dev, len(words), pyxrt.bo.cacheable, kern.group_id(1))
    ib.write(words, 0)
    ib.sync(to)
    buf = pyxrt.bo(dev, N * 4, pyxrt.bo.host_only, kern.group_id(3))
    x = (np.arange(N, dtype=np.int32) * 10 + 1)
    res = dict(variant=name, stream=str(stream.relative_to(ROOT)), bytes=len(words))
    times, ok = [], True
    for i in range(reps):
        buf.write(x.tobytes(), 0)
        buf.sync(to)
        t0 = time.perf_counter()
        st = kern(3, ib, len(words), buf).wait(5000)
        times.append((time.perf_counter() - t0) * 1e3)
        if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            res.update(state=str(st), reps_done=i)
            ok = False
            break
        buf.sync(fr)
        y = np.frombuffer(buf.read(N * 4, 0), np.int32)
        if not np.array_equal(y, expected(name, x)):
            bad = np.nonzero(y != expected(name, x))[0]
            res.update(state="WRONG", first_bad=[int(j) for j in bad[:8]], got=[int(v) for v in y[:16]])
            ok = False
            break
    if ok:
        res.update(state="PASS", reps=reps, median_ms=float(np.median(times[1:] or times)),
                   min_ms=float(min(times)), first_ms=times[0])
    del ib, buf, kern, ctx, xb, dev
    return res


def idle():
    return "No hardware contexts running" in subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"],
                                                            capture_output=True, text=True, timeout=60).stdout


def run(variants, reps):
    if not idle():
        print("NPU not idle: refusing (a foreign context is contention, not a finding)")
        return 3
    for v in variants:
        try:
            r = subprocess.run([sys.executable, __file__, "one", "--variants", v, "--reps", str(reps)],
                               capture_output=True, text=True, timeout=180)
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
            print(f"RUN {v}: {line[0][7:] if line else 'no result, rc=%d %s' % (r.returncode, r.stderr[-400:])}",
                  flush=True)
        except subprocess.TimeoutExpired:
            print(f"RUN {v}: child timed out", flush=True)
        clear = idle()
        print(f"  idle after: {clear}", flush=True)
        if not clear:
            print("  stopping: the device still holds a context", flush=True)
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "run", "one"])
    ap.add_argument("--variants", default="A,B0,B,S,C")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "build":
        return build()
    if a.cmd == "one":
        print("RESULT " + json.dumps(one(a.variants, a.reps)), flush=True)
        return 0
    return run(a.variants.split(","), a.reps)


if __name__ == "__main__":
    sys.exit(main())
