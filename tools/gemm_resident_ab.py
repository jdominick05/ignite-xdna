#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A/B one GEMM shape: the weight-resident build (whole_array_generic.py --b-resident, its own xclbin) against the
current generic build (a shared xclbin plus this shape's .bin), on the same seeded bf16 A and B.

Both contexts are open together. It checks the resident C bit for bit against the generic C (each tile sums K in
the same order, so they must be equal), times --calls calls of each, and then times calls alternating between the
two contexts, which is what a per-(M, K) xclbin set would cost a forward pass.

    python tools/gemm_resident_ab.py --resident scratch/bres-a/g_2048_512_4096 \
        --generic-xclbin scratch/egemm-mmown-m64-epi/g_512_512_512.xclbin \
        --generic-bin scratch/egemm-mmown-m64-epi/g_2048_512_4096.bin -M 2048 -K 512 -N 4096 [--hollow STEM]
"""
import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

XRT_SMI = "C:/Windows/System32/AMD/xrt-smi.exe"
WAIT_MS = 20000


def idle() -> bool:
    out = subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True, timeout=60).stdout
    return "No hardware contexts running" in out


def to_bf16_bits(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resident", type=Path, required=True, help="stem of the resident .xclbin/.bin")
    ap.add_argument("--hollow", type=Path, help="stem of the resident build with no MACs (timing only)")
    ap.add_argument("--generic-hollow", nargs=2, type=Path, metavar=("XCLBIN", "BIN"),
                    help="the generic build with no MACs (timing only), for the same-sitting floor")
    ap.add_argument("--generic-xclbin", type=Path, required=True)
    ap.add_argument("--generic-bin", type=Path, required=True)
    ap.add_argument("-M", type=int, required=True)
    ap.add_argument("-K", type=int, required=True)
    ap.add_argument("-N", type=int, required=True)
    ap.add_argument("--calls", type=int, default=30)
    a = ap.parse_args()
    sys.path.insert(0, "C:/Xilinx/XRT/xrt_sdk/xrt/python")
    import pyxrt
    if not idle():
        print("NPU not idle: refusing (a foreign context is contention, not a finding)")
        return 3
    M, K, N = a.M, a.K, a.N
    rng = np.random.default_rng(3)
    act = to_bf16_bits(rng.standard_normal((M, K)))
    w = to_bf16_bits(rng.standard_normal((K, N)) * 0.02)
    dev = pyxrt.device(0)
    to_dev, from_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    bo = lambda nbytes: pyxrt.bo(dev, nbytes, pyxrt.bo.host_only, 0)
    arms = {}

    def open_arm(name, xclbin, binp):
        xb = pyxrt.xclbin(str(xclbin))
        dev.register_xclbin(xb)
        ctx = pyxrt.hw_context(dev, xb.get_uuid())
        kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
        words = np.frombuffer(Path(binp).read_bytes(), dtype=np.uint32)
        ib = pyxrt.bo(dev, words.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
        ib.write(words.tobytes(), 0)
        ib.sync(to_dev)
        A, B, C = bo(act.nbytes), bo(w.nbytes), bo(M * N * 4)
        A.write(act.tobytes(), 0)
        A.sync(to_dev)
        B.write(w.tobytes(), 0)
        B.sync(to_dev)
        arms[name] = dict(xb=xb, ctx=ctx, kern=kern, ib=ib, nb=words.nbytes, A=A, B=B, C=C)

    def call(name):
        r = arms[name]
        t = time.perf_counter()
        st = r["kern"](3, r["ib"], r["nb"], r["A"], r["B"], r["C"]).wait(WAIT_MS)
        dt = (time.perf_counter() - t) * 1000
        if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"{name}: {st}")
        return dt

    def out(name):
        r = arms[name]
        r["C"].sync(from_dev)
        return np.frombuffer(r["C"].read(M * N * 4, 0), dtype=np.float32).reshape(M, N).copy()

    open_arm("generic", a.generic_xclbin, a.generic_bin)
    open_arm("resident", f"{a.resident}.xclbin", f"{a.resident}.bin")
    if a.hollow:
        open_arm("hollow", f"{a.hollow}.xclbin", f"{a.hollow}.bin")
    if a.generic_hollow:
        open_arm("generic_hollow", *a.generic_hollow)
    print(f"OPEN {len(arms)} contexts: {', '.join(arms)}; {M}x{K}x{N}", flush=True)
    call("generic")
    g = out("generic")
    call("resident")
    r1 = out("resident")
    call("resident")
    r2 = out("resident")
    ref = act.astype(np.uint32) << 16
    ref = ref.view(np.float32) @ (w.astype(np.uint32) << 16).view(np.float32)
    err = float(np.abs(r1 - ref).max() / np.abs(ref).max())
    bit = bool(np.array_equal(r1.view(np.uint32), g.view(np.uint32)))
    rep = bool(np.array_equal(r1.view(np.uint32), r2.view(np.uint32)))
    ndiff = int((r1.view(np.uint32) != g.view(np.uint32)).sum())
    ok = bit and rep and err < 1e-3
    print(f"CHECK resident vs generic: bit-equal {bit} ({ndiff} of {r1.size} words differ), repeat bit-equal {rep}, "
          f"rel err vs fp32 {err:.2e}: {'PASS' if ok else 'FAIL'}", flush=True)
    res = dict(shape=[M, K, N], bit_equal=bit, repeat_equal=rep, rel_err=err, ok=ok)
    for name in arms:
        for _ in range(3):
            call(name)
        ts = [call(name) for _ in range(a.calls)]
        res[f"{name}_ms"] = statistics.median(ts)
        print(f"TIME {name}: median {statistics.median(ts):.3f} ms/call (min {min(ts):.3f}, max {max(ts):.3f}, "
              f"{a.calls} calls)", flush=True)
    ts = {"generic": [], "resident": []}
    for _ in range(a.calls):
        for name in ts:
            ts[name].append(call(name))
    for name, v in ts.items():
        res[f"{name}_alternating_ms"] = statistics.median(v)
        print(f"TIME {name} alternating with the other context: median {statistics.median(v):.3f} ms/call", flush=True)
    print("RESULT " + json.dumps(res), flush=True)
    for r in arms.values():
        for k in ("C", "B", "A", "ib", "kern", "ctx"):
            del r[k]
            gc.collect()
    arms.clear()
    gc.collect()
    del dev
    gc.collect()
    print(f"idle after: {idle()}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
