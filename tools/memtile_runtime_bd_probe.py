#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which runtime-sequence MemTile DMA tasks work on XDNA1? A shim -> MemTile -> shim loopback, one variable at a time.

kernels/gemm_generic/whole_array_generic.py --b-resident 2 configures its MemTile B path from the runtime sequence
(aiex.dma_configure_task on the MemTile, with locks and a repeat count) and hangs on the first call, while the same
design with a static MemTile program (--b-resident 3) runs bit-exact. Each variant here streams L bytes of the input
into a MemTile buffer and back out to the output:

    nolock     runtime S2MM and MM2S tasks, no locks (the MM2S task is pushed after the S2MM one is awaited)
    lock       runtime S2MM (acquire space 1, release full 1) and MM2S (acquire full 1, release space 1), pushed together
    lock_v8    as lock with the GEMM's values: S2MM acquires space 8 (init 16) and releases full R; MM2S repeats R
               times, each repeat taking full 1 and returning space 8 / R; the output holds the buffer R times
    lock_rep   as lock_v8 with R repeats but values 1: S2MM releases full R; MM2S takes full 1 / returns space 1
    static_co  lock, while a static memtile_dma program also runs on the same MemTile (another channel pair, its own
               buffer and locks, fed by nothing so it idles on its lock) - the GEMM's arrangement
    *_ch5      the same on MemTile channel 5 instead of 0 (the GEMM's channel); _ch1 / _ch4 likewise
    *_lo       odd channels with BD ids 24 / 26 instead of 40 / 42

    bash scripts/research-iron.sh tools/memtile_runtime_bd_probe.py build [--variants ...]
    bash scripts/research-iron.sh tools/memtile_runtime_bd_probe.py run [--variants ...]

Written on Desktop 2 (Phoenix), 2026-10-19; what has been run is in its logs.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "memtile_runtime_bd_probe"
XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
L = 16384           # int32 words per transfer (64 KB)
R = 4               # repeats in the *_rep variants
VARIANTS = ("nolock", "lock", "lock_rep", "lock_v8", "static_co", "nolock_ch5", "lock_ch5", "lock_v8_ch5",
            "static_co_ch5", "lock_ch1", "lock_ch1_lo", "lock_ch5_lo", "lock_ch4")


def reps(v):
    return R if ("rep" in v or "v8" in v) else 1


def build(v: str) -> Path:
    import aie.iron as iron
    from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
    from aie.dialects.aie import EndOp, LockAction, dma_bd, use_lock
    from aie.dialects.aiex import bds, dma_await_task, dma_configure_task, dma_free_task, dma_start_task, shim_dma_bd
    from aie.iron import Acquire, Bd, Buffer, DmaChannel, Flow, Lock, Program, Release, Runtime, TileDma
    from aie.iron.device import NPU1, Tile
    from aie.utils.compile.utils import compile_mlir_module

    out = BUILD / v
    (out / "prj").mkdir(parents=True, exist_ok=True)
    iron.set_current_device(NPU1())
    i32 = np.dtype[np.int32]
    ch = 0
    for tag in ("_ch1", "_ch4", "_ch5"):
        if tag in v:
            ch = int(tag[3])
    lo = v.endswith("_lo")   # the lowest BD ids the channel's half allows (16/18 even, 24/26 odd) instead of 40/42
    rep = reps(v)
    v8 = "v8" in v
    shim = Tile(0, 0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(0, 1, tile_type=AIETileType.MemTile)
    buf = Buffer(np.ndarray[(L,), i32], name="mbuf", tile=mem)
    space = Lock(mem, init=16 if v8 else 1, name="space")
    full = Lock(mem, init=0, name="full")
    rt_locks, flows, dmas = [space, full], [Flow(shim, mem, src_channel=0, dst_channel=ch),
                                            Flow(mem, shim, src_channel=ch, dst_channel=0)], []
    if v.startswith("static_co"):   # an idle static program on another channel pair of the same MemTile
        oc = 1 if ch == 0 else 4
        obuf = Buffer(np.ndarray[(256,), i32], name="obuf", tile=mem)
        osp, ofu = Lock(mem, init=1, name="osp"), Lock(mem, init=0, name="ofu")
        rt_locks += [osp, ofu]
        dmas.append(TileDma(mem, [
            DmaChannel(DMAChannelDir.S2MM, oc, [Bd(obuf, acquires=[Acquire(osp)], releases=[Release(ofu)], next=0)]),
            DmaChannel(DMAChannelDir.MM2S, oc, [Bd(obuf, acquires=[Acquire(ofu)], releases=[Release(osp)], next=0)])]))
        flows.append(Flow(shim, mem, src_channel=1, dst_channel=oc))

    def sequence(inp, outp):
        from aie import ir
        with ir.InsertionPoint(ir.InsertionPoint.current.block.owner):
            buf.resolve()
        a_in, a_out = getattr(inp, "op", inp), getattr(outp, "op", outp)
        t0 = dma_configure_task(shim.op, DMAChannelDir.MM2S, 0)
        with bds(t0) as bd:
            with bd[0]:
                shim_dma_bd(a_in, offset=0, sizes=[1, 1, 1, L], strides=[0, 0, 0, 1])
                EndOp()
        t1 = dma_configure_task(mem.op, DMAChannelDir.S2MM, ch, issue_token=v.startswith("nolock"))
        with bds(t1) as bd:
            with bd[0]:
                if v.startswith("nolock") is False:
                    use_lock(space.op, LockAction.AcquireGreaterEqual, 8 if v8 else 1)
                dma_bd(buf.op, offset=0, transfer_len=L, bd_id=(24 if lo else 40) if ch % 2 else 16)
                if v.startswith("nolock") is False:
                    use_lock(full.op, LockAction.Release, rep)
                EndOp()
        t2 = dma_configure_task(mem.op, DMAChannelDir.MM2S, ch, repeat_count=rep - 1)
        with bds(t2) as bd:
            with bd[0]:
                if v.startswith("nolock") is False:
                    use_lock(full.op, LockAction.AcquireGreaterEqual, 1)
                dma_bd(buf.op, offset=0, transfer_len=L, bd_id=(26 if lo else 42) if ch % 2 else 18)
                if v.startswith("nolock") is False:
                    use_lock(space.op, LockAction.Release, 8 // rep if v8 else 1)
                EndOp()
        t3 = dma_configure_task(shim.op, DMAChannelDir.S2MM, 0, issue_token=True)
        with bds(t3) as bd:
            with bd[0]:
                shim_dma_bd(a_out, offset=0, sizes=[1, 1, 1, rep * L], strides=[0, 0, 0, 1])
                EndOp()
        if v.startswith("nolock"):
            dma_start_task(t0, t1)
            dma_await_task(t1)
            dma_start_task(t3, t2)
        else:
            dma_start_task(t0, t1, t2, t3)
        dma_await_task(t3)
        dma_free_task(t0, t1, t2)

    in_t = np.ndarray[(L,), i32]
    out_t = np.ndarray[(R * L,), i32]
    rt = Runtime(sequence, [in_t, out_t])
    for f in flows:
        rt.add_flow(f)
    for lk in rt_locks:
        rt.add_lock(lk)
    for d in dmas:
        rt.add_tile_dma(d)
    module = Program(iron.get_current_device(), rt, workers=[]).resolve_program()
    (out / "design.mlir").write_text(str(module), encoding="utf-8")
    compile_mlir_module(module, insts_path=out / "insts.bin", xclbin_path=out / "probe.xclbin", work_dir=out / "prj",
                        device=iron.get_current_device())
    return out


def run(v: str) -> dict:
    sys.path.insert(0, "C:/Xilinx/XRT/xrt_sdk/xrt/python")
    import pyxrt
    out = BUILD / v
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(str(out / "probe.xclbin"))
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    words = np.frombuffer((out / "insts.bin").read_bytes(), dtype=np.uint32)
    ib = pyxrt.bo(dev, words.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    ib.write(words.tobytes(), 0)
    ib.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    x = np.arange(L, dtype=np.int32) * 7 + 3
    bi, bo_ = pyxrt.bo(dev, x.nbytes, pyxrt.bo.host_only, 0), pyxrt.bo(dev, R * L * 4, pyxrt.bo.host_only, 0)
    bi.write(x.tobytes(), 0)
    bi.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_.write(np.full(R * L, -1, np.int32).tobytes(), 0)
    bo_.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    st = kern(3, ib, words.nbytes, bi, bo_).wait(5000)
    res = dict(variant=v, state=str(st))
    if st == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        bo_.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        y = np.frombuffer(bo_.read(R * L * 4, 0), dtype=np.int32)
        rep = reps(v)
        res["exact"] = bool(all(np.array_equal(y[i * L:(i + 1) * L], x) for i in range(rep)))
    del ib, bi, bo_, kern, ctx, xb, dev
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "run", "one"])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    vs = a.variants.split(",")
    if a.cmd == "build":
        for v in vs:
            try:
                build(v)
                print(f"BUILD {v}: ok", flush=True)
            except Exception as e:  # noqa: BLE001 - report and continue with the other variants
                print(f"BUILD {v}: FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
        return 0
    if a.cmd == "one":
        print("RESULT " + json.dumps(run(vs[0])), flush=True)
        return 0
    for v in vs:   # each variant in its own process: a hang must not take the others with it
        if not (BUILD / v / "probe.xclbin").exists():
            print(f"RUN {v}: no build", flush=True)
            continue
        r = subprocess.run([sys.executable, __file__, "one", "--variants", v], capture_output=True, text=True, timeout=120)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
        print(f"RUN {v}: {line[0][7:] if line else 'no result, rc=%d %s' % (r.returncode, r.stderr[-300:])}", flush=True)
        idle = "No hardware contexts running" in subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"],
                                                                 capture_output=True, text=True).stdout
        print(f"  idle after: {idle}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
