#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single-core check of kernels/embedgemma/attn_local.cc: fused local attention for one head, head_dim 256.

    python kernels/embedgemma/attn_probe.py build --out DIR                     (ironenv)
    python kernels/embedgemma/attn_probe.py run --out DIR --case NAME ...       (pyxrt, NPU, idle device only)

One core (tile 0,2) fed straight from the shim: a Q packet per 16-row query tile (FIFO depth 1), the key window as
a stream of 4 KiB chunks (depth 2: per 32-key block four K^T chunks then four V chunks), and one bf16 [16, 256]
output per tile. The window per tile is NKB blocks starting at s = clamp(floor32(q0 - W), 0, L - 32 NKB); the core
reads q0, s, W and NKB from the packet header, so one xclbin serves every case. Cases (random q/k/v, the q/k scaled
so scores reach ~+-30, like the model's after its norms):
    dense   L = 128, W = 512, NKB = 4      every key visible to every row (global-like)
    band    L = 256, W = 48,  NKB = 5      partial, invisible and clamped windows
    model   L = 1088, W = 512, NKB = 33    the model's local window at full size
The check compares against a NumPy emulation of the kernel's own arithmetic: bf16 operands, the same block order,
P rounded to bf16, fp32 row sums, and, op for op in fp32, the core's table exp (exp_fp32.h: round-half-even
to_fixed of x*log2(e) at 8 fraction bits, 2^(i/256) table, linear remainder, exponent added in the bits) and its
Newton reciprocal (seed 0x7ef311c3, three steps). Matrix products are fp64 then fp32, so the emulation is not
bit-exact: the core accumulates in its own order.

Criterion history. Run 1 (results/aie/embedgemma_attn_local_onecore_desktop2_20261018T212916Z) used numpy exp and
exact division and required every output within 2 bf16 ulps: it FAILED (up to 377 ulps on 0.03-0.26% of
outputs). An ulp distance is unbounded for outputs near zero (O cancels), so from run 2 the criterion, fixed
before run 2, is: |device - emulation| <= 2 bf16 ulps of that output row's max |emulation|, for every output, and
repeat bit-equal, and finite. ``--save`` keeps the device outputs (npz) so any later analysis needs no device.
The frozen end-to-end gate in tools/embedgemma_hybrid.py stays the final judge.
"""
import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IRON = Path.home() / "mlir-aie" / "ironenv"
PEANO = IRON / "Lib/site-packages/llvm-aie"
INSTALL = IRON / "Lib/site-packages/mlir_aie"
LAYA = ROOT / "kernels" / "laya_attention"   # exp_fp32.h
QWORDS, CHUNK, OUTW, STATE = 2064, 1024, 2048, 5424
STACK = 2048
CASES = dict(dense=(128, 512, 4), band=(256, 48, 5), model=(1088, 512, 33))
XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def command(args, cwd):
    r = subprocess.run([str(x) for x in args], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-5000:] + r.stderr[-7000:], flush=True)
        raise RuntimeError(f"{Path(args[0]).name} failed ({r.returncode})")
    return r.stdout


def exp_table_values():
    v = np.exp2(np.arange(256, dtype=np.float64) / 256).astype(np.float32)
    return np.tile(v.reshape(64, 4), (1, 2)).reshape(512)   # two lookup banks: [v0..v3, v0..v3] per 8 words


def harness():
    return f"""module {{
  aie.device(npu1) {{
    func.func private @egattn_score(memref<{QWORDS}xi32>, memref<{CHUNK}xi32>, memref<{STATE}xi32>, i32, i32)
    func.func private @egattn_softmax(memref<{QWORDS}xi32>, memref<{STATE}xi32>, i32)
    func.func private @egattn_pv(memref<{CHUNK}xi32>, memref<{STATE}xi32>, i32)
    func.func private @egattn_finish(memref<{STATE}xi32>, memref<{OUTW}xi32>)
    %s0 = aie.tile(0, 0)
    %t0 = aie.tile(0, 2)
    aie.objectfifo @qF(%s0, {{%t0}}, 1 : i32) : !aie.objectfifo<memref<{QWORDS}xi32>>
    aie.objectfifo @kvF(%s0, {{%t0}}, 2 : i32) : !aie.objectfifo<memref<{CHUNK}xi32>>
    aie.objectfifo @oF(%t0, {{%s0}}, 1 : i32) : !aie.objectfifo<memref<{OUTW}xi32>>
    %state = aie.buffer(%t0) {{sym_name = "egattn_state"}} : memref<{STATE}xi32>
    %core = aie.core(%t0) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %forever = arith.constant 4294967295 : index
      scf.for %it = %c0 to %forever step %c1 {{
        %sq = aie.objectfifo.acquire @qF(Consume, 1) : !aie.objectfifosubview<memref<{QWORDS}xi32>>
        %q = aie.objectfifo.subview.access %sq[0] : !aie.objectfifosubview<memref<{QWORDS}xi32>> -> memref<{QWORDS}xi32>
        %nkb32 = memref.load %q[%c3] : memref<{QWORDS}xi32>
        %nkb = arith.index_cast %nkb32 : i32 to index
        scf.for %kb = %c0 to %nkb step %c1 {{
          %kbi = arith.index_cast %kb : index to i32
          scf.for %c = %c0 to %c4 step %c1 {{
            %ci = arith.index_cast %c : index to i32
            %sk = aie.objectfifo.acquire @kvF(Consume, 1) : !aie.objectfifosubview<memref<{CHUNK}xi32>>
            %k = aie.objectfifo.subview.access %sk[0] : !aie.objectfifosubview<memref<{CHUNK}xi32>> -> memref<{CHUNK}xi32>
            func.call @egattn_score(%q, %k, %state, %ci, %kbi) : (memref<{QWORDS}xi32>, memref<{CHUNK}xi32>, memref<{STATE}xi32>, i32, i32) -> ()
            aie.objectfifo.release @kvF(Consume, 1)
          }}
          func.call @egattn_softmax(%q, %state, %kbi) : (memref<{QWORDS}xi32>, memref<{STATE}xi32>, i32) -> ()
          scf.for %c = %c0 to %c4 step %c1 {{
            %ci = arith.index_cast %c : index to i32
            %sv = aie.objectfifo.acquire @kvF(Consume, 1) : !aie.objectfifosubview<memref<{CHUNK}xi32>>
            %v = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<{CHUNK}xi32>> -> memref<{CHUNK}xi32>
            func.call @egattn_pv(%v, %state, %ci) : (memref<{CHUNK}xi32>, memref<{STATE}xi32>, i32) -> ()
            aie.objectfifo.release @kvF(Consume, 1)
          }}
        }}
        %so = aie.objectfifo.acquire @oF(Produce, 1) : !aie.objectfifosubview<memref<{OUTW}xi32>>
        %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<{OUTW}xi32>> -> memref<{OUTW}xi32>
        func.call @egattn_finish(%state, %o) : (memref<{STATE}xi32>, memref<{OUTW}xi32>) -> ()
        aie.objectfifo.release @oF(Produce, 1)
        aie.objectfifo.release @qF(Consume, 1)
      }}
      aie.end
    }} {{link_with = "kernel.o", stack_size = {STACK} : i32}}
    aie.runtime_sequence(%qa: memref<?xi32>, %kva: memref<?xi32>, %oa: memref<?xi32>) {{
    }}
  }}
}}
"""


def sequence(tiles, nkb):
    """Runtime sequence for one case: every tile's Q packet, its key-window chunks, and its output."""
    nq, nkv, no = tiles * QWORDS, tiles * nkb * 8 * CHUNK, tiles * OUTW
    return "\n".join([
        f"      aiex.npu.dma_memcpy_nd(%qa[0, 0, 0, 0][1, 1, {tiles}, {QWORDS}][0, 0, {QWORDS}, 1]) {{id = 0 : i64, metadata = @qF}} : memref<?xi32>",
        f"      aiex.npu.dma_memcpy_nd(%kva[0, 0, 0, 0][1, {tiles * nkb}, 8, {CHUNK}][0, {8 * CHUNK}, {CHUNK}, 1]) {{id = 1 : i64, metadata = @kvF}} : memref<?xi32>",
        f"      aiex.npu.dma_memcpy_nd(%oa[0, 0, 0, 0][1, 1, {tiles}, {OUTW}][0, 0, {OUTW}, 1]) {{id = 2 : i64, metadata = @oF, issue_token = true}} : memref<?xi32>",
        "      aiex.npu.dma_wait {symbol = @oF}"]), (nq, nkv, no)


def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    clang = PEANO / "bin/clang++.exe"
    flags = ["-O2", "-std=c++20", "--target=aie2-none-unknown-elf", "-DNDEBUG", "-fstack-size-section",
             "-D__AIE_API_AIE_ADF_HPP__", f"-I{INSTALL / 'include'}", f"-I{LAYA}", "-Wno-attributes",
             "-Wno-parentheses"]
    values = ",".join(float(x).hex() + "f" for x in exp_table_values())
    (out / "tables.cc").write_text('extern "C" {alignas(32) float fp_exp_ab[512]={' + values +
                                   '};alignas(32) float fp_exp_cd[512]={' + values + '};}\n', encoding="utf-8")
    command([clang, *flags, "-c", HERE / "attn_local.cc", "-o", out / "core.o"], out)
    command([clang, *flags, "-c", out / "tables.cc", "-o", out / "tables.o"], out)
    command([clang, "--target=aie2-none-unknown-elf", "-r", "-nostdlib", out / "core.o", out / "tables.o",
             "-o", out / "kernel.o"], out)
    undefined = command([PEANO / "bin/llvm-nm.exe", "-u", out / "kernel.o"], out).split()
    if undefined:
        raise ValueError(f"unresolved symbols in kernel.o: {undefined}")
    sizes = command([PEANO / "bin/llvm-size.exe", out / "kernel.o"], out)
    print(sizes, flush=True)
    meta = dict(kernel_sha256=sha(out / "kernel.o"), source_sha256=sha(HERE / "attn_local.cc"), cases={})
    for name, (L, W, nkb) in CASES.items():
        d = out / name
        d.mkdir(exist_ok=True)
        seq, words = sequence(L // 16, nkb)
        text = harness().replace("    aie.runtime_sequence(%qa: memref<?xi32>, %kva: memref<?xi32>, %oa: memref<?xi32>) {\n    }",
                                 f"    aie.runtime_sequence(%qa: memref<?xi32>, %kva: memref<?xi32>, %oa: memref<?xi32>) {{\n{seq}\n    }}")
        (d / "harness.mlir").write_text(text, encoding="utf-8")
        (d / "kernel.o").write_bytes((out / "kernel.o").read_bytes())
        command([IRON / "Scripts/aiecc.exe", d / "harness.mlir", f"--peano={PEANO}", "--get-npu-insts",
                 "--npu-insts-name=insts.bin", "--get-xclbin", "--xclbin-name=final.xclbin", f"--tmpdir={d / 'prj'}"], d)
        meta["cases"][name] = dict(L=L, W=W, nkb=nkb, words=words, xclbin_sha256=sha(d / "final.xclbin"),
                                   insts_sha256=sha(d / "insts.bin"))
        print(f"BUILD {name}: L={L} W={W} NKB={nkb} ok", flush=True)
    (out / "build.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return 0


def bf16(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def bits(x):
    return (bf16(x).view(np.uint32) >> 16).astype(np.uint16)


def tiled(x, r, c):
    rows, cols = x.shape
    return x.reshape(rows // r, r, cols // c, c).transpose(0, 2, 1, 3).reshape(-1)


def window(q0, L, W, nkb):
    return int(min(max((q0 - W) // 32 * 32, 0), L - 32 * nkb))


def pack(q, k, v, W, nkb):
    """q/k/v [L, 256] fp32 (bf16 values) -> (Q packets, chunk stream) as uint32 words."""
    L = q.shape[0]
    T = L // 16
    qp = np.zeros((T, QWORDS), np.uint32)
    kv = np.zeros((T, nkb, 8, CHUNK), np.uint32)
    for t in range(T):
        q0 = 16 * t
        s = window(q0, L, W, nkb)
        qp[t, :4] = [q0, s, W, nkb]
        qp[t, 16:].view(np.uint16)[:] = tiled(bits(q[q0:q0 + 16]), 4, 8)
        for b in range(nkb):
            keys = slice(s + 32 * b, s + 32 * b + 32)
            raw = kv[t, b].view(np.uint16).reshape(8, 2 * CHUNK)
            for c in range(4):
                raw[c] = tiled(bits(k[keys, 64 * c:64 * c + 64].T), 8, 4)
                raw[4 + c] = tiled(bits(v[keys, 64 * c:64 * c + 64]), 8, 4)
    return qp, kv


def exp16(x):
    """exp_fp32.h's exp16 in fp32, op for op (round-half-even to_fixed, as the kernel sets conv_even)."""
    x = np.asarray(x, np.float32)
    tiny = x < np.float32(-80)
    x = np.minimum(np.float32(0), np.maximum(np.float32(-80), x))
    scaled = np.round((x * np.float32(1.4426950408889634)).astype(np.float64) * 256).astype(np.int32)
    exponent = scaled >> 8
    address = scaled & 255
    rem = (x - (scaled.astype(np.float32) * np.float32(.0027076061740622863)).astype(np.float32)).astype(np.float32)
    base = np.exp2(np.arange(256, dtype=np.float64) / 256).astype(np.float32)[address]
    mant = (base * (np.float32(1) + rem).astype(np.float32)).astype(np.float32)
    out = (mant.view(np.int32) + (exponent << 23)).view(np.float32)
    return np.where(tiny, np.float32(0), out).astype(np.float32)


def reciprocal16(x):
    x = np.asarray(x, np.float32)
    zero = x == 0
    x = np.where(zero, np.float32(1), x).astype(np.float32)
    y = (np.int32(0x7ef311c3) - x.view(np.int32)).view(np.float32)
    for _ in range(3):
        y = (y * (np.float32(2) - (x * y).astype(np.float32))).astype(np.float32)
    return np.where(zero, np.float32(0), y).astype(np.float32)


def emulate(q, k, v, W, nkb):
    """The kernel's arithmetic in NumPy: per 16-row tile, NKB 32-key blocks of its window, online softmax."""
    L = q.shape[0]
    out = np.zeros((L, 256), np.float32)
    for t in range(L // 16):
        q0 = 16 * t
        s = window(q0, L, W, nkb)
        rows = np.arange(q0, q0 + 16)
        m = np.full(16, -np.inf, np.float32)
        l = np.zeros(16, np.float32)
        O = np.zeros((16, 256), np.float32)
        for b in range(nkb):
            cols = np.arange(s + 32 * b, s + 32 * b + 32)
            vis = np.abs(rows[:, None] - cols[None, :]) <= W
            if not vis.any():
                continue
            S = (q[rows].astype(np.float64) @ k[cols].T.astype(np.float64)).astype(np.float32)
            S = np.where(vis, S, -np.inf).astype(np.float32)
            mn = np.maximum(m, S.max(axis=1))
            ref = np.where(np.isfinite(mn), mn, 0).astype(np.float32)
            alpha = np.where(l > 0, exp16(np.where(np.isfinite(m), m, -1e30) - ref), 0).astype(np.float32)
            P = bf16(np.where(vis, exp16(np.where(vis, S, -1e30) - ref[:, None]), 0).astype(np.float32))
            l = (alpha * l + P.sum(axis=1)).astype(np.float32)
            O = (alpha[:, None] * O + (P.astype(np.float64) @ v[cols].astype(np.float64))).astype(np.float32)
            m = mn
        out[rows] = bf16(O * reciprocal16(l)[:, None])
    return out


def case_data(name):
    L, W, nkb = CASES[name]
    rng = np.random.default_rng({"dense": 3, "band": 5, "model": 7}[name])
    q = bf16(rng.standard_normal((L, 256)).astype(np.float32) * 0.35)
    k = bf16(rng.standard_normal((L, 256)).astype(np.float32) * 0.35)
    v = bf16(rng.standard_normal((L, 256)).astype(np.float32))
    return q, k, v, L, W, nkb


def idle():
    return "No hardware contexts running" in subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"],
                                                            capture_output=True, text=True, timeout=60).stdout


def run(out: Path, names, reps, save=None):
    sys.path.insert(0, "C:/Xilinx/XRT/xrt_sdk/xrt/python")
    import pyxrt
    meta = json.loads((out / "build.json").read_text(encoding="utf-8"))
    if meta["source_sha256"] != sha(HERE / "attn_local.cc") or meta["kernel_sha256"] != sha(out / "kernel.o"):
        raise RuntimeError("build is stale: kernel source or object changed since build")
    if not idle():
        print("NPU not idle: refusing (a foreign context is contention, not a finding)")
        return 3
    ok = True
    saved = {}
    for name in names:
        c = meta["cases"][name]
        d = out / name
        if sha(d / "final.xclbin") != c["xclbin_sha256"] or sha(d / "insts.bin") != c["insts_sha256"]:
            raise RuntimeError(f"{name}: build artefacts changed")
        q, k, v, L, W, nkb = case_data(name)
        qp, kv = pack(q, k, v, W, nkb)
        ref = emulate(q, k, v, W, nkb)
        dev = pyxrt.device(0)
        xb = pyxrt.xclbin(str(d / "final.xclbin"))
        dev.register_xclbin(xb)
        ctx = pyxrt.hw_context(dev, xb.get_uuid())
        kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
        to, fr = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        words = np.frombuffer((d / "insts.bin").read_bytes(), dtype=np.uint32)
        ib = pyxrt.bo(dev, words.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
        ib.write(words.tobytes(), 0)
        ib.sync(to)
        bq = pyxrt.bo(dev, qp.nbytes, pyxrt.bo.host_only, kern.group_id(3))
        bkv = pyxrt.bo(dev, kv.nbytes, pyxrt.bo.host_only, kern.group_id(4))
        bo = pyxrt.bo(dev, (L // 16) * OUTW * 4, pyxrt.bo.host_only, kern.group_id(5))
        bq.write(qp.tobytes(), 0)
        bq.sync(to)
        bkv.write(kv.tobytes(), 0)
        bkv.sync(to)
        results, times = [], []
        for i in range(1 + reps):
            bo.write(np.full((L // 16) * OUTW, 0x7FC07FC0, np.uint32).tobytes(), 0)   # bf16 NaN poison
            bo.sync(to)
            t0 = time.perf_counter()
            st = kern(3, ib, words.nbytes, bq, bkv, bo).wait(10000)
            times.append((time.perf_counter() - t0) * 1000)
            if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                ok = False
                print(f"CASE {name}: dispatch {st}; stopping without retry", flush=True)
                break
            bo.sync(fr)
            raw = np.frombuffer(bo.read((L // 16) * OUTW * 4, 0), np.uint16).reshape(L, 256)
            results.append(raw.copy())
        if len(results) == 1 + reps:
            got = (results[0].astype(np.uint32) << 16).view(np.float32)
            saved[name] = results[0]
            refb = ref.view(np.uint32) >> 16
            diff = np.abs(results[0].astype(np.int64) - refb.astype(np.int64))
            same = all(np.array_equal(results[0], r) for r in results[1:])
            finite = bool(np.isfinite(got).all())
            frac = float((diff > 0).mean())
            # 2 bf16 ulps of each row's largest |output|: ulp(x) = 2^(floor(log2 x) - 7)
            rowmax = np.abs(ref).max(axis=1, keepdims=True)
            tol = 2 * np.exp2(np.floor(np.log2(np.maximum(rowmax, 1e-30))) - 7)
            worst = float((np.abs(got - ref) / tol).max())
            good = finite and same and worst <= 1.0
            ok &= good
            rel = float(np.abs(got - ref).max() / np.abs(ref).max())
            print(f"CASE {name}: L={L} W={W} NKB={nkb}: worst |diff| / (2 row ulps) {worst:.3f}, max ulp diff "
                  f"{int(diff.max())}, differing {frac:.4%}, rel err {rel:.2e}, finite {finite}, repeat bit-equal {same}: "
                  f"{'PASS' if good else 'FAIL'}; "
                  f"dispatch median {np.median(times[1:]) if reps else times[0]:.3f} ms over {reps}", flush=True)
        del ib, bq, bkv, bo, kern
        gc.collect()
        del ctx
        gc.collect()
        del xb, dev
        gc.collect()
    if save:
        np.savez_compressed(save, **saved)
        print(f"saved device outputs: {save}", flush=True)
    print(f"idle after: {idle()}", flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case", nargs="+", default=list(CASES))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--save", type=Path, help="npz of the device outputs per case")
    a = ap.parse_args()
    return build(a.out.resolve()) if a.cmd == "build" else run(a.out.resolve(), a.case, a.reps, a.save)


if __name__ == "__main__":
    sys.exit(main())
