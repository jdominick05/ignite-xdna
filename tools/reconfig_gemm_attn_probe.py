#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The real designs in one context: the 16-core weight-resident GEMM and the 16-core fused attention, switched
from inside one instruction stream (tools/reconfig_probe.py showed one core switching on XDNA1).

The stream is assembled on the host from parts, the shape a VM's command stream takes:
- a configuration blob per design: a module whose @main sequence is one empty ``aiex.configure @design {}``,
  compiled with aiecc --expand-load-pdis, LOADPDI ops stripped (an xclbin holds no PDI to load). @gemm is
  scratch/egemm-bres/g_512_512_512.prj/aie.mlir; @attn is attn_array.py's doc512 harness;
- each kernel's own compiled instruction stream (scratch/egemm-bres/g_512_512_512.bin, doc512/insts.bin), its
  DDR_PATCH ops renumbered onto @main's buffers.
(aiex.run cannot inline the weight-resident GEMM's sequence: it references MemTile aie.buffers, and the
materialization pass takes only tiles and locks.) The kernel takes five data buffers: A, B, C, Q|KV (one buffer;
KV's patches add Q's byte length) and O. The context is opened on the GEMM's own xclbin.

Plans are ``context:parts`` (one process each, 10 s wait, NPU idle checked after each). The context is ``g``
(the GEMM's xclbin) or ``a`` (the attention's); parts run in order: ``G`` / ``A`` the GEMM / attention
configuration blob, ``g`` / ``a`` the GEMM / attention instruction stream. For example:
    g:g        the GEMM stream, re-assembled, on its own context (checks the assembly)
    g:G        the GEMM configuration alone, nothing run
    g:Gg       the GEMM configuration rewritten over itself, then the GEMM
    a:Aa       the same for the attention
    g:GgAa     gemm, then attention, on one context
    g:GgAaGg   gemm, attention, gemm (one transformer layer's order)
Cost model (timing only, nothing checked): ``P`` / ``Q`` are the GEMM / attention configuration without core
program-memory writes, ``R`` / ``S`` without program or core data memory either (what would remain to write if
both kernels stayed resident).
Each output is compared bit for bit with the same kernel run directly on its own context, in the same process,
and timed against those direct calls.

    bash scripts/research-iron.sh tools/reconfig_gemm_attn_probe.py build
    bash scripts/research-iron.sh tools/reconfig_gemm_attn_probe.py run [--plans G,GA,GAG] [--reps 20]

Written on Desktop 2 (Phoenix), 2026-10-19; what has been run is in its logs.
"""
import argparse
import json
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "kernels/embedgemma"))
import txn_decode  # noqa: E402

BUILD = ROOT / "build" / "reconfig_gemm_attn"
GEMM_PRJ = ROOT / "scratch/egemm-bres/g_512_512_512.prj"
GEMM_XCLBIN = ROOT / "scratch/egemm-bres/g_512_512_512.xclbin"
GEMM_INSTS = ROOT / "scratch/egemm-bres/g_512_512_512.bin"
ATTN = ROOT / "scratch/egemma-attn-hd"
IRON = Path.home() / "mlir-aie" / "ironenv"
PEANO = IRON / "Lib/site-packages/llvm-aie"
XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
M = K = N = 512
def parse(plan):
    ctx, parts = plan.split(":")
    assert ctx in "ga" and all(c in "GAgaPQRS" for c in parts), plan
    return ctx, parts


def region(op):
    """A config op body -> 'pm' (core program memory), 'dm' (core data memory), or 'cfg'."""
    if op[0] not in (0, 1, 3):
        return "cfg"
    reg = struct.unpack_from("<I", op, 8)[0] if op[0] == 1 else struct.unpack_from("<Q", op, 8)[0]
    row, off = reg >> 20 & 0x1F, reg & 0xFFFFF
    if row >= 2 and 0x20000 <= off < 0x30000:
        return "pm"
    if row >= 2 and off < 0x10000:
        return "dm"
    return "cfg"


def without(ops, drop):
    return [o for o in ops if region(o) not in drop]


def device_block(text, name, seq):
    """A module's single ``aie.device(npu1) {...}`` -> the same block named @name, its sequence named @seq."""
    lines = text.strip().splitlines()
    assert lines[0].strip() == "module {" and lines[-1].strip() == "}", "expected one module wrapper"
    body = "\n".join(lines[1:-1])
    assert body.count("aie.device(npu1) {") == 1 and body.count("aie.runtime_sequence(") == 1
    return body.replace("aie.device(npu1) {", f"aie.device(npu1) @{name} {{").replace(
        "aie.runtime_sequence(", f"aie.runtime_sequence @{seq}(")


def attn_sizes():
    import attn_array as A
    nkb, groups = A.layout(512, 512, 4, 256)
    items = sum(len(g) for g in groups)
    return items * A.ROWS * A.QWORDS, 2 * (512 // 32) * 8 * A.CHUNK, items * A.ROWS * A.OUTW


def attn_block():
    import attn_array as A
    nq, nkv, no = attn_sizes()
    text = A.harness(512, 512, 4, 2, 256).replace('link_with = "kernel.o"', 'link_with = "egattn_kernel.o"')
    out = []
    for ln in text.splitlines():   # static argument types, so @main can pass subviews of its own buffers
        if "aie.runtime_sequence(" in ln:
            ln = ln.replace("%qa: memref<?xi32>", f"%qa: memref<{nq}xi32>").replace(
                "%kva: memref<?xi32>", f"%kva: memref<{nkv}xi32>").replace("%oa: memref<?xi32>", f"%oa: memref<{no}xi32>")
        elif "dma_memcpy_nd(%qa" in ln:
            ln = ln.replace(": memref<?xi32>", f": memref<{nq}xi32>")
        elif "dma_memcpy_nd(%kva" in ln:
            ln = ln.replace(": memref<?xi32>", f": memref<{nkv}xi32>")
        elif "dma_memcpy_nd(%oa" in ln:
            ln = ln.replace(": memref<?xi32>", f": memref<{no}xi32>")
        out.append(ln)
    assert "memref<?xi32>" not in "\n".join(out)
    return device_block("\n".join(out), "attn", "attn_seq")


def config_module(name, block):
    return (f"module {{\n  aie.device(npu1) @main {{\n    aie.runtime_sequence @sequence(%x: memref<4xi32>) {{\n"
            f"      aiex.configure @{name} {{\n      }}\n    }}\n  }}\n{block}\n}}\n")


def strip_loadpdi(raw):
    hdr, ops = txn_decode.ops(raw)
    keep = [raw[p:p + n] for p, n, op, _ in ops if op != 8]
    return keep, len(ops) - len(keep)


def build(plans=None):
    """Configuration blobs cfg_gemm.bin / cfg_attn.bin (op bodies only, LOADPDI stripped, as a full stream)."""
    BUILD.mkdir(parents=True, exist_ok=True)
    blocks = dict(gemm=device_block((GEMM_PRJ / "aie.mlir").read_text(encoding="utf-8"), "gemm", "gemm_seq"),
                  attn=attn_block())
    for name, block in blocks.items():
        d = BUILD / f"cfg_{name}"
        d.mkdir(exist_ok=True)
        for o in GEMM_PRJ.glob("*.o"):
            shutil.copy(o, d / o.name)
        shutil.copy(ATTN / "kernel.o", d / "egattn_kernel.o")
        (d / "design.mlir").write_text(config_module(name, block), encoding="utf-8")
        t = time.time()
        r = subprocess.run([str(IRON / "Scripts/aiecc.exe"), "design.mlir", f"--peano={PEANO}", "--expand-load-pdis",
                            "--get-npu-insts", "--npu-insts-name={0}_insts.bin", "--tmpdir=prj"],
                           cwd=d, capture_output=True, text=True)
        if r.returncode:
            print(f"BUILD cfg_{name}: FAIL\n{(r.stdout + r.stderr)[-4000:]}", flush=True)
            return 1
        raw = (d / "main_sequence_insts.bin").read_bytes()
        keep, dropped = strip_loadpdi(raw)
        out = assemble([keep])
        (BUILD / f"cfg_{name}.bin").write_bytes(out)
        hdr, ops = txn_decode.ops(out)
        kinds = {}
        for p_, n_, op, f in ops:
            kinds[txn_decode.NAMES.get(op, hex(op))] = kinds.get(txn_decode.NAMES.get(op, hex(op)), 0) + 1
        pm = sum(f["words"] * 4 for p_, n_, op, f in ops if op == 1 and 0x20000 <= (f["reg"] & 0xFFFFF) < 0x30000)
        print(f"BUILD cfg_{name}: {len(out)} B, {len(ops)} ops {kinds}, {dropped} LOADPDI removed, program-memory "
              f"block writes {pm} B ({time.time() - t:.0f} s)", flush=True)
    return 0


def body_ops(path_or_bytes):
    raw = Path(path_or_bytes).read_bytes() if not isinstance(path_or_bytes, bytes) else path_or_bytes
    hdr, ops = txn_decode.ops(raw)
    return [raw[p:p + n] for p, n, op, _ in ops], hdr


def remap(ops, mapping):
    """DDR_PATCH ops (48 B: argidx u64 @32, argplus u64 @40): arg i -> (new index, byte offset added)."""
    out = []
    for o in ops:
        if o[0] == 0x81:
            b = bytearray(o)
            idx, plus = struct.unpack_from("<QQ", b, 32)
            new, add = mapping[idx]
            struct.pack_into("<QQ", b, 32, new, plus + add)
            o = bytes(b)
        out.append(o)
    return out


def assemble(parts, header=None):
    """Lists of op bodies -> one stream; the header is the first part's source header with counts fixed."""
    ops = [o for part in parts for o in part]
    if header is None:
        header = bytes((BUILD / "cfg_gemm" / "main_sequence_insts.bin").read_bytes()[:16]) if (
            BUILD / "cfg_gemm" / "main_sequence_insts.bin").exists() else GEMM_INSTS.read_bytes()[:16]
    head = bytearray(header)
    body = b"".join(ops)
    struct.pack_into("<II", head, 8, len(ops), 16 + len(body))
    return bytes(head) + body


def plan_stream(name):
    nq, nkv, no = attn_sizes()
    gemm_ops, ghdr = body_ops(GEMM_INSTS)
    attn_ops, _ = body_ops(ATTN / "doc512" / "insts.bin")
    attn_ops = remap(attn_ops, {0: (3, 0), 1: (3, nq * 4), 2: (4, 0)})
    G, A = body_ops(BUILD / "cfg_gemm.bin")[0], body_ops(BUILD / "cfg_attn.bin")[0]
    table = dict(G=G, A=A, g=gemm_ops, a=attn_ops, P=without(G, {"pm"}), Q=without(A, {"pm"}),
                 R=without(G, {"pm", "dm"}), S=without(A, {"pm", "dm"}))
    return assemble([table[c] for c in parse(name)[1]], GEMM_INSTS.read_bytes()[:16])


def inputs():
    import attn_array as A
    rng = np.random.default_rng(7)
    bits = lambda *s: (rng.standard_normal(s).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    a, b = bits(M, K), bits(K, N)
    q, k, v, L, W = A.case_data("doc512")
    qp, kv, _ = A.pack(q, k, v, W)
    return a, b, qp.reshape(-1), kv.reshape(-1)


def one(name, reps):
    sys.path.insert(0, str(ROOT / "src"))
    from ignite_xdna.runtime.driver import setup_xrt_environment
    setup_xrt_environment()
    import pyxrt
    to, fr = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    ok_state = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
    a, b, qp, kv = inputs()
    nq, nkv, no = attn_sizes()
    assert qp.size == nq and kv.size == nkv
    dev = pyxrt.device(0)

    def open_ctx(xclbin):
        xb = pyxrt.xclbin(str(xclbin))
        dev.register_xclbin(xb)
        ctx = pyxrt.hw_context(dev, xb.get_uuid())
        return xb, ctx, pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())

    def instr(kern, path):
        w = Path(path).read_bytes()
        ib = pyxrt.bo(dev, len(w), pyxrt.bo.cacheable, kern.group_id(1))
        ib.write(w, 0)
        ib.sync(to)
        return ib, len(w)

    def bo(kern, i, arr=None, nbytes=None):
        x = pyxrt.bo(dev, arr.nbytes if arr is not None else nbytes, pyxrt.bo.host_only, kern.group_id(3 + i))
        if arr is not None:
            x.write(np.ascontiguousarray(arr).tobytes(), 0)
            x.sync(to)
        return x

    def timed(kern, ib, n, bos, reps_):
        ts = []
        for _ in range(reps_):
            t0 = time.perf_counter()
            st = kern(3, ib, n, *bos).wait(10000)
            ts.append((time.perf_counter() - t0) * 1e3)
            if st != ok_state:
                return None, str(st)
        return float(np.median(ts[1:] or ts)), "ok"

    res = dict(plan=name)
    # direct references, each kernel on its own context
    xb, ctx, kern = open_ctx(GEMM_XCLBIN)
    ib, n = instr(kern, GEMM_INSTS)
    A_, B_, C_ = bo(kern, 0, a), bo(kern, 1, b), bo(kern, 2, nbytes=M * N * 4)
    res["direct_gemm_ms"], _ = timed(kern, ib, n, [A_, B_, C_], reps)
    C_.sync(fr)
    c_ref = np.frombuffer(C_.read(M * N * 4, 0), np.float32).copy()
    del ib, A_, B_, C_, kern, ctx, xb
    xb, ctx, kern = open_ctx(ATTN / "doc512/final.xclbin")
    ib, n = instr(kern, ATTN / "doc512/insts.bin")
    Q_, KV_, O_ = bo(kern, 0, qp), bo(kern, 1, kv), bo(kern, 2, nbytes=no * 4)
    res["direct_attn_ms"], _ = timed(kern, ib, n, [Q_, KV_, O_], reps)
    O_.sync(fr)
    o_ref = np.frombuffer(O_.read(no * 4, 0), np.uint32).copy()
    del ib, Q_, KV_, O_, kern, ctx, xb
    # the assembled stream on the plan's context
    which, parts = parse(name)
    xb, ctx, kern = open_ctx(GEMM_XCLBIN if which == "g" else ATTN / "doc512/final.xclbin")
    stream = plan_stream(name)
    fname = BUILD / ("plan_" + name.replace(":", "_") + ".bin")
    fname.write_bytes(stream)
    txn_decode.ops(stream)
    ib, n = instr(kern, fname)
    bos = [bo(kern, 0, a), bo(kern, 1, b), bo(kern, 2, nbytes=M * N * 4), bo(kern, 3, np.concatenate([qp, kv])),
           bo(kern, 4, nbytes=no * 4)]
    for x in (bos[2], bos[4]):   # poison the outputs: a stale buffer must not read as a pass
        x.write(np.full(x.size() // 4, 0x7FC07FC0, np.uint32).tobytes(), 0)
        x.sync(to)
    med, st = timed(kern, ib, n, bos, reps)
    res.update(stream_bytes=n, state=st, stream_ms=med)
    if st == "ok":
        bos[2].sync(fr)
        bos[4].sync(fr)
        c = np.frombuffer(bos[2].read(M * N * 4, 0), np.float32)
        o = np.frombuffer(bos[4].read(no * 4, 0), np.uint32)
        if "g" in parts:
            res["gemm_bit_equal"] = bool(np.array_equal(c.view(np.uint32), c_ref.view(np.uint32)))
        if "a" in parts:
            res["attn_bit_equal"] = bool(np.array_equal(o, o_ref))
    del ib, bos, kern, ctx, xb, dev
    return res


def idle():
    return "No hardware contexts running" in subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"],
                                                            capture_output=True, text=True, timeout=60).stdout


def run(plans, reps):
    if not idle():
        print("NPU not idle: refusing (a foreign context is contention, not a finding)")
        return 3
    for v in plans:
        try:
            r = subprocess.run([sys.executable, __file__, "one", "--plans", v, "--reps", str(reps)],
                               capture_output=True, text=True, timeout=300)
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
            print(f"RUN {v}: {line[0][7:] if line else 'no result, rc=%d %s' % (r.returncode, r.stderr[-600:])}",
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
    ap.add_argument("--plans", default="g:g,g:G,g:Gg,a:Aa,g:GgAa,g:GgAaGg")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "build":
        return build()
    if a.cmd == "one":
        print("RESULT " + json.dumps(one(a.plans, a.reps)), flush=True)
        return 0
    return run(a.plans.split(","), a.reps)


if __name__ == "__main__":
    sys.exit(main())
