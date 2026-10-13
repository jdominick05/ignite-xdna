#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Can one loaded GEMM design serve all of Laya's weight-product shapes? Step 4a of the Laya study.

Laya step 3 (tools/laya_npu_screen.py) found that running its 120 bf16 products in model order costs 217.1 ms more
than the same products grouped by shape. Each of the six (K, N) shapes is its own mlir-aie ``whole_array`` design,
and IRON keeps up to six hardware contexts resident on npu1 (``NPU_CONTEXT_CACHE_SIZE``), so that cost is the
context switch between resident designs, about 1.8 ms each. kernels/gemm_generic/whole_array_generic.py
``--rtp-generic`` moves the per-shape loop counts out of the core program into a run-time parameter. If every shape
then compiles to the same device image, one context can run all of them, and a shape change is just a different
instruction stream.

    python tools/gemm_generic_probe.py compile --out DIR [--plain]    (offline) compile Laya's six shapes ahead of
                                                                      time: DIR/<tag>_<K>_<N>.xclbin and .bin
    python tools/gemm_generic_probe.py compare --out DIR [--tag g]    (offline) compare the xclbins section by section:
                                                                      which sections are byte-identical across shapes

    python tools/gemm_generic_probe.py single --out DIR --xclbin X --seq "K,N;K,N;..."
                                                                      (NPU) one product per listed shape, in order, in
                                                                      ONE context on design X with the instruction
                                                                      streams g_<K>_<N>.bin: PASS per shape, or the
                                                                      state it stopped in
    python tools/gemm_generic_probe.py run --out DIR [--blocks B --seqs S]
                                                                      (NPU) the six shapes from ONE context (the
                                                                      generic design, six instruction streams) against
                                                                      the plain design per shape, at most FIVE
                                                                      contexts resident with LRU eviction (IRON's
                                                                      npu1 behaviour at the device's ceiling):
                                                                      correctness, bit-equality, then Laya's 120-product
                                                                      order timed in alternating blocks

All shapes use one tile: m=32, k=64, n=64, C single-buffered, bf16 in, f32 out, 4 columns. N=5248 is padded with
zero columns to 5376 (a multiple of 4 x 64), as laya_npu_screen.py's ``--tile best`` does. ``--even-k`` (compile
and run) also pads K with zero rows of B and zero columns of A to an even number of k-tiles: 2624 (41 tiles) runs
as 2688. The generic core timed out on K=2624 on 2026-10-13; the hypothesis is that its depth-2 input FIFOs need an
even K-tile count once that count is a run-time value. Zero padding leaves every output value unchanged (each extra
product is an exact 0), apart from the sign of an exact-zero sum. ``compile`` and ``compare``
run in the mlir-aie ironenv (they compile through IRON) and never open the device. ``run`` needs pyxrt
(PYTHONPATH=C:/Xilinx/XRT/xrt_sdk/xrt/python), refuses to start unless xrt-smi reports no hardware contexts, and
gives every dispatch a timeout; the first timeout or failed state stops it.

``run`` times the device side only, the same for both arms: per product, sync A in, call, wait, sync C out. The
fp32-to-bf16 conversion of the activation is host work that does not depend on the arm, so it is left out here (step
3 measured it). Its activations are fixed random data, like ``laya_npu_screen.py chain``.
What has been run: results/laya/laya_gemm_onectx_desktop2_20261013T012132Z.log (Desktop 2, 2026-10-13).
"""
import argparse
import gc
import hashlib
import os
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "kernels" / "gemm_generic" / "whole_array_generic.py"
M, TILE = 512, dict(m=32, k=64, n=64)
SHAPES = [(1024, 3072), (1024, 1024), (1024, 5248), (2624, 1024), (1024, 4096), (4096, 1024)]   # Laya's six (K, N)


EVEN_K = False   # set by --even-k


def run_k(K: int) -> int:
    """K as run: unchanged, or padded to an even number of k-tiles under --even-k."""
    t = -(-K // TILE["k"])
    return (t + (t % 2 if EVEN_K else 0)) * TILE["k"]


def padded(N: int) -> int:
    return -(-N // 256) * 256


def name(tag: str, K: int, N: int) -> str:
    return f"{tag}_{run_k(K)}_{N}"


def cmd_compile(a) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    tag = "p" if a.plain else "g"
    for K, N in ([tuple(int(v) for v in a.shape.split(","))] if a.shape else SHAPES):
        stem = a.out / name(tag, K, N)
        cmd = [sys.executable, str(DESIGN), "--dev", "npu", "-M", str(M), "-K", str(run_k(K)), "-N", str(padded(N)),
               "-m", str(TILE["m"]), "-k", str(TILE["k"]), "-n", str(TILE["n"]), "--c-single-buffer", "1",
               "--dtype_in", "bf16", "--dtype_out", "f32", "--xclbin-path", f"{stem}.xclbin",
               "--insts-path", f"{stem}.bin"] + ([] if a.plain else ["--rtp-generic"])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        ok = r.returncode == 0 and Path(f"{stem}.xclbin").exists()
        print(f"COMPILE {stem.name} (K run as {run_k(K)}, N run as {padded(N)}): {'ok' if ok else 'FAIL rc=%d' % r.returncode}", flush=True)
        if not ok:
            print(r.stdout[-3000:] + r.stderr[-3000:])
            return 1
    return 0


def sections(p: Path) -> dict[str, bytes]:
    """axlf sections by kind and name. Layout (xclbin.h): 304-byte preamble, then the header, whose m_numSections
    is at 448; section headers start at 456, 40 bytes each (kind u32, name char[16], pad, offset u64, size u64)."""
    b = p.read_bytes()
    if b[:8] != b"xclbin2\0":
        raise SystemExit(f"{p}: not an axlf xclbin")
    n = struct.unpack_from("<I", b, 448)[0]
    out = {}
    for i in range(n):
        kind, nm, off, size = struct.unpack_from("<I16s4xQQ", b, 456 + 40 * i)
        out[f"{kind}:{nm.rstrip(b'\0').decode(errors='replace')}"] = b[off:off + size]
    out["header.uuid"] = b[416:432]
    return out


def cmd_compare(a) -> int:
    files = [a.out / f"{name(a.tag, K, N)}.xclbin" for K, N in SHAPES]
    secs = {f.name: sections(f) for f in files}
    kinds = sorted(set().union(*[s.keys() for s in secs.values()]))
    same_all = True
    print(f"COMPARE {len(files)} xclbins, tag {a.tag}")
    for k in kinds:
        h = [hashlib.sha256(s.get(k, b"<absent>")).hexdigest()[:12] for s in secs.values()]
        sizes = [len(s.get(k, b"")) for s in secs.values()]
        same = len(set(h)) == 1
        same_all &= same or k == "header.uuid"
        print(f"  {k:28s} {'IDENTICAL' if same else 'DIFFERS  '} sizes {sizes} sha {' '.join(h) if not same else h[0]}")
    insts = {f.name: (a.out / f"{f.stem}.bin").stat().st_size for f in files}
    print(f"  instruction streams (bytes): {insts}")
    print(f"VERDICT: {'every section identical (uuid aside): one design can serve all six shapes' if same_all else 'some section differs: see above'}")
    return 0 if same_all else 2


# ---------------------------------------------------------------- NPU
ORDER = [(1024, 3072), (1024, 1024), (1024, 5248), (2624, 1024)] * 28 + \
        [(1024, 3072), (1024, 1024), (1024, 4096), (4096, 1024)] * 2   # Laya's 120 products (laya_npu_screen.ORDER)
XRT_SMI = "C:/Windows/System32/AMD/xrt-smi.exe"
WAIT_MS = 5000
MAX_CTX = 5   # this device holds at most five of these contexts at once (the sixth: 0xc01e0009)


def idle() -> bool:
    out = subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True, timeout=60).stdout
    return "No hardware contexts running" in out


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bits, nearest with ties to even (finite inputs); the same rounding as laya_npu_screen.py."""
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


class Data:
    """One shape's operands, shared by both arms: bf16 A (512 x K) and B (K x N padded), and the fp32 reference."""

    def __init__(self, K, N, rng):
        self.K, self.N, self.np, kr = K, N, padded(N), run_k(K)
        a = np.zeros((M, kr), np.float32)
        a[:, :K] = rng.standard_normal((M, K))   # zero columns past K
        self.a = to_bf16_bits(a)
        w = np.zeros((kr, self.np), np.float32)   # zero rows past K, zero columns past N
        w[:K, :N] = rng.standard_normal((K, N)) * 0.02
        self.b = to_bf16_bits(w)
        f = lambda bits: (bits.astype(np.uint32) << 16).view(np.float32)
        self.ref = f(self.a) @ f(self.b)


class Arm:
    """Loaded designs and BOs. ``designs`` maps each (K, N) to (xclbin, insts): one xclbin for all shapes (generic),
    or one per shape (plain). A design's hw_context is created when a shape first needs it. With ``max_ctx``, at most
    that many stay resident and the least recently used one is destroyed to make room: what IRON's runtime does on
    npu1 when a context creation fails (it evicts and retries), because this device holds at most five of them
    (2026-10-13: the sixth failed with 0xc01e0009). Data BOs are device-level and live for the whole arm; each
    context gets its own instruction BOs, as their memory group comes from its kernel."""

    def __init__(self, pyxrt, dev, designs, data, max_ctx=None):
        from collections import OrderedDict
        self.x, self.dev, self.designs, self.max_ctx = pyxrt, dev, designs, max_ctx
        self.res, self.creates, self.xb, self.d = OrderedDict(), 0, {}, {}
        bo = lambda nbytes: pyxrt.bo(dev, nbytes, pyxrt.bo.host_only, 0)
        for shape, (xclbin, insts) in designs.items():
            if xclbin not in self.xb:
                self.xb[xclbin] = pyxrt.xclbin(str(xclbin))
                dev.register_xclbin(self.xb[xclbin])
            d = data[shape]
            A, B, C = bo(d.a.nbytes), bo(d.b.nbytes), bo(M * d.np * 4)
            A.write(d.a.tobytes(), 0)
            B.write(d.b.tobytes(), 0)
            B.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)   # weights: written once, stay resident
            self.d[shape] = (np.frombuffer(insts.read_bytes(), dtype=np.uint32), A, B, C, d.np)

    def _entry(self, xclbin):
        if xclbin in self.res:
            self.res.move_to_end(xclbin)
            return self.res[xclbin]
        if self.max_ctx is not None and len(self.res) >= self.max_ctx:
            _, old = self.res.popitem(last=False)
            old.clear()   # instruction BOs, kernel, then context go with the last references
            del old
        x = self.x
        ctx = x.hw_context(self.dev, self.xb[xclbin].get_uuid())
        kern = x.kernel(ctx, self.xb[xclbin].get_kernels()[0].get_name())
        ins = {}
        for shape, (xc, _) in self.designs.items():
            if xc == xclbin:
                words = self.d[shape][0]
                ib = x.bo(self.dev, words.nbytes, x.bo.cacheable, kern.group_id(1))
                ib.write(words.tobytes(), 0)
                ib.sync(x.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                ins[shape] = (ib, words.nbytes)
        self.creates += 1
        self.res[xclbin] = {"ins": ins, "kern": kern, "ctx": ctx}
        return self.res[xclbin]

    def step(self, shape):
        e = self._entry(self.designs[shape][0])
        ib, nb = e["ins"][shape]
        _, A, B, C, _ = self.d[shape]
        A.sync(self.x.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        h = e["kern"](3, ib, nb, A, B, C)
        st = h.wait(WAIT_MS)
        if st != self.x.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"{shape}: {st}")
        C.sync(self.x.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def result(self, shape) -> np.ndarray:
        C, npad = self.d[shape][3], self.d[shape][4]
        return np.frombuffer(C.read(M * npad * 4, 0), dtype=np.float32).reshape(M, npad).copy()

    def close(self):
        while self.res:
            _, e = self.res.popitem(last=False)
            e.clear()
        self.d.clear()
        gc.collect()


def cmd_single(a) -> int:
    """Shapes in --seq order, one product each, all in ONE context on design --xclbin; stops at the first failure."""
    sys.path.insert(0, "C:/Xilinx/XRT/xrt_sdk/xrt/python")
    import pyxrt
    if not idle():
        print("NPU not idle: refusing")
        return 3
    seq = [tuple(int(v) for v in x.split(",")) for x in a.seq.split(";")]
    rng = np.random.default_rng(3)
    data = {sh: Data(*sh, rng) for sh in dict.fromkeys(seq)}
    dev = pyxrt.device(0)
    arm = Arm(pyxrt, dev, {sh: (a.out / a.xclbin, a.out / f"{name('g', *sh)}.bin") for sh in data}, data)
    print(f"SEQ on {a.xclbin}, one context: {' -> '.join('%dx%d' % sh for sh in seq)}", flush=True)
    rc = 0
    for i, sh in enumerate(seq):
        K, N = sh
        tag = f"  [{i + 1}] {M}x{K}x{N} (K run as {run_k(K)}, {run_k(K) // TILE['k']} k-tiles, N run as {padded(N)})"
        try:
            arm.step(sh)
        except RuntimeError as e:
            print(f"{tag}: {e}", flush=True)
            rc = 1
            break
        r, d = arm.result(sh), data[sh]
        err = float(np.abs(r[:, :N] - d.ref[:, :N]).max() / np.abs(d.ref[:, :N]).max())
        print(f"{tag}: COMPLETED, |NPU - numpy| / |numpy| {err:.2e}: {'PASS' if err < 1e-3 else 'FAIL'}", flush=True)
        rc |= 0 if err < 1e-3 else 1
    arm.close()
    del dev
    gc.collect()
    print(f"idle after: {idle()}", flush=True)
    return rc


def cmd_run(a) -> int:
    sys.path.insert(0, "C:/Xilinx/XRT/xrt_sdk/xrt/python")
    import pyxrt
    if not idle():
        print("NPU not idle: refusing (a foreign context is contention, not a finding)")
        return 3
    rng = np.random.default_rng(3)
    data = {s: Data(*s, rng) for s in SHAPES}
    one = next(iter(SHAPES))
    gen = {s: (a.out / f"{name('g', *one)}.xclbin", a.out / f"{name('g', *s)}.bin") for s in SHAPES}
    pln = {s: (a.out / f"{name('p', *s)}.xclbin", a.out / f"{name('p', *s)}.bin") for s in SHAPES}
    print(f"RUN one-context arm: {gen[one][0].name} with six instruction streams; per-shape arm: p_*.xclbin, at most 5 contexts resident (LRU)", flush=True)
    dev = pyxrt.device(0)

    # Correctness and bit-equality, each arm on its own (never both open at once).
    out, ok = {}, True
    for arm_name, designs in (("one", gen), ("six", pln)):
        arm = Arm(pyxrt, dev, designs, data, None if arm_name == "one" else MAX_CTX)
        print(f"ARM {arm_name}: {len(arm.xb)} design(s), max resident contexts {arm.max_ctx or len(arm.xb)}", flush=True)
        for s in SHAPES:
            arm.step(s)
            r1 = arm.result(s)
            arm.step(s)
            r2 = arm.result(s)
            d = data[s]
            err = float(np.abs(r1[:, :d.N] - d.ref[:, :d.N]).max() / np.abs(d.ref[:, :d.N]).max())
            pad0 = d.np == d.N or not np.abs(r1[:, d.N:]).any()
            rep = np.array_equal(r1.view(np.uint32), r2.view(np.uint32))
            good = err < 1e-3 and pad0 and rep
            ok &= good
            out[(arm_name, s)] = r1
            print(f"  {arm_name} {M}x{s[0]}x{s[1]}: |NPU - numpy| / |numpy| {err:.2e}, padded columns zero {pad0}, "
                  f"repeat bit-equal {rep}: {'PASS' if good else 'FAIL'}", flush=True)
        arm.close()
    for s in SHAPES:
        eq = np.array_equal(out[("one", s)].view(np.uint32), out[("six", s)].view(np.uint32))
        ok &= eq
        print(f"BITEQUAL {M}x{s[0]}x{s[1]}: one-context generic == per-shape plain: {eq}", flush=True)
    print(f"GATE G1+G2 (correct, repeatable, bit-equal on all six shapes): {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        return 1

    # Timing: Laya's order, arms in alternating blocks; each block opens its arm, warms it, then times.
    seqs, creates = {"one": [], "six": []}, {"one": [], "six": []}
    for blk in range(a.blocks):
        for arm_name, designs in (("one", gen), ("six", pln)) if blk % 2 == 0 else (("six", pln), ("one", gen)):
            arm = Arm(pyxrt, dev, designs, data, None if arm_name == "one" else MAX_CTX)
            for s in ORDER:   # warm-up sequence, untimed
                arm.step(s)
            for _ in range(a.seqs):
                t = time.perf_counter()
                for s in ORDER:
                    arm.step(s)
                seqs[arm_name].append((time.perf_counter() - t) * 1000)
            creates[arm_name].append(arm.creates)
            arm.close()
            print(f"[run] block {blk + 1} {arm_name} done", flush=True)
    for arm_name, v in seqs.items():
        q = statistics.quantiles(v, n=4)
        print(f"CHAIN {arm_name}: {len(v)} sequences of {len(ORDER)} products, ms median {statistics.median(v):.1f}, "
              f"IQR {q[0]:.1f}-{q[2]:.1f}, min {min(v):.1f}, max {max(v):.1f}; context creations per block "
              f"(warm-up + {a.seqs} timed) {creates[arm_name]}", flush=True)
    m1, m6 = statistics.median(seqs["one"]), statistics.median(seqs["six"])
    q1, q6 = statistics.quantiles(seqs["one"], n=4), statistics.quantiles(seqs["six"], n=4)
    gain = q1[2] < q6[0]
    print(f"GATE G3 (one-context IQR wholly below the per-shape IQR): {'PASS' if gain else 'FAIL'}; "
          f"median saving {m6 - m1:.1f} ms per sequence ({(m6 - m1) / len(ORDER):.3f} ms per product)", flush=True)
    print(f"idle after: {idle()}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--plain", action="store_true", help="upstream core (loop counts compiled in), tag p")
    c.add_argument("--even-k", action="store_true", help="pad K to an even number of k-tiles")
    c.add_argument("--shape", help="compile only this K,N (e.g. 1088,1024) instead of Laya's six")
    d = sub.add_parser("compare")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--tag", default="g", choices=["g", "p"])
    d.add_argument("--even-k", action="store_true", help="the builds were compiled with --even-k")
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True, help="directory holding the g_* and p_* builds")
    r.add_argument("--blocks", type=int, default=6, help="timing blocks, arms alternating (ABBA order per pair)")
    r.add_argument("--seqs", type=int, default=5, help="timed 120-product sequences per arm per block")
    r.add_argument("--even-k", action="store_true", help="the builds were compiled with --even-k")
    g = sub.add_parser("single")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--xclbin", required=True, help="xclbin file name inside --out")
    g.add_argument("--seq", required=True, help='shapes in order, e.g. "1024,5248;2624,1024"')
    g.add_argument("--even-k", action="store_true")
    a = ap.parse_args()
    global EVEN_K
    EVEN_K = bool(getattr(a, "even_k", False))
    return {"compile": cmd_compile, "compare": cmd_compare, "run": cmd_run, "single": cmd_single}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
