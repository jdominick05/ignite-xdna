#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Laya's weight products on the NPU in bf16: the screen that bounds a hybrid build before it is built.

Step 3 of the Laya study. Step 2 (tools/laya_accuracy.py) passed bf16 for the 124 weight products with fp32 GEMM
output and everything else in fp32; the hybrid would run those products on the NPU and the rest on the host. At 512
tokens 120 of them have 512 rows and carry all but a rounding error of the 188.6 G MACs; they come in six shapes and
run in a fixed order, each depending on host work before it, so a sequence cannot batch its dispatches. The four
head products with 1-4 rows stay on the host. Before any of that is wired together, these modes measure the parts:

    python tools/laya_npu_screen.py hostshare --model M          (ONNX Runtime) the host's share: ORT profile of the
                                                                  fp32 model, kernel time in the weight products
                                                                  against everything else
    python tools/laya_npu_screen.py noguard --model M            M_noguard.onnx: the Softmax NaN guards removed
    python tools/laya_npu_screen.py samelogits --ref R --arm A   two score files (laya_accuracy.py score): are their
                                                                  logits bit-identical, item by item?
    python tools/laya_npu_screen.py padded --model M --ref R --items I
                                                                  (ONNX Runtime) the first item of each task padded
                                                                  to 512 tokens: M against R, and R against itself
                                                                  unpadded
    python tools/laya_npu_screen.py shapes [--tile T]            (mlir-aie IRON) each shape on the NPU: one check
                                                                  against numpy, then per call the fp32->bf16
                                                                  conversion, the write, the sync in, the call,
                                                                  its NPU bracket, and the fp32 result's sync out
    python tools/laya_npu_screen.py chain [--tile T] --seconds S (mlir-aie IRON) the 120 products in Laya's order,
                                                                  every step above, one [run] frame line per sequence

``chain`` does no host compute between products (the activations are fixed random data), so its time and energy are
the NPU side of a sequence; ``hostshare`` gives the host side. Their sum is DERIVED, not a hybrid run. The GEMM is
mlir-aie's ``whole_array`` design (4 columns, bf16 in, f32 out, so the K reduction accumulates in fp32), imported from
the mlir-aie checkout. ``--tile default`` is its default tile (m=64, k=64, n=32, C double-buffered); ``--tile best``
is n=64 with the C tile single-buffered (a local patch, kernels/gemm_tile_sweep/). N=5248 allows neither m=64 (the C
row-block stride would pass 4 MiB) nor n=64 (5248 is not a multiple of 4 x 64): ``default`` runs it at m=32, n=32,
and ``best`` pads it with zero columns to 5376 and runs m=32, n=64.
"""
import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

M = 512
LAYER = [(1024, 3072), (1024, 1024), (1024, 5248), (2624, 1024)]   # Wqkv, Wo, Wi (GeGLU), Wo2: K, N
HEAD = [(1024, 3072), (1024, 1024), (1024, 4096), (4096, 1024)]    # the decision head's two blocks, full sequence
ORDER = LAYER * 28 + HEAD * 2
WHOLE_ARRAY = Path.home() / "mlir-aie" / "programming_examples" / "basic" / "matrix_multiplication" / "whole_array"


def tile(K: int, N: int, which: str) -> tuple[int, int, int, bool, int]:
    """(m, k, n, C single-buffered, N as run). max N = 4 MiB / (m * 4 rows * 4 B) for f32 out: 4096 at m=64, 8192 at
    m=32. ``best`` pads N with zero columns to a multiple of 4 x 64 so every shape takes n=64 (5248 -> 5376); the
    extra columns are computed, synced and discarded."""
    if which == "best":
        npad = -(-N // 256) * 256
        return (64 if npad <= 4096 else 32), 64, 64, True, npad
    return (64 if N <= 4096 else 32), 64, 32, False, N


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bits, nearest with ties to even (finite inputs)."""
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def keepawake(on: bool) -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    print(f"KEEPAWAKE {'HOLD' if on else 'RELEASE'}", flush=True)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- hostshare (ONNX Runtime)
def cmd_hostshare(a) -> int:
    import onnx
    import onnxruntime as ort
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import laya_baseline as lb
    model = Path(a.model)
    g = onnx.load(str(model), load_external_data=False)
    init = {i.name: list(i.dims) for i in g.graph.initializer}
    wp = {n.name for n in g.graph.node
          if n.op_type in ("MatMul", "Gemm") and n.input[1] in init and len(init[n.input[1]]) == 2}
    feed = lb.inputs(512, 1)
    plain = lb.session(model, "cpu", a.threads)
    for _ in range(3):
        plain.run(None, feed)
    wall = []
    for _ in range(a.calls):
        t = time.perf_counter()
        plain.run(None, feed)
        wall.append((time.perf_counter() - t) * 1000)
    del plain
    prof_s = lb.session(model, "cpu", a.threads, profile=str(Path(a.profile_dir) / "laya_hostshare"))
    for _ in range(3 + a.calls):
        prof_s.run(None, feed)
    prof = Path(prof_s.end_profiling())
    ev = json.loads(prof.read_text(encoding="utf-8"))
    prof.unlink(missing_ok=True)
    runs = [e for e in ev if e.get("cat") == "Session" and e.get("name") == "model_run"]
    last = runs[-a.calls:]
    t0, t1 = last[0]["ts"], last[-1]["ts"] + last[-1]["dur"]
    kern = [e for e in ev if e.get("cat") == "Node" and e["name"].endswith("_kernel_time") and t0 <= e["ts"] <= t1]
    seen, w_us, r_us, by_op = set(), 0.0, 0.0, {}
    for e in kern:
        name = e["name"].removesuffix("_kernel_time")
        if name in wp:
            seen.add(name)
            w_us += e["dur"]
        else:
            r_us += e["dur"]
            op = e.get("args", {}).get("op_name", "?")
            by_op[op] = by_op.get(op, 0.0) + e["dur"]
    prof_wall = sum(r["dur"] for r in last) / a.calls / 1000
    top = sorted(by_op.items(), key=lambda x: -x[1])[:8]
    print(f"HOSTSHARE {model.name} seq 512 batch 1, {a.threads} threads, onnxruntime {ort.__version__}: wall per call "
          f"{statistics.mean(wall):.1f} ms mean, {statistics.median(wall):.1f} median over {a.calls} unprofiled calls; "
          f"profiled {prof_wall:.1f} ms")
    print(f"HOSTSHARE per call, kernel time: the weight products {w_us / a.calls / 1000:.1f} ms ({len(seen)} of "
          f"{len(wp)} named in the profile); everything else {r_us / a.calls / 1000:.1f} ms; outside kernels "
          f"{prof_wall - (w_us + r_us) / a.calls / 1000:.1f} ms")
    print("HOSTSHARE everything else by op, ms per call: " +
          ", ".join(f"{op} {us / a.calls / 1000:.1f}" for op, us in top))
    return 0


def cmd_noguard(a) -> int:
    """<model>_noguard.onnx: every Softmax NaN guard removed. torch's safe softmax exports as
    Where(IsNaN(s), 0, s) over each layer's attention probabilities; a NaN there needs a fully masked row, which one
    unpadded sequence never has, so the guard is an identity for the inputs this study runs (``padded`` checks padded
    ones). A guard is removed only
    in exactly that form, with its IsNaN feeding nothing else; the copy reads the same external weights."""
    import onnx
    from onnx import numpy_helper
    src = Path(a.model)
    dst = src.with_name(src.stem + "_noguard.onnx")
    if dst.exists():
        raise SystemExit(f"{dst.name} exists; this tool never writes over a model")
    m = onnx.load(str(src), load_external_data=False)
    prod = {o: n for n in m.graph.node for o in n.output}
    users = {}
    for n in m.graph.node:
        for x in n.input:
            users.setdefault(x, []).append(n)
    init = {i.name: i for i in m.graph.initializer}
    drop, rename = set(), {}
    for n in m.graph.node:
        isn = prod.get(n.input[0]) if n.op_type == "Where" else None
        if isn is None or isn.op_type != "IsNaN" or isn.input[0] != n.input[2]:
            continue
        sm = prod.get(n.input[2])
        zero = n.input[1] in init and not numpy_helper.to_array(init[n.input[1]]).any()
        if sm is None or sm.op_type != "Softmax" or not zero or len(users[isn.output[0]]) != 1:
            continue
        drop.update({n.name, isn.name})
        rename[n.output[0]] = n.input[2]
    kept = [n for n in m.graph.node if n.name not in drop]
    for n in kept:
        for i, x in enumerate(n.input):
            if x in rename:
                n.input[i] = rename[x]
    del m.graph.node[:]
    m.graph.node.extend(kept)
    onnx.save(m, str(dst))
    print(f"NOGUARD {dst.name} from {src.name}: {len(rename)} Softmax NaN guards removed ({len(drop)} nodes); "
          f"{sum(n.op_type == 'IsNaN' for n in kept)} IsNaN and {sum(n.op_type == 'Where' for n in kept)} Where left; "
          f"sha256 {sha256(dst)}")
    return 0


def cmd_samelogits(a) -> int:
    ra, rb = np.load(a.ref), np.load(a.arm)
    za, zb, k = ra["logits"], rb["logits"], ra["k"]
    if not np.array_equal(k, rb["k"]):
        raise SystemExit("the two files do not score the same items")
    same = sum(np.array_equal(za[j, :k[j]], zb[j, :k[j]]) for j in range(len(k)))
    d = max(float(np.abs(za[j, :k[j]] - zb[j, :k[j]]).max()) for j in range(len(k)))
    print(f"SAMELOGITS {Path(a.arm).name} vs {Path(a.ref).name}: items bit-identical {same}/{len(k)}; "
          f"largest |logit diff| {d:.3e}")
    return 0


def cmd_padded(a) -> int:
    """A fixed-512 build pads a shorter question with [PAD] under attention mask 0, which leaves whole rows of the
    local-attention layers masked: the case the guards exist for. The first item of each task in --items, at 512."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import laya_baseline as lb
    pad_id = 50283   # [PAD], as in tools/laya_accuracy.py
    pick, seen = [], set()
    for x in Path(a.items).read_text(encoding="utf-8").splitlines():
        it = json.loads(x) if x.strip() else None
        if it and it["task"] not in seen:
            seen.add(it["task"])
            pick.append(it)

    def feed(it, pad):
        n, k = len(it["ids"]), len(it["markers"])
        extra = M - n if pad else 0
        return {"input_ids": np.array([it["ids"] + [pad_id] * extra], np.int64),
                "attention_mask": np.array([[1] * n + [0] * extra], np.int64),
                "marker_pos": np.array([it["markers"]], np.int64), "marker_mask": np.ones((1, k), bool),
                "qtype": np.array([it["qtype"]], np.int64)}

    out = {}
    for which, model in (("arm", Path(a.model)), ("ref", Path(a.ref))):
        s = lb.session(model, "cpu", a.threads)
        for j, it in enumerate(pick):
            for pad in ((True, False) if which == "ref" else (True,)):
                out[which, j, pad] = s.run(None, feed(it, pad))[0][0, :it["k"]]
        del s
    same = 0
    for j, it in enumerate(pick):
        z, r, r0 = out["arm", j, True], out["ref", j, True], out["ref", j, False]
        same += np.array_equal(z, r)
        print(f"PADDED {it['task']} item {it['i']}, {len(it['ids'])} tokens padded to {M}: {Path(a.model).name} "
              f"finite {bool(np.isfinite(z).all())}, bit-identical to {Path(a.ref).name} {bool(np.array_equal(z, r))}; "
              f"{Path(a.ref).name} padded against unpadded: largest |logit diff| {float(np.abs(r - r0).max()):.3e}, "
              f"same top option {int(r.argmax()) == int(r0.argmax())}")
    print(f"PADDED {same}/{len(pick)} items bit-identical at {M} tokens")
    return 0


# ---------------------------------------------------------------- NPU (mlir-aie IRON)
class Product:
    """One shape on the NPU: its tensors and its compiled design, called the way a hybrid would call it."""

    def __init__(self, wa, iron, bf16, K, N, which, rng, pool=None):
        self.K, self.N, self.pool = K, N, pool
        self.m, self.k, self.n, self.csb, self.np = tile(K, N, which)
        self.wa, self.bf16 = wa, bf16
        self.x = rng.standard_normal((M, K)).astype(np.float32)             # stands in for the host's activation
        w = np.zeros((K, self.np), np.float32)
        w[:, :N] = rng.standard_normal((K, N)) * 0.02
        self.wb = to_bf16_bits(w).view(bf16)
        self.A = iron.zeros((M, K), dtype=bf16, device="npu")
        self.B = iron.tensor(self.wb, dtype=bf16, device="npu")             # weights: written once, stay resident
        self.C = iron.zeros((M, self.np), dtype=np.float32, device="npu")
        self.kw = dict(M=M, K=K, N=self.np, m=self.m, k=self.k, n=self.n, n_aie_cols=4, dtype_in_str="bf16",
                       dtype_out_str="f32", c_single_buffer=self.csb)

    def step(self) -> tuple[float, float, float, float, float]:
        """convert-and-write, sync in, call, NPU bracket, sync out: seconds. The fp32 activation is assigned into the
        bf16 buffer, so ml_dtypes rounds it (nearest, ties to even: bit-identical to to_bf16_bits) as it copies."""
        t0 = time.perf_counter()
        with self.A.mutate() as arr:
            if self.pool is None:
                arr[...] = self.x
            else:   # row blocks on a thread pool: the cast releases the GIL for part of its work
                step = M // self.pool._max_workers
                list(self.pool.map(lambda i: arr.__setitem__(slice(i, i + step), self.x[i:i + step]), range(0, M, step)))
        t1 = time.perf_counter()
        self.A.to("npu")
        t2 = time.perf_counter()
        ret = self.wa.whole_array(self.A, self.B, self.C, **self.kw)
        t3 = time.perf_counter()
        self.C.numpy()
        t4 = time.perf_counter()
        npu = getattr(ret[1] if isinstance(ret, tuple) else ret, "npu_time", 0) / 1e9
        return t1 - t0, t2 - t1, t3 - t2, npu, t4 - t3

    def check(self) -> float:
        """The NPU result against numpy on the same bf16 operands in fp32: largest |diff| over largest |ref|, over
        the real columns; the padded columns must come back 0."""
        self.step()
        got = np.array(self.C.numpy(), dtype=np.float32)
        ref = to_bf16_bits(self.x).view(self.bf16).astype(np.float32) @ self.wb.astype(np.float32)
        if not np.array_equal(np.array(self.A.numpy()).view(np.uint16), to_bf16_bits(self.x)):
            return float("inf")   # the buffer's rounding is not nearest-even
        if self.np > self.N and np.abs(got[:, self.N:]).max() != 0:
            return float("inf")
        return float(np.abs(got[:, :self.N] - ref[:, :self.N]).max() / np.abs(ref[:, :self.N]).max())


def npu_setup(which: str, conv_threads: int = 1):
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(conv_threads) if conv_threads > 1 else None
    sys.path.insert(0, str(WHOLE_ARRAY))
    import aie.iron as iron
    from aie.iron import str_to_dtype
    from aie.iron.device import from_name
    from aie.utils.hostruntime import set_current_device
    import whole_array as wa
    set_current_device(from_name("npu", n_cols=4))
    src = (WHOLE_ARRAY / "whole_array.py").read_text(encoding="utf-8")
    print(f"[summary] NPU whole_array.py sha256 {sha256(WHOLE_ARRAY / 'whole_array.py')}; C single-buffer patch "
          f"{'present' if 'c_single_buffer' in src else 'absent'}; tile {which}; conversion threads {conv_threads}; "
          f"python {sys.version.split()[0]}, "
          f"numpy {np.__version__}", flush=True)
    bf16 = str_to_dtype("bf16")
    rng = np.random.default_rng(3)
    prods, t0 = {}, time.perf_counter()
    for K, N in dict.fromkeys(ORDER):
        p = Product(wa, iron, bf16, K, N, which, rng, pool)
        t = time.perf_counter()
        err = p.check()   # the first call compiles the design, or loads it from the JIT cache
        print(f"[summary] NPU {M}x{K}x{N}{f' run as N={p.np}' if p.np != N else ''} tile m{p.m} k{p.k} n{p.n}"
              f"{' C single-buffered' if p.csb else ''}: first call "
              f"{time.perf_counter() - t:.1f} s; largest |NPU - numpy| / largest |numpy| {err:.2e} "
              f"({'PASS' if err < 1e-3 else 'FAIL'})", flush=True)
        prods[(K, N)] = p
    print(f"[summary] NPU {len(prods)} designs ready in {time.perf_counter() - t0:.0f} s", flush=True)
    return prods


def cmd_shapes(a) -> int:
    prods = npu_setup(a.tile, a.conv_threads)
    keepawake(True)
    try:
        total = [0.0] * 5
        for (K, N), p in prods.items():
            for _ in range(a.warmup):
                p.step()
            s = np.array([p.step() for _ in range(a.iters)]) * 1000   # ms
            mean = s.mean(axis=0)
            cnt = ORDER.count((K, N))
            total = [x + cnt * y for x, y in zip(total, mean)]
            print(f"[summary] SHAPE {M}x{K}x{N} x{cnt}: ms per call mean over {a.iters} (median): convert-and-write "
                  f"{mean[0]:.3f}, sync in {mean[1]:.3f}, call {mean[2]:.3f} ({np.median(s[:, 2]):.3f}), NPU bracket "
                  f"{mean[3]:.3f} ({np.median(s[:, 3]):.3f}), sync out {mean[4]:.3f}; "
                  f"{2 * M * K * N / (mean[3] / 1000) / 1e9:.0f} GFLOPS of Laya's own work in the NPU bracket", flush=True)
        print(f"[summary] SHAPES per sequence (120 products, each shape repeated alone, no switching), ms: convert-and-write "
              f"{total[0]:.1f}, sync in {total[1]:.1f}, call {total[2]:.1f} of which NPU bracket {total[3]:.1f}, sync out "
              f"{total[4]:.1f}; sum {total[0] + total[1] + total[2] + total[4]:.1f}", flush=True)
    finally:
        keepawake(False)
    return 0


def cmd_chain(a) -> int:
    prods = npu_setup(a.tile, a.conv_threads)
    for (K, N), p in prods.items():   # warm every design once more before the frames start
        p.step()
    keepawake(True)
    seqs, parts, frame, start = [], np.zeros(5), 0, time.perf_counter()
    try:
        while frame < a.min_frames or time.perf_counter() - start < a.seconds:
            t = time.perf_counter()
            for K, N in ORDER:
                parts += np.array(prods[(K, N)].step())
            seqs.append((time.perf_counter() - t) * 1000)
            frame += 1
            print(f"[run] frame {frame}", flush=True)
    finally:
        keepawake(False)
    pm = parts / frame * 1000
    print(f"[summary] CHAIN {frame} sequences of 120 products in Laya's order, tile {a.tile}: ms per sequence mean "
          f"{statistics.mean(seqs):.1f}, median {statistics.median(seqs):.1f}, min {min(seqs):.1f}, max {max(seqs):.1f}",
          flush=True)
    print(f"[summary] CHAIN per sequence, ms: convert-and-write {pm[0]:.1f}, sync in {pm[1]:.1f}, call {pm[2]:.1f} of which NPU "
          f"bracket {pm[3]:.1f}, sync out {pm[4]:.1f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("hostshare", "noguard", "samelogits", "padded", "shapes", "chain"))
    ap.add_argument("--model", help="hostshare, noguard, padded: an fp32 ONNX model")
    ap.add_argument("--ref", help="samelogits: the reference score file; padded: the reference model")
    ap.add_argument("--arm", help="samelogits: the score file compared")
    ap.add_argument("--items", help="padded: the items file (tools/laya_accuracy.py items)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--calls", type=int, default=10, help="hostshare: calls timed and profiled")
    ap.add_argument("--profile-dir", default=".")
    ap.add_argument("--tile", choices=("default", "best"), default="best")
    ap.add_argument("--conv-threads", type=int, default=1, help="NPU modes: threads for the fp32->bf16 conversion")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--min-frames", type=int, default=10)
    a = ap.parse_args()
    return {"hostshare": cmd_hostshare, "noguard": cmd_noguard, "samelogits": cmd_samelogits, "padded": cmd_padded,
            "shapes": cmd_shapes, "chain": cmd_chain}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
