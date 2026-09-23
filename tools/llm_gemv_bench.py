#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""M = 1 decode GEMV on the CPU and on the Radeon 780M (DirectML) through ONNX Runtime.

The LLM study's yardsticks: the NPU earns a decode role only if it beats both of these, in speed or
in accuracy. No NPU is touched here.

    python tools/llm_gemv_bench.py controls --ep {cpu,dml}
        packing and placement checks, no timing: MatMulNBits against a float64 reference on a small
        shape, and (dml) a negative control proving session.disable_cpu_ep_fallback refuses a node
        DirectML cannot run.
    python tools/llm_gemv_bench.py build --shape K N --block B
        writes the int4 weights for one shape and block (fp32- and fp16-scale variants) and the dense
        dequantized weights into scratch/llm/. Deterministic from --seed; no timing.
    python tools/llm_gemv_bench.py readbw --ep {cpu,dml} --dtype {fp32,fp16}
        read bandwidth through ONNX Runtime: ReduceSum over a 1 GiB initializer.
    python tools/llm_gemv_bench.py gemv --ep {cpu,dml} --arm {nbits,dense} --shape K N ...
        one row: y = x W^T at M = 1, timed over R distinct weight copies totalling >= 1 GiB so every
        run streams from DRAM (the 8700G's L3 is 16 MiB), plus the error of one GEMV against float64.

Weights: w ~ N(0, 0.02), round-to-nearest asymmetric uint4 per block of K (MatMulNBits' layout: two
per byte, low nibble first, one zero point per block, packed two per byte). Scales are rounded to
fp16 first, so the fp32- and fp16-scale variants dequantize to the same values. The dense arm stores
the block-32 copy's dequantized weights, so every arm at a shape is scored against one float64
reference: y_ref = x W^T in float64 with x rounded to fp16.

Every measured row prints one ROW_JSON line; tools/llm_decode_verdict.py reads them.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch" / "llm"
TARGET_BYTES = 1 << 30            # weight bytes streamed per timed run
W_STD = 0.02                      # Llama-like weight scale; values do not change speed
SEED = 20260923
ALIGN = 4096
GEN = "rtn-asym-uint4-v1"         # bump if the generator changes; it names the scratch dirs


# ---------------------------------------------------------------- weights

def quantize_rtn(w: np.ndarray, block: int):
    """[N, K] float32 -> q uint8 [N, K], scales float32 [N, K/block] (fp16-exact), zp uint8 [N, K/block]."""
    n, k = w.shape
    wb = w.reshape(n, k // block, block)
    mn, mx = wb.min(axis=2), wb.max(axis=2)
    s = np.maximum((mx - mn) / 15.0, 1e-8).astype(np.float16).astype(np.float32)
    zp = np.clip(np.rint(-mn / s), 0, 15).astype(np.uint8)
    q = np.clip(np.rint(wb / s[:, :, None]) + zp[:, :, None], 0, 15).astype(np.uint8)
    return q.reshape(n, k), s, zp


def dequant(q, s, zp, block, dtype=np.float64):
    return ((q.astype(dtype) - np.repeat(zp.astype(dtype), block, axis=1))
            * np.repeat(s.astype(dtype), block, axis=1))


def pack_nibbles(a: np.ndarray) -> np.ndarray:
    """Two uint4 per byte along the last axis, low nibble first."""
    return (a[..., 0::2] | (a[..., 1::2] << 4)).astype(np.uint8)


def copy_weights(k: int, n: int, block: int, i: int, seed: int):
    rng = np.random.default_rng(seed + i)
    w = (rng.standard_normal((n, k), dtype=np.float32) * W_STD)
    return quantize_rtn(w, block)


def x_vector(k: int, seed: int) -> np.ndarray:
    """The activation, fp16-exact so every arm sees the same numbers."""
    return np.random.default_rng(seed + 10**6 + k).standard_normal((1, k)).astype(np.float16).astype(np.float32)


class ExtWriter:
    """Append numpy arrays to one external-data file; return TensorProtos that point at them."""

    def __init__(self, path: Path):
        self.path, self.fh = path, open(path, "wb")

    def add(self, name: str, arr: np.ndarray) -> TensorProto:
        pad = (-self.fh.tell()) % ALIGN
        self.fh.write(b"\0" * pad)
        off = self.fh.tell()
        self.fh.write(np.ascontiguousarray(arr).tobytes())
        t = TensorProto(name=name, data_type=helper.np_dtype_to_tensor_dtype(arr.dtype), dims=arr.shape)
        t.data_location = TensorProto.EXTERNAL
        for key, val in (("location", self.path.name), ("offset", str(off)), ("length", str(arr.nbytes))):
            e = t.external_data.add()
            e.key, e.value = key, val
        return t

    def add_stream(self, name: str, dtype, shape, chunks) -> TensorProto:
        """One tensor written chunk by chunk, so a 1 GiB initializer never sits in memory whole."""
        pad = (-self.fh.tell()) % ALIGN
        self.fh.write(b"\0" * pad)
        off = self.fh.tell()
        for c in chunks:
            self.fh.write(np.ascontiguousarray(c, dtype=dtype).tobytes())
        length = self.fh.tell() - off
        if length != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            raise ValueError(f"{name}: wrote {length} bytes for shape {shape}")
        t = TensorProto(name=name, data_type=helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), dims=shape)
        t.data_location = TensorProto.EXTERNAL
        for key, val in (("location", self.path.name), ("offset", str(off)), ("length", str(length))):
            e = t.external_data.add()
            e.key, e.value = key, val
        return t

    def close(self):
        self.fh.close()


def shape_dir(kind: str, k: int, n: int, block: int, seed: int) -> Path:
    return SCRATCH / f"{kind}_k{k}_n{n}_b{block}_s{seed}_m{TARGET_BYTES >> 20}_{GEN}"


def nbits_copy_bytes(k, n, block, t1):
    kb = k // block
    return n * k // 2 + n * kb * (4 if t1 == "fp32" else 2) + n * kb // 2


def copies_for(bytes_per_copy: int) -> int:
    return max(1, -(-TARGET_BYTES // bytes_per_copy))


def cmd_build(args) -> int:
    k, n, block = args.shape[0], args.shape[1], args.block
    if k % block or (k // block) % 2:
        raise SystemExit(f"K={k} needs an even number of blocks of {block}")
    d = shape_dir("nbits", k, n, block, args.seed)
    r = copies_for(nbits_copy_bytes(k, n, block, "fp32"))
    manifest = d / "manifest.json"
    if manifest.exists():
        print(f"exists: {d.relative_to(ROOT)}")
    else:
        d.mkdir(parents=True, exist_ok=True)
        writers = {t1: ExtWriter(d / f"weights_{t1}.bin") for t1 in ("fp32", "fp16")}
        tensors = {t1: [] for t1 in writers}
        t0 = time.perf_counter()
        for i in range(r):
            q, s, zp = copy_weights(k, n, block, i, args.seed)
            b = pack_nibbles(q).reshape(n, k // block, block // 2)
            z = pack_nibbles(zp).reshape(-1)
            for t1, wr in writers.items():
                sd = s.astype(np.float32 if t1 == "fp32" else np.float16).reshape(-1)
                tensors[t1].append([wr.add(f"B{i}", b), wr.add(f"S{i}", sd), wr.add(f"Z{i}", z)])
        for t1, wr in writers.items():
            wr.close()
            for acc in (0, 4):
                write_nbits_graphs(d, t1, acc, k, n, block, tensors[t1])
        info = {"k": k, "n": n, "block": block, "copies": r, "seed": args.seed, "gen": GEN,
                "copy_bytes": {t1: nbits_copy_bytes(k, n, block, t1) for t1 in writers},
                "build_s": round(time.perf_counter() - t0, 1)}
        manifest.write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(f"built {d.relative_to(ROOT)}: {r} copies in {info['build_s']} s")
    if block == 32:
        build_dense(k, n, args.seed)
    return 0


def write_nbits_graphs(d, t1, acc, k, n, block, tensors):
    tt = TensorProto.FLOAT if t1 == "fp32" else TensorProto.FLOAT16
    for mode, use in (("time", tensors), ("one", tensors[:1])):
        nodes = [helper.make_node("MatMulNBits", ["x", *[t.name for t in tr]], [f"y{i}"], domain="com.microsoft",
                                  K=k, N=n, bits=4, block_size=block, accuracy_level=acc)
                 for i, tr in enumerate(use)]
        outs = [f"y{i}" for i in range(len(use))]
        if len(outs) > 1:
            nodes.append(helper.make_node("Sum", outs, ["y"]))
        else:
            nodes.append(helper.make_node("Identity", outs, ["y"]))
        g = helper.make_graph(nodes, f"{mode}_{t1}", [helper.make_tensor_value_info("x", tt, [1, k])],
                              [helper.make_tensor_value_info("y", tt, [1, n])], [t for tr in use for t in tr])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)])
        m.ir_version = 9
        (d / f"{mode}_{t1}_acc{acc}.onnx").write_bytes(m.SerializeToString())


def build_dense(k, n, seed):
    d = shape_dir("dense", k, n, 32, seed)
    manifest = d / "manifest.json"
    if manifest.exists():
        print(f"exists: {d.relative_to(ROOT)}")
        return
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    info = {"k": k, "n": n, "block": 32, "seed": seed, "gen": GEN, "copies": {}}
    for t1, npt, tt in (("fp32", np.float32, TensorProto.FLOAT), ("fp16", np.float16, TensorProto.FLOAT16)):
        r = copies_for(k * n * np.dtype(npt).itemsize)
        wr = ExtWriter(d / f"weights_{t1}.bin")
        tensors = []
        for i in range(r):
            q, s, zp = copy_weights(k, n, 32, i, seed)
            tensors.append(wr.add(f"W{i}", np.ascontiguousarray(dequant(q, s, zp, 32, np.float32).T.astype(npt))))
        wr.close()
        for mode, use in (("time", tensors), ("one", tensors[:1])):
            nodes = [helper.make_node("MatMul", ["x", t.name], [f"y{i}"]) for i, t in enumerate(use)]
            outs = [f"y{i}" for i in range(len(use))]
            nodes.append(helper.make_node("Sum" if len(outs) > 1 else "Identity", outs, ["y"]))
            g = helper.make_graph(nodes, f"{mode}_{t1}", [helper.make_tensor_value_info("x", tt, [1, k])],
                                  [helper.make_tensor_value_info("y", tt, [1, n])], use)
            m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
            m.ir_version = 9
            (d / f"{mode}_{t1}_acc0.onnx").write_bytes(m.SerializeToString())
        info["copies"][t1] = r
    info["build_s"] = round(time.perf_counter() - t0, 1)
    manifest.write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"built {d.relative_to(ROOT)}: {info['copies']} copies in {info['build_s']} s")


# ---------------------------------------------------------------- sessions

def make_session(model, ep: str, threads: int, opt: str = "default"):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = {"default": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                                   "disable_all": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}[opt]
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if ep == "cpu":
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
    else:
        so.enable_mem_pattern = False
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers = [("DmlExecutionProvider", {"device_id": 0})]
    src = str(model) if isinstance(model, Path) else model
    return ort.InferenceSession(src, so, providers=providers)


def header(ep: str) -> dict:
    import onnxruntime as ort
    h = {"python": platform.python_version(), "onnxruntime": ort.__version__, "numpy": np.__version__,
         "onnx": onnx.__version__, "env": os.environ.get("CONDA_DEFAULT_ENV", "?"), "host": platform.node(),
         "ep": ep, "cpu": platform.processor()}
    print("HEADER " + json.dumps(h, sort_keys=True))
    return h


def timed(sess, feed, warmup: int, reps: int):
    for _ in range(warmup):
        sess.run(None, feed)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sess.run(None, feed)
        ts.append(time.perf_counter() - t0)
    return ts


def errors(y: np.ndarray, ref: np.ndarray) -> dict:
    y = y.astype(np.float64).reshape(-1)
    ref = ref.reshape(-1)
    return {"rel_l2": float(np.linalg.norm(y - ref) / np.linalg.norm(ref)),
            "max_abs_rel": float(np.max(np.abs(y - ref)) / np.max(np.abs(ref))),
            "finite": bool(np.all(np.isfinite(y)))}


# ---------------------------------------------------------------- commands

def cmd_controls(args) -> int:
    header(args.ep)
    k, n, block = 256, 128, 32
    x = x_vector(k, args.seed)
    q, s, zp = copy_weights(k, n, block, 0, args.seed)
    ref = x.astype(np.float64) @ dequant(q, s, zp, block).T
    b = pack_nibbles(q).reshape(n, k // block, block // 2)
    z = pack_nibbles(zp).reshape(-1)
    fails = 0
    t1s = ("fp32", "fp16")
    accs = (0, 4) if args.ep == "cpu" else (0,)
    for t1 in t1s:
        npt, tt = (np.float32, TensorProto.FLOAT) if t1 == "fp32" else (np.float16, TensorProto.FLOAT16)
        for acc in accs:
            node = helper.make_node("MatMulNBits", ["x", "B", "S", "Z"], ["y"], domain="com.microsoft",
                                    K=k, N=n, bits=4, block_size=block, accuracy_level=acc)
            g = helper.make_graph([node], "c", [helper.make_tensor_value_info("x", tt, [1, k])],
                                  [helper.make_tensor_value_info("y", tt, [1, n])],
                                  [onnx.numpy_helper.from_array(b, "B"),
                                   onnx.numpy_helper.from_array(s.astype(npt).reshape(-1), "S"),
                                   onnx.numpy_helper.from_array(z, "Z")])
            m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17),
                                                    helper.make_opsetid("com.microsoft", 1)])
            m.ir_version = 9
            sess = make_session(m.SerializeToString(), args.ep, 1)
            e = errors(sess.run(None, {"x": x.astype(npt)})[0], ref)
            # fp32 activations with fp32 compute must reproduce the float64 result to fp32 rounding
            exact = t1 == "fp32" and acc == 0
            ok = e["finite"] and (e["rel_l2"] < 1e-6 if exact else e["rel_l2"] < 3e-2)
            fails += not ok
            print(f"[controls] {args.ep} MatMulNBits t1={t1} acc={acc}: rel_l2 {e['rel_l2']:.3e} "
                  f"max_abs_rel {e['max_abs_rel']:.3e} -> {'PASS' if ok else 'FAIL'}"
                  f"{' (packing check: must be < 1e-6)' if exact else ''}")
    if args.ep == "dml":
        # negative control: a string op DirectML has no kernel for must be refused, not run on the CPU
        node = helper.make_node("StringNormalizer", ["s"], ["o"], case_change_action="LOWER")
        g = helper.make_graph([node], "neg", [helper.make_tensor_value_info("s", TensorProto.STRING, [2])],
                              [helper.make_tensor_value_info("o", TensorProto.STRING, [2])])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        try:
            make_session(m.SerializeToString(), "dml", 1)
            print("[controls] dml negative control FAIL: a StringNormalizer session was created, so "
                  "disable_cpu_ep_fallback does not guarantee DirectML placement")
            fails += 1
        except Exception as ex:
            print(f"[controls] dml negative control PASS: refused ({type(ex).__name__}: "
                  f"{str(ex).splitlines()[0][:140]})")
    print(f"[controls] {'PASS' if not fails else f'FAIL ({fails})'}")
    return 1 if fails else 0


def cmd_readbw(args) -> int:
    header(args.ep)
    npt, tt = (np.float32, TensorProto.FLOAT) if args.dtype == "fp32" else (np.float16, TensorProto.FLOAT16)
    cols = 4096
    rows = TARGET_BYTES // (cols * np.dtype(npt).itemsize)
    d = SCRATCH / f"readbw_{args.dtype}_{rows}x{cols}"
    path = d / "model.onnx"
    if not path.exists():
        d.mkdir(parents=True, exist_ok=True)
        wr = ExtWriter(d / "weights.bin")
        rng = np.random.default_rng(args.seed)
        chunk = 8192
        w = wr.add_stream("W", npt, [rows, cols],
                          (rng.random((min(chunk, rows - r0), cols), dtype=np.float32) for r0 in range(0, rows, chunk)))
        wr.close()
        axes = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")
        nodes = [helper.make_node("ReduceSum", ["W", "axes"], ["r"], keepdims=0),
                 helper.make_node("Add", ["r", "x"], ["y"])]
        g = helper.make_graph(nodes, "readbw", [helper.make_tensor_value_info("x", tt, [1])],
                              [helper.make_tensor_value_info("y", tt, [rows])], [w, axes])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        path.write_bytes(m.SerializeToString())
    nbytes = rows * cols * np.dtype(npt).itemsize
    # ORT_DISABLE_ALL: ReduceSum of an initializer would otherwise be constant-folded away
    sess = make_session(path, args.ep, args.threads, opt="disable_all")
    ts = timed(sess, {"x": np.zeros(1, npt)}, args.warmup, args.reps)
    med = statistics.median(ts)
    row = {"kind": "readbw", "ep": args.ep, "dtype": args.dtype, "bytes": nbytes, "threads": args.threads if args.ep == "cpu" else None,
           "median_ms": med * 1e3, "min_ms": min(ts) * 1e3, "max_ms": max(ts) * 1e3, "reps": args.reps,
           "gbps": nbytes / med / 1e9, "gbps_best": nbytes / min(ts) / 1e9}
    print(f"readbw {args.ep} {args.dtype}: {nbytes / 2**30:.2f} GiB, median {med * 1e3:.2f} ms -> "
          f"{row['gbps']:.2f} GB/s (best {row['gbps_best']:.2f})")
    print("ROW_JSON " + json.dumps(row, sort_keys=True))
    return 0


def cmd_gemv(args) -> int:
    header(args.ep)
    k, n = args.shape
    if args.arm == "nbits":
        d = shape_dir("nbits", k, n, args.block, args.seed)
    else:
        d = shape_dir("dense", k, n, 32, args.seed)
    man = d / "manifest.json"
    if not man.exists():
        raise SystemExit(f"missing {d}: run build --shape {k} {n} --block {args.block if args.arm == 'nbits' else 32}")
    info = json.loads(man.read_text(encoding="utf-8"))
    acc = args.acc if (args.arm == "nbits" and args.ep == "cpu") else 0
    npt = np.float32 if args.t1 == "fp32" else np.float16
    if args.arm == "nbits":
        copies, per_copy = info["copies"], info["copy_bytes"][args.t1]
    else:
        copies, per_copy = info["copies"][args.t1], k * n * np.dtype(npt).itemsize
    t_model = d / f"time_{args.t1}_acc{acc}.onnx"
    o_model = d / f"one_{args.t1}_acc{acc}.onnx"
    x = x_vector(k, args.seed)
    feed = {"x": x.astype(npt)}

    t0 = time.perf_counter()
    sess = make_session(t_model, args.ep, args.threads)
    load_s = time.perf_counter() - t0
    ts = timed(sess, feed, args.warmup, args.reps)
    del sess
    med = statistics.median(ts)

    blk = args.block if args.arm == "nbits" else 32          # the dense arm stores the block-32 copy
    q, s, zp = copy_weights(k, n, blk, 0, args.seed)
    ref = x.astype(np.float64) @ dequant(q, s, zp, blk).T
    one = make_session(o_model, args.ep, args.threads)
    e = errors(one.run(None, feed)[0], ref)
    with open(d / f"weights_{args.t1}.bin", "rb") as fh:
        wsha = hashlib.sha256(fh.read(1 << 24)).hexdigest()

    run_bytes = copies * per_copy
    row = {"kind": "gemv", "ep": args.ep, "arm": args.arm, "k": k, "n": n,
           "block": args.block if args.arm == "nbits" else None, "t1": args.t1, "acc": acc if args.arm == "nbits" else None,
           "threads": args.threads if args.ep == "cpu" else None, "copies": copies, "copy_bytes": per_copy,
           "run_bytes": run_bytes, "reps": args.reps, "warmup": args.warmup, "load_s": round(load_s, 2),
           "median_ms": med * 1e3, "min_ms": min(ts) * 1e3, "max_ms": max(ts) * 1e3,
           "ms_per_gemv": med * 1e3 / copies, "gbps": run_bytes / med / 1e9, **e,
           "weights_sha256_first16mib": wsha, "gen": GEN, "seed": args.seed}
    print(f"gemv {args.ep} {args.arm} K={k} N={n} block={row['block']} t1={args.t1} acc={row['acc']} "
          f"threads={row['threads']}: {copies} copies, {run_bytes / 2**30:.3f} GiB/run, median {med * 1e3:.2f} ms "
          f"(min {min(ts) * 1e3:.2f}, max {max(ts) * 1e3:.2f}) -> {row['ms_per_gemv']:.4f} ms/GEMV, "
          f"{row['gbps']:.2f} GB/s; rel_l2 {e['rel_l2']:.3e}, max_abs_rel {e['max_abs_rel']:.3e}")
    print("ROW_JSON " + json.dumps(row, sort_keys=True))
    return 0


def main() -> int:
    global TARGET_BYTES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("controls", "build", "readbw", "gemv"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=SEED)
        if name != "controls":
            p.add_argument("--mib", type=int, default=TARGET_BYTES >> 20,
                           help="weight MiB per timed run (default 1024; smaller only for smoke tests)")
        if name != "build":
            p.add_argument("--ep", choices=("cpu", "dml"), required=True)
            p.add_argument("--threads", type=int, default=8, help="CPU intra-op threads (ignored on dml)")
        if name in ("readbw", "gemv"):
            p.add_argument("--warmup", type=int, default=3)
            p.add_argument("--reps", type=int, default=20)
        if name in ("build", "gemv"):
            p.add_argument("--shape", type=int, nargs=2, metavar=("K", "N"), required=True)
            p.add_argument("--block", type=int, default=32)
        if name == "readbw":
            p.add_argument("--dtype", choices=("fp32", "fp16"), required=True)
        if name == "gemv":
            p.add_argument("--arm", choices=("nbits", "dense"), required=True)
            p.add_argument("--t1", choices=("fp32", "fp16"), default="fp32")
            p.add_argument("--acc", type=int, choices=(0, 4), default=0, help="MatMulNBits accuracy_level (cpu)")
    args = ap.parse_args()
    TARGET_BYTES = getattr(args, "mib", TARGET_BYTES >> 20) << 20
    return {"controls": cmd_controls, "build": cmd_build, "readbw": cmd_readbw, "gemv": cmd_gemv}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
