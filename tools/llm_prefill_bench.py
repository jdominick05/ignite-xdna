#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prefill GEMM at Llama-2-7B's shapes on the CPU, DirectML and the NPU (LLM study stage 3).

The bar, the user's: the NPU earns a role in prefill only if it is faster than both the CPU (ONNX
Runtime) and DirectML on the Radeon 780M, or more accurate than both.

One Llama-2-7B layer's weight GEMMs, M = prompt tokens:
  S1  (M, 4096) x (4096, 4096)    four per layer: Q, K, V, O
  S2  (M, 4096) x (4096, 11008)   two: gate, up
  S3  (M, 11008) x (11008, 4096)  one: down
The arms keep their native input and output dtypes, the weights resident, and the input and output
in host memory:
  cpu  ONNX Runtime's CPU EP, 8 and 16 threads: fp32 MatMul; int8 MatMulInteger (u8 x s8, A zero
       point 128, so the int32 output equals s8 x s8)
  dml  ONNX Runtime's DirectML EP: fp16 MatMul; fp32 MatMul; int8 MatMulInteger (s8 x s8, else the
       CPU's u8 form; neither placed = UNAVAILABLE)
  npu  mlir-aie's whole_array GEMM (bf16 in, f32 out; int8 in, int32 out), compiled here by IRON and
       dispatched through raw pyxrt, with the tiles below

    python tools/llm_prefill_bench.py inputs                 # the fixed inputs, into scratch/
    python tools/llm_prefill_bench.py build                  # compile the NPU configs, no chip (ironenv)
    python tools/llm_prefill_bench.py prereg                 # the pre-registration text
    python tools/llm_prefill_bench.py npu                    # the NPU rows (ironenv; the sitting)
    python tools/llm_prefill_bench.py ort --ep {cpu,dml}     # the CPU or DirectML rows (resnet_env17)
    python tools/llm_prefill_bench.py verdict NPU CPU DML    # the mechanical verdict from the three logs
    python tools/llm_prefill_bench.py selftest               # small CPU-only checks and a synthetic verdict
"""
import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
INPUTS = ROOT / "scratch/llm/prefill"
BUILD = ROOT / "build/llm_prefill"
WA_DIR = Path.home() / "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
MM_CC = Path.home() / "mlir-aie/aie_kernels/aie2/mm.cc"

SEED = 20260923
X_STD, W_STD = 1.0, 0.02
M_MAX = 2048
MS = (512, 2048)
SHAPES = {"S1": (4096, 4096), "S2": (4096, 11008), "S3": (11008, 4096)}   # K, N
PER_LAYER = {"S1": 4, "S2": 2, "S3": 1}
LAYERS = 32
X_OF = {"S1": "x4", "S2": "x4", "S3": "x11"}
SLICES = (4096, 4096, 2816)          # S2's column slices on the NPU: two S1-shaped, one 2816 wide
WARMUP, REPS = 3, 10
REPEAT_MAX = 0.10                    # |pass 1 - pass 2| / mean, per row
MARGIN = 1.10                        # "faster" means at least 1.10x faster
ACC_TIE = 1.10                       # an arm is at least as accurate as X if its rel-L2 <= 1.10 x X's
BF16_ACC_MAX = 1e-4                  # NPU bf16 against the product of its own bf16-rounded inputs
SWITCH_MS = 0.748                    # MEASURED before (BENCHMARKS): alternating two hardware contexts
SWITCHES = 3                         # per layer: S1 -> S2 -> S3 -> the next layer's S1
CPU_THREADS = (8, 16)
ZP = 128
WAIT_MS = 120000
L1_BYTES, STACK = 65536, 3328

# (dtype, label, K, N, m, k, n, c_single_buffer). P is the best tile on record, F the one that
# already ran at this shape (or its nearest). T is S2's third column slice, 2816 wide.
NPU_TILES = [
    ("bf16", "S1-P", 4096, 4096, 32, 64, 128, 1),
    ("bf16", "S1-F", 4096, 4096, 64, 64, 64, 1),
    ("bf16", "S2u", 4096, 11008, 16, 64, 64, 0),
    ("bf16", "T-F", 4096, 2816, 64, 64, 64, 1),
    ("bf16", "S3-P", 11008, 4096, 32, 64, 128, 1),
    ("bf16", "S3-F", 11008, 4096, 64, 64, 32, 0),
    ("i8", "S1-P", 4096, 4096, 64, 128, 64, 1),
    ("i8", "S1-F", 4096, 4096, 64, 64, 64, 0),
    ("i8", "S2u", 4096, 11008, 16, 64, 64, 0),
    ("i8", "T-P", 4096, 2816, 64, 128, 64, 1),
    ("i8", "T-F", 4096, 2816, 64, 64, 64, 0),
    ("i8", "S3-P", 11008, 4096, 64, 128, 64, 1),
    ("i8", "S3-F", 11008, 4096, 64, 64, 64, 0),
]
SMOKE = [("bf16", "smoke", 512, 512, 64, 64, 64, 1), ("i8", "smoke", 512, 512, 64, 64, 64, 0)]
NPU_ROWS = {"S1": ["S1-P", "S1-F"], "S2": ["S2u", "S2s-P", "S2s-F"], "S3": ["S3-P", "S3-F"]}
S2S = {"bf16": {"S2s-P": ("S1-P", "T-F"), "S2s-F": ("S1-F", "T-F")},
       "i8": {"S2s-P": ("S1-P", "T-P"), "S2s-F": ("S1-F", "T-F")}}
ARM = {"bf16": "bf16", "i8": "int8"}
ARMS = {"cpu": ("fp32", "int8"), "dml": ("fp16", "fp32", "int8"), "npu": ("bf16", "int8")}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- the NPU tiles

def tiles():
    out = [dict(dt=dt, label=lb, M=M, K=K, N=N, m=m, k=k, n=n, cs=cs)
           for M in MS for dt, lb, K, N, m, k, n, cs in NPU_TILES]
    out += [dict(dt=dt, label=lb, M=512, K=K, N=N, m=m, k=k, n=n, cs=cs) for dt, lb, K, N, m, k, n, cs in SMOKE]
    return out


def tile(dt: str, label: str, M: int) -> dict:
    return next(t for t in tiles() if t["dt"] == dt and t["label"] == label and t["M"] == M)


def tdir(t: dict) -> Path:
    return BUILD / f"{t['dt']}_M{t['M']}_K{t['K']}_N{t['N']}_m{t['m']}k{t['k']}n{t['n']}cs{t['cs']}"


def constraints(t: dict):
    """whole_array's asserts and the three FFN DMA limits (SILICON 2.6) for one tile."""
    M, K, N, m, k, n = (t[x] for x in "MKNmkn")
    bi = 2 if t["dt"] == "bf16" else 1
    l1 = 2 * m * k * bi + 2 * k * n * bi + (1 if t["cs"] else 2) * m * n * 4 + STACK
    bad = []
    if M % (m * 4) or (M // (m * 4)) % 2:
        bad.append("M")
    if K % k or N % (n * 4):
        bad.append("K/N")
    if m % 16 or k % 8 or n % 16:
        bad.append("kernel dims")
    if l1 > L1_BYTES:
        bad.append(f"L1 {l1}")
    if m * 4 * N > 2 ** 20:
        bad.append("C step over 2^20 words")
    if N // (n * 4) > 64:
        bad.append("A repeat over 64")
    if k * n * bi // 4 > 16383:
        bad.append("B tile over 16383 words")
    return bad, l1


def build() -> int:
    sys.path.insert(0, str(WA_DIR))
    import aie.iron as iron
    import whole_array as wa
    iron.set_current_device(wa._device_for("npu", 4))
    BUILD.mkdir(parents=True, exist_ok=True)
    rc = 0
    for t in tiles():
        bad, l1 = constraints(t)
        d = tdir(t)
        if bad:
            print("REFUSED", d.name, bad)
            rc = 1
            continue
        if (d / "final.xclbin").exists() and (d / "insts.bin").exists():
            print("BUILT", d.name, "(kept)")
            continue
        d.mkdir(exist_ok=True)
        t0 = time.perf_counter()
        spec = wa.whole_array.specialize(M=t["M"], K=t["K"], N=t["N"], m=t["m"], k=t["k"], n=t["n"], n_aie_cols=4,
                                         dtype_in_str=t["dt"], dtype_out_str="f32" if t["dt"] == "bf16" else "i32",
                                         c_single_buffer=bool(t["cs"]))
        spec.compile(xclbin_path=d / "final.xclbin", inst_path=d / "insts.bin")
        print("BUILT", d.name, f"{time.perf_counter() - t0:.1f} s, L1 {l1} B")
    return rc


# ---------------------------------------------------------------- the inputs

def quant_x(x: np.ndarray):
    s = float(np.abs(x).max()) / 127.0
    return np.clip(np.rint(x.astype(np.float64) / s), -127, 127).astype(np.int8), s


def quant_w(w: np.ndarray):
    s = np.abs(w.astype(np.float64)).max(axis=0) / 127.0
    return np.clip(np.rint(w.astype(np.float64) / s), -127, 127).astype(np.int8), s


def inputs() -> int:
    if (INPUTS / "manifest.json").exists():
        manifest = json.loads((INPUTS / "manifest.json").read_text(encoding="utf-8"))
        if all((INPUTS / f"{n}.npy").exists() and sha(INPUTS / f"{n}.npy") == h for n, h in manifest.items()):
            print("INPUTS_MANIFEST_SHA256", sha(INPUTS / "manifest.json"), "(kept, every file verified)")
            return 0
    INPUTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    arrs = {"x4": rng.standard_normal((M_MAX, 4096), dtype=np.float32) * np.float32(X_STD),
            "x11": rng.standard_normal((M_MAX, 11008), dtype=np.float32) * np.float32(X_STD)}
    for s, (K, N) in SHAPES.items():
        arrs[f"w_{s}"] = rng.standard_normal((K, N), dtype=np.float32) * np.float32(W_STD)
    for x in ("x4", "x11"):
        arrs[f"{x}_q"], sx = quant_x(arrs[x])
        arrs[f"sx_{x}"] = np.array([sx], dtype=np.float64)
    for s in SHAPES:
        arrs[f"w_{s}_q"], arrs[f"sw_{s}"] = quant_w(arrs[f"w_{s}"])
    manifest = {}
    for name, a in arrs.items():
        p = INPUTS / f"{name}.npy"
        np.save(p, a)
        manifest[name] = sha(p)
    (INPUTS / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    print("INPUTS_MANIFEST_SHA256", sha(INPUTS / "manifest.json"))
    return 0


def load(names) -> dict:
    manifest = json.loads((INPUTS / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for n in names:
        p = INPUTS / f"{n}.npy"
        if sha(p) != manifest[n]:
            raise SystemExit(f"input {n} does not match the manifest")
        out[n] = np.load(p)
    print("INPUTS_MANIFEST_SHA256", sha(INPUTS / "manifest.json"), flush=True)
    return out


def reference(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x.astype(np.float64) @ w.astype(np.float64)


def dequant(y: np.ndarray, sx: float, sw: np.ndarray) -> np.ndarray:
    return y.astype(np.float64) * (sx * sw.astype(np.float64))[None, :]


def errors(y: np.ndarray, ref: np.ndarray) -> dict:
    y = y.astype(np.float64).reshape(-1)
    ref = ref.reshape(-1)
    return {"rel_l2": float(np.linalg.norm(y - ref) / np.linalg.norm(ref)),
            "max_abs_rel": float(np.max(np.abs(y - ref)) / np.max(np.abs(ref))),
            "finite": bool(np.all(np.isfinite(y)))}


def host_gate():
    host = subprocess.run(["powershell", "-NoProfile", "-File", str(ROOT / "tools/host_load.ps1")],
                          capture_output=True, text=True)
    summary = next(s for s in host.stdout.splitlines() if s.startswith("HOST_LOAD busy_cores="))
    peers = [float(v) for v in re.findall(r"HOST_LOAD_PEER ([\d.]+)", host.stdout)]
    print(summary, "peer_busy_cores", peers, flush=True)
    assert float(re.search(r"busy_cores=([\d.]+)", summary)[1]) < 2, "host busy"
    assert max(peers, default=0) < .5, "a heavy peer holds the CPU"


def stats(ts) -> dict:
    ms = [round(t * 1e3, 4) for t in ts]
    return {"median_ms": round(statistics.median(ms), 4), "ms": ms}


# ---------------------------------------------------------------- CPU and DirectML (ONNX Runtime)

def ort_model(kind: str, w: np.ndarray, form: str = "") -> bytes:
    from onnx import TensorProto, helper, numpy_helper
    K, N = w.shape
    inits = [numpy_helper.from_array(w, "w")]
    if kind == "int8":
        ins = ["x", "w"]
        et = TensorProto.INT8
        if form == "u8zp":
            et = TensorProto.UINT8
            inits.append(numpy_helper.from_array(np.array(ZP, dtype=np.uint8), "azp"))
            ins.append("azp")
        node, yt = helper.make_node("MatMulInteger", ins, ["y"]), TensorProto.INT32
    else:
        et = yt = TensorProto.FLOAT if kind == "fp32" else TensorProto.FLOAT16
        node = helper.make_node("MatMul", ["x", "w"], ["y"])
    g = helper.make_graph([node], "prefill_gemm", [helper.make_tensor_value_info("x", et, ["M", K])],
                          [helper.make_tensor_value_info("y", yt, ["M", N])], inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=9).SerializeToString()


def make_session(model: bytes, ep: str, threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if ep == "cpu":
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
    else:
        so.enable_mem_pattern = False
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers = [("DmlExecutionProvider", {"device_id": 0})]
    return ort.InferenceSession(model, so, providers=providers)


def ort_weights(kind: str, s: str, data: dict, form: str) -> np.ndarray:
    if kind == "int8":
        return data[f"w_{s}_q"]
    return data[f"w_{s}"].astype(np.float16 if kind == "fp16" else np.float32)


def ort_rows(ep: str, ms=MS, shapes=None, reps=REPS, warmup=WARMUP, threads_set=CPU_THREADS) -> int:
    import onnx
    import onnxruntime as ort
    from concurrent_read_bw import gpu_engines
    shapes = shapes or SHAPES
    print("HEADER " + json.dumps({"python": platform.python_version(), "onnxruntime": ort.__version__,
                                  "numpy": np.__version__, "onnx": onnx.__version__, "ep": ep,
                                  "env": os.environ.get("CONDA_DEFAULT_ENV", "?"), "host": platform.node()},
                                 sort_keys=True), flush=True)
    host_gate()
    if ep == "dml":
        gpu_engines()
    names = ["x4", "x11", "x4_q", "x11_q", "sx_x4", "sx_x11"] + [f"w_{s}{t}" for s in SHAPES for t in ("", "_q")] \
        + [f"sw_{s}" for s in SHAPES]
    data = load(names)
    refs = {s: reference(data[X_OF[s]], data[f"w_{s}"]) for s in shapes}
    feeds = {}
    for x in ("x4", "x11"):
        feeds[("fp32", "", x)] = data[x]
        feeds[("fp16", "", x)] = data[x].astype(np.float16)
        feeds[("int8", "s8", x)] = data[f"{x}_q"]
        feeds[("int8", "u8zp", x)] = (data[f"{x}_q"].astype(np.int16) + ZP).astype(np.uint8)
    arms = [(a, th) for a in ARMS["cpu"] for th in threads_set] if ep == "cpu" else [(a, 0) for a in ARMS["dml"]]
    sessions = {}
    for arm, th in arms:
        for s in shapes:
            forms = [""] if arm != "int8" else (["u8zp"] if ep == "cpu" else ["s8", "u8zp"])
            for form in forms:
                try:
                    sess = make_session(ort_model(arm, ort_weights(arm, s, data, form), form), ep, th)
                except Exception as e:                            # DirectML may refuse an int8 form
                    print("SESSION_REFUSED " + json.dumps({"ep": ep, "arm": arm, "shape": s, "form": form,
                                                           "error": str(e)[:300]}), flush=True)
                    continue
                sessions[(arm, th, s)] = (sess, form)
                print("SESSION " + json.dumps({"ep": ep, "arm": arm, "threads": th, "shape": s, "form": form,
                                               "providers": sess.get_providers()}), flush=True)
                break
            else:
                print("ARM_UNAVAILABLE " + json.dumps({"ep": ep, "arm": arm, "shape": s}), flush=True)
    rows = [(arm, th, s, M) for (arm, th, s) in sessions for M in ms]
    for p, order in ((1, rows), (2, rows[::-1])):
        for arm, th, s, M in order:
            sess, form = sessions[(arm, th, s)]
            feed = {"x": feeds[(arm, form, X_OF[s])][:M]}
            for _ in range(warmup):
                sess.run(None, feed)
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                y = sess.run(None, feed)[0]
                ts.append(time.perf_counter() - t0)
            K, N = SHAPES[s]
            row = {"chip": ep, "arm": arm, "threads": th, "form": form, "config": "", "shape": s, "M": M, "K": K,
                   "N": N, "pass": p, **stats(ts)}
            if arm == "int8":
                row["int32_sha"] = hashlib.sha256(np.ascontiguousarray(y, dtype=np.int32).tobytes()).hexdigest()
                y = dequant(y, float(data[f"sx_{X_OF[s]}"][0]), data[f"sw_{s}"])
            row.update(errors(y, refs[s][:M]))
            print("ROW_JSON " + json.dumps(row), flush=True)
            time.sleep(0.5)                                       # let the previous pool's spinning threads park
    if ep == "dml":
        gpu_engines()
    return 0


# ---------------------------------------------------------------- the NPU (raw pyxrt)

class NpuGemm:
    """One compiled whole_array GEMM in its own hardware context: an A buffer, and one B and C
    buffer and one run per weight given."""

    def __init__(self, h, reg: dict, d: Path, a: np.ndarray, bs, c_bytes: int):
        xrt = h.pyxrt
        self.to_dev = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.from_dev = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.done = xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
        if d not in reg:                                          # register each xclbin once per process
            x = xrt.xclbin(str(d / "final.xclbin"))
            reg[d] = (x, h.dev.register_xclbin(x))
        self.xclbin, uuid = reg[d]
        self.ctx = xrt.hw_context(h.dev, uuid)
        self.kernel = xrt.kernel(self.ctx, "MLIR_AIE")
        insts = (d / "insts.bin").read_bytes()
        self.instr = xrt.bo(h.dev, len(insts), xrt.bo.cacheable, self.kernel.group_id(1))
        self.instr.write(insts, 0)
        self.instr.sync(self.to_dev)
        self.a = xrt.bo(h.dev, a.nbytes, xrt.bo.host_only, self.kernel.group_id(3))
        self.a.write(a, 0)
        self.c_bytes = c_bytes
        self.b, self.c, self.runs = [], [], []
        for w in bs:
            b = xrt.bo(h.dev, w.nbytes, xrt.bo.host_only, self.kernel.group_id(4))
            b.write(w, 0)
            b.sync(self.to_dev)
            c = xrt.bo(h.dev, c_bytes, xrt.bo.host_only, self.kernel.group_id(5))
            r = xrt.run(self.kernel)
            for i, v in enumerate((3, self.instr, len(insts), self.a, b, c)):
                r.set_arg(i, v)
            self.b.append(b)
            self.c.append(c)
            self.runs.append(r)

    def step(self, i: int, sync_a: bool):
        if sync_a:
            self.a.sync(self.to_dev)
        self.runs[i].start()
        if self.runs[i].wait(WAIT_MS) != self.done:
            raise RuntimeError("npu dispatch did not complete")
        self.c[i].sync(self.from_dev)

    def read(self, i: int, dtype, cols: int) -> np.ndarray:
        return np.frombuffer(self.c[i].read(self.c_bytes, 0), dtype=dtype).reshape(-1, cols).copy()

    def close(self):
        self.runs = self.c = self.b = []
        self.a = self.instr = self.kernel = self.ctx = self.xclbin = None


def npu() -> int:
    import ml_dtypes
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    bf16 = ml_dtypes.bfloat16
    print("HEADER " + json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                                  "ml_dtypes": ml_dtypes.__version__, "host": platform.node(),
                                  "whole_array_sha256": sha(WA_DIR / "whole_array.py"),
                                  "mm_cc_sha256": sha(MM_CC)}, sort_keys=True), flush=True)
    host_gate()
    for t in tiles():
        print("ARTIFACT", tdir(t).name, "insts_sha256", sha(tdir(t) / "insts.bin"), flush=True)
    names = ["x4", "x11", "x4_q", "x11_q", "sx_x4", "sx_x11"] + [f"w_{s}{t}" for s in SHAPES for t in ("", "_q")] \
        + [f"sw_{s}" for s in SHAPES]
    data = load(names)
    xin = {"bf16": {x: data[x].astype(bf16) for x in ("x4", "x11")}, "i8": {x: data[f"{x}_q"] for x in ("x4", "x11")}}
    win = {"bf16": {s: data[f"w_{s}"].astype(bf16) for s in SHAPES}, "i8": {s: data[f"w_{s}_q"] for s in SHAPES}}
    t0 = time.perf_counter()
    ref = {s: reference(data[X_OF[s]], data[f"w_{s}"]) for s in SHAPES}
    ref_own = {(dt, s): reference(xin[dt][X_OF[s]].astype(np.float32), win[dt][s].astype(np.float32))
               for dt in ("bf16", "i8") for s in SHAPES}      # int8: exact (K * 127^2 < 2^53)
    print(f"REFERENCES {time.perf_counter() - t0:.1f} s", flush=True)
    raw = lambda a: np.ascontiguousarray(a).view(np.uint16) if a.dtype == bf16 else np.ascontiguousarray(a)
    out_dt = {"bf16": np.float32, "i8": np.int32}

    def check(dt, s, M, y):
        own = ref_own[(dt, s)][:M]
        if dt == "bf16":
            acc = errors(y, own)["rel_l2"]
            ok, extra = acc <= BF16_ACC_MAX, {"acc_rel_l2": acc}
            e = errors(y, ref[s][:M])
        else:
            ok, extra = bool(np.array_equal(y.astype(np.float64), own)), {}
            extra["int32_sha"] = hashlib.sha256(np.ascontiguousarray(y, dtype=np.int32).tobytes()).hexdigest()
            e = errors(dequant(y, float(data[f"sx_{X_OF[s]}"][0]), data[f"sw_{s}"]), ref[s][:M])
        return {"ok": bool(ok and e["finite"]), **extra, **e}

    reg = {}
    with XrtSiliconHarness(0) as h:
        # the smoke: a 512^3 GEMM per dtype through this raw-pyxrt path, before any timed row, each
        # twice with a fresh hardware context (as pass 2 will open every tile's context again)
        for dt in ("bf16", "i8"):
            t = tile(dt, "smoke", 512)
            a, w = xin[dt]["x4"][:512, :512], win[dt]["S1"][:512, :512]
            own = reference(a.astype(np.float32), w.astype(np.float32))
            oks = []
            for _ in range(2):
                g = NpuGemm(h, reg, tdir(t), raw(a), [raw(w)], 512 * 512 * 4)
                try:
                    g.step(0, True)
                    y = g.read(0, out_dt[dt], 512)
                finally:
                    g.close()
                oks.append(errors(y, own)["rel_l2"] <= BF16_ACC_MAX if dt == "bf16"
                           else bool(np.array_equal(y.astype(np.float64), own)))
            ok = all(oks)
            print("SMOKE_JSON " + json.dumps({"dtype": dt, "ok": bool(ok), "runs": [bool(o) for o in oks],
                                              "rel_l2_own": errors(y, own)["rel_l2"]}), flush=True)
            if not ok:
                print("SMOKE_FAIL", dt, flush=True)
                return 3
        rows = [(dt, s, lb, M) for dt in ("bf16", "i8") for s in SHAPES for lb in NPU_ROWS[s] for M in MS]
        for p, order in ((1, rows), (2, rows[::-1])):
            for dt, s, lb, M in order:
                K, N = SHAPES[s]
                a = raw(xin[dt][X_OF[s]][:M])
                w = win[dt][s]
                gs, row = [], {"chip": "npu", "arm": ARM[dt], "threads": 0, "form": "", "config": lb, "shape": s,
                               "M": M, "K": K, "N": N, "pass": p}
                try:
                    if lb.startswith("S2s"):
                        t1, t3 = (tile(dt, x, M) for x in S2S[dt][lb])
                        cut = np.cumsum((0,) + SLICES)
                        parts = [raw(w[:, cut[i]:cut[i + 1]]) for i in range(3)]
                        gs = [NpuGemm(h, reg, tdir(t1), a, parts[:2], M * SLICES[0] * 4),
                              NpuGemm(h, reg, tdir(t3), a, parts[2:], M * SLICES[2] * 4)]
                        steps = [(gs[0], 0, True), (gs[0], 1, False), (gs[1], 0, True)]
                        row["tiles"] = [tdir(t1).name, tdir(t3).name]
                    else:
                        t = tile(dt, lb, M)
                        gs = [NpuGemm(h, reg, tdir(t), a, [raw(w)], M * N * 4)]
                        steps = [(gs[0], 0, True)]
                        row["tiles"] = [tdir(t).name]
                    for _ in range(WARMUP):
                        for g, i, sync_a in steps:
                            g.step(i, sync_a)
                    ts = []
                    for _ in range(REPS):
                        t0 = time.perf_counter()
                        for g, i, sync_a in steps:
                            g.step(i, sync_a)
                        ts.append(time.perf_counter() - t0)
                    outs = [g.read(i, out_dt[dt], g.c_bytes // 4 // M) for g, i, _ in steps]
                    if len(outs) > 1:
                        cts = []
                        for _ in range(REPS):
                            t0 = time.perf_counter()
                            y = np.concatenate(outs, axis=1)
                            cts.append(time.perf_counter() - t0)
                        row["concat_ms"] = stats(cts)["median_ms"]
                    else:
                        y = outs[0]
                    row.update(stats(ts))
                    row.update(check(dt, s, M, y))               # wrong numbers void the row
                except Exception as e:                            # a raise or a timeout stops the sitting
                    print("ROW_ERROR " + json.dumps({**row, "error": f"{type(e).__name__}: {str(e)[:300]}"}),
                          flush=True)
                    raise
                finally:
                    for g in gs:
                        g.close()
                print("ROW_JSON " + json.dumps(row), flush=True)
        reg.clear()
    return 0


# ---------------------------------------------------------------- the verdict

def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {"rows": [], "unavailable": [], "smoke": [], "exit": None, "smoke_fail": "SMOKE_FAIL" in text}
    for line in text.splitlines():
        if line.startswith("ROW_JSON "):
            out["rows"].append(json.loads(line[9:]))
        elif line.startswith("ARM_UNAVAILABLE "):
            out["unavailable"].append(json.loads(line[16:]))
        elif line.startswith("SMOKE_JSON "):
            out["smoke"].append(json.loads(line[11:]))
        elif line.startswith("EXIT_CODE:"):
            out["exit"] = int(line.split(":")[1])
    return out


def flops(K, N, M) -> float:
    return 2.0 * M * K * N


def verdict(paths) -> int:
    return evaluate(paths)[0]


def evaluate(paths):
    """Print the verdict; return (exit code, the M values kept)."""
    logs = {c: parse(Path(p)) for c, p in zip(("npu", "cpu", "dml"), paths)}
    problems = []
    for c, lg in logs.items():
        if lg["exit"] != 0:
            problems.append(f"{c} log: EXIT_CODE {lg['exit']}")
    if logs["npu"]["smoke_fail"] or sorted(s["dtype"] for s in logs["npu"]["smoke"] if s["ok"]) != ["bf16", "i8"]:
        problems.append("the NPU smoke did not pass for both dtypes")
    rows = {}
    for lg in logs.values():
        for r in lg["rows"]:
            rows.setdefault((r["chip"], r["arm"], r["threads"], r["config"], r["shape"], r["M"]), {})[r["pass"]] = r
    vals = {}
    for key, ps in rows.items():
        if set(ps) != {1, 2}:
            problems.append(f"{key}: passes {sorted(ps)}")
            continue
        a, b = ps[1], ps[2]
        if key[0] == "npu" and not (a.get("ok") and b.get("ok")):
            print("VOID", key, a.get("error") or b.get("error") or "failed its correctness check")
            continue
        v = (a["median_ms"] + b["median_ms"]) / 2
        spread = abs(a["median_ms"] - b["median_ms"]) / v
        if spread > REPEAT_MAX:
            problems.append(f"{key}: passes {a['median_ms']} / {b['median_ms']} ms, {100 * spread:.1f}% apart")
        if not (a["finite"] and b["finite"]):
            problems.append(f"{key}: non-finite output")
        vals[key] = {"ms": v, "spread": spread, "rel_l2": b["rel_l2"], "sha": b.get("int32_sha"),
                     "concat_ms": b.get("concat_ms", 0.0)}
    unavailable = {(u["arm"], u["shape"]) for u in logs["dml"]["unavailable"]}
    expected = [("cpu", a, th, "", s, M) for a in ARMS["cpu"] for th in CPU_THREADS for s in SHAPES for M in MS]
    expected += [("dml", a, 0, "", s, M) for a in ARMS["dml"] for s in SHAPES for M in MS
                 if not (a == "int8" and (a, s) in unavailable)]
    expected += [("npu", ARM[dt], 0, lb, s, M) for dt in ("bf16", "i8") for s in SHAPES for lb in NPU_ROWS[s] for M in MS]
    for key in expected:
        if key not in rows:
            problems.append(f"{key}: missing")
    for (a, s) in unavailable:
        if a != "int8":
            problems.append(f"dml {a} {s}: unavailable")
    # the int8 control: the shared quantization should give every exact int8 arm the same int32 output
    for s in SHAPES:
        for M in MS:
            shas = {k: v["sha"] for k, v in vals.items() if k[1] == "int8" and k[4] == s and k[5] == M}
            agree = len(set(shas.values())) == 1
            print(f"INT8_CONTROL {s} M={M}: {len(shas)} int8 rows, int32 outputs "
                  + ("identical" if agree else "DIFFER: " + ", ".join(f"{k[0]}/{k[2] or k[3]}" for k in shas)))

    print(f"per layer = {' + '.join(f'{PER_LAYER[s]} x {s}' for s in SHAPES)}; ms are the mean of two pass medians")
    verdicts = {}
    for M in MS:
        arms = {}
        for chip in ("cpu", "dml", "npu"):
            for arm in ARMS[chip]:
                best = None
                for th in (CPU_THREADS if chip == "cpu" else (0,)):
                    per, cfg = {}, {}
                    for s in SHAPES:
                        if chip == "npu":
                            cands = [(vals[k]["ms"], lb) for lb in NPU_ROWS[s]
                                     for k in [(chip, arm, 0, lb, s, M)] if k in vals]
                            if cands:
                                per[s], cfg[s] = min(cands)
                        elif (chip, arm, th, "", s, M) in vals:
                            per[s], cfg[s] = vals[(chip, arm, th, "", s, M)]["ms"], ""
                    if len(per) < len(SHAPES):
                        continue
                    T = sum(PER_LAYER[s] * per[s] for s in SHAPES)
                    err = max(vals[(chip, arm, th, cfg[s], s, M)]["rel_l2"] for s in SHAPES)
                    concat = sum(PER_LAYER[s] * vals[(chip, arm, th, cfg[s], s, M)]["concat_ms"] for s in SHAPES)
                    if best is None or T < best["T"]:
                        best = {"T": T, "err": err, "per": per, "cfg": cfg, "threads": th, "concat": concat}
                if best:
                    arms[(chip, arm)] = best
                elif chip == "npu":
                    print(f"M={M} npu {arm}: FAILED, no passing config at some shape; it beats nothing")
        verdicts[M] = decide(M, arms)
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(" ", p)
        print("VERDICT INCOMPLETE")
        return 2, []
    keep = [M for M, v in verdicts.items() if v]
    print("\nVERDICT " + (f"KEEP: the NPU earns a prefill role at M = {', '.join(map(str, keep))}" if keep
                          else "KILL: at neither M does an NPU arm beat both the CPU and DirectML"))
    return 0, keep


def decide(M: int, arms: dict):
    print(f"\nM = {M}")
    print(f"  {'chip arm':16s} {'S1 ms':>9s} {'S2 ms':>9s} {'S3 ms':>9s} {'layer ms':>10s} {'32 layers s':>12s} "
          f"{'TFLOPS':>7s} {'rel-L2':>9s}  config")
    layer_flops = sum(PER_LAYER[s] * flops(*SHAPES[s], M) for s in SHAPES)
    for (chip, arm), a in sorted(arms.items(), key=lambda kv: kv[1]["T"]):
        cfg = " ".join(f"{s}:{a['cfg'][s]}" for s in SHAPES if a["cfg"][s])
        if chip == "cpu":
            cfg = f"{a['threads']} threads"
        print(f"  {chip + ' ' + arm:16s} " + " ".join(f"{a['per'][s]:9.2f}" for s in SHAPES)
              + f" {a['T']:10.2f} {a['T'] * LAYERS / 1e3:12.3f} {layer_flops / a['T'] / 1e9:7.2f} {a['err']:9.2e}  {cfg}")
    won = []
    for arm in ARMS["npu"]:
        X = arms.get(("npu", arm))
        if not X:
            continue
        for label, T in (("", X["T"]), ("with context switches and S2's concatenation added",
                                         X["T"] + SWITCHES * SWITCH_MS + X["concat"])):
            beats = {}
            for chip in ("cpu", "dml"):
                rivals = {k: v for k, v in arms.items() if k[0] == chip and v["err"] <= ACC_TIE * X["err"]}
                slow = {k[1]: v["T"] / T for k, v in rivals.items()}
                beats[chip] = all(T * MARGIN <= v["T"] for v in rivals.values())
                how = ("more accurate than every arm" if not rivals else
                       ", ".join(f"{r} {x:.2f}x" for r, x in slow.items()) + " (rival time / NPU time)")
                print(f"  npu {arm}{' ' + label if label else ''} vs {chip}: "
                      f"{'BEATS' if beats[chip] else 'does not beat'} -- {how}")
            if not label:
                if not all(beats.values()):
                    break
            elif all(beats.values()):
                won.append(arm)
    print(f"  M = {M}: " + (f"KEEP, the NPU's {' and '.join(won)} arm beats both chips" if won
                            else "no NPU arm beats both chips"))
    return won


# ---------------------------------------------------------------- the pre-registration

PREREG = f"""\
LLM study stage 3: prefill GEMM at Llama-2-7B's shapes on the CPU, DirectML and the NPU,
pre-registered before any sitting

Question (the user's third item; the gate's go after the R5 sync test)
  Does the NPU earn a prefill role? The bar is the user's: an NPU arm must be faster than both the
  CPU (ONNX Runtime) and DirectML on the Radeon 780M, or more accurate than both.

Workload: one Llama-2-7B layer's weight GEMMs, M = prompt tokens, M in {MS}
  S1  (M, 4096) x (4096, 4096)    4 per layer (Q, K, V, O)
  S2  (M, 4096) x (4096, 11008)   2 per layer (gate, up)
  S3  (M, 11008) x (11008, 4096)  1 per layer (down)
  An arm's layer time = 4 t(S1) + 2 t(S2) + t(S3) (DERIVED from MEASURED rows); 32 layers are
  printed beside it. Attention, norms, RoPE and the LM head are outside this stage.

Inputs (tools/llm_prefill_bench.py inputs, seed {SEED}; SHA-256 pinned below)
  X ~ N(0, {X_STD}) fp32, [2048, 4096] and [2048, 11008]; W ~ N(0, {W_STD}) fp32 per shape. M = 512
  uses the first 512 rows.
  Each arm rounds the same fp32 X and W to its own dtype: fp16, bf16 (round to nearest even), or
  int8 with one shared quantization: X symmetric per tensor, W symmetric per output column, both
  to [-127, 127]. Every exact int8 arm therefore returns the same int32 output, and ties on error.

Arms (native dtypes in and out, weights resident, input and output in host memory)
  cpu  ONNX Runtime 1.23.3 CPU EP, resnet_env17, 8 and 16 threads (the faster counts, per arm):
       fp32 MatMul; int8 MatMulInteger, u8 x s8 with A zero point {ZP} (MLAS's VNNI form; the int32 output
       equals s8 x s8). s8 x s8 and u8 x s8 both build and are exact on this CPU (checked, no timing).
  dml  ONNX Runtime DirectML EP, device 0 (the 780M, Phase 1's adapter log), CPU fallback disabled:
       fp16 MatMul; fp32 MatMul; int8 MatMulInteger as s8 x s8, else u8 x s8 with zero point {ZP}. If
       DirectML places neither, its int8 arm is UNAVAILABLE, which does not void the verdict.
  npu  mlir-aie v1.4.2's whole_array (live file pinned below; it carries the two local patches in
       kernels/ and one unrecorded change to its own integer input generator, which this stage
       does not use), 4 columns x 4 rows, compiled by IRON's compile-only path here and dispatched
       through raw pyxrt with IRON's XRT ABI: kernel(3, instr, instr bytes, A, B, C).
       bf16 in, f32 out; int8 in, int32 out. Tiles (M = 512 and 2048 each; P = the best tile on
       record, F = the one that already ran at the shape; the faster passing config counts):
{{TILES}}
       S2 runs three ways: unsliced (S2u, forced to m = 16 by the 2^20-word C step), and as column
       slices 4096 + 4096 + 2816 (S2s-P, S2s-F: two dispatches on the S1 tile, one on the T tile,
       two hardware contexts held, no padded work). Its three outputs stay in their own buffers;
       their host concatenation is timed separately (concat_ms) and added only to a KEEP check.

Timing (all chips)
  CPU and DirectML: one session.run per GEMM, numpy in and out (DirectML uploads x and downloads y).
  NPU: sync A to the device, start, wait, sync C from the device (A and B written beforehand; B
  synced once). Prior NPU GEMM figures here (1.33x, 1.10x, 1.13x and the int8 table) used IRON's
  inner bracket, without the input sync, against torch baselines. They are not comparable.
  {WARMUP} warmup runs, then {REPS} timed; the row is the median. Two passes, the second in reverse row
  order; a row's value is the mean of its two pass medians.

Accuracy
  rel-L2 of each arm's output (int8 dequantized) against the float64 product of the fp32 X and W; an
  arm's error is its worst of the three shapes at that M.
  Controls: NPU bf16 against the float64 product of its own bf16-rounded inputs must be <= {BF16_ACC_MAX:g}
  (fp32 accumulation); NPU int8 must equal the int8 inputs' product exactly; a failing NPU row is void.
  Every int8 row prints the SHA-256 of its int32 output, and the verdict prints whether they agree
  across chips. That is a control, not a void: the rule uses each arm's own measured rel-L2.

Rule (kill or keep, per M)
  An NPU arm X beats a chip if it is at least {MARGIN:.2f}x faster (layer time) than every arm of that chip
  whose rel-L2 is <= {ACC_TIE:.2f} x X's. If that chip has no such arm, X is more accurate than all of it.
  KEEP at M if some NPU arm beats both the CPU and DirectML, and still does with S2's concatenation
  and {SWITCHES} context switches per layer added. {SWITCHES} is the most a layer needs beyond what its rows
  already hold (S1 -> S2 -> S3 -> the next layer's S1; the sliced S2 rows alternate their two
  contexts inside their own timed runs). Each switch costs {SWITCH_MS} ms, the cross-context alternation
  penalty BENCHMARKS measured with a passthrough on the driver bench; it is not remeasured on GEMMs.
  The verdict: KEEP at the M values where that holds; KILL if it holds at neither.
  An NPU arm with no passing config at some shape beats nothing at that M.

INCOMPLETE (no verdict)
  A sitting log without EXIT_CODE 0, which includes any NPU dispatch that raises or times out (it
  stops the sitting; only a tile that completes with wrong numbers voids just its row); the NPU
  smoke (512^3 per dtype, each twice with a fresh hardware context, before any timed row) not
  passing; a missing row or pass; a row whose pass medians differ by more than {100 * REPEAT_MAX:.0f}% of their
  mean; a non-finite output; DirectML fp16 or fp32 unavailable.

Written predictions (stated expectations; the rule decides, these do not)
  Q1 CPU fp32, S1 at M = 2048: 0.5-1.0 TFLOPS. Prior: torch fp32 0.71 at this shape (MEASURED,
     results/aie/int8_matmul_sweep_npu.log).
  Q2 CPU int8, S1 at M = 2048: 1.4-2.4 TOPS. Prior: ORT MatMulInteger u8 x s8 1.70 (same log).
  Q3 DirectML fp16, S1 at M = 2048: 2.5-9 TFLOPS. No DirectML GEMM above M = 1 has run here. The
     780M's fp16 peak is 8.9 TFLOPS with one of RDNA 3's dual issue or packed fp16, and 17.8 if both
     apply, as AMD's RDNA 3 peak figures count them; which applies here is not established (DERIVED
     from SPEC: 768 shaders at 2.9 GHz).
  Q4 DirectML fp32, S1 at M = 2048: 1.5-5 TFLOPS.
  Q5 NPU bf16 at M = 2048: S1 2.2-2.8 TFLOPS (IRON's inner bracket read 2.70 at this shape and tile);
     S2 0.8-1.0 unsliced (0.91 before), 2.0-2.7 sliced; S3 1.7-2.7.
  Q6 NPU int8, S1 at M = 2048: 3.0-4.9 TOPS (3.26 at the F tile, 4.85 at the P tile but only at
     2048^3, both through IRON's inner bracket).
  Q7 rel-L2: CPU and DirectML fp32 under 1e-6; DirectML fp16 2e-4 to 1.5e-3; NPU bf16 1.5e-3 to 5e-3;
     every int8 arm the same, 5e-3 to 3e-2. So the NPU cannot win on accuracy.
  Q8 KILL at both M. DirectML fp16 decides: it is at least as accurate as both NPU arms and at least
     {MARGIN:.2f}x faster than each. If it lands under about 3.5 TFLOPS, NPU int8 could beat it; that is the
     prediction most likely to be wrong.
  Q9 No chip runs faster per FLOP at M = 512 than at M = 2048.

Unverified by design: attention, norms, RoPE and the LM head; the activations between GEMMs (f32 to
bf16, int8 quantization and dequantization are outside every bracket, for every arm); DirectML IO
binding; CPU stacks other than ONNX Runtime (torch ran faster than ORT in int8 before); tiles outside
the menu above; the NPU's power modes; running prefill beside decode.
"""


def prereg() -> int:
    lines = []
    for t in tiles():
        bad, l1 = constraints(t)
        d = tdir(t)
        if bad or not (d / "insts.bin").exists():
            raise SystemExit(f"{d.name}: {'refused ' + str(bad) if bad else 'not built'}")
        lines.append(f"         {t['dt']:4s} {t['label']:5s} M={t['M']:<4d} K={t['K']:<5d} N={t['N']:<5d} "
                     f"m={t['m']:<2d} k={t['k']:<3d} n={t['n']:<3d} c_single={t['cs']}  L1 {l1:5d} B  "
                     f"insts {sha(d / 'insts.bin')[:16]}  xclbin {sha(d / 'final.xclbin')[:16]}")
    print(PREREG.replace("{TILES}", "\n".join(lines)))
    print("PINS inputs manifest", sha(INPUTS / "manifest.json"))
    print("PINS whole_array.py", sha(WA_DIR / "whole_array.py"), "mm.cc", sha(MM_CC))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.split("\n")
    print(f"git HEAD {head} (+{len([d for d in dirty if d.strip()])} uncommitted paths, this stage's own files)")
    print("PREREG_JSON " + json.dumps({"ms": MS, "shapes": SHAPES, "per_layer": PER_LAYER, "margin": MARGIN,
                                       "acc_tie": ACC_TIE, "repeat_max": REPEAT_MAX, "bf16_acc_max": BF16_ACC_MAX,
                                       "switch_ms": SWITCH_MS, "switches": SWITCHES, "warmup": WARMUP, "reps": REPS,
                                       "cpu_threads": CPU_THREADS, "seed": SEED, "git_head": head}))
    return 0


# ---------------------------------------------------------------- selftest (no chip, no timing)

def selftest() -> int:
    import onnxruntime as ort
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 64), dtype=np.float32)
    w = rng.standard_normal((64, 48), dtype=np.float32) * np.float32(W_STD)
    xq, sx = quant_x(x)
    wq, sw = quant_w(w)
    exact = xq.astype(np.int64) @ wq.astype(np.int64)
    for form, feed in (("s8", xq), ("u8zp", (xq.astype(np.int16) + ZP).astype(np.uint8))):
        y = make_session(ort_model("int8", wq, form), "cpu", 1).run(None, {"x": feed})[0]
        print("selftest cpu int8", form, "exact", bool(np.array_equal(y, exact)))
        assert np.array_equal(y, exact)
    y = make_session(ort_model("fp32", w), "cpu", 1).run(None, {"x": x})[0]
    e = errors(y, reference(x, w))["rel_l2"]
    print("selftest cpu fp32 rel_l2", f"{e:.2e}")
    assert e < 1e-6
    e8 = errors(dequant(exact, sx, sw), reference(x, w))["rel_l2"]
    print("selftest int8 quantization rel_l2", f"{e8:.2e}", "onnxruntime", ort.__version__)
    for t in tiles():
        bad, _ = constraints(t)
        assert not bad, (tdir(t).name, bad)
    print("selftest constraints: all", len(tiles()), "tiles pass")
    # a synthetic verdict: one where DirectML fp16 wins (KILL), one where the NPU int8 wins (KEEP)
    tmp = ROOT / "scratch/llm/prefill_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    for name, dml16, want in (("kill", 5.0, (0, [])), ("keep", 1.0, (0, list(MS)))):
        tf = {"cpu fp32": .8, "cpu int8": 1.7, "dml fp16": dml16, "dml fp32": 1.0, "dml int8": .9,
              "npu bf16": 2.5, "npu int8": 3.5}
        err = {"fp32": 1e-7, "fp16": 5e-4, "bf16": 3e-3, "int8": 1.5e-2}
        logs = {}
        for chip in ("npu", "cpu", "dml"):
            out = []
            if chip == "npu":
                out += [f"SMOKE_JSON {json.dumps({'dtype': d, 'ok': True})}" for d in ("bf16", "i8")]
            for arm in ARMS[chip]:
                for th in (CPU_THREADS if chip == "cpu" else (0,)):
                    for s, (K, N) in SHAPES.items():
                        for M in MS:
                            for lb in (NPU_ROWS[s] if chip == "npu" else [""]):
                                ms = flops(K, N, M) / (tf[f"{chip} {arm}"] * 1e12) * 1e3 * (1.3 if lb == "S2u" else 1)
                                for p in (1, 2):
                                    r = {"chip": chip, "arm": arm, "threads": th, "form": "", "config": lb, "shape": s,
                                         "M": M, "K": K, "N": N, "pass": p, "median_ms": ms * (1 + .01 * p), "ms": [],
                                         "rel_l2": err[arm], "max_abs_rel": 0, "finite": True, "ok": True,
                                         "int32_sha": f"{s}{M}" if arm == "int8" else None,
                                         "concat_ms": .5 if lb.startswith("S2s") else 0.0}
                                    out.append("ROW_JSON " + json.dumps(r))
            out.append("EXIT_CODE: 0")
            logs[chip] = tmp / f"{name}_{chip}.log"
            logs[chip].write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"\n---- synthetic verdict, {name} (DirectML fp16 at {dml16} TFLOPS)")
        assert evaluate([logs["npu"], logs["cpu"], logs["dml"]]) == want, name
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("inputs", "build", "prereg", "npu", "selftest"):
        sub.add_parser(c)
    o = sub.add_parser("ort")
    o.add_argument("--ep", choices=("cpu", "dml"), required=True)
    v = sub.add_parser("verdict")
    v.add_argument("logs", nargs=3, help="the NPU, CPU and DirectML sitting logs")
    a = ap.parse_args()
    if a.cmd == "ort":
        return ort_rows(a.ep)
    if a.cmd == "verdict":
        return verdict(a.logs)
    return {"inputs": inputs, "build": build, "prereg": prereg, "npu": npu, "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
