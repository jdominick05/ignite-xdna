#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Laya end to end with its weight products on the NPU: step 5 of the Laya study (the hybrid).

Laya's 120 weight MatMuls at 512 rows (tools/laya_accuracy.py ``weight_products``, minus the four head Gemms of 1-4
rows) run as bf16 GEMMs on the NPU; everything between them runs on the CPU through ONNX Runtime. The graph is cut
at those 120 products into 121 host segments, each its own CPU session, so a sequence is: segment, product,
segment, ... product, segment. The products can run three ways, which is how the split and the NPU are checked
separately:

  fp32   numpy fp32 matmul: the split alone. Its logits must match the whole fp32 model to BLAS-vs-MLAS noise.
  bf16   numpy on bf16-rounded operands, fp32 accumulate: the same computation as the NPU's, and the same as
         tools/laya_accuracy.py ``bf16copy``'s emulation, up to accumulation order. This calibrates the band the
         NPU is held to.
  npu    the one-context generic GEMM (kernels/gemm_generic/whole_array_generic.py --rtp-generic, built by
         tools/gemm_generic_probe.py compile into a directory such as scratch/gemm_generic/aot6): one hardware
         context, six instruction streams, 120 weight buffers written once and resident. Activations are rounded
         to bf16 (nearest, ties to even) straight into a mapped input buffer; results are read from a mapped
         output buffer, zero-copy where the next segment is the only consumer.

    python tools/laya_hybrid.py check --mode fp32|bf16|npu --model-dir D [--model M] [--ref R] [--gemm-dir G]
    python tools/laya_hybrid.py time --mode npu|fp32|bf16 --model-dir D [--gemm-dir G] [--calls 20]

``check`` runs Laya's reference support ticket (three questions) and three synthetic inputs, all padded to 512
tokens with [PAD] under attention mask 0, through the hybrid and through the whole model ``--ref`` on the CPU
(default: the served model for fp32; for bf16 and npu, pass the ``bf16copy`` of it), and prints the largest
|logit difference| and whether the top option agrees. ``time`` prints the per-call split: host segments,
rounding and writes, NPU calls, reads. Every input is 512 tokens: the hybrid is built for that shape only.

Runs in the mlir-aie-iron env (ONNX Runtime CPU and pyxrt in one process). The NPU mode refuses to start unless
xrt-smi reports no hardware contexts, gives every dispatch a timeout, and stops at the first failed state.
"""
import argparse
import gc
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya_accuracy as la  # noqa: E402

SEQ = 512
THREADS = 8
DEFAULT_MODEL = "laya_az0_noguard.onnx"
XRT_PY = "C:/Xilinx/XRT/xrt_sdk/xrt/python"
XRT_SMI = "C:/Windows/System32/AMD/xrt-smi.exe"
WAIT_MS = 5000
GEMM_SHAPES = {(1024, 3072), (1024, 1024), (1024, 5248), (2624, 1024), (1024, 4096), (4096, 1024)}


def padded_n(n: int) -> int:
    return -(-n // 256) * 256


# ---------------------------------------------------------------- the split
class Plan:
    """The 121 host segments and the 120 products between them, from one ONNX model."""

    def __init__(self, model_path: Path, threads: int = THREADS):
        import onnx
        import onnxruntime as ort
        from onnx import helper, shape_inference
        m = onnx.load(str(model_path))                       # with external data: the weights come along
        g = m.graph
        init = {i.name: i for i in g.initializer}
        # Cut at every weight MatMul (120). The four head Gemms stay inside host segments. A product goes to the NPU
        # only if its activation has 512 rows and its shape is one of the six designs; the rest run in numpy.
        prods = [j for j in la.weight_products(m) if g.node[j].op_type == "MatMul"]
        if len(prods) != 120:
            raise SystemExit(f"expected 120 weight MatMuls, found {len(prods)}")
        # Element types of every tensor, from shape inference at 512 tokens.
        light = onnx.load(str(model_path), load_external_data=False)
        for vi in light.graph.input:
            if vi.name in ("input_ids", "attention_mask"):
                d = vi.type.tensor_type.shape.dim
                d[0].dim_value, d[1].dim_value = 1, SEQ
        inferred = shape_inference.infer_shapes(light)
        etype = {vi.name: vi.type.tensor_type.elem_type
                 for vi in list(inferred.graph.value_info) + list(g.input) + list(g.output)}
        etype.update({name: t.data_type for name, t in init.items()})
        graph_inputs = {vi.name for vi in g.input}
        graph_outputs = [vi.name for vi in g.output]
        # Products: activation, weight, output, (K, N).
        self.products = []
        for j in prods:
            n = g.node[j]
            w = onnx.numpy_helper.to_array(init[n.input[1]]).astype(np.float32)
            K, N = w.shape
            self.products.append({"act": n.input[0], "out": n.output[0], "w": w, "K": K, "N": N, "name": n.name})
        # Segments: the nodes strictly between consecutive products.
        bounds = [-1] + prods + [len(g.node)]
        seg_nodes = [list(range(a + 1, b)) for a, b in zip(bounds[:-1], bounds[1:])]
        consumers = {}
        for s, idx in enumerate(seg_nodes):
            for j in idx:
                for t in g.node[j].input:
                    consumers.setdefault(t, set()).add(("seg", s))
        for p, pr in enumerate(self.products):
            consumers.setdefault(pr["act"], set()).add(("prod", p))
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        # 121 sessions each own an intra-op pool; spinning pools left running after a segment steal the cores the
        # next segment and the activation rounding need.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.segments = []
        for s, idx in enumerate(seg_nodes):
            nodes = [g.node[j] for j in idx]
            produced = {o for n in nodes for o in n.output if o}
            needed = [t for n in nodes for t in n.input if t and t not in produced]
            needed = list(dict.fromkeys(needed))
            inits = [init[t] for t in needed if t in init]
            ins = [t for t in needed if t not in init]
            outs = [t for t in produced
                    if t in graph_outputs or any(c != ("seg", s) for c in consumers.get(t, ()))]
            if not nodes:
                self.segments.append(None)
                continue
            gi = [helper.make_tensor_value_info(t, etype[t], None) for t in ins]
            go = [helper.make_tensor_value_info(t, etype[t], None) for t in outs]
            sg = helper.make_graph(nodes, f"seg{s}", gi, go, initializer=inits)
            sm = helper.make_model(sg, opset_imports=m.opset_import, ir_version=m.ir_version)
            sess = ort.InferenceSession(sm.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"])
            self.segments.append({"sess": sess, "ins": ins, "outs": outs})
            missing = [t for t in ins if t not in graph_inputs and t not in
                       {o for x in self.segments[:-1] if x for o in x["outs"]} | {p["out"] for p in self.products}]
            if missing:
                raise SystemExit(f"segment {s}: inputs produced nowhere earlier: {missing[:4]}")
        self.graph_outputs = graph_outputs
        # Whether a product's output is read by anything but the next segment (then it must be copied off the
        # mapped output buffer before a later product of the same shape overwrites it).
        for p, pr in enumerate(self.products):
            pr["only_next"] = consumers.get(pr["out"], set()) <= {("seg", p + 1)} and pr["out"] not in graph_outputs
        del m

    def run(self, feed: dict, product) -> list[np.ndarray]:
        env = dict(feed)
        for s, seg in enumerate(self.segments):
            if seg is not None:
                res = seg["sess"].run(seg["outs"], {t: env[t] for t in seg["ins"]})
                env.update(zip(seg["outs"], res))
            if s < len(self.products):
                pr = self.products[s]
                x = env[pr["act"]]
                y = product(s, x.reshape(-1, pr["K"]))
                env[pr["out"]] = y.reshape(x.shape[:-1] + (pr["N"],))
        return [env[t] for t in self.graph_outputs]


# ---------------------------------------------------------------- the products
def bf16_round(x: np.ndarray) -> np.ndarray:
    """fp32 -> fp32 holding the nearest bf16 (ties to even), as tools/gemm_generic_probe.to_bf16_bits."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)).view(np.float32)


class NumpyProducts:
    def __init__(self, plan: Plan, bf16: bool):
        self.w = [bf16_round(p["w"]) if bf16 else p["w"] for p in plan.products]
        self.bf16 = bf16

    def __call__(self, i, x):
        return (bf16_round(x) if self.bf16 else x) @ self.w[i]


def idle() -> bool:
    import subprocess
    out = subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True,
                         timeout=60).stdout
    return "No hardware contexts running" in out


class NpuProducts:
    """One context on the generic design; per shape an instruction BO, a mapped A (bf16 bits) and a mapped C (fp32);
    per product a resident B (bf16 bits, N padded with zero columns)."""

    def __init__(self, plan: Plan, gemm_dir: Path):
        sys.path.insert(0, XRT_PY)
        import pyxrt
        if not idle():
            raise SystemExit("NPU not idle: refusing (a foreign context is contention, not a finding)")
        self.x = x = pyxrt
        self.dev = x.device(0)
        any_xclbin = gemm_dir / "g_1024_3072.xclbin"
        self.xb = x.xclbin(str(any_xclbin))
        self.dev.register_xclbin(self.xb)
        self.ctx = x.hw_context(self.dev, self.xb.get_uuid())
        self.kern = x.kernel(self.ctx, self.xb.get_kernels()[0].get_name())
        self.pool = ThreadPoolExecutor(THREADS)
        self.scratch = {}
        self.shape = {}
        # Products the GEMM designs do not cover (the 1024x1 one), bf16 on the CPU; only their weights are kept.
        self.host_w = {i: bf16_round(p["w"]) for i, p in enumerate(plan.products)
                       if (p["K"], p["N"]) not in GEMM_SHAPES}
        self.counts = {"npu": 0, "host": 0}
        for K, N in sorted({(p["K"], p["N"]) for p in plan.products} & GEMM_SHAPES):
            words = np.frombuffer((gemm_dir / f"g_{K}_{N}.bin").read_bytes(), dtype=np.uint32)
            ib = x.bo(self.dev, words.nbytes, x.bo.cacheable, self.kern.group_id(1))
            ib.write(words.tobytes(), 0)
            ib.sync(x.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            npad = padded_n(N)
            A = x.bo(self.dev, SEQ * K * 2, x.bo.host_only, 0)
            C = x.bo(self.dev, SEQ * npad * 4, x.bo.host_only, 0)
            a_view = np.frombuffer(A.map(), dtype=np.uint16).reshape(SEQ, K)
            c_view = np.frombuffer(C.map(), dtype=np.float32).reshape(SEQ, npad)
            self.shape[K, N] = {"ib": ib, "nb": words.nbytes, "A": A, "C": C, "a": a_view, "c": c_view, "np": npad}
        self.plan_w = [p["w"] for p in plan.products]   # a 512-row product with fewer rows falls back here
        self.B, self.prod_shape, self.only_next = [], [], []
        for p in plan.products:
            K, N = p["K"], p["N"]
            if (K, N) not in GEMM_SHAPES:
                self.B.append(None)
                self.prod_shape.append((K, N))
                self.only_next.append(False)
                continue
            w = np.zeros((K, padded_n(N)), np.float32)
            w[:, :N] = p["w"]
            bits = (bf16_round(w).view(np.uint32) >> 16).astype(np.uint16)
            B = x.bo(self.dev, bits.nbytes, x.bo.host_only, 0)
            B.write(bits.tobytes(), 0)
            B.sync(x.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self.B.append(B)
            self.prod_shape.append((K, N))
            self.only_next.append(p["only_next"])
        self.t = {"round": 0.0, "npu": 0.0, "read": 0.0}

    def _round_into(self, x, out):
        """bf16 bits of x into out (the mapped A). Rounded into a local buffer with rows split across threads (numpy
        releases the GIL), then copied in one contiguous pass: element-wise stores straight into the mapped host
        buffer ran about 20x slower (1.41 s per sequence, 2026-10-14)."""
        local = self.scratch.setdefault(x.shape, np.empty(x.shape, np.uint16))

        def part(r0, r1):
            u = x[r0:r1].view(np.uint32)
            t = u >> 16
            t &= np.uint32(1)
            t += np.uint32(0x7FFF)
            t += u
            t >>= 16
            local[r0:r1] = t
        step = SEQ // THREADS
        list(self.pool.map(lambda r: part(r, r + step), range(0, SEQ, step)))
        np.copyto(out, local)

    def __call__(self, i, x):
        K, N = self.prod_shape[i]
        if self.B[i] is None or x.shape[0] != SEQ:
            self.counts["host"] += 1
            w = self.host_w[i] if i in self.host_w else bf16_round(self.plan_w[i])
            return bf16_round(x) @ w
        self.counts["npu"] += 1
        sh = self.shape[K, N]
        t0 = time.perf_counter()
        self._round_into(np.ascontiguousarray(x, np.float32), sh["a"])
        t1 = time.perf_counter()
        S = self.x.xclBOSyncDirection
        sh["A"].sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
        h = self.kern(3, sh["ib"], sh["nb"], sh["A"], self.B[i], sh["C"])
        st = h.wait(WAIT_MS)
        if st != self.x.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"product {i} ({K}x{N}): {st}")
        sh["C"].sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
        t2 = time.perf_counter()
        c = sh["c"]
        y = c if sh["np"] == N else c[:, :N]
        if not (self.only_next[i] and sh["np"] == N):
            y = np.array(y, copy=True)
        t3 = time.perf_counter()
        self.t["round"] += t1 - t0
        self.t["npu"] += t2 - t1
        self.t["read"] += t3 - t2
        return y

    def close(self):
        self.pool.shutdown()
        for sh in self.shape.values():
            sh.clear()
        self.shape.clear()
        self.B.clear()
        del self.kern, self.ctx
        gc.collect()


# ---------------------------------------------------------------- inputs
def padded_feed(ids, markers, qt) -> dict:
    n = len(ids)
    return {"input_ids": np.array([ids + [la.PAD] * (SEQ - n)], np.int64),
            "attention_mask": np.array([[1] * n + [0] * (SEQ - n)], np.int64),
            "marker_pos": np.array([markers], np.int64), "marker_mask": np.ones((1, len(markers)), bool),
            "qtype": np.array([qt], np.int64)}


def inputs(model_dir: Path) -> list[tuple[str, dict, int]]:
    cfg = json.loads((model_dir / "laya_config.json").read_text(encoding="utf-8"))
    enc = la.tokenizer(model_dir)
    out = []
    for qid, qdef in la.FIXTURE_QS.items():
        q = la.to_internal(qdef)
        ids, mk = la.build_sequence(enc, la.FIXTURE_STATE, q, cfg["max_len"], cfg["head_max_len"])
        out.append((f"ticket/{qid}", padded_feed(ids, mk, la.QTYPES[q["t"]]), len(mk)))
    rng = np.random.default_rng(7)
    for r in range(3):
        n = [180, 400, SEQ][r]
        ids = [la.CLS] + rng.integers(1000, 50000, n - 2).tolist() + [la.SEP]
        mk = sorted(rng.choice(np.arange(5, n - 5), 4, replace=False).tolist())
        for m_ in mk:
            ids[m_] = la.MASK
        out.append((f"synthetic/{n}", padded_feed(ids, mk, 0), 4))
    return out


def build(a):
    model_dir = Path(a.model_dir)
    model = Path(a.model) if a.model else model_dir / DEFAULT_MODEL
    t0 = time.perf_counter()
    plan = Plan(model)
    print(f"[plan] {model.name}: {len(plan.products)} products, {sum(s is not None for s in plan.segments)} host "
          f"segments, built in {time.perf_counter() - t0:.1f} s; outputs read zero-copy for "
          f"{sum(p['only_next'] for p in plan.products)} products", flush=True)
    if a.mode == "npu":
        prod = NpuProducts(plan, Path(a.gemm_dir))
    else:
        prod = NumpyProducts(plan, a.mode == "bf16")
    return model_dir, model, plan, prod


def cmd_check(a) -> int:
    import laya_baseline as lb
    model_dir, model, plan, prod = build(a)
    ref = lb.session(Path(a.ref) if a.ref else model, "cpu", THREADS)
    worst, agree, rows = 0.0, 0, inputs(model_dir)
    emu = NumpyProducts(plan, bf16=True) if a.mode == "npu" else None
    worst_emu = 0.0
    try:
        for name, feed, k in rows:
            z = plan.run(feed, prod)[0][0, :k]
            r = ref.run(None, feed)[0][0, :k]
            d = float(np.abs(z - r).max())
            same = int(z.argmax()) == int(r.argmax())
            worst, agree = max(worst, d), agree + same
            extra = ""
            if emu is not None:   # the same split with the products in numpy bf16: differs only in GEMM summation
                e = plan.run(feed, emu)[0][0, :k]
                de = float(np.abs(z - e).max())
                worst_emu = max(worst_emu, de)
                extra = f", max |logit - numpy-bf16 hybrid| {de:.3e}"
            print(f"CHECK {a.mode} {name}: finite {bool(np.isfinite(z).all())}, max |logit - ref| {d:.3e}, "
                  f"logit range {float(np.abs(r).max()):.2f}, same top option {same}{extra}", flush=True)
        if emu is not None:
            print(f"CHECK npu products: {prod.counts['npu']} on the NPU, {prod.counts['host']} on the host (bf16); "
                  f"worst max |logit - numpy-bf16 hybrid| {worst_emu:.3e}", flush=True)
    finally:
        if a.mode == "npu":
            prod.close()
    print(f"CHECK {a.mode} against {Path(a.ref).name if a.ref else model.name}: worst max |logit diff| {worst:.3e}; "
          f"same top option {agree}/{len(rows)}", flush=True)
    if a.mode == "npu":
        print(f"idle after: {idle()}", flush=True)
    return 0


def cmd_time(a) -> int:
    model_dir, model, plan, prod = build(a)
    feed = inputs(model_dir)[-1][1]
    try:
        for _ in range(3):
            plan.run(feed, prod)
        if a.mode == "npu":
            prod.t = {k: 0.0 for k in prod.t}
        ts = []
        for _ in range(a.calls):
            t0 = time.perf_counter()
            plan.run(feed, prod)
            ts.append((time.perf_counter() - t0) * 1e3)
        med = float(np.median(ts))
        line = f"TIME {a.mode} 512 tokens: median {med:.1f} ms (min {min(ts):.1f}, max {max(ts):.1f}) over {a.calls}"
        if a.mode == "npu":
            per = {k: v * 1e3 / a.calls for k, v in prod.t.items()}
            line += (f"; per call: rounding+write {per['round']:.1f}, NPU (sync, call, wait, sync) {per['npu']:.1f}, "
                     f"read {per['read']:.1f}, host segments {med - sum(per.values()):.1f} ms (by difference)")
        print(line, flush=True)
    finally:
        if a.mode == "npu":
            prod.close()
            print(f"idle after: {idle()}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "time"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=("fp32", "bf16", "npu"), required=True)
        p.add_argument("--model-dir", required=True)
        p.add_argument("--model", default=None, help=f"default <model-dir>/{DEFAULT_MODEL}")
        p.add_argument("--gemm-dir", default="scratch/gemm_generic/aot6")
        if name == "check":
            p.add_argument("--ref", default=None, help="whole model on the CPU to compare with (default: --model)")
        else:
            p.add_argument("--calls", type=int, default=20)
    a = ap.parse_args()
    return {"check": cmd_check, "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
