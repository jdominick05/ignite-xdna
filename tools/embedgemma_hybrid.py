#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""EmbeddingGemma 2 end to end with its weight products on the NPU and everything else on the host (NumPy fp32).

    python tools/embedgemma_hybrid.py --backend cpu --docs DOCS --reference REF.npz --out OUT.json
    python tools/embedgemma_hybrid.py --backend npu --builds DIR --docs DOCS --reference REF.npz --out OUT.json

``kernels/embedgemma/model.py`` is the model. ``cpu`` runs it with fp32 NumPy products (the oracle check);
``npu`` routes all 146 weight products per document (148 calls: the per-layer-input projection is three 4096-wide
chunks) through ONE context of the generic GEMM design that tools/embedgemma_gemm_probe.py compiled (bf16 in, f32
out), weights resident in device BOs. Each product's input is rounded to BF16 once, nearest-even, straight into
the mapped A buffer, and C is read through its mapped view (no ``bo.read`` copy). Rows are zero-padded to the
smallest compiled M >= rows. ``--batch N`` runs N same-length documents as one row batch (block-diagonal
attention, per-document positions and pooling), so two 128-token documents fill M = 256. DOCS is the token-id json
the GPU/CPU baselines used; REF is HF transformers' fp32 embeddings of the same documents (``embeddings`` [n, 768]
in DOCS order).

Gates, fixed before any run (2026-10-18):
  cpu: cosine to the HF fp32 reference >= 0.99999 on every document (an independent re-implementation must agree
       to fp32 rounding).
  npu: cosine to the HF fp32 reference >= 0.9999 on every document, and every document keeps its HF nearest
       neighbour among the documents (CPU torch bf16 reached 0.999928 and the 780M 0.99998 on these documents).
  cpu with --bf16-inputs or --attention blockwise is a device-numerics emulation, held to the npu gate.
--attention blockwise replaces the host attention with kernels/embedgemma/attention_oracle.py (the fused kernel's
online-softmax form: key blocks of --k-block, BF16 Q/K/V, BF16 exp and P); --bf16-inputs rounds every weight
product's input to BF16 as the NPU GEMM does. --attention npu-local runs every local (sliding-window) layer's
attention on the device: kernels/embedgemma/attn_array.py's 16-core fused kernel, one dispatch per document per
layer (all four heads), in its own context on --attn-build's doc2048 xclbin; global layers stay host fp32. It is
held to the npu gate. Both are numerics checks: their timings mean nothing.
Timing: wall time per batch divided by its documents, median per length, after one untimed warm-up batch per
(length, batch size). Phases: ``gemm`` is the whole weight-product hook; inside it ``npu_pack`` (round to BF16
into the mapped A), ``npu_write`` (A sync), ``npu_call`` (submit + wait), ``npu_read`` (C sync) and ``npu_unpack``
(chunk concatenation); ``--round-dll`` does the packing natively. attention, norm, rope, gelu and embed are host NumPy fp32 (single-threaded, not a tuned CPU
path). Run the npu backend under scripts/research-lowlevel.sh --npu.

History: the first run (results/aie/embedgemma_hybrid_npu_desktop2_20261018T203211Z) rounded every input twice
(``round_in`` in the model, again while packing) and copied each C readback twice (``bo.read`` then ``.copy()``).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels" / "embedgemma"))
from model import Encoder, bf16_bits  # noqa: E402

XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
WAIT_MS = 10000
CHUNK = 4096   # widest N the tile's shim DMA accepts here (N = 12288 overflows a stride)


def padded(N):
    return -(-N // 256) * 256


def idle():
    out = subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True, timeout=60).stdout
    return "No hardware contexts running" in out


class Npu:
    """Every weight product through one generic-GEMM context. Weight BOs resident; one A/C BO pair per (M, K, N)."""

    def __init__(self, enc, builds, ms, rounder=None, geglu_mode=0, w8=None, b_resident=False):
        """w8: {key: (q int8 [K, N], s [K / 64, N])} -> upload W8 tiles (builds compiled with --w8) instead of bf16.
        b_resident: builds compiled with --b-resident read B strip by strip in L1 tile order, so upload it swizzled."""
        self.rounder, self.geglu_mode = rounder, geglu_mode
        sys.path.insert(0, "C:/Xilinx/XRT/xrt_sdk/xrt/python")
        import pyxrt
        self.x, self.ms, self.builds = pyxrt, sorted(ms), builds
        self.to_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.from_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.dev = pyxrt.device(0)
        self.xb = pyxrt.xclbin(str(builds / f"g_{min(ms)}_512_512.xclbin"))   # every shape shares one context
        self.dev.register_xclbin(self.xb)
        self.ctx = pyxrt.hw_context(self.dev, self.xb.get_uuid())
        self.kern = pyxrt.kernel(self.ctx, self.xb.get_kernels()[0].get_name())
        self.t = dict(npu_pack=0.0, npu_write=0.0, npu_call=0.0, npu_read=0.0, npu_unpack=0.0)
        self.B, self.io = {}, {}
        if w8:
            from embedgemma_gemm_probe import w8_tiles
        if b_resident:
            assert not w8, "--npu-b-resident is bf16 only"
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels" / "gemm_generic"))
            from whole_array_generic import swizzle_b_resident

        def inter(a):   # per 64-column tile: 32 gate columns, then the same 32 up columns
            return a.reshape(a.shape[0], 2, a.shape[1] // 64, 32).transpose(0, 2, 1, 3).reshape(a.shape)
        for key, w in enc.weights.items():   # [K, N] fp32 holding exact BF16 values
            K, N = w.shape
            q, sc = w8[key] if w8 else (None, None)
            if geglu_mode and key[0] == "gate_up":   # q and its scales take the same column permutation
                w = inter(w)
                if w8:
                    q, sc = inter(q), inter(sc)
            chunks = []
            for c in range(0, N, CHUNK):
                part = w[:, c:c + CHUNK]
                n = part.shape[1]
                if w8:   # padded columns: q = 0, s = 1 -> exactly 0
                    qp, sp = np.zeros((K, padded(n)), np.int8), np.ones((K // 64, padded(n)), np.float32)
                    qp[:, :n], sp[:, :n] = q[:, c:c + n], sc[:, c:c + n]
                    bits = w8_tiles(qp, sp)
                else:
                    bits = np.zeros((K, padded(n)), np.uint16)
                    bits[:, :n] = bf16_bits(part)
                    if b_resident:
                        bits = swizzle_b_resident(bits)
                bo = self._bo(bits.nbytes)
                bo.write(bits.tobytes(), 0)
                bo.sync(self.to_dev)
                chunks.append((bo, n))
            self.B[key] = chunks

    def _bo(self, nbytes):
        return self.x.bo(self.dev, nbytes, self.x.bo.host_only, 0)

    def _io(self, M, K, Np, mode=0):
        key = (M, K, Np, mode)
        if key not in self.io:
            name = f"g_{M}_{K}_{Np}" + (f"_m{mode}" if mode else "")
            words = np.frombuffer((self.builds / f"{name}.bin").read_bytes(), dtype=np.uint32)
            ib = self.x.bo(self.dev, words.nbytes, self.x.bo.cacheable, self.kern.group_id(1))
            ib.write(words.tobytes(), 0)
            ib.sync(self.to_dev)
            A, C = self._bo(M * K * 2), self._bo(M * Np * 4)
            av = np.frombuffer(A.map(), dtype=np.uint16, count=M * K).reshape(M, K)
            av[:] = 0   # padding rows: zero once; a shorter batch at the same M leaves stale rows, which only feed
            #             discarded output rows (each output row depends on its own input row alone)
            cv = np.frombuffer(C.map(), dtype=np.float32, count=M * Np).reshape(M, Np)
            self.io[key] = (ib, words.nbytes, A, C, av, cv)
        return self.io[key]

    def matmul(self, x, name, layer):
        """fp32 [L, K] -> fp32 [L, N]. A one-chunk result is a view into C's mapping, valid until the next product
        of the same (M, K, N); every consumer in model.py derives a new array from it before that."""
        L, K = x.shape
        M = next(m for m in self.ms if m >= L)
        chunks = self.B[(name, layer)]
        mode = self.geglu_mode if name == "gate_up" else 0
        outs = []
        packed = None
        for bo_w, n in chunks:
            Np = padded(n)
            ib, nb, A, C, av, cv = self._io(M, K, Np, mode)
            t0 = time.perf_counter()
            if packed is None:   # round once per product; every chunk shares the input
                if self.rounder:
                    self.rounder.round_into(x, av)
                else:
                    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
                    av[:L] = (u + ((u >> 16) & 1) + 0x7FFF) >> 16
                packed = av
            elif packed is not av:
                av[:L] = packed[:L]
            t1 = time.perf_counter()
            A.sync(self.to_dev)
            t2 = time.perf_counter()
            st = self.kern(3, ib, nb, A, bo_w, C).wait(WAIT_MS)
            if st != self.x.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"{name}[{layer}] M={M} K={K} N={Np}: {st}")
            t3 = time.perf_counter()
            C.sync(self.from_dev)
            t4 = time.perf_counter()
            # chunks of one product share an (M, K, N) and so one C buffer: copy each before the next call
            if mode:   # the device's GeGLU sits at each tile's gate columns: ff [L, n / 2]
                outs.append(cv[:L, :n].reshape(L, n // 64, 2, 32)[:, :, 0, :].reshape(L, n // 2))
            else:
                outs.append(cv[:L, :n] if len(chunks) == 1 else cv[:L, :n].copy())
            self.t["npu_pack"] += t1 - t0
            self.t["npu_write"] += t2 - t1
            self.t["npu_call"] += t3 - t2
            self.t["npu_read"] += t4 - t3
        t5 = time.perf_counter()
        y = outs[0] if len(outs) == 1 else np.concatenate(outs, axis=1)
        self.t["npu_unpack"] += time.perf_counter() - t5
        return y

    def close(self):
        """Release BOs, kernel, context, xclbin, device in that order (a BO outliving the device segfaults at exit)."""
        self.B.clear()
        self.io.clear()
        gc.collect()
        del self.kern
        gc.collect()
        del self.ctx
        gc.collect()
        del self.xb, self.dev
        gc.collect()


def serve_queue(addr, enc, matmul, bf16, rows):
    """The bulk lane: pull a pack of documents (at most `rows` tokens), embed it as one row batch, send it back."""
    import base64
    import socket
    host, port = addr.rsplit(":", 1)
    with socket.create_connection((host, int(port))) as sk:
        f = sk.makefile("rw", encoding="utf-8", newline="\n")
        f.write(json.dumps(dict(hello="npu", rows=rows)) + "\n")
        f.flush()
        n = 0
        while True:
            f.write(json.dumps(dict(want=rows)) + "\n")
            f.flush()
            msg = json.loads(f.readline())
            if msg.get("done"):
                break
            idx = [i for i, _ in msg["docs"]]
            e = enc.embed_documents([ids for _, ids in msg["docs"]], matmul, bf16).astype(np.float32)
            f.write(json.dumps(dict(results=[[i, base64.b64encode(v.tobytes()).decode()] for i, v in zip(idx, e)]))
                    + "\n")
            f.flush()
            n += len(idx)
    print(f"QUEUE_DONE {n} documents", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cpu", "npu"], required=True)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--builds", type=Path, help="tools/embedgemma_gemm_probe.py compile --out DIR (npu)")
    ap.add_argument("--ms", type=int, nargs="+", default=[256, 512, 2048])
    ap.add_argument("--batch", type=int, default=1, help="consecutive same-length documents per row batch")
    ap.add_argument("--attention", choices=["exact", "blockwise", "npu-local"], default="exact")
    ap.add_argument("--geglu", choices=["host", "model1", "model2", "model5"], default="host",
                    help="MLP GeGLU: host fp32, or (cpu emulation) geglu_epilogue.cc's mode 1 / mode 2 arithmetic")
    ap.add_argument("--wchannel", action="store_true", help="--w4: one scale per output column over all of K "
                    "(group = K; W8A8's weights), --w4's value is then ignored")
    ap.add_argument("--a8", action="store_true", help="cpu: quantize every weight product's input to int8 per token "
                    "(w4.quantize_rows) and dequantize it before the fp32 product (W8A8 emulation)")
    ap.add_argument("--wbits", type=int, choices=[4, 8], default=4, help="bits for --w4 (kernels/embedgemma/w4.py)")
    ap.add_argument("--wclip", type=int, default=0, help="--w4: per-group MSE clip search points over alpha in [0.85, 1]; 0 = RTN")
    ap.add_argument("--w4", type=int, default=0, help="quantize every weight product to int4 (kernels/embedgemma/w4.py) "
                    "with this group size along K, and run on its bf16 dequantization; 0 = bf16 weights")
    ap.add_argument("--native-rowops", type=Path, help="run every other norm, residual add and the per-layer-input "
                    "GELU through this eg_rowops.dll (native.py build --rowops)")
    ap.add_argument("--native-qkv", type=Path, help="run the post-QKV norms and RoPE through this "
                    "eg_qkv_norm_rope.dll (kernels/embedgemma/native.py build --qkv) instead of model.py's numpy")
    ap.add_argument("--npu-b-resident", action="store_true", help="npu: --builds compiled with "
                    "embedgemma_gemm_probe.py compile --b-resident (weights uploaded in their swizzled order)")
    ap.add_argument("--npu-w8", action="store_true", help="npu: upload the --w4 64 --wbits 8 weights as W8 tiles "
                    "(--builds compiled with embedgemma_gemm_probe.py compile --w8 --w8-dequant)")
    ap.add_argument("--npu-geglu", type=int, default=0, help="npu: run gate|up with this geglu_epilogue.cc mode "
                    "(--builds compiled with tools/embedgemma_gemm_probe.py compile --epilogue); 0 = host GeGLU")
    ap.add_argument("--attn-build", type=Path, help="npu-local: kernels/embedgemma/attn_array.py build --out DIR")
    ap.add_argument("--attn-pack-dll", type=Path, help="npu-local: egattn_pack.dll (native.py build --attn)")
    ap.add_argument("--k-block", type=int, default=32)
    ap.add_argument("--exp", choices=["exact", "lut", "fixed"], default="exact",
                    help="blockwise: exp model (lut = v4 kernel table exp, fixed = v5 integer softmax)")
    ap.add_argument("--tau", type=float, default=0.0, help="blockwise: lazy-rescale threshold (kernel: 8)")
    ap.add_argument("--bf16-inputs", action="store_true", help="cpu: round weight-product inputs to BF16")
    ap.add_argument("--round-dll", type=Path, help="npu: kernels/embedgemma/native.py build's embedgemma_round.dll "
                                                    "(AVX2 + OpenMP, bit-identical) instead of NumPy rounding")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--loop-go", type=Path, help="after the gated run: write FILE.ready, wait for FILE to hold "
                    "'START STOP' (epoch seconds), then encode --loop-length documents in --batch groups from START "
                    "until STOP and write OUT.loop.json with each group's wall-clock span "
                    "(tools/embedgemma_concurrency_energy.py)")
    ap.add_argument("--loop-length", type=int, default=2048)
    ap.add_argument("--queue", help="after the gated run: serve as the NPU lane of tools/embedgemma_bulk.py at "
                    "HOST:PORT (one JSON line each way: {want: rows} -> {docs: [[i, ids]...]} or {done: true}; "
                    "then {results: [[i, base64 float32 embedding]...]})")
    ap.add_argument("--save-embeddings", type=Path, help="also write the embeddings as an npz (key 'embeddings'), "
                    "e.g. to serve as a W8-emulation --reference")
    ap.add_argument("--quality-reference", type=Path, help="W8 gate: when --reference is a W8 CPU emulation, also "
                    "require every document to keep its nearest neighbour under this HF fp32 reference")
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    docs = json.loads(a.docs.read_text(encoding="utf-8"))["docs"]
    ref = np.load(a.reference)["embeddings"]
    if len(ref) != len(docs):
        raise ValueError("reference and docs differ in count")
    t = time.perf_counter()
    enc = Encoder()
    w8_raw = {}
    if a.w4:
        import w4
        for key, w in list(enc.weights.items()):
            q, sc = w4.quantize(w, w.shape[0] if a.wchannel else a.w4, a.wbits, a.wclip)
            enc.weights[key] = w4.dequant(q, sc)
            if a.npu_w8:
                w8_raw[key] = (q, sc)
        print(f"W{a.wbits}: {len(enc.weights)} weight products quantized, group "
              f"{'K (per channel)' if a.wchannel else a.w4} along K, clip points {a.wclip}", flush=True)
    if a.attention == "blockwise":
        import attention_oracle

        def blockwise(q, k, v, layer, segs):   # rows are independent, so one query block per document is exact
            return np.concatenate([attention_oracle.attention(q[x:y], k[x:y], v[x:y], layer["kind"], enc.window,
                                                              q_block=y - x, k_block=a.k_block, exp=a.exp,
                                                              tau=a.tau) for x, y in segs])
        enc._attention = blockwise
    if a.geglu.startswith("model"):
        import geglu_model
        mode = int(a.geglu[-1])

        def geglu_emulated(gu):
            t = time.perf_counter()
            h = gu.shape[1] // 2
            out = geglu_model.geglu(gu[:, :h], gu[:, h:], mode)
            enc.timings["gelu"] += time.perf_counter() - t
            return out
        enc.geglu_hook = geglu_emulated
    if a.native_rowops:
        import ctypes
        rl = ctypes.CDLL(str(a.native_rowops.resolve()))
        P_, I64, F = ctypes.c_void_p, ctypes.c_int64, ctypes.c_float
        rl.eg_rms.argtypes = [P_, I64, I64, P_, F, P_]
        rl.eg_add_rms.argtypes = [P_, P_, I64, I64, P_, F, F, P_]
        rl.eg_gelu_mul.argtypes = [P_, P_, I64, P_]

        def c32(x):
            return np.ascontiguousarray(x, np.float32)

        class Rowops:
            @staticmethod
            def rms(x, w):
                x = c32(x)
                out = np.empty_like(x)
                wv = c32(w) if w is not None else None
                rl.eg_rms(x.ctypes.data, x.size // x.shape[-1], x.shape[-1], wv.ctypes.data if wv is not None else None,
                          float(enc.eps), out.ctypes.data)
                return out

            @staticmethod
            def add_rms(res, a_, w, scalar):
                res, a_, wv = c32(res), c32(a_), c32(w)
                out = np.empty_like(res)
                rl.eg_add_rms(res.ctypes.data, a_.ctypes.data, res.size // res.shape[-1], res.shape[-1], wv.ctypes.data,
                              float(enc.eps), float(scalar), out.ctypes.data)
                return out

            @staticmethod
            def gelu_mul(a_, b):
                a_, b = c32(a_), c32(b)
                out = np.empty_like(a_)
                rl.eg_gelu_mul(a_.ctypes.data, b.ctypes.data, a_.size, out.ctypes.data)
                return out
        enc.native = Rowops()
    if a.native_qkv:
        import ctypes
        lib = ctypes.CDLL(str(a.native_qkv.resolve()))
        P = ctypes.c_void_p
        lib.eg_qkv_norm_rope.argtypes = [P, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, P, P, P, P,
                                         ctypes.c_float, P, P, P]

        def qkv_native(qkv, lay, cos, sin):
            t = time.perf_counter()
            L, hd, kv, H = qkv.shape[0], lay["hd"], lay["kv"], enc.H
            src = np.ascontiguousarray(qkv, np.float32)
            q, k, v = (np.empty((L, n, hd), np.float32) for n in (H, kv, kv))
            qw, kw = (np.ascontiguousarray(lay[n], np.float32) for n in ("q_norm", "k_norm"))
            c, s = np.ascontiguousarray(cos, np.float32), np.ascontiguousarray(sin, np.float32)
            lib.eg_qkv_norm_rope(src.ctypes.data, L, H, kv, hd, qw.ctypes.data, kw.ctypes.data, c.ctypes.data,
                                 s.ctypes.data, float(enc.eps), q.ctypes.data, k.ctypes.data, v.ctypes.data)
            enc.timings["qkv_norm_rope"] += time.perf_counter() - t
            return q, k, v
        enc.qkv_hook = qkv_native
    attn = None
    if a.attention == "npu-local":
        if not idle():
            print("NPU not idle: refusing (a foreign context is contention, not a finding)")
            return 3
        import attn_array
        attn = attn_array.Session(a.attn_build, "doc2048", a.attn_pack_dll)
        host_attention = enc._attention

        def npu_local(q, k, v, layer, segs):   # q [R, H, 256], k/v [R, KV, 256]; one dispatch per document
            if layer["kind"] != "sliding_attention":
                return host_attention(q, k, v, layer, segs)
            t = time.perf_counter()
            out = np.concatenate([attn.run(q[x:y], k[x:y], v[x:y], enc.window).reshape(y - x, -1) for x, y in segs])
            enc.timings["attn_npu"] = enc.timings.get("attn_npu", 0.0) + time.perf_counter() - t
            for key, val in attn.t.items():
                enc.timings[key] = enc.timings.get(key, 0.0) + val
            attn.t = dict.fromkeys(attn.t, 0.0)
            return out
        enc._attention = npu_local
    print(f"weights loaded in {time.perf_counter() - t:.1f} s; {len(enc.shapes())} weight products per document", flush=True)
    npu = None
    if a.backend == "npu":
        if attn is None and not idle():   # with npu-local, idle was checked before the attention context opened
            print("NPU not idle: refusing (a foreign context is contention, not a finding)")
            return 3
        rounder = None
        if a.round_dll:
            from native import Rounder
            rounder = Rounder(a.round_dll)
        if a.npu_w8 and not (a.w4 == 64 and a.wbits == 8):
            raise SystemExit("--npu-w8 needs --w4 64 --wbits 8 (the device's group is its 64-row K tile)")
        npu = Npu(enc, a.builds, a.ms, rounder, a.npu_geglu, w8_raw or None, b_resident=a.npu_b_resident)
        if a.npu_geglu:
            enc.geglu_hook = lambda gu: gu   # gate|up already returned ff from the device
    matmul = npu.matmul if npu else None
    if a.a8 and not npu:
        import w4 as _w4

        def matmul(x, name, layer):   # W8A8 emulation: int8 per-token activations against the (dequantized) weights
            q, s = _w4.quantize_rows(x)
            return (q.astype(np.float32) * s[:, None]) @ enc.weights[(name, layer)]
    bf16 = a.backend == "npu" or a.bf16_inputs
    emulation = a.backend == "cpu" and (a.bf16_inputs or a.attention == "blockwise" or a.geglu != "host" or bool(a.w4)
                                        or a.a8)
    try:
        embeds, times, phases = [None] * len(docs), {}, {}
        groups = []   # consecutive same-length documents, up to --batch and (npu) the largest compiled M in rows
        rows = max(a.ms) if npu else float("inf")
        for j, d in enumerate(docs):
            g = groups[-1] if groups else None
            if (g and docs[g[0]]["length"] == d["length"] and len(g) < a.batch
                    and (len(g) + 1) * d["length"] <= rows):
                groups[-1].append(j)
            else:
                groups.append([j])
        warmed = set()
        for g in groups:
            L, n = docs[g[0]]["length"], len(g)
            ids = [docs[j]["ids"] for j in g]
            if (L, n) not in warmed:
                enc.embed_documents(ids, matmul, bf16)
                warmed.add((L, n))
            enc.timings.clear()
            if npu:
                npu.t = dict.fromkeys(npu.t, 0.0)
            t0 = time.perf_counter()
            for j, v in zip(g, enc.embed_documents(ids, matmul, bf16)):
                embeds[j] = v
            times.setdefault(L, []).append((time.perf_counter() - t0) / n)
            ph = dict(enc.timings) | (dict(npu.t) if npu else {})
            for k, v in ph.items():
                phases.setdefault(L, {}).setdefault(k, []).append(v / n)
        e = np.stack(embeds)
        cos = np.sum(e * ref, axis=1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(ref, axis=1))
        lengths = np.array([d["length"] for d in docs])

        def nn(x):
            s = x @ x.T
            np.fill_diagonal(s, -np.inf)
            return s.argmax(axis=1)
        nn_kept = float(np.mean(nn(e) == nn(ref)))
        nn_quality = None
        if a.quality_reference:
            nn_quality = float(np.mean(nn(e) == nn(np.load(a.quality_reference)["embeddings"])))
        if a.save_embeddings:
            np.savez_compressed(a.save_embeddings, embeddings=e)
        strict = a.backend == "cpu" and not emulation and a.attention == "exact"
        threshold = 0.99999 if strict else 0.9999
        passed = bool(cos.min() >= threshold and (strict or nn_kept == 1.0) and nn_quality in (None, 1.0))
        rec = dict(backend=a.backend, docs=str(a.docs), reference=str(a.reference), batch=a.batch, round_dll=bool(a.round_dll), attention=a.attention, geglu=a.geglu, npu_geglu=a.npu_geglu, npu_w8=a.npu_w8, npu_b_resident=a.npu_b_resident, native_qkv=bool(a.native_qkv), native_rowops=bool(a.native_rowops), w4=a.w4, wchannel=a.wchannel, a8=a.a8, wbits=a.wbits, wclip=a.wclip, k_block=a.k_block, exp=a.exp, tau=a.tau,
                   bf16_inputs=a.bf16_inputs, passed=passed,
                   gate=dict(cos_min=threshold, nn_kept=None if strict else 1.0),
                   cos_min=float(cos.min()), cos_mean=float(cos.mean()), nn_kept=nn_kept,
                   quality_reference=str(a.quality_reference) if a.quality_reference else None, nn_kept_quality=nn_quality,
                   per_length={})
        print(f"GATE {a.backend}: cos min {cos.min():.7f} (>= {threshold}), mean {cos.mean():.7f}, NN kept {nn_kept:.1%}"
              + (f", NN kept vs quality reference {nn_quality:.1%}" if nn_quality is not None else "") + ": "
              f"{'PASS' if passed else 'FAIL'}", flush=True)
        for L in sorted(times):
            med = statistics.median(times[L]) * 1000
            ph = {k: statistics.median(v) * 1000 for k, v in phases[L].items()}
            rec["per_length"][str(L)] = dict(batches=len(times[L]), median_ms_per_doc=med,
                                             cos_min=float(cos[lengths == L].min()), phases_ms_per_doc=ph)
            print(f"L={L:5d}: {len(times[L])} batches, median {med:8.1f} ms/doc; cos min {cos[lengths == L].min():.7f}; "
                  + ", ".join(f"{k} {v:.1f}" for k, v in sorted(ph.items(), key=lambda kv: -kv[1])), flush=True)
        a.out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(a.out, flush=True)
        if a.queue and passed:
            serve_queue(a.queue, enc, matmul, bf16, rows if npu else 2048)
        if a.loop_go and passed:
            loop_groups = [[docs[j]["ids"] for j in g] for g in groups if docs[g[0]]["length"] == a.loop_length]
            Path(str(a.loop_go) + ".ready").write_text(f"{time.time()}\n", encoding="utf-8")
            print("LOOP_READY", flush=True)
            while not a.loop_go.exists():
                time.sleep(0.1)
            time.sleep(0.2)
            start, stop = map(float, a.loop_go.read_text(encoding="utf-8").split())
            while time.time() < start:
                time.sleep(0.005)
            spans, i = [], 0
            while time.time() < stop:
                ids = loop_groups[i % len(loop_groups)]
                t0 = time.time()
                enc.embed_documents(ids, matmul, bf16)
                spans.append([t0, time.time(), len(ids)])
                i += 1
            Path(str(a.out) + ".loop.json").write_text(json.dumps(dict(
                device="npu", length=a.loop_length, start=start, stop=stop, spans=spans)) + "\n", encoding="utf-8")
            print(f"LOOP_DONE {len(spans)} groups", flush=True)
        return 0 if passed else 1
    finally:
        if attn:
            attn.close()
        if npu:
            npu.close()
        if npu or attn:
            print(f"idle after: {idle()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
