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
product's input to BF16 as the NPU GEMM does. Both are numerics checks: their timings mean nothing.
Timing: wall time per batch divided by its documents, median per length, after one untimed warm-up batch per
(length, batch size). Phases: ``gemm`` is the whole weight-product hook; inside it ``npu_pack`` (round to BF16
into the mapped A), ``npu_write`` (A sync), ``npu_call`` (submit + wait), ``npu_read`` (C sync) and ``npu_unpack``
(chunk concatenation). attention, norm, rope, gelu and embed are host NumPy fp32 (single-threaded, not a tuned CPU
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

    def __init__(self, enc, builds, ms):
        sys.path.insert(0, "C:/Xilinx/XRT/xrt_sdk/xrt/python")
        import pyxrt
        self.x, self.ms, self.builds = pyxrt, sorted(ms), builds
        self.to_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.from_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.dev = pyxrt.device(0)
        self.xb = pyxrt.xclbin(str(builds / "g_256_512_512.xclbin"))
        self.dev.register_xclbin(self.xb)
        self.ctx = pyxrt.hw_context(self.dev, self.xb.get_uuid())
        self.kern = pyxrt.kernel(self.ctx, self.xb.get_kernels()[0].get_name())
        self.t = dict(npu_pack=0.0, npu_write=0.0, npu_call=0.0, npu_read=0.0, npu_unpack=0.0)
        self.B, self.io = {}, {}
        for key, w in enc.weights.items():   # [K, N] fp32 holding exact BF16 values
            K, N = w.shape
            chunks = []
            for c in range(0, N, CHUNK):
                part = w[:, c:c + CHUNK]
                n = part.shape[1]
                bits = np.zeros((K, padded(n)), np.uint16)
                bits[:, :n] = bf16_bits(part)
                bo = self._bo(bits.nbytes)
                bo.write(bits.tobytes(), 0)
                bo.sync(self.to_dev)
                chunks.append((bo, n))
            self.B[key] = chunks

    def _bo(self, nbytes):
        return self.x.bo(self.dev, nbytes, self.x.bo.host_only, 0)

    def _io(self, M, K, Np):
        key = (M, K, Np)
        if key not in self.io:
            words = np.frombuffer((self.builds / f"g_{M}_{K}_{Np}.bin").read_bytes(), dtype=np.uint32)
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
        outs = []
        packed = None
        for bo_w, n in chunks:
            Np = padded(n)
            ib, nb, A, C, av, cv = self._io(M, K, Np)
            t0 = time.perf_counter()
            if packed is None:   # round once per product; every chunk shares the input
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cpu", "npu"], required=True)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--builds", type=Path, help="tools/embedgemma_gemm_probe.py compile --out DIR (npu)")
    ap.add_argument("--ms", type=int, nargs="+", default=[256, 512, 2048])
    ap.add_argument("--batch", type=int, default=1, help="consecutive same-length documents per row batch")
    ap.add_argument("--attention", choices=["exact", "blockwise"], default="exact")
    ap.add_argument("--k-block", type=int, default=32)
    ap.add_argument("--bf16-inputs", action="store_true", help="cpu: round weight-product inputs to BF16")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    docs = json.loads(a.docs.read_text(encoding="utf-8"))["docs"]
    ref = np.load(a.reference)["embeddings"]
    if len(ref) != len(docs):
        raise ValueError("reference and docs differ in count")
    t = time.perf_counter()
    enc = Encoder()
    if a.attention == "blockwise":
        import attention_oracle

        def blockwise(q, k, v, layer, segs):   # rows are independent, so one query block per document is exact
            return np.concatenate([attention_oracle.attention(q[x:y], k[x:y], v[x:y], layer["kind"], enc.window,
                                                              q_block=y - x, k_block=a.k_block) for x, y in segs])
        enc._attention = blockwise
    print(f"weights loaded in {time.perf_counter() - t:.1f} s; {len(enc.shapes())} weight products per document", flush=True)
    npu = None
    if a.backend == "npu":
        if not idle():
            print("NPU not idle: refusing (a foreign context is contention, not a finding)")
            return 3
        npu = Npu(enc, a.builds, a.ms)
    matmul = npu.matmul if npu else None
    bf16 = a.backend == "npu" or a.bf16_inputs
    emulation = a.backend == "cpu" and (a.bf16_inputs or a.attention != "exact")
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
        strict = a.backend == "cpu" and not emulation
        threshold = 0.99999 if strict else 0.9999
        passed = bool(cos.min() >= threshold and (strict or nn_kept == 1.0))
        rec = dict(backend=a.backend, docs=str(a.docs), reference=str(a.reference), batch=a.batch, attention=a.attention, k_block=a.k_block,
                   bf16_inputs=a.bf16_inputs, passed=passed,
                   gate=dict(cos_min=threshold, nn_kept=None if strict else 1.0),
                   cos_min=float(cos.min()), cos_mean=float(cos.mean()), nn_kept=nn_kept,
                   per_length={})
        print(f"GATE {a.backend}: cos min {cos.min():.7f} (>= {threshold}), mean {cos.mean():.7f}, NN kept {nn_kept:.1%}: "
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
        return 0 if passed else 1
    finally:
        if npu:
            npu.close()
            print(f"idle after: {idle()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
