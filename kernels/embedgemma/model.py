# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""EmbeddingGemma 2 text encoder in NumPy, with every weight product behind one pluggable ``matmul``.

An independent reference of ``transformers``' ``EmbeddingGemma2TextModel`` (5.19) written from its source: scaled
token embedding; projection-only per-layer inputs; 24 bidirectional layers of RMSNorm -> attention (q/k RMSNorm,
scale-free v norm, RoPE, scaling 1.0, local layers masked to |i - j| <= sliding_window) -> RMSNorm -> residual,
RMSNorm -> GELU-tanh gated MLP -> RMSNorm -> residual, the per-layer-input block, then ``layer_scalar``; final
RMSNorm, the 512 -> 768 projection, mean pooling over every token and L2 normalisation. Non-GEMM arithmetic is
fp32. Weights are read straight from the checkpoint's safetensors (BF16, widened exactly to fp32).

Weight products are grouped as the NPU runs them: ``qkv`` (q|k|v concatenated), ``o``, ``gate_up`` (gate|up),
``down``, ``ple_gate``, ``ple_proj``, ``ple_model`` (the per-layer-input projection) and ``out``. ``matmul(x, name,
layer)`` receives fp32 activations [rows, K] and must return fp32 [rows, N]; the default is fp32 NumPy. With
``bf16_inputs=True`` the activations are rounded to BF16 (nearest-even) before every weight product, which is what
a BF16-in GEMM consumes; the products themselves stay whatever the backend computes.

``timings`` accumulates wall time per phase (seconds) so a caller can attribute an end-to-end run by op type.
"""
import json
import math
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SNAPSHOT = Path(r"L:\huggingface\hub\models--google--embeddinggemma-2\snapshots\914f7f89142e33e77833254d9c9b90c3cef7303b")
PREFIX = "language_model."


def bf16_round(x: np.ndarray) -> np.ndarray:
    """fp32 -> nearest-even BF16, returned as fp32 (finite inputs)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 -> nearest-even BF16 bits (uint16)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def read_safetensors(path: Path, prefix: str) -> dict[str, np.ndarray]:
    """BF16 tensors whose name starts with ``prefix``, widened to fp32 (exact)."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for name, meta in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0] if kv[0] != "__metadata__" else -1):
            if name == "__metadata__" or not name.startswith(prefix):
                continue
            if meta["dtype"] != "BF16":
                raise ValueError(f"{name}: {meta['dtype']}, expected BF16")
            a, b = meta["data_offsets"]
            f.seek(base + a)
            raw = np.frombuffer(f.read(b - a), dtype=np.uint16)
            out[name[len(prefix):]] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
    return out


class Encoder:
    def __init__(self, snapshot: Path = SNAPSHOT):
        cfg = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))["text_config"]
        self.cfg = cfg
        w = read_safetensors(snapshot / "model.safetensors", PREFIX)
        self.d, self.H, self.nl = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
        self.eps, self.window, self.ple_dim = cfg["rms_norm_eps"], cfg["sliding_window"], cfg["hidden_size_per_layer_input"]
        self.embed = w["embed_tokens.weight"]
        self.final_norm = w["norm.weight"]
        # Weight products as [K, N] matrices, products sharing an input concatenated along N.
        self.weights = {("ple_model", -1): w["ple.per_layer_model_projection.weight"].T.copy(),
                        ("out", -1): w["embedding_projection.weight"].T.copy()}
        self.ple_norm = w["ple.per_layer_projection_norm.weight"]
        self.layers = []
        for i, kind in enumerate(cfg["layer_types"]):
            pc = cfg["per_layer_config"].get(f"{i:02d}", {})
            hd, kv = pc.get("head_dim", cfg["head_dim"]), pc.get("num_key_value_heads", cfg["num_key_value_heads"])
            p = f"layers.{i}."
            a = p + "self_attn."
            self.weights[("qkv", i)] = np.concatenate([w[a + n + "_proj.weight"].T for n in "qkv"], axis=1)
            self.weights[("o", i)] = w[a + "o_proj.weight"].T.copy()
            self.weights[("gate_up", i)] = np.concatenate([w[p + "mlp.gate_proj.weight"].T, w[p + "mlp.up_proj.weight"].T], axis=1)
            self.weights[("down", i)] = w[p + "mlp.down_proj.weight"].T.copy()
            self.weights[("ple_gate", i)] = w[p + "ple_block.per_layer_input_gate.weight"].T.copy()
            self.weights[("ple_proj", i)] = w[p + "ple_block.per_layer_projection.weight"].T.copy()
            theta = cfg["rope_parameters"][kind]["rope_theta"]
            self.layers.append(dict(
                kind=kind, hd=hd, kv=kv, theta=theta,
                inv_freq=(1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))).astype(np.float32),
                input_norm=w[p + "input_layernorm.weight"], post_attn_norm=w[p + "post_attention_layernorm.weight"],
                pre_ff_norm=w[p + "pre_feedforward_layernorm.weight"], post_ff_norm=w[p + "post_feedforward_layernorm.weight"],
                q_norm=w[a + "q_norm.weight"], k_norm=w[a + "k_norm.weight"],
                ple_norm=w[p + "ple_block.post_per_layer_input_norm.weight"], scalar=float(w[p + "layer_scalar"][0])))
        self.timings = defaultdict(float)

    def shapes(self) -> list[tuple[str, int, int, int]]:
        """(name, layer, K, N) of every weight product, in forward order."""
        seq = [("ple_model", -1, *self.weights[("ple_model", -1)].shape)]
        for i in range(self.nl):
            for n in ("qkv", "o", "gate_up", "down", "ple_gate", "ple_proj"):
                seq.append((n, i, *self.weights[(n, i)].shape))
        return seq + [("out", -1, *self.weights[("out", -1)].shape)]

    # -- ops -------------------------------------------------------------------------------------------------------
    def _rms(self, x, w=None):
        t = time.perf_counter()
        y = x * (np.mean(x * x, axis=-1, keepdims=True) + self.eps) ** -0.5
        if w is not None:
            y = y * w
        self.timings["norm"] += time.perf_counter() - t
        return y.astype(np.float32)

    def _gelu(self, x):
        t = time.perf_counter()
        y = 0.5 * x * (1.0 + np.tanh(np.float32(math.sqrt(2.0 / math.pi)) * (x + np.float32(0.044715) * x * x * x)))
        self.timings["gelu"] += time.perf_counter() - t
        return y.astype(np.float32)

    def _mm(self, x, name, layer, matmul, bf16_inputs):
        t = time.perf_counter()
        if bf16_inputs:
            x = bf16_round(x)
        self.timings["round_in"] += time.perf_counter() - t
        t = time.perf_counter()
        y = matmul(x, name, layer) if matmul else x @ self.weights[(name, layer)]
        self.timings["gemm"] += time.perf_counter() - t
        return y

    def _attention(self, q, k, v, layer, L):
        """q [L, H, hd], k/v [L, kv, hd] (normed, rotated): fp32 softmax attention -> [L, H*hd]."""
        t = time.perf_counter()
        H, kv, hd = self.H, layer["kv"], layer["hd"]
        rep = H // kv
        out = np.empty((L, H, hd), np.float32)
        if layer["kind"] == "sliding_attention":
            i = np.arange(L)
            masked = np.abs(i[:, None] - i[None, :]) > self.window
        else:
            masked = None
        for h in range(H):
            s = q[:, h, :] @ k[:, h // rep, :].T          # scaling 1.0
            if masked is not None:
                s[masked] = -np.inf
            s -= s.max(axis=1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(axis=1, keepdims=True)
            out[:, h, :] = s @ v[:, h // rep, :]
        self.timings["attention"] += time.perf_counter() - t
        return out.reshape(L, H * hd)

    def _rope(self, x, cos, sin):
        t = time.perf_counter()
        half = x.shape[-1] // 2
        rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        y = (x * cos[:, None, :] + rot * sin[:, None, :]).astype(np.float32)
        self.timings["rope"] += time.perf_counter() - t
        return y

    # -- forward ---------------------------------------------------------------------------------------------------
    def embed_document(self, ids, matmul=None, bf16_inputs=False) -> np.ndarray:
        ids = np.asarray(ids)
        L, d, H = len(ids), self.d, self.H
        t = time.perf_counter()
        x = (self.embed[ids] * np.float32(math.sqrt(d))).astype(np.float32)
        self.timings["embed"] += time.perf_counter() - t
        ple = self._mm(x, "ple_model", -1, matmul, bf16_inputs) * np.float32(d ** -0.5)
        ple = self._rms(ple.reshape(L, self.nl, self.ple_dim), self.ple_norm)
        pos = np.arange(L, dtype=np.float32)
        rope = {}
        for i, lay in enumerate(self.layers):
            hd, kv = lay["hd"], lay["kv"]
            if lay["kind"] not in rope:
                f = pos[:, None] * lay["inv_freq"][None, :]
                emb = np.concatenate([f, f], axis=-1)
                rope[lay["kind"]] = (np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32))
            cos, sin = rope[lay["kind"]]
            res = x
            hs = self._rms(x, lay["input_norm"])
            qkv = self._mm(hs, "qkv", i, matmul, bf16_inputs)
            q = qkv[:, :H * hd].reshape(L, H, hd)
            k = qkv[:, H * hd:(H + kv) * hd].reshape(L, kv, hd)
            v = qkv[:, (H + kv) * hd:].reshape(L, kv, hd)
            q = self._rope(self._rms(q, lay["q_norm"]), cos, sin)
            k = self._rope(self._rms(k, lay["k_norm"]), cos, sin)
            v = self._rms(v)
            a = self._mm(self._attention(q, k, v, lay, L), "o", i, matmul, bf16_inputs)
            x = res + self._rms(a, lay["post_attn_norm"])
            res = x
            gu = self._mm(self._rms(x, lay["pre_ff_norm"]), "gate_up", i, matmul, bf16_inputs)
            ff = self._gelu(gu[:, :gu.shape[1] // 2]) * gu[:, gu.shape[1] // 2:]
            x = res + self._rms(self._mm(ff, "down", i, matmul, bf16_inputs), lay["post_ff_norm"])
            g = self._gelu(self._mm(x, "ple_gate", i, matmul, bf16_inputs)) * ple[:, i, :]
            x = x + self._rms(self._mm(g, "ple_proj", i, matmul, bf16_inputs), lay["ple_norm"])
            x = (x * np.float32(lay["scalar"])).astype(np.float32)
        h = self._mm(self._rms(x, self.final_norm), "out", -1, matmul, bf16_inputs).astype(np.float64)
        v = h.mean(axis=0)
        return v / np.linalg.norm(v)
