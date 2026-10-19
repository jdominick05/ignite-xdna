"""SmolLM2-135M decode with the rounding points the NPU decoder will have: the reference and, later, the oracle.

Every value that crosses from one op to the next is bf16, and each op is built only from primitives whose bits are
known on this core: the mmul MAC (``engine_bf16_emulator.mac``, exp_sum), bf16 x bf16 products (exact in fp32),
round-to-nearest-even to bf16, and tables indexed by bf16 bits, computed on the host with correct rounding. So the
semantics below are what each kernel must reproduce bit for bit.

Rounding points, per layer at position p:
  RMSNorm   ss = sum(h*h) (MAC, fp32); r = bf16(1/sqrt(ss/576 + eps)); y = bf16(gamma * bf16(h * r))
  qkv       projection (fp32 MAC over K, one rounding)
  RoPE      bf16(q*cos + rotate_half(q)*sin), the two products exact and summed once; cos/sin are HF's bf16 tables
  KV        k (after RoPE) and v appended at row p
  scores    s_j = bf16(MAC(q, k_j)) / 8 (a power of two, exact); d_j = bf16(s_j - max s); e_j = EXP[d_j], a table of
            correctly rounded exp over bf16; z = bf16(sum e_j); out = bf16(bf16(sum e_j v_j) * RECIP[z])
  o, res    projection; h = bf16(h + o), one rounding of the exact sum
  MLP       gate, up projections; m = bf16(SILU[gate] * up), SILU a correctly rounded table; down; h = bf16(h + d)
Then the final RMSNorm, the tied head (projection) and argmax (lowest index on a tie).

``exact=True`` runs every MAC through the exp_sum emulator. The default computes each MAC-accumulated sum in float64
and rounds once, which differs from exp_sum only in the last bit of rare sums: right for judging quality, wrong as a
device oracle.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from ignite_xdna.compiler import engine_bf16_emulator as em

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
LAYERS, HIDDEN, HEADS, KV_HEADS, HEAD_DIM, MLP, VOCAB = 30, 576, 9, 3, 64, 1536, 49152
EPS, THETA = 1e-5, 100000.0


def model_file() -> Path:
    """The cached model.safetensors (fetch it once with ``hf download HuggingFaceTB/SmolLM2-135M-Instruct``)."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    found = sorted((home / "hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots").glob("*/model.safetensors"))
    if len(found) != 1:
        raise SystemExit(f"expected one cached {MODEL} snapshot under {home}, found {len(found)}")
    return found[0]


def safetensors(path) -> Dict[str, np.ndarray]:
    """name -> uint16 array of bf16 bits, read with numpy alone (the iron env has no safetensors package)."""
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    head = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, meta in head.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] != "BF16":
            raise ValueError(f"{name}: {meta['dtype']}, expected BF16")
        a, b = meta["data_offsets"]
        out[name] = np.frombuffer(raw, np.uint16, (b - a) // 2, base + a).reshape(meta["shape"])
    return out


def bf16(a) -> np.ndarray:
    """Round to bf16 (RNE), kept in float32."""
    return em.to_bf16(np.asarray(a, dtype=np.float32))


def bf16_from64(a) -> np.ndarray:
    """Round a float64 value to bf16 once (no float32 step in between): correct rounding of an exact sum."""
    a = np.asarray(a, dtype=np.float64)
    f = a.astype(np.float32)
    # float64 -> float32 rounds; if that rounding lands on a bf16 tie it could double-round. Resolve ties by the
    # residual's sign: nudge the float32 one ulp toward the true value before the bf16 rounding.
    resid = a - f.astype(np.float64)
    u = f.view(np.uint32)
    tie = (u & 0xFFFF) == 0x8000
    nudge = np.where(tie & (resid > 0), u + np.where(f >= 0, 1, -1), np.where(tie & (resid < 0), u - np.where(f >= 0, 1, -1), u))
    return bf16(nudge.astype(np.uint32).view(np.float32))


def _table(fn) -> np.ndarray:
    """fn over every bf16 bit pattern (float64), correctly rounded back to bf16: a 65536-entry table by bits.
    NaN and infinite inputs map to themselves; overflow to infinity and underflow are kept as they round."""
    with np.errstate(all="ignore"):
        vals = em.from_bf16_bits(np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)).astype(np.float64)
        out = np.where(np.isfinite(vals), fn(vals), vals)
        return bf16_from64(out)


EXP = _table(np.exp)
SILU = _table(lambda x: x / (1.0 + np.exp(-x)))
RECIP = _table(lambda x: 1.0 / x)


def lut(table: np.ndarray, x: np.ndarray) -> np.ndarray:
    return table[em.bf16_bits(bf16(x)).astype(np.int64)]


def dot(a: np.ndarray, b: np.ndarray, exact: bool) -> np.ndarray:
    """Sums of bf16 products over the last axis: [..., K] x [K] or [..., K] x [..., K] -> fp32."""
    if not exact:
        return (a.astype(np.float64) * b.astype(np.float64)).sum(-1)
    a, b = np.broadcast_arrays(np.asarray(a, np.float32), np.asarray(b, np.float32))
    lead, k = a.shape[:-1], a.shape[-1]
    pad = (-k) % 8
    a = np.pad(a.reshape(-1, k), ((0, 0), (0, pad)))
    b = np.pad(b.reshape(-1, k), ((0, 0), (0, pad)))
    acc = np.zeros((a.shape[0], 1, 1, 1), np.float32)
    for s in range(0, a.shape[1], 8):
        # one lane per row: a row's 8 inputs times its own 8 weights, as an mmul with a single output column
        acc = np.stack([em.mac(acc[i:i + 1], a[i, s:s + 8].reshape(1, 1, 8), b[i, s:s + 8].reshape(1, 8, 1),
                               em.MAC_MODEL)[0] for i in range(a.shape[0])])
    return acc.reshape(lead).astype(np.float64)


def matvec(w: np.ndarray, x: np.ndarray, exact: bool) -> np.ndarray:
    """bf16 W [N][K] (float32) times bf16 x -> bf16 y, one rounding: the GEMV."""
    if exact:
        from kernels.llm_gemv import oracle
        return em.from_bf16_bits(oracle.gemv(em.bf16_bits(w), em.bf16_bits(x)))
    return bf16_from64(w @ x.astype(np.float64))


def rmsnorm(h, gamma, exact):
    ss = dot(h, h, exact)
    r = bf16_from64(1.0 / np.sqrt(np.float64(np.float32(ss)) / HIDDEN + EPS))
    return bf16(gamma * bf16(h * r))


def rope_tables(positions: int):
    """HF's cos/sin for the non-interleaved rotation, computed in float32 and cast to bf16, [positions][64]."""
    inv = (1.0 / (THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.int64).astype(np.float32) / HEAD_DIM))).astype(np.float32)
    f = np.arange(positions, dtype=np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([f, f], axis=1).astype(np.float32)
    return bf16(np.cos(emb)), bf16(np.sin(emb))


def rope(x, cos, sin):
    """x [heads][64] bf16 -> bf16(x*cos + rotate_half(x)*sin), the two exact products summed once."""
    rot = np.concatenate([-x[:, HEAD_DIM // 2:], x[:, :HEAD_DIM // 2]], axis=1)
    return bf16_from64(x.astype(np.float64) * cos + rot.astype(np.float64) * sin)


@dataclass
class Weights:
    embed: np.ndarray                       # [V][576] bf16 as float32; also the head
    layers: List[Dict[str, np.ndarray]]
    norm: np.ndarray

    @classmethod
    def load(cls, tensors: Dict[str, np.ndarray]):
        # float64 holds bf16 exactly, and keeps every fast-mode matvec from re-widening 269 MB of weights per token
        f = lambda n: em.from_bf16_bits(tensors[n]).astype(np.float64)  # noqa: E731
        layers = []
        for i in range(LAYERS):
            p = f"model.layers.{i}."
            layers.append(dict(
                n1=f(p + "input_layernorm.weight"), n2=f(p + "post_attention_layernorm.weight"),
                qkv=np.concatenate([f(p + "self_attn.q_proj.weight"), f(p + "self_attn.k_proj.weight"),
                                    f(p + "self_attn.v_proj.weight")]),
                o=f(p + "self_attn.o_proj.weight"),
                gu=np.concatenate([f(p + "mlp.gate_proj.weight"), f(p + "mlp.up_proj.weight")]),
                down=f(p + "mlp.down_proj.weight")))
        return cls(f("model.embed_tokens.weight"), layers, f("model.norm.weight"))


@dataclass
class Decoder:
    w: Weights
    max_positions: int = 2048
    exact: bool = False
    k: List[list] = field(default_factory=lambda: [[] for _ in range(LAYERS)])
    v: List[list] = field(default_factory=lambda: [[] for _ in range(LAYERS)])

    def __post_init__(self):
        self.cos, self.sin = rope_tables(self.max_positions)

    def attention(self, i, q, p):
        K = np.stack(self.k[i])                         # [p+1][3][64]
        V = np.stack(self.v[i])
        out = np.zeros((HEADS, HEAD_DIM), np.float32)
        for hd in range(HEADS):
            kv = hd // (HEADS // KV_HEADS)
            s = bf16(dot(K[:, kv], q[hd], self.exact)) / 8.0      # exact: /8 is a power of two
            d = bf16(s - s.max())
            e = lut(EXP, d)
            z = bf16_from64(dot(e, np.ones_like(e), self.exact) if self.exact else e.astype(np.float64).sum())
            acc = bf16_from64(dot(V[:, kv].T, e, self.exact))
            out[hd] = bf16(acc * lut(RECIP, z))
        return out.reshape(-1)

    def step(self, token: int, p: int) -> np.ndarray:
        """One decode step at position p: returns the bf16 logits [V]."""
        h = self.w.embed[token].copy()
        cos, sin = self.cos[p], self.sin[p]
        for i, L in enumerate(self.w.layers):
            a = rmsnorm(h, L["n1"], self.exact)
            qkv = matvec(L["qkv"], a, self.exact)
            q = rope(qkv[:HIDDEN].reshape(HEADS, HEAD_DIM), cos, sin)
            k = rope(qkv[HIDDEN:HIDDEN + KV_HEADS * HEAD_DIM].reshape(KV_HEADS, HEAD_DIM), cos, sin)
            v = qkv[HIDDEN + KV_HEADS * HEAD_DIM:].reshape(KV_HEADS, HEAD_DIM)
            self.k[i].append(k)
            self.v[i].append(v)
            o = matvec(L["o"], self.attention(i, q, p), self.exact)
            h = bf16_from64(h.astype(np.float64) + o)
            b = rmsnorm(h, L["n2"], self.exact)
            gu = matvec(L["gu"], b, self.exact)
            m = bf16(lut(SILU, gu[:MLP]) * gu[MLP:])
            h = bf16_from64(h.astype(np.float64) + matvec(L["down"], m, self.exact))
        return matvec(self.w.embed, rmsnorm(h, self.w.norm, self.exact), self.exact)

    def generate(self, prompt: List[int], new_tokens: int, eos: int | None = None) -> List[int]:
        out = []
        logits = None
        for p, t in enumerate(prompt):
            logits = self.step(t, p)
        for _ in range(new_tokens):
            t = int(np.argmax(logits))
            out.append(t)
            if eos is not None and t == eos:
                break
            logits = self.step(t, len(prompt) + len(out) - 1)
        return out
