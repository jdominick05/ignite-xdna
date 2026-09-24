#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM study, Gemma 3 4B, pre-registration (c): decode on the CPU and DirectML with ONNX Runtime
GenAI 0.11.2. This file holds (c)'s build: five models that read the release's exact weights.

The release is google/gemma-3-4b-it-qat-q4_0-gguf: 238 q4_0 linears and a tied F16 head. The
arms differ only in compute and head format:
  C0-H4   CPU, MatMulNBits accuracy_level 0 (fp32 compute), int4 head (RTN from the release head)
  C4-H4   CPU, accuracy_level 4 (int8 compute, the builder's CPU default), int4 head
  D-H4    DirectML (fp16 io and compute), int4 head
  C0-H16  CPU, accuracy_level 0, the release's F16 head held in fp32
  D-H16   DirectML, the release's F16 head
How the build gets there (each step logged and verified):
  - The text-only copy: -qat-q4_0-unquantized (7c0881d8) as a Gemma3ForCausalLM folder, with the
    release's F16 head (upcast to F32) as its embedding and vocab_size 262,144 (the GGUF's rows).
  - The builder runs with its model load wrapped: every linear carries the GGUF's q4_0 codes and
    scales as a pre-quantized MatMulNBits (the builder's own qweight path), so its float linears
    never enter the graph. The head is left to the builder: RTN int4 (H4) or excluded (H16).
  - Every weight-bearing initializer is then read back and compared with the release by value;
    two 0.11.2 constants are corrected and verified (F1 the global RoPE's linear factor 8, F2 the
    embedding scale sqrt(2560)).
  - B3 DirectML placement, a fidelity gate (C0-H16 against transformers' fp32 Gemma 3 with the
    release's weights, streamed layer by layer), a smoke, the thread-affinity read-back, pins.

    python tools/gemma_decode.py build [--steps a,b,...]   # the build (resnet_env17); all steps by default
    python tools/gemma_decode.py builder --arm ARM         # one builder run (spawned by build)
    python tools/gemma_decode.py placement ARM             # B3: a DirectML session, verbose placement (spawned)
    python tools/gemma_decode.py selftest                  # small CPU-only checks, no model files
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gemma_compress as gc  # noqa: E402

# ---------------------------------------------------------------- pins (the Hub's metadata)

CK_REPO, CK_REV = "google/gemma-3-4b-it-qat-q4_0-unquantized", "7c0881d809c27a8356951b1abbf5be1b827e875c"
CK_LFS = {"model-00001-of-00002.safetensors": (4961251752, "51e4642dc4d431d53fc2030993e00e6becd3ef0d852bf6d00235caa0c453eb4e"),
          "model-00002-of-00002.safetensors": (3639026128, "49e1429f3bb29eded09f33029a5a09c31660cc22766b9bd29ecfd1451dbcba1c"),
          "tokenizer.json": (33384570, "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c"),
          "tokenizer.model": (4689074, "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c")}
CK_BLOB = {"config.json": (1610, "ac44fe0dc956cf40ee4466e48b8dd1346c282ba4"),
           "generation_config.json": (173, "3d589876faad712a9bfb6d3a98073e0f4ab5f75c"),
           "added_tokens.json": (35, "e17bde03d42feda32d1abfca6d3b598b9a020df7"),
           "special_tokens_map.json": (662, "1a6193244714d3d78be48666cb02cdbfac62ad86"),
           "tokenizer_config.json": (1157001, "72c9f6bc082907666631052305438b5e7db2a8e2"),
           "chat_template.json": (1615, "719b0cd0d7a373a400b0c119ee0e051f41ea88d9"),
           "model.safetensors.index.json": (90558, "4b95241f208f06d324d17c9675568ec58dafd9fb")}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "generation_config.json", "chat_template.json")

# ---------------------------------------------------------------- shapes and places

HIDDEN, VOCAB, LAYERS, HEAD_DIM = 2560, 262_144, 34, 256
CK_VOCAB = 262_208                       # the checkpoint's rows: the GGUF's 262,144 plus 64 (padding and the image token)
WORK = ROOT / "scratch" / "llm" / "gemma"
TEXT = WORK / "text_only"
ONNX = WORK / "onnx"
CACHE = WORK / "builder_cache"
ARMS = {"C0-H4": ("cpu", 0, False), "C4-H4": ("cpu", 4, False), "D-H4": ("dml", None, False),
        "C0-H16": ("cpu", 0, True), "D-H16": ("dml", None, True)}
GG_PROJ = {"attn_q": ("self_attn", "q_proj"), "attn_k": ("self_attn", "k_proj"), "attn_v": ("self_attn", "v_proj"),
           "attn_output": ("self_attn", "o_proj"), "ffn_gate": ("mlp", "gate_proj"), "ffn_up": ("mlp", "up_proj"),
           "ffn_down": ("mlp", "down_proj")}
ONNX_PROJ = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output",
             "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}
# the GGUF's F32 norms and the checkpoint's names (HF adds 1 in the forward; llama.cpp stores 1 + w)
GG_NORM = {"attn_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm",
           "ffn_norm": "pre_feedforward_layernorm", "post_ffw_norm": "post_feedforward_layernorm",
           "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}
FID_LENGTHS = (64, 1100)                 # 1100 > the local layers' 1024-token window
FID_KL_MAX = 1e-4                        # nats, max over positions: above it the build is wrong
FID_SEED = 20260924
FID_MIN_FREE_GB = 12                     # the reference (~5 GB streamed) plus the C0-H16 session (~7.5 GB), DERIVED
SMOKE_PROMPT = "The capital of France is"
SMOKE_TOKENS = 16
BOS = 2                              # Gemma 3's <bos>, config.json's bos_token_id
PIN_CPUS = [0, 2, 4, 6, 8, 10, 12, 14]   # one logical CPU per physical core; the caller takes the first


def sha256(path: Path) -> str:
    return gc.sha256_file(path)


def blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def say(tag: str, obj) -> None:
    print(f"{tag} " + json.dumps(obj), flush=True)


# ---------------------------------------------------------------- q4_0 <-> MatMulNBits

def pack_ort(codes: np.ndarray) -> np.ndarray:
    """codes [N, kb, 32] (0..15, element order along K) -> MatMulNBits B [N, kb, 16]: consecutive
    elements share a byte, the even one in the low nibble."""
    c = codes.astype(np.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)


def unpack_ort(b: np.ndarray) -> np.ndarray:
    """MatMulNBits B [N, kb, 16] -> codes [N, kb, 32]."""
    out = np.empty(b.shape[:-1] + (32,), dtype=np.uint8)
    out[..., 0::2] = b & 0x0F
    out[..., 1::2] = b >> 4
    return out


def gguf_linear(mm, t):
    """A q4_0 GGUF tensor as (codes [N, kb, 32] uint8, d [N * kb] f16 bits); ggml dims are [K, N]."""
    K, N = t["dims"]
    nb = t["n"] // 32
    d, codes = gc.q4_0_split(np.asarray(mm[t["start"]:t["start"] + nb * 18]).reshape(nb, 18))
    return codes.reshape(N, K // 32, 32), d


def dequant(codes: np.ndarray, scales_f32: np.ndarray) -> np.ndarray:
    """s * (q - 8) in fp32, [N, kb, 32] -> [N, K]."""
    q = codes.astype(np.float32) - np.float32(8)
    return (q * scales_f32.reshape(codes.shape[0], codes.shape[1], 1)).reshape(codes.shape[0], -1)


# ---------------------------------------------------------------- fetch and the GGUF

def fetch() -> dict:
    """The GGUF (pinned in (a)) and every checkpoint file at 7c0881d8, verified; exits on a mismatch."""
    from huggingface_hub import hf_hub_download
    paths, rows, ok = {}, [], True
    g = gc.GGUF_PIN
    p = Path(hf_hub_download(g["repo"], g["file"], revision=g["revision"]))
    good = p.stat().st_size == g["size"] and sha256(p) == g["sha256"]
    ok &= good
    rows.append({"file": g["file"], "revision": g["revision"], "sha256": g["sha256"], "ok": good})
    paths["gguf"] = p
    for f, (size, sha) in CK_LFS.items():
        p = Path(hf_hub_download(CK_REPO, f, revision=CK_REV))
        got = sha256(p)
        good = p.stat().st_size == size and got == sha
        ok &= good
        rows.append({"file": f, "revision": CK_REV, "sha256": got, "ok": good})
        paths[f] = p
    for f, (size, blob) in CK_BLOB.items():
        p = Path(hf_hub_download(CK_REPO, f, revision=CK_REV))
        got = blob_sha1(p)
        good = p.stat().st_size == size and got == blob
        ok &= good
        rows.append({"file": f, "revision": CK_REV, "git_blob": got, "ok": good})
        paths[f] = p
    say("FETCH_JSON", {"ok": ok, "files": rows})
    if not ok:
        sys.exit("a fetched file differs from its pin; nothing is built")
    return paths


def release_paths() -> dict:
    """The GGUF's and the checkpoint shards' cache paths (fetched and verified by build's first step)."""
    from huggingface_hub import hf_hub_download
    g = gc.GGUF_PIN
    paths = {"gguf": Path(hf_hub_download(g["repo"], g["file"], revision=g["revision"]))}
    for f in CK_LFS:
        if f.endswith(".safetensors"):
            paths[f] = Path(hf_hub_download(CK_REPO, f, revision=CK_REV))
    return paths


def open_release(paths):
    mm, version, meta, tensors = gc.read_gguf(paths["gguf"])
    by = {t["name"]: t for t in tensors}
    st = gc.read_st([paths[f] for f in CK_LFS if f.endswith(".safetensors") and f in paths])
    return mm, by, st, meta


def gguf_f32(mm, t) -> np.ndarray:
    assert t["tname"] == "F32", t
    return np.asarray(mm[t["start"]:t["start"] + t["n"] * 4]).view(np.float32).copy()


def gguf_head(mm, by) -> np.ndarray:
    """The release's tied F16 head as [262144, 2560] float16 (a view on the file)."""
    t = by["token_embd.weight"]
    assert t["tname"] == "F16" and t["dims"] == [HIDDEN, VOCAB], t
    return np.asarray(mm[t["start"]:t["start"] + t["n"] * 2]).view(np.float16).reshape(VOCAB, HIDDEN)


def ck_f32(st, name) -> np.ndarray:
    return gc.bf16_to_f32(np.asarray(gc.st_u16(st[name])))


# ---------------------------------------------------------------- B1: the grid tests

def ort_q4(quantize_matmul_4bits, w: np.ndarray, dt):
    """ONNX Runtime's symmetric block-32 int4 quantizer (what the builder's default path calls) on
    w [N, K] in dtype dt: (B [N, K/32, 16] uint8, scales [N * K/32] in dt)."""
    N, K = w.shape
    wt = np.ascontiguousarray(w.T.astype(dt))                    # MatMul's B is [K, N]
    packed = np.zeros((N, K // 32, 16), dtype=np.uint8)
    scales = np.zeros(N * K // 32, dtype=dt)
    zp = np.zeros(N * ((K // 32 + 1) // 2), dtype=np.uint8)
    quantize_matmul_4bits(packed, wt, scales, zp, 32, N, K, True)
    return packed, scales


def b1(mm, by, st) -> dict:
    """q4_0 of the checkpoint by llama.cpp's reference rule (B1), and by ONNX Runtime's
    quantize_matmul_4bits in fp32 and fp16 (B1b, what the builder's default path would write),
    against the GGUF's codes and scales; the head and the norms."""
    from onnxruntime.capi._pybind_state import quantize_matmul_4bits
    tot = {"weights": 0, "blocks": 0, "rule_codes": 0, "rule_scales": 0, "rule_blocks": 0}
    for k in ("ort32", "ort16"):
        tot.update({f"{k}_codes": 0, f"{k}_scales": 0})
    t0 = time.time()
    lin = [t for t in by.values() if t["tname"] == "Q4_0"]
    for i, t in enumerate(lin):
        K, N = t["dims"]
        codes, d = gguf_linear(mm, t)
        w = ck_f32(st, gc.hf_name(t["name"])).reshape(N, K)
        d_rule, q_rule = gc.q4_0_rule(w.reshape(-1, 32))
        q_rule = q_rule.reshape(codes.shape).astype(np.uint8)
        d_gg = gc.f16_to_f32(d)
        sc_eq = d_rule == d_gg
        cd_eq = (q_rule == codes)
        row = {"name": t["name"], "n": t["n"], "blocks": len(d), "rule_codes": int(cd_eq.sum()),
               "rule_scales": int(sc_eq.sum()), "rule_blocks": int((cd_eq.all(axis=2).ravel() & sc_eq).sum())}
        for k, dt in (("ort32", np.float32), ("ort16", np.float16)):
            try:                                                 # B1b is informational: a failure is recorded
                packed, scales = ort_q4(quantize_matmul_4bits, w, dt)
                row[f"{k}_codes"] = int((unpack_ort(packed) == codes).sum())
                row[f"{k}_scales"] = int((scales.astype(np.float32) == d_gg).sum())
            except Exception as e:                               # noqa: BLE001
                row[f"{k}_codes"] = row[f"{k}_scales"] = 0
                row[f"{k}_error"] = f"{type(e).__name__}: {e}"[:200]
        for k in row:
            if k in tot:
                tot[k] += row[k]
        tot["weights"] += t["n"]
        tot["blocks"] += len(d)
        say("B1_ROW", row)
        print(f"[{i + 1}/{len(lin)}] {t['name']}  rule codes {row['rule_codes'] / t['n']:.6f} scales "
              f"{row['rule_scales'] / len(d):.6f}  ort32 codes {row['ort32_codes'] / t['n']:.6f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    # the head: the release's F16 head against f16 of the checkpoint's embedding, first 262,144 rows
    head = gguf_head(mm, by)
    emb = gc.st_u16(st["language_model.model.embed_tokens.weight"])
    head_eq = 0
    step = 8192
    for r in range(0, VOCAB, step):
        m = min(step, VOCAB - r)
        w = gc.bf16_to_f32(np.asarray(emb[r * HIDDEN:(r + m) * HIDDEN]))
        head_eq += int((w.astype(np.float16) == head[r:r + m].reshape(-1)).sum())
    # the norms: the GGUF's F32 against the checkpoint + 1 (llama.cpp's convention for Gemma), and + 0
    norm_rows = []
    for L in range(LAYERS):
        for gg, hf in GG_NORM.items():
            g = gguf_f32(mm, by[f"blk.{L}.{gg}.weight"])
            c = ck_f32(st, f"language_model.model.layers.{L}.{hf}.weight")
            norm_rows.append((g, c))
    g = gguf_f32(mm, by["output_norm.weight"])
    norm_rows.append((g, ck_f32(st, "language_model.model.norm.weight")))
    n_norm = sum(len(g) for g, _ in norm_rows)
    plus1 = sum(int((g == (c + np.float32(1))).sum()) for g, c in norm_rows)
    plus0 = sum(int((g == c).sum()) for g, c in norm_rows)
    res = {**tot, "head_exact": head_eq, "head_n": VOCAB * HIDDEN, "norm_n": n_norm, "norm_plus1_exact": plus1,
           "norm_plus0_exact": plus0, "seconds": round(time.time() - t0, 1)}
    say("B1_JSON", res)
    print(f"B1 rule: codes {tot['rule_codes'] / tot['weights']:.6f}, scales {tot['rule_scales'] / tot['blocks']:.6f}, "
          f"whole blocks {tot['rule_blocks'] / tot['blocks']:.6f}; B1b ORT fp32: codes "
          f"{tot['ort32_codes'] / tot['weights']:.6f} scales {tot['ort32_scales'] / tot['blocks']:.6f}; ORT fp16: "
          f"codes {tot['ort16_codes'] / tot['weights']:.6f} scales {tot['ort16_scales'] / tot['blocks']:.6f}; head "
          f"{head_eq / (VOCAB * HIDDEN):.6f}; norms +1 {plus1 / n_norm:.6f}, +0 {plus0 / n_norm:.6f}", flush=True)
    return res


def b1_common(mm, by, st) -> dict:
    """POST HOC (added after B1's first run showed the GGUF's norms within bf16 rounding of the
    checkpoint + 1, but almost never equal): is the checkpoint the bf16 rounding of what the GGUF
    holds? Per group, the fraction where bf16(GGUF value) equals the checkpoint's BF16 bits:
      norms    bf16(g - 1) (g the GGUF's F32 norm, 1 + w by llama.cpp's convention)
      head     bf16(f32(the GGUF's F16 head)), the first 262,144 rows
      linears  bf16(d * (q - 8)), (a)'s grid test, on this checkpoint"""
    res = {"label": "POST HOC"}
    n = e = 0
    for L in range(LAYERS):
        for gg, hf in GG_NORM.items():
            g = gguf_f32(mm, by[f"blk.{L}.{gg}.weight"])
            c = np.asarray(gc.st_u16(st[f"language_model.model.layers.{L}.{hf}.weight"]))
            e += int((gc.f32_to_bf16((g.astype(np.float64) - 1.0).astype(np.float32)) == c).sum())
            n += g.size
    g = gguf_f32(mm, by["output_norm.weight"])
    c = np.asarray(gc.st_u16(st["language_model.model.norm.weight"]))
    e += int((gc.f32_to_bf16((g.astype(np.float64) - 1.0).astype(np.float32)) == c).sum())
    res["norms"] = {"n": n + g.size, "bf16_equal": e}
    head = gguf_head(mm, by)
    emb = gc.st_u16(st["language_model.model.embed_tokens.weight"])
    e = 0
    for r in range(0, VOCAB, 8192):
        m = min(8192, VOCAB - r)
        e += int((gc.f32_to_bf16(head[r:r + m].astype(np.float32).ravel()) == np.asarray(emb[r * HIDDEN:(r + m) * HIDDEN])).sum())
    res["head"] = {"n": VOCAB * HIDDEN, "bf16_equal": e}
    n = e = 0
    for t in by.values():
        if t["tname"] != "Q4_0":
            continue
        codes, d = gguf_linear(mm, t)
        ex, _ = gc.grid_linear(d, codes.reshape(-1, 32), gc.st_u16(st[gc.hf_name(t["name"])]))
        e += ex
        n += t["n"]
    res["linears"] = {"n": n, "bf16_equal": e}
    say("B1_COMMON_JSON", res)
    print("B1 common source (POST HOC): bf16(GGUF) == checkpoint for " + "; ".join(
        f"{k} {v['bf16_equal'] / v['n']:.6f}" for k, v in res.items() if k != "label"), flush=True)
    return res


# ---------------------------------------------------------------- the text-only copy

ST_DTYPE = {"BF16": 2, "F32": 4, "F16": 2}


def write_safetensors(path: Path, items) -> None:
    """items: [(name, dtype, shape, writer)] where writer(f) streams the tensor's bytes."""
    header, off = {}, 0
    for name, dt, shape, _ in items:
        n = int(np.prod(shape)) * ST_DTYPE[dt]
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt"}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for name, dt, shape, writer in items:
            start = f.tell()
            writer(f)
            assert f.tell() - start == header[name]["data_offsets"][1] - header[name]["data_offsets"][0], name


def text_only(paths, mm, by, st) -> dict:
    """The checkpoint as a Gemma3ForCausalLM folder: its tensors renamed and copied byte for byte
    (BF16), the embedding replaced by the release's F16 head upcast to F32 (262,144 rows)."""
    TEXT.mkdir(parents=True, exist_ok=True)
    ck = json.loads(Path(paths["config.json"]).read_text(encoding="utf-8"))
    cfg = dict(ck["text_config"])
    cfg.update({"architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text", "vocab_size": VOCAB,
                "tie_word_embeddings": True, "torch_dtype": "bfloat16", "eos_token_id": ck["eos_token_id"],
                "bos_token_id": 2, "pad_token_id": 0})
    (TEXT / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    for f in TOKENIZER_FILES:
        if f in paths:                                               # all of them in the build; none in the selftest
            shutil.copyfile(paths[f], TEXT / f)
    head = gguf_head(mm, by)

    def raw(e):
        def w(f):
            mmap = np.memmap(e["path"], dtype=np.uint8, mode="r", offset=e["begin"], shape=(e["end"] - e["begin"],))
            for i in range(0, len(mmap), 1 << 26):
                f.write(mmap[i:i + (1 << 26)].tobytes())
        return w

    def head_f32(f):
        for r in range(0, VOCAB, 8192):
            f.write(head[r:r + 8192].astype(np.float32).tobytes())

    items = []
    for name in sorted(st):
        if not name.startswith("language_model.model.") or name.endswith("embed_tokens.weight"):
            continue
        e = st[name]
        assert e["dtype"] == "BF16", (name, e["dtype"])
        items.append(("model." + name[len("language_model.model."):], "BF16", e["shape"], raw(e)))
    items.append(("model.embed_tokens.weight", "F32", [VOCAB, HIDDEN], head_f32))
    out = TEXT / "model.safetensors"
    write_safetensors(out, items)
    res = {"tensors": len(items), "config": cfg, "safetensors_bytes": out.stat().st_size,
           "safetensors_sha256": sha256(out), "dropped": sorted({n.split(".")[0] for n in st} - {"language_model"}),
           "embedding": "the GGUF's F16 head upcast to F32, [262144, 2560]",
           "dropped_rows": f"the checkpoint's rows {VOCAB}-{CK_VOCAB - 1} (padding and the image token, id "
                           f"{ck.get('image_token_index')})"}
    say("TEXTONLY_JSON", res)
    return res


# ---------------------------------------------------------------- one builder run (a subprocess)

def builder_run(arm: str, paths=None) -> int:
    """ORT GenAI 0.11.2's builder with its model load wrapped: the linears carry the GGUF's codes and
    scales (the builder's pre-quantized qweight path), the norms are fp32 from the checkpoint."""
    import torch
    import onnxruntime_genai.models.builder as B
    ep, acc, h16 = ARMS[arm]
    mm, by, st, _ = open_release(paths or release_paths())
    real = B.AutoModelForCausalLM

    class Load:
        @staticmethod
        def from_pretrained(path, **kw):
            keep = {k: v for k, v in kw.items() if k in ("cache_dir", "token", "trust_remote_code")}
            model = real.from_pretrained(path, dtype=torch.float16, low_cpu_mem_usage=True, **keep)
            for L, layer in enumerate(model.model.layers):
                for gg, (blk, proj) in GG_PROJ.items():
                    mod = getattr(getattr(layer, blk), proj)
                    codes, d = gguf_linear(mm, by[f"blk.{L}.{gg}.weight"])
                    assert codes.shape == (mod.out_features, mod.in_features // 32, 32), (L, gg, codes.shape)
                    mod.qweight = torch.from_numpy(pack_ort(codes))
                    mod.scales = torch.from_numpy(gc.f16_to_f32(d))
                    mod.bits, mod.group_size = 4, 32
                    del mod.weight                                   # the float linear never enters the graph
                for gg, hf in GG_NORM.items():
                    mod = layer.get_submodule(hf)
                    mod.weight = torch.nn.Parameter(torch.from_numpy(
                        ck_f32(st, f"language_model.model.layers.{L}.{hf}.weight")), requires_grad=False)
            model.model.norm.weight = torch.nn.Parameter(torch.from_numpy(
                ck_f32(st, "language_model.model.norm.weight")), requires_grad=False)
            print(f"WRAPPED_LOAD {len(model.model.layers)} layers: linears from the GGUF, norms fp32", flush=True)
            return model

    B.AutoModelForCausalLM = Load
    extra = [] if acc is None else [f"int4_accuracy_level={acc}"]
    if h16:
        extra.append("int4_nodes_to_exclude=/lm_head/MatMul")
    opts = B.parse_extra_options(extra, ep)
    out = ONNX / arm
    if out.exists():
        shutil.rmtree(out)
    try:
        B.create_model(None, str(TEXT), str(out), "int4", ep, str(CACHE / arm), **opts)
    finally:
        B.AutoModelForCausalLM = real
    return 0


def run_builders() -> dict:
    """Each arm's builder in its own process; its peak private bytes (commit) and the system's lowest
    available RAM are sampled once a second, so a build that paged is on the record."""
    import psutil
    res = {}
    for arm in ARMS:
        t0 = time.time()
        cmd = [sys.executable, "tools/gemma_decode.py", "builder", "--arm", arm]
        out = WORK / f"builder_{arm}.txt"
        with open(out, "w", encoding="utf-8") as fh:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"})
            proc, peak, low = psutil.Process(p.pid), 0, psutil.virtual_memory().available
            while p.poll() is None:
                try:
                    peak = max(peak, proc.memory_info().private)
                except psutil.Error:
                    pass
                low = min(low, psutil.virtual_memory().available)
                time.sleep(1.0)
        lines = out.read_text(encoding="utf-8", errors="replace").splitlines()
        keep = [s for s in lines if s.strip() and not s.startswith(("Reading decoder layer", "Saving "))
                and "matmul_nbits_quantizer [INFO] - skip" not in s and not s.startswith("/model/")]
        for s in keep[-25:]:
            print(f"  {arm}| {s}")
        res[arm] = {"exit": p.returncode, "seconds": round(time.time() - t0, 1),
                    "peak_private_gb": round(peak / 1e9, 2), "min_available_gb": round(low / 1e9, 2),
                    "cmd": f"python[{os.environ.get('CONDA_DEFAULT_ENV', '?')}] " + " ".join(cmd[1:])}
        say("BUILDER_JSON", {"arm": arm, **res[arm]})
        if p.returncode:
            sys.exit(f"the builder failed for {arm}")
    return res


# ---------------------------------------------------------------- reading and writing initializers

NP = {1: np.float32, 10: np.float16, 2: np.uint8, 7: np.int64, 6: np.int32}


class Graph:
    """model.onnx with its external data: initializers read and overwritten in place, by value."""

    def __init__(self, d: Path):
        import onnx
        self.d, self.path = d, d / "model.onnx"
        self.m = onnx.load(str(self.path), load_external_data=False)
        self.init = {t.name: t for t in self.m.graph.initializer}
        self.dirty = False

    def _loc(self, t):
        kv = {e.key: e.value for e in t.external_data}
        return self.d / kv["location"], int(kv.get("offset", 0)), int(kv["length"])

    def read(self, name) -> np.ndarray:
        t = self.init[name]
        dt = NP[t.data_type]
        if t.data_location == 1:
            f, off, n = self._loc(t)
            a = np.fromfile(f, dtype=dt, count=n // np.dtype(dt).itemsize, offset=off)
        else:
            import onnx.numpy_helper as nh
            a = nh.to_array(t)
        return a.reshape(tuple(t.dims))

    def write(self, name, a: np.ndarray) -> None:
        t = self.init[name]
        a = np.ascontiguousarray(a, dtype=NP[t.data_type]).reshape(tuple(t.dims))
        if t.data_location == 1:
            f, off, n = self._loc(t)
            assert a.nbytes == n, (name, a.nbytes, n)
            with open(f, "r+b") as fh:
                fh.seek(off)
                fh.write(a.tobytes())
        else:
            t.raw_data = a.tobytes()
            del t.float_data[:]
            self.dirty = True

    def save(self) -> None:
        if self.dirty:
            import onnx
            onnx.save(self.m, str(self.path))
            self.dirty = False


def rope_caches(cfg_path: Path):
    """transformers' own Gemma 3 cos/sin for the global (linear factor) and local layers, first half,
    [max_position_embeddings, head_dim / 2] fp32."""
    import copy
    import torch
    from transformers import AutoConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
    cfg = AutoConfig.from_pretrained(str(cfg_path))
    loc = copy.deepcopy(cfg)
    loc.rope_theta = loc.rope_local_base_freq
    loc.rope_scaling = {"rope_type": "default"}
    out = {}
    pos = torch.arange(cfg.max_position_embeddings)[None]
    x = torch.zeros(1, dtype=torch.float32)
    for k, c in (("global", cfg), ("local", loc)):
        cos, sin = Gemma3RotaryEmbedding(config=c)(x, pos)
        out[k] = (cos[0, :, :cfg.head_dim // 2].numpy(), sin[0, :, :cfg.head_dim // 2].numpy())
    return out


FIXES = ("N1", "F1", "F2", "F3")
# N1  the release's norms: the builder writes the checkpoint's (bf16) + 1 in fp32; the GGUF's F32 norms
#     are within bf16 rounding of those but not equal (B1's norm check), so the GGUF's are written
# F1  onnxruntime-genai 0.11.2's builder ignores rope_scaling {linear, factor 8}: the global cos/sin
#     caches are written with transformers' values (fixed on genai main, 4859742e)
# F2  the builder rounds the embedding scale to np.round(sqrt(2560), 2) = 50.6: set to sqrt(2560)
#     (fp32 50.596443; in fp16 both are 50.59375, so the DML arms do not change)
# F3  onnxruntime 1.23.3's GroupQueryAttention CPU kernel keeps local_window_size + 1 keys (fixed on
#     ORT main by 8af9f58086, #25927): the CPU arms' local layers get sliding_window - 1, so they keep
#     1024 keys as transformers does. DirectML's GQA ignores local_window_size (ORT 1.23 and main
#     source): the DML arms are left as built and are Gemma 3 only up to 1024 tokens of context (F4).


def release(arm: str, mm, by, st, fixes=()) -> dict:
    """Read back every weight-bearing initializer against the release by value; apply and verify the
    named fixes (F1, F2, F3 above). With no fixes, nothing is written."""
    g = Graph(ONNX / arm)
    ep, acc, h16 = ARMS[arm]
    io =np.float32 if ep == "cpu" else np.float16
    seen, rows = set(), {}
    nodes = list(g.m.graph.node)
    # the 238 linears: MatMulNBits by value, s * (q - 8) against d * (q - 8), exact
    lin = {"tensors": 0, "weights": 0, "exact": 0, "attrs_ok": True}
    for n in nodes:
        if n.op_type != "MatMulNBits" or n.name.startswith("/lm_head"):
            continue
        mt = re.match(r"/model/layers\.(\d+)/(attn|mlp)/(\w+_proj)/", n.name)
        assert mt, n.name
        L, proj = int(mt.group(1)), mt.group(3)
        codes, d = gguf_linear(mm, by[f"blk.{L}.{ONNX_PROJ[proj]}.weight"])
        b, s = g.read(n.input[1]), g.read(n.input[2]).astype(np.float32).ravel()
        seen.update(n.input[1:3])
        got = dequant(unpack_ort(b), s)
        want = dequant(codes, gc.f16_to_f32(d))
        attrs = {a.name: a.i for a in n.attribute}
        lin["tensors"] += 1
        lin["weights"] += want.size
        lin["exact"] += int((got == want).sum())
        lin["attrs_ok"] &= (len(n.input) == 3 and attrs.get("bits") == 4 and attrs.get("block_size") == 32
                            and attrs.get("accuracy_level", 0) == (acc or 0) and attrs.get("K") == want.shape[1]
                            and attrs.get("N") == want.shape[0])
    rows["linears"] = lin
    head = gguf_head(mm, by)
    step = 16384
    # the Gather table: fp32 (CPU) or fp16 (DML) of the release's F16 head, row blocks to spare memory
    emb = g.read("model.embed_tokens.weight")
    seen.add("model.embed_tokens.weight")
    eq = sum(int((emb[r:r + step].astype(np.float32) == head[r:r + step].astype(np.float32)).sum())
             for r in range(0, VOCAB, step))
    rows["embedding"] = {"dtype": str(emb.dtype), "shape": list(emb.shape), "exact": eq, "n": head.size}
    del emb
    hn = [n for n in nodes if n.name.startswith("/lm_head/MatMul")]
    assert len(hn) == 1, [n.name for n in hn]
    hn = hn[0]
    if h16:
        assert hn.op_type == "MatMul", hn.op_type
        w = g.read(hn.input[1])                                      # [2560, 262144]
        seen.add(hn.input[1])
        eq = sum(int((w[:, r:r + step].T.astype(np.float32) == head[r:r + step].astype(np.float32)).sum())
                 for r in range(0, VOCAB, step))
        rows["head"] = {"format": "H16", "op": hn.op_type, "name": hn.name, "dtype": str(w.dtype),
                        "shape": list(w.shape), "exact": eq, "n": head.size}
        del w
    else:
        assert hn.op_type == "MatMulNBits", hn.op_type
        b, s = g.read(hn.input[1]), g.read(hn.input[2])
        seen.update(hn.input[1:3])
        attrs = {a.name: a.i for a in hn.attribute}
        kb = HIDDEN // 32
        s32 = s.astype(np.float32).reshape(VOCAB, kb)
        e2 = r2 = 0.0
        for r in range(0, VOCAB, step):
            got = dequant(unpack_ort(b[r:r + step]), s32[r:r + step].ravel()).astype(np.float64)
            ref = head[r:r + step].astype(np.float64)
            e2 += float(((got - ref) ** 2).sum())
            r2 += float((ref ** 2).sum())
        rows["head"] = {"format": "H4", "op": hn.op_type, "name": hn.name, "inputs": len(hn.input), "attrs": attrs,
                        "scales_dtype": str(s.dtype), "bytes": int(b.nbytes + s.nbytes),
                        "rel_l2_vs_release": (e2 / r2) ** 0.5}
    # the norms: fp32 on both EPs (Gemma 2/3 compute their norms in fp32), 1 + w, against the GGUF's F32
    norm = {"tensors": 0, "n": 0, "exact": 0, "injected": []}
    where = {"attn_norm": "model.layers.{L}.input_layernorm.weight",
             "post_attention_norm": "model.layers.{L}.post_attention_layernorm.weight",
             "ffn_norm": "model.layers.{L}.pre_feedforward_layernorm.weight",
             "post_ffw_norm": "model.layers.{L}.post_feedforward_layernorm.weight",
             "attn_q_norm": "model.layers.{L}.attn.q_norm.layernorm.weight",
             "attn_k_norm": "model.layers.{L}.attn.k_norm.layernorm.weight"}
    pairs = [(v.format(L=L), f"blk.{L}.{gg}.weight") for L in range(LAYERS) for gg, v in where.items()]
    pairs.append((f"model.layers.{LAYERS}.final_norm_layernorm.weight", "output_norm.weight"))
    norm["exact_on_disk"] = 0
    for name, gg in pairs:
        want = gguf_f32(mm, by[gg])
        got = g.read(name).ravel()
        seen.add(name)
        e = int((got == want).sum())
        norm["tensors"] += 1
        norm["n"] += want.size
        norm["exact"] += e
        norm["dtype"] = str(got.dtype)
        if e != want.size and "N1" in fixes:
            g.write(name, want)
            norm["injected"].append(name)
            e = int((g.read(name).ravel() == want).sum())            # read back
        norm["exact_on_disk"] += e
    norm["injected"] = len(norm["injected"])
    rows["norms"] = norm
    # RoPE caches and the embedding scale (F1, F2)
    rc = rope_caches(TEXT)
    rope = {}
    for k in ("global", "local"):
        for j, which in enumerate(("cos", "sin")):
            name = f"{which}_cache_{k}"
            got = g.read(name)
            want = rc[k][j].astype(io)
            seen.add(name)
            rope[name] = {"shape": list(got.shape), "max_abs_diff": float(np.abs(got.astype(np.float32) - want.astype(np.float32)).max())}
            if "F1" in fixes and k == "global":
                g.write(name, want)
                rope[name]["after_fix_max_abs_diff"] = float(np.abs(g.read(name).astype(np.float32) - want.astype(np.float32)).max())
    rows["rope"] = rope
    # F2, the embedding scale: a Constant node (or an initializer) feeding /model/embed_tokens/Mul
    import onnx.numpy_helper as nh
    mul = next(n for n in nodes if n.name == "/model/embed_tokens/Mul")
    cname = mul.input[1]
    want = np.array(np.sqrt(np.float64(HIDDEN)), dtype=io)
    consumers = [n.name for n in nodes if cname in n.input]
    if cname in g.init:
        got = g.read(cname)
        seen.add(cname)
        rows["embed_scale"] = {"name": cname, "value": float(got.ravel()[0]), "dtype": str(got.dtype),
                               "want": float(want), "consumers": consumers}
        if "F2" in fixes and got.ravel()[0] != want:
            g.write(cname, want.reshape(got.shape))
    else:
        cn = next(n for n in nodes if n.op_type == "Constant" and cname in n.output)
        at = next(a for a in cn.attribute if a.name == "value")
        val = nh.to_array(at.t)
        rows["embed_scale"] = {"name": cname, "value": float(val.ravel()[0]), "dtype": str(val.dtype),
                               "want": float(want), "constant_node": True, "consumers": consumers}
        if "F2" in fixes and val.ravel()[0] != want:
            at.t.CopyFrom(nh.from_array(want.reshape(val.shape), at.t.name))
            g.dirty = True
    # F3, the CPU arms' local window: sliding_window - 1 on this runtime's GQA CPU kernel
    window = json.loads((TEXT / "config.json").read_text(encoding="utf-8"))["sliding_window"]
    for n in nodes:
        if n.op_type == "GroupQueryAttention" and "F3" in fixes and ep == "cpu":
            for x in n.attribute:
                if x.name == "local_window_size" and x.i == window:
                    x.i = window - 1
                    g.dirty = True
    g.save()
    # read the edited file back: F2's constant and every attention node's attributes
    g2 = Graph(ONNX / arm)
    nodes2 = list(g2.m.graph.node)
    if cname in g2.init:
        rows["embed_scale"]["on_disk"] = float(g2.read(cname).ravel()[0])
    else:
        cn = next(n for n in nodes2 if n.op_type == "Constant" and cname in n.output)
        rows["embed_scale"]["on_disk"] = float(nh.to_array(next(a for a in cn.attribute if a.name == "value").t).ravel()[0])
    att, windows = [], {}
    for n in nodes2:
        if n.op_type in ("GroupQueryAttention", "RotaryEmbedding"):
            a = {x.name: (x.i if x.type == 2 else x.f if x.type == 1 else None) for x in n.attribute}
            att.append({"op": n.op_type, "name": n.name, **{k: v for k, v in a.items() if v is not None}})
            if n.op_type == "GroupQueryAttention":
                w = a.get("local_window_size", -1)
                windows[w] = windows.get(w, 0) + 1
    rows["attention"] = att
    rows["gqa_windows"] = {str(k): v for k, v in sorted(windows.items())}
    rest = []
    for name, t in g.init.items():
        if name not in seen:
            rest.append({"name": name, "dtype": int(t.data_type), "dims": list(t.dims)})
    rows["other_initializers"] = rest
    say("RELEASE_JSON", {"arm": arm, "fixes": list(fixes), **rows})
    n_global = sum(1 for i in range(LAYERS) if (i + 1) % 6 == 0)
    want_local = window - 1 if ("F3" in fixes and ep == "cpu") else window
    ok = (lin["exact"] == lin["weights"] and lin["tensors"] == LAYERS * len(GG_PROJ) and lin["attrs_ok"]
          and rows["embedding"]["exact"] == rows["embedding"]["n"] and norm["tensors"] == LAYERS * 6 + 1
          and norm["exact_on_disk"] == norm["n"]
          and (not h16 or rows["head"]["exact"] == rows["head"]["n"])
          and windows == {-1: n_global, want_local: LAYERS - n_global}
          and ("F2" not in fixes or rows["embed_scale"]["on_disk"] == float(want))
          and ("F1" not in fixes or all(rope[f"{w}_cache_global"]["after_fix_max_abs_diff"] == 0.0 for w in ("cos", "sin"))))
    print(f"{arm} [{'+'.join(fixes) or 'as built'}]: linears {lin['exact']}/{lin['weights']} exact ({lin['tensors']} "
          f"tensors, attrs {'ok' if lin['attrs_ok'] else 'WRONG'}); embedding {rows['embedding']['exact']}/"
          f"{rows['embedding']['n']}; head {rows['head']}; norms {norm['exact']}/{norm['n']} ({norm['tensors']}), on disk "
          f"{norm['exact_on_disk']}/{norm['n']}; "
          f"rope global cos {rope['cos_cache_global']['max_abs_diff']:.3g}, local cos "
          f"{rope['cos_cache_local']['max_abs_diff']:.3g}; embed scale {rows['embed_scale']['value']} -> on disk "
          f"{rows['embed_scale']['on_disk']}; GQA windows {rows['gqa_windows']}; {len(rest)} other initializers -> "
          f"{'RELEASE OK' if ok else 'RELEASE MISMATCH'}", flush=True)
    return {"ok": ok, **rows}


# ---------------------------------------------------------------- B3: DirectML placement (a subprocess)

def placement(arm: str) -> int:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 0
    so.log_verbosity_level = 0
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    ort.InferenceSession(str(ONNX / arm / "model.onnx"), so,
                         providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
    print("PLACEMENT_SESSION_OK", flush=True)
    return 0


def b3() -> dict:
    res = {}
    for arm in ("D-H4", "D-H16"):
        p = subprocess.run([sys.executable, "tools/gemma_decode.py", "placement", arm], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        text = p.stdout + p.stderr
        placed = re.findall(r"(?:All nodes placed on|Node\(s\) placed on) \[(\w+)\]\. Number of nodes: (\d+)", text)
        cpu_nodes, cur = [], None
        for s in text.splitlines():
            m = re.search(r"placed on \[(\w+)\]", s)
            if m:
                cur = m.group(1)
                continue
            msg = s.split("] ", 1)[-1]
            m = re.match(r"^\s*(\w+) \((.*)\)\s*$", msg)
            if cur == "CPUExecutionProvider" and m:
                cpu_nodes.append(f"{m.group(1)} ({m.group(2)})")
            elif not m:
                cur = None
        placed = {ep: int(n) for ep, n in placed}
        # an unmatched log must not read as "all on DirectML"
        status = ("UNPARSED" if not placed or "PLACEMENT_SESSION_OK" not in text
                  else "ALL_DML" if set(placed) == {"DmlExecutionProvider"} else "CPU_NODES")
        res[arm] = {"exit": p.returncode, "session_ok": "PLACEMENT_SESSION_OK" in text, "status": status,
                    "placed": placed, "cpu_nodes": cpu_nodes[:80], "cpu_ops": sorted({c.split(" ")[0] for c in cpu_nodes}),
                    "placement_lines": [s.split("] ", 1)[-1].strip() for s in text.splitlines() if "placed on [" in s][:10]}
        print(f"B3 {arm}: {status} {placed}" + (f", CPU ops {res[arm]['cpu_ops']}" if cpu_nodes else ""), flush=True)
        say("B3_JSON", {"arm": arm, **res[arm]})
    return res


# ---------------------------------------------------------------- the fidelity gate

def release_norm(mm, by, name: str) -> np.ndarray:
    """A GGUF F32 norm (llama.cpp stores 1 + w) as transformers' Gemma weight w = g - 1. Rounding w to
    fp32 loses up to a few ulps of g where g is near 0, so the reference does not multiply by 1 + w:
    release_norms() hands every norm module g itself."""
    return (gguf_f32(mm, by[name]).astype(np.float64) - 1.0).astype(np.float32)


def release_norms(model, mm, by) -> list:
    """Give every Gemma3RMSNorm of a Gemma3ForCausalLM the GGUF's F32 norm g as `release_g`; while
    exact_norms() is active those modules multiply by g, as ORT's SimplifiedLayerNormalization does
    with the weight N1 writes. Returns [(module, GGUF name)]."""
    import torch
    out = []
    for L, layer in enumerate(model.model.layers):
        for gg, hf in GG_NORM.items():
            out.append((layer.get_submodule(hf), f"blk.{L}.{gg}.weight"))
    out.append((model.model.norm, "output_norm.weight"))
    for mod, name in out:
        mod.release_g = torch.from_numpy(gguf_f32(mm, by[name]))    # a plain attribute: it stays on the CPU
    return out


class exact_norms:
    """Patch Gemma3RMSNorm.forward for the block: a module with release_g computes norm(x) * g in fp32
    (transformers' own normalization, its 1 + w replaced by g); any other module is unchanged."""

    def __enter__(self):
        from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
        self.cls, self.real = Gemma3RMSNorm, Gemma3RMSNorm.forward
        real = self.real

        def forward(mod, x):
            g = mod.__dict__.get("release_g")
            if g is None:
                return real(mod, x)
            return (mod._norm(x.float()) * g).type_as(x)
        Gemma3RMSNorm.forward = forward
        return self

    def __exit__(self, *exc):
        self.cls.forward = self.real


def reference_norm_check(mm, by) -> dict:
    """The reference's norms, read back: on a meta-device copy of the reference model, every norm
    module's multiplier equals the GGUF's F32 norm bit for bit, and the patched forward computes
    norm(x) * g (checked on one module against numpy)."""
    import torch
    from transformers import AutoConfig, Gemma3ForCausalLM
    cfg = AutoConfig.from_pretrained(str(TEXT))
    with torch.device("meta"):
        model = Gemma3ForCausalLM(cfg)
    mods = release_norms(model, mm, by)
    n = exact = 0
    for mod, name in mods:
        g = gguf_f32(mm, by[name])
        n += g.size
        exact += int((mod.release_g.numpy().view(np.uint32) == g.view(np.uint32)).sum())
    mod, name = mods[0]
    probe = type(mod)(mod.release_g.numel(), eps=cfg.rms_norm_eps)
    probe.release_g = mod.release_g
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((3, mod.release_g.numel())).astype(np.float32))
    with exact_norms():
        y = probe(x).numpy()
    xn = x.numpy().astype(np.float64)
    want = xn / np.sqrt((xn ** 2).mean(-1, keepdims=True) + cfg.rms_norm_eps) * gguf_f32(mm, by[name])
    res = {"tensors": len(mods), "n": n, "bit_exact": exact, "forward_rel_err": float(np.abs(y - want).max() / np.abs(want).max())}
    say("REFERENCE_NORMS_JSON", res)
    print(f"REFERENCE NORMS: the reference multiplies by the GGUF's F32 norm, bit-exact for {exact}/{n} ({len(mods)} "
          f"tensors); its forward on {name} matches norm(x) * g to {res['forward_rel_err']:.2g} (fp32 against float64)",
          flush=True)
    return res


def reference_logits(mm, by, st, ids) -> np.ndarray:
    """transformers' Gemma3ForCausalLM in fp32 with the release's weights (the GGUF's dequantized q4_0
    linears, F16 head and F32 norms), eager attention, one decoder layer in memory at a time: each
    layer is built on the meta device and materialized by a forward pre-hook, freed after."""
    import torch
    from transformers import AutoConfig, Gemma3ForCausalLM
    from transformers.models.gemma3.modeling_gemma3 import (Gemma3RMSNorm, Gemma3RotaryEmbedding,
                                                            Gemma3TextScaledWordEmbedding)
    import copy
    cfg = AutoConfig.from_pretrained(str(TEXT))
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = Gemma3ForCausalLM(cfg)
    tm = model.model
    head = torch.from_numpy(gguf_head(mm, by).astype(np.float32))
    emb = Gemma3TextScaledWordEmbedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id, embed_scale=cfg.hidden_size ** 0.5)
    emb.weight = torch.nn.Parameter(head, requires_grad=False)
    tm.embed_tokens = emb
    norm = Gemma3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight = torch.nn.Parameter(torch.from_numpy(release_norm(mm, by, "output_norm.weight")), requires_grad=False)
    tm.norm = norm
    tm.rotary_emb = Gemma3RotaryEmbedding(config=cfg)
    loc = copy.deepcopy(cfg)
    loc.rope_theta = loc.rope_local_base_freq
    loc.rope_scaling = {"rope_type": "default"}
    tm.rotary_emb_local = Gemma3RotaryEmbedding(config=loc)
    lm = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device="cpu")
    lm.weight = emb.weight                                           # tied
    model.lm_head = lm

    def state(L):
        sd = {}
        for gg, (blk, proj) in GG_PROJ.items():
            codes, d = gguf_linear(mm, by[f"blk.{L}.{gg}.weight"])
            sd[f"{blk}.{proj}.weight"] = torch.from_numpy(dequant(codes, gc.f16_to_f32(d)))
        for gg, hf in GG_NORM.items():
            sd[f"{hf}.weight"] = torch.from_numpy(release_norm(mm, by, f"blk.{L}.{gg}.weight"))
        return sd

    def pre(mod, args, kwargs):
        mod.to_empty(device="cpu")
        mod.load_state_dict(state(mod.self_attn.layer_idx), strict=True)

    def post(mod, args, kwargs, out):
        mod.to("meta")

    for layer in tm.layers:
        layer.register_forward_pre_hook(pre, with_kwargs=True)
        layer.register_forward_hook(post, with_kwargs=True)
    release_norms(model, mm, by)
    with torch.no_grad(), exact_norms():
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
    return out.logits[0].float().numpy()


def ort_logits(arm: str, ids) -> np.ndarray:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 8
    if ARMS[arm][0] == "dml":
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(ONNX / arm / "model.onnx"), so, providers=providers)
    feed = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}
    for i in sess.get_inputs():
        if i.name == "position_ids":
            feed[i.name] = np.arange(len(ids), dtype=np.int64)[None]
        elif i.name.startswith("past_key_values"):                   # [batch, kv heads, 0, head_dim], an empty cache
            dt = np.float16 if i.type == "tensor(float16)" else np.float32
            feed[i.name] = np.zeros((1, i.shape[1], 0, i.shape[3]), dtype=dt)
    out = sess.run(["logits"], feed)[0][0].astype(np.float32)
    del sess
    return out


def kl_rows(ref: np.ndarray, got: np.ndarray) -> np.ndarray:
    """KL(ref || got) per position, in float64, from logits."""
    out = np.empty(len(ref))
    for i in range(len(ref)):
        a, b = ref[i].astype(np.float64), got[i].astype(np.float64)
        la = a - (a.max() + np.log(np.exp(a - a.max()).sum()))
        lb = b - (b.max() + np.log(np.exp(b - b.max()).sum()))
        out[i] = float((np.exp(la) * (la - lb)).sum())
    return out


_REF = {}                                # reference logits by length: they do not depend on the ONNX edits


def free_ram_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 1e9


def fidelity(mm, by, st, tag: str, arms=("C0-H16",)) -> dict:
    """The gate: C0-H16 against the fp32 reference at each length, max per-position KL <= FID_KL_MAX.
    Any other arm listed (D-H16) is scored against the same reference and reported; it decides nothing."""
    window = json.loads((TEXT / "config.json").read_text(encoding="utf-8"))["sliding_window"]
    if free_ram_gb() < FID_MIN_FREE_GB:
        sys.exit(f"FIDELITY_STOP: {free_ram_gb():.1f} GB free, below {FID_MIN_FREE_GB} GB; a reference that swaps "
                 f"or is killed voids the gate, so none is run")
    rng = np.random.default_rng(FID_SEED)
    ids_all = [2] + rng.integers(3, min(VOCAB, 256_000), max(FID_LENGTHS) - 1).tolist()
    res = {"tag": tag, "gate_arm": "C0-H16", "seed": FID_SEED, "kl_max": FID_KL_MAX, "window": window,
           "ram_free_gb_before": round(free_ram_gb(), 1), "passes": []}
    for n in FID_LENGTHS:
        ids = ids_all[:n]
        t0 = time.time()
        if n not in _REF:
            _REF[n] = reference_logits(mm, by, st, ids)
        ref = _REF[n]
        t1 = time.time()
        for arm in arms:
            got = ort_logits(arm, ids)
            t2 = time.time()
            kl = kl_rows(ref, got)
            top1 = float((ref.argmax(1) == got.argmax(1)).mean())
            row = {"arm": arm, "tokens": n, "max_kl": float(kl.max()), "mean_kl": float(kl.mean()),
                   "argmax_kl": int(kl.argmax()), "max_kl_past_window": float(kl[window:].max()) if n > window else None,
                   "max_kl_within_window": float(kl[:window].max()), "top1_agree": top1,
                   "ref_seconds": round(t1 - t0, 1), "ort_seconds": round(t2 - t1, 1),
                   "ram_free_gb": round(free_ram_gb(), 1)}
            res["passes"].append(row)
            print(f"FIDELITY {tag} {arm} {n} tokens: max KL {row['max_kl']:.3g} at {row['argmax_kl']}, mean "
                  f"{row['mean_kl']:.3g}, top-1 {top1:.4f}"
                  + (f", past the {window}-token window {row['max_kl_past_window']:.3g}" if n > window else ""), flush=True)
            del got
            t1 = time.time()
    res["pass"] = all(r["max_kl"] <= FID_KL_MAX for r in res["passes"] if r["arm"] == "C0-H16")
    say("FIDELITY_JSON", res)
    return res


# ---------------------------------------------------------------- smoke, affinity, pins

def smoke() -> dict:
    """Greedy SMOKE_TOKENS tokens per arm, twice: the prompt as og.Tokenizer.encode gives it, and with
    BOS first. GenAI 0.11.2's encode adds no BOS for this tokenizer (transformers' adds it), and Gemma 3
    without BOS degenerates (build 3's log: ' is' 16 times on every arm); the suite prepends BOS itself."""
    import onnxruntime_genai as og
    res = {}
    for arm in ARMS:
        t0 = time.time()
        model = og.Model(str(ONNX / arm))
        tok = og.Tokenizer(model)
        enc = [int(t) for t in tok.encode(SMOKE_PROMPT)]
        res[arm] = []
        for bos_first in (False, True):
            ids = [BOS] + enc if bos_first and enc[:1] != [BOS] else enc
            params = og.GeneratorParams(model)
            params.set_search_options(do_sample=False, max_length=64)
            gen = og.Generator(model, params)
            gen.append_tokens(np.array(ids, dtype=np.int32))
            out = []
            while not gen.is_done() and len(out) < SMOKE_TOKENS:
                gen.generate_next_token()
                out.append(int(gen.get_next_tokens()[0]))
            row = {"bos_first": bos_first, "prompt_ids": ids, "tokens": out,
                   "text": tok.decode(np.array(out, dtype=np.int32)), "seconds": round(time.time() - t0, 1),
                   "onnxruntime_dll": loaded_dll("onnxruntime.dll")}
            res[arm].append(row)
            say("SMOKE_JSON", {"arm": arm, **row})
            del gen, params
        del tok, model
    return res


def loaded_dll(name: str):
    """The path of a DLL loaded in this process (the runtime GenAI actually uses), scrubbed."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.windll.kernel32
    k32.GetModuleHandleW.restype = wintypes.HMODULE
    h = k32.GetModuleHandleW(name)
    if not h:
        return None
    buf = ctypes.create_unicode_buffer(1024)
    k32.GetModuleFileNameW(wintypes.HMODULE(h), buf, 1024)
    return re.sub(r"(?i)C:\\Users\\[^\\]+", r"C:\\Users\\<user>", buf.value)


def affinity() -> dict:
    import onnxruntime_genai as og
    import measure_noise as mn
    mn.pin_calling_thread(PIN_CPUS[0])
    before = mn.thread_ids()
    cfg = og.Config(str(ONNX / "C4-H4"))
    entries = ";".join(str(c + 1) for c in PIN_CPUS[1:])
    cfg.overlay(json.dumps({"model": {"decoder": {"session_options": {
        "intra_op_num_threads": len(PIN_CPUS), "inter_op_num_threads": 1,
        "config_entries": {"session.intra_op_thread_affinities": entries}}}}}))
    model = og.Model(cfg)
    tok = og.Tokenizer(model)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=32)
    gen = og.Generator(model, params)
    gen.append_tokens(tok.encode(SMOKE_PROMPT))
    gen.generate_next_token()
    new = mn.thread_placement(mn.thread_ids() - before)
    pinned = sorted(tuple(t["cpus"]) for t in new if len(t["cpus"]) == 1)
    want = sorted((c,) for c in PIN_CPUS[1:])
    res = {"entries": entries, "new_threads": len(new), "single_cpu_threads": [list(p) for p in pinned],
           "takes_affinities": all(w in pinned for w in want)}
    say("AFFINITY_JSON", res)
    del gen, params, tok, model
    return res


def genai_configs() -> dict:
    """Each arm's genai_config.json as written: what onnxruntime-genai runs. 0.11.2 writes a decoder
    sliding_window (a KV ring for every layer) only for trt-rtx; its absence is checked here."""
    res = {}
    for arm in ARMS:
        cfg = json.loads((ONNX / arm / "genai_config.json").read_text(encoding="utf-8"))
        dec = cfg["model"]["decoder"]
        res[arm] = {"sliding_window_key": "sliding_window" in dec, "config": cfg}
        say("GENAI_CONFIG_JSON", {"arm": arm, **res[arm]})
        print(f"GENAI_CONFIG {arm}: decoder sliding_window key {'PRESENT' if 'sliding_window' in dec else 'absent'}; "
              f"provider options {dec.get('session_options', {}).get('provider_options')}; context_length "
              f"{cfg['model'].get('context_length')}", flush=True)
    return res


def pins() -> dict:
    res = {}
    for arm in ARMS:
        d = ONNX / arm
        res[arm] = {f.name: {"bytes": f.stat().st_size, "sha256": sha256(f)} for f in sorted(d.iterdir()) if f.is_file()}
    say("MODEL_SHA_JSON", res)
    return res


STEPS = ("fetch", "b1", "textonly", "builders", "release", "fidelity", "b3", "smoke", "affinity", "pins")


def build(steps) -> int:
    t0 = time.time()
    env = os.environ.get("CONDA_DEFAULT_ENV", "?")
    print("BUILD_BEGIN", json.dumps({"cmd": f"python[{env}] tools/gemma_decode.py build --steps {','.join(steps)}",
                                     "steps": steps, "env": env, "ram_free_gb": round(free_ram_gb(), 1)}), flush=True)
    paths = fetch()
    mm, by, st, meta = open_release(paths)
    census = {"q4_0": sum(t["tname"] == "Q4_0" for t in by.values()), "f32": sum(t["tname"] == "F32" for t in by.values()),
              "f16": [t["name"] for t in by.values() if t["tname"] == "F16"], "tensors": len(by),
              "rope_scaling": {k: v for k, v in meta.items() if "rope" in k}}
    say("GGUF_CENSUS_JSON", census)
    if "b1" in steps:
        b1(mm, by, st)
        b1_common(mm, by, st)
    if "textonly" in steps:
        text_only(paths, mm, by, st)
    if "builders" in steps:
        run_builders()
    ok = True
    if "release" in steps:
        genai_configs()
        # C0-H16 as built, then N1, then N1+F1+F2, then every arm with all four: the fidelity at each
        # stage attributes the edits (the as-built pass is the gate's optional item 5; only the last
        # stage's C0-H16 rows decide)
        release("C0-H16", mm, by, st)
        if "fidelity" in steps:
            reference_norm_check(mm, by)
            fidelity(mm, by, st, "as built (0.11.2 and ORT 1.23.3; decides nothing)")
            release("C0-H16", mm, by, st, ("N1",))
            fidelity(mm, by, st, "N1 (attributes F1, F2, F3; decides nothing)")
            release("C0-H16", mm, by, st, ("N1", "F1", "F2"))
            fidelity(mm, by, st, "N1+F1+F2 (attributes F3; decides nothing)")
        for arm in ARMS:
            ok &= release(arm, mm, by, st, FIXES)["ok"]
    if "fidelity" in steps:
        f = fidelity(mm, by, st, "N1+F1+F2+F3 (the gate: C0-H16; D-H16 reported, F4)", arms=("C0-H16", "D-H16"))
        ok &= f["pass"]
        if not f["pass"]:
            print(f"FIDELITY_FAIL: max KL above {FID_KL_MAX}; the build is wrong, stopping", flush=True)
            return 3
    if "b3" in steps:
        # a placement that was not read is no placement (build 3's subprocesses exited 0xC0000142
        # before ORT loaded, and that log's BUILD_DONE still read OK)
        unread = [a for a, r in b3().items() if r["status"] == "UNPARSED"]
        if unread:
            print(f"B3_UNREAD {unread}: no DirectML placement was read", flush=True)
        ok &= not unread
    if "smoke" in steps:
        smoke()
    if "affinity" in steps:
        affinity()
    if "pins" in steps:
        pins()
    print("BUILD_DONE", "OK" if ok else "NOT OK (RELEASE_JSON, B3_JSON)", f"{time.time() - t0:.0f} s", flush=True)
    return 0 if ok else 3


# ---------------------------------------------------------------- selftest

def selftest() -> int:
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import TensorProto, helper, numpy_helper
    fails = []

    def expect(what, got, want):
        print(f"  {what}: {got}" + ("" if got == want else f"  (expected {want})"))
        if got != want:
            fails.append(what)

    rng = np.random.default_rng(1)
    # 1. q4_0 blocks -> MatMulNBits, run by ORT, against numpy's d * (q - 8)
    K, N = 96, 40
    codes = rng.integers(0, 16, (N, K // 32, 32)).astype(np.uint8)
    d = rng.standard_normal(N * K // 32).astype(np.float16).view(np.uint16)
    expect("unpack(pack(codes)) == codes", bool((unpack_ort(pack_ort(codes)) == codes).all()), True)
    w = dequant(codes, gc.f16_to_f32(d))                                 # [N, K]
    x = rng.standard_normal((3, K)).astype(np.float32)
    node = helper.make_node("MatMulNBits", ["x", "b", "s"], ["y"], domain="com.microsoft", K=K, N=N, bits=4,
                            block_size=32, accuracy_level=0)
    g = helper.make_graph([node], "t", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, K])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, N])],
                          [numpy_helper.from_array(pack_ort(codes), "b"),
                           numpy_helper.from_array(gc.f16_to_f32(d), "s")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
                          ir_version=9)
    y = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    ref = x.astype(np.float64) @ w.T.astype(np.float64)
    expect("ORT MatMulNBits on repacked q4_0 matches d*(q-8) (rel err < 1e-5)",
           bool(np.abs(y - ref).max() / np.abs(ref).max() < 1e-5), True)
    # B1b's quantizer runs in both dtypes and round-trips (it is called 476 times in the build)
    from onnxruntime.capi._pybind_state import quantize_matmul_4bits
    wq = rng.standard_normal((40, 96)).astype(np.float32)
    for dt in (np.float32, np.float16):
        pk, sc = ort_q4(quantize_matmul_4bits, wq, dt)
        err = np.linalg.norm(dequant(unpack_ort(pk), sc.astype(np.float32)) - wq) / np.linalg.norm(wq)
        expect(f"ORT quantize_matmul_4bits ({np.dtype(dt).name}) round trip rel L2 < 0.2", bool(err < 0.2), True)
    # a q4_0 block row from gguf bytes: low nibble j, high nibble j + 16
    blk = np.zeros((1, 18), dtype=np.uint8)
    blk[0, 2:] = np.arange(16, dtype=np.uint8) | (np.uint8(15) << 4)
    dd, cc = gc.q4_0_split(blk)
    expect("q4_0_split order", cc[0].tolist(), list(range(16)) + [15] * 16)
    # 2. safetensors writer, read back by the safetensors package
    import tempfile
    a = rng.standard_normal((5, 7)).astype(np.float32)
    bq = rng.integers(0, 65535, (3, 4)).astype(np.uint16)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.safetensors"
        write_safetensors(p, [("a", "F32", [5, 7], lambda f: f.write(a.tobytes())),
                              ("b", "BF16", [3, 4], lambda f: f.write(bq.tobytes()))])
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            base = 8 + n
        raw = p.read_bytes()
        ra = np.frombuffer(raw[base + hdr["a"]["data_offsets"][0]:base + hdr["a"]["data_offsets"][1]], np.float32)
        expect("safetensors writer round trip", bool((ra.reshape(5, 7) == a).all()), True)
        import safetensors
        with safetensors.safe_open(str(p), framework="pt") as f:
            expect("safetensors package reads the file", sorted(f.keys()), ["a", "b"])
    # 3. the streamed reference equals transformers' own forward on a tiny Gemma 3
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    cfg = Gemma3TextConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=7,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=32, sliding_window=8,
                           rope_scaling={"rope_type": "linear", "factor": 8.0}, max_position_embeddings=64)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    full = Gemma3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for prm in full.parameters():
            prm.normal_(0, 0.2)
    ids = torch.tensor([rng.integers(0, 97, 20).tolist()])
    with torch.no_grad():
        want = full(input_ids=ids, use_cache=False).logits
    sds = [{k: v.clone() for k, v in lay.state_dict().items()} for lay in full.model.layers]
    import copy
    streamed = copy.deepcopy(full)
    for i, lay in enumerate(streamed.model.layers):
        lay.to("meta")

        def pre(mod, args, kwargs, i=i):
            mod.to_empty(device="cpu")
            mod.load_state_dict(sds[mod.self_attn.layer_idx], strict=True)

        def post(mod, args, kwargs, out):
            mod.to("meta")                                           # returns None: the output is kept

        lay.register_forward_pre_hook(pre, with_kwargs=True)
        lay.register_forward_hook(post, with_kwargs=True)
    with torch.no_grad():
        got = streamed(input_ids=ids, use_cache=False).logits
    expect("streamed layers == transformers forward (20 tokens > window 8)", bool(torch.equal(got, want)), True)
    # 4. KL
    lg = rng.standard_normal((4, 11))
    expect("KL(p||p) == 0", float(np.abs(kl_rows(lg, lg)).max()) < 1e-12, True)
    kl = kl_rows(lg, lg + rng.standard_normal((4, 11)))
    expect("KL > 0 for different logits", bool((kl > 0).all()), True)
    # 5. end to end on a tiny synthetic release: GGUF + checkpoint -> text-only copy -> the wrapped
    #    builder -> the release check (unfixed, fixed) -> the fidelity gate past the sliding window
    tiny_end_to_end(expect)
    print("SELFTEST", "FAIL " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


def tiny_end_to_end(expect) -> None:
    import onnxruntime_genai.models.builder as B
    rng = np.random.default_rng(3)
    H, V, L, HD, NH, KV, FF = 96, 300, 7, 32, 2, 1, 128           # sqrt(96) = 9.798, not 9.8: F2 shows
    tmp = ROOT / "scratch" / "llm" / "gemma_decode_selftest"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shapes = {"attn_q": (NH * HD, H), "attn_k": (KV * HD, H), "attn_v": (KV * HD, H), "attn_output": (H, NH * HD),
              "ffn_gate": (FF, H), "ffn_up": (FF, H), "ffn_down": (H, FF)}
    norm_len = {"attn_norm": H, "post_attention_norm": H, "ffn_norm": H, "post_ffw_norm": H,
                "attn_q_norm": HD, "attn_k_norm": HD}
    gg, st_items = [], []

    def bf16(shape, sd):
        return gc.f32_to_bf16(rng.standard_normal(shape).astype(np.float32) * np.float32(sd))

    def st_add(name, arr):
        st_items.append((name, "BF16", list(arr.shape), lambda f, a=arr: f.write(a.tobytes())))

    head = (rng.standard_normal((V, H)) * 0.05).astype(np.float16)   # logits ~0.5 std: KL sees small changes
    gg.append(("token_embd.weight", [H, V], 1, head.tobytes()))
    st_add("language_model.model.embed_tokens.weight", bf16((V + 4, H), 0.5))
    for layer in range(L):
        for k, (n_out, k_in) in shapes.items():
            codes = rng.integers(0, 16, (n_out * k_in // 32, 32)).astype(np.uint8)
            d = (rng.standard_normal(len(codes)) * 0.03).astype(np.float16)
            raw = np.concatenate([d.view(np.uint8).reshape(-1, 2), codes[:, :16] | (codes[:, 16:] << 4)], axis=1)
            gg.append((f"blk.{layer}.{k}.weight", [k_in, n_out], 2, raw.astype(np.uint8).tobytes()))
            blk, proj = GG_PROJ[k]
            st_add(f"language_model.model.layers.{layer}.{blk}.{proj}.weight", bf16((n_out, k_in), 0.02))
        for k, hf in GG_NORM.items():                           # the GGUF's norms: checkpoint + 1, nudged, as in the
            w = bf16((norm_len[k],), 0.3)                         # release (N1 has something to do)
            st_add(f"language_model.model.layers.{layer}.{hf}.weight", w)
            g1 = gc.bf16_to_f32(w) + np.float32(1) + (rng.standard_normal(norm_len[k]) * 0.01).astype(np.float32)
            gg.append((f"blk.{layer}.{k}.weight", [norm_len[k]], 0, g1.astype(np.float32).tobytes()))
    w = bf16((H,), 0.3)
    st_add("language_model.model.norm.weight", w)
    g1 = gc.bf16_to_f32(w) + np.float32(1) + (rng.standard_normal(H) * 0.01).astype(np.float32)
    gg.append(("output_norm.weight", [H], 0, g1.astype(np.float32).tobytes()))
    st_add("vision_tower.dummy.weight", bf16((4,), 1.0))
    gc.write_gguf(tmp / "tiny.gguf", gg)
    write_safetensors(tmp / "tiny.safetensors", st_items)
    cfg = {"architectures": ["Gemma3ForConditionalGeneration"], "eos_token_id": [1, 106], "image_token_index": V,
           "text_config": {"model_type": "gemma3_text", "hidden_size": H, "intermediate_size": FF,
                           "num_hidden_layers": L, "num_attention_heads": NH, "num_key_value_heads": KV, "head_dim": HD,
                           "vocab_size": V + 4, "sliding_window": 8, "sliding_window_pattern": 6,
                           "max_position_embeddings": 64, "rope_theta": 1000000, "rope_local_base_freq": 10000,
                           "rope_scaling": {"factor": 8.0, "rope_type": "linear"}, "query_pre_attn_scalar": HD,
                           "rms_norm_eps": 1e-6, "hidden_activation": "gelu_pytorch_tanh",
                           "attn_logit_softcapping": None, "final_logit_softcapping": None}}
    (tmp / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    paths = {"gguf": tmp / "tiny.gguf", "config.json": tmp / "config.json",
             "model-00001-of-00002.safetensors": tmp / "tiny.safetensors"}
    g = globals()
    keep = {k: g[k] for k in ("HIDDEN", "VOCAB", "LAYERS", "TEXT", "ONNX", "CACHE", "FID_LENGTHS", "FID_MIN_FREE_GB")}
    hooks = (B.Model.make_genai_config, B.Model.save_processing)
    g.update(HIDDEN=H, VOCAB=V, LAYERS=L, TEXT=tmp / "text_only", ONNX=tmp / "onnx", CACHE=tmp / "cache",
             FID_LENGTHS=(6, 20), FID_MIN_FREE_GB=0)
    B.Model.make_genai_config = B.Model.save_processing = lambda *a, **k: None   # no tokenizer in the tiny copy
    _REF.clear()
    try:
        mm, by, st, _ = open_release(paths)
        rb = b1(mm, by, st)
        rc = b1_common(mm, by, st)
        expect("tiny: B1 and its post-hoc check run over every weight",
               (rb["weights"], rb["head_n"], rb["norm_n"], rc["linears"]["n"], rc["head"]["n"], rc["norms"]["n"]),
               (sum(a * b for a, b in shapes.values()) * L, V * H, 3232, sum(a * b for a, b in shapes.values()) * L, V * H, 3232))
        text_only(paths, mm, by, st)
        for arm in ("C0-H16", "C0-H4"):
            builder_run(arm, paths)
        sq = float(np.float32(np.sqrt(96.0)))
        r0 = release("C0-H16", mm, by, st)
        expect("tiny: C0-H16 as built reads the release's linears, embedding and F16 head exactly",
               (r0["linears"]["exact"] == r0["linears"]["weights"], r0["embedding"]["exact"] == r0["embedding"]["n"],
                r0["head"]["exact"] == r0["head"]["n"]), (True, True, True))
        expect("tiny: the norms as built are the checkpoint's + 1, not the release's", (r0["ok"], r0["norms"]["injected"]),
               (False, 0))
        expect("tiny: F1 shows (global RoPE cache differs from transformers')",
               r0["rope"]["cos_cache_global"]["max_abs_diff"] > 1e-3, True)
        expect("tiny: local RoPE cache matches transformers' (< 1e-6)",
               r0["rope"]["cos_cache_local"]["max_abs_diff"] < 1e-6 and r0["rope"]["sin_cache_local"]["max_abs_diff"] < 1e-6, True)
        expect("tiny: F2 shows (embed scale 9.8 as built)", round(r0["embed_scale"]["value"], 4), 9.8)
        expect("tiny: GQA windows as built (6 local at 8, 1 global)", r0["gqa_windows"], {"-1": 1, "8": 6})
        rr = reference_norm_check(mm, by)
        expect("tiny: the reference's norms are the GGUF's, bit for bit, and its forward is norm(x) * g",
               (rr["n"], rr["bit_exact"], rr["forward_rel_err"] < 1e-6), (3232, 3232, True))
        f0 = fidelity(mm, by, st, "tiny as built")
        rn = release("C0-H16", mm, by, st, ("N1",))
        expect("tiny: N1 writes the release's norms, read back", (rn["ok"], rn["norms"]["injected"]), (True, 43))
        fn = fidelity(mm, by, st, "tiny N1")
        r12 = release("C0-H16", mm, by, st, ("N1", "F1", "F2"))
        expect("tiny: N1+F1+F2 applied and read back", (r12["ok"], r12["embed_scale"]["on_disk"]), (True, sq))
        f12 = fidelity(mm, by, st, "tiny N1+F1+F2")
        r1 = release("C0-H16", mm, by, st, FIXES)
        expect("tiny: F3 applied (local windows 7)", (r1["ok"], r1["gqa_windows"]), (True, {"-1": 1, "7": 6}))
        r2 = release("C0-H16", mm, by, st)
        expect("tiny: the edits persist on disk", (r2["ok"], r2["rope"]["cos_cache_global"]["max_abs_diff"],
                                                   r2["embed_scale"]["value"], r2["gqa_windows"]),
               (False, 0.0, sq, {"-1": 1, "7": 6}))         # ok is False only because F3's windows are not asked for
        f1 = fidelity(mm, by, st, "tiny all")
        worst = [max(p["max_kl"] for p in f["passes"]) for f in (f0, fn, f12, f1)]
        inside = [max(p["max_kl_within_window"] for p in f["passes"]) for f in (f0, fn, f12, f1)]
        past12 = max(p["max_kl_past_window"] or 0.0 for p in f12["passes"])
        print(f"  tiny: max KL as built {worst[0]:.3g}, N1 {worst[1]:.3g}, N1+F1+F2 {worst[2]:.3g} (within the window "
              f"{inside[2]:.3g}, past it {past12:.3g}), all {worst[3]:.3g}")
        expect("tiny: all edits: C0-H16 == transformers past the window (max KL <= 1e-9)", worst[3] <= 1e-9, True)
        expect("tiny: N1 matters within the window (as built > 100 x N1+F1+F2 there)", inside[0] > 100 * max(inside[2], 1e-15),
               True)
        expect("tiny: F1+F2 matter within the window (N1 > 1000 x N1+F1+F2 there)", inside[1] > 1000 * max(inside[2], 1e-15),
               True)
        expect("tiny: F3 matters only past the window (N1+F1+F2: within <= 1e-9, past > 1000 x all)",
               (inside[2] <= 1e-9, past12 > 1000 * worst[3]), (True, True))
        r3 = release("C0-H4", mm, by, st, FIXES)
        expect("tiny: C0-H4 reads the release, head int4 by the builder",
               (r3["ok"], r3["head"]["format"], r3["head"]["op"]), (True, "H4", "MatMulNBits"))
    finally:
        g.update(keep)
        B.Model.make_genai_config, B.Model.save_processing = hooks
        _REF.clear()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("build", "builder", "placement", "selftest"))
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--arm", dest="arm_opt", choices=tuple(ARMS))
    ap.add_argument("--steps", default=",".join(STEPS))
    a = ap.parse_args()
    if a.mode == "builder":
        return builder_run(a.arm_opt)
    if a.mode == "placement":
        return placement(a.arm)
    if a.mode == "selftest":
        return selftest()
    steps = [s for s in a.steps.split(",") if s]
    bad = set(steps) - set(STEPS)
    if bad:
        sys.exit(f"unknown steps {sorted(bad)}")
    return build(steps)


if __name__ == "__main__":
    sys.exit(main())
