#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM study, Gemma 3 4B, pre-registration (a): how far the QAT q4_0 weights compress losslessly.

The shipped q4_0 GGUF of Gemma 3 4B streams 3,147,038,720 B per decoded token: the 238 q4_0
linears (1,804,861,440 B at 4.5 bits) and the tied F16 head (1,342,177,280 B). Decode reads every
one of those bytes per token, so a lossless coder that shrinks them cuts decode time on any chip
that can decode at stream rate. Offline and on the CPU only, this tool measures:
  Q1  the q4_0 layout and grid: the GGUF's codes and scales against the unquantized QAT checkpoint,
      the q4_0 rule on the checkpoint's embedding, and the GGUF's F16 head against that embedding;
  Q2  order-0 entropies (on paper);
  Q3  real coders, each round-tripped to the exact input, and the pre-registered kill line.

    python tools/gemma_compress.py prereg          # the pre-registration text (no network)
    python tools/gemma_compress.py fetch           # the pinned files into the HF cache, SHA-256 verified
    python tools/gemma_compress.py run             # fetch and verify, then Q1-Q3 (resnet_env17; heavy CPU)
    python tools/gemma_compress.py verdict RUNLOG  # the mechanical verdict from the run log
    python tools/gemma_compress.py selftest        # synthetic GGUF, grid, entropy, coders and verdicts

Runner: scripts/llm-study.sh compress-prereg | compress | compress-verdict.
"""
import argparse
import heapq
import hashlib
import json
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- pins (the Hub's LFS metadata)

GGUF_PIN = {"repo": "google/gemma-3-4b-it-qat-q4_0-gguf", "revision": "15f73f5eee9c28f53afefef5723e29680c2fc78a",
            "file": "gemma-3-4b-it-q4_0.gguf", "size": 3155051328,
            "sha256": "76aed0a8285b83102f18b5d60e53c70d09eb4e9917a20ce8956bd546452b56e2"}
ST_REPO, ST_REV = "google/gemma-3-4b-it-qat-int4-unquantized", "554bd242505753eef6dfae71f76ddd50c335fc46"
ST_PINS = [{"repo": ST_REPO, "revision": ST_REV, "file": "model-00001-of-00002.safetensors", "size": 4961251752,
            "sha256": "069e91c41faa7febba9c62cf97b9167acfc87f42b67d6061c52a89fed93cb56c"},
           {"repo": ST_REPO, "revision": ST_REV, "file": "model-00002-of-00002.safetensors", "size": 3639026128,
            "sha256": "efcb40cfc23a68ed0f45e1d089373f351fcdc050dee639481171535de243d636"}]
PINS = [GGUF_PIN] + ST_PINS

# ---------------------------------------------------------------- constants (SPEC from the headers, DERIVED)

N_LIN, N_LIN_TENSORS = 3_208_642_560, 238   # the 34 x 7 q4_0 linears
N_HEAD = 671_088_640                        # token_embd.weight [2560, 262144] F16, the tied head
HIDDEN, HEAD_ROWS, ST_EMBED_ROWS = 2560, 262_144, 262_208
SHIPPED_L = N_LIN // 32 * 18                # 1,804,861,440 B
SHIPPED_H = N_HEAD * 2                      # 1,342,177,280 B
SHIPPED_T = SHIPPED_L + SHIPPED_H           # 3,147,038,720 B per token as shipped
MARGIN_NUM, MARGIN_DEN = 11, 10             # >= 1.10x fewer bytes; exactly 1.10x passes
K1_BITS = 4.5 * MARGIN_DEN / MARGIN_NUM     # 4.0909 bits/weight
K2_BYTES = SHIPPED_T * MARGIN_DEN / MARGIN_NUM   # 2,860,944,290.9 B/token
REF_BLOCK128 = 4.125                        # block-128 f16 scales: a lossy re-quantization, reported only
CHUNK = 1 << 20                             # symbols per independently coded Huffman chunk
HUFF_MAX_LEN = 20                           # length-limited canonical Huffman
ZSTD_LEVEL, ZSTD_THREADS = 19, 8
LEN_FIELD, CHUNK_FIELD = 8, 4               # bytes: a stream's symbol count; one chunk's byte length
CORES, CLOCK_GHZ, READ_GBPS = 16, 1.7972, 47.62   # SILICON:252-258 and :63; MEASURED clock and read rate

PRED = {"P1": (0.999, None), "P2": (1.0, None), "P3": (3.60, 3.85), "P4": (0.28, 0.38), "P5": (3.95, 4.20),
        "P6": (13.0, 14.2), "P7": (0.08, 0.13), "P8": (4.3, None), "P9": (None, 0.10), "P10": (0.999, None)}

GG2HF = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
         "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
         "ffn_down": "mlp.down_proj"}


def hf_name(gg: str) -> str:
    if gg == "token_embd.weight":
        return "language_model.model.embed_tokens.weight"
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", gg)
    return f"language_model.model.layers.{m.group(1)}.{GG2HF[m.group(2)]}.weight"


def k_pass(shipped: int, coded: int) -> bool:
    """The kill line: at least 1.10x fewer bytes, in integers so exactly 1.10x passes."""
    return shipped * MARGIN_DEN >= coded * MARGIN_NUM


# ---------------------------------------------------------------- files

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fetch() -> dict:
    """Download (or find in the HF cache) every pinned file at its pinned revision; verify size and
    SHA-256. Returns {file: path}; exits non-zero on any mismatch."""
    from huggingface_hub import hf_hub_download
    paths, rows, ok = {}, [], True
    for p in PINS:
        t0 = time.time()
        path = Path(hf_hub_download(p["repo"], p["file"], revision=p["revision"]))
        size, sha = path.stat().st_size, sha256_file(path)
        good = size == p["size"] and sha == p["sha256"]
        ok &= good
        rows.append({"file": p["file"], "revision": p["revision"], "size": size, "sha256": sha, "ok": good,
                     "seconds": round(time.time() - t0, 1)})
        paths[p["file"]] = path
        print(f"FETCH {p['file']}  {size} B  sha256 {sha}  {'matches the pin' if good else 'DIFFERS FROM THE PIN'}")
    print("FETCH_JSON " + json.dumps({"ok": ok, "files": rows}))
    if not ok:
        sys.exit("a fetched file differs from its pin; nothing is measured")
    return paths


GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18)}   # name, elements per block, bytes per block
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_gguf(path):
    """Parse a GGUF v3 header in place (memory-mapped). Returns (mm, version, meta, tensors), each
    tensor with its absolute byte start and length."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, mm, pos)
        pos += struct.calcsize("<" + fmt)
        return v[0]

    def string():
        nonlocal pos
        n = take("Q")
        s = bytes(mm[pos:pos + n]).decode("utf-8", "replace")
        pos += n
        return s

    def value(t):
        if t in SCALAR:
            return take(SCALAR[t])
        if t == 8:
            return string()
        if t == 9:
            et, n = take("I"), take("Q")
            items = [value(et) for _ in range(n)]
            return items if n <= 16 else f"<array of {n}>"
        raise ValueError(f"GGUF value type {t}")

    if bytes(mm[:4]) != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    pos = 4
    version, n_t, n_kv = take("I"), take("Q"), take("Q")
    meta = {}
    for _ in range(n_kv):
        k = string()
        meta[k] = value(take("I"))
    tensors = []
    for _ in range(n_t):
        name = string()
        dims = [take("Q") for _ in range(take("I"))]
        tensors.append({"name": name, "dims": dims, "type": take("I"), "offset": take("Q")})
    align = int(meta.get("general.alignment", 32))
    data = (pos + align - 1) // align * align
    for t in tensors:
        n = int(np.prod(t["dims"], dtype=np.int64))
        tname, per, nb = GGML.get(t["type"], (str(t["type"]), 0, 0))
        t.update(n=n, tname=tname, start=data + t["offset"], nbytes=n // per * nb if per else None)
    return mm, version, meta, tensors


def read_st(paths) -> dict:
    """Safetensors headers of every shard: {name: entry with absolute byte range}."""
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = {"path": p, "dtype": v["dtype"], "shape": v["shape"],
                          "begin": 8 + n + v["data_offsets"][0], "end": 8 + n + v["data_offsets"][1]}
    return out


def st_u16(e) -> np.ndarray:
    assert e["dtype"] == "BF16", e["dtype"]
    return np.memmap(e["path"], dtype=np.uint16, mode="r", offset=e["begin"], shape=((e["end"] - e["begin"]) // 2,))


# ---------------------------------------------------------------- number formats

def f16_to_f32(u16):
    return np.asarray(u16, dtype=np.uint16).view(np.float16).astype(np.float32)


def bf16_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(v):
    b = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)
    return ((b + np.uint32(0x7FFF) + ((b >> 16) & np.uint32(1))) >> 16).astype(np.uint16)   # round to nearest even


def q4_0_split(blocks):
    """ggml's block_q4_0: an f16 d, then 16 bytes; the low nibble is element j, the high nibble j + 16."""
    d = blocks[:, :2].copy().view(np.uint16).ravel()
    qs = blocks[:, 2:]
    codes = np.empty((len(blocks), 32), dtype=np.uint8)
    codes[:, :16] = qs & 0x0F
    codes[:, 16:] = qs >> 4
    return d, codes


def q4_0_rule(w):
    """llama.cpp's reference q4_0 quantizer on fp32 rows of 32: the first signed extreme sets
    d = max / -8 (fp32, stored f16), q = min(15, trunc(x / d + 8.5)). Returns (d_f16_as_f32, codes)."""
    idx = np.abs(w).argmax(axis=1)
    mx = w[np.arange(len(w)), idx]
    d = (mx / np.float32(-8)).astype(np.float32)
    inv = np.where(d != 0, np.float32(1) / np.where(d != 0, d, np.float32(1)), np.float32(0)).astype(np.float32)
    q = np.minimum(15, np.trunc(w * inv[:, None] + np.float32(8.5))).astype(np.int16)
    return d.astype(np.float16).astype(np.float32), q


def entropy(counts) -> float:
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def sym_entropy(x) -> float:
    return entropy(np.unique(np.asarray(x), return_counts=True)[1])


# ---------------------------------------------------------------- coders

def huff_lengths(counts, max_len=HUFF_MAX_LEN) -> np.ndarray:
    """Huffman code lengths, limited to max_len by flattening the rarest counts until the code fits
    (the cost of the limit is in the coded size)."""
    counts = np.asarray(counts, dtype=np.int64)
    present = [int(s) for s in np.flatnonzero(counts)]
    lens = np.zeros(len(counts), dtype=np.int64)
    if not present:
        return lens
    if len(present) == 1:
        lens[present[0]] = 1
        return lens

    def build(c):
        heap = [(int(c[s]), i, [s]) for i, s in enumerate(present)]
        heapq.heapify(heap)
        depth = dict.fromkeys(present, 0)
        uid = len(heap)
        while len(heap) > 1:
            a, b = heapq.heappop(heap), heapq.heappop(heap)
            for s in a[2] + b[2]:
                depth[s] += 1
            heapq.heappush(heap, (a[0] + b[0], uid, a[2] + b[2]))
            uid += 1
        out = np.zeros(len(c), dtype=np.int64)
        for s, dd in depth.items():
            out[s] = dd
        return out

    lens = build(counts)
    total, shift = int(counts.sum()), max_len - 1
    while lens.max() > max_len:
        floor = max(1, total >> shift)
        lens = build(np.where(counts > 0, np.maximum(counts, floor), 0))
        shift -= 1
    return lens


def canonical(lens) -> np.ndarray:
    codes = np.zeros(len(lens), dtype=np.uint32)
    code, prev = 0, 0
    for s in sorted(np.flatnonzero(lens), key=lambda s: (lens[s], s)):
        code <<= int(lens[s]) - prev
        prev = int(lens[s])
        codes[s] = code
        code += 1
    return codes


def decode_table(lens, width):
    """The decoder's table, built from the stored code lengths alone: every width-bit window names a
    symbol and its length."""
    sym = np.full(1 << width, 0xFFFF, dtype=np.uint16)
    ln = np.zeros(1 << width, dtype=np.int32)
    codes = canonical(lens)
    for s in np.flatnonzero(lens):
        n = int(lens[s])
        lo, hi = int(codes[s]) << (width - n), (int(codes[s]) + 1) << (width - n)
        sym[lo:hi], ln[lo:hi] = s, n
    return sym, ln


def huff_stream(x, nsym: int, fault_chunk=None):
    """One static canonical Huffman table for the symbol stream x, coded in CHUNK-symbol chunks each
    padded to a byte, and every chunk decoded back from its own bytes: the width-bit window at each
    code start must name that code's symbol and length, which by induction is what a sequential
    decoder produces. Returns (bytes: payload + table + length fields, round-trip ok)."""
    x = np.asarray(x)
    lens = huff_lengths(np.bincount(x.ravel(), minlength=nsym))
    codes = canonical(lens)
    width = int(lens.max())
    dsym, dln = decode_table(lens, width)
    lens32 = lens.astype(np.int32)
    flat = x.ravel()
    n = len(flat)
    nchunks = (n + CHUNK - 1) // CHUNK
    payload, ok = 0, True
    for c in range(nchunks):
        xs = flat[c * CHUNK:(c + 1) * CHUNK]
        ln = lens32[xs]
        ends = np.cumsum(ln, dtype=np.int64)
        total = int(ends[-1])
        starts = ends - ln
        rep = np.repeat(np.arange(len(xs), dtype=np.int32), ln)
        shift = (ln[rep] - 1 - (np.arange(total, dtype=np.int64) - starts[rep])).astype(np.uint32)
        packed = np.packbits(((codes[xs][rep] >> shift) & np.uint32(1)).astype(np.uint8))
        del rep, shift
        if fault_chunk == c:
            packed[len(packed) // 2] ^= np.uint8(0x10)
        payload += len(packed)
        buf = np.concatenate([packed, np.zeros(4, dtype=np.uint8)]).astype(np.uint32)
        b0, sh = starts >> 3, (starts & 7).astype(np.uint32)
        w = (buf[b0] << 24) | (buf[b0 + 1] << 16) | (buf[b0 + 2] << 8) | buf[b0 + 3]
        win = (w >> (np.uint32(32 - width) - sh)) & np.uint32((1 << width) - 1)
        ok &= bool(np.array_equal(dsym[win], xs) and np.array_equal(dln[win], ln) and len(packed) == (total + 7) // 8)
    return payload + nsym + LEN_FIELD + CHUNK_FIELD * nchunks, ok


def zstd_size(arr):
    """zstd level 19 (8 worker threads) on the array's bytes, decompressed back and compared.
    Returns (frame bytes + the length field, round-trip ok)."""
    import zstandard
    a = np.ascontiguousarray(arr)
    raw = memoryview(a).cast("B")
    comp = zstandard.ZstdCompressor(level=ZSTD_LEVEL, threads=ZSTD_THREADS).compress(raw)
    back = zstandard.ZstdDecompressor().decompress(comp, max_output_size=raw.nbytes)
    ok = len(back) == raw.nbytes and np.array_equal(np.frombuffer(back, np.uint8), np.frombuffer(raw, np.uint8))
    return len(comp) + LEN_FIELD, bool(ok)


# ---------------------------------------------------------------- the run

def grid_linear(d_u16, codes, hf_u16, step=1 << 16):
    """Q1 on one linear: bf16(d * (q - 8)) against the checkpoint's BF16 values, as values (+0 = -0).
    Returns (exact matches, max |diff| / |d| over blocks with d != 0)."""
    exact, worst = 0, 0.0
    d = f16_to_f32(d_u16)
    for i in range(0, len(d), step):
        dd = d[i:i + step]
        v = (dd[:, None] * (codes[i:i + step].astype(np.float32) - np.float32(8))).ravel()
        w = bf16_to_f32(np.asarray(hf_u16[i * 32:(i + step) * 32]))
        exact += int((bf16_to_f32(f32_to_bf16(v)) == w).sum())
        nz = np.repeat(dd != 0, 32)
        if nz.any():
            worst = max(worst, float((np.abs(w - v)[nz] / np.abs(np.repeat(dd, 32)[nz])).max()))
    return exact, worst


def run() -> int:
    paths = fetch()
    mm, version, meta, tensors = read_gguf(paths[GGUF_PIN["file"]])
    st = read_st([paths[p["file"]] for p in ST_PINS])
    lin = [t for t in tensors if t["tname"] == "Q4_0"]
    head = [t for t in tensors if t["name"] == "token_embd.weight"]
    f32 = [t for t in tensors if t["tname"] == "F32"]
    census = {"version": version, "tensors": len(tensors), "q4_0": len(lin), "q4_0_params": sum(t["n"] for t in lin),
              "head": [head[0]["tname"], head[0]["dims"]] if head else None, "f32": len(f32),
              "f32_params": sum(t["n"] for t in f32), "alignment": int(meta.get("general.alignment", 32)),
              "other": sorted({t["tname"] for t in tensors} - {"Q4_0", "F16", "F32"})}
    census["ok"] = (census["q4_0"] == N_LIN_TENSORS and census["q4_0_params"] == N_LIN and len(head) == 1
                    and head[0]["tname"] == "F16" and head[0]["n"] == N_HEAD and not census["other"]
                    and all(hf_name(t["name"]) in st for t in lin + head))
    print("CENSUS_JSON " + json.dumps(census))
    if not census["ok"]:
        print("the GGUF's census differs from the pinned header facts; nothing is measured")
        return 1
    t_start = time.time()
    # zstd 19 on a tensor of at most 26 MB is one job, so one core: the linears' zstd jobs run 8 at a time
    # in a thread pool (zstandard releases the GIL). Scheduling only; each frame is what it would be alone.
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=ZSTD_THREADS)
    pending = []
    for k, t in enumerate(lin):
        t0 = time.time()
        e = st[hf_name(t["name"])]
        assert e["shape"] == t["dims"][::-1], (t["name"], e["shape"], t["dims"])
        nb = t["n"] // 32
        blocks = np.asarray(mm[t["start"]:t["start"] + nb * 18]).reshape(nb, 18)
        d, codes = q4_0_split(blocks)
        futs = [pool.submit(zstd_size, a) for a in (blocks, codes, d)]
        exact, worst = grid_linear(d, codes, st_u16(e))
        counts = np.bincount(codes.ravel(), minlength=16)
        h_codes, hc_ok = huff_stream(codes, 16)
        h_hi, hh_ok = huff_stream((d >> 8).astype(np.uint8), 256)
        row = {"name": t["name"], "dims": t["dims"], "n": t["n"], "blocks": nb,
               "q1_exact": exact, "q1_max_diff_over_d": worst,
               "q0_blocks": int((codes == 0).any(axis=1).sum()), "zero_d_blocks": int(((d & 0x7FFF) == 0).sum()),
               "code_counts": counts.tolist(), "h_codes": entropy(counts),
               "h_scale16": sym_entropy(d), "h_scale_se": sym_entropy(d >> 10), "h_scale_man": sym_entropy(d & 0x3FF),
               "c3": h_codes + 2 * nb, "c3_ok": hc_ok, "c4": h_codes + h_hi + nb, "c4_ok": hc_ok and hh_ok,
               "seconds_main": round(time.time() - t0, 1)}
        pending.append((row, futs))
        print(f"[{k + 1}/{len(lin)}] {t['name']}  grid {exact / t['n']:.6f}  H0 {row['h_codes']:.4f}  "
              f"C3 {row['c3'] * 8 / t['n']:.4f}  C4 {row['c4'] * 8 / t['n']:.4f} bits/weight  "
              f"({row['seconds_main']} s, {time.time() - t_start:.0f} s so far)", flush=True)
    for row, futs in pending:
        (c1, c1_ok), (z_codes, zc_ok), (z_scales, zs_ok) = (f.result() for f in futs)
        row.update(c1=c1, c1_ok=c1_ok, c2=z_codes + z_scales, c2_ok=zc_ok and zs_ok)
        print("ROW_JSON " + json.dumps(row))
    pool.shutdown()
    print(f"linears done, {time.time() - t_start:.0f} s", flush=True)
    # the head: Q2, the coders, Q1c against the checkpoint's embedding, and the q4_0 rule on that embedding
    t0 = time.time()
    h = np.asarray(mm[head[0]["start"]:head[0]["start"] + SHIPPED_H]).view(np.uint16)
    hi, lo = (h >> 8).astype(np.uint8), (h & 0xFF).astype(np.uint8)
    c1, c1_ok = zstd_size(h)
    z_hi, zh_ok = zstd_size(hi)
    z_lo, zl_ok = zstd_size(lo)
    c5_hi, c5_ok = huff_stream(hi, 256)
    emb = st_u16(st[hf_name("token_embd.weight")])
    q1c = q1b = 0
    rows_step = 8192
    for r in range(0, ST_EMBED_ROWS, rows_step):
        w16 = np.asarray(emb[r * HIDDEN:min(r + rows_step, ST_EMBED_ROWS) * HIDDEN])
        w = bf16_to_f32(w16)
        if r < HEAD_ROWS:
            m = min(r + rows_step, HEAD_ROWS) - r
            g = h[r * HIDDEN:(r + m) * HIDDEN].view(np.float16)
            q1c += int((w[:m * HIDDEN].astype(np.float16) == g).sum())
        d32, q = q4_0_rule(w.reshape(-1, 32))
        v = (d32[:, None] * (q.astype(np.float32) - np.float32(8))).ravel()
        q1b += int((bf16_to_f32(f32_to_bf16(v)) == w).sum())
    row = {"name": "token_embd.weight", "n": N_HEAD, "h_hi": sym_entropy(hi), "h_lo": sym_entropy(lo),
           "h_se": sym_entropy(h >> 10), "c1": c1, "c1_ok": c1_ok, "c2": z_hi + z_lo, "c2_ok": zh_ok and zl_ok,
           "c5": c5_hi + N_HEAD, "c5_ok": c5_ok, "q1c_exact": q1c, "q1c_n": HEAD_ROWS * HIDDEN,
           "q1b_exact": q1b, "q1b_n": ST_EMBED_ROWS * HIDDEN, "seconds": round(time.time() - t0, 1)}
    print("HEAD_JSON " + json.dumps(row))
    print(f"head: C5 {row['c5'] * 8 / N_HEAD:.4f} bits/weight, Q1c {q1c / row['q1c_n']:.6f}, "
          f"q4_0 rule on the embedding {q1b / row['q1b_n']:.6f} ({row['seconds']} s)")
    print(f"RUN_DONE {time.time() - t_start:.0f} s")
    return 0


# ---------------------------------------------------------------- the verdict

def score(rows, headrow, fetch_ok, census_ok):
    """The verdict from the run's rows. Returns (verdict, facts)."""
    f = {"fetch_ok": fetch_ok, "census_ok": census_ok, "linears": len(rows), "head": headrow is not None}
    if not (fetch_ok and census_ok and len(rows) == N_LIN_TENSORS and headrow is not None
            and sum(r["n"] for r in rows) == N_LIN and headrow["n"] == N_HEAD):
        return "INCOMPLETE", f
    tot = {k: sum(r[k] for r in rows) for k in ("c1", "c2", "c3", "c4", "q1_exact", "blocks", "q0_blocks",
                                               "zero_d_blocks")}
    ok = {k: all(r[k + "_ok"] for r in rows) for k in ("c1", "c2", "c3", "c4")}
    ok["c5"], ok["h1"], ok["h2"] = headrow["c5_ok"], headrow["c1_ok"], headrow["c2_ok"]
    counts = np.sum([r["code_counts"] for r in rows], axis=0)
    f.update(tot)
    f["ok"] = ok
    f["q1_frac"] = tot["q1_exact"] / N_LIN
    f["q0_frac"] = tot["q0_blocks"] / tot["blocks"]
    f["h_codes_pooled"] = entropy(counts)
    f["h_codes_weighted"] = sum(r["h_codes"] * r["n"] for r in rows) / N_LIN
    f["scale_bits"] = sum(r["h_scale16"] * r["blocks"] for r in rows) / N_LIN
    f["scale_split_bits"] = sum((r["h_scale_se"] + r["h_scale_man"]) * r["blocks"] for r in rows) / N_LIN
    bits = {k: tot[k] * 8 / N_LIN for k in ("c1", "c2", "c3", "c4")}
    bits.update(h1=headrow["c1"] * 8 / N_HEAD, h2=headrow["c2"] * 8 / N_HEAD, c5=headrow["c5"] * 8 / N_HEAD)
    f["bits"] = bits
    f["head_entropy"] = {"hi": headrow["h_hi"], "lo": headrow["h_lo"], "se": headrow["h_se"]}
    f["q1c_frac"] = headrow["q1c_exact"] / headrow["q1c_n"]
    f["q1b_frac"] = headrow["q1b_exact"] / headrow["q1b_n"]
    best_l = min(tot["c3"], tot["c4"])
    f["best_l"], f["best_l_bits"] = best_l, best_l * 8 / N_LIN
    f["t_bytes"] = best_l + headrow["c5"]
    f["t_saving"] = 1 - f["t_bytes"] / SHIPPED_T
    f["t_bits"] = f["t_bytes"] * 8 / (N_LIN + N_HEAD)
    f["k1"] = k_pass(SHIPPED_L, best_l)
    f["k2"] = k_pass(SHIPPED_T, f["t_bytes"])
    if not (ok["c3"] and ok["c4"] and ok["c5"]):
        return "INCOMPLETE", f
    return ("SURVIVES" if f["k1"] or f["k2"] else "KILLED"), f


def predictions(f) -> list:
    def inside(v, lo, hi):
        return (lo is None or v >= lo) and (hi is None or v <= hi)
    rows = [("P1", "≥ 99.9% of linear weights match the grid exactly", f["q1_frac"], inside(f["q1_frac"], *PRED["P1"])),
            ("P2", "100% of blocks hold a q = 0 code", f["q0_frac"], f["q0_frac"] == 1.0),
            ("P3", "pooled H0 of the codes is 3.60-3.85 bits", f["h_codes_pooled"], inside(f["h_codes_pooled"], *PRED["P3"])),
            ("P4", "scales cost 0.28-0.38 bits/weight at order 0", f["scale_bits"], inside(f["scale_bits"], *PRED["P4"])),
            ("P5 range", "the best C3/C4 on L is 3.95-4.20 bits/weight", f["best_l_bits"],
             inside(f["best_l_bits"], *PRED["P5"])),
            ("P5 outcome", "K1 FAILS", f["k1"], not f["k1"]),
            ("P6", "C5 codes the head to 13.0-14.2 bits/weight", f["bits"]["c5"], inside(f["bits"]["c5"], *PRED["P6"])),
            ("P7 range", "K2 saves 8-13% of 3.147 GB", f["t_saving"], inside(f["t_saving"], *PRED["P7"])),
            ("P7 outcome", "K2 PASSES", f["k2"], f["k2"]),
            ("P8", "C1 on the raw blocks is ≥ 4.3 bits/weight", f["bits"]["c1"], inside(f["bits"]["c1"], *PRED["P8"])),
            ("P9", "the embedding is not on an int4 grid (< 10% exact)", f["q1b_frac"], f["q1b_frac"] < PRED["P9"][1]),
            ("P10", "Q1c: the F16 head equals bf16 -> f16 of the embedding for ≥ 99.9%", f["q1c_frac"],
             inside(f["q1c_frac"], *PRED["P10"]))]
    return rows


def verdict(path) -> int:
    rows, headrow, fetch_ok, census_ok = [], None, False, False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("FETCH_JSON "):
            fetch_ok = json.loads(line[11:])["ok"]
        elif line.startswith("CENSUS_JSON "):
            census_ok = json.loads(line[12:])["ok"]
        elif line.startswith("ROW_JSON "):
            rows.append(json.loads(line[9:]))
        elif line.startswith("HEAD_JSON "):
            headrow = json.loads(line[10:])
    v, f = score(rows, headrow, fetch_ok, census_ok)
    print(f"inputs: fetch {'verified' if f['fetch_ok'] else 'NOT VERIFIED'}, census {'ok' if f['census_ok'] else 'NOT OK'}, "
          f"{f['linears']} of {N_LIN_TENSORS} linears, head row {'present' if f['head'] else 'MISSING'}")
    if "bits" in f:
        b = f["bits"]
        print(f"Q1 grid: {f['q1_exact']:,} of {N_LIN:,} linear weights exact ({f['q1_frac']:.6f}); "
              f"blocks with a q = 0 code {f['q0_blocks']:,} of {f['blocks']:,} ({f['q0_frac']:.6f}); "
              f"blocks with d = 0 {f['zero_d_blocks']:,}")
        print(f"Q1c: the GGUF's F16 head equals f16(bf16 embedding) for {f['q1c_frac']:.6f} of the shared rows; "
              f"the q4_0 rule reproduces the embedding for {f['q1b_frac']:.6f}")
        print(f"Q2 (on paper): codes H0 pooled {f['h_codes_pooled']:.4f}, per tensor weighted {f['h_codes_weighted']:.4f} bits; "
              f"scales {f['scale_bits']:.4f} bits/weight as 16-bit symbols ({f['scale_split_bits']:.4f} split "
              f"sign+exponent / mantissa); head high byte {f['head_entropy']['hi']:.4f}, low byte "
              f"{f['head_entropy']['lo']:.4f}, sign+exponent {f['head_entropy']['se']:.4f} bits")
        print(f"Q3 L, bits/weight (shipped 4.5; block-128 4.125 is lossy, for reference): C1 {b['c1']:.4f}, C2 {b['c2']:.4f}, "
              f"C3 {b['c3']:.4f}, C4 {b['c4']:.4f}; round trips C1 {f['ok']['c1']}, C2 {f['ok']['c2']}, "
              f"C3 {f['ok']['c3']}, C4 {f['ok']['c4']}")
        print(f"Q3 H, bits/weight (shipped 16): C1 {b['h1']:.4f}, byte-split zstd {b['h2']:.4f}, C5 {b['c5']:.4f}; "
              f"round trips C1 {f['ok']['h1']}, byte-split {f['ok']['h2']}, C5 {f['ok']['c5']}")
        print(f"K1 (the linear stack, table-driven): best C3/C4 {f['best_l']:,} B = {f['best_l_bits']:.4f} bits/weight "
              f"against ≤ {K1_BITS:.4f}: {'PASS' if f['k1'] else 'FAIL'} ({SHIPPED_L / f['best_l']:.4f}x fewer bytes)")
        print(f"K2 (the whole shipped stream, table-driven): {f['t_bytes']:,} B per token against ≤ {K2_BYTES:,.1f}: "
              f"{'PASS' if f['k2'] else 'FAIL'} ({SHIPPED_T / f['t_bytes']:.4f}x fewer bytes, saving "
              f"{100 * f['t_saving']:.2f}%, {f['t_bits']:.4f} bits/weight over L and H)")
        for pid, text, val, hit in predictions(f):
            shown = val if isinstance(val, bool) else f"{val:.6g}"
            print(f"{pid}: {text}: {shown}  {'HIT' if hit else 'MISS'}")
    print(f"VERDICT (a): {v}" + {
        "SURVIVES": " - lossless compaction survives as a byte cut (necessary, not sufficient: (d) and a decode-rate "
                    "test decide whether the NPU gains)",
        "KILLED": " - no table-driven coder saves 1/11 of the bytes on K1 or K2; lossless compaction is killed as a "
                  "byte cut",
        "INCOMPLETE": " - an input failed its pin or census, a row is missing, or a table-driven coder failed its "
                      "round trip; no kill, no survival"}[v])
    print("VERDICT_JSON " + json.dumps({"verdict": v, **{k: f[k] for k in ("k1", "k2", "best_l", "t_bytes")
                                                           if k in f}}))
    return 0 if v in ("SURVIVES", "KILLED") else 2


# ---------------------------------------------------------------- the pre-registration

PREREG = f"""\
LLM study, Gemma 3 4B, pre-registration (a): lossless compressibility of the QAT q4_0 weights.
Designed on the user's decisions of 2026-09-23 (locked decision 10) and the gate's review of the
drafts (v2). Offline and CPU-only: no timing, no NPU or GPU. Committed before any file is fetched.

INPUTS, pinned from the Hub's LFS metadata and verified after download (size and SHA-256); a
mismatch stops the run and nothing is measured. They go to the HF cache; nothing enters git.
  {GGUF_PIN['repo']} @ {GGUF_PIN['revision']}
    {GGUF_PIN['file']}  {GGUF_PIN['size']:,} B  sha256 {GGUF_PIN['sha256']}
  {ST_REPO} @ {ST_REV}
    {ST_PINS[0]['file']}  {ST_PINS[0]['size']:,} B  sha256 {ST_PINS[0]['sha256']}
    {ST_PINS[1]['file']}  {ST_PINS[1]['size']:,} B  sha256 {ST_PINS[1]['sha256']}
  The run also checks the census from the header (SPEC): 238 Q4_0 tensors, 3,208,642,560 params;
  token_embd.weight [2560, 262144] F16, 671,088,640 params, the tied head (no separate output
  tensor); the rest F32. Per token as shipped: 1,804,861,440 + 1,342,177,280 = 3,147,038,720 B.

Q1, LAYOUT AND GRID (a check, not assumed; reported, it does not decide the verdict)
  For each of the 238 linears: parse the q4_0 blocks as ggml defines them, an f16 d, then 16 bytes
  where the low nibble is element j and the high nibble element j + 16; w = d * (q - 8); 32-element
  blocks along ne0, the input dimension. Compare bf16(d * (q - 8)) with the BF16 checkpoint's weight
  at the same index (HF [out, in]), as values (+0 equals -0). Report the exact-match fraction and
  max |diff| / |d| per tensor. A wrong layout shows as a failed match, so Q1 also checks the layout.
  Also report the fraction of blocks with a code at q = 0 (q4_0 maps a block's signed extreme to -8,
  so it should be 100%: a structural redundancy a coder can use), and the blocks with d = 0.
  The same grid test on the embedding: llama.cpp's reference q4_0 rule (the first signed extreme,
  d = max / -8 in fp32 and stored f16, q = min(15, trunc(x / d + 8.5))) applied to the checkpoint's
  BF16 embedding (all 262,208 rows), to see whether the head was trained on an int4 grid.
  Q1c: the GGUF's F16 head against f16(bf16) of the checkpoint's embedding over the shared 262,144
  rows (the exact-match fraction). This fixes what "the release's head" is for (c)'s reference.
  The kill line does not depend on Q1: order-0 entropies and the table-driven coders do not depend
  on the order of the codes within a tensor.

Q2, ENTROPY (on paper, DERIVED)
  Order-0 entropy of the codes, per tensor (weighted by size) and pooled over all 238; of the f16
  scales as 16-bit symbols per tensor (the empirical histogram; biased low on small tensors), and
  split sign+exponent (6 bits) / mantissa (10 bits); for the F16 head, of its high bytes, its low
  bytes and its sign+exponent fields.

Q3, REAL CODERS. Sizes include every table and length field. Each coder round-trips to the exact
input, or its row is void.
  C1: zstd level 19 on each tensor's raw q4_0 blocks.
  C2: zstd level 19 on split streams: the codes one per byte, then the f16 scales.
  C3: a static per-tensor canonical Huffman on the codes (16 symbols, in numpy), scales raw.
  C4: C3, with the scales' high byte coded by a static per-tensor canonical Huffman and their low
      byte raw.
  C5 (the head): a static canonical Huffman on the F16 high byte (sign, the 5 exponent bits and
      the top 2 mantissa bits), the low byte raw.
  On the head, also reported: C1 (zstd 19 on the raw F16 bytes) and a byte-split zstd (the high and
  the low bytes as separate streams).
  C3, C4 and C5 are the table-driven class, the simplest class a core might decode: one table
  lookup per symbol. zstd is not in that class. No coder's decode rate is measured in (a).
  Implementation (fixed here):
    zstd: zstandard {ZSTD_LEVEL}, with {ZSTD_THREADS} worker threads, the content size in each frame;
      a stream costs its frame plus an {LEN_FIELD}-byte length. The linears' frames are compressed
      {ZSTD_THREADS} at a time in a thread pool (scheduling only: each frame is the one it would be alone).
    Huffman: code lengths limited to {HUFF_MAX_LEN} bits (the rarest counts flattened until the code
      fits; the cost is in the size). The stream is coded in chunks of {CHUNK:,} symbols, each padded
      to a byte. A stream costs its chunks plus one byte of code length per symbol of its alphabet
      (16 or 256), an {LEN_FIELD}-byte symbol count, and {CHUNK_FIELD} bytes per chunk.
    Round trip: every Huffman chunk is decoded from its own bytes through a table built from the
      stored code lengths alone. The {HUFF_MAX_LEN}-bit-or-less window at each code start must name
      that code's symbol and length, which by induction is what a sequential decoder produces.
      Every zstd frame is decompressed and compared byte for byte.

DECODE BUDGET (DERIVED, so the class is not called NPU-ready)
  16 reachable compute cores x the MEASURED 1.7972 GHz = 28.755 G core-cycles/s. SILICON:252-258
  gives the core counts, 20 physical and 16 on the 4x4 overlay. SILICON:63 gives the unreachable
  column, column 0 by elimination.
  An NPU decode fed at 47.62 GB/s must produce:
    84.66 G codes/s at the shipped 4.5 bits: 2.94 codes per cycle per core;
    93.12 G codes/s at K1's 4.0909 bits: 3.24 per cycle per core;
    for the head, 23.81 G weights/s at 16 bits: 0.83 high-byte symbols per cycle per core,
      rising as the coder saves.
  Whether a core can sustain that is a later decode-rate test, not (a).

METRIC: bits per streamed weight, (coded bytes x 8) / weights, for
  L, the linear stack, against 4.5 as shipped and 4.125 (block-128 f16 scales: a lossy
     re-quantization, reported, not the baseline);
  H, the head, against 16;
  T, the whole per-token stream as shipped (3,147,038,720 B).
  The best C3/C4 is the smaller of the two whole-stack totals.

KILL LINE (DERIVED from the study's 1.10 line; decode time is proportional to bytes read)
  Lossless compaction survives as a byte cut for decode only if a table-driven real coder saves at
  least 1/11 of the bytes (at least 1.10x fewer) on EITHER
    K1, the linear stack alone, C3/C4: at most 4.0909 bits/weight
        (exactly: 1,804,861,440 x 10 >= coded x 11);
    K2, the whole shipped stream, the linears by C3/C4 and the head by C5: at most
        3,147,038,720 / 1.1 = 2,860,944,290.9 B/token (exactly: 3,147,038,720 x 10 >= coded x 11).
  Exactly 1.10x passes. If neither holds, compaction is KILLED as a lossless byte cut. Entropy (Q2)
  and zstd (C1/C2) are reported, but cannot pass it: entropy is on paper, and zstd is not
  table-driven.
  INCOMPLETE, with no kill and no survival, if:
    a fetched file differs from its pin;
    the census differs;
    a row is missing;
    or C3, C4 or C5 fails a round trip.
  A zstd round-trip failure voids only that reported row.

PASSING (a) IS NECESSARY, NOT SUFFICIENT. The CPU and the 780M could read the same coded stream, so a
byte cut can help every chip. Whether the NPU gains relative to them is (d)'s question, plus a
decode-rate test against the budget above.

PREDICTIONS (scored by the verdict; P5 and P7 in two parts, the range and the outcome)
  P1: at least 99.9% of linear weights match the grid exactly.
  P2: 100% of blocks hold a q = 0 code.
  P3: pooled H0 of the codes is 3.60-3.85 bits.
  P4: scales cost 0.28-0.38 bits/weight at order 0.
  P5: the best C3/C4 on L is 3.95-4.20 bits/weight (point 4.12, a 8.4% saving), so K1 FAILS,
      narrowly.
  P6: C5 codes the head to 13.0-14.2 bits/weight (saves 11-19% of the head).
  P7: K2 saves 8-13% of 3.147 GB (point 11%), so it PASSES, narrowly.
  P8: C1 on the raw blocks is at least 4.3 bits/weight.
  P9: the embedding is not on an int4 grid (under 10% exact).
  P10: Q1c, the GGUF's F16 head equals bf16 -> f16 of the checkpoint's embedding for at least 99.9%
       of the shared rows.

RUNNER: bash scripts/llm-study.sh compress-prereg | compress | compress-verdict (resnet_env17).
  Logs: results/llm/gemma_compress_{{prereg,run,verdict}}_desktop2_<date>.log.
  The run is a heavy CPU load, not a timing sitting. It is announced to BFP16, and the gate hears
  the result before (b) starts.
"""


def prereg() -> int:
    print(PREREG)
    budget = {"core_cycles_per_s": CORES * CLOCK_GHZ * 1e9,
              "codes_per_cycle_core_4.5": READ_GBPS * 8 / 4.5 / (CORES * CLOCK_GHZ),
              "codes_per_cycle_core_k1": READ_GBPS * 8 / K1_BITS / (CORES * CLOCK_GHZ),
              "head_symbols_per_cycle_core_16": READ_GBPS * 8 / 16 / (CORES * CLOCK_GHZ)}
    print("BUDGET " + ", ".join(f"{k} {v:.4g}" for k, v in budget.items()))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.split("\n")
    n = sum(1 for d in dirty if d.strip())
    print(f"git HEAD {head}" + (f" (+{n} uncommitted paths, this stage's own files)" if n else ""))
    print("PREREG_JSON " + json.dumps({
        "stage": "gemma (a) compress", "pins": PINS, "n_lin": N_LIN, "n_head": N_HEAD, "shipped": {
            "l": SHIPPED_L, "h": SHIPPED_H, "t": SHIPPED_T}, "margin": [MARGIN_NUM, MARGIN_DEN], "k1_bits": K1_BITS,
        "k2_bytes": K2_BYTES, "chunk": CHUNK, "huff_max_len": HUFF_MAX_LEN, "zstd": [ZSTD_LEVEL, ZSTD_THREADS],
        "len_field": LEN_FIELD, "chunk_field": CHUNK_FIELD, "predictions": PRED, "budget": budget, "git_head": head}))
    return 0


# ---------------------------------------------------------------- selftest (no network, no model)

def write_gguf(path, tensors):
    """A minimal GGUF v3 writer for the selftest: tensors = [(name, dims, type, raw bytes)]."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [("general.architecture", 8, s("gemma3")), ("general.alignment", 4, struct.pack("<I", 32)),
          ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 20) + b"".join(s(f"t{i}") for i in range(20)))]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv:
        head += s(k) + struct.pack("<I", t) + v
    offs, off = [], 0
    for _, _, _, raw in tensors:
        offs.append(off)
        off = (off + len(raw) + 31) // 32 * 32
    for (name, dims, t, raw), o in zip(tensors, offs):
        head += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<IQ", t, o)
    head += b"\0" * ((-len(head)) % 32)
    body = bytearray(off)
    for (_, _, _, raw), o in zip(tensors, offs):
        body[o:o + len(raw)] = raw
    Path(path).write_bytes(head + bytes(body))


def selftest() -> int:
    rng = np.random.default_rng(7)
    tmp = ROOT / "scratch" / "llm" / "gemma_compress_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    # entropy
    assert abs(entropy([1] * 16) - 4.0) < 1e-12 and abs(entropy([5, 5]) - 1.0) < 1e-12 and entropy([7]) == 0.0
    print("selftest entropy: uniform 16 = 4 bits, a fair coin = 1 bit, one symbol = 0")
    # Huffman: a skewed 16-symbol stream over three chunks, then a fault
    p = np.array([.30, .20, .12, .09, .07, .05, .04, .03, .025, .02, .015, .012, .01, .006, .003, .001])
    x = rng.choice(16, size=3 * CHUNK + 12345, p=p / p.sum()).astype(np.uint8)
    size, ok = huff_stream(x, 16)
    lens = huff_lengths(np.bincount(x, minlength=16))
    expect = sum((int(lens[x[i:i + CHUNK]].sum()) + 7) // 8 for i in range(0, len(x), CHUNK)) + 16 + LEN_FIELD + 4 * 4
    assert ok and size == expect, (ok, size, expect)
    h0 = entropy(np.bincount(x, minlength=16))
    print(f"selftest C3 coder: {len(x):,} symbols round-trip, {size * 8 / len(x):.4f} bits/symbol against H0 {h0:.4f}")
    size_f, ok_f = huff_stream(x, 16, fault_chunk=1)
    assert not ok_f
    print("selftest C3 coder: one flipped bit in chunk 1 fails the round trip (the row would be void)")
    # length limit on a 256-symbol stream with a very skewed tail, and a single-symbol stream
    counts = np.array([2 ** 30 >> min(i, 29) for i in range(256)], dtype=np.int64)
    assert huff_lengths(counts).max() <= HUFF_MAX_LEN
    y = np.repeat(np.arange(256, dtype=np.uint8), [max(1, 400000 >> min(i, 18)) for i in range(256)])
    rng.shuffle(y)
    assert huff_stream(y, 256)[1] and huff_stream(np.zeros(1000, np.uint8), 256)[1]
    print(f"selftest C4/C5 coder: 256 symbols limited to {HUFF_MAX_LEN} bits and round-trips; a one-symbol stream too")
    # zstd
    zs, zok = zstd_size(x)
    assert zok and zs < len(x)
    print(f"selftest zstd: round-trips ({zs:,} B for {len(x):,})")
    # a synthetic GGUF: q4_0 codes and scales on a bf16-exact grid, an F16 head, an F32 norm
    nb = 8
    d16 = np.array([2.0 ** -k * (1 + m / 8) for k, m in zip(range(4, 12), range(8))], dtype=np.float16)
    codes = rng.integers(0, 16, size=(nb, 32)).astype(np.uint8)
    codes[:, 5] = 0
    qs = (codes[:, :16] | (codes[:, 16:] << 4)).astype(np.uint8)
    blocks = np.concatenate([d16.view(np.uint8).reshape(nb, 2), qs], axis=1)
    headv = rng.standard_normal(64 * 3).astype(np.float16)
    path = tmp / "mini.gguf"
    write_gguf(path, [("blk.0.attn_q.weight", [64, 4], 2, blocks.tobytes()),
                      ("token_embd.weight", [64, 3], 1, headv.tobytes()),
                      ("blk.0.attn_norm.weight", [64], 0, np.ones(64, np.float32).tobytes())])
    mm, version, meta, tensors = read_gguf(path)
    t = {x["name"]: x for x in tensors}
    q = t["blk.0.attn_q.weight"]
    assert version == 3 and q["tname"] == "Q4_0" and q["n"] == 256 and q["nbytes"] == nb * 18
    d_back, c_back = q4_0_split(np.asarray(mm[q["start"]:q["start"] + nb * 18]).reshape(nb, 18))
    assert np.array_equal(c_back, codes) and np.array_equal(d_back, d16.view(np.uint16))
    hh = t["token_embd.weight"]
    assert np.array_equal(np.asarray(mm[hh["start"]:hh["start"] + hh["nbytes"]]).view(np.float16), headv)
    print("selftest GGUF: header, alignment, tensor offsets, the q4_0 nibble order and the F16 head read back")
    w = f32_to_bf16((f16_to_f32(d_back)[:, None] * (codes.astype(np.float32) - 8)).ravel())
    exact, worst = grid_linear(d_back, codes, w)
    assert exact == 256 and worst == 0.0
    w2 = w.copy()
    w2[17] ^= np.uint16(1)
    assert grid_linear(d_back, codes, w2)[0] == 255
    print("selftest Q1 grid: an on-grid tensor matches 256 of 256, one flipped bit gives 255")
    d32, qq = q4_0_rule(bf16_to_f32(w).reshape(-1, 32))
    v = (d32[:, None] * (qq.astype(np.float32) - 8)).ravel()
    assert int((bf16_to_f32(f32_to_bf16(v)) == bf16_to_f32(w)).sum()) == 256
    noise = bf16_to_f32(f32_to_bf16(rng.standard_normal(32 * 64).astype(np.float32)))
    d32, qq = q4_0_rule(noise.reshape(-1, 32))
    frac = float((bf16_to_f32(f32_to_bf16((d32[:, None] * (qq.astype(np.float32) - 8)).ravel())) == noise).mean())
    assert frac < 0.5
    print(f"selftest q4_0 rule: reproduces an on-grid tensor exactly, and {frac:.3f} of Gaussian bf16 values")
    # verdicts on synthetic rows, with the real per-tensor sizes
    per_layer = [("attn_q", 2560 * 2048), ("attn_k", 2560 * 1024), ("attn_v", 2560 * 1024),
                 ("attn_output", 2048 * 2560), ("ffn_gate", 2560 * 10240), ("ffn_up", 2560 * 10240),
                 ("ffn_down", 10240 * 2560)]
    assert sum(n for _, n in per_layer) * 34 == N_LIN

    def fake(name, l_bits, h_bits, fault=None, fetch_ok=True):
        out = [f"FETCH_JSON {json.dumps({'ok': fetch_ok})}", f"CENSUS_JSON {json.dumps({'ok': True})}"]
        for i in range(34):
            for gg, n in per_layer:
                size = int(round(n * l_bits / 8))
                r = {"name": f"blk.{i}.{gg}.weight", "n": n, "blocks": n // 32, "q1_exact": n, "q0_blocks": n // 32,
                     "zero_d_blocks": 0, "code_counts": [n // 16] * 16, "h_codes": 4.0, "h_scale16": 10.0,
                     "h_scale_se": 4.0, "h_scale_man": 7.0, "c1": int(n * 0.56), "c1_ok": True, "c2": int(n * 0.55),
                     "c2_ok": True, "c3": size + 1000, "c3_ok": True, "c4": size, "c4_ok": not (fault and i == 0)}
                out.append("ROW_JSON " + json.dumps(r))
        out.append("HEAD_JSON " + json.dumps({"n": N_HEAD, "h_hi": 5.8, "h_lo": 8.0, "h_se": 3.0, "c1": SHIPPED_H,
                                              "c1_ok": True, "c2": SHIPPED_H, "c2_ok": True,
                                              "c5": int(round(N_HEAD * h_bits / 8)), "c5_ok": True, "q1c_exact": 99,
                                              "q1c_n": 100, "q1b_exact": 1, "q1b_n": 100}))
        p = tmp / f"verdict_{name}.log"
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
        return p

    cases = (("killed", 4.2, 15.0, None, True, "KILLED"), ("k1-pass", 4.0, 16.0, None, True, "SURVIVES"),
             ("k2-pass", 4.2, 13.0, None, True, "SURVIVES"), ("void", 4.0, 16.0, "c4", True, "INCOMPLETE"),
             ("sha", 4.0, 16.0, None, False, "INCOMPLETE"))
    for name, lb, hb, fault, fok, want in cases:
        print(f"\n---- synthetic (a) verdict: {name}")
        got = verdict(fake(name, lb, hb, fault, fok))
        text = [ln for ln in (tmp / f"verdict_{name}.log").read_text(encoding="utf-8").splitlines()]
        v, _ = score([json.loads(ln[9:]) for ln in text if ln.startswith("ROW_JSON ")],
                     json.loads(next(ln[10:] for ln in text if ln.startswith("HEAD_JSON "))), fok, True)
        assert v == want and got == (0 if want != "INCOMPLETE" else 2), (name, v, got)
    assert k_pass(110, 100) and not k_pass(110, 101) and k_pass(SHIPPED_L, SHIPPED_L * 10 // 11)
    assert not k_pass(SHIPPED_T, SHIPPED_T * 10 // 11 + 1)
    print("\nselftest: exactly 1.10x passes, a byte over fails; the entropy, coders, GGUF reader, grid tests and "
          "all five synthetic verdicts as expected")
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prereg", "fetch", "run", "selftest"):
        sub.add_parser(c)
    v = sub.add_parser("verdict")
    v.add_argument("log", help="the run log")
    a = ap.parse_args()
    if a.cmd == "verdict":
        return verdict(a.log)
    if a.cmd == "fetch":
        fetch()
        return 0
    return {"prereg": prereg, "run": run, "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
