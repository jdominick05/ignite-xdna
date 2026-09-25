#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid stack, S1: the NPU numerics gate, CPU only. Do 3c's int8 and bf16 arithmetics on Gemma 3 4B's
seven linears keep the model's quality on real text? No NPU and no GPU: N1 and N2 reproduce the NPU's
int32 bit for bit (3c: the NPU's int8 output equalled the exact product in every window), and N3
reproduces N-bf16 up to fp32 accumulation order. The plan is PREREG below; the rules are this file's
verdict code (VERDICT_FUNCS, verdict_constants()), frozen by VERDICT_CODE_SHA256.

The five models share (c)'s built C0-H16 graph (tools/gemma_decode.py), head cut, hidden states out:
  R0  MatMulNBits accuracy_level 0 (exact d.(q - 8), fp32 activations): the reference
  R   accuracy_level 4 (int8 activations per 32-block, ORT's closest form to llama.cpp): report-only
  N1  int8, X per tensor x W per column (3c's N-i8), MatMulInteger, float64 quantize and epilogue
  N2  int8, X per token x W per column, the same kernel and codes
  N3  bf16 (RNE) X and W, fp32 MatMul (N-bf16's arithmetic)

    python tools/hybrid_s1.py selftest   # synthetic checks: no model, no chip
    python tools/hybrid_s1.py prereg     # the plan, the text pins, PROTOCOL_JSON, the frozen hashes
    python tools/hybrid_s1.py build      # the five models, C3, one profiled pass per model (children)
    python tools/hybrid_s1.py check      # C1, C2 (the form), C4, C5, C6 (children)
    python tools/hybrid_s1.py states     # the five models' hidden states, then band C (children)
    python tools/hybrid_s1.py verdict    # the metrics and the frozen verdict
Every heavy child holds one session (or the torch reference) and refuses below MIN_AVAIL_GB available.
"""
import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gemma_compress as gc  # noqa: E402
import gemma_decode as gd  # noqa: E402
import gemma_decode_suite as gds  # noqa: E402
import llm_prefill3c as p3c  # noqa: E402

WORK = ROOT / "scratch/llm/hybrid_s1"
MODELS = WORK / "models"
CHECK = WORK / "check"
STATES = WORK / "states"
KVDIR = WORK / "kv"
PROF = WORK / "prof"
RESULTS = ROOT / "results/llm"

# ---------------------------------------------------------------- the protocol (pre-registered)

ARMS = ("R0", "R", "N1", "N2", "N3")
DECIDING = ("N1", "N2", "N3")
REPORT_ONLY = ("R",)
N_SEQ, SEQ_LEN, STRIDE, BOS = 16, 2048, 2047, 2
C_SEQ, C_LEN = 4, 3072
BANDS = {"L": (0, 1023), "H": (1024, 2046)}             # positions i (the logits after ids[0..i]), inclusive
BAND_C = (2048, 3070)
T_KL, T_TOP1 = 0.0123, 0.956                            # (c)'s C0-H4: 0.012269, 0.95601 (three figures)
ANCHOR = {"C0-H4": [0.012269480565198573, 0.9560117302052786],
          "C4-H4": [0.013382885385917895, 0.9540566959921799],
          "D-H4": [0.012289061755960435, 0.9560117302052786]}
C4_KL_MAX, C6_KL_MAX = gd.FID_KL_MAX, 1e-5
BF16_ACC_MAX = p3c.BF16_ACC_MAX
SEED, BOOT_N, CHUNK = 20260924, 10_000, 256
MIN_AVAIL_GB = 15.0
THREADS = 8
SESSION = {"graph_optimization_level": "ORT_DISABLE_ALL", "intra_op_num_threads": THREADS,
           "providers": ["CPUExecutionProvider"], "prepacking": "on (ORT's default)"}
FUSED_FORBIDDEN = ("MatMulIntegerToFloat", "DynamicQuantizeMatMul", "FusedMatMul", "DynamicQuantizeLinear")
HIDDEN_NAME = "/model/layers.34/final_norm_layernorm/output_0"
HEAD_NODE, HEAD_W = "/lm_head/MatMul", "lm_head.MatMul.weight"
NB_RE = re.compile(r"^/model/layers\.(\d+)/(attn|mlp)/(q|k|v|o|gate|up|down)_proj/MatMulNBits$")
L16_INPUTS = {"x_attn": "/model/layers.16/input_layernorm/output_0",
              "x_o": "/model/layers.16/attn/GroupQueryAttention/output_0",
              "x_ffn": "/model/layers.16/pre_feedforward_layernorm/output_0",
              "x_down": "/model/layers.16/mlp/Mul/output_0"}
SRC_PIN = {"model.onnx": (433879, gds.MODEL_SHA["C0-H16"]["model.onnx"]),
           "model.onnx.data": (7644053504, gds.MODEL_SHA["C0-H16"]["model.onnx.data"])}
P3C_INPUTS_LOG = RESULTS / "llm_prefill3c_inputs_desktop2_20260924.log"
C_BUILD_LOG = RESULTS / "gemma_decode_build_desktop2_20260924.log"
PREDICTIONS = [
    ("P1", "N1 FAILS: per-tensor int8 over real activations, whose BOS row and outlier channels set the tile's amax"),
    ("P2", "N2 PASSES, against R0"),
    ("P3", "N3 PASSES, against R0, with mean KL below 0.1 x 0.0123 in both bands"),
    ("P4", "KL(R0 || R) is below 0.0123 in both bands"),
]


def protocol() -> dict:
    return {"stage": "hybrid S1", "plan": "v2", "arms": ARMS, "deciding": DECIDING, "report_only": REPORT_ONLY,
            "reference": "R0", "n_seq": N_SEQ, "seq_len": SEQ_LEN, "stride": STRIDE, "bos": BOS,
            "c_seq": C_SEQ, "c_len": C_LEN, "bands": BANDS, "band_c": BAND_C, "t_kl": T_KL, "t_top1": T_TOP1,
            "anchor": ANCHOR, "c4_kl_max": C4_KL_MAX, "c6_kl_max": C6_KL_MAX, "bf16_acc_max": BF16_ACC_MAX,
            "seed": SEED, "boot_n": BOOT_N, "chunk": CHUNK, "min_avail_gb": MIN_AVAIL_GB, "session": SESSION,
            "fused_forbidden": FUSED_FORBIDDEN, "text_pin": gds.TEXT_PIN, "c_ids_sha": gds.IDS_SHA,
            "gguf_pin": {k: gc.GGUF_PIN[k] for k in ("repo", "file", "revision", "size", "sha256")},
            "src_pin": SRC_PIN, "quantize": "float64: s = float64(max|X|) / 127.0; q = Clip(Round(X / s), -127, 127); "
            "u8 = q + 128 (zero point 128); a zero scale divides by 1", "epilogue": "fp32(float64(int32) * s_x * s_w[n])",
            "bf16": "Cast(float->bfloat16) RNE, Cast(->float); W by ml_dtypes RNE, stored bf16; MatMul fp32",
            "pick": ["N2", "N1 (requires 2,048-row prompt tiles: llama.cpp n_ubatch = 2,048)", "N3", "none"]}


def sha_bytes(b) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    return gc.sha256_file(path)


def sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def arr_sha(a: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(a).tobytes())


def npy_sha(a: np.ndarray) -> str:
    """sha256 of the .npy file np.save would write (3c's INPUT_FILE_JSON form)."""
    f = io.BytesIO()
    np.save(f, a)
    return sha_bytes(f.getvalue())


def say(tag: str, obj) -> None:
    print(f"{tag} " + json.dumps(obj), flush=True)


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def avail_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 1e9


def own_memory() -> dict:
    import psutil
    m = psutil.Process().memory_info()
    return {"wset_gb": round(m.rss / 1e9, 2), "peak_wset_gb": round(getattr(m, "peak_wset", 0) / 1e9, 2),
            "private_gb": round(getattr(m, "private", 0) / 1e9, 2),
            "peak_private_gb": round(getattr(m, "peak_pagefile", 0) / 1e9, 2)}


def start_gate(tag: str) -> None:
    a = avail_gb()
    say("MEMORY_START_JSON", {"child": tag, "available_gb": round(a, 1), "min_gb": MIN_AVAIL_GB})
    if a < MIN_AVAIL_GB:
        print(f"MEMORY_REFUSE {tag}: {a:.1f} GB available, below {MIN_AVAIL_GB} GB; not a verdict, re-run later",
              flush=True)
        sys.exit(4)


# ---------------------------------------------------------------- the text

def text_tokens() -> list:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    t = gds.TEXT_PIN
    p = Path(hf_hub_download(t["repo"], t["file"], revision=t["revision"], repo_type="dataset"))
    if p.stat().st_size != t["size"] or sha_file(p) != t["sha256"]:
        sys.exit("the text differs from its pin")
    text = "\n\n".join(pq.read_table(p).column("text").to_pylist())
    return gds.quiet_tokenizer()(text, add_special_tokens=False)["input_ids"]


def sequences(T: list) -> tuple:
    seqs = [[BOS] + T[STRIDE * s:STRIDE * (s + 1)] for s in range(N_SEQ)]
    cseqs = [[BOS] + T[STRIDE * c:STRIDE * c + C_LEN - 1] for c in range(C_SEQ)]
    assert all(len(s) == SEQ_LEN for s in seqs) and all(len(c) == C_LEN for c in cseqs)
    assert all(cseqs[c][:SEQ_LEN] == seqs[c] for c in range(C_SEQ))
    return seqs, cseqs


def text_pins(T: list) -> dict:
    seqs, cseqs = sequences(T)
    return {"text_tokens": len(T), "seq_sha": [gds.ids_sha(s) for s in seqs],
            "all_sha": gds.ids_sha([i for s in seqs for i in s]), "c_sha": [gds.ids_sha(c) for c in cseqs],
            "c_link_equal": gds.ids_sha(seqs[0][:1024]) == gds.IDS_SHA}


def pinned_sequences() -> tuple:
    """The ids, checked against the committed prereg log's TEXT_JSON."""
    T = text_tokens()
    got = text_pins(T)
    want = logged_json(latest("hybrid_s1_prereg_*.log"), "TEXT_JSON")
    for k in ("text_tokens", "seq_sha", "all_sha", "c_sha"):
        if got[k] != want[k]:
            sys.exit(f"the ids differ from the prereg's TEXT_JSON ({k})")
    return sequences(T)


# ---------------------------------------------------------------- logs

def latest(pattern: str):
    got = sorted(RESULTS.glob(pattern))
    if not got:
        sys.exit(f"no {pattern} under results/llm")
    return got[-1]


def logged_json(path: Path, tag: str, every: bool = False):
    out = [json.loads(s.split(" ", 1)[1]) for s in path.read_text(encoding="utf-8").splitlines()
           if s.startswith(tag + " ")]
    if every:
        return out
    if not out:
        sys.exit(f"{path.name} has no {tag}")
    return out[-1]


def frozen_ok(tag_log: str = "hybrid_s1_prereg_*.log") -> bool:
    p = latest(tag_log)
    text = p.read_text(encoding="utf-8")
    want = dict(re.findall(r"^(PREREG_TEXT_SHA256|PROTOCOL_JSON_SHA256|VERDICT_CODE_SHA256) ([0-9a-f]{64})$",
                           text, re.M))
    got = hashes()
    ok = all(want.get(k) == v for k, v in got.items())
    say("FROZEN_JSON", {"prereg_log": p.name, "prereg_log_lf_sha256": sha_lf(p), "equal": ok, **got})
    return ok


# ---------------------------------------------------------------- the release

def open_gguf(hash_check: bool):
    from huggingface_hub import hf_hub_download
    g = gc.GGUF_PIN
    p = Path(hf_hub_download(g["repo"], g["file"], revision=g["revision"]))
    if p.stat().st_size != g["size"] or (hash_check and sha_file(p) != g["sha256"]):
        sys.exit("the GGUF differs from its pin")
    mm, _, _, tensors = gc.read_gguf(p)
    return mm, {t["name"]: t for t in tensors}


def gguf_of(node_name: str) -> tuple:
    L, _, proj = NB_RE.match(node_name).groups()
    return int(L), gd.ONNX_PROJ[f"{proj}_proj"]


# ---------------------------------------------------------------- the subgraphs (one builder for models and probes)

def _t(name, dtype, dims, vals):
    from onnx import helper
    return helper.make_tensor(name, dtype, dims, vals)


def shared_consts() -> list:
    from onnx import TensorProto as P
    return [_t("s1/c127", P.DOUBLE, [], [127.0]), _t("s1/zero", P.DOUBLE, [], [0.0]), _t("s1/one", P.DOUBLE, [], [1.0]),
            _t("s1/lo", P.DOUBLE, [], [-127.0]), _t("s1/hi", P.DOUBLE, [], [127.0]), _t("s1/c128", P.DOUBLE, [], [128.0]),
            _t("s1/azp", P.UINT8, [], [128]), _t("s1/axes_last", P.INT64, [1], [-1])]


def used_consts(nodes) -> list:
    """Only the shared constants these nodes read (ORT warns about, and drops, unused initializers)."""
    names = {i for n in nodes for i in n.input}
    return [t for t in shared_consts() if t.name in names]


def quant_nodes(a: str, per_token: bool) -> tuple:
    """X (fp32) -> u8 codes and the float64 scale, 3c's quant_x (per tensor) or per row (per token)."""
    from onnx import TensorProto as P, helper
    p = f"/s1/q{'t' if per_token else 'p'}{a}"
    n = []
    n.append(helper.make_node("Abs", [a], [p + "/abs"], name=p + "/Abs"))
    n.append(helper.make_node("ReduceMax", [p + "/abs"] + (["s1/axes_last"] if per_token else []), [p + "/amax"],
                              name=p + "/ReduceMax", keepdims=1))
    n.append(helper.make_node("Cast", [p + "/amax"], [p + "/amax64"], name=p + "/CastAmax", to=P.DOUBLE))
    n.append(helper.make_node("Div", [p + "/amax64", "s1/c127"], [p + "/s"], name=p + "/Scale"))
    n.append(helper.make_node("Equal", [p + "/s", "s1/zero"], [p + "/z"], name=p + "/IsZero"))
    n.append(helper.make_node("Where", [p + "/z", "s1/one", p + "/s"], [p + "/sd"], name=p + "/Guard"))
    n.append(helper.make_node("Cast", [a], [p + "/x64"], name=p + "/CastX", to=P.DOUBLE))
    n.append(helper.make_node("Div", [p + "/x64", p + "/sd"], [p + "/xs"], name=p + "/DivX"))
    n.append(helper.make_node("Round", [p + "/xs"], [p + "/xr"], name=p + "/Round"))
    n.append(helper.make_node("Clip", [p + "/xr", "s1/lo", "s1/hi"], [p + "/xc"], name=p + "/Clip"))
    n.append(helper.make_node("Add", [p + "/xc", "s1/c128"], [p + "/xu"], name=p + "/Add128"))
    n.append(helper.make_node("Cast", [p + "/xu"], [p + "/u8"], name=p + "/CastU8", to=P.UINT8))
    return n, p + "/u8", p + "/s"


def int8_nodes(name: str, u8: str, s_x: str, wq: str, sw: str, y: str, i32: str = None) -> list:
    """MatMulInteger(u8 X, zero point 128; s8 W_q) -> int32; y = fp32(float64(int32) * s_x * s_w[n])."""
    from onnx import TensorProto as P, helper
    p = name + "/s1"
    i32 = i32 or p + "/i32"
    return [helper.make_node("MatMulInteger", [u8, wq, "s1/azp"], [i32], name=p + "/MatMulInteger"),
            helper.make_node("Cast", [i32], [p + "/i64"], name=p + "/CastI", to=P.DOUBLE),
            helper.make_node("Mul", [p + "/i64", s_x], [p + "/ys"], name=p + "/MulSx"),
            helper.make_node("Mul", [p + "/ys", sw], [p + "/yd"], name=p + "/MulSw"),
            helper.make_node("Cast", [p + "/yd"], [y], name=p + "/CastY", to=P.FLOAT)]


def bf16_in_nodes(a: str) -> tuple:
    from onnx import TensorProto as P, helper
    p = f"/s1/b{a}"
    return [helper.make_node("Cast", [a], [p + "/bf"], name=p + "/CastBf16", to=P.BFLOAT16),
            helper.make_node("Cast", [p + "/bf"], [p + "/f"], name=p + "/CastF", to=P.FLOAT)], p + "/f"


def bf16_nodes(name: str, af: str, wbf: str, y: str, wf: str = None) -> list:
    from onnx import TensorProto as P, helper
    p = name + "/s1"
    wf = wf or p + "/wf"
    return [helper.make_node("Cast", [wbf], [wf], name=p + "/CastW", to=P.FLOAT),
            helper.make_node("MatMul", [af, wf], [y], name=p + "/MatMul")]


class DataFile:
    """An external-data file, written once: every tensor at a 4096-aligned offset, hashed as written."""

    def __init__(self, path: Path):
        self.path, self.f, self.off, self.h = path, open(path, "wb"), 0, hashlib.sha256()

    def add(self, name: str, a: np.ndarray, dtype: int, dims) -> "object":
        from onnx import TensorProto
        pad = (-self.off) % 4096
        if pad:
            z = b"\0" * pad
            self.f.write(z)
            self.h.update(z)
            self.off += pad
        b = np.ascontiguousarray(a).tobytes()
        self.f.write(b)
        self.h.update(b)
        t = TensorProto()
        t.name, t.data_type = name, dtype
        t.dims.extend(list(dims))
        t.data_location = TensorProto.EXTERNAL
        for k, v in (("location", self.path.name), ("offset", str(self.off)), ("length", str(len(b)))):
            e = t.external_data.add()
            e.key, e.value = k, v
        self.off += len(b)
        return t

    def close(self) -> dict:
        self.f.close()
        return {"file": self.path.name, "bytes": self.off, "sha256": self.h.hexdigest()}


# ---------------------------------------------------------------- the surgery

def base_cut(src: Path, location: str, hidden: int = gd.HIDDEN):
    """(c)'s C0-H16 proto: external data re-pointed at `location`, the head removed, the hidden states out."""
    import onnx
    from onnx import TensorProto
    m = onnx.load(str(src), load_external_data=False)
    g = m.graph
    for t in g.initializer:
        for e in t.external_data:
            if e.key == "location":
                e.value = location
    head = [n for n in g.node if n.name == HEAD_NODE]
    assert len(head) == 1 and list(head[0].input) == [HIDDEN_NAME, HEAD_W], head
    g.node.remove(head[0])
    w = next(t for t in g.initializer if t.name == HEAD_W)
    g.initializer.remove(w)
    logits = next(o for o in g.output if o.name == "logits")
    g.output.remove(logits)
    g.output.insert(0, onnx.helper.make_tensor_value_info(HIDDEN_NAME, TensorProto.FLOAT,
                                                          ["batch_size", "sequence_length", hidden]))
    return m


def nb_nodes(g) -> list:
    out = [n for n in g.node if n.op_type == "MatMulNBits"]
    assert len(out) == gd.LAYERS * 7 and all(NB_RE.match(n.name) for n in out), len(out)
    return out


def set_accuracy(m, level: int):
    for n in nb_nodes(m.graph):
        for a in n.attribute:
            if a.name == "accuracy_level":
                a.i = level
    return m


def replace_linears(m, kind: str, weights, data: DataFile, shared: dict = None):
    """kind N1 | N2 | N3. weights(node_name) -> W [K, N] fp32 (exact d.(q - 8)). N1 and N2 share one int8 file:
    `shared` maps node names to the TensorProtos already written (the second call writes nothing)."""
    from onnx import NodeProto, TensorProto as P
    g = m.graph
    new, inits, done = [], [], {}
    shared = {} if shared is None else shared
    for n0 in g.node:
        n = NodeProto()
        n.CopyFrom(n0)
        if n.op_type != "MatMulNBits":
            new.append(n)
            continue
        a, y = n.input[0], n.output[0]
        attrs = {x.name: x.i for x in n.attribute}
        K, N = attrs["K"], attrs["N"]
        if kind in ("N1", "N2"):
            if a not in done:
                qn, u8, s = quant_nodes(a, per_token=kind == "N2")
                new += qn
                done[a] = (u8, s)
            u8, s = done[a]
            if n.name not in shared:
                W = weights(n.name)
                assert W.shape == (K, N) and W.dtype == np.float32, (n.name, W.shape)
                wq, sw = p3c.quant_w(W)
                shared[n.name] = (data.add(n.name + "/s1/wq", wq, P.INT8, [K, N]),
                                  data.add(n.name + "/s1/sw", sw.astype(np.float64), P.DOUBLE, [N]))
            twq, tsw = shared[n.name]
            inits += [twq, tsw]
            new += int8_nodes(n.name, u8, s, twq.name, tsw.name, y)
        else:
            import ml_dtypes
            if a not in done:
                bn, af = bf16_in_nodes(a)
                new += bn
                done[a] = af
            W = weights(n.name)
            assert W.shape == (K, N) and W.dtype == np.float32, (n.name, W.shape)
            wb = W.astype(ml_dtypes.bfloat16).view(np.uint16)
            t = data.add(n.name + "/s1/wbf16", wb, P.BFLOAT16, [K, N])
            inits.append(t)
            new += bf16_nodes(n.name, done[a], t.name, y)
    old = {x.input[1] for x in g.node if x.op_type == "MatMulNBits"} | \
          {x.input[2] for x in g.node if x.op_type == "MatMulNBits"}
    keep = []
    for t0 in g.initializer:
        if t0.name not in old:
            t = P()
            t.CopyFrom(t0)
            keep.append(t)
    del g.node[:]
    g.node.extend(new)
    del g.initializer[:]
    g.initializer.extend(keep + used_consts(new) + inits)
    return m


def c3_compare(base, arm, kind: str) -> dict:
    """C3: arm equals base outside the 238 replaced linears; R equals R0 but for accuracy_level."""
    bn = [n for n in base.graph.node if n.op_type != "MatMulNBits"]
    an = [n for n in arm.graph.node if not n.name.startswith("/s1/") and "/s1/" not in n.name
          and n.op_type != "MatMulNBits"]
    res = {"kind": kind, "io_equal": [i.SerializeToString() for i in base.graph.input] ==
           [i.SerializeToString() for i in arm.graph.input] and
           [o.SerializeToString() for o in base.graph.output] == [o.SerializeToString() for o in arm.graph.output]}
    res["nodes_equal"] = [n.SerializeToString() for n in bn] == [n.SerializeToString() for n in an]
    old = {x.input[k] for x in base.graph.node if x.op_type == "MatMulNBits" for k in (1, 2)}
    b_init = {t.name: t.SerializeToString() for t in base.graph.initializer if t.name not in old or kind in ("R0", "R")}
    a_init = {t.name: t.SerializeToString() for t in arm.graph.initializer}
    res["initializers_equal"] = all(a_init.get(k) == v for k, v in b_init.items())
    if kind in ("R0", "R"):
        want = 0 if kind == "R0" else 4
        anb = nb_nodes(arm.graph)
        bnb = nb_nodes(base.graph)
        same = True
        for x, y in zip(bnb, anb):
            xa = {a.name: a.SerializeToString() for a in x.attribute if a.name != "accuracy_level"}
            ya = {a.name: a.SerializeToString() for a in y.attribute if a.name != "accuracy_level"}
            same &= (list(x.input), list(x.output), x.name, xa) == (list(y.input), list(y.output), y.name, ya)
            same &= next(a.i for a in y.attribute if a.name == "accuracy_level") == want
        res["nbits_equal_but_level"] = same and len(anb) == len(bnb)
    res["ok"] = all(v for k, v in res.items() if k != "kind")
    return res


# ---------------------------------------------------------------- sessions

def session(path: Path, profile: Path = None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = THREADS
    so.log_severity_level = 3
    if profile is not None:
        so.enable_profiling = True
        so.profile_file_prefix = str(profile)
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def present_names(sess) -> list:
    return [o.name for o in sess.get_outputs() if o.name.startswith("present.")]


def run(sess, ids, past: dict = None, want_kv: bool = False) -> tuple:
    n = len(ids)
    plen = 0 if past is None else past["past_key_values.0.key"].shape[2]
    feed = {"input_ids": np.asarray([ids], dtype=np.int64), "attention_mask": np.ones((1, plen + n), dtype=np.int64)}
    for i in sess.get_inputs():
        if i.name.startswith("past_key_values"):
            feed[i.name] = past[i.name] if past is not None else np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32)
    names = [HIDDEN_NAME] + (present_names(sess) if want_kv else [])
    out = sess.run(names, feed)
    kv = None
    if want_kv:
        kv = {nm.replace("present.", "past_key_values."): o for nm, o in zip(names[1:], out[1:])}
    return np.ascontiguousarray(out[0][0], dtype=np.float32), kv


def profile_counts(path: str) -> dict:
    ev = json.loads(Path(path).read_text(encoding="utf-8"))
    c = {}
    for e in ev:
        if e.get("cat") == "Node" and str(e.get("name", "")).endswith("_kernel_time"):
            op = e.get("args", {}).get("op_name")
            c[op] = c.get(op, 0) + 1
    return dict(sorted(c.items()))


def static_counts(m) -> dict:
    c = {}
    for n in m.graph.node:
        if n.op_type != "Constant":                    # ORT folds Constant nodes into initializers at load
            c[n.op_type] = c.get(n.op_type, 0) + 1
    return dict(sorted(c.items()))


# ---------------------------------------------------------------- the metrics (frozen: VERDICT_FUNCS)

def log_softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    m = z.max(axis=1, keepdims=True)
    z = z - m
    z -= np.log(np.exp(z).sum(axis=1, keepdims=True))
    return z


def logits_chunk(h: np.ndarray, head: np.ndarray) -> np.ndarray:
    return np.asarray(h, dtype=np.float32) @ head.T


def kl_top1_chunk(lp_ref: np.ndarray, p_ref: np.ndarray, top_ref: np.ndarray, z: np.ndarray, nxt) -> dict:
    lp = log_softmax_rows(z)
    kl = (p_ref * (lp_ref - lp)).sum(axis=1)
    return {"kl": kl, "top1": (np.argmax(z, axis=1) == top_ref).astype(np.float64),
            "nll": -lp[np.arange(len(nxt)), nxt], "finite": bool(np.isfinite(z).all())}


def band_rows(band: tuple) -> slice:
    return slice(band[0], band[1] + 1)


def bootstrap(per_seq: np.ndarray) -> list:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per_seq), (BOOT_N, len(per_seq)))
    m = per_seq[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def aggregate(kl: np.ndarray, top1: np.ndarray, nll: np.ndarray) -> dict:
    """kl, top1, nll: [n_seq, positions] for one band."""
    return {"positions": int(kl.size), "kl_mean": float(kl.mean()), "kl_p50": float(np.median(kl)),
            "kl_p99": float(np.quantile(kl, 0.99)), "kl_max": float(kl.max()), "top1": float(top1.mean()),
            "ppl": float(np.exp(nll.mean())), "kl_ci": bootstrap(kl.mean(axis=1)), "top1_ci": bootstrap(top1.mean(axis=1))}


def rule(m: dict) -> tuple:
    """PASS iff in both bands L and H: mean KL <= T_KL and top-1 >= T_TOP1. A non-finite value is a FAIL.
    NARROW names each band whose bootstrap interval holds a threshold; it changes nothing."""
    if not m["finite"]:
        return "FAIL", []
    ok = all(m[b]["kl_mean"] <= T_KL and m[b]["top1"] >= T_TOP1 for b in ("L", "H"))
    narrow = [b for b in ("L", "H") if m[b]["kl_ci"][0] <= T_KL <= m[b]["kl_ci"][1]
              or m[b]["top1_ci"][0] <= T_TOP1 <= m[b]["top1_ci"][1]]
    return ("PASS" if ok else "FAIL"), narrow


def outcomes(metrics: dict, c1: dict, gates_ok: bool, complete: dict) -> dict:
    out = {}
    for arm in DECIDING + REPORT_ONLY:
        if not gates_ok or not complete.get(arm, False) or (arm in DECIDING and not c1.get(arm, False)):
            out[arm] = {"outcome": "INCOMPLETE", "narrow": []}
            continue
        o, nb = rule(metrics[arm])
        out[arm] = {"outcome": o, "narrow": nb}
    return out


def pick(out: dict) -> str:
    """N2 if it passes; N1 only if N2 fails and N1 passes (2,048-row tiles required); N3 if neither int8 arm
    passes; an INCOMPLETE earlier in that order leaves the pick INCOMPLETE."""
    for arm, label in (("N2", "N2"), ("N1", "N1: the hybrid must feed 2,048-row prompt tiles (llama.cpp n_ubatch = 2,048)"),
                       ("N3", "N3: bf16 (3c: 1.40x on energy above idle, 1.04x with the idle, 2.09x slower than DirectML)")):
        o = out[arm]["outcome"]
        if o == "PASS":
            return label
        if o == "INCOMPLETE":
            return "INCOMPLETE"
    return "NONE: the hybrid has no NPU numerics; U6 or stop is the user's call (band C is reported for it)"


def score(out: dict, metrics: dict) -> dict:
    def s(cond, arms):
        if any(out[a]["outcome"] == "INCOMPLETE" for a in arms):
            return "NOT SCORED"
        return "HIT" if cond() else "MISS"
    return {"P1": s(lambda: out["N1"]["outcome"] == "FAIL", ["N1"]),
            "P2": s(lambda: out["N2"]["outcome"] == "PASS", ["N2"]),
            "P3": s(lambda: out["N3"]["outcome"] == "PASS" and all(metrics["N3"][b]["kl_mean"] < 0.1 * T_KL
                                                                  for b in ("L", "H")), ["N3"]),
            "P4": s(lambda: metrics["R"]["finite"] and all(metrics["R"][b]["kl_mean"] < T_KL for b in ("L", "H")), ["R"])}


VERDICT_FUNCS = ("log_softmax_rows", "logits_chunk", "kl_top1_chunk", "band_rows", "bootstrap", "aggregate", "rule",
                 "outcomes", "pick", "score")


def verdict_constants() -> dict:
    return {"ARMS": ARMS, "DECIDING": DECIDING, "REPORT_ONLY": REPORT_ONLY, "N_SEQ": N_SEQ, "SEQ_LEN": SEQ_LEN,
            "STRIDE": STRIDE, "BOS": BOS, "C_SEQ": C_SEQ, "C_LEN": C_LEN, "BANDS": BANDS, "BAND_C": BAND_C,
            "T_KL": T_KL, "T_TOP1": T_TOP1, "C4_KL_MAX": C4_KL_MAX, "C6_KL_MAX": C6_KL_MAX,
            "BF16_ACC_MAX": BF16_ACC_MAX, "SEED": SEED, "BOOT_N": BOOT_N, "CHUNK": CHUNK,
            "TEXT_PIN": gds.TEXT_PIN, "GGUF_SHA256": gc.GGUF_PIN["sha256"], "SRC_PIN": SRC_PIN}


def verdict_code_sha() -> str:
    """sha256 over the sources of VERDICT_FUNCS (inspect.getsource: universal newlines, so CRLF and LF
    checkouts agree) and the JSON of verdict_constants()."""
    import inspect
    g = globals()
    src = "".join(inspect.getsource(g[f]) for f in VERDICT_FUNCS)
    return sha_bytes((src + json.dumps(verdict_constants(), sort_keys=True)).encode("utf-8"))


def hashes() -> dict:
    return {"PREREG_TEXT_SHA256": sha_bytes(PREREG.encode("utf-8")),
            "PROTOCOL_JSON_SHA256": sha_bytes(json.dumps(protocol(), sort_keys=True).encode("utf-8")),
            "VERDICT_CODE_SHA256": verdict_code_sha()}


def print_hashes() -> None:
    for k, v in hashes().items():
        print(f"{k} {v}", flush=True)


# ---------------------------------------------------------------- the emulations (C2 and the diagnostics)

def blocks(x: np.ndarray) -> np.ndarray:
    M, K = x.shape
    return x.reshape(M, K // 32, 32)


def q8_ort(x: np.ndarray) -> tuple:
    """MLAS CompInt8's A: per 32-block, scale = amax / 127 (fp32, kept fp32), codes RNE(x * (127 / amax))."""
    xb = blocks(np.asarray(x, dtype=np.float32))
    amax = np.abs(xb).max(axis=2)
    scale = (amax / np.float32(127)).astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(amax != 0, np.float32(127) / amax, np.float32(0)).astype(np.float32)
    q = np.rint((xb * inv[..., None]).astype(np.float32))
    return q, scale


def q8_llama(x: np.ndarray) -> tuple:
    """llama.cpp's x86 quantize_row_q8_0: codes RNE(x * (127 / amax)) in fp32, the scale stored fp16(amax / 127)."""
    q, scale = q8_ort(x)
    return q, scale.astype(np.float16).astype(np.float32)


def q4_dot(q: np.ndarray, sx: np.ndarray, codes: np.ndarray, d: np.ndarray) -> np.ndarray:
    """sum over blocks of (integer dot of the block) * sx * d_w, accumulated in float64. codes [N, kb, 32]."""
    M, kb, _ = q.shape
    N = codes.shape[0]
    c8 = codes.astype(np.float64) - 8.0
    dw = gc.f16_to_f32(d).reshape(N, kb).astype(np.float64)
    y = np.zeros((M, N))
    for b in range(kb):
        y += (q[:, b, :].astype(np.float64) @ c8[:, b, :].T) * (sx[:, b:b + 1].astype(np.float64) * dw[None, :, b])
    return y


def rel_l2(y, ref) -> float:
    y, ref = np.asarray(y, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(y - ref) / np.linalg.norm(ref))


def quant_x_rows(x: np.ndarray) -> tuple:
    """N2's quantize in numpy: per row, float64, 3c's quant_x form; a zero row divides by 1."""
    s = np.abs(x).max(axis=1).astype(np.float64) / 127.0
    sd = np.where(s == 0, 1.0, s)
    q = np.clip(np.rint(x.astype(np.float64) / sd[:, None]), -127, 127).astype(np.int8)
    return q, s


# ---------------------------------------------------------------- the plan (PREREG)

PREREG = r"""S1 PLAN AND PREREG: the NPU numerics gate. Do 3c's int8 and bf16 arithmetics on the seven linears keep
Gemma 3 4B's quality on real text? CPU only.

Plan v2, 2026-09-24.

The approvals:
- The user approved S1: "ok let INT4 start S1".
- The gate reviewed v1 (ea649322) and approved it with changes, none of which needs the user.
- v2 applies those changes: R0 is the reference (§3), the pick order (§7), band C (§6), S4's wording (§4), and
  N3's runtime (§8). The gate's rulings on v1's questions are in §10.

The rules:
- This text is committed before anything heavy runs.
- The rules are tools/hybrid_s1.py's verdict code.
- Branch worktree-hybrid-stack, from worktree-llm-study at 25cbc09. Held local.

Tags: MEASURED (with its log), DERIVED (arithmetic here), SPEC (a vendor source, read 2026-09-24), ESTIMATE.

## 0. What S1 decides, and what it does not

- **It decides**, per numerics N1, N2 and N3 (§4): PASS, FAIL or INCOMPLETE on model quality, against the
  reference R0 (§3), on real 2,048-token prompts.
- **It feeds**:
  - the hybrid's NPU numerics (§7);
  - the user's U6 call ("Decide after S1"). If nothing passes, U6 or stop is the user's call.
- **It does not decide**:
  - speed or energy (3c measured them; S5 measures the whole prompt);
  - generation, which stays on the GPU in Q4_0;
  - prompts of 8,192 tokens, which would be a separate run;
  - the GPU runtimes' accuracy (S0 and S3).
- **Why no NPU is needed** (MEASURED in 3c):
  - N1 and N2 reproduce the NPU's int32 output bit for bit. 3c's N-i8 int32 equalled the exact int8 product in
    every window, as C-i8's MatMulInteger did.
  - N3 reproduces N-bf16 up to fp32 accumulation order (§4).

## 1. A correction to the design (accepted by the gate)

- 3c's N-i8 quantizes W per output column, not per tensor:
  - `quant_w` (tools/llm_prefill3c.py:162-165) sets s_w[n] = max_k |W[k, n]| / 127;
  - X is quantized per tensor, one scale over the whole M-row tile (`quant_x`, :168-171; `quant_x(a[:M])`,
    :587);
  - so N2 differs from N1 only in X, which becomes per token.
- b1()'s printed fractions are fractions that match (gemma_decode.py:239-253).

## 2. The text set

- **Corpus:** WikiText-2 raw, test split, as (c) pinned it (gemma_decode_suite.py:96-98):
  - Salesforce/wikitext @ b08601e04326c79dfdd32d625aee71d232d685c3;
  - wikitext-2-raw-v1/test-00000-of-00001.parquet;
  - 732,610 B, sha256 5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91.

  It is real encyclopedic English, and it is llama.cpp's perplexity text.
- **Tokenizer:** (c)'s, the text-only copy's AutoTokenizer:
  - its files are the checkpoint's: tokenizer.json sha256 7d4046bf…, tokenizer.model 1299c11d…
    (gemma_decode.CK_LFS), and the others by git blob sha1 (CK_BLOB);
  - (c)'s prereg checked it equal to SentencePiece and to the raw tokenizer.json.
- **Ids:**
  1. The parquet's text column is joined with "\n\n", as (c) did.
  2. The whole joined text is tokenized with add_special_tokens=False, giving T. (c)'s prereg counts 292,282
     tokens; S1 prints its own count.
  3. Sequence s = 0…15 is [2] + T[2047·s : 2047·(s+1)]: BOS and 2,047 consecutive text tokens. That is 2,048
     ids, no overlap, no chat template.
  4. Band C's sequences c = 0…3 are [2] + T[2047·c : 2047·c + 3071]: 3,072 ids, whose first 2,048 are
     sequence c.
- **Pins:** the prereg log prints the sha256 of every sequence's ids (int32 little-endian, (c)'s ids_sha form)
  and of all 16 concatenated. Every later step refuses other ids.
- **The link to (c)** (report-only): the prereg reports whether sequence 0's first 1,024 ids equal (c)'s
  IDS_SHA c87fda3e….

## 3. The model, and the reference R0

- **Weights:** the pinned GGUF (google/gemma-3-4b-it-qat-q4_0-gguf @ 15f73f5e, 3,155,051,328 B, sha256
  76aed0a8…).
- **The graph:** (c)'s built C0-H16 model, copied from the LLM study's scratch, not rebuilt:
  - checked against (c)'s pins: model.onnx sha256 33c47ffa…, model.onnx.data 7,644,053,504 B, 11edd445…
    (gemma_decode_suite.MODEL_SHA; build log line 596);
  - the 238 linears are MatMulNBits (bits 4, block 32, the release's codes and d, no zero points);
  - the F16 embedding and head are held in fp32;
  - GQA local windows are 1023 (F3), with RoPE caches (F1);
  - from text_only, only config.json and the tokenizer files are copied. The torch reference takes every
    weight from the GGUF.
- **Surgery, identical for every model:** /lm_head/MatMul and its weight are removed, and the graph's output
  is the final norm's output: the hidden states, [2048, 2560] fp32. The head is applied in the verdict,
  identically for every model (§6).
- **R0, the reference:** the graph with accuracy_level 0 on the 238 MatMulNBits. That is exact d·(q − 8)
  against fp32 activations: (c)'s C0-H16 with the head cut.
  - The anchor (§7) was measured against torch fp32, and C4 ties R0 to torch at ≤ 1e-4 per position. So R0 is
    the like-for-like reference.
- **R** (a report-only arm, scored like the others): the same graph with accuracy_level 4. It is the closest
  ORT form to llama.cpp's Q4_0 × Q8_0:

  | | llama.cpp, x86 | ORT MLAS, CompInt8 |
  |---|---|---|
  | Function | quantize_row_q8_0 | QuantizeARow_CompInt8_avx2 |
  | Codes | RNE of x · 127/amax | the same |
  | Activation block scale | stored fp16(amax/127) | kept in fp32 |
  | Dot | per-block integer sum × fp16(d_w) × fp16(d_x), float accumulator (ggml_vec_dot_q4_0_q8_0) | integer dot per block × d_w × d_x, fp32 accumulate |

  Both columns are SPEC, read from source on 2026-09-24: llama.cpp master, and ORT rel-1.23.2's AVX2 kernel.
  - KL(R0 ‖ R) is P4: what an int8-activation path costs.
  - The gap, named:
    1. the block scale: fp32 in ORT, fp16 in llama.cpp;
    2. fp32 summation order;
    3. llama.cpp's GPU backends: their MMQ kernels use Q8_1 blocks (an fp16 d, plus an fp16 sum of the
       block's unquantized values); a Vulkan device without integer dot products dequantizes to fp16 instead;
    4. the head: llama.cpp's CPU converts the activations to fp16 for an F16 weight. It is common to every
       model here.
  - The ORT here is onnxruntime 1.23.3.dev20260320, as resnet_env17 logs it, on Zen 4 (AVX-512 VNNI). So C2
    measures the form R takes, not assumes it.
- **Everything outside the 238 linears** is identical in R0, R, N1, N2 and N3 (C3): the embedding, the norms,
  QK-norm, RoPE, GQA, GELU and the residuals, in fp32 on the CPU EP.

## 4. The three numerics (the arms)

- Each arm replaces all 238 MatMulNBits (7 linears × 34 layers) with the subgraph below. Nothing else changes.
- **W** is the release's exact d·(q − 8), 3c's dense().
- **The input X** is the linear's fp32 input, [1, 2048, K]:
  - q, k and v share one X, and so one quantization, as in 3c;
  - gate and up share another.
- **Quantization is float64, exactly 3c's quant_x:**
  1. s = float64(max |X|) / 127.0;
  2. q = Round(float64(X) / s), where ONNX Round is half to even, as np.rint is;
  3. q = Clip(q, −127, 127), then u8 = q + 128.

  In ONNX: Abs → ReduceMax → Cast(double) → Div → Round → Clip → Add → Cast(uint8). A zero scale divides by 1
  instead of 0, which gives codes of 0 and an output of 0.

**N1: int8, X per tensor × W per column. 3c's N-i8 exactly.**
- s_x is one scale over the whole [2048, K] prompt tile, BOS row included. This is 3c's M = 2048 tile.
- W_q = quant_w(W): per output column, float64, RNE, clipped to ±127.
- The product: MatMulInteger(u8 X with zero point 128, s8 W_q) → int32. This is 3c's C-i8 form, which
  returned the exact int8 product in every window.
- The epilogue: y = fp32(float64(int32) · s_x · s_w[n]), evaluated left to right in float64, with one rounding
  to fp32.
- **S4's requirement:**
  - the NPU's int32 must be bit-exact for identical int8 codes;
  - the hybrid's quantize and epilogue (likely fp32, on the GPU or host) must hold the codes and outputs within
    a stated tolerance of S1's, with the number of flipped codes reported.

**N2: int8, X per token × W per column.**
- As N1, except s_x[m] = float64(max_k |X[m, k]|) / 127.0 per row.
- The epilogue is y = fp32(float64(int32) · s_x[m] · s_w[n]).
- Same NPU kernel and same W codes as N1; only the per-row scale is new, in the epilogue. N2 is invariant to
  how the prompt is batched.

**N3: bf16, N-bf16's arithmetic.**
- X is rounded to bf16 by Cast(float→bfloat16), round to nearest even, then Cast(→float).
- W is rounded on the host to bf16 by ml_dtypes (RNE), as 3c rounded N-bf16's W, and stored as bf16. Cast(→float)
  runs at run time. No graph optimizer runs (§8), so it is neither folded nor removed.
- The product is MatMul in fp32 (MLAS SGEMM, fp32 accumulator). The output is fp32, as N-bf16's is.
- **The gap to the NPU, named:** the accumulation order only.
  - 3c MEASURED N-bf16 against the float64 product of its bf16-rounded X and W at rel-L2 3.03e-7 to 3.44e-7 per
    linear (bf16_own, the same in A2 and B). That includes the NPU's split-K down.
  - C1 prints S1's own N3 against the same product.
  - Their sum, about 1e-6, is the bound between S1's N3 and the NPU, against bf16 rounding itself at about
    3e-3 (DERIVED).

## 5. Checks (the check log). A failed check stops what it guards.

- **C1, the bit-exact link to 3c.** It runs on 3c's own inputs:
  - X is regenerated from 3c's seed 20260924 (default_rng, standard_normal float32, INPUTS order);
  - W and W_q for blk.16 are re-derived from the GGUF;
  - every one's .npy sha256 must equal 3c's INPUT_FILE_JSON pin (llm_prefill3c_inputs_desktop2_20260924.log).
  - Then one probe per arm and linear is built with S1's own subgraph builder. It sees 2,048 rows and exposes
    the int32 (or the cast values):

  | Arm | What must hold |
  |---|---|
  | N1 | the int32's sha256 equals that log's EXACT_INT32_SHA256 i8_{name}_M2048, all seven |
  | N2 | the int32 equals the exact per-token product (float64 BLAS, 3c's exact_int), all seven |
  | N3 | the cast X and W equal ml_dtypes' RNE values; the output's rel-L2 against the float64 product of the rounded X and W is at most 1e-4 (3c's BF16_ACC_MAX), and it is printed |

  A failure voids that arm (INCOMPLETE).
- **C2, R's form** (report-only). It uses sequence 0's layer-16 linear inputs from R0:
  - for each of the seven linears, ORT's accuracy_level 4 output is compared by rel-L2 with two emulations:
    - (a) the ORT form: fp32 block scale, RNE;
    - (b) llama.cpp's x86 form: fp16 block scale, and codes from the fp32 127/amax;
  - and (a) is compared with (b).
  - The check log prints only these form figures. R's error against the exact product goes into the verdict
    log.
- **C3, graph identity:**
  - the five graphs are equal node by node outside the 238 replaced linears;
  - their shared initializers point at the same bytes of the same data file;
  - R and R0 differ only in 238 accuracy_level attributes;
  - every MatMulNBits node's qweight and scales equal the GGUF tensor it names (pack_ort(codes),
    f16→f32(d)), K and N included;
  - the verdict's head (the GGUF's F16 token_embd, upcast) equals the removed lm_head weight by sha256.
- **C4, R0 against torch at 2,048 real tokens.** It is a gate:
  - sequence 0, R0's hidden states against (c)'s reference: transformers' fp32 Gemma 3 with the release's
    weights, streamed (reference_hidden);
  - both pass through the same head;
  - the rule: max per-position KL ≤ 1e-4 (the builder's FID_KL_MAX).
  - The evidence so far: (c)'s fidelity gate ran this model as one plain-ORT prefill (ort_logits) at 1,100
    random ids. Its max KL was 3.73e-09 past the window (build log line 582), against 0.901 before F3
    (line 568). C4 repeats that test on real text, at 2,048 tokens and under S1's session options.
  - A failure stops S1 (INCOMPLETE).
  - The check log prints PASS or FAIL only; the value goes into the verdict log.
- **C5, determinism:** R0 runs sequence 0 twice, and the two hidden states' sha256 must be equal.
- **C6, band C's control:**
  - R0 runs chunked on band C's sequence 0: its own KV from [0, 2048), then [2048, 3072) with that past;
  - it is compared with R0 one-shot over [0, 3072);
  - the rule: max per-position KL ≤ 1e-5 over positions 2048-3070;
  - PASS or FAIL in the check log, the value in the verdict log;
  - a failure stops band C only.

## 6. The metrics (computed only in the verdict step)

- **The head:** the release's F16 token_embd upcast to fp32, [262,144, 2,560] (C3).
  - logits = h · headᵀ in fp32, by numpy's BLAS, in chunks of 256 positions;
  - the same code for every model.
- **Per position** i (the logits after ids[0..i], predicting ids[i+1]), with log-softmax in float64:
  - KL_i(R0 ‖ A) = Σ_v p_R0(v) · (log p_R0(v) − log p_A(v)), over the full 262,144 vocabulary;
  - top-1 = [argmax z_A = argmax z_R0], with numpy's first index on ties.
- **Band L:** i = 0…1,023. Every local layer sees the whole prefix.
- **Band H:** i = 1,024…2,046. The 1,024 window is active. With n_ctx = 2,048, these are the positions
  llama.cpp's --kl-divergence scores.
- **Band C** (REPORT-ONLY; the rule is unchanged):
  - The hybrid runs the prompt under the arm, and everything after it on the GPU, attending to the arm's KV
    cache. L and H score every position under the arm, which is stricter.
  - So a PASS carries over to the hybrid, but a FAIL does not prove the hybrid fails. Band C measures that
    case, for the U6 call.
  - For c = 0…3:
    1. the arm runs [0, 2048), and its present KV is kept on disk;
    2. R0 runs [2048, 3072) with that past;
    3. positions 2048-3070 (4 × 1,023) are scored against R0's own one-shot run over [0, 3072).
  - It reports mean KL, p99 and top-1. It runs for R, N1, N2 and N3, and for R0 itself, which is C6 on all four
    sequences.
- **Per arm, in bands L and H:**
  - mean KL, pooled over the 16 sequences, with p50, p99 and max;
  - top-1 agreement;
  - the arm's perplexity on the next tokens (R0's too).
- **Report-only:**
  - a bootstrap 95% interval of mean KL and top-1 over the 16 sequences (10,000 resamples, seed 20260924);
  - the values of C2 against the exact product, C4 and C6;
  - for layer 16 of sequence 0:
    - the per-linear rel-L2 of N1, N2 and N3 (numpy emulations of §4) against the exact product on R0's real
      inputs;
    - the outlier ratio: per-tensor amax / median per-token amax, per linear input.
- **Any non-finite hidden state or logit** is a FAIL for that arm, as in (c).

## 7. The thresholds and the verdict

**The anchor: (c)'s int4-head arms** (gemma_decode_accuracy_desktop2_20260924.log, MEASURED, 1,023 positions,
against torch fp32):

| Arm | Mean KL | Top-1 |
|---|---|---|
| C0-H4 | 0.012269 | 0.95601 |
| C4-H4 | 0.013383 | 0.95406 |
| D-H4 | 0.012289 | 0.95601 |

C0-H4's secondary KL, 1.4e-11, shows its 0.0123 is the int4 head's cost alone.

**The rule:** an NPU numerics may cost the model no more than the int4 head cost it in (c). Per arm N1, N2 and
N3, against R0:
- **PASS** if, in both bands L and H:
  - mean KL ≤ 0.0123 (C0-H4's 0.012269, rounded to three figures);
  - and top-1 agreement ≥ 0.956 (C0-H4's 0.95601, rounded to three figures).
- **FAIL** otherwise, or on a non-finite value.
- **INCOMPLETE** if:
  - C1 voided the arm;
  - C3, C4 or C5 failed;
  - or any of its 16 sequences has no states after one re-run of its process.
- The verdict reads the point estimates. If a threshold lies inside the bootstrap interval, the line adds
  NARROW, but the verdict stands.
- **The comparability, named:** the anchor is a size, not a like-for-like. It is one 1,023-position text,
  where S1 is 16 × 2,047 positions.
- R is scored like the arms and printed with its would-be outcome under the rule, but it decides nothing.

**The pick** (printed; it decides nothing beyond S1):
1. **N2**, if it passes.
2. **N1**, only if N2 fails and N1 passes. Then it is named as a requirement: "the hybrid must feed 2,048-row
   prompt tiles (llama.cpp n_ubatch = 2,048)". llama.cpp's default ubatch is 512, and N2 is batch-invariant.
3. **N3**, if neither int8 arm passes and N3 passes. Its caveats come from 3c:
   - 1.40× on energy above idle;
   - 1.04× with the idle, under the 1.10 margin (post hoc);
   - 2.09× slower than DirectML.
4. **None:** the hybrid has no NPU numerics, and U6 or stop is the user's call. Band C is reported for that
   call.

**Predictions** (report-only, scored HIT or MISS, decide nothing):
- P1: N1 FAILS. Per-tensor int8 over real activations, whose BOS row and outlier channels set the tile's amax.
- P2: N2 PASSES, against R0.
- P3: N3 PASSES, against R0, with mean KL below 0.1 × 0.0123 in both bands.
- P4: KL(R0 ‖ R) is below 0.0123 in both bands.

**Frozen, as in 3c:**
- VERDICT_CODE_SHA256, over the named functions and constants;
- PROTOCOL_JSON and its sha256;
- this text's sha256.

All three are printed in the prereg log and in every later S1 log, and the verdict refuses a mismatch.

## 8. RAM, runtime, and the announcements

- **(c)'s incident:** the accuracy pass sat at 12.9 GB private when a memory guard fired, with 6.0 GB free and
  4.0 GB commit free (99a7eec). Most of that was three fp32 heads.
- **This machine:** 31.1 GB of RAM, 16.9 GB free at the read, and a page file of only 2.4 GB.
- **One model at a time.** Every step runs as a sequence of child processes, each holding one session or the
  torch reference, never two:
  - the build's probes;
  - the five states runs, then R0's band-C continuations;
  - C4's torch forward;
  - the verdict's head.
- **The start gate:** each heavy child refuses to start below 15 GB available. That is not a verdict; it is
  re-run later.
- **Session options**, pinned in PROTOCOL_JSON and identical everywhere:
  - graph_optimization_level = ORT_DISABLE_ALL, so no fusion, constant folding or cast elimination can touch
    an arm;
  - the built graph already holds its fused contrib ops (GQA, SkipSimplifiedLayerNormalization, MatMulNBits),
    and kernel prepacking still runs;
  - 8 intra-op threads, unpinned. S1 times nothing that decides.
- **What survives, checked from what actually ran:**
  - the build's one pass per model runs with ORT's profiler, and the executed nodes are counted by op type:
    - R0 and R: 238 MatMulNBits;
    - N1 and N2: 238 MatMulInteger and 136 Round;
    - N3: 238 MatMul, and 510 Cast beyond the base graph's 2;
  - every other op count must equal R0's, and no fused op (MatMulIntegerToFloat, DynamicQuantizeMatMul,
    FusedMatMul) may appear;
  - a full optimized-model dump would rewrite about 10 GB per model; the profile shows what ran.
- **Memory:** the build MEASURES each model's session memory and its one pass on sequence 0 (peak working set
  and private bytes, by psutil).
- **N3's runtime:** its W is a Cast output, so MLAS cannot prepack it, and every run re-packs. The build's
  timed pass MEASURES it, and the states window is sized from that.
- **ESTIMATE of each process's peak:**

  | Process | Peak |
  |---|---|
  | R0 and R | about 5-7 GB |
  | N1 and N2 | about 6-9.5 GB |
  | N3 | about 9.5-10.5 GB |
  | activations and float64 temporaries | about 1-1.5 GB more |
  | torch (C4) | about 5 GB |
  | the verdict | about 5 GB |

  Band C's KV (570 MB per model and sequence, 11.4 GB for all) goes to disk and is deleted after the
  continuations.
- **BFP16:**
  - START REQUEST 1: build and check, with RAM and duration (about 30 min). FINISHED after.
  - START REQUEST 2: states and band C, sized from the build's measured times. FINISHED after.
  - The verdict (about 5 GB, about 15 min) is announced on its own.

## 9. Files, logs and order

- **New:**
  - tools/hybrid_s1.py (selftest, prereg, build, check, states, verdict, and their child modes);
  - scripts/hybrid-stack.sh: stages s1-selftest, s1-prereg, s1-build, s1-check, s1-states, s1-verdict, with
    refuse, logged and use_env as in llm-study.sh, and resnet_env17 for all.
- **Scratch** (git-ignored):
  - scratch/llm/hybrid_s1/models/: c0h16.onnx and base.onnx.data, copied from (c)'s C0-H16 and checked
    against its pins, plus the five graphs and int8.onnx.data and bf16.onnx.data;
  - scratch/llm/gemma/text_only/: config.json and the tokenizer files, checked (config.json against build
    log line 484's record);
  - scratch/llm/hybrid_s1/{check,states,kv,prof}/.

  The copies are manual, and no committed log names their source path.
- **Logs:** results/llm/hybrid_s1_{prereg,build,check,states,verdict}_desktop2_YYYYMMDD.log, UTF-8, the profile
  path scrubbed, never overwritten.
- **The order:**
  1. The code, with synthetic tests (no model).
  2. s1-prereg: the text pins, PROTOCOL_JSON and the frozen hashes. Committed before anything heavy.
  3. START REQUEST 1, then s1-build and s1-check, then FINISHED.
  4. START REQUEST 2, then s1-states (the five models, then band C), then FINISHED.
  5. Commit the build, check and states logs as measured, report the HEAD, and wait. No KL or agreement figure
     exists before the verdict.
  6. The gate checks them blind; then s1-verdict, commit, report.

## 10. The gate's rulings on v1 (2026-09-24)

- N = 16 and the 15 GB start gate: accepted.
- The thresholds: 0.0123 / 0.956 in both bands, against R0, with no stricter anchor.
- C4: PASS or FAIL only in the check log, the value in the verdict log.
- The copy of C0-H16 and text_only's config and tokenizer: accepted, with C3 and C4 re-proving it.
- ORT_DISABLE_ALL: accepted; C4 re-proves R0 under it.
- The design corrections: accepted, and applied by the gate."""


def prereg() -> int:
    print(PREREG, flush=True)
    say("PROTOCOL_JSON", protocol())
    say("PREDICTIONS_JSON", [{"id": q, "text": t} for q, t in PREDICTIONS])
    T = text_tokens()
    pins = text_pins(T)
    say("TEXT_JSON", pins)
    print(f"TEXT: {pins['text_tokens']} tokens in the joined test split; 16 sequences of 2,048 ids, sha256 of all "
          f"{pins['all_sha']}; sequence 0's first 1,024 ids {'equal' if pins['c_link_equal'] else 'DIFFER FROM'} (c)'s "
          f"IDS_SHA (report-only)", flush=True)
    for p in (P3C_INPUTS_LOG, C_BUILD_LOG):
        print(f"SOURCE_LOG {rel(p)}: LF sha256 {sha_lf(p)}", flush=True)
    print_hashes()
    print("PREREG OK", flush=True)
    return 0


# ---------------------------------------------------------------- build

def verify_sources() -> dict:
    res = {}
    src = MODELS / "c0h16.onnx"
    data = MODELS / "base.onnx.data"
    for p, (size, sha) in ((src, SRC_PIN["model.onnx"]), (data, SRC_PIN["model.onnx.data"])):
        got = sha_file(p) if p.exists() else None
        res[p.name] = {"bytes": p.stat().st_size if p.exists() else None, "sha256": got,
                       "ok": p.exists() and p.stat().st_size == size and got == sha}
    t = gd.TEXT
    for f, (size, sha) in gd.CK_LFS.items():
        if f.endswith(".safetensors"):
            continue
        p = t / f
        res["text_only/" + f] = {"ok": p.exists() and p.stat().st_size == size and sha_file(p) == sha}
    for f, (size, sha1) in gd.CK_BLOB.items():
        if f in ("config.json", "model.safetensors.index.json"):
            continue
        p = t / f
        ok = p.exists() and p.stat().st_size == size
        if ok:
            b = p.read_bytes()
            ok = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == sha1
        res["text_only/" + f] = {"ok": ok}
    want = logged_json(C_BUILD_LOG, "TEXTONLY_JSON")["config"]
    p = t / "config.json"
    res["text_only/config.json"] = {"ok": p.exists() and json.loads(p.read_text(encoding="utf-8")) == want}
    return res


def head_shas(mm, by) -> dict:
    """The verdict's head [V, 2560] fp32 and its transpose (the removed lm_head weight's layout), by sha256."""
    h16 = gd.gguf_head(mm, by)
    a, b = hashlib.sha256(), hashlib.sha256()
    for r in range(0, gd.VOCAB, 16384):
        a.update(np.ascontiguousarray(h16[r:r + 16384].astype(np.float32)).tobytes())
    for c in range(0, gd.HIDDEN, 64):
        b.update(np.ascontiguousarray(h16[:, c:c + 64].astype(np.float32).T).tobytes())
    return {"head_sha256": a.hexdigest(), "head_t_sha256": b.hexdigest()}


def data_slice_sha(path: Path, off: int, length: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        left = length
        while left:
            b = f.read(min(left, 1 << 24))
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def ext(t) -> dict:
    return {e.key: e.value for e in t.external_data}


def build() -> int:
    import onnx
    print(f"BUILD (hybrid S1), no chip. {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    print_hashes()
    if not frozen_ok():
        print("BUILD STOP: the frozen hashes differ from the prereg log", flush=True)
        return 2
    src = verify_sources()
    say("SOURCES_JSON", src)
    if not all(v["ok"] for v in src.values()):
        print("BUILD STOP: a copied source differs from its pin", flush=True)
        return 2
    seqs, _ = pinned_sequences()
    mm, by = open_gguf(hash_check=True)
    base = base_cut(MODELS / "c0h16.onnx", "base.onnx.data")
    data = MODELS / "base.onnx.data"
    # C3's mapping: every MatMulNBits node's qweight and scales equal the GGUF tensor it names
    raw = np.memmap(data, dtype=np.uint8, mode="r")
    init = {t.name: t for t in base.graph.initializer}
    bad = []
    for n in nb_nodes(base.graph):
        L, gg = gguf_of(n.name)
        t = by[f"blk.{L}.{gg}.weight"]
        codes, d = gd.gguf_linear(mm, t)
        K, N = t["dims"]
        attrs = {x.name: x.i for x in n.attribute}
        qe, se = ext(init[n.input[1]]), ext(init[n.input[2]])
        qb = raw[int(qe["offset"]):int(qe["offset"]) + int(qe["length"])]
        sb = raw[int(se["offset"]):int(se["offset"]) + int(se["length"])]
        ok = (attrs["K"], attrs["N"]) == (K, N) and qb.tobytes() == gd.pack_ort(codes).tobytes() and \
            sb.tobytes() == gc.f16_to_f32(d).astype(np.float32).tobytes()
        if not ok:
            bad.append(n.name)
    del raw
    heads = head_shas(mm, by)
    he = ext(next(t for t in onnx.load(str(MODELS / "c0h16.onnx"), load_external_data=False).graph.initializer
                  if t.name == HEAD_W))
    removed = data_slice_sha(data, int(he["offset"]), int(he["length"]))
    say("C3_MAPPING_JSON", {"nodes": gd.LAYERS * 7, "mismatched": bad, "head_equal": removed == heads["head_t_sha256"],
                            **heads, "removed_lm_head_sha256": removed})

    def weights(name):
        L, gg = gguf_of(name)
        codes, d = gd.gguf_linear(mm, by[f"blk.{L}.{gg}.weight"])
        return p3c.dense(codes, d)

    models = {}
    models["R0"] = set_accuracy(base_cut(MODELS / "c0h16.onnx", "base.onnx.data"), 0)
    models["R"] = set_accuracy(base_cut(MODELS / "c0h16.onnx", "base.onnx.data"), 4)
    files = {}
    i8 = DataFile(MODELS / "int8.onnx.data")
    shared = {}
    models["N1"] = replace_linears(base_cut(MODELS / "c0h16.onnx", "base.onnx.data"), "N1", weights, i8, shared)
    models["N2"] = replace_linears(base_cut(MODELS / "c0h16.onnx", "base.onnx.data"), "N2", weights, i8, shared)
    files["int8.onnx.data"] = i8.close()
    bf = DataFile(MODELS / "bf16.onnx.data")
    models["N3"] = replace_linears(base_cut(MODELS / "c0h16.onnx", "base.onnx.data"), "N3", weights, bf)
    files["bf16.onnx.data"] = bf.close()
    probe = set_accuracy(base_cut(MODELS / "c0h16.onnx", "base.onnx.data"), 0)
    for k, nm in L16_INPUTS.items():
        K = {"x_attn": 2560, "x_o": 2048, "x_ffn": 2560, "x_down": 10240}[k]
        probe.graph.output.append(onnx.helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT,
                                                                     ["batch_size", "sequence_length", K]))
    models["R0_probe16"] = probe
    c3 = {k: c3_compare(base, models[k], k) for k in ARMS}
    for k, m in models.items():
        p = MODELS / f"{k}.onnx"
        onnx.save(m, str(p))
        files[p.name] = {"bytes": p.stat().st_size, "sha256": sha_file(p)}
    counts = {k: static_counts(models[k]) for k in ARMS}
    base_ops = counts["R0"]
    expect = {"R0": base_ops.get("MatMulNBits") == 238, "R": counts["R"] == base_ops,
              "N1": counts["N1"].get("MatMulInteger") == 238 and counts["N1"].get("Round") == 136 and "MatMulNBits" not in counts["N1"],
              "N2": counts["N2"].get("MatMulInteger") == 238 and counts["N2"].get("Round") == 136 and "MatMulNBits" not in counts["N2"],
              "N3": counts["N3"].get("MatMul") == 238 and counts["N3"].get("Cast", 0) - base_ops.get("Cast", 0) == 510
              and "MatMulNBits" not in counts["N3"]}
    for k in ("N1", "N2", "N3"):
        rest = {op: c for op, c in base_ops.items() if op not in ("MatMulNBits",)}
        expect[k] &= all(counts[k].get(op, 0) >= c for op, c in rest.items())
    say("STATIC_COUNTS_JSON", counts)
    say("C3_JSON", {"graphs": c3, "mapping_ok": not bad, "head_equal": removed == heads["head_t_sha256"],
                    "static_counts_ok": expect})
    c3_ok = all(v["ok"] for v in c3.values()) and not bad and removed == heads["head_t_sha256"] and all(expect.values())
    del mm
    probes = {}
    for arm in ARMS:
        rc = child(["_probe", arm], retries=1)
        probes[arm] = rc
        if rc == 4:
            print("BUILD STOP: memory gate; not a verdict, re-run later", flush=True)
            return 4
    got = {r["arm"]: r for r in logged_stdout_json("PROBE_JSON")}
    survive = {}
    for arm in ARMS:
        r = got.get(arm)
        ok = r is not None and r["executed"] == counts[arm] and not any(f in r["executed"] for f in FUSED_FORBIDDEN) \
            and r["finite"]
        survive[arm] = ok
    say("SURVIVE_JSON", survive)
    ok = c3_ok and all(survive.values()) and all(v == 0 for v in probes.values())
    say("BUILD_JSON", {"files": files, "c3": c3_ok, "survive": survive, "probe_rc": probes, "ok": ok, **heads})
    print("BUILD", "OK" if ok else "INCOMPLETE", flush=True)
    return 0 if ok else 2


_STDOUT_JSON = []


def child(args: list, retries: int = 0) -> int:
    """Run a child of this tool, its output into this log, and keep its tagged JSON for the parent."""
    for attempt in range(retries + 1):
        sys.stdout.flush()
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve())] + args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end="", flush=True)
            _STDOUT_JSON.append(line.rstrip("\n"))
        rc = p.wait()
        print(f"CHILD {' '.join(args)} rc {rc}" + (f" (attempt {attempt + 1})" if retries else ""), flush=True)
        if rc == 0 or rc == 4:
            return rc
    return rc


def logged_stdout_json(tag: str) -> list:
    return [json.loads(s.split(" ", 1)[1]) for s in _STDOUT_JSON if s.startswith(tag + " ")]


def probe_child(arm: str) -> int:
    start_gate(f"probe {arm}")
    seqs, _ = pinned_sequences()
    PROF.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sess = session(MODELS / f"{arm}.onnx", profile=PROF / arm)
    t1 = time.perf_counter()
    mem_load = own_memory()
    h, _ = run(sess, seqs[0])
    t2 = time.perf_counter()
    prof = sess.end_profiling()
    say("PROBE_JSON", {"arm": arm, "session_s": round(t1 - t0, 1), "seq0_s": round(t2 - t1, 1),
                       "executed": profile_counts(prof), "finite": bool(np.isfinite(h).all()),
                       "memory_after_load": mem_load, "memory_after_run": own_memory()})
    return 0


# ---------------------------------------------------------------- check

def check() -> int:
    print(f"CHECK (hybrid S1), no chip. {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    print_hashes()
    if not frozen_ok():
        print("CHECK STOP: the frozen hashes differ from the prereg log", flush=True)
        return 2
    b = logged_json(latest("hybrid_s1_build_*.log"), "BUILD_JSON")
    if not b["ok"]:
        print("CHECK STOP: the build is not OK", flush=True)
        return 2
    CHECK.mkdir(parents=True, exist_ok=True)
    rcs = {}
    for c in (["_c1"], ["_r0"], ["_probe16"], ["_torch"], ["_compare"]):
        rcs[c[0]] = child(c, retries=1)
        if rcs[c[0]] == 4:
            print("CHECK STOP: memory gate; not a verdict, re-run later", flush=True)
            return 4
    c1 = {r["arm"]: r["ok"] for r in logged_stdout_json("C1_ARM_JSON")}
    c5 = logged_stdout_json("C5_JSON")
    cmp_ = logged_stdout_json("GATES_JSON")
    res = {"C1": c1, "C5": c5[-1]["outcome"] if c5 else "MISSING",
           "C4": cmp_[-1]["C4"] if cmp_ else "MISSING", "C6": cmp_[-1]["C6"] if cmp_ else "MISSING", "rc": rcs}
    res["stop"] = res["C4"] != "PASS" or res["C5"] != "PASS"
    say("CHECK_JSON", res)
    print("CHECK", "STOP (C4 or C5)" if res["stop"] else "OK", flush=True)
    return 2 if res["stop"] else 0


def c1_child() -> int:
    import ml_dtypes
    import onnx
    from onnx import TensorProto as P, helper
    pins = {r["name"]: r["sha256"] for r in logged_json(P3C_INPUTS_LOG, "INPUT_FILE_JSON", every=True)}
    exact = logged_json(P3C_INPUTS_LOG, "EXACT_INT32_SHA256_JSON")
    rng = np.random.default_rng(p3c.SEED)
    xs = {x: rng.standard_normal((max(p3c.MS), K), dtype=np.float32) for x, K in p3c.INPUTS.items()}
    x_ok = {x: npy_sha(a) == pins[x] for x, a in xs.items()}
    mm, by = open_gguf(hash_check=False)
    M = 2048
    res = {"N1": True, "N2": True, "N3": True}
    rows = []
    for name, gg, K, N, x in p3c.LINEARS:
        codes, d = gd.gguf_linear(mm, by[f"blk.{p3c.LAYER}.{gg}.weight"])
        w = p3c.dense(codes, d)
        wq, sw = p3c.quant_w(w)
        w_ok = npy_sha(w) == pins[f"w_{name}"] and npy_sha(wq) == pins[f"w_{name}_q"] and npy_sha(sw) == pins[f"sw_{name}"]
        a = xs[x][:M][None]
        row = {"name": name, "x_pin": x_ok[x], "w_pin": w_ok}
        for kind in ("N1", "N2", "N3"):
            if kind in ("N1", "N2"):
                qn, u8, s = quant_nodes("x", per_token=kind == "N2")
                nodes = qn + int8_nodes("/lin", u8, s, "wq", "sw", "y", i32="i32")
                inits = used_consts(nodes) + [helper.make_tensor("wq", P.INT8, [K, N], wq.tobytes(), raw=True),
                                  helper.make_tensor("sw", P.DOUBLE, [N], sw.astype(np.float64).tobytes(), raw=True)]
                outs = [helper.make_tensor_value_info("i32", P.INT32, [1, M, N]),
                        helper.make_tensor_value_info("y", P.FLOAT, [1, M, N])]
            else:
                bn, af = bf16_in_nodes("x")
                wb = w.astype(ml_dtypes.bfloat16).view(np.uint16)
                nodes = bn + bf16_nodes("/lin", af, "wbf", "y", wf="wf")
                inits = used_consts(nodes) + [helper.make_tensor("wbf", P.BFLOAT16, [K, N], wb.tobytes(), raw=True)]
                outs = [helper.make_tensor_value_info(af, P.FLOAT, [1, M, K]),
                        helper.make_tensor_value_info("wf", P.FLOAT, [K, N]),
                        helper.make_tensor_value_info("y", P.FLOAT, [1, M, N])]
            g = helper.make_graph(nodes, f"c1_{kind}_{name}", [helper.make_tensor_value_info("x", P.FLOAT, [1, M, K])],
                                  outs, inits)
            m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            so.intra_op_num_threads = THREADS
            sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
            out = sess.run(None, {"x": a})
            if kind == "N1":
                ok = p3c.int32_sha(out[0][0]) == exact[f"i8_{name}_M{M}"]
                row["N1_int32_equal_3c"] = ok
            elif kind == "N2":
                q, _ = quant_x_rows(xs[x][:M])
                ok = np.array_equal(out[0][0], p3c.exact_int(q, wq))
                row["N2_int32_exact"] = ok
            else:
                xf = xs[x][:M].astype(ml_dtypes.bfloat16).astype(np.float32)
                wf = w.astype(ml_dtypes.bfloat16).astype(np.float32)
                bits = np.array_equal(out[0][0].view(np.uint32), xf.view(np.uint32)) and \
                    np.array_equal(out[1].view(np.uint32), wf.view(np.uint32))
                e = rel_l2(out[2][0], xf.astype(np.float64) @ wf.astype(np.float64))
                ok = bits and e <= BF16_ACC_MAX
                row.update({"N3_bits_rne": bits, "N3_rel_l2": e})
            ok &= x_ok[x] and w_ok
            res[kind] &= bool(ok)
            del sess
        say("C1_ROW_JSON", row)
        rows.append(row)
    for kind in ("N1", "N2", "N3"):
        say("C1_ARM_JSON", {"arm": kind, "ok": res[kind]})
    return 0


def r0_child() -> int:
    start_gate("R0 (C5, C6)")
    seqs, cseqs = pinned_sequences()
    sess = session(MODELS / "R0.onnx")
    h1, _ = run(sess, seqs[0])
    h2, _ = run(sess, seqs[0])
    np.save(CHECK / "r0_seq0.npy", h1)
    say("C5_JSON", {"sha_run1": arr_sha(h1), "sha_run2": arr_sha(h2),
                    "outcome": "PASS" if arr_sha(h1) == arr_sha(h2) else "FAIL"})
    ids = cseqs[0]
    _, kv = run(sess, ids[:SEQ_LEN], want_kv=True)
    hc, _ = run(sess, ids[SEQ_LEN:], past=kv)
    del kv
    ho, _ = run(sess, ids)
    np.save(CHECK / "r0_c0_chunked.npy", hc)
    np.save(CHECK / "r0_c0_oneshot.npy", ho)
    say("C6_RUNS_JSON", {"chunked_sha": arr_sha(hc), "oneshot_sha": arr_sha(ho), "chunked_rows": len(hc),
                         "oneshot_rows": len(ho)})
    say("MEM_JSON", {"child": "R0 (C5, C6)", **own_memory()})
    return 0


def probe16_child() -> int:
    start_gate("R0_probe16 (C2's inputs)")
    seqs, _ = pinned_sequences()
    sess = session(MODELS / "R0_probe16.onnx")
    names = [HIDDEN_NAME] + list(L16_INPUTS.values())
    feed = {"input_ids": np.asarray([seqs[0]], dtype=np.int64), "attention_mask": np.ones((1, SEQ_LEN), np.int64)}
    for i in sess.get_inputs():
        if i.name.startswith("past_key_values"):
            feed[i.name] = np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32)
    out = sess.run(names, feed)
    same = arr_sha(np.ascontiguousarray(out[0][0], dtype=np.float32)) == arr_sha(np.load(CHECK / "r0_seq0.npy"))
    for (k, _), a in zip(L16_INPUTS.items(), out[1:]):
        np.save(CHECK / f"x16_{k}.npy", np.ascontiguousarray(a[0]))
    say("PROBE16_JSON", {"hidden_equals_r0": bool(same), "inputs": list(L16_INPUTS)})
    return 0


def torch_child() -> int:
    start_gate("torch reference (C4)")
    seqs, _ = pinned_sequences()
    mm, by = open_gguf(hash_check=False)
    t0 = time.time()
    h = gds.reference_hidden(mm, by, None, seqs[0])
    np.save(CHECK / "torch_seq0.npy", np.ascontiguousarray(h, dtype=np.float32))
    say("TORCH_JSON", {"rows": len(h), "seconds": round(time.time() - t0, 1), "finite": bool(np.isfinite(h).all()),
                       **own_memory()})
    return 0


def max_kl(h_ref: np.ndarray, h: np.ndarray, head: np.ndarray) -> float:
    worst = 0.0
    for r in range(0, len(h_ref), CHUNK):
        lr = log_softmax_rows(logits_chunk(h_ref[r:r + CHUNK], head))
        la = log_softmax_rows(logits_chunk(h[r:r + CHUNK], head))
        worst = max(worst, float((np.exp(lr) * (lr - la)).sum(axis=1).max()))
    return worst


def load_head():
    mm, by = open_gguf(hash_check=False)
    head = np.ascontiguousarray(gd.gguf_head(mm, by).astype(np.float32))
    return head, mm, by


def compare_child() -> int:
    start_gate("compare (C4, C6, C2)")
    head, mm, by = load_head()
    r0 = np.load(CHECK / "r0_seq0.npy")
    tr = np.load(CHECK / "torch_seq0.npy")
    c4 = max_kl(tr, r0, head) if np.isfinite(tr).all() and np.isfinite(r0).all() else float("inf")
    hc = np.load(CHECK / "r0_c0_chunked.npy")
    ho = np.load(CHECK / "r0_c0_oneshot.npy")
    n = BAND_C[1] - BAND_C[0] + 1
    c6 = max_kl(ho[BAND_C[0]:BAND_C[1] + 1], hc[:n], head)
    np.save(CHECK / "gates_values.npy", np.array([c4, c6]))
    say("GATES_JSON", {"C4": "PASS" if c4 <= C4_KL_MAX else "FAIL", "C6": "PASS" if c6 <= C6_KL_MAX else "FAIL"})
    del head
    import onnxruntime as ort
    from onnx import TensorProto as P, helper
    for name, gg, K, N, x in p3c.LINEARS:
        X = np.load(CHECK / f"x16_{x}.npy")
        codes, d = gd.gguf_linear(mm, by[f"blk.16.{gg}.weight"])
        node = helper.make_node("MatMulNBits", ["x", "b", "s"], ["y"], domain="com.microsoft", K=K, N=N, bits=4,
                                block_size=32, accuracy_level=4)
        g = helper.make_graph([node], "c2", [helper.make_tensor_value_info("x", P.FLOAT, [len(X), K])],
                              [helper.make_tensor_value_info("y", P.FLOAT, [len(X), N])],
                              [helper.make_tensor("b", P.UINT8, list(gd.pack_ort(codes).shape), gd.pack_ort(codes).tobytes(), raw=True),
                               helper.make_tensor("s", P.FLOAT, [N * (K // 32)], gc.f16_to_f32(d).astype(np.float32).tobytes(), raw=True)])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                              ir_version=10)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.intra_op_num_threads = THREADS
        y = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"]).run(None, {"x": X})[0]
        np.save(CHECK / f"c2_nb4_{name}.npy", y)
        qa, sa = q8_ort(X)
        qb, sb = q8_llama(X)
        ya, yb = q4_dot(qa, sa, codes, d), q4_dot(qb, sb, codes, d)
        say("C2_JSON", {"name": name, "nb4_vs_ort_form": rel_l2(y, ya), "nb4_vs_llama_form": rel_l2(y, yb),
                        "ort_form_vs_llama_form": rel_l2(ya, yb)})
    return 0


# ---------------------------------------------------------------- states

def states() -> int:
    print(f"STATES (hybrid S1), no chip. {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    print_hashes()
    if not frozen_ok():
        print("STATES STOP: the frozen hashes differ from the prereg log", flush=True)
        return 2
    b = logged_json(latest("hybrid_s1_build_*.log"), "BUILD_JSON")
    c = logged_json(latest("hybrid_s1_check_*.log"), "CHECK_JSON")
    say("GATES_READ_JSON", {"build_ok": b["ok"], "C4": c["C4"], "C5": c["C5"], "C6": c["C6"], "C1": c["C1"]})
    if not b["ok"] or c["stop"]:
        print("STATES STOP: the build or C4/C5 is not OK", flush=True)
        return 2
    for f, v in b["files"].items():
        if sha_file(MODELS / f) != v["sha256"]:
            print(f"STATES STOP: {f} differs from the build log", flush=True)
            return 2
    rcs = {}
    for arm in ARMS:
        rcs[arm] = child(["_states", arm], retries=1)
        if rcs[arm] == 4:
            print("STATES STOP: memory gate; not a verdict, re-run later", flush=True)
            return 4
    if c["C6"] == "PASS":
        rcs["cont"] = child(["_cont"], retries=1)
    else:
        print("BAND C NOT RUN: C6 failed (band C is report-only)", flush=True)
    if KVDIR.exists():
        shutil.rmtree(KVDIR)
    print(f"KV_DELETED {not KVDIR.exists()}", flush=True)
    say("STATES_DONE_JSON", {"rc": rcs})
    ok = all(rcs[a] == 0 for a in ARMS)                        # band C is report-only: it never fails STATES
    if rcs.get("cont", 0) != 0:
        print("BAND C INCOMPLETE (report-only; STATES stands on the five models)", flush=True)
    print("STATES", "OK" if ok else "INCOMPLETE", flush=True)
    return 0 if ok else 2


def states_child(arm: str) -> int:
    start_gate(f"states {arm}")
    seqs, _ = pinned_sequences()
    d = STATES / arm
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    kd = KVDIR / arm
    kd.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sess = session(MODELS / f"{arm}.onnx")
    say("SESSION_JSON", {"arm": arm, "seconds": round(time.perf_counter() - t0, 1), **own_memory()})
    for s, ids in enumerate(seqs):
        t1 = time.perf_counter()
        h, kv = run(sess, ids, want_kv=s < C_SEQ)
        np.save(d / f"seq{s:02d}.npy", h)
        if kv is not None:
            np.savez(kd / f"c{s}.npz", **kv)
            del kv
        say("STATE_JSON", {"arm": arm, "seq": s, "sha256": arr_sha(h), "finite": bool(np.isfinite(h).all()),
                           "seconds": round(time.perf_counter() - t1, 1)})
    say("MEM_JSON", {"child": f"states {arm}", **own_memory()})
    return 0


def cont_child() -> int:
    start_gate("band C (R0 continuations)")
    _, cseqs = pinned_sequences()
    d = STATES / "C"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    sess = session(MODELS / "R0.onnx")
    for c, ids in enumerate(cseqs):
        h, _ = run(sess, ids)
        np.save(d / f"oneshot_c{c}.npy", h)
        say("ONESHOT_JSON", {"c": c, "sha256": arr_sha(h), "finite": bool(np.isfinite(h).all())})
    for arm in ARMS:
        for c, ids in enumerate(cseqs):
            if not (KVDIR / arm / f"c{c}.npz").exists():             # its states child failed twice: report-only gap
                say("CONT_SKIPPED_JSON", {"arm": arm, "c": c, "why": "no KV from the arm's states run"})
                continue
            kv = dict(np.load(KVDIR / arm / f"c{c}.npz"))
            h, _ = run(sess, ids[SEQ_LEN:], past=kv)
            del kv
            np.save(d / f"{arm}_c{c}.npy", h)
            say("CONT_JSON", {"arm": arm, "c": c, "sha256": arr_sha(h), "finite": bool(np.isfinite(h).all())})
    say("MEM_JSON", {"child": "band C", **own_memory()})
    return 0


# ---------------------------------------------------------------- verdict

def verdict() -> int:
    print(f"VERDICT (hybrid S1), no chip. {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    print_hashes()
    if not frozen_ok():
        print("VERDICT REFUSED: the frozen hashes differ from the prereg log", flush=True)
        return 3
    start_gate("verdict")
    bl, cl, sl = latest("hybrid_s1_build_*.log"), latest("hybrid_s1_check_*.log"), latest("hybrid_s1_states_*.log")
    for p in (bl, cl, sl):
        print(f"INPUT_LOG {p.name}: LF sha256 {sha_lf(p)}", flush=True)
    b, c = logged_json(bl, "BUILD_JSON"), logged_json(cl, "CHECK_JSON")
    st = {(r["arm"], r["seq"]): r for r in logged_json(sl, "STATE_JSON", every=True)}
    cont = {(r["arm"], r["c"]): r for r in logged_json(sl, "CONT_JSON", every=True)}
    one = {r["c"]: r for r in logged_json(sl, "ONESHOT_JSON", every=True)}
    seqs, cseqs = pinned_sequences()
    head, mm, by = load_head()
    if arr_sha(head) != b["head_sha256"]:
        print("VERDICT REFUSED: the head differs from the build's", flush=True)
        return 3

    def load(path: Path, want: str):
        if not path.exists():
            return None
        a = np.load(path)
        return a if arr_sha(a) == want else None

    complete = {arm: all((arm, s) in st and st[(arm, s)]["finite"] is not None for s in range(N_SEQ)) for arm in ARMS}
    per = {arm: {"kl": [], "top1": [], "nll": [], "finite": True} for arm in ARMS if arm != "R0"}
    ref_nll = []
    for s, ids in enumerate(seqs):
        h0 = load(STATES / "R0" / f"seq{s:02d}.npy", st.get(("R0", s), {}).get("sha256", ""))
        if h0 is None:
            complete = {a: False for a in ARMS}
            break
        hs = {}
        for arm in per:
            hs[arm] = load(STATES / arm / f"seq{s:02d}.npy", st.get((arm, s), {}).get("sha256", ""))
            if hs[arm] is None:
                complete[arm] = False
        rows = {arm: {"kl": [], "top1": [], "nll": []} for arm in per}
        r_nll = []
        for r in range(0, SEQ_LEN - 1, CHUNK):
            e = min(r + CHUNK, SEQ_LEN - 1)
            nxt = np.asarray(ids[r + 1:e + 1])
            z0 = logits_chunk(h0[r:e], head)
            lp0 = log_softmax_rows(z0)
            p0 = np.exp(lp0)
            top0 = np.argmax(z0, axis=1)
            r_nll.append(-lp0[np.arange(len(nxt)), nxt])
            for arm in per:
                if hs[arm] is None:
                    continue
                k = kl_top1_chunk(lp0, p0, top0, logits_chunk(hs[arm][r:e], head), nxt)
                per[arm]["finite"] &= k["finite"] and bool(np.isfinite(hs[arm]).all())
                for f in ("kl", "top1", "nll"):
                    rows[arm][f].append(k[f])
        ref_nll.append(np.concatenate(r_nll))
        for arm in per:
            if hs[arm] is not None:
                for f in ("kl", "top1", "nll"):
                    per[arm][f].append(np.concatenate(rows[arm][f]))
        print(f"SEQUENCE {s} scored", flush=True)
    metrics = {}
    for arm, v in per.items():
        if not complete[arm] or len(v["kl"]) != N_SEQ:
            metrics[arm] = {"finite": v["finite"], "complete": False}
            continue
        kl, t1, nl = np.stack(v["kl"]), np.stack(v["top1"]), np.stack(v["nll"])
        metrics[arm] = {"finite": v["finite"], "complete": True}
        for band, rr in BANDS.items():
            metrics[arm][band] = aggregate(kl[:, band_rows(rr)], t1[:, band_rows(rr)], nl[:, band_rows(rr)])
    rn = np.stack(ref_nll) if len(ref_nll) == N_SEQ else None
    gates_ok = bool(b["ok"]) and c["C4"] == "PASS" and c["C5"] == "PASS"
    out = outcomes(metrics, c["C1"], gates_ok, {a: metrics.get(a, {}).get("complete", False) for a in per})
    sc = score(out, metrics) if all(metrics.get(a, {}).get("complete") for a in ("N1", "N2", "N3", "R")) else \
        {q: "NOT SCORED" for q, _ in PREDICTIONS}
    # band C (report-only)
    bandc = {}
    if c["C6"] == "PASS" and len(one) == C_SEQ:
        n = BAND_C[1] - BAND_C[0] + 1
        for arm in ARMS:
            kl, t1 = [], []
            for cc, ids in enumerate(cseqs):
                ho = load(STATES / "C" / f"oneshot_c{cc}.npy", one[cc]["sha256"])
                h = load(STATES / "C" / f"{arm}_c{cc}.npy", cont.get((arm, cc), {}).get("sha256", ""))
                if ho is None or h is None:
                    kl = None
                    break
                for r in range(0, n, CHUNK):
                    e = min(r + CHUNK, n)
                    nxt = np.asarray(ids[BAND_C[0] + r + 1:BAND_C[0] + e + 1])
                    z0 = logits_chunk(ho[BAND_C[0] + r:BAND_C[0] + e], head)
                    lp0 = log_softmax_rows(z0)
                    k = kl_top1_chunk(lp0, np.exp(lp0), np.argmax(z0, axis=1), logits_chunk(h[r:e], head), nxt)
                    kl.append(k["kl"])
                    t1.append(k["top1"])
            if kl is not None:
                kl, t1 = np.concatenate(kl), np.concatenate(t1)
                bandc[arm] = {"positions": int(kl.size), "kl_mean": float(kl.mean()),
                              "kl_p99": float(np.quantile(kl, 0.99)), "top1": float(t1.mean())}
    gv = np.load(CHECK / "gates_values.npy") if (CHECK / "gates_values.npy").exists() else [None, None]
    diag = layer16_diagnostics(mm, by)
    report(metrics, out, sc, bandc, rn, gv, diag)
    say("VERDICT_JSON", {"outcomes": {a: out[a]["outcome"] for a in out}, "narrow": {a: out[a]["narrow"] for a in out},
                         "pick": pick(out)})
    say("PREDICTIONS_SCORED_JSON", sc)
    return 0 if all(out[a]["outcome"] != "INCOMPLETE" for a in DECIDING) else 2


def layer16_diagnostics(mm, by) -> dict:
    """Report-only: layer 16 of sequence 0 on R0's real inputs: R (C2's ORT output), N1, N2 and N3 (numpy
    emulations of section 4) against the exact product, and the outlier ratio per input."""
    import ml_dtypes
    out = {"rel_l2": {}, "outlier_ratio": {}}
    for x in L16_INPUTS:
        X = np.load(CHECK / f"x16_{x}.npy")
        amax_rows = np.abs(X).max(axis=1)
        out["outlier_ratio"][x] = float(np.abs(X).max() / np.median(amax_rows))
    for name, gg, K, N, x in p3c.LINEARS:
        X = np.load(CHECK / f"x16_{x}.npy")
        codes, d = gd.gguf_linear(mm, by[f"blk.16.{gg}.weight"])
        w = p3c.dense(codes, d)
        exact = X.astype(np.float64) @ w.astype(np.float64)
        wq, sw = p3c.quant_w(w)
        q1, s1 = p3c.quant_x(X)
        y1 = p3c.exact_int(q1, wq).astype(np.float64) * s1 * sw
        q2, s2 = quant_x_rows(X)
        y2 = p3c.exact_int(q2, wq).astype(np.float64) * s2[:, None] * sw
        xb = X.astype(ml_dtypes.bfloat16).astype(np.float64)
        wb = w.astype(ml_dtypes.bfloat16).astype(np.float64)
        y3 = xb @ wb
        yr = np.load(CHECK / f"c2_nb4_{name}.npy")
        out["rel_l2"][name] = {"R": rel_l2(yr, exact), "N1": rel_l2(y1, exact), "N2": rel_l2(y2, exact),
                               "N3": rel_l2(y3, exact)}
    return out


def fmt(x, spec=".4g"):
    return "-" if x is None else format(x, spec)


def report(metrics, out, sc, bandc, rn, gv, diag) -> None:
    print(f"\nTHE CHECK VALUES: C4 max per-position KL (R0 against torch, sequence 0) {fmt(gv[0], '.3g')} "
          f"(<= {C4_KL_MAX:g}); C6 max KL (R0 chunked against one-shot, positions 2048-3070) {fmt(gv[1], '.3g')} "
          f"(<= {C6_KL_MAX:g})", flush=True)
    if rn is not None:
        for band, rr in BANDS.items():
            print(f"R0's perplexity on the next tokens, band {band}: {np.exp(rn[:, band_rows(rr)].mean()):.4f}", flush=True)
    print(f"\nAgainst R0 (the rule: mean KL <= {T_KL} and top-1 >= {T_TOP1} in both bands L and H; R is report-only)")
    print(f"  {'arm':4s} {'band':4s} {'mean KL':>10s} {'95% CI':>23s} {'p50':>9s} {'p99':>9s} {'max':>9s} "
          f"{'top-1':>7s} {'95% CI':>17s} {'ppl':>8s}")
    for arm in ("N1", "N2", "N3", "R"):
        m = metrics.get(arm, {})
        if not m.get("complete"):
            print(f"  {arm:4s} INCOMPLETE (states missing)")
            continue
        for band in BANDS:
            a = m[band]
            print(f"  {arm:4s} {band:4s} {a['kl_mean']:10.5f} [{a['kl_ci'][0]:.5f}, {a['kl_ci'][1]:.5f}] "
                  f"{a['kl_p50']:9.5f} {a['kl_p99']:9.4f} {a['kl_max']:9.3f} {a['top1']:7.4f} "
                  f"[{a['top1_ci'][0]:.4f}, {a['top1_ci'][1]:.4f}] {a['ppl']:8.4f}")
    print("\nOutcomes:")
    for arm in ("N1", "N2", "N3", "R"):
        tag = " (report-only; decides nothing)" if arm in REPORT_ONLY else ""
        nb = f", NARROW in band {', '.join(out[arm]['narrow'])}" if out[arm]["narrow"] else ""
        print(f"  {arm}: {out[arm]['outcome']}{nb}{tag}")
    print(f"\nTHE PICK: {pick(out)}")
    if bandc:
        print("\nBand C (REPORT-ONLY): the arm's prompt KV, then R0 over positions 2048-3071, against R0 one-shot")
        for arm, v in bandc.items():
            print(f"  {arm:4s} mean KL {v['kl_mean']:.5f}, p99 {v['kl_p99']:.4f}, top-1 {v['top1']:.4f} "
                  f"({v['positions']} positions)")
    print("\nLayer 16, sequence 0 (report-only): rel-L2 against the exact product on R0's real inputs")
    for name, v in diag["rel_l2"].items():
        print(f"  {name:5s} " + "  ".join(f"{k} {v[k]:.3e}" for k in ("R", "N1", "N2", "N3")))
    print("  outlier ratio (per-tensor amax / median per-token amax): " +
          ", ".join(f"{k} {v:.1f}" for k, v in diag["outlier_ratio"].items()))
    print("\nPredictions (they decide nothing):")
    for q, text in PREDICTIONS:
        print(f"  {q} {sc[q]}: {text}")
    say("METRICS_JSON", metrics)
    say("BAND_C_JSON", bandc)
    say("LAYER16_JSON", diag)


# ---------------------------------------------------------------- selftest

def selftest() -> int:
    import tempfile
    import ml_dtypes
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto as P, helper
    fails = []

    def expect(what, got, want=True):
        print(f"  {what}: {got}" + ("" if got == want else f"  (expected {want})"), flush=True)
        if got != want:
            fails.append(what)

    def sess_of(m):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.intra_op_num_threads = 2
        return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(1)
    print("Quantize subgraph against 3c's quant_x (float64, RNE), ties and a zero row included:")
    M, K, N = 6, 64, 8
    x = rng.standard_normal((M, K)).astype(np.float32)
    x[0, :4] = [2.5, -2.5, 0.5, 1.5]
    x[0, 5] = 127 * 0.05                                  # amax set so that exact halves occur
    x[3] = 0.0
    for per_token in (False, True):
        qn, u8, s = quant_nodes("x", per_token)
        g = helper.make_graph(qn, "q", [helper.make_tensor_value_info("x", P.FLOAT, [1, M, K])],
                              [helper.make_tensor_value_info(u8, P.UINT8, [1, M, K]),
                               helper.make_tensor_value_info(s, P.DOUBLE, None)], used_consts(qn))
        u, sv = sess_of(helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)).run(None, {"x": x[None]})
        if per_token:
            q, sw = quant_x_rows(x)
            expect("per token: codes equal numpy's", bool(np.array_equal(u[0].astype(np.int16) - 128, q)))
            expect("per token: the zero row gives codes 0 and scale 0", bool((u[0][3] == 128).all() and sv[0, 3, 0] == 0))
        else:
            xn = x.copy()
            xn[3] = 0.0
            q, sx = p3c.quant_x(xn)
            expect("per tensor: codes equal 3c's quant_x", bool(np.array_equal(u[0].astype(np.int16) - 128, q)))
            expect("per tensor: scale equals 3c's", float(sv.reshape(-1)[0]) == sx)
    print("int8 linear: MatMulInteger's int32 against 3c's exact_int, the float64 epilogue:")
    w = (rng.standard_normal((K, N)) * 0.05).astype(np.float32)
    wq, sw = p3c.quant_w(w)
    for per_token in (False, True):
        qn, u8, s = quant_nodes("x", per_token)
        nodes = qn + int8_nodes("/lin", u8, s, "wq", "sw", "y", i32="i32")
        inits = used_consts(nodes) + [helper.make_tensor("wq", P.INT8, [K, N], wq.tobytes(), raw=True),
                                   helper.make_tensor("sw", P.DOUBLE, [N], sw.tobytes(), raw=True)]
        g = helper.make_graph(nodes, "l", [helper.make_tensor_value_info("x", P.FLOAT, [1, M, K])],
                              [helper.make_tensor_value_info("i32", P.INT32, [1, M, N]),
                               helper.make_tensor_value_info("y", P.FLOAT, [1, M, N])], inits)
        i32, y = sess_of(helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)).run(None, {"x": x[None]})
        if per_token:
            q, sx = quant_x_rows(x)
            want = (p3c.exact_int(q, wq).astype(np.float64) * sx[:, None] * sw).astype(np.float32)
        else:
            q, sx = p3c.quant_x(x)
            want = (p3c.exact_int(q, wq).astype(np.float64) * sx * sw).astype(np.float32)
        tag = "per token" if per_token else "per tensor"
        expect(f"{tag}: int32 exact", bool(np.array_equal(i32[0], p3c.exact_int(q, wq))))
        expect(f"{tag}: epilogue fp32(float64(int32) * s_x * s_w)", bool(np.array_equal(y[0], want)))
    print("bf16 casts round to nearest even (ties included), as ml_dtypes does:")
    one = np.float32(1.0)
    ties = np.array([1 + 2 ** -8, 1 + 3 * 2 ** -8, -(1 + 2 ** -8), 1 + 2 ** -8 + 2 ** -20, 3.14159, 1e-30, 65504.0],
                    dtype=np.float32)
    xt = np.concatenate([ties, rng.standard_normal(57).astype(np.float32)]).reshape(1, 1, 64)
    bn, af = bf16_in_nodes("x")
    g = helper.make_graph(bn, "b", [helper.make_tensor_value_info("x", P.FLOAT, [1, 1, 64])],
                          [helper.make_tensor_value_info(af, P.FLOAT, [1, 1, 64])])
    got = sess_of(helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)).run(None, {"x": xt})[0]
    want = xt.astype(ml_dtypes.bfloat16).astype(np.float32)
    expect("ORT Cast float->bf16->float equals ml_dtypes RNE bit for bit", bool(np.array_equal(got.view(np.uint32), want.view(np.uint32))))
    expect("the tie 1 + 2^-8 rounds to 1 (even)", float(want.reshape(-1)[0]) == float(one))
    print("The surgery on a synthetic graph shaped like (c)'s (two layers, a head):")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        K2, N2 = 64, 32
        wts = {}
        data = DataFile(td / "model.onnx.data")
        nodes, inits = [], []
        prev = "x"
        for L in range(2):
            for proj, kin, nout, inp in (("q", K2, K2, prev), ("k", K2, K2, prev), ("o", K2, K2, None)):
                nm = f"/model/layers.{L}/attn/{proj}_proj/MatMulNBits"
                codes = rng.integers(0, 16, (nout, kin // 32, 32)).astype(np.uint8)
                d = (np.abs(rng.standard_normal(nout * kin // 32)) * 0.01 + 0.001).astype(np.float16).view(np.uint16)
                wts[nm] = p3c.dense(codes, d)
                a = inp if inp is not None else f"/model/layers.{L}/add/output_0"
                qb = data.add(f"model.layers.{L}.{proj}.qweight", gd.pack_ort(codes), P.UINT8, list(gd.pack_ort(codes).shape))
                sc_ = data.add(f"model.layers.{L}.{proj}.scales", gc.f16_to_f32(d).astype(np.float32), P.FLOAT, [nout * kin // 32])
                inits += [qb, sc_]
                nodes.append(helper.make_node("MatMulNBits", [a, qb.name, sc_.name], [nm + "/output_0"], name=nm,
                                              domain="com.microsoft", K=kin, N=nout, bits=4, block_size=32, accuracy_level=0))
                if proj == "k":
                    nodes.append(helper.make_node("Add", [f"/model/layers.{L}/attn/q_proj/MatMulNBits/output_0", nm + "/output_0"],
                                                  [f"/model/layers.{L}/add/output_0"], name=f"/model/layers.{L}/add"))
            prev = f"/model/layers.{L}/attn/o_proj/MatMulNBits/output_0"
        nodes.append(helper.make_node("Identity", [prev], [HIDDEN_NAME], name="/model/layers.34/final_norm_layernorm"))
        hw = data.add(HEAD_W, rng.standard_normal((K2, 16)).astype(np.float32), P.FLOAT, [K2, 16])
        inits.append(hw)
        nodes.append(helper.make_node("MatMul", [HIDDEN_NAME, HEAD_W], ["logits"], name=HEAD_NODE))
        data.close()
        g = helper.make_graph(nodes, "syn", [helper.make_tensor_value_info("x", P.FLOAT, [1, "S", K2])],
                              [helper.make_tensor_value_info("logits", P.FLOAT, [1, "S", 16])], inits)
        onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                                    ir_version=10), str(td / "model.onnx"))
        real_nb = globals()["nb_nodes"]
        globals()["nb_nodes"] = lambda gr: [n for n in gr.node if n.op_type == "MatMulNBits"]
        try:
            base = base_cut(td / "model.onnx", "model.onnx.data", K2)
            expect("head cut: the output is the hidden states", [o.name for o in base.graph.output], [HIDDEN_NAME])
            xin = rng.standard_normal((1, 5, K2)).astype(np.float32)
            outs = {}
            shared = {}
            i8 = DataFile(td / "int8.onnx.data")
            arms = {"R0": set_accuracy(base_cut(td / "model.onnx", "model.onnx.data", K2), 0),
                    "R": set_accuracy(base_cut(td / "model.onnx", "model.onnx.data", K2), 4),
                    "N1": replace_linears(base_cut(td / "model.onnx", "model.onnx.data", K2), "N1", lambda n: wts[n], i8, shared),
                    "N2": replace_linears(base_cut(td / "model.onnx", "model.onnx.data", K2), "N2", lambda n: wts[n], i8, shared)}
            i8.close()
            bf = DataFile(td / "bf16.onnx.data")
            arms["N3"] = replace_linears(base_cut(td / "model.onnx", "model.onnx.data", K2), "N3", lambda n: wts[n], bf)
            bf.close()
            for k, m in arms.items():
                onnx.save(m, str(td / f"{k}.onnx"))
                c3 = c3_compare(base, m, k)
                expect(f"C3 {k}", c3["ok"])
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                so.enable_profiling = True
                so.profile_file_prefix = str(td / f"prof_{k}")
                ss = ort.InferenceSession(str(td / f"{k}.onnx"), so, providers=["CPUExecutionProvider"])
                outs[k] = ss.run(None, {"x": xin})[0]
                prof = ss.end_profiling()
                expect(f"{k}: executed ops equal the graph's", profile_counts(prof), static_counts(m))
            bad = set_accuracy(base_cut(td / "model.onnx", "model.onnx.data", K2), 0)
            bad.graph.node[0].name = "changed"
            expect("C3 catches a changed node", c3_compare(base, bad, "R0")["ok"], False)

            def emulate(kind):
                h = xin[0]
                for L in range(2):
                    def lin(nm, a):
                        W = wts[nm]
                        if kind == "R0":
                            return (a.astype(np.float64) @ W.astype(np.float64)).astype(np.float32)
                        if kind == "N3":
                            ab = a.astype(ml_dtypes.bfloat16).astype(np.float64)
                            return (ab @ W.astype(ml_dtypes.bfloat16).astype(np.float64)).astype(np.float32)
                        q_, sw_ = p3c.quant_w(W)
                        if kind == "N1":
                            qa, sa = p3c.quant_x(a)
                            return (p3c.exact_int(qa, q_).astype(np.float64) * sa * sw_).astype(np.float32)
                        qa, sa = quant_x_rows(a)
                        return (p3c.exact_int(qa, q_).astype(np.float64) * sa[:, None] * sw_).astype(np.float32)
                    q = lin(f"/model/layers.{L}/attn/q_proj/MatMulNBits", h)
                    k = lin(f"/model/layers.{L}/attn/k_proj/MatMulNBits", h)
                    h = lin(f"/model/layers.{L}/attn/o_proj/MatMulNBits", (q + k).astype(np.float32))
                return h
            for k in ("R0", "N1", "N2", "N3"):
                e = rel_l2(outs[k][0], emulate(k))
                expect(f"{k}: the graph equals its numpy emulation (rel-L2 {e:.1e} <= 1e-5)", e <= 1e-5)
            expect("R (int8 activations) differs from R0", not np.array_equal(outs["R"], outs["R0"]))
        finally:
            globals()["nb_nodes"] = real_nb
    print("The metrics, the rule, the pick and the scoring:")
    z = rng.standard_normal((4, 50)).astype(np.float32)
    lp = log_softmax_rows(z)
    direct = z.astype(np.float64) - np.log(np.exp(z.astype(np.float64)).sum(axis=1, keepdims=True))
    expect("log-softmax equals the direct form", bool(np.allclose(lp, direct, rtol=0, atol=1e-12)))
    lsd = gds.log_softmax(z[1])
    expect("log-softmax equals (c)'s", bool(np.allclose(lp[1], lsd, rtol=0, atol=1e-12)))
    k = kl_top1_chunk(lp, np.exp(lp), np.argmax(z, axis=1), z, np.array([0, 1, 2, 3]))
    expect("KL of a model against itself is 0", bool(np.all(np.abs(k["kl"]) < 1e-12)))
    expect("top-1 against itself is 1", bool(np.all(k["top1"] == 1)))
    expect("band L is 1,024 positions, H 1,023, C 1,023", (len(range(2047)[band_rows(BANDS["L"])]),
                                                           len(range(2047)[band_rows(BANDS["H"])]),
                                                           BAND_C[1] - BAND_C[0] + 1), (1024, 1023, 1023))
    expect("bootstrap is deterministic", bootstrap(np.arange(16.0)) == bootstrap(np.arange(16.0)))

    def m_of(kl, t1, finite=True, ci=None):
        b = {"kl_mean": kl, "top1": t1, "kl_ci": ci or [kl, kl], "top1_ci": [t1, t1]}
        return {"finite": finite, "L": dict(b), "H": dict(b)}
    expect("rule: PASS at the thresholds", rule(m_of(T_KL, T_TOP1))[0], "PASS")
    expect("rule: FAIL above T_KL", rule(m_of(0.0124, 0.99))[0], "FAIL")
    expect("rule: FAIL below T_TOP1", rule(m_of(0.001, 0.955))[0], "FAIL")
    expect("rule: FAIL when non-finite", rule(m_of(0.0, 1.0, finite=False))[0], "FAIL")
    expect("rule: NARROW when the interval holds T_KL", rule(m_of(0.012, 0.99, ci=[0.011, 0.013]))[1], ["L", "H"])
    one_band = m_of(0.001, 0.99)
    one_band["H"]["kl_mean"] = 0.02
    expect("rule: both bands must pass", rule(one_band)[0], "FAIL")

    def o(**kw):
        return {a: {"outcome": kw.get(a, "FAIL"), "narrow": []} for a in ("N1", "N2", "N3", "R")}
    expect("pick: N2 first", pick(o(N1="PASS", N2="PASS", N3="PASS")), "N2")
    expect("pick: N1 with the tile requirement", pick(o(N1="PASS", N3="PASS")).startswith("N1: the hybrid must feed 2,048-row"), True)
    expect("pick: N3 when neither int8 arm passes", pick(o(N3="PASS")).startswith("N3"), True)
    expect("pick: none", pick(o()).startswith("NONE"), True)
    expect("pick: an INCOMPLETE N2 leaves the pick INCOMPLETE", pick(o(N2="INCOMPLETE", N1="PASS")), "INCOMPLETE")
    expect("outcomes: a C1 failure voids the arm", outcomes({"N1": m_of(0.001, 0.99), "N2": m_of(0.001, 0.99),
                                                             "N3": m_of(0.001, 0.99), "R": m_of(0.001, 0.99)},
                                                            {"N1": False, "N2": True, "N3": True}, True,
                                                            {a: True for a in ("N1", "N2", "N3", "R")})["N1"]["outcome"],
           "INCOMPLETE")
    mets = {"N1": m_of(0.05, 0.9), "N2": m_of(0.005, 0.97), "N3": m_of(0.0005, 0.995), "R": m_of(0.002, 0.98)}
    outs_ = outcomes(mets, {"N1": True, "N2": True, "N3": True}, True, {a: True for a in mets})
    expect("scores", score(outs_, mets), {"P1": "HIT", "P2": "HIT", "P3": "HIT", "P4": "HIT"})
    print("The emulations of C2 on synthetic blocks:")
    xx = rng.standard_normal((3, 64)).astype(np.float32)
    codes = rng.integers(0, 16, (5, 2, 32)).astype(np.uint8)
    d = (np.abs(rng.standard_normal(10)) * 0.01 + 0.001).astype(np.float16).view(np.uint16)
    qa, sa = q8_ort(xx)
    y = q4_dot(qa, sa, codes, d)
    brute = np.zeros((3, 5))
    dw = gc.f16_to_f32(d).reshape(5, 2)
    for m in range(3):
        for n in range(5):
            for b in range(2):
                amax = np.abs(xx[m, 32 * b:32 * b + 32]).max()
                inv = np.float32(127) / amax
                qq = np.rint((xx[m, 32 * b:32 * b + 32] * inv).astype(np.float32))
                brute[m, n] += float((qq * (codes[n, b].astype(np.float64) - 8)).sum()) * float(np.float32(amax / np.float32(127))) * float(dw[n, b])
    expect("q4_dot equals a brute-force loop", bool(np.allclose(y, brute, rtol=1e-12, atol=0)))
    ql, sl = q8_llama(xx)
    expect("llama form differs only in the fp16 scale", bool(np.array_equal(ql, qa)) and not np.array_equal(sl, sa))
    print("Frozen hashes:")
    for kk, v in hashes().items():
        print(f"  {kk} {v}")
    expect("PREREG holds the plan", "<<PLAN>>" not in PREREG and PREREG.startswith("S1 PLAN AND PREREG"))
    print("SELFTEST", "OK" if not fails else f"FAILED: {fails}", flush=True)
    return 0 if not fails else 1


# ---------------------------------------------------------------- main

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("selftest", "prereg", "build", "check", "states", "verdict", "_probe", "_c1", "_r0",
                                     "_probe16", "_torch", "_compare", "_states", "_cont"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.mode == "_probe":
        return probe_child(a.args[0])
    if a.mode == "_states":
        return states_child(a.args[0])
    return {"selftest": selftest, "prereg": prereg, "build": build, "check": check, "states": states, "verdict": verdict,
            "_c1": c1_child, "_r0": r0_child, "_probe16": probe16_child, "_torch": torch_child, "_compare": compare_child,
            "_cont": cont_child}[a.mode]()


if __name__ == "__main__":
    sys.exit(main())
