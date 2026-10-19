#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small-LLM decode, step 1: every weight projection of one SmolLM2-135M token on the whole-array GEMV.

The question before any attention, norm or sampling work: can an M=1 projection stream a token's weights from a
resident DDR buffer over all 8 shim channels into compute cores, exactly, and what does a token's set of projections
cost once dispatch overhead is in? One token is 121 GEMVs of the real bf16 weights (30 layers x {qkv 960x576,
o 576x576, gate+up 3072x576, down 576x1536} + the tied 49152x576 head), 269 MB, every one checked bit for bit against
the emulator's MAC model (kernels/llm_gemv/oracle.py). Three ways to issue them:

    A  one dispatch per GEMV (121), each on a sub-buffer of the arena at that matrix
    B  one dispatch per layer (31): the layer's 4 GEMVs in one instruction stream, plus the head
    C  one dispatch per token: all 121 in one stream

    python tools/llm_gemv_probe.py compile                   # build/llm_gemv/<program>/ (mlir-aie toolchain)
    python tools/llm_gemv_probe.py oracle --seed 0           # exact expected outputs, build/llm_gemv/oracle_s0.npz
    python tools/llm_gemv_probe.py run --seeds 0 1           # the sitting (NPU): exactness, then timing rounds
    python tools/llm_gemv_probe.py run --seeds 0 --exact-only   # exactness alone (a first run of a new build)

On the NPU, run it through the wrapper, with the iron env's python as the child (resnet_env17 has no pyxrt):

    ./scripts/research-lowlevel.sh --log results/aie/<name>.log --npu --seconds 600 -- \\
        <mlir-aie-iron>/python.exe tools/llm_gemv_probe.py run

x is random per seed (one vector per shape, reused by every layer); the weights are the model's. The bandwidth
figures count weight bytes only. This is not a decoder: nothing feeds one projection's output into the next.
"""
import argparse
import gc
import json
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kernels.llm_gemv import oracle, pack  # noqa: E402

BUILD = ROOT / "build" / "llm_gemv"
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
LAYERS, HIDDEN, MLP, VOCAB, KV = 30, 576, 1536, 49152, 192
SHAPES = {"qkv": (HIDDEN + 2 * KV, HIDDEN), "o": (HIDDEN, HIDDEN), "gu": (2 * MLP, HIDDEN),
          "down": (HIDDEN, MLP), "head": (VOCAB, HIDDEN)}
LAYER_ORDER = ("qkv", "o", "gu", "down")
WARMUP, TOKENS, ROUNDS = 3, 20, 2
WAIT_MS = 5000


def words(name):
    return pack.block_words(*SHAPES[name])


LAYER_WORDS = sum(words(s) for s in LAYER_ORDER)          # 1,769,472: 7.08 MB, a multiple of 4 KiB
ARENA_WORDS = LAYERS * LAYER_WORDS + words("head")
X_SLOTS = {s: i * pack.X_SLOT_WORDS for i, s in enumerate(SHAPES)}
X_WORDS = len(SHAPES) * pack.X_SLOT_WORDS
OUT = {}
_o = 0
for _s in SHAPES:
    OUT[_s] = _o
    _o += SHAPES[_s][0] // 2
OUT_WORDS = _o


def layer_offset(layer):
    return layer * LAYER_WORDS


def matrix_offset(layer, shape):
    if shape == "head":
        return LAYERS * LAYER_WORDS
    off = layer_offset(layer)
    for s in LAYER_ORDER:
        if s == shape:
            return off
        off += words(s)
    raise KeyError(shape)


# ---------------------------------------------------------------------------- weights

def safetensors(path):
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


def model_file():
    """The cached model.safetensors (fetch it once with ``hf download HuggingFaceTB/SmolLM2-135M-Instruct``)."""
    import os
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    found = sorted((home / "hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots").glob("*/model.safetensors"))
    if len(found) != 1:
        raise SystemExit(f"expected one cached {MODEL} snapshot under {home}, found {len(found)}")
    return found[0]


def matrices(path):
    """(layer, shape) -> uint16 [N][K], the projection as y = W x."""
    t = safetensors(path)
    p = "model.layers.{}.{}"
    m = {}
    for i in range(LAYERS):
        g = lambda k: t[p.format(i, k)]  # noqa: E731
        m[i, "qkv"] = np.concatenate([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                                      g("self_attn.v_proj.weight")])
        m[i, "o"] = g("self_attn.o_proj.weight")
        m[i, "gu"] = np.concatenate([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")])
        m[i, "down"] = g("mlp.down_proj.weight")
    m[0, "head"] = t["model.embed_tokens.weight"]
    for (i, s), w in m.items():
        if w.shape != SHAPES[s]:
            raise ValueError(f"layer {i} {s}: {w.shape}, expected {SHAPES[s]}")
    return m


def token_calls():
    """The 121 GEMVs of one token, in order: (layer, shape)."""
    return [(i, s) for i in range(LAYERS) for s in LAYER_ORDER] + [(0, "head")]


def xs(seed):
    from ignite_xdna.compiler import engine_bf16_emulator as em
    rng = np.random.default_rng(seed)
    return {s: em.bf16_bits(em.to_bf16(rng.normal(0, 1, SHAPES[s][1]).astype(np.float32))) for s in SHAPES}


# ---------------------------------------------------------------------------- programs

def programs():
    from kernels.llm_gemv.design import Call
    P = {}
    for s in SHAPES:          # arm A: w is a sub-buffer starting at the matrix
        n, k = SHAPES[s]
        P[f"one_{s}"] = ([Call(0, X_SLOTS[s], OUT[s], n, k)], words(s))
    layer, off = [], 0         # arm B: w is a sub-buffer starting at the layer
    for s in LAYER_ORDER:
        n, k = SHAPES[s]
        layer.append(Call(off, X_SLOTS[s], OUT[s], n, k))
        off += words(s)
    P["layer"] = (layer, LAYER_WORDS)
    P["token"] = ([Call(matrix_offset(i, s), X_SLOTS[s], OUT[s], *SHAPES[s]) for i, s in token_calls()],
                  ARENA_WORDS)  # arm C: the whole arena
    return P


def compile_all():
    from kernels.llm_gemv import design
    from kernels.transformer_bf16.common import sha
    configs = {}
    for label, (calls, w_words) in programs().items():
        d = BUILD / label
        if (d / "build.json").exists():
            meta = json.loads((d / "build.json").read_text())
            print("EXISTS", label, meta["artifacts"]["insts.bin"], flush=True)
        else:
            meta = design.build(d, calls, w_words, X_WORDS, OUT_WORDS, label)
        # Every program runs in one hardware context (the token program's xclbin), so what configures the array -
        # the CDOs, the core images and the kernel signature - must be byte-identical across them. The xclbins
        # themselves are not: each build stamps a fresh UUID and a few PDI header bytes.
        prj = d / "prj"
        configs[label] = tuple(sorted((str(p.relative_to(prj)), sha(p)) for p in
                                      [*prj.glob("cdo_main/*.bin"), *prj.rglob("*core_*.elf"), prj / "kernels_main.json"]))
    distinct = set(configs.values())
    print("ARRAY_CONFIG_SETS", len(distinct), "files_per_set", len(next(iter(distinct))), flush=True)
    if len(distinct) != 1:
        raise SystemExit("the programs configure the array differently; they cannot share one context")


# ---------------------------------------------------------------------------- oracle

def oracle_path(seed):
    return BUILD / f"oracle_s{seed}.npz"


def compute_oracle(seed):
    path = oracle_path(seed)
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    m, x = matrices(model_file()), xs(seed)
    out = {}
    t0 = time.perf_counter()
    for i, s in token_calls():
        out[f"{i}_{s}"] = oracle.gemv(m[i, s], x[s])
    np.savez(path, **out, **{f"x_{s}": v for s, v in x.items()})
    print(f"ORACLE seed {seed}: {len(out)} GEMVs in {time.perf_counter() - t0:.1f} s -> {path}", flush=True)


# ---------------------------------------------------------------------------- device

def witness(tag, wait_s=0.0):
    """xrt-smi idle? With wait_s, poll that long first: a released context takes a moment to leave the report."""
    deadline = time.monotonic() + wait_s
    while True:
        out = subprocess.run(["C:/Windows/System32/AMD/xrt-smi.exe", "examine", "-r", "aie-partitions"],
                             capture_output=True, text=True).stdout
        idle = "No hardware contexts running" in out
        if idle or time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    print(f"WITNESS {tag} idle={idle}", flush=True)
    return idle


class Device:
    def __init__(self, h, m):
        self.h, self.xrt = h, h.pyxrt
        xclbin = BUILD / "token" / "final.xclbin"
        h.load_xclbin(str(xclbin))
        self.to = self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.fr = self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.arena = h.create_host_bo(ARENA_WORDS * 4, 3)
        view = np.frombuffer(self.arena.map(), np.uint32)
        for i, s in token_calls():
            o = matrix_offset(i, s)
            view[o:o + words(s)] = pack.pack(m[i, s])
        self.arena.sync(self.to)
        self.x = h.create_host_bo(X_WORDS * 4, 4)
        self.out = h.create_host_bo(OUT_WORDS * 4, 5)
        self.insts = {label: h.create_instruction_bo((BUILD / label / "insts.bin").as_posix())
                      for label in programs()}
        self.subs = []   # keep sub-buffers alive while their runs exist

    def sub(self, offset_words, n_words):
        b = self.xrt.bo(self.arena, n_words * 4, offset_words * 4)
        self.subs.append(b)
        return b

    def run(self, label, w):
        bo, count = self.insts[label]
        r = self.xrt.run(self.h.kernel)
        for i, v in enumerate((3, bo, count, w, self.x, self.out)):
            r.set_arg(i, v)
        return r

    def arms(self):
        """Arm -> the run objects one token issues, in order, built once."""
        a = [self.run(f"one_{s}", self.sub(matrix_offset(i, s), words(s))) for i, s in token_calls()]
        b = [self.run("layer", self.sub(layer_offset(i), LAYER_WORDS)) for i in range(LAYERS)]
        b.append(self.run("one_head", self.sub(matrix_offset(0, "head"), words("head"))))
        c = [self.run("token", self.arena)]
        return {"A": a, "B": b, "C": c}

    def set_x(self, x):
        v = np.concatenate([pack.x_slot(x[s], SHAPES[s][0]) for s in SHAPES])
        self.x.write(v.tobytes(), 0)
        self.x.sync(self.to)

    def poison(self):
        self.out.write(np.full(OUT_WORDS, 0x7FC9_8765, np.uint32).tobytes(), 0)
        self.out.sync(self.to)

    def y(self, shape):
        self.out.sync(self.fr)
        w = np.frombuffer(self.out.read(OUT_WORDS * 4, 0), np.uint32)
        return pack.unpack_y(w[OUT[shape]:], SHAPES[shape][0])

    def go(self, r):
        r.start()
        state = r.wait(WAIT_MS)
        if state != self.xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"dispatch did not complete: {state}; stopping without retry")


def check(dev, arms, seed, ref):
    """Exactness, every arm: A checks all 121 outputs, B every layer's 4 and the head, C the last layer and head."""
    bad = {}
    calls = token_calls()
    dev.poison()
    for r, (i, s) in zip(arms["A"], calls):
        dev.go(r)
        bad[f"A_{i}_{s}"] = int(np.count_nonzero(dev.y(s) != ref[f"{i}_{s}"]))
    dev.poison()
    for i, r in enumerate(arms["B"]):
        dev.go(r)
        for s in (LAYER_ORDER if i < LAYERS else ("head",)):
            bad[f"B_{0 if s == 'head' else i}_{s}"] = int(np.count_nonzero(dev.y(s) != ref[f"{0 if s == 'head' else i}_{s}"]))
    dev.poison()
    dev.go(arms["C"][0])
    for s in LAYER_ORDER:
        bad[f"C_{LAYERS - 1}_{s}"] = int(np.count_nonzero(dev.y(s) != ref[f"{LAYERS - 1}_{s}"]))
    bad["C_0_head"] = int(np.count_nonzero(dev.y("head") != ref["0_head"]))
    total = sum(bad.values())
    print("EXACT_JSON " + json.dumps(dict(seed=seed, checked=len(bad), mismatched_elements=total,
                                          failures={k: v for k, v in bad.items() if v})), flush=True)
    return total == 0


def time_arm(dev, runs):
    per_token = []
    for t in range(WARMUP + TOKENS):
        t0 = time.perf_counter_ns()
        for r in runs:
            dev.go(r)
        dt = (time.perf_counter_ns() - t0) / 1e6
        if t >= WARMUP:
            per_token.append(dt)
    return per_token


def run(seeds, exact_only=False):
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    if not witness("pre"):
        raise SystemExit("NPU is busy; did not dispatch")
    m = matrices(model_file())
    weight_bytes = ARENA_WORDS * 4
    ok = True
    try:
        with XrtSiliconHarness(0) as h:
            dev = Device(h, m)
            arms = dev.arms()
            print("ARMS " + json.dumps({k: len(v) for k, v in arms.items()}), flush=True)
            for seed in seeds:
                ref = dict(np.load(oracle_path(seed)))
                x = {s: ref[f"x_{s}"] for s in SHAPES}
                if any(not np.array_equal(x[s], v) for s, v in xs(seed).items()):
                    raise SystemExit(f"oracle_s{seed}.npz was made from different x")
                dev.set_x(x)
                ok &= check(dev, arms, seed, ref)
            if not ok:
                print("TIMING SKIPPED: not exact", flush=True)
                return 1
            if exact_only:
                print("TIMING SKIPPED: --exact-only", flush=True)
                return 0
            for rnd in range(1, ROUNDS + 1):
                for arm in ("A", "B", "C"):
                    t = time_arm(dev, arms[arm])
                    med = statistics.median(t)
                    print("RESULT_JSON " + json.dumps(dict(
                        round=rnd, arm=arm, dispatches_per_token=len(arms[arm]), tokens=TOKENS, warmup=WARMUP,
                        times_ms=t, median_ms=med, min_ms=min(t), max_ms=max(t),
                        weight_bytes=weight_bytes, gbps=weight_bytes / med / 1e6,
                        tokens_per_s_weights_only=1e3 / med)), flush=True)
            del arms, dev
    finally:
        gc.collect()
        witness("post", wait_s=5)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("compile", "oracle", "run", "layout"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--exact-only", action="store_true", help="run: the exactness checks, no timing")
    a = ap.parse_args()
    if a.mode == "compile":
        compile_all()
    elif a.mode == "oracle":
        BUILD.mkdir(parents=True, exist_ok=True)
        compute_oracle(a.seed)
    elif a.mode == "layout":
        print(json.dumps(dict(layer_words=LAYER_WORDS, arena_words=ARENA_WORDS, arena_mb=ARENA_WORDS * 4 / 1e6,
                              x_slots=X_SLOTS, out=OUT, out_words=OUT_WORDS), indent=1))
    else:
        return run(a.seeds, a.exact_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
