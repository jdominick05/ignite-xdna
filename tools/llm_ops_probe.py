#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small-LLM decode, step 2: the projections with RMSNorm and the residual add fused in, exact on the device.

The decoder folds two ops into the GEMV (kernels/llm_gemv/reference.py, fused semantics): RMSNorm, with gamma in the
weights and r = RSQ[bf16(sum x*x)] applied in the epilogue (header flag F_NORM), and the residual add, with h as the
accumulator's starting value (F_INIT). This checks both on SmolLM2-135M's layer 0 and its head, bit for bit against
the reference's exact mode (the exp_sum MAC), two ways:

    one    each fused call on host-built x (and h): qkv+norm, o+res, gate/up+norm, down+res, head+norm, and a plain
           o-projection (no flags, the step-1 path)
    chain  o+res on host x and h, then gate/up+norm on its output, down+res on part of that output with the first
           call's output as h, and head+norm on that: x, h and every norm input DMA'd from earlier outputs

    python tools/llm_ops_probe.py compile
    python tools/llm_ops_probe.py run --seeds 0 1           # NPU, through research-lowlevel.sh --npu --checks-only

The seeds set the inputs' scale (seed s draws h at standard deviation 1 + 29*s), so the norm's sum of squares spans
the range real decode reaches. Exactness only: nothing here is timed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from kernels.llm_gemv import pack  # noqa: E402
from kernels.llm_gemv import reference as R  # noqa: E402

BUILD = ROOT / "build" / "llm_ops"
HIDDEN, MLP, VOCAB, KV = 576, 1536, 49152, 192
SHAPES = {"qkv": (HIDDEN + 2 * KV, HIDDEN), "o": (HIDDEN, HIDDEN), "gu": (2 * MLP, HIDDEN), "down": (HIDDEN, MLP),
          "head": (VOCAB, HIDDEN)}
FLAGS = {"qkv": pack.F_NORM, "o": pack.F_INIT, "gu": pack.F_NORM, "down": pack.F_INIT, "head": pack.F_NORM}
WAIT_MS = 5000

W_OFF, _w = {}, 0
for _s in SHAPES:
    W_OFF[_s] = _w
    _w += pack.block_words(*SHAPES[_s])
W_WORDS = _w
# x: a host slot per one-shot call (the five fused and the plain o), then the chain's tails
ONE = ["qkv", "o", "gu", "down", "head", "plain_o"]
X_SLOT = {s: i * pack.X_SLOT_WORDS for i, s in enumerate(ONE)}
CHAIN_TAILS = {"gu": pack.tail(*SHAPES["gu"], pack.F_NORM), "down": pack.tail(*SHAPES["down"], pack.F_INIT),
               "head": pack.tail(*SHAPES["head"], pack.F_NORM)}
X_TAIL, _x = {}, len(ONE) * pack.X_SLOT_WORDS
for _s, _t in CHAIN_TAILS.items():
    X_TAIL[_s] = _x
    _x += _t.size
X_WORDS = _x
OUT, _o = {}, 0
for _s, _n in [("qkv", 960), ("o", 576), ("gu", 3072), ("down", 576), ("head", VOCAB), ("plain_o", 576),
               ("c_o", 576), ("c_gu", 3072), ("c_down", 576), ("c_head", VOCAB)]:
    OUT[_s] = _o
    _o += _n // 2
OUT_WORDS = _o


def weights():
    """name -> float64 [N][K] bf16 values: layer 0's projections with the norms' gamma folded in, and the head."""
    w = R.Weights.load(R.safetensors(R.model_file()))
    L = w.layers[0]
    return {"qkv": L["qkv_g"], "o": L["o"], "gu": L["gu_g"], "down": L["down"], "head": w.head_g}


def bits(a):
    from ignite_xdna.compiler import engine_bf16_emulator as em
    return em.bf16_bits(np.asarray(a, np.float32))


def inputs(seed):
    rng = np.random.default_rng(seed)
    scale = 1.0 + 29.0 * seed
    f = lambda n, s: R.bf16(rng.normal(0, s, n))  # noqa: E731
    return dict(h=f(HIDDEN, scale), attn=f(HIDDEN, 1.0), m=f(MLP, 1.0), h2=f(HIDDEN, scale))


def programs():
    from kernels.llm_gemv.design import Call
    one = []
    for s in ["qkv", "o", "gu", "down", "head"]:
        one.append(Call(W_OFF[s], X_SLOT[s], OUT[s], *SHAPES[s], flags=FLAGS[s]))
    one.append(Call(W_OFF["o"], X_SLOT["plain_o"], OUT["plain_o"], *SHAPES["o"]))
    chain = [Call(W_OFF["o"], X_SLOT["o"], OUT["c_o"], *SHAPES["o"], flags=pack.F_INIT),
             Call(W_OFF["gu"], X_TAIL["gu"], OUT["c_gu"], *SHAPES["gu"], x_out=OUT["c_o"], flags=pack.F_NORM),
             Call(W_OFF["down"], X_TAIL["down"], OUT["c_down"], *SHAPES["down"], x_out=OUT["c_gu"],
                  flags=pack.F_INIT, h_out=OUT["c_o"]),
             Call(W_OFF["head"], X_TAIL["head"], OUT["c_head"], *SHAPES["head"], x_out=OUT["c_down"],
                  flags=pack.F_NORM)]
    for c in chain[1:]:
        if not np.array_equal(c.header(), CHAIN_TAILS[{OUT["c_gu"]: "gu", OUT["c_down"]: "down",
                                                       OUT["c_head"]: "head"}[c.out_offset]]):
            raise AssertionError("a chained call's tail is not the one the x buffer holds")
    return {"one": one, "chain": chain}


def oracle(w, x):
    """Every checked output, from reference.py's exact mode."""
    f32 = lambda a: np.asarray(a, np.float32)  # noqa: E731
    h, attn, m, h2 = (f32(x[k]) for k in ("h", "attn", "m", "h2"))
    o = {"qkv": R.normed_matvec(w["qkv"], h, True), "o": R.matvec(w["o"], attn, True, init=h),
         "gu": R.normed_matvec(w["gu"], h2, True), "down": R.matvec(w["down"], m, True, init=h2),
         "head": R.normed_matvec(w["head"], h, True), "plain_o": R.matvec(w["o"], attn, True)}
    c_o = o["o"]                                                  # the chain's first call is the one-shot o+res
    c_gu = R.normed_matvec(w["gu"], f32(c_o), True)
    c_down = R.matvec(w["down"], f32(c_gu[:MLP]), True, init=f32(c_o))
    o.update(c_o=c_o, c_gu=c_gu, c_down=c_down, c_head=R.normed_matvec(w["head"], f32(c_down), True))
    return {k: bits(v) for k, v in o.items()}


def x_buffer(x):
    h, attn, m, h2 = (bits(x[k]) for k in ("h", "attn", "m", "h2"))
    slots = {"qkv": pack.x_slot(h, 960, None, pack.F_NORM), "o": pack.x_slot(attn, 576, h, pack.F_INIT),
             "gu": pack.x_slot(h2, 3072, None, pack.F_NORM), "down": pack.x_slot(m, 576, h2, pack.F_INIT),
             "head": pack.x_slot(h, VOCAB, None, pack.F_NORM), "plain_o": pack.x_slot(attn, 576)}
    v = np.concatenate([slots[s] for s in ONE] + list(CHAIN_TAILS.values()))
    assert v.size == X_WORDS
    return v


def compile_all():
    from kernels.llm_gemv import design
    from kernels.transformer_bf16.common import sha
    configs = {}
    for label, calls in programs().items():
        d = BUILD / label
        if not (d / "build.json").exists():
            design.build(d, calls, W_WORDS, X_WORDS, OUT_WORDS, label)
        prj = d / "prj"
        configs[label] = tuple(sorted((str(p.relative_to(prj)), sha(p)) for p in
                                      [*prj.glob("cdo_main/*.bin"), *prj.rglob("*core_*.elf"), prj / "kernels_main.json"]))
        print("BUILD", label, json.dumps(json.loads((d / "build.json").read_text())["cores"][0]), flush=True)
    if len(set(configs.values())) != 1:
        raise SystemExit("the programs configure the array differently; they cannot share one context")


def run(seeds):
    import gc
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    from llm_gemv_probe import witness
    w = weights()
    print("WEIGHTS " + json.dumps({k: list(v.shape) for k, v in w.items()}), flush=True)
    refs = {}
    for seed in seeds:
        refs[seed] = (inputs(seed), oracle(w, inputs(seed)))
        print(f"ORACLE seed {seed} ready", flush=True)
    if not witness("pre"):
        raise SystemExit("NPU is busy; did not dispatch")
    total = 0
    try:
        with XrtSiliconHarness(0) as h:
            h.load_xclbin(str(BUILD / "chain" / "final.xclbin"))
            xrt = h.pyxrt
            to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            fr = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            arena = h.create_host_bo(W_WORDS * 4, 3)
            view = np.frombuffer(arena.map(), np.uint32)
            for s in SHAPES:
                view[W_OFF[s]:W_OFF[s] + pack.block_words(*SHAPES[s])] = pack.pack(bits(w[s]))
            arena.sync(to)
            xb, ob = h.create_host_bo(X_WORDS * 4, 4), h.create_host_bo(OUT_WORDS * 4, 5)
            insts = {label: h.create_instruction_bo((BUILD / label / "insts.bin").as_posix()) for label in programs()}
            for seed in seeds:
                x, ref = refs[seed]
                xb.write(x_buffer(x).tobytes(), 0)
                xb.sync(to)
                ob.write(np.full(OUT_WORDS, 0x7FC9_8765, np.uint32).tobytes(), 0)
                ob.sync(to)
                for label in ("one", "chain"):
                    bo, count = insts[label]
                    r = xrt.run(h.kernel)
                    for i, v in enumerate((3, bo, count, arena, xb, ob)):
                        r.set_arg(i, v)
                    r.start()
                    state = r.wait(WAIT_MS)
                    if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                        raise RuntimeError(f"{label} did not complete: {state}; stopping without retry")
                    r = bo = None
                ob.sync(fr)
                words = np.frombuffer(ob.read(OUT_WORDS * 4, 0), np.uint32)
                bad = {}
                for name, want in ref.items():
                    got = words[OUT[name]:OUT[name] + want.size // 2].view(np.uint16)
                    bad[name] = int(np.count_nonzero(got != want))
                total += sum(bad.values())
                print("EXACT_JSON " + json.dumps(dict(
                    seed=seed, checked=len(bad), elements=int(sum(v.size for v in ref.values())),
                    mismatched_elements=sum(bad.values()), failures={k: v for k, v in bad.items() if v})), flush=True)
            arena = xb = ob = insts = view = None
    finally:
        gc.collect()
        witness("post_in_process", wait_s=5)   # informational; the wrapper's post witness decides
    print("VERDICT", "EXACT" if total == 0 else f"MISMATCH {total}", flush=True)
    return 0 if total == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("compile")
    r = sub.add_parser("run")
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    if a.cmd == "compile":
        compile_all()
        return 0
    return run(a.seeds)


if __name__ == "__main__":
    sys.exit(main())
