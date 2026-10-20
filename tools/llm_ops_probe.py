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

# v2: the kernel with RoPE, SiLU and split outputs. v1's build (residual and norm only) stays in build/llm_ops.
BUILD = ROOT / "build" / "llm_ops_v2"
HIDDEN, MLP, VOCAB, KV = 576, 1536, 49152, 192
QK = HIDDEN + KV                       # rows RoPE rotates: 9 q heads and 3 k heads
POS, ROW, CACHE_ROWS = 37, 2, 4        # the position RoPE uses, and the cache row the KV append writes
POISON = 0x7FC9_8765


def bits(a):
    from ignite_xdna.compiler import engine_bf16_emulator as em
    return em.bf16_bits(np.asarray(a, np.float32))


F = pack
# name -> (N, K, flags) of each weight block in the arena
WSPEC = {"qkv": (960, HIDDEN, F.F_NORM), "o": (HIDDEN, HIDDEN, F.F_INIT), "gu": (2 * MLP, HIDDEN, F.F_NORM),
         "down": (HIDDEN, MLP, F.F_INIT), "head": (VOCAB, HIDDEN, F.F_NORM),
         "qkv_r": (960, HIDDEN, F.F_NORM | F.F_ROPE), "gu_s": (MLP, HIDDEN, F.F_NORM | F.F_SILU)}
ROPE_END = {"qkv_r": QK}
WAIT_MS = 5000
ORDER = ("one", "chain_short", "chain")    # --sequence replaces this (any order, repeats allowed)
OUTPUTS = {"one": ("qkv", "o", "gu", "down", "head", "plain_o", "q1", "kc1", "vc1", "m1"),
           "chain_short": ("s_o", "s_gu", "s_down", "s_head"),
           "chain": ("c_o", "c_gu", "c_down", "c_head", "c_q", "kc2", "vc2", "c_m"),
           "rope_one": ("q1", "kc1", "vc1"), "silu_one": ("m1",),
           "head_one": ("head",), "headc": ("s_head",)}
# headc reads its x from s_down, which the host fills with the expected down output before each seed

W_OFF, _w = {}, 0
for _s, (_n, _k, _f) in WSPEC.items():
    W_OFF[_s] = _w
    _w += pack.weight_words(_n, _k, _f)
W_WORDS = _w
OUT, _o = {}, 0
for _s, _n in [("qkv", 960), ("o", 576), ("gu", 3072), ("down", 576), ("head", VOCAB), ("plain_o", 576),
               ("c_o", 576), ("c_gu", 3072), ("c_down", 576), ("c_head", VOCAB),
               ("q1", HIDDEN), ("kc1", CACHE_ROWS * KV), ("vc1", CACHE_ROWS * KV), ("m1", MLP),
               ("c_q", HIDDEN), ("kc2", CACHE_ROWS * KV), ("vc2", CACHE_ROWS * KV), ("c_m", MLP),
               ("s_o", 576), ("s_gu", 3072), ("s_down", 576), ("s_head", VOCAB)]:
    OUT[_s] = _o
    _o += _n // 2
OUT_WORDS = _o


def cs_row():
    """cos[0:32] then sin[0:32] at POS, bf16 bits: what a RoPE call's tail opens with."""
    cos, sin = R.rope_tables(POS + 1)
    return bits(np.concatenate([cos[POS, :32], sin[POS, :32]]))


def kv_map(q, kc, vc):
    """qkv_r's out_map: q to its region, k and v to cache row ROW."""
    return ((0, HIDDEN, OUT[q]), (HIDDEN, KV, OUT[kc] + ROW * KV // 2), (QK, KV, OUT[vc] + ROW * KV // 2))


def chain_tails():
    return {"gu": pack.tail(*WSPEC["gu"]), "down": pack.tail(*WSPEC["down"]), "head": pack.tail(*WSPEC["head"]),
            "qkv_r": pack.tail(*WSPEC["qkv_r"], ROPE_END["qkv_r"], cs_row()), "gu_s": pack.tail(*WSPEC["gu_s"])}


ONE = ["qkv", "o", "gu", "down", "head", "plain_o", "qkv_r", "gu_s"]
X_SLOT = {s: i * pack.X_SLOT_WORDS for i, s in enumerate(ONE)}
X_TAIL, _x = {}, len(ONE) * pack.X_SLOT_WORDS
for _s, _t in chain_tails().items():
    X_TAIL[_s] = _x
    _x += _t.size
X_WORDS = _x


def weights():
    """name -> float64 [N][K] bf16 values: layer 0's projections with the norms' gamma folded in, and the head."""
    w = R.Weights.load(R.safetensors(R.model_file()))
    L = w.layers[0]
    return {"qkv": L["qkv_g"], "o": L["o"], "gu": L["gu_g"], "down": L["down"], "head": w.head_g}


def packed(w, name):
    """The arena block of one WSPEC entry."""
    if name == "qkv_r":
        return pack.pack(bits(w["qkv"][pack.rope_perm(960, QK)]))
    if name == "gu_s":
        return pack.pack_silu(bits(w["gu"][:MLP]), bits(w["gu"][MLP:]))
    return pack.pack(bits(w[name]))


def inputs(seed):
    rng = np.random.default_rng(seed)
    scale = 1.0 + 29.0 * seed
    f = lambda n, s: R.bf16(rng.normal(0, s, n))  # noqa: E731
    return dict(h=f(HIDDEN, scale), attn=f(HIDDEN, 1.0), m=f(MLP, 1.0), h2=f(HIDDEN, scale))


def programs():
    from kernels.llm_gemv.design import Call
    one = []
    for s in ["qkv", "o", "gu", "down", "head"]:
        n, k, fl = WSPEC[s]
        one.append(Call(W_OFF[s], X_SLOT[s], OUT[s], n, k, flags=fl))
    one.append(Call(W_OFF["o"], X_SLOT["plain_o"], OUT["plain_o"], HIDDEN, HIDDEN))
    one.append(Call(W_OFF["qkv_r"], X_SLOT["qkv_r"], 0, *WSPEC["qkv_r"][:2], flags=WSPEC["qkv_r"][2],
                    rope_end=QK, out_map=kv_map("q1", "kc1", "vc1")))
    one.append(Call(W_OFF["gu_s"], X_SLOT["gu_s"], OUT["m1"], *WSPEC["gu_s"][:2], flags=WSPEC["gu_s"][2]))
    chain = [Call(W_OFF["o"], X_SLOT["o"], OUT["c_o"], HIDDEN, HIDDEN, flags=F.F_INIT),
             Call(W_OFF["gu"], X_TAIL["gu"], OUT["c_gu"], *WSPEC["gu"][:2], x_out=OUT["c_o"], flags=F.F_NORM),
             Call(W_OFF["down"], X_TAIL["down"], OUT["c_down"], *WSPEC["down"][:2], x_out=OUT["c_gu"],
                  flags=F.F_INIT, h_out=OUT["c_o"]),
             Call(W_OFF["head"], X_TAIL["head"], OUT["c_head"], *WSPEC["head"][:2], x_out=OUT["c_down"],
                  flags=F.F_NORM),
             Call(W_OFF["qkv_r"], X_TAIL["qkv_r"], 0, *WSPEC["qkv_r"][:2], x_out=OUT["c_down"],
                  flags=WSPEC["qkv_r"][2], rope_end=QK, out_map=kv_map("c_q", "kc2", "vc2")),
             Call(W_OFF["gu_s"], X_TAIL["gu_s"], OUT["c_m"], *WSPEC["gu_s"][:2], x_out=OUT["c_o"],
                  flags=WSPEC["gu_s"][2])]
    tails = chain_tails()
    for name, c in zip(["gu", "down", "head", "qkv_r", "gu_s"], chain[1:]):
        cs = cs_row() if c.flags & F.F_ROPE else None
        if not np.array_equal(c.header(cs), tails[name]) or c.x_offset != X_TAIL[name]:
            raise AssertionError(f"{name}: the chained call's tail is not the one the x buffer holds")
    # chain_short: v1's chain (o+res, gu+norm, down+res, head+norm) on this kernel, outputs to their own regions
    short = [Call(c.w_offset, c.x_offset, OUT["s" + {OUT["c_o"]: "_o", OUT["c_gu"]: "_gu", OUT["c_down"]: "_down",
                                                     OUT["c_head"]: "_head"}[c.out_offset]], c.n, c.k,
                  x_out=None if c.x_out is None else OUT["s" + {OUT["c_o"]: "_o", OUT["c_gu"]: "_gu",
                                                                OUT["c_down"]: "_down"}[c.x_out]],
                  flags=c.flags, h_out=None if c.h_out is None else OUT["s_o"]) for c in chain[:4]]
    return {"one": one, "chain_short": short, "chain": chain, "rope_one": [one[6]], "silu_one": [one[7]],
            "head_one": [one[4]], "headc": [short[3]]}


def _qkv_rope(w, h):
    """The device's qkv_r output in its row order: norm, then RoPE on q and k (exact), rows by rope_perm."""
    cos, sin = R.rope_tables(POS + 1)
    y = R.normed_matvec(w["qkv"], h, True)
    q = R.rope(y[:HIDDEN].reshape(9, 64), cos[POS], sin[POS], True).reshape(-1)
    k = R.rope(y[HIDDEN:QK].reshape(3, 64), cos[POS], sin[POS], True).reshape(-1)
    return np.concatenate([q, k, y[QK:]])[pack.rope_perm(960, QK)]


def _cache(row_bits):
    """A cache region as the device should leave it: poison, with ROW written."""
    region = np.full(CACHE_ROWS * KV // 2, POISON, np.uint32).view(np.uint16).copy()
    region[ROW * KV:(ROW + 1) * KV] = row_bits
    return region


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
    o.update(s_o=c_o, s_gu=c_gu, s_down=c_down, s_head=o["c_head"])     # chain_short computes the same values
    out = {k: bits(v) for k, v in o.items()}
    for tag, src in (("1", h), ("2", f32(c_down))):
        y = bits(_qkv_rope(w, src))
        out["q1" if tag == "1" else "c_q"] = y[:HIDDEN]
        out["kc" + tag], out["vc" + tag] = _cache(y[HIDDEN:QK]), _cache(y[QK:])
    for name, src in (("m1", h2), ("c_m", f32(c_o))):
        gu = R.normed_matvec(w["gu"], src, True)
        out[name] = bits(R.bf16(R.dev(R.silu_bits, gu[:MLP]) * gu[MLP:]))
    return out


def x_buffer(x):
    h, attn, m, h2 = (bits(x[k]) for k in ("h", "attn", "m", "h2"))
    slots = {"qkv": pack.x_slot(h, 960, None, F.F_NORM), "o": pack.x_slot(attn, 576, h, F.F_INIT),
             "gu": pack.x_slot(h2, 3072, None, F.F_NORM), "down": pack.x_slot(m, 576, h2, F.F_INIT),
             "head": pack.x_slot(h, VOCAB, None, F.F_NORM), "plain_o": pack.x_slot(attn, 576),
             "qkv_r": pack.x_slot(h, 960, None, WSPEC["qkv_r"][2], QK, cs_row()),
             "gu_s": pack.x_slot(h2, MLP, None, WSPEC["gu_s"][2])}
    v = np.concatenate([slots[s] for s in ONE] + list(chain_tails().values()))
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
            for s in WSPEC:
                view[W_OFF[s]:W_OFF[s] + pack.weight_words(*WSPEC[s])] = packed(w, s)
            arena.sync(to)
            xb, ob = h.create_host_bo(X_WORDS * 4, 4), h.create_host_bo(OUT_WORDS * 4, 5)
            insts = {label: h.create_instruction_bo((BUILD / label / "insts.bin").as_posix()) for label in programs()}
            for seed in seeds:
                x, ref = refs[seed]
                xb.write(x_buffer(x).tobytes(), 0)
                xb.sync(to)
                ob.write(np.full(OUT_WORDS, POISON, np.uint32).tobytes(), 0)
                if "headc" in ORDER:      # headc's x: the down output, as the device computes it (oracle)
                    ob.write(ref["s_down"].view(np.uint32).tobytes(), OUT["s_down"] * 4)
                ob.sync(to)
                # every step re-poisons its own outputs, runs, and is checked at once, so a step that goes wrong
                # after an earlier one names which earlier calls it followed in this context
                for step, label in enumerate(ORDER):
                    for name in OUTPUTS[label]:
                        if name in ref:
                            ob.write(np.full(ref[name].size // 2, POISON, np.uint32).tobytes(), OUT[name] * 4)
                    ob.sync(to)
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
                    for name in OUTPUTS[label]:
                        want = ref[name]
                        got = words[OUT[name]:OUT[name] + want.size // 2].view(np.uint16)
                        bad[name] = int(np.count_nonzero(got != want))
                        if bad[name]:     # where, and whether the device left poison (never written) or a value
                            idx = np.flatnonzero(got != want)
                            pw = got.view(np.uint32) == POISON
                            print("MISMATCH_DETAIL " + json.dumps(dict(
                                seed=seed, step=step, program=label, name=name, first=int(idx[0]), last=int(idx[-1]),
                                count=int(idx.size), quarters=np.bincount(idx * 4 // want.size, minlength=4).tolist(),
                                poison_words=int(pw.sum()), sample_got=got[idx[:4]].tolist(),
                                sample_want=want[idx[:4]].tolist(),
                                equals_other={o: int(np.count_nonzero(got[idx[:64]] == ref[o][idx[:64]]))
                                              for o in ref if o != name and ref[o].size == want.size},
                                got_in_other={o: bool(np.isin(got[idx[:8]], ref[o]).all())
                                              for o in ref if o != name})), flush=True)
                    total += sum(bad.values())
                    print("EXACT_JSON " + json.dumps(dict(
                        seed=seed, step=step, program=label, checked=len(bad),
                        elements=int(sum(ref[n].size for n in OUTPUTS[label])),
                        mismatched_elements=sum(bad.values()), failures={k: v for k, v in bad.items() if v})),
                        flush=True)
            # the weights the device read: unchanged in DDR after every step?
            arena.sync(fr)
            back = np.frombuffer(arena.map(), np.uint32)
            for s_ in WSPEC:
                want_w = packed(w, s_)
                diff = int(np.count_nonzero(back[W_OFF[s_]:W_OFF[s_] + want_w.size] != want_w))
                print("ARENA_JSON " + json.dumps(dict(block=s_, words=int(want_w.size), changed_words=diff)),
                      flush=True)
                total += diff
            back = None
            arena = xb = ob = insts = view = None
    finally:
        gc.collect()
        witness("post_in_process", wait_s=5)   # informational; the wrapper's post witness decides
    print("VERDICT", "EXACT" if total == 0 else f"MISMATCH {total}", flush=True)
    return 0 if total == 0 else 1


def main():
    global ORDER
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("compile")
    r = sub.add_parser("run")
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    r.add_argument("--sequence", nargs="+", choices=tuple(OUTPUTS),
                   help="the programs to run per seed, in this order in one context (repeats allowed); each is "
                        "checked right after it runs")
    a = ap.parse_args()
    if a.cmd == "run" and a.sequence:
        ORDER = tuple(a.sequence)
    if a.cmd == "compile":
        compile_all()
        return 0
    return run(a.seeds)


if __name__ == "__main__":
    sys.exit(main())
