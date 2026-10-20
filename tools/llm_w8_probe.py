#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small-LLM decode, W8A16: each fused projection with int8 weights, exact on the device.

The W8 path (pack.py's F_W8, gemv.cc's w8_chunk and w8_emit) streams int8 codes with one bf16 scale per row and
forms y = bf16(h + hi*s + lo*s) from the fp32 accumulator (reference.w8_matvec). This checks it on SmolLM2-135M's
layer 0 (Q8 of the gamma-folded weights) and the whole head, bit for bit against the reference's exact mode,
in one dispatch of seven calls:

    qkv    norm + RoPE, 960 rows padded to 1,152 (the padding to a junk region)
    o      o + residual, 576 rows on 3 columns (padded to 768: the fourth column gets no transfer)
    gu     gate/up + norm + SiLU
    down   down + residual, 576 rows on 3 columns
    head   norm, all 49,152 rows in one call (the scales run on into two extra lead objects)
    plain  the o projection with no flag
    chead  the head again, its x DMA'd from the down call's output

Every x and h is DMA'd from the out argument (W8 calls are chained), where the host writes the inputs.

    python tools/llm_w8_probe.py compile
    python tools/llm_w8_probe.py run --seeds 0 1            # NPU, through research-lowlevel.sh --npu

Seed s draws h at standard deviation 1 + 29*s. Exactness only: nothing here is timed.
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

# v2: the whole head in one call; v1 (a head quarter) stays in build/llm_w8
BUILD = ROOT / "build" / "llm_w8_v2"
HIDDEN, MLP, KV, VOCAB = 576, 1536, 192, 49152
QK = HIDDEN + KV
POS = 37
POISON = 0x7FC9_8765
WAIT_MS = 5000
F = pack
NORM_ROPE, NORM_SILU = F.F_NORM | F.F_ROPE, F.F_NORM | F.F_SILU
# name -> (padded N, K, flags, cols)
SPEC = {"qkv": (1152, HIDDEN, NORM_ROPE, 4), "o": (768, HIDDEN, F.F_INIT, 3), "gu": (MLP, HIDDEN, NORM_SILU, 4),
        "down": (768, MLP, F.F_INIT, 3), "head": (VOCAB, HIDDEN, F.F_NORM, 4), "plain": (768, HIDDEN, 0, 3)}
CHECKED = {"qkv": 960, "o": HIDDEN, "m": MLP, "down": HIDDEN, "head": VOCAB, "plain": HIDDEN, "chead": VOCAB}


def bits(a):
    return R.em.bf16_bits(np.asarray(a, np.float32))


def _offsets(sizes):
    at, out = 0, {}
    for name, words in sizes:
        out[name] = at
        at += words
    return out, at


OUT, OUT_WORDS = _offsets([("in_h", HIDDEN // 2), ("in_attn", HIDDEN // 2), ("in_h2", HIDDEN // 2),
                           ("in_m", MLP // 2), ("qkv", 960 // 2), ("junk", 192 // 2), ("o", HIDDEN // 2),
                           ("m", MLP // 2), ("down", HIDDEN // 2), ("head", VOCAB // 2), ("plain", HIDDEN // 2),
                           ("chead", VOCAB // 2)])
X_CS, X_WORDS = 0, F.CS_ELEMS // 2


def _call(name, w_off, out_offset, x_out, **kw):
    from kernels.llm_gemv.design import Call
    n, k, flags, cols = SPEC[name]
    return Call(w_off, X_CS, out_offset, n, k, x_out=x_out, flags=flags, w8=True, cols=cols, **kw)


def _w_layout():
    sizes = [(name, _call(name, 0, 0, 0, **_extra(name)).weight_words()) for name in SPEC]
    return _offsets(sizes)


def _extra(name):
    if name == "qkv":
        return dict(rope_end=QK, out_map=((0, 960, OUT["qkv"]), (960, 192, OUT["junk"])))
    if name == "o":
        return dict(h_out=OUT["in_h"])
    if name == "down":
        return dict(h_out=OUT["in_h2"])
    return {}


W_OFF, W_WORDS = _w_layout()


def program():
    x_of = {"qkv": "in_h", "o": "in_attn", "gu": "in_h2", "down": "in_m", "head": "in_h", "plain": "in_attn"}
    out_of = {"qkv": 0, "o": OUT["o"], "gu": OUT["m"], "down": OUT["down"], "head": OUT["head"],
              "plain": OUT["plain"]}
    calls = [_call(name, W_OFF[name], out_of[name], OUT[x_of[name]], **_extra(name)) for name in SPEC]
    calls.append(_call("head", W_OFF["head"], OUT["chead"], OUT["down"]))
    return calls


def cs_row():
    cos, sin = R.rope_tables(POS + 1)
    return bits(np.concatenate([cos[POS, :32], sin[POS, :32]]))


def quantized():
    """name -> Q8 of layer 0's gamma-folded projections (natural row order) and the head."""
    w = R.Weights.load(R.safetensors(R.model_file()))
    L = w.layers[0]
    return {"qkv": R.Q8.of(L["qkv_g"]), "o": R.Q8.of(L["o"]), "gate": R.Q8.of(L["gu_g"][:MLP]),
            "up": R.Q8.of(L["gu_g"][MLP:]), "down": R.Q8.of(L["down"]), "head": R.Q8.of(w.head_g)}


def _padded(q, n):
    codes = np.zeros((n, q.codes.shape[1]), np.int8)
    codes[:q.codes.shape[0]] = q.codes
    s = np.zeros(n, np.uint16)
    s[:q.s.shape[0]] = bits(q.s)
    return codes, s


def arena(q):
    out = np.zeros(W_WORDS, np.uint32)
    perm = pack.rope_perm(960, QK)
    src = {"qkv": [R.Q8(q["qkv"].codes[perm], q["qkv"].s[perm])], "o": [q["o"]], "gu": [q["gate"], q["up"]],
           "down": [q["down"]], "head": [q["head"]], "plain": [q["o"]]}
    for name, (n, k, flags, cols) in SPEC.items():
        mats = [_padded(m, n) for m in src[name]]
        block = pack.pack8(*(c for c, _ in mats), cols=cols)
        tails = pack.tails8(n, k, flags, *(s for _, s in mats), rope_end=QK if flags & F.F_ROPE else 0, cols=cols)
        both = np.concatenate([block, tails])
        call = _call(name, 0, 0, 0, **_extra(name))
        assert both.size == call.weight_words()
        out[W_OFF[name]:W_OFF[name] + both.size] = both
    return out


def inputs(seed):
    rng = np.random.default_rng(seed)
    scale = 1.0 + 29.0 * seed
    f = lambda n, s: R.bf16(rng.normal(0, s, n))  # noqa: E731
    return dict(h=f(HIDDEN, scale), attn=f(HIDDEN, 1.0), h2=f(HIDDEN, scale), m=f(MLP, 1.0))


def oracle(q, x):
    """Every checked output (bf16 bits), from reference.py's exact mode on the Q8 weights."""
    f32 = lambda a: np.asarray(a, np.float32)  # noqa: E731
    h, attn, h2, m = (f32(x[k]) for k in ("h", "attn", "h2", "m"))
    cos, sin = R.rope_tables(POS + 1)
    y = R.normed_matvec(q["qkv"], h, True)
    qr = R.rope(y[:HIDDEN].reshape(9, 64), cos[POS], sin[POS], True).reshape(-1)
    kr = R.rope(y[HIDDEN:QK].reshape(3, 64), cos[POS], sin[POS], True).reshape(-1)
    qkv = np.concatenate([qr, kr, y[QK:]])[pack.rope_perm(960, QK)]
    gate, up = R.normed_matvec(q["gate"], h2, True), R.normed_matvec(q["up"], h2, True)
    down = R.matvec(q["down"], m, True, init=h2)
    o = {"qkv": qkv, "o": R.matvec(q["o"], attn, True, init=h), "m": R.bf16(R.dev(R.silu_bits, gate) * up),
         "down": down, "head": R.normed_matvec(q["head"], h, True), "plain": R.matvec(q["o"], attn, True),
         "chead": R.normed_matvec(q["head"], f32(down), True)}
    return {k: bits(v) for k, v in o.items()}


def out_init(x):
    v = np.full(OUT_WORDS, POISON, np.uint32)
    for name, key in (("in_h", "h"), ("in_attn", "attn"), ("in_h2", "h2"), ("in_m", "m")):
        b = bits(x[key])
        v[OUT[name]:OUT[name] + b.size // 2] = b.view(np.uint32)
    return v


def compile_all():
    from kernels.llm_gemv import design
    if not (BUILD / "build.json").exists():
        design.build(BUILD, program(), W_WORDS, X_WORDS, OUT_WORDS, "w8")
    meta = json.loads((BUILD / "build.json").read_text())
    print("BUILD w8 " + json.dumps(dict(calls=len(program()), insts_sha256=meta.get("insts_sha256"),
                                        cores=meta.get("cores", [None])[0], tokens=meta.get("tokens"))), flush=True)


def run(seeds):
    import gc
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    from llm_gemv_probe import witness
    q = quantized()
    weights = arena(q)
    refs = {s: (inputs(s), oracle(q, inputs(s))) for s in seeds}
    print("ORACLE ready " + json.dumps({k: int(v.size) for k, v in refs[seeds[0]][1].items()}), flush=True)
    if not witness("pre"):
        raise SystemExit("NPU is busy; did not dispatch")
    total = 0
    try:
        with XrtSiliconHarness(0) as h:
            h.load_xclbin(str(BUILD / "final.xclbin"))
            xrt = h.pyxrt
            to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            fr = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            ab = h.create_host_bo(W_WORDS * 4, 3)
            view = np.frombuffer(ab.map(), np.uint32)
            view[:] = weights
            ab.sync(to)
            xb, ob = h.create_host_bo(X_WORDS * 4, 4), h.create_host_bo(OUT_WORDS * 4, 5)
            xb.write(cs_row().view(np.uint32).tobytes(), 0)
            xb.sync(to)
            bo, count = h.create_instruction_bo((BUILD / "insts.bin").as_posix())
            for seed in seeds:
                x, ref = refs[seed]
                ob.write(out_init(x).tobytes(), 0)
                ob.sync(to)
                r = xrt.run(h.kernel)
                for i, v in enumerate((3, bo, count, ab, xb, ob)):
                    r.set_arg(i, v)
                r.start()
                state = r.wait(WAIT_MS)
                if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    raise RuntimeError(f"seed {seed} did not complete: {state}; stopping without retry")
                r = None
                ob.sync(fr)
                words = np.frombuffer(ob.read(OUT_WORDS * 4, 0), np.uint32)
                bad = {}
                for name, n in CHECKED.items():
                    got = words[OUT[name]:OUT[name] + n // 2].view(np.uint16)
                    want = ref[name]
                    bad[name] = int(np.count_nonzero(got != want))
                    if bad[name]:
                        idx = np.flatnonzero(got != want)
                        print("MISMATCH_DETAIL " + json.dumps(dict(
                            seed=seed, name=name, first=int(idx[0]), last=int(idx[-1]), count=int(idx.size),
                            poison_words=int(np.count_nonzero(got.view(np.uint32) == POISON)),
                            sample_got=got[idx[:4]].tolist(), sample_want=want[idx[:4]].tolist())), flush=True)
                total += sum(bad.values())
                print("EXACT_JSON " + json.dumps(dict(seed=seed, checked=len(bad), elements=sum(CHECKED.values()),
                                                      mismatched_elements=sum(bad.values()),
                                                      failures={k: v for k, v in bad.items() if v})), flush=True)
            ab.sync(fr)
            changed = int(np.count_nonzero(np.frombuffer(ab.map(), np.uint32) != weights))
            print("ARENA_JSON " + json.dumps(dict(words=int(W_WORDS), changed_words=changed)), flush=True)
            total += changed
            ab = xb = ob = bo = view = None
    finally:
        gc.collect()
        witness("post_in_process", wait_s=5)
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
