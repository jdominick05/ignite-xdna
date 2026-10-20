#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small-LLM decode, step 2c: one whole decoder layer on the device, exact against the reference.

SmolLM2-135M's layer 0 as one dispatch of six chained calls, every input after the first DMA'd from an earlier
call's output in DDR:
    qkv_r  RMSNorm + q/k/v + RoPE; q to its region, k and v appended at cache row p (the KV append, patched per p)
    score  F_SCORE over the K cache            pv    F_PV over the V cache: attention, in attn_out_perm order
    o      o-projection + residual (W_o's columns in attn_out_perm order)
    gu_s   RMSNorm + gate/up + SiLU            down  down-projection + residual: the layer's output h
The host writes h (the token's embedding), the position's cos/sin row and p in the tails, and patches the KV append's
two destinations; the cache itself is the device's from one position to the next.

    walk   positions 0..P-1 in order from an empty cache, one random token each: every intermediate checked
    jump   the cache's rows 0..p-1 written by the host (random), then one step at p, for long positions

    python tools/llm_layer_probe.py compile
    python tools/llm_layer_probe.py run --walk 40 --jump 150 575    # NPU, through research-lowlevel.sh --npu

Exactness only: nothing here is timed.
"""
import argparse
import dataclasses
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

BUILD = ROOT / "build" / "llm_layer"
CAP = 576
HIDDEN, MLP, KV, QK = 576, 1536, 192, 768
POISON = 0x7FC9_8765
WAIT_MS = 5000
F = pack
WSPEC = {"qkv_r": (960, HIDDEN, F.F_NORM | F.F_ROPE), "o": (HIDDEN, HIDDEN, F.F_INIT),
         "gu_s": (MLP, HIDDEN, F.F_NORM | F.F_SILU), "down": (HIDDEN, MLP, F.F_INIT)}
W_OFF, _w = {}, 0
for _s, (_n, _k, _f) in WSPEC.items():
    W_OFF[_s] = _w
    _w += pack.weight_words(_n, _k, _f)
W_WORDS = _w
OUT, _o = {}, 0
for _s, _n in [("h0", HIDDEN), ("q", HIDDEN), ("kc", CAP * KV), ("vc", CAP * KV),
               ("s", pack.KV_COLS * CAP * pack.GQA), ("a", HIDDEN), ("h1", HIDDEN), ("m", MLP), ("h2", HIDDEN)]:
    OUT[_s] = _o
    _o += _n // 2
OUT_WORDS = _o
CHECKED = ("q", "krow", "vrow", "a", "h1", "m", "h2")


def bits(a):
    from ignite_xdna.compiler import engine_bf16_emulator as em
    return em.bf16_bits(np.asarray(a, np.float32))


def program():
    from kernels.llm_gemv.design import Attn, Call
    kv0 = ((0, HIDDEN, OUT["q"]), (HIDDEN, KV, OUT["kc"]), (QK, KV, OUT["vc"]))   # row 0; patched per position
    calls = [Call(W_OFF["qkv_r"], 0, 0, 960, HIDDEN, x_out=OUT["h0"], flags=WSPEC["qkv_r"][2], rope_end=QK,
                  out_map=kv0),
             Attn(F.F_SCORE, CAP, OUT["q"], 0, OUT["kc"], OUT["s"]),
             Attn(F.F_PV, CAP, OUT["s"], 0, OUT["vc"], OUT["a"]),
             Call(W_OFF["o"], 0, OUT["h1"], HIDDEN, HIDDEN, x_out=OUT["a"], flags=F.F_INIT, h_out=OUT["h0"]),
             Call(W_OFF["gu_s"], 0, OUT["m"], MLP, HIDDEN, x_out=OUT["h1"], flags=WSPEC["gu_s"][2]),
             Call(W_OFF["down"], 0, OUT["h2"], HIDDEN, MLP, x_out=OUT["m"], flags=F.F_INIT, h_out=OUT["h1"])]
    # every call's tail at its own x offset, in program order
    out, at = [], 0
    for c in calls:
        size = (pack.OBJ_WORDS - c.lead_words()) if isinstance(c, Attn) else c.header().size
        out.append(dataclasses.replace(c, x_offset=at))
        at += size
    return out, at


CALLS, X_WORDS = program()


def x_arg(p):
    """Every call's tail for position p: cos/sin row p in qkv_r's, p in pv's."""
    from kernels.llm_gemv.design import Attn
    cos, sin = R.rope_tables(p + 1)
    cs = bits(np.concatenate([cos[p, :32], sin[p, :32]]))
    parts = []
    for c in CALLS:
        if isinstance(c, Attn):
            parts.append(pack.attn_tail(c.flags, CAP, p if c.flags == F.F_PV else 0))
        else:
            parts.append(c.header(cs if c.flags & F.F_ROPE else None))
    x = np.concatenate(parts)
    assert x.size == X_WORDS
    return x


def weights():
    w = R.Weights.load(R.safetensors(R.model_file()))
    return w


def arena(w):
    L = w.layers[0]
    blocks = {"qkv_r": pack.pack(bits(L["qkv_g"][pack.rope_perm(960, QK)])),
              "o": pack.pack(bits(L["o"][:, pack.attn_out_perm()])),
              "gu_s": pack.pack_silu(bits(L["gu_g"][:MLP]), bits(L["gu_g"][MLP:])),
              "down": pack.pack(bits(L["down"]))}
    out = np.zeros(W_WORDS, np.uint32)
    for s, b in blocks.items():
        out[W_OFF[s]:W_OFF[s] + b.size] = b
    return out


def expected(t):
    """The device's view of a reference layer trace: q and k in rope_perm order within a head, attn in its order."""
    dev = R.DEV_DIMS
    return dict(q=bits(np.asarray(t["q"])[:, dev].reshape(-1)), krow=bits(np.asarray(t["k"])[:, dev].reshape(-1)),
                vrow=bits(np.asarray(t["v"]).reshape(-1)), a=bits(np.asarray(t["attn"])[pack.attn_out_perm()]),
                h1=bits(t["h1"]), m=bits(t["m"]), h2=bits(t["h2"]))


def got_of(words, p):
    u = words.view(np.uint16)
    row = lambda base: u[2 * base + p * KV:2 * base + (p + 1) * KV]  # noqa: E731
    span = lambda n, size: u[2 * OUT[n]:2 * OUT[n] + size]           # noqa: E731
    return dict(q=span("q", HIDDEN), krow=row(OUT["kc"]), vrow=row(OUT["vc"]), a=span("a", HIDDEN),
                h1=span("h1", HIDDEN), m=span("m", MLP), h2=span("h2", HIDDEN))


def cases(w, walk, jumps, seed):
    """[(label, p, h0 bits, host cache rows or None, expected)] with the reference stepping alongside."""
    rng = np.random.default_rng(seed)
    out = []
    dec = R.Decoder(w, exact=True)
    for p in range(walk):
        tok = int(rng.integers(0, R.VOCAB))
        h0 = R.bf16(w.embed[tok])
        t = {}
        dec.layer(0, h0, p, trace=t)
        out.append((f"walk tok {tok}", p, bits(h0), None, expected(t)))
    for p in jumps:
        dec = R.Decoder(w, exact=True)
        Kr = R.bf16(rng.normal(0, 3, (p, 3, 64)).astype(np.float32))
        Vr = R.bf16(rng.normal(0, 1, (p, 3, 64)).astype(np.float32))
        dec.k[0], dec.v[0] = list(Kr), list(Vr)
        tok = int(rng.integers(0, R.VOCAB))
        h0 = R.bf16(w.embed[tok])
        t = {}
        dec.layer(0, h0, p, trace=t)
        rows = (bits(Kr[:, :, R.DEV_DIMS]).reshape(p, -1), bits(Vr).reshape(p, -1))
        out.append((f"jump tok {tok}", p, bits(h0), rows, expected(t)))
    return out


def insts_for(raw, p):
    from kernels.llm_gemv import patch
    sites = patch.calls(raw, CALLS)
    c0 = CALLS[0]
    changes = {**patch.move_piece(sites[0], c0, 1, OUT["kc"] + p * KV // 2),
               **patch.move_piece(sites[0], c0, 2, OUT["vc"] + p * KV // 2)}
    return patch.apply(raw, changes)


def compile_all():
    from kernels.llm_gemv import design
    d = BUILD / "layer"
    if not (d / "build.json").exists():
        design.build(d, CALLS, W_WORDS, X_WORDS, OUT_WORDS, "layer")
    meta = json.loads((d / "build.json").read_text())
    raw = (d / "insts.bin").read_bytes()
    for p in (0, 1, CAP - 1):              # the patch finds every transfer and moves only the KV append
        insts_for(raw, p)
    print("BUILD layer " + json.dumps(dict(cores=meta["cores"][0], tokens=meta["tokens"], calls=len(CALLS))),
          flush=True)


def run(walk, jumps, seed):
    import gc
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    from llm_gemv_probe import witness
    w = weights()
    todo = cases(w, walk, jumps, seed)
    print(f"ORACLE {len(todo)} cases ready", flush=True)
    raw = (BUILD / "layer" / "insts.bin").read_bytes()
    if not witness("pre"):
        raise SystemExit("NPU is busy; did not dispatch")
    total = 0
    try:
        with XrtSiliconHarness(0) as h:
            h.load_xclbin(str(BUILD / "layer" / "final.xclbin"))
            xrt = h.pyxrt
            to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            fr = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            wb = h.create_host_bo(W_WORDS * 4, 3)
            np.frombuffer(wb.map(), np.uint32)[:] = arena(w)
            wb.sync(to)
            xb, ob = h.create_host_bo(X_WORDS * 4, 4), h.create_host_bo(OUT_WORDS * 4, 5)
            ob.write(np.full(OUT_WORDS, POISON, np.uint32).tobytes(), 0)     # the cache starts as poison
            ob.sync(to)
            for label, p, h0, rows, want in todo:
                xb.write(x_arg(p).tobytes(), 0)
                xb.sync(to)
                words = np.frombuffer(ob.map(), np.uint32)
                words[OUT["h0"]:OUT["h0"] + HIDDEN // 2] = h0.view(np.uint32)
                if rows is not None:              # a jump: the host's cache rows, poison past them
                    for name, r in zip(("kc", "vc"), rows):
                        region = words[OUT[name]:OUT[name] + CAP * KV // 2]
                        region[:] = POISON
                        region[:p * KV // 2] = r.reshape(-1).view(np.uint32)
                for name in ("q", "s", "a", "h1", "m", "h2"):
                    words[OUT[name]:OUT[name] + {"s": pack.KV_COLS * CAP * pack.GQA // 2, "m": MLP // 2}.get(
                        name, HIDDEN // 2)] = POISON
                words = None
                ob.sync(to)
                bo, count = h.create_instruction_bo_from_bytes(insts_for(raw, p))
                r = xrt.run(h.kernel)
                for i, v in enumerate((3, bo, count, wb, xb, ob)):
                    r.set_arg(i, v)
                r.start()
                state = r.wait(WAIT_MS)
                if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    raise RuntimeError(f"{label} p {p} did not complete: {state}; stopping without retry")
                r = bo = None
                ob.sync(fr)
                got = got_of(np.frombuffer(ob.read(OUT_WORDS * 4, 0), np.uint32), p)
                bad = {k: int(np.count_nonzero(got[k] != want[k])) for k in CHECKED}
                for k in CHECKED:
                    if bad[k]:
                        idx = np.flatnonzero(got[k] != want[k])
                        print("MISMATCH_DETAIL " + json.dumps(dict(
                            case=label, p=p, name=k, count=int(idx.size), first=int(idx[0]),
                            poison=int(np.count_nonzero(got[k] == (POISON & 0xFFFF))),
                            sample_got=got[k][idx[:4]].tolist(), sample_want=want[k][idx[:4]].tolist())), flush=True)
                total += sum(bad.values())
                print("EXACT_JSON " + json.dumps(dict(case=label, p=p, elements=int(sum(want[k].size for k in CHECKED)),
                                                      mismatched=sum(bad.values()),
                                                      failures={k: v for k, v in bad.items() if v})), flush=True)
            wb = xb = ob = None
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
    r.add_argument("--walk", type=int, default=40, help="positions 0..WALK-1 in order")
    r.add_argument("--jump", type=int, nargs="*", default=[150, CAP - 1])
    r.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "compile":
        compile_all()
        return 0
    if not 0 <= a.walk <= CAP or any(not 0 < p < CAP for p in a.jump):
        raise SystemExit(f"positions are 0..{CAP - 1}")
    return run(a.walk, a.jump, a.seed)


if __name__ == "__main__":
    sys.exit(main())
