#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small-LLM decode, step 2b: attention on the device, exact against the reference.

Two calls of the whole-array kernel (kernels/llm_gemv, header flags F_SCORE and F_PV), columns 0-2 each one KV head
and its three q heads: F_SCORE streams the K cache and writes bf16(K q) per position and head; F_PV reads those
scores and the V cache and writes softmax(scores / 8) V for positions <= p (reference.attention_group, bit for bit).
Every stream covers CAP cached positions whatever p is; only the header's p changes.

Per seed and per p, the host writes q (9 heads, rope_perm order within a head, as the RoPE projection leaves it),
the K and V caches and the F_PV tail, poisons both outputs, dispatches once, and checks:
    scores  every position's score where K is finite past p (even seeds), positions <= p where it is poison (odd)
    out     all 576 outputs; V past p is always poison, so a masking slip shows as NaN

    python tools/llm_attn_probe.py compile
    python tools/llm_attn_probe.py run --seeds 0 1 --positions 0 1 5 37 150 575   # NPU, research-lowlevel.sh --npu

Exactness only: nothing here is timed.
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

BUILD = ROOT / "build" / "llm_attn"
CAP = 576
POISON = 0x7FC9_8765
WAIT_MS = 5000
HEADS, HD, G = 9, 64, pack.GQA
# out argument (words): q, K cache, V cache, scores, attention output
Q_OFF = 0
K_OFF = Q_OFF + HEADS * HD // 2
V_OFF = K_OFF + CAP * pack.CACHE_WORDS
S_OFF = V_OFF + CAP * pack.CACHE_WORDS
S_WORDS = pack.KV_COLS * CAP * G // 2
A_OFF = S_OFF + S_WORDS
OUT_WORDS = A_OFF + HEADS * HD // 2
# x argument: the two tails
X_SCORE = 0
X_PV = X_SCORE + pack.OBJ_WORDS - pack.attn_lead_words(pack.F_SCORE, CAP)
X_WORDS = X_PV + pack.OBJ_WORDS - pack.attn_lead_words(pack.F_PV, CAP)
W_WORDS = 16


def calls():
    from kernels.llm_gemv import design
    return [design.Attn(pack.F_SCORE, CAP, Q_OFF, X_SCORE, K_OFF, S_OFF),
            design.Attn(pack.F_PV, CAP, S_OFF, X_PV, V_OFF, A_OFF)]


def bits(a):
    from ignite_xdna.compiler import engine_bf16_emulator as em
    return em.bf16_bits(np.asarray(a, np.float32))


def inputs(seed, p):
    """q [9][64], K and V [CAP][3][64] as bf16 values (natural dims); K past p finite on even seeds."""
    rng = np.random.default_rng(1000 * seed + p)
    scale = 1.0 + 2.0 * seed
    q = R.bf16(rng.normal(0, scale, (HEADS, HD)).astype(np.float32))
    K = R.bf16(rng.normal(0, scale, (CAP, pack.KV_COLS, HD)).astype(np.float32))
    V = R.bf16(rng.normal(0, 1, (CAP, pack.KV_COLS, HD)).astype(np.float32))
    return q, K, V


def out_arg(seed, p, q, K, V):
    """The out argument as the device should find it: q and K in device dim order, V past p (and K past p on odd
    seeds) poison, both outputs poison."""
    words = np.full(OUT_WORDS, POISON, np.uint32)
    dev = R.DEV_DIMS
    words[Q_OFF:K_OFF] = bits(q[:, dev]).reshape(-1).view(np.uint32)
    kc = np.full((CAP, pack.CACHE_WORDS * 2), 0, np.uint16)
    kc[:] = bits(K[:, :, dev]).reshape(CAP, -1)
    vc = bits(V).reshape(CAP, -1).copy()
    kc_w, vc_w = kc.view(np.uint32).copy(), vc.view(np.uint32).copy()
    vc_w[p + 1:] = POISON
    if seed % 2:
        kc_w[p + 1:] = POISON
    words[K_OFF:V_OFF] = kc_w.reshape(-1)
    words[V_OFF:S_OFF] = vc_w.reshape(-1)
    return words


def x_arg(p):
    return np.concatenate([pack.attn_tail(pack.F_SCORE, CAP), pack.attn_tail(pack.F_PV, CAP, p)])


def oracle(seed, p, q, K, V):
    """(scores uint16 [3 * CAP * 3] in device order with a mask of the checked ones, out uint16 [576] device order)."""
    scores = np.zeros(S_WORDS * 2, np.uint16)
    checked = np.zeros(S_WORDS * 2, bool)
    rows = CAP if seed % 2 == 0 else p + 1
    pos, head = np.meshgrid(np.arange(rows), np.arange(G), indexing="ij")
    out = np.zeros(HEADS * HD, np.float32)
    for c in range(pack.KV_COLS):
        qc = q[c * G:(c + 1) * G]
        s = bits(R.attention_scores(qc, K[:rows, c]))                    # [rows][3]
        at = c * CAP * G + pack.score_index(pos, head)
        scores[at] = s
        checked[at] = True
        out[c * G * HD:(c + 1) * G * HD] = R.attention_group(qc, K[:, c], V[:, c], p).reshape(-1)
    return scores, checked, bits(out[pack.attn_out_perm()])


def compile_all():
    from kernels.llm_gemv import design
    d = BUILD / "attn"
    if not (d / "build.json").exists():
        design.build(d, calls(), W_WORDS, X_WORDS, OUT_WORDS, "attn")
    meta = json.loads((d / "build.json").read_text())
    print("BUILD attn " + json.dumps(dict(cores=meta["cores"][0], tokens=meta["tokens"])), flush=True)


def run(seeds, positions):
    import gc
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    from llm_gemv_probe import witness
    cases = []
    for seed in seeds:
        for p in positions:
            q, K, V = inputs(seed, p)
            cases.append((seed, p, out_arg(seed, p, q, K, V), x_arg(p), oracle(seed, p, q, K, V)))
    print(f"ORACLE {len(cases)} cases ready", flush=True)
    if not witness("pre"):
        raise SystemExit("NPU is busy; did not dispatch")
    total = 0
    try:
        with XrtSiliconHarness(0) as h:
            h.load_xclbin(str(BUILD / "attn" / "final.xclbin"))
            xrt = h.pyxrt
            to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            fr = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            wb = h.create_host_bo(W_WORDS * 4, 3)
            xb, ob = h.create_host_bo(X_WORDS * 4, 4), h.create_host_bo(OUT_WORDS * 4, 5)
            bo, count = h.create_instruction_bo((BUILD / "attn" / "insts.bin").as_posix())
            for seed, p, words, xw, (want_s, mask, want_o) in cases:
                xb.write(xw.tobytes(), 0)
                xb.sync(to)
                ob.write(words.tobytes(), 0)
                ob.sync(to)
                r = xrt.run(h.kernel)
                for i, v in enumerate((3, bo, count, wb, xb, ob)):
                    r.set_arg(i, v)
                r.start()
                state = r.wait(WAIT_MS)
                if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    raise RuntimeError(f"seed {seed} p {p} did not complete: {state}; stopping without retry")
                r = None
                ob.sync(fr)
                got = np.frombuffer(ob.read(OUT_WORDS * 4, 0), np.uint32)
                got_s = got[S_OFF:A_OFF].view(np.uint16)
                got_o = got[A_OFF:].view(np.uint16)
                bad_s = int(np.count_nonzero((got_s != want_s) & mask))
                bad_o = int(np.count_nonzero(got_o != want_o))
                if bad_o:
                    idx = np.flatnonzero(got_o != want_o)
                    print("MISMATCH_DETAIL " + json.dumps(dict(
                        seed=seed, p=p, name="out", count=int(idx.size), first=int(idx[0]),
                        columns=np.bincount(idx // 192, minlength=3).tolist(),
                        poison_words=int(np.count_nonzero(got[A_OFF:] == POISON)),
                        sample_got=got_o[idx[:4]].tolist(), sample_want=want_o[idx[:4]].tolist())), flush=True)
                if bad_s:
                    idx = np.flatnonzero((got_s != want_s) & mask)
                    print("MISMATCH_DETAIL " + json.dumps(dict(
                        seed=seed, p=p, name="scores", count=int(idx.size), first=int(idx[0]),
                        columns=np.bincount(idx // (CAP * G), minlength=3).tolist(),
                        sample_got=got_s[idx[:4]].tolist(), sample_want=want_s[idx[:4]].tolist())), flush=True)
                total += bad_s + bad_o
                print("EXACT_JSON " + json.dumps(dict(seed=seed, p=p, scores_checked=int(mask.sum()),
                                                      scores_mismatched=bad_s, out_checked=int(want_o.size),
                                                      out_mismatched=bad_o)), flush=True)
            wb = xb = ob = bo = None
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
    r.add_argument("--positions", type=int, nargs="+", default=[0, 1, 5, 37, 150, CAP - 1])
    a = ap.parse_args()
    if a.cmd == "compile":
        compile_all()
        return 0
    if any(not 0 <= p < CAP for p in a.positions):
        raise SystemExit(f"positions are 0..{CAP - 1}")
    return run(a.seeds, a.positions)


if __name__ == "__main__":
    sys.exit(main())
