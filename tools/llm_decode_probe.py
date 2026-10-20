#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small-LLM decode, step 3: SmolLM2-135M-Instruct generating text with every neural op on the NPU.

One dispatch per token (kernels/llm_gemv/decode.py): 30 decoder layers and the tied head, 181 chained calls, the KV
cache kept on the device. The host embeds the token (a row copy), patches the position into the cached stream and
picks the argmax of the device's logits.

    compile   build/llm_decode/token
    reference prompt(s) through reference.Decoder in exact mode (about 3 s a token on the CPU), every step's logits
              saved to an .npz, so the device sitting itself is short
    check     the saved steps through the device, teacher-forced: every step's 49,152 logits compared bit for bit;
              then free-running greedy on the device's own argmax, compared token for token
    chat      generate from a prompt on the device and print the text as it comes (no reference)

    python tools/llm_decode_probe.py compile
    python tools/llm_decode_probe.py reference --tokens 16 --out ref.npz
    python tools/llm_decode_probe.py check ref.npz          # NPU, through research-lowlevel.sh --npu --checks-only
    python tools/llm_decode_probe.py chat "What is the capital of France?"

Any command with --w8 (first, before the command) runs the W8A16 decoder (decode.py's LLM_DECODE_W8): its own
build folder, Q8 weights in the reference and on the device.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

if "--w8" in sys.argv[1:2]:
    os.environ["LLM_DECODE_W8"] = "1"
from kernels.llm_gemv import decode as D  # noqa: E402
from kernels.llm_gemv import reference as R  # noqa: E402

BUILD = ROOT / "build" / "llm_decode" / ("token_w8" if D.W8 else "token")


def weights():
    """The decoder's weights: bf16, or under --w8 Q8 of the gamma-folded matrices (the embedding stays bf16)."""
    w = R.Weights.load(R.safetensors(R.model_file()))
    return R.quantize_weights(w, "w8a16-dev") if D.W8 else w
SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
PROMPTS = ["What is the capital of France? Answer in one sentence.", "List three uses for a paperclip."]


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(R.model_file()).parent / "tokenizer.json"))


def chat_ids(tok, prompt):
    """The model's chat template (tokenizer_config.json), with its default system message."""
    text = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n")
    return tok.encode(text, add_special_tokens=False).ids


def compile_all():
    from kernels.llm_gemv import design
    if not (BUILD / "build.json").exists():
        design.build(BUILD, D.CALLS, D.W_WORDS, D.X_WORDS, D.OUT_WORDS, BUILD.name)
    meta = json.loads((BUILD / "build.json").read_text())
    D.Patcher((BUILD / "insts.bin").read_bytes()).stream(D.CAP - 1)
    print("BUILD token " + json.dumps(dict(calls=len(D.CALLS), insts_bytes=(BUILD / "insts.bin").stat().st_size,
                                            w_bytes=D.W_WORDS * 4, out_bytes=D.OUT_WORDS * 4,
                                            cores=meta["cores"][0], tokens=meta["tokens"])), flush=True)


def reference(prompts, new_tokens, out_path):
    tok = tokenizer()
    eos = tok.token_to_id("<|im_end|>")
    w = weights()
    save = {}
    for n, prompt in enumerate(prompts):
        ids = chat_ids(tok, prompt)
        dec = R.Decoder(w, exact=True)
        t0 = time.perf_counter()
        tokens, positions, logits_all, logits, out = [], [], [], None, []
        for p, t in enumerate(ids):
            logits = dec.step(t, p)
            tokens.append(t), positions.append(p), logits_all.append(R.em.bf16_bits(np.asarray(logits, np.float32)))
        for _ in range(new_tokens):
            t = int(np.argmax(logits))
            out.append(t)
            if t == eos:
                break
            p = len(ids) + len(out) - 1
            logits = dec.step(t, p)
            tokens.append(t), positions.append(p), logits_all.append(R.em.bf16_bits(np.asarray(logits, np.float32)))
        save.update({f"{n}_ids": np.array(ids), f"{n}_out": np.array(out), f"{n}_tokens": np.array(tokens),
                     f"{n}_positions": np.array(positions), f"{n}_logits": np.stack(logits_all)})
        print("REFERENCE " + json.dumps(dict(prompt=prompt, prompt_tokens=len(ids), new_tokens=len(out),
                                             seconds=round(time.perf_counter() - t0, 1),
                                             text=tok.decode(out))), flush=True)
    np.savez(out_path, prompts=np.array(prompts), **save)
    return 0


def check(ref_path):
    import gc
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    from llm_gemv_probe import witness
    tok = tokenizer()
    eos = tok.token_to_id("<|im_end|>")
    w = weights()
    ref = np.load(ref_path)
    plans = []
    for n, prompt in enumerate(ref["prompts"].tolist()):
        if not np.array_equal(ref[f"{n}_ids"], chat_ids(tok, prompt)):
            raise SystemExit("the reference file's prompt ids differ from this tokenizer's")
        steps = list(zip(ref[f"{n}_tokens"].tolist(), ref[f"{n}_positions"].tolist(), ref[f"{n}_logits"]))
        plans.append((prompt, ref[f"{n}_ids"].tolist(), ref[f"{n}_out"].tolist(), steps))
        print("REFERENCE " + json.dumps(dict(prompt=prompt, steps=len(steps), text=tok.decode(plans[-1][2]))),
              flush=True)
    if not witness("pre"):
        raise SystemExit("NPU is busy; did not dispatch")
    total = 0
    try:
        with XrtSiliconHarness(0) as h:
            npu = D.NpuDecoder(h, BUILD, w)
            try:
                for prompt, ids, ref_out, steps in plans:
                    bad_steps, first_bad = 0, None
                    for t, p, want in steps:            # teacher-forced on the reference's tokens
                        got = npu.step(t, p)
                        n = int(np.count_nonzero(got != want))
                        if n:
                            bad_steps += 1
                            first_bad = p if first_bad is None else first_bad
                            idx = np.flatnonzero(got != want)
                            print("MISMATCH_DETAIL " + json.dumps(dict(
                                prompt=prompt, p=p, count=n, first=int(idx[0]), sample_got=got[idx[:4]].tolist(),
                                sample_want=want[idx[:4]].tolist())), flush=True)
                        total += n
                    # and free-running on the device's own argmax
                    dev_out = npu.generate(ids, len(ref_out), eos)
                    print("EXACT_JSON " + json.dumps(dict(
                        prompt=prompt, steps=len(steps), logits_per_step=D.VOCAB, mismatched_steps=bad_steps,
                        first_mismatch_position=first_bad, tokens_equal=dev_out == ref_out,
                        device_text=tok.decode(dev_out))), flush=True)
                    total += 0 if dev_out == ref_out else 1
            finally:
                npu.close()
                npu = None
    finally:
        gc.collect()
        witness("post_in_process", wait_s=5)
    print("VERDICT", "EXACT" if total == 0 else f"MISMATCH {total}", flush=True)
    return 0 if total == 0 else 1


def bench(ref_path, rounds):
    """Timed: each saved prompt generated ``rounds`` times on the device, every step timed (host wall, and dispatch
    start to completion). Each generation must stay token-identical to the reference."""
    import gc
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    from llm_gemv_probe import witness
    tok = tokenizer()
    eos = tok.token_to_id("<|im_end|>")
    w = weights()
    ref = np.load(ref_path)
    prompts = [(ref[f"{n}_ids"].tolist(), ref[f"{n}_out"].tolist()) for n in range(len(ref["prompts"]))]
    if not witness("pre"):
        raise SystemExit("NPU is busy; did not dispatch")
    wall, device, bad = [], [], 0
    try:
        with XrtSiliconHarness(0) as h:
            npu = D.NpuDecoder(h, BUILD, w)
            try:
                npu.generate(prompts[0][0][:4], 1)                 # warm: first dispatch, caches, page-in
                for rnd in range(rounds):
                    for ids, want in prompts:
                        times = []
                        step = npu.step

                        def timed(t, p):
                            t0 = time.perf_counter()
                            out = step(t, p)
                            times.append((time.perf_counter() - t0, npu.device_s))
                            return out
                        npu.step = timed
                        got = npu.generate(ids, len(want), eos)
                        npu.step = step
                        bad += got != want
                        wall += [a for a, _ in times]
                        device += [b for _, b in times]
                        print("ROUND " + json.dumps(dict(round=rnd, prompt_tokens=len(ids), new_tokens=len(got),
                                                         tokens_equal=got == want,
                                                         median_ms=round(1e3 * float(np.median([a for a, _ in times])), 3))),
                              flush=True)
            finally:
                npu.close()
                npu = None
    finally:
        gc.collect()
        witness("post_in_process", wait_s=5)
    wall, device = np.array(wall) * 1e3, np.array(device) * 1e3
    print("TIMING_JSON " + json.dumps(dict(
        steps=int(wall.size), ms_per_token_median=round(float(np.median(wall)), 3),
        ms_per_token_p90=round(float(np.percentile(wall, 90)), 3),
        tokens_per_second_median=round(1e3 / float(np.median(wall)), 2),
        dispatch_ms_median=round(float(np.median(device)), 3),
        host_ms_median=round(float(np.median(wall - device)), 3))), flush=True)
    print("VERDICT", "EXACT" if bad == 0 else f"MISMATCH {bad}", flush=True)
    return 0 if bad == 0 else 1


def chat(prompt, new_tokens):
    import gc
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    tok = tokenizer()
    eos = tok.token_to_id("<|im_end|>")
    w = weights()
    ids = chat_ids(tok, prompt)
    try:
        with XrtSiliconHarness(0) as h:
            npu = D.NpuDecoder(h, BUILD, w)
            try:
                seen = []

                def show(t):
                    seen.append(t)
                    if t != eos:
                        sys.stdout.write(tok.decode([t]))
                        sys.stdout.flush()
                t0 = time.perf_counter()
                out = npu.generate(ids, new_tokens, eos, on_token=show)
                secs = time.perf_counter() - t0
                print()
                print("CHAT " + json.dumps(dict(prompt_tokens=len(ids), new_tokens=len(out),
                                                seconds_including_prompt=round(secs, 2))), flush=True)
            finally:
                npu.close()
                npu = None
    finally:
        gc.collect()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w8", action="store_true", help="the W8A16 decoder (must come first)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("compile")
    r = sub.add_parser("reference")
    r.add_argument("--tokens", type=int, default=16)
    r.add_argument("--prompt", nargs="*", default=PROMPTS)
    r.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("reference_file")
    b = sub.add_parser("bench")
    b.add_argument("reference_file")
    b.add_argument("--rounds", type=int, default=3)
    g = sub.add_parser("chat")
    g.add_argument("prompt")
    g.add_argument("--tokens", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "compile":
        compile_all()
        return 0
    if a.cmd == "reference":
        return reference(a.prompt, a.tokens, a.out)
    if a.cmd == "check":
        return check(a.reference_file)
    if a.cmd == "bench":
        return bench(a.reference_file, a.rounds)
    return chat(a.prompt, a.tokens)


if __name__ == "__main__":
    sys.exit(main())
