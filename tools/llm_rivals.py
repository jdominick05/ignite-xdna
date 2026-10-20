#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SmolLM2-135M-Instruct on the CPU and the iGPU, as rivals to the NPU decoder (tools/llm_decode_probe.py).

ONNX Runtime GenAI 0.11.2 (resnet_env17, the stack the Gemma study used), its own model builder on the Hub's
weights, three arms:
    cpu-fp32   CPU, fp32 weights and compute (the bf16 checkpoint upcast: no weight is changed)
    cpu-int4   CPU, int4 weights (the builder's RTN, block 32), int8 compute (accuracy_level 4, its CPU default)
    dml-fp16   DirectML on the Radeon 780M, fp16 weights and compute

Each arm greedy-decodes the NPU check's two chat prompts (the same token ids, from build/llm_decode/ref_2x16.npz)
for the same number of new tokens, `rounds` times after a warm-up. Timed separately: the prompt (one batched
forward pass, which the NPU decoder does not have: it runs a prompt one token at a time) and every generated token
(one decode step each, the number to set against the NPU's ms per token). Token agreement with the exact reference
is reported, not required: these arms round differently by design.

    python tools/llm_rivals.py build                    # resnet_env17; build/llm_rivals/<arm>/
    python tools/llm_rivals.py run --arm cpu-int4       # resnet_env17, through research-lowlevel.sh for a timed sitting
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "llm_rivals"
REF = ROOT / "build" / "llm_decode" / "ref_2x16.npz"
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
ARMS = {"cpu-fp32": ("fp32", "cpu", {}), "cpu-int4": ("int4", "cpu", {"int4_accuracy_level": "4"}),
        "dml-fp16": ("fp16", "dml", {})}


def snapshot():
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL, local_files_only=True)


def build(arms):
    import onnxruntime_genai.models.builder as B
    src = snapshot()
    for arm in arms:
        prec, ep, extra = ARMS[arm]
        out = OUT / arm
        if (out / "genai_config.json").exists():
            print("BUILT", arm, "(exists)", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        B.create_model(MODEL, src, str(out), prec, ep, str(OUT / "cache"), **extra)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        print("BUILT " + json.dumps(dict(arm=arm, precision=prec, ep=ep, extra=extra, bytes=size,
                                         seconds=round(time.perf_counter() - t0, 1))), flush=True)
    return 0


def run(arm, rounds):
    import onnxruntime_genai as og
    ref = np.load(REF)
    prompts = [(ref[f"{n}_ids"].tolist(), ref[f"{n}_out"].tolist()) for n in range(len(ref["prompts"]))]
    model = og.Model(str(OUT / arm))
    print("ARM " + json.dumps(dict(arm=arm, genai=og.__version__, config=json.loads(
        (OUT / arm / "genai_config.json").read_text())["model"]["decoder"]["session_options"])), flush=True)

    def generate(ids, n):
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=len(ids) + n)
        gen = og.Generator(model, params)
        t0 = time.perf_counter()
        gen.append_tokens(ids)                       # the prompt, one batched forward pass
        gen.generate_next_token()                    # the first new token: picked from the prompt's logits
        prompt_s = time.perf_counter() - t0
        out, steps = [int(gen.get_next_tokens()[0])], []
        while len(out) < n and out[-1] != 2:         # 2 = <|im_end|>, as the NPU run stops
            t0 = time.perf_counter()
            gen.generate_next_token()                # one forward pass on the last token, then the pick
            steps.append(time.perf_counter() - t0)
            out.append(int(gen.get_next_tokens()[0]))
        return out, prompt_s, steps

    generate(prompts[0][0][:8], 4)                   # warm-up
    decode, prefill, agree = [], [], []
    for rnd in range(rounds):
        for ids, want in prompts:
            got, ps, steps = generate(ids, len(want))
            decode += steps
            prefill.append((len(ids), ps))
            agree.append(sum(a == b for a, b in zip(got, want)))
            print("ROUND " + json.dumps(dict(round=rnd, prompt_tokens=len(ids), prompt_ms=round(ps * 1e3, 2),
                                             new_tokens=len(got), agree_with_exact_reference=agree[-1],
                                             of=len(want), decode_ms_median=round(1e3 * float(np.median(steps)), 3))),
                  flush=True)
    d = np.array(decode) * 1e3
    print("TIMING_JSON " + json.dumps(dict(
        arm=arm, decode_steps=int(d.size), ms_per_token_median=round(float(np.median(d)), 3),
        ms_per_token_p90=round(float(np.percentile(d, 90)), 3),
        tokens_per_second_median=round(1e3 / float(np.median(d)), 2),
        prompt_ms_median={str(n): round(1e3 * float(np.median([s for m, s in prefill if m == n])), 2)
                          for n in sorted({m for m, _ in prefill})})), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=tuple(ARMS), required=True)
    r.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    return build(a.arms) if a.cmd == "build" else run(a.arm, a.rounds)


if __name__ == "__main__":
    sys.exit(main())
