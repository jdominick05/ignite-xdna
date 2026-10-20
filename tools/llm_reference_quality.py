#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small-LLM decode, step 2 gate: do the NPU decoder's rounding points keep SmolLM2-135M's quality?

kernels/llm_gemv/reference.py is the decode step with every rounding point the device will have (bf16 between every
op, table-based exp/SiLU/reciprocal, one rounding per sum). Before any of those ops becomes a kernel, this compares it
with Hugging Face's own model on the CPU, in fp32 and in bf16:

  greedy   the same chat prompts, N new tokens each: positional token match and first divergence, against HF fp32 and
           HF bf16 (HF bf16 against HF fp32 is the yardstick: bf16 itself already diverges on open-ended prompts)
  nll      teacher-forced mean negative log-likelihood (and perplexity) of a fixed passage under each model

The reference runs in its fast mode (each sum in float64, rounded once), which can differ from the exp_sum MAC only in
the last bit of rare sums. Run in resnet_env (torch + transformers):

    python tools/llm_reference_quality.py --tokens 64 --out <json>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

# By file path: resnet_env has Hugging Face's own `kernels` package installed, which shadows this repo's kernels/.
import importlib.util  # noqa: E402
import types  # noqa: E402

# reference.py imports kernels.llm_gemv.pack: register this repo's package and pack under those names first
_pkg = types.ModuleType("kernels.llm_gemv")
_pkg.__path__ = [str(ROOT / "kernels" / "llm_gemv")]
sys.modules["kernels.llm_gemv"] = _pkg
_pspec = importlib.util.spec_from_file_location("kernels.llm_gemv.pack", ROOT / "kernels" / "llm_gemv" / "pack.py")
sys.modules["kernels.llm_gemv.pack"] = _pack = importlib.util.module_from_spec(_pspec)
_pspec.loader.exec_module(_pack)

_spec = importlib.util.spec_from_file_location("llm_reference", ROOT / "kernels" / "llm_gemv" / "reference.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["llm_reference"] = R
_spec.loader.exec_module(R)
MODEL, model_file, safetensors = R.MODEL, R.model_file, R.safetensors

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a short poem about the ocean.",
    "Explain what a neural network is to a ten-year-old.",
    "List three uses for a paperclip.",
    "Summarize the water cycle in two sentences.",
]
# A fixed passage for teacher-forced likelihood (written for this test, no external text).
PASSAGE = ("The water cycle describes how water moves through the environment. Heat from the sun evaporates water from "
           "oceans, lakes and rivers, and plants release vapour through their leaves. The vapour rises, cools and "
           "condenses into clouds. When the droplets in a cloud grow heavy enough, they fall as rain or snow. Some of "
           "that water soaks into the ground and refills underground stores, while the rest runs off into streams that "
           "carry it back to the sea, where the cycle begins again.")


def match(a, b):
    n = max(len(a), len(b))
    first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    return dict(match=sum(x == y for x, y in zip(a, b)), of=n, first_divergence=first)


def hf_greedy(model, ids, n, eos):
    import torch
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False, pad_token_id=eos,
                             attention_mask=torch.ones(1, len(ids), dtype=torch.long))
    return out[0, len(ids):].tolist()


def hf_nll(model, ids):
    import torch
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float()
    lp = torch.log_softmax(logits[:-1], -1)
    return float(-lp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].mean())


def ref_nll(dec, ids):
    nll = []
    for p, t in enumerate(ids[:-1]):
        logits = dec.step(t, p).astype(np.float64)
        m = logits.max()
        nll.append(-(logits[ids[p + 1]] - m - np.log(np.exp(logits - m).sum())))
    return float(np.mean(nll))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--unfused", action="store_true", help="v0 semantics: norms as their own op, rounded residual adds")
    ap.add_argument("--quant", choices=("w8a16", "w8a8", "w4a16"),
                    help="fake-quantize the fused reference: int8 per-row or int4 per-32 weights, W8A8's per-token "
                         "int8 activations (reference.quantize_weights)")
    a = ap.parse_args()
    if a.out and a.out.exists():
        raise SystemExit(f"{a.out} exists")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = tok.eos_token_id
    hf = {d: AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, d)).eval()
          for d in ("float32", "bfloat16")}
    w = R.Weights.load(safetensors(model_file()))
    if a.quant:
        if a.unfused:
            raise SystemExit("--quant applies to the fused semantics")
        w = R.quantize_weights(w, a.quant)
    rec = dict(model=MODEL, tokens=a.tokens, reference_mode="fast (float64 sums, one rounding)", quant=a.quant,
               semantics="v0 unfused" if a.unfused else "fused (norm in the projection, residual as accumulator start)",
               prompts=[])
    totals = {k: [0, 0] for k in ("ref_vs_fp32", "ref_vs_bf16", "bf16_vs_fp32")}
    for prompt in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
        t0 = time.perf_counter()
        ref = R.Decoder(w, fused=not a.unfused).generate(ids, a.tokens, eos)
        secs = time.perf_counter() - t0
        f32, b16 = hf_greedy(hf["float32"], ids, a.tokens, eos), hf_greedy(hf["bfloat16"], ids, a.tokens, eos)
        row = dict(prompt=prompt, prompt_tokens=len(ids), ref_seconds=round(secs, 1),
                   ref_vs_fp32=match(ref, f32), ref_vs_bf16=match(ref, b16), bf16_vs_fp32=match(b16, f32),
                   ref_text=tok.decode(ref), fp32_text=tok.decode(f32), bf16_text=tok.decode(b16))
        for k in totals:
            totals[k][0] += row[k]["match"]
            totals[k][1] += row[k]["of"]
        rec["prompts"].append(row)
        print("GREEDY " + json.dumps({k: row[k] for k in ("prompt", "ref_vs_fp32", "ref_vs_bf16", "bf16_vs_fp32",
                                                            "ref_seconds")}), flush=True)
        print("  ref :", repr(row["ref_text"][:150]), flush=True)
        print("  fp32:", repr(row["fp32_text"][:150]), flush=True)
    rec["greedy_totals"] = {k: dict(match=m, of=n, rate=round(m / n, 4)) for k, (m, n) in totals.items()}
    ids = tok(PASSAGE)["input_ids"]
    rec["passage_tokens"] = len(ids)
    rec["nll"] = dict(hf_fp32=hf_nll(hf["float32"], ids), hf_bf16=hf_nll(hf["bfloat16"], ids),
                      reference=ref_nll(R.Decoder(w, fused=not a.unfused), ids))
    rec["perplexity"] = {k: float(np.exp(v)) for k, v in rec["nll"].items()}
    print("GREEDY_TOTALS " + json.dumps(rec["greedy_totals"]), flush=True)
    print("NLL " + json.dumps(rec["nll"]) + " PPL " + json.dumps(rec["perplexity"]), flush=True)
    if a.out:
        a.out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
