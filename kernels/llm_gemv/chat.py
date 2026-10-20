# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SmolLM2-135M-Instruct chat on the NPU: the decoder in decode.py behind a small API, for demos.

    from kernels.llm_gemv.chat import SmolChat
    with SmolChat() as chat:                      # W8A16 by default; SmolChat(w8=False) for bf16
        for piece in chat.generate("What is the capital of France?"):
            print(piece, end="", flush=True)

Greedy decoding, the model's chat template with its default system message, a 576-position context (prompt and
reply together). The compiled decoder must exist under build/llm_decode (tools/llm_decode_probe.py [--w8] compile);
the weights come from the Hugging Face cache, packed at open (about 5 s). This is a demonstration: the CPU at int4
decodes this model about twice as fast (docs/BENCHMARKS.md, the small-LLM verdict).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator, Optional

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def _decode_module(w8: bool):
    """decode.py in the mode asked for. Its mode is fixed when it is first imported, so a second mode in one process
    is refused rather than silently given the first."""
    import sys
    if "kernels.llm_gemv.decode" not in sys.modules:
        os.environ["LLM_DECODE_W8"] = "1" if w8 else "0"
    from kernels.llm_gemv import decode
    if decode.W8 != w8:
        raise RuntimeError(f"decode.py is already loaded in {'W8A16' if decode.W8 else 'bf16'} mode in this process")
    return decode


class SmolChat:
    """Opens the NPU, loads the decoder and its weights; generate() streams one reply. Use as a context manager."""

    def __init__(self, w8: bool = True, build: Optional[Path] = None):
        self.D = _decode_module(w8)
        self.w8 = w8
        self.build = Path(build) if build else ROOT / "build" / "llm_decode" / ("token_w8" if w8 else "token")
        if not (self.build / "final.xclbin").exists():
            flag = "--w8 " if w8 else ""
            raise FileNotFoundError(f"no compiled decoder at {self.build}; build it with "
                                    f"python tools/llm_decode_probe.py {flag}compile")
        self._harness = self._npu = None
        self.stats = {}

    def __enter__(self) -> "SmolChat":
        from tokenizers import Tokenizer
        from ignite_xdna.runtime.driver import XrtSiliconHarness
        from kernels.llm_gemv import reference as R
        t0 = time.perf_counter()
        self.tok = Tokenizer.from_file(str(Path(R.model_file()).parent / "tokenizer.json"))
        self.eos = self.tok.token_to_id("<|im_end|>")
        w = R.Weights.load(R.safetensors(R.model_file()))
        if self.w8:
            w = R.quantize_weights(w, "w8a16-dev")
        self._harness = XrtSiliconHarness(0)
        h = self._harness.__enter__()
        try:
            self._npu = self.D.NpuDecoder(h, self.build, w)
        except BaseException:
            self._harness.__exit__(None, None, None)
            raise
        self.stats["open_s"] = time.perf_counter() - t0
        return self

    def __exit__(self, *exc):
        if self._npu is not None:
            self._npu.close()
            self._npu = None
        if self._harness is not None:
            self._harness.__exit__(*exc)
            self._harness = None
        import gc
        gc.collect()

    def prompt_ids(self, prompt: str) -> list:
        text = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n")
        return self.tok.encode(text, add_special_tokens=False).ids

    def generate(self, prompt: str, max_tokens: int = 128) -> Iterator[str]:
        """Yields the reply's text as it is decoded, until the end-of-reply token, max_tokens or the context's end.
        stats then holds prompt_tokens, new_tokens, prompt_s and tokens_per_s (the reply's decode rate)."""
        if self._npu is None:
            raise RuntimeError("open the chat first (with SmolChat() as chat: ...)")
        ids = self.prompt_ids(prompt)
        cap = self.D.CAP
        if len(ids) >= cap:
            raise ValueError(f"the prompt is {len(ids)} tokens with the chat template; the context holds {cap}")
        import numpy as np
        from kernels.llm_gemv import reference as R
        npu, out, shown = self._npu, [], ""
        t0 = time.perf_counter()
        logits = None
        for p, t in enumerate(ids):
            logits = npu.step(t, p)
        t1 = time.perf_counter()
        while len(out) < max_tokens:
            t = int(np.argmax(R.em.from_bf16_bits(logits)))
            if t == self.eos:
                break
            out.append(t)
            text = self.tok.decode(out)
            if not text.endswith("�"):                  # hold back a split multi-byte character
                yield text[len(shown):]
                shown = text
            p = len(ids) + len(out) - 1
            if p >= cap:
                break
            logits = npu.step(t, p)
        t2 = time.perf_counter()
        text = self.tok.decode(out)
        if text != shown:
            yield text[len(shown):]
        self.stats.update(prompt_tokens=len(ids), new_tokens=len(out), prompt_s=t1 - t0,
                          tokens_per_s=(len(out) / (t2 - t1)) if out and t2 > t1 else 0.0,
                          stopped=("end" if len(out) < max_tokens and p < cap else "limit"))
