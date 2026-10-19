#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host half of unfused NPU attention for EmbeddingGemma 2: mask + softmax of the f32 scores, P rounded to bf16.

tools/embedgemma_attn_gemm_probe.py times the score and value products on the NPU; between them the host must turn
each f32 score matrix S [rows, L] (query heads of one KV head stacked along rows) into bf16 probabilities: apply the
local band (|i - j| <= window, local layers only), softmax along keys in fp32, round to bf16. This times exactly that
in torch on the CPU, per document, at the probe's shapes and call counts (read from the HF config), so the two
together price the unfused path. Run in an env with torch (laya-ref); ``--threads`` as the CPU baselines (8).
One untimed warm-up per shape; median of ``--reps``.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedgemma_attn_gemm_probe import CONFIG, products  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ls", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(a.threads)
    window = json.loads(a.config.read_text(encoding="utf-8"))["text_config"]["sliding_window"]
    rec = dict(threads=a.threads, torch=torch.__version__, window=window, per_length={})
    for L in a.ls:
        rows_ = []
        for name, M, K, N, calls in products(a.config, L):
            if not name.endswith("scores"):
                continue
            g = torch.Generator().manual_seed(L)
            S = torch.randn(M, N, generator=g)
            band = None
            if name.startswith("local"):
                i = torch.arange(L)
                band = ((i[:, None] - i[None, :]).abs() > window).repeat(M // L, 1)

            def step():
                x = S.masked_fill(band, float("-inf")) if band is not None else S
                return torch.softmax(x, dim=-1).to(torch.bfloat16)

            step()
            ts = []
            for _ in range(a.reps):
                t = time.perf_counter()
                step()
                ts.append((time.perf_counter() - t) * 1000)
            med = statistics.median(ts)
            rows_.append(dict(name=name, rows=M, keys=N, calls=calls, median_ms=med))
            print(f"L={L} {name} [{M}, {N}]: median {med:.2f} ms/call, x{calls} per document", flush=True)
        total = sum(r["median_ms"] * r["calls"] for r in rows_)
        rec["per_length"][str(L)] = dict(calls=rows_, per_document_ms=total)
        print(f"L={L}: host mask+softmax+bf16 per document {total:.1f} ms", flush=True)
    a.out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
