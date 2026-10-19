#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""EmbeddingGemma 2's ops on the 780M iGPU (llama.cpp ROCm backend), per op and per forward pass, for the iGPU/NPU
placement question (the "logical device" decision).

Runs tools/ggml_gpu_opbench/opbench.exe (build: cmd /d /c tools\\ggml_gpu_opbench\\build.cmd) over:
  every weight GEMM shape of a forward pass (tools/embedgemma_gemm_probe.py forward(), bf16 weights, f32 activations)
    at each M, with its count per pass;
  local attention (hd 256, 4 heads on 2 KV heads, window 512) and global attention (hd 512, 1 KV head, no window),
    flash and classic, at each document length L (20 local + 4 global layers per pass);
  RMSNorm rows (L x 512) and host<->device copies of 1 to 32 MiB (the hand-off cost).
Per-pass totals multiply each op by its count. Check contention first; this is a GPU timing.

    python tools/embedgemma_igpu_opbench.py --bin ~/tools/llamacpp-b1343-rocm [--ms 256 512 2048] --out LOG.json
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedgemma_gemm_probe as probe  # noqa: E402

EXE = Path(__file__).resolve().parents[1] / "scratch" / "ggml-opbench" / "opbench.exe"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--ms", type=int, nargs="+", default=[256, 512, 2048], help="GEMM rows M")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048], help="attention document lengths")
    ap.add_argument("--exe", type=Path, default=EXE)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cnt = Counter((K, N) for _, K, N in probe.forward(probe.CONFIG))
    spec = [f"gemm {M} {K} {N}" for M in a.ms for (K, N) in sorted(cnt)]
    spec += [f"attn {L} 256 4 2 512" for L in a.lengths] + [f"attn {L} 512 4 1 0" for L in a.lengths]
    spec += [f"rms {L} 512" for L in a.lengths] + [f"copy {mb << 20}" for mb in (1, 4, 16, 32)]
    sp = a.out.with_suffix(".spec")
    sp.write_text("\n".join(spec) + "\n", encoding="utf-8")
    r = subprocess.run([str(a.exe), str(a.bin), str(sp)], capture_output=True, text=True, timeout=3600)
    rows = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]
    if r.returncode:
        print(r.stdout + r.stderr)
        return r.returncode
    for x in rows:
        print(json.dumps(x))
    summary = {}
    for M in a.ms:
        g = {(x["K"], x["N"]): x["ms"] for x in rows if x["op"] == "gemm" and x["M"] == M}
        summary[f"gemm_pass_ms_M{M}"] = sum(cnt[k] * g[k] for k in cnt)
    for L in a.lengths:
        for impl in ("flash", "classic"):
            loc = next(x["ms"] for x in rows if x["op"] == "attn" and x["L"] == L and x["window"] and x["impl"] == impl)
            glo = next(x["ms"] for x in rows if x["op"] == "attn" and x["L"] == L and not x["window"] and x["impl"] == impl)
            if loc >= 0 and glo >= 0:   # -1: ROCm0 does not support it
                summary[f"attn_{impl}_pass_ms_L{L}"] = 20 * loc + 4 * glo
            summary[f"attn_{impl}_local_ms_L{L}"], summary[f"attn_{impl}_global_ms_L{L}"] = loc, glo
    for k, v in summary.items():
        print(f"SUMMARY {k}: {v:.2f}")
    a.out.write_text(json.dumps(dict(rows=rows, summary=summary, spec=spec), indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
