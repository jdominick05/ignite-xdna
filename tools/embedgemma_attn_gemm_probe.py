#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""EmbeddingGemma 2's attention as plain GEMMs on the one-context generic design: the unfused NPU baseline.

Question: what does attention cost on the NPU with no fused kernel, the score and value products issued to the same
generic GEMM as the weights and softmax left to the host? That is the bar a fused attention kernel has to beat.

Per document of L tokens, query heads that share a KV head are stacked along M, so one call serves them all (GQA):
  local layers (head_dim 256, 2 KV heads, 2 query heads each), per KV head:
      scores  S = Q_stack [2L, 256] @ K^T [256, L]       value  O = P [2L, L] @ V [L, 256]
  global layers (head_dim 512, 1 KV head, 4 query heads):
      scores  S = Q_stack [4L, 512] @ K^T [512, L]       value  O = P [4L, L] @ V [L, 512]
Shapes and counts are read from the HF config. K^T is laid out on the host (the shared xclbin takes row-major B). The
local layers' +-512 window is not exploited: at L = 2048 every score is computed and the band is applied on the host,
which a fused kernel would not need. Softmax, the mask and P's f32 -> bf16 rounding are host work and are NOT timed
here (``hf_split`` priced attention as a whole on the CPU).

    python tools/embedgemma_attn_gemm_probe.py compile --out DIR --ls 512 2048        (ironenv, offline)
    python tools/embedgemma_attn_gemm_probe.py run --out DIR --ls 512 2048 [--reps 5]  (pyxrt, NPU)

``run`` loads the same shared xclbin as tools/embedgemma_gemm_probe.py (``g_256_512_512.xclbin`` in DIR), checks
every product against numpy (relative max error < 1e-3, repeat bit-equal) before timing, then times each product
as sync A in, call, wait, sync C out, with B written once. Per-document totals are per-product medians times their
counts. Run it under scripts/research-lowlevel.sh --npu.
"""
import argparse
import gc
import json
import statistics
import subprocess
import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from embedgemma_gemm_probe import CONFIG, DESIGN, TILE, WAIT_MS, bf16_float, idle, stem, to_bf16_bits  # noqa: E402


def products(config: Path, L: int) -> list[tuple[str, int, int, int, int]]:
    """(name, M, K, N, calls per document) for one document of L tokens."""
    t = json.loads(config.read_text(encoding="utf-8"))["text_config"]
    H, per = t["num_attention_heads"], t["per_layer_config"]
    groups = {}
    for i, kind in enumerate(t["layer_types"]):
        pc = per.get(f"{i:02d}", {})
        hd, kv = pc.get("head_dim", t["head_dim"]), pc.get("num_key_value_heads", t["num_key_value_heads"])
        groups.setdefault((kind, hd, kv), 0)
        groups[(kind, hd, kv)] += 1
    out = []
    for (kind, hd, kv), layers in sorted(groups.items()):
        rows = (H // kv) * L
        tag = "local" if kind == "sliding_attention" else "global"
        out += [(f"{tag} scores", rows, hd, L, layers * kv), (f"{tag} values", rows, L, hd, layers * kv)]
    return out


def cmd_compile(a) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    for L in a.ls:
        for name, M, K, N, _ in products(a.config, L):
            s = stem(a.out, M, K, N)
            if Path(f"{s}.bin").exists():
                print(f"COMPILE {s.name} ({name}, L={L}): exists", flush=True)
                continue
            cmd = [sys.executable, str(DESIGN), "--dev", "npu", "-M", str(M), "-K", str(K), "-N", str(N),
                   "-m", str(TILE["m"]), "-k", str(TILE["k"]), "-n", str(TILE["n"]), "--c-single-buffer", "1",
                   "--dtype_in", "bf16", "--dtype_out", "f32", "--xclbin-path", f"{s}.xclbin",
                   "--insts-path", f"{s}.bin", "--rtp-generic"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            ok = r.returncode == 0 and Path(f"{s}.bin").exists()
            print(f"COMPILE {s.name} ({name}, L={L}): {'ok' if ok else 'FAIL rc=%d' % r.returncode}", flush=True)
            if not ok:
                print(r.stdout[-3000:] + r.stderr[-3000:])
                return 1
    return 0


def cmd_run(a) -> int:
    sys.path.insert(0, "C:/Xilinx/XRT/xrt_sdk/xrt/python")
    import pyxrt
    if not idle():
        print("NPU not idle: refusing (a foreign context is contention, not a finding)")
        return 3
    shared = a.out / "g_256_512_512.xclbin"
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(str(shared))
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    to_dev, from_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    bo = lambda nbytes: pyxrt.bo(dev, nbytes, pyxrt.bo.host_only, 0)
    rng = np.random.default_rng(5)
    ok, report = True, {}
    print(f"RUN one context on {shared.name}: attention products at L in {a.ls}", flush=True)
    for L in a.ls:
        rows = []
        for name, M, K, N, calls in products(a.config, L):
            words = np.frombuffer(Path(f"{stem(a.out, M, K, N)}.bin").read_bytes(), dtype=np.uint32)
            ib = pyxrt.bo(dev, words.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
            ib.write(words.tobytes(), 0)
            ib.sync(to_dev)
            act = to_bf16_bits(rng.standard_normal((M, K)).astype(np.float32))
            w = to_bf16_bits(rng.standard_normal((K, N)).astype(np.float32))
            A, B, C = bo(act.nbytes), bo(w.nbytes), bo(M * N * 4)
            A.write(act.tobytes(), 0)
            B.write(w.tobytes(), 0)
            B.sync(to_dev)

            def step():
                A.sync(to_dev)
                st = kern(3, ib, words.nbytes, A, B, C).wait(WAIT_MS)
                if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    raise RuntimeError(f"{name} {M}x{K}x{N}: {st}")
                C.sync(from_dev)

            step()
            r1 = np.frombuffer(C.read(M * N * 4, 0), dtype=np.float32).copy()
            step()
            r2 = np.frombuffer(C.read(M * N * 4, 0), dtype=np.float32).copy()
            ref = (bf16_float(act) @ bf16_float(w)).ravel()
            err = float(np.abs(r1 - ref).max() / np.abs(ref).max())
            good = err < 1e-3 and np.array_equal(r1.view(np.uint32), r2.view(np.uint32))
            ok &= good
            print(f"  CHECK L={L} {name} {M}x{K}x{N}: rel err {err:.2e}: {'PASS' if good else 'FAIL'}", flush=True)
            if not good:
                break
            ts = []
            for _ in range(a.reps):
                t = time.perf_counter()
                step()
                ts.append((time.perf_counter() - t) * 1000)
            med = statistics.median(ts)
            rows.append(dict(name=name, M=M, K=K, N=N, calls=calls, median_ms=med, min_ms=min(ts),
                             gflops=2.0 * M * K * N / med / 1e6, c_mbytes=M * N * 4 / 1e6))
            print(f"    {name} {M}x{K}x{N}: median {med:.3f} ms/call (min {min(ts):.3f}), {2.0 * M * K * N / med / 1e6:.0f} "
                  f"GFLOPS, x{calls} per document", flush=True)
            del step, ib, A, B, C
            gc.collect()
        if not ok:
            break
        total = sum(r["median_ms"] * r["calls"] for r in rows)
        report[L] = dict(products=rows, per_document_ms=total)
        print(f"PASS L={L}: attention score+value GEMMs per document {total:.1f} ms (host softmax/mask/rounding excluded)",
              flush=True)
    print("RESULT " + json.dumps(dict(ok=ok, report={str(k): v for k, v in report.items()})), flush=True)
    gc.collect()
    del kern
    gc.collect()
    del ctx
    gc.collect()
    del xb, dev
    gc.collect()
    print(f"idle after: {idle()}", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("compile", "run"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--ls", type=int, nargs="+", default=[512, 2048], help="document lengths in tokens")
        p.add_argument("--config", type=Path, default=CONFIG)
        if name == "run":
            p.add_argument("--reps", type=int, default=5, help="timed calls per product")
    a = ap.parse_args()
    return cmd_compile(a) if a.cmd == "compile" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
