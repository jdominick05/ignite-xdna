#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""EmbeddingGemma 2's weight GEMMs on the one-context generic design, at several sequence lengths M.

Question: what do the encoder's weight products cost on Phoenix when one loaded design serves every shape, which is
the first thing a fused encoder must beat or build on? A derived estimate priced them at older whole_array rates,
with every M=128 shape extrapolated; this measures them.

Shapes come from the HF config (``--config``), with the products that share an input concatenated: q|k|v as one GEMM
per layer, gate|up as one. Per layer: QKV (512 -> q+k+v), o (H*head_dim -> 512), gate|up (512 -> 2*2048), down
(2048 -> 512), PLE gate and PLE projection (512 -> 512); plus the per-layer-input projection once (512 -> 24*512,
issued as three 4096-column chunks because N = 12288 overflows a shim DMA stride) and the output projection
(512 -> 768) once. Attention's per-head score and value products are NOT included: they need
transposed/banded operands, not weight GEMMs.

    python tools/embedgemma_gemm_probe.py compile --out DIR --ms 256 512 2048     (ironenv, offline)
    python tools/embedgemma_gemm_probe.py run --out DIR --ms 256 512 2048 [--seqs 5] (pyxrt, NPU)

The design is kernels/gemm_generic/whole_array_generic.py --rtp-generic, the tile Laya's hybrid runs on (m32 k64
n64, C single-buffered, bf16 in, f32 out, 4 columns). ``run`` uses ONE xclbin for every (M, K, N) and checks every
product against numpy (relative max error < 1e-3, padded columns zero, repeat bit-equal) before timing; a shape whose
device image differs from the shared one shows up as a failed check, not as a silent wrong number.
Timing is the device side per product, as tools/gemm_generic_probe.py does: sync A in, call, wait, sync C out. It
excludes the fp32 -> bf16 conversion of each output (host work Laya measured separately), every non-GEMM op and
attention. Activations are fixed random data. Run it under scripts/research-lowlevel.sh --npu so foreign contexts
are watched for the whole run.
"""
import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "kernels" / "gemm_generic" / "whole_array_generic.py"
CONFIG = Path(r"L:\huggingface\hub\models--google--embeddinggemma-2\snapshots"
              r"\914f7f89142e33e77833254d9c9b90c3cef7303b\config.json")
TILE = dict(m=32, k=64, n=64)
XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
WAIT_MS = 10000


def padded(N: int) -> int:
    return -(-N // 256) * 256


def forward(config: Path) -> list[tuple[str, int, int]]:
    """(name, K, N) for every weight product of one forward pass, in model order, shared-input products concatenated."""
    t = json.loads(config.read_text(encoding="utf-8"))["text_config"]
    d, H, ff, ple = t["hidden_size"], t["num_attention_heads"], t["intermediate_size"], t["hidden_size_per_layer_input"]
    # N = 24*512 = 12288 overflows a shim DMA stride (20-bit step); issue it as column chunks of at most 4096
    width = t["num_hidden_layers"] * ple
    seq = [("per-layer-input", d, min(4096, width - c)) for c in range(0, width, 4096)]
    for i, kind in enumerate(t["layer_types"]):
        pc = t["per_layer_config"].get(f"{i:02d}", {})
        hd, kv = pc.get("head_dim", t["head_dim"]), pc.get("num_key_value_heads", t["num_key_value_heads"])
        tag = "global" if kind == "full_attention" else "local"
        seq += [(f"{tag} qkv", d, (H + 2 * kv) * hd), (f"{tag} o", H * hd, d), ("gate|up", d, 2 * ff),
                ("down", ff, d), ("ple gate", d, ple), ("ple proj", ple, d)]
    return seq + [("output", d, t["embedding_dim"])]


def stem(out: Path, M: int, K: int, N: int) -> Path:
    return out / f"g_{M}_{K}_{N}"


def cmd_compile(a) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    shapes = sorted({(K, N) for _, K, N in forward(a.config)})
    for M in a.ms:
        for K, N in shapes:
            s = stem(a.out, M, K, N)
            if Path(f"{s}.xclbin").exists():
                print(f"COMPILE {s.name}: exists", flush=True)
                continue
            cmd = [sys.executable, str(DESIGN), "--dev", "npu", "-M", str(M), "-K", str(K), "-N", str(padded(N)),
                   "-m", str(TILE["m"]), "-k", str(TILE["k"]), "-n", str(TILE["n"]), "--c-single-buffer", "1",
                   "--dtype_in", "bf16", "--dtype_out", "f32", "--xclbin-path", f"{s}.xclbin",
                   "--insts-path", f"{s}.bin", "--rtp-generic"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            ok = r.returncode == 0 and Path(f"{s}.xclbin").exists()
            print(f"COMPILE {s.name} (N run as {padded(N)}): {'ok' if ok else 'FAIL rc=%d' % r.returncode}", flush=True)
            if not ok:
                print(r.stdout[-3000:] + r.stderr[-3000:])
                return 1
    return 0


def idle() -> bool:
    out = subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True, timeout=60).stdout
    return "No hardware contexts running" in out


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def bf16_float(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def cmd_run(a) -> int:
    sys.path.insert(0, "C:/Xilinx/XRT/xrt_sdk/xrt/python")
    import pyxrt
    if not idle():
        print("NPU not idle: refusing (a foreign context is contention, not a finding)")
        return 3
    seq = forward(a.config)
    shapes = sorted({(K, N) for _, K, N in seq})
    rng = np.random.default_rng(3)
    weights = {}
    for K, N in shapes:
        w = np.zeros((K, padded(N)), np.float32)
        w[:, :N] = rng.standard_normal((K, N)) * 0.02
        weights[(K, N)] = to_bf16_bits(w)
    shared = Path(f"{stem(a.out, a.ms[0], *shapes[0])}.xclbin")
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(str(shared))
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    to_dev, from_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    bo = lambda nbytes: pyxrt.bo(dev, nbytes, pyxrt.bo.host_only, 0)
    Bs = {}
    for sh, w in weights.items():
        Bs[sh] = bo(w.nbytes)
        Bs[sh].write(w.tobytes(), 0)
        Bs[sh].sync(to_dev)   # weights written once, resident
    print(f"RUN one context on {shared.name}: {len(seq)} products per forward pass, {len(shapes)} distinct (K, N), "
          f"M in {a.ms}", flush=True)
    ok, report = True, {}
    for M in a.ms:
        prods = {}
        for K, N in shapes:
            words = np.frombuffer(Path(f"{stem(a.out, M, K, N)}.bin").read_bytes(), dtype=np.uint32)
            ib = pyxrt.bo(dev, words.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
            ib.write(words.tobytes(), 0)
            ib.sync(to_dev)
            act = to_bf16_bits(rng.standard_normal((M, K)).astype(np.float32))
            A, C = bo(act.nbytes), bo(M * padded(N) * 4)
            A.write(act.tobytes(), 0)
            prods[(K, N)] = (ib, words.nbytes, A, C, act)

        def step(sh):
            ib, nb, A, C, _ = prods[sh]
            A.sync(to_dev)
            st = kern(3, ib, nb, A, Bs[sh], C).wait(WAIT_MS)
            if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"M={M} {sh}: {st}")
            C.sync(from_dev)

        for sh in shapes:   # correctness gate before any timing
            K, N = sh
            step(sh)
            r1 = np.frombuffer(prods[sh][3].read(M * padded(N) * 4, 0), dtype=np.float32).reshape(M, -1).copy()
            step(sh)
            r2 = np.frombuffer(prods[sh][3].read(M * padded(N) * 4, 0), dtype=np.float32).reshape(M, -1).copy()
            ref = bf16_float(prods[sh][4]) @ bf16_float(weights[sh])
            err = float(np.abs(r1[:, :N] - ref[:, :N]).max() / np.abs(ref[:, :N]).max())
            good = err < 1e-3 and not np.abs(r1[:, N:]).any() and np.array_equal(r1.view(np.uint32), r2.view(np.uint32))
            ok &= good
            print(f"  CHECK {M}x{K}x{N}: rel err {err:.2e}: {'PASS' if good else 'FAIL'}", flush=True)
        if not ok:
            break
        for name, K, N in seq:   # warm-up pass, untimed
            step((K, N))
        per, passes = {}, []
        for _ in range(a.seqs):
            t0 = time.perf_counter()
            for name, K, N in seq:
                t = time.perf_counter()
                step((K, N))
                per.setdefault((K, N), []).append((time.perf_counter() - t) * 1000)
            passes.append((time.perf_counter() - t0) * 1000)
        flops = sum(2.0 * M * K * N for _, K, N in seq)
        med = statistics.median(passes)
        report[M] = dict(pass_ms=passes, median_ms=med, gflops=flops / med / 1e6,
                         per_shape_ms={f"{K}x{N}": statistics.median(v) for (K, N), v in per.items()})
        print(f"PASS M={M}: forward-pass weight GEMMs median {med:.1f} ms (min {min(passes):.1f}, max {max(passes):.1f}, "
              f"{a.seqs} passes), {flops / 1e9:.1f} GFLOP -> {flops / med / 1e6:.0f} GFLOPS", flush=True)
        for (K, N), v in sorted(per.items()):
            m = statistics.median(v)
            print(f"    {M}x{K}x{N}: median {m:.3f} ms/call, {2.0 * M * K * N / m / 1e6:.0f} GFLOPS", flush=True)
        prods.clear()
        gc.collect()
    print("RESULT " + json.dumps(dict(ok=ok, report={str(k): v for k, v in report.items()})), flush=True)
    # Release in dependency order (BOs, kernel, context, xclbin, device) before returning. The setup loop's own
    # ib/A/C names still hold the last M's BOs; left bound, they were freed after the device when cmd_run's frame was
    # torn down, and the process segfaulted (exit 139) after every result had printed (faulthandler: at return).
    prods.clear()
    Bs.clear()
    del step, ib, A, C
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
        p.add_argument("--ms", type=int, nargs="+", default=[256, 512, 2048], help="sequence rows M; this tile needs multiples of 256 (two 128-token docs = 256)")
        p.add_argument("--config", type=Path, default=CONFIG)
        if name == "run":
            p.add_argument("--seqs", type=int, default=5, help="timed forward passes per M")
    a = ap.parse_args()
    return cmd_compile(a) if a.cmd == "compile" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
