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
``compile --epilogue FILE`` links FILE's egemm_epilogue into every shape (mode 0, so every product above must still
pass) and adds a mode-1 stream ``g_M_K_N_m1`` / ``_m2`` for gate|up (kernels/embedgemma/geglu_epilogue.cc: GeGLU on the
accumulator, mode 1 fp32, mode 2 native bf16). ``run`` then also checks each ``_m1`` shape: gate|up's columns interleaved per 64-column tile, the
SAME A through mode 0 and mode 1, mode 1's gate columns against fp64 gelu_tanh(gate) * up computed from mode 0's
own C. Criteria, fixed before each mode's first run (2026-10-18): mode 1, worst |device - fp64| <= 1e-5 of that
row's max |reference|; mode 2, worst |device - geglu_model mode 2| <= 2 bf16 ulps of that row's max, its error
against fp64 reported beside it; both finite and repeat bit-equal. The added device time per call (mode 1 minus mode 0) is reported.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels" / "embedgemma"))
import geglu_model  # noqa: E402

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


def stem(out: Path, M: int, K: int, N: int, mode: int = 0) -> Path:
    return out / (f"g_{M}_{K}_{N}" + (f"_m{mode}" if mode else ""))


def gate_up_shape(config: Path) -> tuple[int, int]:
    return next((K, N) for name, K, N in forward(config) if name == "gate|up")


def interleave(w: np.ndarray) -> np.ndarray:
    """gate|up [K, 2F] -> per 64-column output tile p: gate columns 32p..32p+31 then up columns 32p..32p+31."""
    K, N = w.shape
    return w.reshape(K, 2, N // 64, 32).transpose(0, 2, 1, 3).reshape(K, N)


def gate_columns(c: np.ndarray, N: int) -> np.ndarray:
    """An interleaved gate|up output [M, N] -> its gate positions [M, N / 2] (mode 1: the GeGLU result)."""
    return c[:, :N].reshape(c.shape[0], N // 64, 2, 32)[:, :, 0, :].reshape(c.shape[0], N // 2)


def gelu_tanh(x):
    x = np.asarray(x, np.float64)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def cmd_compile(a) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    shapes = sorted({(K, N) for _, K, N in forward(a.config)})
    jobs = [(M, K, N, 0) for M in a.ms for K, N in shapes]
    if a.epilogue:
        jobs += [(M, *gate_up_shape(a.config), md) for M in a.ms for md in a.modes]
    for M, K, N, mode in jobs:
            s = stem(a.out, M, K, N, mode)
            if Path(f"{s}.xclbin").exists():
                print(f"COMPILE {s.name}: exists", flush=True)
                continue
            cmd = [sys.executable, str(DESIGN), "--dev", "npu", "-M", str(M), "-K", str(K), "-N", str(padded(N)),
                   "-m", str(TILE["m"]), "-k", str(TILE["k"]), "-n", str(TILE["n"]), "--c-single-buffer", "1",
                   "--dtype_in", "bf16", "--dtype_out", "f32", "--xclbin-path", f"{s}.xclbin",
                   "--insts-path", f"{s}.bin", "--rtp-generic"]
            if a.epilogue:
                cmd += ["--epilogue-src", str(a.epilogue.resolve()), "--epilogue-mode", str(mode)]
            if a.hollow_matmul:
                cmd += ["--hollow-matmul", str(a.hollow_matmul.resolve())]
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

        gu = gate_up_shape(a.config)
        modes = [md for md in (1, 2, 3, 4, 5, 6) if Path(f"{stem(a.out, M, *gu, md)}.bin").exists()]
        if modes:   # the epilogue: mode 0 and each mode on the same A and the interleaved weights
            K, N = gu
            wp = to_bf16_bits(interleave(bf16_float(weights[gu])[:, :N]))
            Bp = bo(wp.nbytes)
            Bp.write(wp.tobytes(), 0)
            Bp.sync(to_dev)
            ib0, nb0, A, C, _ = prods[gu]
            streams = {0: (ib0, nb0)}
            for md in modes:
                words_ = np.frombuffer(Path(f"{stem(a.out, M, *gu, md)}.bin").read_bytes(), dtype=np.uint32)
                ib_ = pyxrt.bo(dev, words_.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
                ib_.write(words_.tobytes(), 0)
                ib_.sync(to_dev)
                streams[md] = (ib_, words_.nbytes)
            outs, times = {}, {md: [] for md in streams}
            for rep_ in range(1 + a.seqs):
                for mode, (ib_, nb_) in streams.items():
                    A.sync(to_dev)
                    t = time.perf_counter()
                    st = kern(3, ib_, nb_, A, Bp, C).wait(WAIT_MS)
                    if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                        raise RuntimeError(f"M={M} {gu} mode {mode}: {st}")
                    times[mode].append((time.perf_counter() - t) * 1000)
                    C.sync(from_dev)
                    outs.setdefault(mode, []).append(
                        np.frombuffer(C.read(M * padded(N) * 4, 0), dtype=np.float32).reshape(M, -1).copy())
            c0 = outs[0][0]
            g0 = c0[:, :N].reshape(M, N // 64, 2, 32)[:, :, 0, :].reshape(M, N // 2)
            u0 = c0[:, :N].reshape(M, N // 64, 2, 32)[:, :, 1, :].reshape(M, N // 2)
            exact = gelu_tanh(g0) * u0.astype(np.float64)
            plain = np.abs(c0[:, :N] - (bf16_float(prods[gu][4]) @ bf16_float(wp))).max() / np.abs(c0[:, :N]).max()
            d0 = statistics.median(times[0][1:])
            for md in modes:
                if md in (3, 4, 7):   # timing probe: wrong output by construction, no check
                    dm = statistics.median(times[md][1:])
                    extra = ""
                    if md == 7:   # each tile's epilogue cycles, written over its row 0, column 32
                        cyc = outs[7][0][::32, 32:N:64].view(np.int32).astype(np.float64)
                        extra = (f"; epilogue cycles per tile median {np.median(cyc):.0f} (min {cyc.min():.0f}, max "
                                 f"{cyc.max():.0f}) over {cyc.size} tiles -> {np.median(cyc) * cyc.size / 16 / 1.8e6:.3f} ms "
                                 f"per core at 1.80 GHz")
                    print(f"  PROBE mode {md} {M}x{K}x{N}: call median {dm:.3f} ms (+{dm - d0:.3f} over mode 0){extra}",
                          flush=True)
                    report.setdefault("epilogue", {})[f"{M}_m{md}"] = dict(mode0_ms=d0, mode_ms=dm, probe=True)
                    continue
                got = gate_columns(outs[md][0], N).astype(np.float64)
                vs_exact = float((np.abs(got - exact) / np.abs(exact).max(axis=1, keepdims=True)).max())
                if md == 1:   # fp32: against fp64 at 1e-5 of the row max
                    worst, crit = vs_exact, "<= 1e-5 of row max vs fp64"
                    passed_ = worst <= 1e-5
                else:         # bf16: against its own op-for-op model at 2 bf16 ulps of the row max
                    ref = geglu_model.geglu(g0, u0, 5 if md == 6 else md).astype(np.float64)
                    rowmax = np.abs(ref).max(axis=1, keepdims=True)
                    tol = 2 * np.exp2(np.floor(np.log2(np.maximum(rowmax, 1e-30))) - 7)
                    worst, crit = float((np.abs(got - ref) / tol).max()), "<= 1 (2 bf16 ulps of row max) vs model"
                    passed_ = worst <= 1.0
                same = all(np.array_equal(outs[md][0].view(np.uint32), o.view(np.uint32)) for o in outs[md][1:])
                good = bool(np.isfinite(got).all()) and same and passed_ and plain < 1e-3
                ok &= good
                dm = statistics.median(times[md][1:])
                print(f"  EPILOGUE mode {md} {M}x{K}x{N} GeGLU: worst {worst:.3e} ({crit}), vs fp64 {vs_exact:.2e} of row max, "
                      f"mode-0 product rel err {plain:.2e}, repeat bit-equal {same}: {'PASS' if good else 'FAIL'}; "
                      f"call median mode 0 {d0:.3f} ms, mode {md} {dm:.3f} ms (+{dm - d0:.3f})", flush=True)
                report.setdefault("epilogue", {})[f"{M}_m{md}"] = dict(worst=worst, vs_fp64=vs_exact, mode0_ms=d0,
                                                                      mode_ms=dm, passed=bool(good))
            np.savez_compressed(a.out / f"epilogue_{M}.npz", **{f"c{md}": outs[md][0] for md in outs})
            del Bp, streams, ib_, ib0, words_   # every BO handle before the device (a leftover one segfaults teardown, exit 139)
            if a.epilogue_only:
                continue
        for sh in ([] if a.timing_only else shapes):   # correctness gate before any timing
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
    print("RESULT " + json.dumps(dict(ok=bool(ok), report={str(k): v for k, v in report.items()})), flush=True)
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
            p.add_argument("--epilogue-only", action="store_true", help="only the epilogue check (no plain checks, no timing)")
            p.add_argument("--timing-only", action="store_true", help="hollow builds: skip the correctness checks "
                           "(their output is wrong by construction) and report timing only")
        else:
            p.add_argument("--epilogue", type=Path, help="link this egemm_epilogue source; adds gate|up mode streams")
            p.add_argument("--hollow-matmul", type=Path, help="timing probe: link this no-op in matmul's place")
            p.add_argument("--modes", type=int, nargs="+", default=[1, 2],
                           help="epilogue modes to build gate|up streams for (3, 4: timing probes)")
    a = ap.parse_args()
    return cmd_compile(a) if a.cmd == "compile" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
