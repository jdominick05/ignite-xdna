"""Does the core's aligned multiply-accumulate differ from fp32 where a model can see it?

WHY THIS EXISTS. `tools/onnx_bf16_cast.py` scores a bf16 model on the CPU, and its accuracy
number is what decides whether a bf16 engine is worth building. It modelled the core as
accumulating in fp32, and said so in its docstring. The accumulate probe
(results/aie/engine_bf16_mac_model_probe_npu_20260921.log) measured otherwise: one
`mmul<4,8,4>::mac` aligns its nine operands to the largest exponent among them and rounds each
separately to a 24-bit grid, ties to even. That is not IEEE addition, so the CPU score stopped
being a prediction of silicon and became an upper bound on it - unless the difference is below
what a bf16 store can carry, which is what this measures.

Both questions are asked against the emulator's own `mac()`, so this tests the MEASURED model
rather than a paraphrase of it, and against an fp64 reference that shares no code with either.

  --depth    relative error and stored-value disagreement by tap count, for the layer shapes a
             CNN actually has, including the post-ReLU all-positive case where errors accumulate
             instead of cancelling.
  --cancel   the regime the margin argument does not cover: a small difference of large terms.
             This repo has met it - YOLO-World's four C2fAttn cv2 convs cancel that way - so it
             is not hypothetical. Brackets the cancellation ratio at which the core loses a
             value the exact sum keeps.

Offline. No device, no timing claim.

    bash scripts/research-lowlevel.sh --log results/aie/bf16_accum_fidelity_<date>.log \\
        --checks-only -- bash scripts/research-iron.sh tools/bf16_accum_fidelity.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ignite_xdna.compiler.engine_bf16_emulator import mac, to_bf16, NCO  # noqa: E402

R, C = 2, 4          # every accumulator lane is independent, so the patch shape changes nothing


def _rel(x, ref):
    return float(np.median(np.abs(x - ref) / np.maximum(np.abs(ref), 1e-30)))


def _stored_differ(x, y):
    """Fraction of lanes disagreeing AFTER the bf16 store - the only difference a next layer sees."""
    return float(np.mean(to_bf16(np.asarray(x, np.float32)) != to_bf16(np.asarray(y, np.float32))))


def _accumulate(rng, taps, w_scale, positive):
    """One output tile over `taps` multiply-accumulates: the core's model, an exact sum, fp64."""
    acc = np.zeros((NCO, R, C, 4), np.float32)
    exact = np.zeros((NCO, R, C, 4), np.float32)
    ref = np.zeros((NCO, R, C, 4), np.float64)
    for _ in range(taps):
        a = rng.normal(size=(R, C, 8)).astype(np.float32)
        if positive:
            a = np.abs(a)                      # post-ReLU activations never cancel each other
        a = to_bf16(a)
        w = to_bf16(rng.normal(scale=w_scale, size=(NCO, 8, 4)).astype(np.float32))
        acc = mac(acc, a, w, "aligned")
        exact = mac(exact, a, w, "wide")
        ref += np.einsum("rxk,bkn->brxn", a.astype(np.float64), w.astype(np.float64))
    return acc, exact, ref


# Tap count is k*k*ncin: one multiply-accumulate per (ky, kx, input channel block of 8).
DEPTH_CASES = [
    ("k1 over 32 channels", 4, False),
    ("k1 over 64 channels", 8, False),
    ("k3 over 8 channels", 9, False),
    ("k3 over 32 channels", 36, False),
    ("k3 over 64 channels", 72, False),
    ("k3 over 128 channels", 144, False),
    ("k3 over 64, post-ReLU", 72, True),
    ("k3 over 128, post-ReLU", 144, True),
]


def depth(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for name, taps, positive in DEPTH_CASES:
        al, ex, ref = _accumulate(rng, taps, 0.1, positive)
        print("BF16_ACCUM_DEPTH " + json.dumps({
            "case": name, "taps": taps, "post_relu": positive,
            "aligned_relerr": _rel(al, ref), "exact_relerr": _rel(ex, ref),
            "stored_lanes_differ": _stored_differ(al, ex),
        }, sort_keys=True), flush=True)


def cancel(seed: int) -> None:
    """Drive the accumulator to M, cancel it in one instruction, leave r. Sweep M/r."""
    del seed                                   # this construction is exact, not sampled
    for power in range(4, 30, 2):
        M = np.float32(2.0 ** 10)
        r = np.float32(M / 2.0 ** power)
        acc = np.full((NCO, R, C, 4), M, np.float32)
        a = np.zeros((R, C, 8), np.float32)
        w = np.zeros((NCO, 8, 4), np.float32)
        a[..., 0], w[:, 0, :] = 1.0, -M        # cancels the accumulator exactly
        a[..., 1], w[:, 1, :] = 1.0, r         # the surviving result
        a, w = to_bf16(a), to_bf16(w)
        al = to_bf16(mac(acc.copy(), a, w, "aligned"))
        ex = to_bf16(mac(acc.copy(), a, w, "wide"))
        print("BF16_ACCUM_CANCEL " + json.dumps({
            "ratio_log2": power, "result": float(r),
            "aligned_stores": float(al[0, 0, 0, 0]), "exact_stores": float(ex[0, 0, 0, 0]),
            "stored_lanes_differ": _stored_differ(al, ex),
        }, sort_keys=True), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--cancel", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.all or args.depth:
        depth(args.seed)
    if args.all or args.cancel:
        cancel(args.seed)
    if not (args.all or args.depth or args.cancel):
        ap.error("give --depth, --cancel or --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
