"""Integer operands through the bf16 core's aligned multiply-accumulate: exact, and until when.

WHY THIS EXISTS. The bf16 engine was forked to run float models, and every argument for it so far
has been a float argument. The opposite question is cheaper and had never been asked: the int8
domain fits inside bf16 exactly, so can the bf16 datapath carry INTEGER work?

It can, and the reason is specific. `tools/bf16_accum_fidelity.py` records the measured core model
(results/aie/engine_bf16_mac_model_probe_npu_20260921.log): one `mmul<4,8,4>::mac` aligns its nine
operands to the largest exponent among them and rounds each separately to a 24-bit grid, ties to
even. That is not IEEE addition, and for general floats it costs precision, because a small operand
aligned to a large exponent rounds away. **An integer below 2^24 does not.** It sits exactly on the
24-bit grid at every alignment at or above its own exponent, so the aligned add - the whole reason
the CPU bf16 score is an upper bound rather than a prediction - is exact on integers.

WHAT BOUNDS IT. Only the running sum. uint8 activations and int8 weights are each exact in bf16
(8-bit significand covers [-256, 256]); their products reach 255 * 128 = 32,640, well inside the
grid; and every partial sum is exact while |sum| < 2^24 = 16,777,216. The psum buffer is float32
(`kernels/bf16_conv/design.py`, PSUM_BYTES = 6400 as 1,600 float32), so a chain across weight
packets does not round-trip through bf16 and the bound applies to the whole chain, not one packet.

WHAT THIS DOES NOT ESTABLISH, and it matters:

* **The OUTPUT STORE is bf16, not float32.** An accumulator above 256 does not survive the store.
  Carrying int8 end to end therefore needs the epilogue to bring the accumulator back into int8
  range before it stores - and `engine_bf16_emulator.epilogue` has no output scale today, only a
  floor at 0 and a ceiling at 6. This probe measures the ACCUMULATOR, not a working int8 path.
* This is the emulator's `mac()` in "aligned" mode, which is the measured model of the core, not
  the core. Device-vs-emulator byte-exactness is established separately
  (results/aie/engine_bf16_16core_npu_20260922.log).
* Uniform random operands are not a model's activations. Real post-ReLU activations are all
  positive and correlate with their weights, so a real layer's sums sit above the random case and
  below the adversarial one. The adversarial arm is the only guarantee here.

Offline. No device, no timing claim.

    bash scripts/research-lowlevel.sh --log results/aie/bf16_integer_exactness_<date>.log \\
        --checks-only -- bash scripts/research-iron.sh tools/bf16_integer_exactness.py --all
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
GRID = 2 ** 24       # the measured accumulator grid: integers below this are exact at any alignment

# Tap count is k*k*(ncin/8): one mac() per (ky, kx, input channel block of 8), so 8 MACs a tap.
SWEEP = [4, 8, 9, 36, 72, 144, 288, 576, 1152, 2304]


def accumulate(taps, a_lo, a_hi, w_lo, w_hi, adversarial, seed):
    """Accumulate `taps` integer MACs through the measured core model.

    Returns (max abs deviation from the exact integer sum, max abs value the sum reached).
    The reference is int64 - exact by construction, and sharing no code with the emulator.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((NCO, R, C, 4), np.float32)
    ref = np.zeros((NCO, R, C, 4), np.int64)
    for _ in range(taps):
        if adversarial:
            # every product at max magnitude with the same sign: the sum grows as fast as it can
            a = np.full((R, C, 8), a_hi, np.int64)
            w = np.full((NCO, 8, 4), w_hi, np.int64)
        else:
            a = rng.integers(a_lo, a_hi + 1, size=(R, C, 8)).astype(np.int64)
            w = rng.integers(w_lo, w_hi + 1, size=(NCO, 8, 4)).astype(np.int64)
        af, wf = to_bf16(a.astype(np.float32)), to_bf16(w.astype(np.float32))
        # If bf16 could not carry the operands this would be a test of the cast, not the adder.
        if not np.array_equal(af.astype(np.int64), a):
            raise SystemExit("activation operand is not exact in bf16")
        if not np.array_equal(wf.astype(np.int64), w):
            raise SystemExit("weight operand is not exact in bf16")
        acc = mac(acc, af, wf, "aligned")
        ref += np.einsum("rxk,bkn->brxn", a, w)
    dev = np.abs(acc.astype(np.float64) - ref.astype(np.float64)).max()
    return float(dev), int(np.abs(ref).max())


CASES = [
    # name, activation range, weight range, adversarial
    ("uint8_x_int8", (0, 255), (-128, 127), False),     # the real QDQ combination
    ("int8_x_int8", (-128, 127), (-128, 127), False),
    ("adversarial_max_same_sign", (255, 255), (127, 127), True),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="accepted for symmetry; every case runs")
    args = ap.parse_args()
    del args.all

    print("BF16_INT_EXACT_ENV " + json.dumps(
        {"numpy": np.__version__, "grid": GRID, "nco": int(NCO), "patch": [R, C],
         "taps_are": "8 MACs each (one input channel block)"}, sort_keys=True), flush=True)

    worst_exact_sum, first_break = 0, None
    for name, (a_lo, a_hi), (w_lo, w_hi), adversarial in CASES:
        for taps in SWEEP:
            dev, peak = accumulate(taps, a_lo, a_hi, w_lo, w_hi, adversarial, args.seed)
            exact = dev == 0.0
            if exact:
                worst_exact_sum = max(worst_exact_sum, peak)
            elif adversarial and first_break is None:
                first_break = {"taps": taps, "macs": taps * 8, "peak_sum": peak}
            print("BF16_INT_EXACT " + json.dumps(
                {"case": name, "taps": taps, "macs": taps * 8, "max_abs_deviation": dev,
                 "peak_abs_sum": peak, "exact": exact, "peak_over_grid": peak >= GRID},
                sort_keys=True), flush=True)

    print("BF16_INT_EXACT_VERDICT " + json.dumps(
        {"grid": GRID, "largest_sum_still_exact": worst_exact_sum,
         "first_adversarial_break": first_break,
         "rule": "integer operands accumulate exactly while |running sum| < 2^24; the bound is "
                 "the accumulator grid, not the operand width",
         "not_established": "the bf16 OUTPUT STORE carries only 8 significand bits, so an "
                            "accumulator above 256 needs an epilogue scale to survive it; "
                            "epilogue() has no output scale today"},
        sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
