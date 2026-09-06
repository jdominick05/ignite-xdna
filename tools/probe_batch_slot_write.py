"""Minimal, label-free reproduction of the batch>1 unwritten-slot bug
(docs/DECISIONS.md locked #1, filed as amd/RyzenAI-SW#401).

Generalizes to whatever static batch size N the given model was exported
with (read from the graph, never assumed): builds two batches where every
one of the N slots differs from its counterpart, runs both, and reports per
slot whether the output actually changed. No ImageNet, no accuracy
computation, no labels -- just numerical identity across two different
inputs. On a working batch>1 EP every slot must differ (different input ->
different output); a slot that stays identical across the two runs is an
unwritten/stale buffer, not a coincidence.

    python tools/probe_batch_slot_write.py --model models/resnet50_b2_xint8_c64.onnx
    python tools/probe_batch_slot_write.py --model models/resnet50_b4_xint8_c64.onnx

Only batch 2 has been run against real hardware so far (see
results/batch/slot_probe_b2.log) -- this script takes whatever static batch
size the --model was exported at, but no batch-4 (or higher) XINT8 export
has been produced or tested yet. Passing a batch-4 model here is how that
would be checked; nothing above assumes 2.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

from npu.session import build_session, clear_cache
from npu.paths import RESNET_CACHE_KEY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="a static batch-N quantized CNN, e.g. models/resnet50_b2_xint8_c64.onnx")
    ap.add_argument("--cache-key", default=RESNET_CACHE_KEY)
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.fresh:
        clear_cache(args.cache_key)

    rng = np.random.default_rng(args.seed)

    for ep in ("npu", "cpu"):
        sess = build_session(args.model, ep, args.cache_key, args.xclbin, log_severity=2)
        input_name = sess.get_inputs()[0].name
        shape = sess.get_inputs()[0].shape  # e.g. [N, 3, 224, 224]; N must be a fixed int
        n, chw = shape[0], shape[1:]
        if not isinstance(n, int):
            raise SystemExit(f"input batch dim is not static ({shape}) -- this probe "
                              "only makes sense for a genuinely static-batch export")

        # Every slot gets independent noise in both batches, so a slot that
        # matches across A and B can only be an unwritten/stale buffer, not
        # luck: with continuous random floats, two independent draws colliding
        # is a probability-zero event.
        batch_a = rng.standard_normal((n, *chw), dtype=np.float32)
        batch_b = rng.standard_normal((n, *chw), dtype=np.float32)

        out_a = sess.run(None, {input_name: batch_a})[0]
        out_b = sess.run(None, {input_name: batch_b})[0]

        print(f"\n=== {ep.upper()} EP, batch {n} ===")
        stale = []
        for i in range(n):
            same = np.array_equal(out_a[i], out_b[i])
            diff = np.abs(out_a[i].astype(np.float64) - out_b[i].astype(np.float64)).max()
            print(f"slot {i}: identical despite different input = {same}"
                  f"  (max abs diff {diff:.6f})"
                  + ("  <-- STALE/UNWRITTEN" if same else ""))
            if same:
                stale.append(i)
        if stale:
            print(f"stale slots: {stale} of {n}")
        else:
            print(f"no stale slots detected out of {n}")


if __name__ == "__main__":
    main()
