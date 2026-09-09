"""Check that AdaRound's cpu path is bit-for-bit unchanged by the device refactor.

Ignition's AdaRound is byte-identical to a fresh Quark XINT8_ADAROUND oracle, and that
property is the reason the module exists. Adding an OptimDevice therefore has exactly one
dangerous failure mode: silently perturbing the cpu path -- a reordered constant, a moved
tensor construction, an RNG draw in a different place -- so that the oracle match is lost
without anything raising.

This runs the transcription on a synthetic QDQ Conv and writes the emitted integer weights
plus the loop's own trace (per-iteration losses, recon metrics, changed-weight counts). Run
it against the OLD code to record a reference, then against the NEW code to compare:

    # reference, from a checkout that predates the change
    PYTHONPATH=<old checkout> python tools/quant_adaround_device_checks.py --write ref.npz
    # this checkout
    python tools/quant_adaround_device_checks.py --compare ref.npz

Needs torch, so run it in resnet_env, not resnet_env17. It builds no InferenceSession on the
NPU -- the only onnxruntime use is Quark's own CPU activation extraction -- so it is safe
under peer device load. It is small by construction (a 2-channel 4x4 Conv, a handful of
iterations), not a producer run.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# --repo picks WHICH checkout `quant` is imported from, so one copy of this script can run
# the pre-change code in another worktree and the post-change code here. The fixtures
# always come from this file's own directory.
_REPO = Path(sys.argv[sys.argv.index("--repo") + 1]).resolve() if "--repo" in sys.argv else ROOT
sys.path.insert(0, str(_REPO))
# See quant_shift_cut_checks.py: site-packages ships an unrelated `tools` package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_fixtures as fx


def run(device: str, iterations: int, data_size: int, batch_size: int, seed: int):
    """Run the transcription once and return (weights, trace, report_fields)."""
    from quant.adaround import FastFinetuneConfig, finetune
    from quant.graph import Graph

    float_model, quant_model = fx.adaround_pair()
    float_graph, quant_graph = Graph(float_model), Graph(quant_model)
    images = fx.images(data_size)

    kwargs = dict(DataSize=data_size, FixedSeed=seed, NumIterations=iterations, BatchSize=batch_size)
    if device != "cpu":
        # Only the post-refactor config carries these; a reference run is always cpu.
        kwargs.update(OptimDevice=device, AllowNonParityDevice=True)
    cfg = FastFinetuneConfig(**kwargs)

    lines = []
    report = finetune(float_graph, quant_graph, images, cfg, log=lines.append)
    weights = quant_graph.initializer("W_quantized")
    fields = {
        "layers": len(report.layers),
        "iterations": [l.iterations for l in report.layers],
        "early_stop": [bool(l.early_stop) for l in report.layers],
        "recons_before": [float(l.recons_before) for l in report.layers],
        "recons_after": [float(l.recons_after) for l in report.layers],
        "changed_elements": [int(l.changed_elements) for l in report.layers],
        "max_lsb": [int(l.max_lsb) for l in report.layers],
    }
    # The loop's own log, minus three kinds of line that cannot be compared across runs:
    # wall-clock timings, and Ignition's own banner/warning lines. The banner is this
    # project's telemetry, not transcribed Quark output, so extending it is a deliberate
    # change rather than a parity break -- but everything Quark itself prints ("Module ...",
    # "adaround iterations=...", "LAYER ...") stays in and is compared exactly, because a
    # drift there means the optimisation itself moved.
    def transcribed(line: str) -> bool:
        return not (line.startswith("ADAROUND")
                    or "latency_profiler" in line
                    or line.startswith("ONNX inference costs"))

    trace = [l for l in lines if transcribed(l)]
    return np.asarray(weights), trace, fields


def check_guards():
    """The fail-closed boundaries: leaving the cpu must be asked for, and must be possible."""
    from quant.adaround import FastFinetuneConfig

    failures = []

    def expect(label, fn, exc, needle):
        try:
            fn()
        except exc as e:
            if needle in str(e):
                print(f"  PASS  {label}")
                return
            failures.append(f"{label}: {exc.__name__} raised but did not mention {needle!r}: {e}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{label}: raised {type(e).__name__}, want {exc.__name__}: {e}")
            return
        else:
            failures.append(f"{label}: did not raise")

    print("a non-cpu OptimDevice must be acknowledged")
    expect("cuda without AllowNonParityDevice is refused",
           lambda: FastFinetuneConfig(OptimDevice="cuda").check(),
           NotImplementedError, "byte-parity")
    print("  PASS  cpu is accepted with no acknowledgement"
          if FastFinetuneConfig().check() is None else "")
    try:
        FastFinetuneConfig(OptimDevice="cuda", AllowNonParityDevice=True).check()
        print("  PASS  cuda with AllowNonParityDevice passes check()")
    except Exception as e:  # noqa: BLE001
        failures.append(f"cuda with AllowNonParityDevice should pass check(): {e}")

    print("\nORT activation extraction stays on the cpu")
    expect("a non-cpu InferDevice is refused",
           lambda: FastFinetuneConfig(InferDevice="cuda", AllowNonParityDevice=True).check(),
           NotImplementedError, "cpu inference")

    print("\nan acknowledged device that torch cannot reach fails loudly, not silently on cpu")
    import torch
    if torch.cuda.is_available():
        print("  SKIP  this torch build has cuda; the unavailable-device path cannot be checked here")
    else:
        try:
            run("cuda", 2, 2, 1, 1705472343)
            failures.append("an unavailable cuda device did not raise")
        except RuntimeError as e:
            if "not available to this torch build" in str(e):
                print("  PASS  unavailable cuda raises RuntimeError naming the torch build")
            else:
                failures.append(f"unavailable cuda raised the wrong RuntimeError: {e}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"unavailable cuda raised {type(e).__name__}: {e}")

    print()
    for f in failures:
        print(f"  FAIL  {f}")
    print(f"{'FAILED' if failures else 'ALL GUARDS PASS'}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", type=Path, help="Record this run as the reference .json")
    parser.add_argument("--compare", type=Path, help="Compare this run against a reference .json")
    parser.add_argument("--device", default="cpu", help="OptimDevice to run (default cpu)")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--data-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1705472343)
    parser.add_argument("--repo", type=Path, default=ROOT,
                        help="Checkout to import quant/ from (default: this file's own). Point it at "
                             "a pre-change worktree to record the reference this run is compared to")
    parser.add_argument("--guards", action="store_true",
                        help="Check only the fail-closed boundaries around a non-cpu device")
    args = parser.parse_args()
    print(f"quant/ imported from {_REPO}")
    if args.guards:
        return check_guards()
    if not args.write and not args.compare:
        parser.error("Give --write to record a reference, --compare to check one, or --guards")

    weights, trace, fields = run(args.device, args.iters, args.data_size, args.batch_size, args.seed)
    print(f"device={args.device} iterations={args.iters} images={args.data_size} "
          f"batch={args.batch_size} seed={args.seed}")
    print(f"emitted W_quantized: shape={weights.shape} dtype={weights.dtype} "
          f"sum={int(weights.astype(np.int64).sum())}")
    for line in trace:
        print(f"  | {line}")

    # Plain JSON, no pickle: the fixture weight is a few dozen int8 values, and a reference
    # file that a later session has to trust should not be a code-execution vector.
    if args.write:
        args.write.write_text(json.dumps(
            {"dtype": str(weights.dtype), "shape": list(weights.shape),
             "weights": weights.reshape(-1).tolist(), "trace": trace, "fields": fields},
            indent=2), encoding="utf-8")
        print(f"\nreference written to {args.write}")
        return 0

    ref = json.loads(args.compare.read_text(encoding="utf-8"))
    ref_weights = np.array(ref["weights"], dtype=ref["dtype"]).reshape(ref["shape"])
    ref_trace = list(ref["trace"])
    ref_fields = ref["fields"]

    failures = []
    if ref_weights.shape != weights.shape or ref_weights.dtype != weights.dtype:
        failures.append(f"shape/dtype: {ref_weights.shape}/{ref_weights.dtype} -> "
                        f"{weights.shape}/{weights.dtype}")
    elif not np.array_equal(ref_weights, weights):
        differing = int(np.count_nonzero(ref_weights != weights))
        worst = int(np.abs(ref_weights.astype(np.int16) - weights.astype(np.int16)).max())
        failures.append(f"weights differ in {differing}/{weights.size} values, max |delta| = {worst} LSB")
    if ref_fields != fields:
        for key in sorted(set(ref_fields) | set(fields)):
            if ref_fields.get(key) != fields.get(key):
                failures.append(f"report.{key}: {ref_fields.get(key)!r} -> {fields.get(key)!r}")
    if ref_trace != trace:
        for i, (a, b) in enumerate(zip(ref_trace, trace)):
            if a != b:
                failures.append(f"trace line {i}:\n      was: {a}\n      now: {b}")
                break
        if len(ref_trace) != len(trace):
            failures.append(f"trace length: {len(ref_trace)} -> {len(trace)}")

    print()
    if failures:
        print(f"MISMATCH against {args.compare} ({len(failures)} difference(s)):")
        for f in failures:
            print(f"  - {f}")
        print("\nOn the cpu path this is a REGRESSION: the byte-parity property is lost.")
        return 1
    print(f"IDENTICAL to {args.compare}: emitted weights, report figures and loop trace all match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
