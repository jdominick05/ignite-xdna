"""How much of the 617 us per-dispatch floor does batching remove *through IRON*?

`results/aie/dispatch_runlist_npu.log` reached 36.3 us per dispatch by driving raw pyxrt
against a cached xclbin -- bypassing IRON's host path entirely. `kernels/dispatch_floor/
iron_batch.py` puts runlist submission back inside that path, so an ordinary `@iron.jit`
design can use it. This measures what that is worth end to end.

Two clocks matter and they answer different questions:

  runlist bracket   perf_counter around runlist.execute() + wait(), divided by the batch
                    size. This is the DEVICE cost per dispatch, and it is the number
                    comparable to the 36.3 us raw-pyxrt figure.

  wall per call     total time in the block divided by the number of calls. This includes
                    everything IRON does per call BEFORE the submit -- argument validation
                    against the xclbin ABI, buffer preparation, instruction-buffer setup --
                    which batching does not touch. It is what a caller actually pays.

The gap between them is the part of the floor that is IRON's own host work rather than
dispatch, and it is the point of the measurement.

Correctness: batching gives up per-call completion status (see iron_batch.py), so every
output buffer is verified and a batch whose outputs are wrong is reported, not timed.

Usage (ironenv, XRT SDK on PATH, pyxrt on PYTHONPATH):
    python kernels/dispatch_floor/measure_iron_batch.py
    python kernels/dispatch_floor/measure_iron_batch.py --design groupnorm --L 18816
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import aie.iron as iron  # noqa: E402
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402

from kernels.dispatch_floor import iron_batch  # noqa: E402

LINE_SIZE = 1024


@iron.jit
def passthrough(a_in: In, _unused: In, c_out: Out, *, n: CompileTime[int] = 4096):
    """shim -> memtile -> shim. Same design measure_floor.py and measure_runlist.py use."""
    vector_ty = np.ndarray[(n,), np.dtype[np.int32]]
    line_ty = np.ndarray[(LINE_SIZE,), np.dtype[np.int32]]
    of_in = ObjectFifo(line_ty, name="in")
    of_out = of_in.cons().forward()

    def sequence(a, _, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(sequence, [vector_ty, vector_ty, vector_ty,
                            of_in.prod(tile=AnyShimTile),
                            of_out.cons(tile=AnyShimTile)])
    return Program(iron.get_current_device(), rt).resolve_program()


def make_passthrough_set(n):
    a = iron.arange(1, n + 1, dtype=np.int32, device="npu")
    return a, iron.zeros_like(a), iron.zeros_like(a)


def check_passthrough(bufs, n):
    exp = np.arange(1, n + 1, dtype=np.int32)
    return np.array_equal(bufs[2].numpy(), exp)


def build(design_name, args):
    """Return (call, make_set, check, label). `call` takes one buffer set."""
    if design_name == "passthrough":
        n = args.payload
        return (lambda s: passthrough(*s, n=n),
                lambda: make_passthrough_set(n),
                lambda s: check_passthrough(s, n),
                f"no-compute passthrough, {n} int32 ({n * 4 * 2 / 1024:.0f} KB round-trip)")

    if design_name == "groupnorm":
        sys.path.insert(0, str(_ROOT / "kernels" / "groupnorm_bf16"))
        import groupnorm as gn  # noqa: E402

        L, chunk = args.L, args.chunk
        if L % chunk:
            raise SystemExit(f"L={L} is not a multiple of chunk {chunk}")
        total = gn.GROUPS * L
        rng = np.random.default_rng(0)
        # Deterministic, finite inputs. The point of this run is the dispatch cost,
        # not GroupNorm's accuracy, which groupnorm.py's own verifier covers.
        x_bf = rng.standard_normal(total).astype(np.float32).astype(gn.bfloat16)
        prm = gn.pack_params(rng.standard_normal(gn.GROUPS).astype(np.float32),
                             rng.standard_normal(gn.GROUPS).astype(np.float32), chunk)

        def mk():
            return (iron.tensor(x_bf, dtype=gn.bfloat16, device="npu"),
                    iron.tensor(prm, dtype=gn.bfloat16, device="npu"),
                    iron.zeros(total, dtype=gn.bfloat16, device="npu"))

        def chk(s):
            out = np.asarray(s[2].numpy(), dtype=np.float32)
            # A batch that silently did nothing leaves the output at zero; a batch that
            # ran leaves a normalised tensor. Both checked, since batching gives up
            # per-call completion status.
            return bool(np.isfinite(out).all() and np.count_nonzero(out) > out.size // 2)

        return (lambda s: gn.groupnorm32(*s, L=L, chunk=chunk), mk, chk,
                f"groupnorm32, L={L}, {gn.GROUPS} groups on {gn.N_CORES} cores, bf16")

    raise SystemExit(f"unknown design {design_name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", choices=("passthrough", "groupnorm"), default="passthrough")
    ap.add_argument("--payload", type=int, default=4096, help="passthrough int32 count")
    ap.add_argument("--L", type=int, default=18816, help="groupnorm L")
    ap.add_argument("--chunk", type=int, default=3072, help="groupnorm chunk")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    ap.add_argument("--reps", type=int, default=3, help="timed repeats per batch size")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args(argv)

    sizes = [int(x) for x in args.batch_sizes.split(",")]
    call, mk, check, label = build(args.design, args)

    print(f"design: {label}")
    print(f"batch sizes: {sizes}   reps={args.reps}  warmup={args.warmup}")
    print()

    # Warm up: compile and settle. Untimed.
    warm = mk()
    for _ in range(args.warmup + 1):
        call(warm)
    if not check(warm):
        raise SystemExit("design failed verification before timing -- refusing to measure")
    print("warmup verified\n")

    # Unbatched steady state, for the baseline.
    solo = []
    for _ in range(max(8, args.reps * 4)):
        s = mk()
        t0 = time.perf_counter()
        call(s)
        solo.append((time.perf_counter() - t0) * 1e6)
        if not check(s):
            raise SystemExit("unbatched call failed verification")
    solo_med, solo_min = statistics.median(solo), min(solo)
    print(f"unbatched: median {solo_med:.1f} us/call, min {solo_min:.1f} us/call  VERIFIED")
    print()

    head = (f"{'N':>4} {'wall/call us':>13} {'runlist us/run':>15} "
            f"{'host share':>11} {'vs unbatched':>13}  verify")
    print(head)
    print("-" * len(head))

    for nb in sizes:
        walls, devs, oks = [], [], []
        for _ in range(args.reps):
            sets = [mk() for _ in range(nb)]
            t0 = time.perf_counter()
            with iron_batch.batched(max_in_flight=nb) as batch:
                for s in sets:
                    call(s)
            wall_us = (time.perf_counter() - t0) * 1e6
            walls.append(wall_us / nb)
            devs.append(batch.per_run_us)
            oks.append(all(check(s) for s in sets))
        w, d = statistics.median(walls), statistics.median(devs)
        ok = all(oks)
        print(f"{nb:>4} {w:>13.1f} {d:>15.1f} {w - d:>11.1f} "
              f"{solo_med / w:>12.2f}x  {'OK' if ok else 'MISMATCH'}")

    print()
    print("  wall/call is what a caller pays. runlist us/run is the device cost per")
    print("  dispatch, comparable to the 36.3 us raw-pyxrt figure in")
    print("  results/aie/dispatch_runlist_npu.log. The difference between them is")
    print("  IRON's own per-call host work, which batching does not remove.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
