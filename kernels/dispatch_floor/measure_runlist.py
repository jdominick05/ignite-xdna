#!/usr/bin/env python3
"""Measure the per-dispatch cost amortisation with pyxrt.runlist batching.

WHY THIS EXISTS
---------------
`measure_floor.py` found that the per-dispatch fixed cost through IRON's @iron.jit
path is 617.0 us (169.8 us hardware, 447.3 us host-side), and that
`pyxrt.runlist` exists and is bound (add/execute/wait). This script answers D1 in
docs/SILICON.md: how much of that cost amortises when N dispatches are batched
into a single execute()+wait() round trip?

WHAT IT MEASURES
----------------
Three layers, each on the same no-compute passthrough design as measure_floor.py:

  raw_single   pyxrt raw kernel(3,...) + run.wait(), one dispatch at a time,
               NO IRON wrapper at all. This is the raw-pyxrt floor: it removes
               IRON's Python host-side cost and says what the pyxrt/XRT binding
               alone costs per call. If it lands near 169.8 us, IRON's 447 us is
               all host-side Python. If it's substantially higher, there's
               pyxrt-binding cost above the hardware bracket.

  runlist_N    pyxrt.runlist with N run objects added, one execute()+wait() for
               the batch. Wall time divided by N is the amortised per-dispatch
               cost. Sweep N = 1, 2, 4, 8, 16, 32, 64.

Both are compared against measure_floor.py's published numbers (617.0 / 169.8 us)
at the same payload size, so the three numbers (IRON, raw-single, runlist-amortised)
form a stack.

CORRECTNESS GATE
----------------
Every payload is verified (output == input) before its timings are kept. A
passthrough that silently did nothing would be meaningless.

APPROACH TO THE CACHE
---------------------
IRON compiles designs into ~/.npu/cache/<hash>/. Rather than hunting for the right
hash, this script compiles a fresh passthrough via IRON (same design as
measure_floor.py), then finds the compiled artifacts in the cache directory by
modification time. The IRON run itself is untimed — it is only used to ensure the
xclbin and instruction file exist.

USAGE (from the mlir-aie ironenv, NOT resnet_env17 -- see kernels/README.md)
    . C:\\Users\\<user>\\mlir-aie\\iron_env.ps1
    $env:PATH = "C:\\Xilinx\\XRT\\xrt_sdk\\xrt;$env:PATH"
    python kernels/dispatch_floor/measure_runlist.py
    python kernels/dispatch_floor/measure_runlist.py --iters 100 --batch-sizes 1,4,16,64
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Step 1: compile the passthrough design via IRON (same as measure_floor.py)    #
# --------------------------------------------------------------------------- #

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import AnyShimTile

LINE_SIZE = 1024  # transfer chunk, same as measure_floor.py


@iron.jit
def passthrough(a_in: In, _unused: In, c_out: Out, *, n: CompileTime[int] = 4096):
    """shim -> memtile -> shim. No compute tile; identical to measure_floor.py."""
    vector_ty = np.ndarray[(n,), np.dtype[np.int32]]
    line_ty = np.ndarray[(LINE_SIZE,), np.dtype[np.int32]]

    of_in = ObjectFifo(line_ty, name="in")
    of_out = of_in.cons().forward()

    def sequence(a, _, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [
            vector_ty,
            vector_ty,
            vector_ty,
            of_in.prod(tile=AnyShimTile),
            of_out.cons(tile=AnyShimTile),
        ],
    )
    return Program(iron.get_current_device(), rt).resolve_program()


def _capture_compiled_paths():
    """Context manager yielding a dict that IRON fills with the design's real paths.

    CompilableDesign.compile() returns (xclbin_path, inst_path) for the design being
    built, and CallableDesign._build_kernel calls it on the first use of a
    specialization in a process -- on a cache HIT too, because compile() is what
    consults the on-disk cache and returns the existing artifacts. Spying on it is
    therefore exact for both paths, and needs no private hash or path attribute.

    This replaces an mtime heuristic that produced a wrong answer; see the comment in
    compile_and_find_cache. Two other resolvers were tried first and are recorded here
    so they are not re-attempted: CallableDesign.compilable.compile() recompiles,
    because .compilable is the UNSPECIALIZED design and misses the cache (it then dies
    on xclbinutil not being on PATH); and importing this module under a synthetic name
    changes IRON's recipe hash, so the cache is missed that way too.
    """
    import contextlib

    from aie.utils.compile.jit.compilabledesign import CompilableDesign

    captured = {}

    @contextlib.contextmanager
    def _ctx():
        original = CompilableDesign.compile

        def spy(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            try:
                xclbin_path, inst_path = result
                captured["xclbin"] = Path(xclbin_path)
                captured["insts"] = Path(inst_path)
            except Exception:
                pass
            return result

        CompilableDesign.compile = spy
        try:
            yield captured
        finally:
            CompilableDesign.compile = original

    return _ctx()


def compile_and_find_cache(n):
    """Run the passthrough once via IRON to compile it, then find the xclbin
    and instruction file in ~/.npu/cache/. Returns (xclbin_path, insts_path, kernel_name).
    """
    a = iron.arange(1, n + 1, dtype=np.int32, device="npu")
    b = iron.zeros_like(a)
    c = iron.zeros_like(a)

    # Run once to trigger compilation and cache population, capturing the paths IRON
    # itself resolved for THIS design rather than inferring them afterwards.
    with _capture_compiled_paths() as captured:
        passthrough(a, b, c, n=n)

    # Verify the design actually works
    if not np.array_equal(c.numpy(), a.numpy()):
        raise RuntimeError("IRON passthrough failed verification — cannot trust the design")

    # Try to extract paths from IRON internals first (version-dependent)
    for attr in ("_last_npu_kernel", "_npu_kernel", "_cached_kernel"):
        npu_kernel = getattr(passthrough, attr, None)
        if npu_kernel is not None:
            xclbin_path = getattr(npu_kernel, "xclbin_path", None)
            insts_path = getattr(npu_kernel, "insts_path", None)
            kernel_name = getattr(npu_kernel, "kernel_name", None)
            if xclbin_path and insts_path and Path(xclbin_path).exists():
                return str(Path(xclbin_path).resolve()), str(Path(insts_path).resolve()), kernel_name

    # Resolve by WHICH ENTRY IRON JUST LOCKED, not by modification time.
    #
    # The previous rule here -- newest entry holding both final.xclbin and insts.bin --
    # is wrong and produced a WRONG ANSWER rather than an error on 2026-09-09: every
    # batch read FAILED VERIFICATION and the single-dispatch figure came out at 777.7 us
    # against the ~140 us this design costs. Its own comment claimed that requiring
    # insts.bin stopped "a GEMM, say" from being picked up, but EVERY IRON design writes
    # insts.bin, so the extra condition excluded nothing. Once the bank_placement and
    # gemm_reblock designs were compiled later the same day, "newest" was one of those,
    # and this harness happily timed a different design through the passthrough's
    # argument layout.
    #
    # The .lock diff is exact: IRON acquires the lock inside the design's own kernel_dir
    # on every compile() call, cache hit included, so exactly one entry moves.
    cache_dir = Path.home() / ".npu" / "cache"
    if not cache_dir.exists():
        sys.exit("Cannot find IRON cache at ~/.npu/cache/ — run measure_floor.py first")

    xclbin_path = captured.get("xclbin")
    insts_path = captured.get("insts")
    if xclbin_path and insts_path and xclbin_path.is_file() and insts_path.is_file():
        return str(xclbin_path.resolve()), str(insts_path.resolve()), None

    # Refuse rather than guess. A wrong design here is not a failed run, it is a
    # plausible number for the wrong thing.
    sys.exit(
        "Could not capture the design's compiled paths from IRON.\n"
        f"  captured: {captured}\n"
        "Refusing to fall back to 'newest entry in ~/.npu/cache/', which is exactly "
        "what timed the wrong design on 2026-09-09."
    )


# --------------------------------------------------------------------------- #
# Step 2: raw pyxrt submission (single and runlist)                             #
# --------------------------------------------------------------------------- #

def setup_pyxrt(xclbin_path, insts_path, kernel_name):
    """Load the compiled passthrough design via raw pyxrt. Returns the handles
    needed for kernel invocation."""
    import pyxrt

    device = pyxrt.device(0)
    xclbin = pyxrt.xclbin(xclbin_path)
    device.register_xclbin(xclbin)
    context = pyxrt.hw_context(device, xclbin.get_uuid())

    # Resolve the kernel name from the xclbin if not provided
    if kernel_name is None:
        kernels = xclbin.get_kernels()
        if not kernels:
            raise RuntimeError("No kernels found in xclbin")
        kernel_name = kernels[0].get_name()

    kernel = pyxrt.kernel(context, kernel_name)

    # Load instruction binary into a buffer object
    insts_data = np.fromfile(insts_path, dtype=np.uint32)
    insts_bo = pyxrt.bo(device, len(insts_data) * 4, pyxrt.bo.cacheable, kernel.group_id(1))
    insts_bo.write(insts_data.tobytes(), 0)
    insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    return device, context, kernel, insts_bo, insts_data


def make_buffers(device, kernel, n):
    """Create one set of input/output buffer objects for a single run."""
    import pyxrt

    bo_a = pyxrt.bo(device, n * 4, pyxrt.bo.host_only, kernel.group_id(3))
    bo_b = pyxrt.bo(device, n * 4, pyxrt.bo.host_only, kernel.group_id(4))
    bo_c = pyxrt.bo(device, n * 4, pyxrt.bo.host_only, kernel.group_id(5))
    return bo_a, bo_b, bo_c


def fill_input(bo_a, n):
    """Write arange(1, n+1) into the input buffer."""
    import pyxrt

    data = np.arange(1, n + 1, dtype=np.int32)
    bo_a.write(data.tobytes(), 0)
    bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def verify_output(bo_c, n):
    """Read the output buffer and check it matches arange(1, n+1)."""
    import pyxrt

    bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    out = np.frombuffer(bo_c.read(n * 4, 0), dtype=np.int32)
    expected = np.arange(1, n + 1, dtype=np.int32)
    return bool(np.array_equal(out, expected))


def time_raw_single(device, kernel, insts_bo, insts_data, n, iters, warmup):
    """Time N=1 raw pyxrt dispatches (no IRON, no runlist).
    Returns (wall_times_ms, verified)."""
    bo_a, bo_b, bo_c = make_buffers(device, kernel, n)
    fill_input(bo_a, n)

    # Warmup
    for _ in range(warmup):
        run = kernel(3, insts_bo, len(insts_data), bo_a, bo_b, bo_c)
        run.wait()

    verified = verify_output(bo_c, n)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run = kernel(3, insts_bo, len(insts_data), bo_a, bo_b, bo_c)
        run.wait()
        times.append((time.perf_counter() - t0) * 1e3)

    return times, verified


def time_runlist_batch(device, context, kernel, insts_bo, insts_data, n,
                       batch_size, iters, warmup):
    """Time batched dispatches via pyxrt.runlist.
    Returns (wall_times_ms, verified, per_run_ms)."""
    import pyxrt

    # Create batch_size separate buffer sets (each run needs its own to avoid
    # undefined behaviour from concurrent buffer access within a runlist).
    buf_sets = []
    for _ in range(batch_size):
        bo_a, bo_b, bo_c = make_buffers(device, kernel, n)
        fill_input(bo_a, n)
        buf_sets.append((bo_a, bo_b, bo_c))

    # Warmup: single dispatches to prime caches
    bo_a0, bo_b0, bo_c0 = buf_sets[0]
    for _ in range(warmup):
        run = kernel(3, insts_bo, len(insts_data), bo_a0, bo_b0, bo_c0)
        run.wait()

    # Verify at least the first buffer set
    verified = verify_output(bo_c0, n)
    if not verified:
        return [], False, float("nan")

    times = []
    for _ in range(iters):
        # Create a fresh runlist each iteration (the XRT API docs say the list
        # can be reused by calling execute() again, but creating a fresh one is
        # the safer path for a first measurement).
        rl = pyxrt.runlist(context)
        runs = []
        for bo_a, bo_b, bo_c in buf_sets:
            # Build the run WITHOUT starting it. `kernel(...)` in pyxrt both
            # creates and STARTS a run, so adding one to a runlist hands
            # execute() a run that is already in flight -- which is why every
            # batch failed verification before this was fixed. `pyxrt.run(kernel)`
            # plus set_arg leaves it unstarted for the runlist to launch.
            run = pyxrt.run(kernel)
            run.set_arg(0, 3)
            run.set_arg(1, insts_bo)
            run.set_arg(2, len(insts_data))
            run.set_arg(3, bo_a)
            run.set_arg(4, bo_b)
            run.set_arg(5, bo_c)
            rl.add(run)
            runs.append(run)

        t0 = time.perf_counter()
        rl.execute()
        rl.wait()
        wall_ms = (time.perf_counter() - t0) * 1e3
        times.append(wall_ms)
        # Drop the runlist before the next iteration allocates another. Leaving
        # these alive to be torn down at interpreter exit crashes with an access
        # violation on this driver.
        del runs
        del rl

    # Verify all outputs after the timed runs
    all_ok = all(verify_output(bo_c, n) for _, _, bo_c in buf_sets)

    per_run_mean = statistics.mean(times) / batch_size
    return times, all_ok, per_run_mean


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iters", type=int, default=100, help="timed iterations per measurement point")
    p.add_argument("--warmup", type=int, default=5, help="untimed warmup calls")
    p.add_argument("--payload", type=int, default=4096,
                   help="element count (int32), must be a multiple of 1024")
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64",
                   help="comma-separated batch sizes to sweep")
    args = p.parse_args()

    n = args.payload
    if n % LINE_SIZE:
        sys.exit(f"payload {n} is not a multiple of {LINE_SIZE}")

    batch_sizes = [int(s) for s in args.batch_sizes.split(",")]

    # Step 1: compile via IRON and extract paths
    print("Step 1: compiling passthrough via IRON (one-time, result cached)...")
    xclbin_path, insts_path, kernel_name = compile_and_find_cache(n)
    print(f"  xclbin: {xclbin_path}")
    print(f"  instr:  {insts_path}")
    print(f"  kernel: {kernel_name}")
    print(f"  cache:  {Path(xclbin_path).parent.name}   "
          f"(resolved by newest entry holding BOTH final.xclbin and insts.bin)")

    # Step 2: set up raw pyxrt
    print("\nStep 2: loading design via raw pyxrt...")
    device, context, kernel, insts_bo, insts_data = setup_pyxrt(
        xclbin_path, insts_path, kernel_name
    )
    kb_moved = n * 4 * 2 / 1024
    print(f"  loaded, payload = {n} int32 ({kb_moved:.0f} KB round-trip)")

    # Step 3: raw single dispatch baseline
    print(f"\nStep 3: raw pyxrt single dispatch (no IRON, no runlist)...")
    print(f"  iters={args.iters} warmup={args.warmup}\n")
    raw_times, raw_ok = time_raw_single(
        device, kernel, insts_bo, insts_data, n, args.iters, args.warmup
    )
    raw_mean = statistics.mean(raw_times)
    raw_min = min(raw_times)
    raw_stdev = statistics.stdev(raw_times) if len(raw_times) > 1 else 0
    print(f"  raw single:  mean {raw_mean:.4f} ms  min {raw_min:.4f} ms  "
          f"stdev {raw_stdev:.4f} ms  "
          f"{'VERIFIED' if raw_ok else '!! FAILED VERIFICATION !!'}")
    print(f"\n  Comparison stack (same {kb_moved:.0f} KB passthrough):")
    print(f"    IRON wall  617.0 us  (measure_floor.py)")
    print(f"    HW bracket 169.8 us  (measure_floor.py)")
    print(f"    raw pyxrt  {raw_mean * 1e3:.1f} us  <-- this measurement")

    # Step 4: runlist batching sweep
    print(f"\nStep 4: pyxrt.runlist batching sweep...")
    print(f"  batch sizes: {batch_sizes}")
    print(f"  iters={args.iters} warmup={args.warmup}\n")

    head = (f"{'N':>4}  {'total ms':>10}  {'total min':>10}  "
            f"{'per-run ms':>10}  {'per-run us':>10}  "
            f"{'vs raw':>8}  {'vs IRON':>8}  ok")
    print(head)
    print("-" * len(head))

    results = []
    for batch_n in batch_sizes:
        try:
            times, ok, per_run = time_runlist_batch(
                device, context, kernel, insts_bo, insts_data, n,
                batch_n, args.iters, args.warmup
            )
            if not times:
                print(f"{batch_n:>4}  FAILED VERIFICATION — excluded")
                continue
            total_mean = statistics.mean(times)
            total_min = min(times)
            per_run_mean = total_mean / batch_n
            per_run_us = per_run_mean * 1e3
            vs_raw = raw_mean / per_run_mean if per_run_mean > 0 else float("nan")
            iron_wall_us = 617.0
            vs_iron = iron_wall_us / per_run_us if per_run_us > 0 else float("nan")
            print(f"{batch_n:>4}  {total_mean:>10.4f}  {total_min:>10.4f}  "
                  f"{per_run_mean:>10.4f}  {per_run_us:>10.1f}  "
                  f"{vs_raw:>8.2f}x  {vs_iron:>8.2f}x  "
                  f"{'yes' if ok else 'NO'}")
            results.append({
                "N": batch_n, "total_mean": total_mean, "total_min": total_min,
                "per_run": per_run_mean, "per_run_us": per_run_us,
                "vs_raw": vs_raw, "vs_iron": vs_iron, "ok": ok,
            })
        except Exception as e:
            print(f"{batch_n:>4}  ERROR: {e}")

    # Summary
    print("\n=== SUMMARY ===\n")
    print(f"Design:  no-compute passthrough (shim->memtile->shim)")
    print(f"Payload: {n} int32 ({kb_moved:.0f} KB round-trip)")
    print(f"Machine: (record machine name here when running)")
    print(f"\nPer-dispatch cost stack:")
    print(f"  IRON @iron.jit wall : 617.0 us  (measure_floor.py, published)")
    print(f"  IRON HW bracket     : 169.8 us  (measure_floor.py, published)")
    print(f"  raw pyxrt single    : {raw_mean * 1e3:.1f} us  <-- THIS MEASUREMENT")
    if results:
        best = min(results, key=lambda r: r["per_run_us"])
        print(f"  runlist N={best['N']:>2} amort : {best['per_run_us']:.1f} us/dispatch  "
              f"({best['vs_raw']:.2f}x over raw single, "
              f"{best['vs_iron']:.2f}x over IRON)")
    print()
    print("WHAT THIS TELLS US:")
    if results:
        # Diagnose where the cost lives
        best_us = min(r["per_run_us"] for r in results)
        raw_us = raw_mean * 1e3
        if raw_us < 250:
            print(f"  raw pyxrt ({raw_us:.0f} us) is near the HW bracket (170 us),")
            print(f"  so most of IRON's 447 us host cost is Python/caching overhead.")
        elif raw_us < 500:
            print(f"  raw pyxrt ({raw_us:.0f} us) is between the HW bracket (170 us)")
            print(f"  and IRON wall (617 us), so there's pyxrt-binding cost beyond")
            print(f"  the hardware but less than IRON's full overhead.")
        else:
            print(f"  raw pyxrt ({raw_us:.0f} us) is near IRON wall (617 us),")
            print(f"  so IRON's Python overhead is small — the cost is in pyxrt/XRT.")

        if best_us < raw_us * 0.5:
            print(f"  Batching helps: {best_us:.0f} us amortised vs {raw_us:.0f} us single.")
            print(f"  The host round-trip cost amortises over the batch.")
        elif best_us < raw_us * 0.8:
            print(f"  Batching helps modestly: {best_us:.0f} us vs {raw_us:.0f} us.")
        else:
            print(f"  Batching does NOT help much: {best_us:.0f} us vs {raw_us:.0f} us.")
            print(f"  The per-dispatch cost is mostly hardware, not host overhead.")

    print("\nThe go/no-go threshold from measure_floor.py:")
    print(f"  current: an op must cost >617 us on the CPU to win through IRON")
    if results:
        best_us = min(r["per_run_us"] for r in results)
        print(f"  with D1: an op must cost >{best_us:.0f} us on the CPU to win through")
        print(f"           a batched raw-pyxrt path")

    # Tear the pyxrt handles down in dependency order. Left to the interpreter's
    # own exit-time collection they are destroyed in arbitrary order and this
    # driver faults with an access violation -- harmless, since every result has
    # already been printed, but it makes the script look like it crashed and
    # gives any log assembler a non-zero exit code to reject.
    del kernel
    del insts_bo
    del context
    del device


if __name__ == "__main__":
    main()
