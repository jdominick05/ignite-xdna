"""Predict a tiled-GEMM kernel's cycles from its machine code, before running it.

Two measurements on this machine make this possible and neither is an estimate:

  * A hardware-loop body's bundle count IS its cycle count on AIE2, which is a
    statically scheduled VLIW with an exposed pipeline (results/aie/aie2_isa_static.log).
  * A buffer handed to a core costs a constant number of cycles on top of its work,
    independent of how much work that is over a 4,096x range
    (results/aie/pmu_probe_npu.log).

Together they say a kernel's issuing time is computable from its object file. This tool
does that computation for the mm.cc-shaped tiled GEMM: count the MAC operations one call
must issue from the tile geometry, read how many the compiled loop issues per cycle,
add the once-per-call prologue and the per-buffer handoff, and multiply by the calls the
problem shape requires.

What it produces is an ISSUING-cycle prediction: how long the core would take if it were
never starved. Comparing that with a measured wall time gives the fraction of the
dispatch the core actually spent issuing, and one minus that fraction is the part of the
loss that lives outside the instruction schedule -- the number a stall trace has to
account for.

Usage:
    python tools/gemm_cost_model.py <kernel.o> --tile 64,64,64 --mac-dims 4,8,8 \\
        --shape 4096,2048,2048 --cores 16 --measured-us 7458.1

    # every default is the int8 whole_array design at its best tile:
    python tools/gemm_cost_model.py ~/.npu/cache/<key>/matmul_i8_i32_86901378.o \\
        --measured-us 7458.1

The mac dims are the compiler's, not a guess: `_MM_MAC_DIMS` in
`python/iron/kernels/linalg.py` gives AIE2 (4, 8, 8) for every int8 input, so one AIE2
`vmac` retires 4*8*8 = 256 int8 MACs. Run from the ironenv so Peano's llvm-objdump is
reachable through tools/aie_disasm.py.
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools import aie_disasm as ad  # noqa: E402

# Measured on this machine; see the module docstring for the backing logs.
CLOCK_GHZ = 1.7983

# results/aie/pmu_probe_npu.log measured a constant 717 cycles per buffer on top of the
# work, and decomposed it: 512 cycles were that probe's own trace-flush loop, 190 were
# its kernel prologue and epilogue, and about 15 were the ObjectFifo acquire and release.
# Only the last of those transfers to another kernel. This tool counts the GEMM kernel's
# OWN prologue and epilogue from its own disassembly, so adding the probe's 190 here
# would double-count it, and the probe's 512-cycle flush loop does not exist at all in
# production code. The default is therefore the acquire/release term alone, per fifo.
ACQUIRE_RELEASE_CYCLES = 15

# The pessimistic bound: charge the whole 717 anyway. Reported alongside the default so
# the sensitivity of the conclusion to this one transferred constant is visible.
FULL_BUFFER_COST = 717


def triple(s: str) -> tuple[int, int, int]:
    a, b, c = (int(x) for x in s.split(","))
    return a, b, c


def count_op(bundles, op: str) -> int:
    return sum(1 for b in bundles for f in b.live if f.split()[0].startswith(op))


def pick(sections, want: str):
    for s in sections:
        if s.name.endswith(want) or s.name == want:
            return s
    return None


def analyse_kernel(section, op: str):
    """Split a function into its steady-state loop and everything executed once."""
    loops = ad.find_loops(section)
    hot = max(loops, key=lambda L: count_op(L.bundles, op)) if loops else None
    if hot is None or count_op(hot.bundles, op) == 0:
        raise SystemExit(f"no loop in {section.name} issues a '{op}'")
    in_loop = count_op(hot.bundles, op)
    total = count_op(section.bundles, op)
    return {
        "loop": hot,
        "loop_bundles": hot.n_bundles,
        "op_in_loop": in_loop,
        "op_total": total,
        "op_peeled": total - in_loop,
        "once_bundles": len(section.bundles) - hot.n_bundles,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("obj", help="compiled kernel object (mm.cc's matmul_*.o)")
    ap.add_argument("--tile", type=triple, default=(64, 64, 64), metavar="m,k,n")
    ap.add_argument("--mac-dims", type=triple, default=(4, 8, 8), metavar="r,s,t")
    ap.add_argument("--shape", type=triple, default=(4096, 2048, 2048), metavar="M,K,N")
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--fn", default="matmul_i8_i32", help="vectorized matmul symbol")
    ap.add_argument("--zero-fn", default="zero_i32", help="output-tile zero symbol")
    ap.add_argument("--out-bytes", type=int, default=4, help="bytes per output element")
    ap.add_argument("--in-bytes", type=int, default=1, help="bytes per input element")
    ap.add_argument("--measured-us", type=float, default=None,
                    help="measured NPU-time bracket for this shape, microseconds")
    ap.add_argument("--clock-ghz", type=float, default=CLOCK_GHZ)
    ap.add_argument("--fifos", type=int, default=2,
                    help="input ObjectFifos acquired and released per call")
    ap.add_argument("--objdump", default=None)
    args = ap.parse_args(argv)

    m, k, n = args.tile
    r, s, t = args.mac_dims
    M, K, N = args.shape

    obj = os.path.expanduser(args.obj)
    sections = ad.parse(ad.disassemble(obj, ad.find_objdump(args.objdump)))
    mm = pick(sections, args.fn)
    if mm is None:
        raise SystemExit(f"{args.fn} not found in {obj}")

    a = analyse_kernel(mm, "vmac")
    macs_per_op = r * s * t
    ops_per_call = (m * k * n) // macs_per_op
    if (m * k * n) % macs_per_op:
        raise SystemExit("tile is not a whole number of MAC operations")

    iters = (ops_per_call - a["op_peeled"]) / a["op_in_loop"]
    loop_cycles = iters * a["loop_bundles"]
    call_cycles = loop_cycles + a["once_bundles"]

    print(f"== {obj}")
    print(f"   tile m={m} k={k} n={n}   mac dims {r}x{s}x{t} = {macs_per_op} MACs per vmac")
    print()
    print("the machine code")
    print(f"  {args.fn:<22} {len(mm.bundles):>7} bundles total")
    print(f"  steady-state loop      {a['loop_bundles']:>7} bundles = "
          f"{a['loop_bundles']} cycles per iteration")
    print(f"  vmac in that loop      {a['op_in_loop']:>7}  -> "
          f"{a['op_in_loop'] / a['loop_bundles']:.3f} vmac/cycle, "
          f"{100 * a['op_in_loop'] / a['loop_bundles']:.1f}% of one per cycle")
    print(f"  vmac peeled outside    {a['op_peeled']:>7}  (software-pipeline prologue "
          f"and epilogue)")
    print(f"  bundles executed once  {a['once_bundles']:>7}")
    print()

    zcycles = 0.0
    zsec = pick(sections, args.zero_fn)
    calls_per_out_tile = K // k
    if zsec is not None:
        zloops = ad.find_loops(zsec)
        if zloops:
            zl = max(zloops, key=lambda L: count_op(L.bundles, "vst"))
            vst = count_op(zl.bundles, "vst")
            # 256-bit vector store: wl/wh are the halves of a 512-bit x register.
            elems = vst * (256 // (8 * args.out_bytes))
            ziters = (m * n) / elems
            ztotal = ziters * zl.n_bundles + (len(zsec.bundles) - zl.n_bundles)
            zcycles = ztotal / calls_per_out_tile
            print(f"  {args.zero_fn:<22} {zl.n_bundles} bundles/iter, {vst} vst/iter -> "
                  f"{ztotal:.0f} cycles per {m}x{n} output tile,")
            print(f"  {'':<22} amortised over {calls_per_out_tile} calls = "
                  f"{zcycles:.1f} cycles per call")
            print()

    handoff = args.fifos * ACQUIRE_RELEASE_CYCLES
    per_call = call_cycles + zcycles + handoff
    # Pessimistic variant: charge the probe's whole per-buffer constant instead.
    per_call_hi = call_cycles + zcycles + args.fifos * FULL_BUFFER_COST
    calls = (M // m) * (N // n) * (K // k)
    per_core = calls / args.cores
    cycles = per_core * per_call
    us = cycles / (args.clock_ghz * 1000.0)

    print("one call, one buffer pair")
    print(f"  vmac the tile requires   {ops_per_call:>10}")
    print(f"  loop iterations          {iters:>10.1f}")
    print(f"  loop cycles              {loop_cycles:>10.0f}")
    print(f"  prologue/epilogue        {a['once_bundles']:>10}   (from this object's own code)")
    print(f"  zero, amortised          {zcycles:>10.1f}")
    print(f"  handoff, {args.fifos} fifos        {handoff:>10}   "
          f"(acquire+release only; see the note in this file)")
    print(f"  ISSUING CYCLES PER CALL  {per_call:>10.0f}")
    print(f"  MAC issue rate           {macs_per_op * ops_per_call / per_call:>10.1f} "
          f"MACs/cycle over the whole call, "
          f"{100 * macs_per_op * ops_per_call / per_call / 256:.1f}% of 256")
    print(f"  pessimistic variant      {per_call_hi:>10.0f}   cycles/call if the probe's "
          f"full {FULL_BUFFER_COST} is charged per fifo")
    print()

    print(f"the problem: {M}x{K}x{N} on {args.cores} cores at {args.clock_ghz} GHz")
    print(f"  calls total              {calls:>10}")
    print(f"  calls per core           {per_core:>10.0f}")
    print(f"  predicted issuing cycles {cycles:>10.0f}")
    print(f"  predicted time           {us:>10.1f} us   (never starved)")

    in_bytes_per_call = (m * k + k * n) * args.in_bytes
    print(f"  input bytes per call     {in_bytes_per_call:>10}")
    # Bytes ARRIVING IN A CORE'S L1. The aggregate is the L1-side total across cores and
    # is not a DRAM figure: whole_array broadcasts each A tile along a row and each B tile
    # down a column, so off-chip traffic is a fraction of this.
    print(f"  demand at that rate      {in_bytes_per_call / per_call:>10.2f} B/cycle "
          f"into one core's L1, {args.cores * in_bytes_per_call / per_call:.1f} B/cycle "
          f"summed over {args.cores} cores")

    if args.measured_us is not None:
        meas_cycles = args.measured_us * args.clock_ghz * 1000.0
        frac = cycles / meas_cycles
        frac_hi = per_core * per_call_hi / meas_cycles
        meas_per_call = meas_cycles / per_core
        achieved_gops = 2 * M * K * N / (args.measured_us * 1e-6) / 1e9
        peak_gops = 2 * 256 * args.cores * args.clock_ghz
        print()
        print("against the measurement")
        print(f"  measured NPU time        {args.measured_us:>10.1f} us")
        print(f"  measured cycles/core     {meas_cycles:>10.0f}")
        print(f"  measured cycles/call     {meas_per_call:>10.0f}")
        print(f"  CORE ISSUING FRACTION    {100 * frac:>10.1f} %   "
              f"(pessimistic handoff: {100 * frac_hi:.1f} %)")
        print(f"  not issuing              {100 * (1 - frac):>10.1f} %  "
              f"({meas_per_call - per_call:.0f} cycles per call)")
        print(f"  achieved input rate      "
              f"{in_bytes_per_call / meas_per_call:>10.2f} B/cycle per core")
        print()
        print("  closure check -- the two factors must multiply to the measured rate")
        print(f"  schedule    {100 * macs_per_op * ops_per_call / per_call / 256:>6.1f} %"
              f"  x  issuing {100 * frac:>5.1f} %"
              f"  =  {100 * macs_per_op * ops_per_call / per_call / 256 * frac:>5.1f} %")
        print(f"  measured {achieved_gops:.1f} GOPS / peak {peak_gops:.0f} GOPS"
              f"  =  {100 * achieved_gops / peak_gops:>5.1f} %")
        print()
        print("  PREDICTION: if a port trace measures a sustained input rate at or below")
        print(f"  {in_bytes_per_call / meas_per_call:.2f} B/cycle into a core, this GEMM is "
              "bandwidth-bound and the")
        print(f"  {100 * (1 - frac):.0f}% is fully explained. If the ports read faster than "
              "that, it is not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
