#!/usr/bin/env python3
"""SUPERSEDED positive control: can the AIE2 trace unit see a same-bank dual-load stall?

SUPERSEDED BY kernels/memory_placement/probe.py --events; ITS HEADLINE WAS RETRACTED.
This probe concluded MEMORY_STALL cannot see a bank conflict. That is wrong, and the fault
was this kernel rather than the event: the loop below runs at 15.0 cycles/iteration for 4
loads, far too loose for the compiler to be issuing two of them in ONE instruction, and the
same-bank penalty exists only for PAIRED loads. There was no conflict here to detect.
memory_placement/probe.py is the valid control because it ASSERTS the pairing -- it greps
the disassembly for `vlda` and `vldb` on one line and refuses to run otherwise -- and with
that assertion the event is exact: one same-bank paired load costs one cycle and raises one
MEMORY_STALL (results/aie/bank_stall_observable_npu.log). Read the three-outcome table below
knowing the middle row is what this probe reported and it was the wrong reading of it.
What still stands from this file: the three structural facts about .bss and stack_size, and
the eight-event trace-buffer trap, both recorded below.

WHY THIS EXISTS
---------------
results/aie/bank_ab_h12_npu.log separated the int8 GEMM's operands into different 16 KB
banks exactly as designed and could not resolve the effect -- seven alternating series,
the sign changing between them, because wall time on a shared machine cannot see a ~3%
core-side change. That log names the fix: read the trace unit's stall taxonomy from
inside the dispatch, where host contention cannot reach.

But results/aie/pmu_probe_npu.log established that only LOCK_STALL has ever read nonzero
on this machine, and stated that MEMORY_STALL, STREAM_STALL and CASCADE_STALL "are not
shown to work by this run. A kernel that uses them is needed before any of those three
can be trusted." A same-bank dual-load conflict should surface as MEMORY_STALL, and this
toolchain ships no BANK_CONFLICT event (nearest: GROUP_STALL 22, MEMORY_STALL 23,
DM_ACCESS_TO_UNAVAILABLE 66).

So tracing the GEMM's two arms first would be UNINTERPRETABLE: MEMORY_STALL = 0 in both
arms cannot distinguish "no conflict" from "the event does not capture bank conflicts".
This runs the control first, on a kernel built to collide as hard as it can.

  nonzero when colliding, ~zero when not   the instrument works AND the effect is real;
                                           tracing the GEMM's arms is then worth a sitting
  zero in both, but cycles differ          the stall is real, MEMORY_STALL does not see
                                           it; the GEMM trace is pointless until another
                                           event does
  zero in both, cycles match               no dual-load bank penalty at this access
                                           pattern; nothing for a GEMM trace to find

Every outcome is a result. None requires trusting an untested event.

HOW THE ARMS ARE MADE
---------------------
Not by a mode flag, which would compile two different code paths. The kernel always
issues the same two load streams, 2 KB apart, from one array.
The arms come from SLIDING both streams up inside one array by a compile-time index
offset, until the 2 KB gap between them straddles a 16 KB bank boundary. stack_size was
tried first, because bank_ab_h12 used it to move the GEMM's ObjectFifo buffers -- it does
NOT work here, and that is itself worth recording: at stack 0x400 and 0x1400 the streams
landed at 0x74100/0x76100 both times. The allocator lays ObjectFifo buffers above the
stack, but a kernel's own .bss is placed by the linker and the stack does not touch it.
An index offset cannot depend on any of that. The HOST classifies each run from the
addresses the kernel reports -- an arm is never assumed, it is read back.

USAGE (from the mlir-aie ironenv, NOT resnet_env17 -- see kernels/README.md)
    export PATH=/c/Users/<user>/mlir-aie/ironenv/Scripts:$PATH
    export PYTHONPATH=/c/Xilinx/XRT/xrt_sdk/xrt/python
    python3 kernels/bank_placement/bank_stall_probe.py
    python3 kernels/bank_placement/bank_stall_probe.py --target 16384 --repeats 3
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.trace import TraceConfig

# Reuse the PMU probe's harness and the clock probe's verified frame decoder rather than
# forking either: that decoder is the one validated against two loops whose cycles per
# iteration are already measured on this machine.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "pmu_probe"))
sys.path.insert(0, str(_HERE.parent / "clock_probe"))
import clock_probe as cp  # noqa: E402
import pmu_probe as pp  # noqa: E402

N_IO = 1024  # stream B reads 1 KB into this buffer for a 1 KB window: needs >= 512
BANK_BYTES = 16384
_KERNEL_SRC = _HERE / "bank_stall_kernels.cc"

# ACTIVE and LOCK_STALL are the two already validated on this machine; MEMORY_STALL and
# GROUP_STALL are the ones under test; INSTR_LOAD and INSTR_VECTOR say whether the loop
# issued what we think it issued. The trace unit takes at most 8 events per tile.
DEFAULT_EVENTS = [
    "INSTR_EVENT_0",
    "INSTR_EVENT_1",
    "ACTIVE",
    "MEMORY_STALL",
    "GROUP_STALL",
    "LOCK_STALL",
    "INSTR_LOAD",
    "INSTR_VECTOR",
]

# 1 KB is the default. Stepping by 4 KB across a 16 KB bank should land the 8 KB-separated
# streams both ways: same bank when (base mod 16384) < 8192, split otherwise.
# The relocation lever, in BYTES, applied as an index offset inside one array. Chosen
# against the base address measured on this machine (the streams landed at 0x74100, i.e.
# 1792 bytes into bank 29): with a 2 KB stream separation, pads up to ~12 KB keep both
# streams in one bank and ~12.5 KB pushes the second into the next. The harness never
# assumes this -- it classifies each run from the addresses the kernel reports.
DEFAULT_PADS = "0,4096,8192,12544"


@iron.jit
def bank_stall(a_in: In, c_out: Out, *, trace_size: CompileTime[int] = 0,
               pad_words: CompileTime[int] = 0,
               a_from_fifo: CompileTime[int] = 0,
               events: CompileTime[str] = ",".join(DEFAULT_EVENTS)):
    """kernels/pmu_probe's single-core design, with the dual-load kernel and a
    parameterised pad offset -- the lever that relocates the two load streams."""
    io_ty = np.ndarray[(N_IO,), np.dtype[np.int32]]
    probe = ExternalFunction(
        "bank_stall_probe",
        source_file=str(_KERNEL_SRC),
        object_file_name="bank_stall_kernels.o",
        arg_types=[io_ty, io_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[f"-DN_OUT={N_IO}", f"-DPAD_WORDS={pad_words}",
                       f"-DA_FROM_FIFO={a_from_fifo}"],
    )
    of_in = ObjectFifo(io_ty, name="in")
    of_out = ObjectFifo(io_ty, name="out")

    def core_fn(of_in, of_out, fn):
        e = of_in.acquire(1)
        o = of_out.acquire(1)
        fn(e, o)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        fn_args=[of_in.cons(), of_out.prod(), probe],
        tile=Tile(0, 2),
        trace=1 if trace_size > 0 else None,
    )

    def sequence(a, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [io_ty, io_ty, of_in.prod(tile=Tile(0, 0)), of_out.cons(tile=Tile(0, 0))],
    )
    prog = Program(iron.get_current_device(), rt, workers=[worker])
    if trace_size > 0:
        prog.enable_trace(
            trace_size=trace_size,
            workers=[worker],
            coretile_events=pp.resolve_events(events.split(",")),
        )
    return prog.resolve_program()


def bank_of(addr):
    """Which 16 KB bank a core-tile local address falls in."""
    return addr // BANK_BYTES


def run_one(a, c_out, target, arm, names, trace_size, tc):
    """One dispatch of one arm. None if the kernel's own echo check failed."""
    a[0] = 0
    a[1] = target
    bank_stall(a, c_out, trace_size=trace_size, pad_words=0,
               a_from_fifo=arm, events=",".join(names))
    out = np.asarray(c_out.numpy()).astype(np.int64)
    if int(out[1]) != target:
        return None

    row, col, bs = pp.core_byte_stream(tc)
    counts, first, last = pp.event_counts(bs, names)
    idx = {n: i for i, n in enumerate(names)}
    loop = None
    if idx.get("INSTR_EVENT_0") in first and idx.get("INSTR_EVENT_1") in first:
        loop = first[idx["INSTR_EVENT_1"]] - first[idx["INSTR_EVENT_0"]]

    addr_a = int(out[6]) & 0xFFFFFFFF
    addr_b = int(out[7]) & 0xFFFFFFFF
    return {
        "arm": arm,
        "cycles": loop,
        "counts": {n: counts.get(idx[n], 0) for n in names},
        "addr_a": addr_a,
        "addr_b": addr_b,
        "colliding": bank_of(addr_a) == bank_of(addr_b),
        "check": int(out[4]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", type=int, default=8192,
                    help="loop iterations (default 8192)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="passes over the stack sweep (default 3)")
    ap.add_argument("--arms", default="0,1",
                    help="0 = A from .bss (separated), 1 = A from the fifo (colliding)")
    ap.add_argument("--trace-size", type=int, default=65536)
    ap.add_argument("--events", default=",".join(DEFAULT_EVENTS))
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.events.split(",") if n.strip()]
    pp.resolve_events(names)  # fail early on a bad event name
    arms = [int(s, 0) for s in args.arms.split(",") if s.strip()]

    print(f"== bank_stall_probe  target={args.target}  repeats={args.repeats}")
    print("   events:", ", ".join(f"{i}:{n}" for i, n in enumerate(names)))
    print("   arms:", ", ".join("A-from-fifo" if x else "A-from-bss" for x in arms))
    print("   the arm is READ BACK from the reported addresses, never assumed")
    print(cp.platform_report())

    a = iron.zeros(N_IO, dtype=np.int32, device="npu")
    c_out = iron.zeros_like(a)
    trace_file = os.path.join(os.environ.get("TEMP", "/tmp"),
                              f"bank_stall_{os.getpid()}.txt")
    tc = TraceConfig(trace_size=args.trace_size, trace_file=trace_file)
    bank_stall.trace_config = tc

    show = [n for n in names if n not in ("INSTR_EVENT_0", "INSTR_EVENT_1")]
    rows = []
    for rep in range(args.repeats):
        # Reverse the order on alternate passes so a monotonic drift in the machine
        # cannot masquerade as an arm difference -- the shape bank_ab_h12 used.
        order = arms if rep % 2 == 0 else list(reversed(arms))
        for a_src in order:
            r = run_one(a, c_out, args.target, a_src, names, args.trace_size, tc)
            if r is None:
                print(f"rep {rep}  arm={a_src}  FAILED ECHO CHECK -- excluded")
                continue
            rows.append(r)
            arm = "COLLIDING" if r["colliding"] else "separated"
            print(f"rep {rep}  arm={a_src}  A=0x{r['addr_a']:05x} "
                  f"B=0x{r['addr_b']:05x}  bank {bank_of(r['addr_a'])}/"
                  f"{bank_of(r['addr_b'])} {arm:>9}  cycles={r['cycles']}  "
                  + "  ".join(f"{n}={r['counts'][n]}" for n in show))

    print("\n-- verdict --")
    coll = [r for r in rows if r["colliding"]]
    sep = [r for r in rows if not r["colliding"]]
    print(f"colliding runs: {len(coll)}   separated runs: {len(sep)}")
    if not coll or not sep:
        print("THE INTERVENTION DID NOT LAND: the stack sweep did not produce both a")
        print("colliding and a separated placement, so there is nothing to compare.")
        print("Check --arms; the addresses printed above say where each stream landed.")
        return 1

    ms_c = [r["counts"].get("MEMORY_STALL", 0) for r in coll]
    ms_s = [r["counts"].get("MEMORY_STALL", 0) for r in sep]
    gs_c = [r["counts"].get("GROUP_STALL", 0) for r in coll]
    gs_s = [r["counts"].get("GROUP_STALL", 0) for r in sep]
    cy_c = [r["cycles"] for r in coll if r["cycles"] is not None]
    cy_s = [r["cycles"] for r in sep if r["cycles"] is not None]
    print(f"MEMORY_STALL  colliding={ms_c}  separated={ms_s}")
    print(f"GROUP_STALL   colliding={gs_c}  separated={gs_s}")
    print(f"cycles        colliding={cy_c}  separated={cy_s}")

    # INSTRUCTION PARITY FIRST. A raw cycle difference means nothing unless both arms
    # issued the same instructions -- and here they do not: changing stream A's base from
    # .bss to the fifo buffer changes the codegen, roughly doubling the issued loads.
    # Comparing 15628 against 30730 cycles across that would attribute a scheduling
    # difference to the banks, which is precisely the error this repo's standing rule on
    # attribution forbids. Normalise by issued loads, and say plainly when the arms are
    # not comparable raw.
    ld_c = [r["counts"].get("INSTR_LOAD", 0) for r in coll]
    ld_s = [r["counts"].get("INSTR_LOAD", 0) for r in sep]
    print(f"INSTR_LOAD    colliding={ld_c}  separated={ld_s}")

    def per_load(rs):
        out = []
        for r in rs:
            n = r["counts"].get("INSTR_LOAD", 0)
            if r["cycles"] is not None and n:
                out.append(round(r["cycles"] / n, 4))
        return out

    pl_c, pl_s = per_load(coll), per_load(sep)
    print(f"cycles/load   colliding={pl_c}  separated={pl_s}")

    matched = bool(ld_c) and bool(ld_s) and min(ld_c) and min(ld_s) and         abs(min(ld_c) - min(ld_s)) / max(min(ld_c), min(ld_s)) < 0.05
    if not matched:
        print("\n" + "ARMS ARE NOT INSTRUCTION-MATCHED: the issued load counts differ by more")
        print("than 5%, so raw cycles are NOT comparable between them. Only the per-load")
        print("figures above can be read, and only as a weak check.")

    if max(ms_c) > 0 and max(ms_s) == 0:
        print("\n" + "INSTRUMENT VALIDATED and the effect is real: MEMORY_STALL sees the")
        print("conflict. Tracing the int8 GEMM's two arms is now worth a sitting.")
    elif max(ms_c) == 0 and max(ms_s) == 0:
        print("\n" + "MEMORY_STALL READS ZERO IN BOTH PLACEMENTS, including one built to")
        print("collide as hard as this design allows. Together with pmu_probe_npu.log")
        print("never seeing it nonzero, MEMORY_STALL CANNOT BE TRUSTED as a bank-conflict")
        print("signal on this machine, and tracing the GEMM's arms with it would be")
        print("uninterpretable -- zero there would not distinguish 'no conflict' from")
        print("'the event does not fire'. H12 stays unresolved and the instrument its own")
        print("log named as the fix does not work.")
        if pl_c and pl_s:
            d = abs(min(pl_c) - min(pl_s)) / max(min(pl_c), min(pl_s))
            if d < 0.02:
                print(f"Per issued load the two arms agree to {d*100:.2f}%, which is a weak")
                print("but real hint that there is no large dual-load bank penalty here.")
    else:
        print("\n" + "Mixed or unexpected: report the rows as measured, do not summarise past")
        print("them. MEMORY_STALL nonzero in the SEPARATED arm would mean the event is")
        print("counting something other than a bank conflict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
