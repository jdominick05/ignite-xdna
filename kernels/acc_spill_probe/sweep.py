"""How many live mmul accumulators fit in AIE2 registers before the compiler spills.

Compiles `acc_kernels.cc` once per live-accumulator count with Peano and reads the
result back with `tools/aie_disasm.py`. Compile only -- no NPU, no hardware context,
no IRON. The whole sweep takes a few seconds and can run while the device is busy.

Why it exists. `docs/DECISIONS.md` records "AIE2 has only 6 hardware accumulator
registers", inferred from one upstream kernel that spilled at 8 and was fixed by
capping at 4. `docs/SILICON.md` records "<=4 stays in registers ... TO VERIFY the exact
register count from the AIE-ML ISA rather than from that one kernel". The two disagree
and neither was measured directly. This sweeps the count and reads the answer off the
machine code.

What is measured. For each K the probe holds K live `aie::mmul<4,8,8,int8,int8,acc32>`
accumulators across a k-reduction loop, the same shape upstream's conv2dk3 uses, so the
register allocator cannot retire one early. Two signals come back: stack references
anywhere in the function, which are zero while everything stays in registers, and the
bundle count of the reduction loop body, which is that loop's cycle count because AIE2
is a statically scheduled VLIW (see `tools/aie_disasm.py`). Cycles per MAC is the
figure that decides how a kernel should be written.

The `-D__AIE_API_AIE_ADF_HPP__=1` flag predefines the include guard of aie_api's
graph-level ADF header so its body is skipped. That header includes `adf.h`, which
ships with Vitis and exists nowhere on this machine, and nothing in a Peano kernel uses
it. Compiling `clock_probe.cc` this way reproduces the object IRON itself built,
bundle for bundle and frame for frame, which is what makes the flag safe.

Usage:
    python kernels/acc_spill_probe/sweep.py
    python kernels/acc_spill_probe/sweep.py --max 12 --keep out/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import aie_disasm  # noqa: E402

CLANG = os.path.expanduser(
    "~/mlir-aie/ironenv/Lib/site-packages/llvm-aie/bin/clang++.exe"
)
INCLUDE = os.path.expanduser(
    "~/mlir-aie/ironenv/Lib/site-packages/mlir_aie/include"
)
SOURCE = os.path.join(_HERE, "acc_kernels.cc")


def compile_one(nacc: int, outdir: str, clang: str, include: str) -> str | None:
    obj = os.path.join(outdir, f"acc_n{nacc}.o")
    cmd = [
        clang,
        "--target=aie2-none-unknown-elf",
        "-std=c++20",
        "-O2",
        f"-DNACC={nacc}",
        "-D__AIE_API_AIE_ADF_HPP__=1",
        "-c",
        "-I",
        include,
        SOURCE,
        "-o",
        obj,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  NACC={nacc}: compile failed")
        print("   ", proc.stderr.strip().splitlines()[0] if proc.stderr else "")
        return None
    return obj


def analyse(obj: str, objdump: str) -> dict:
    sections = aie_disasm.parse(aie_disasm.disassemble(obj, objdump))
    body = [s for s in sections if "acc_probe" in s.name]
    s = body[0] if body else sections[0]
    # The reduction loop is the longest hardware loop in the function.
    loop = max(s.loops, key=lambda lp: lp.n_bundles) if s.loops else None
    acc = s.registers("acc")
    return {
        "bundles": len(s.bundles),
        "stack_refs": s.stack_traffic,
        "frame": s.frame_bytes,
        "loop_cycles": loop.n_bundles if loop else None,
        "loop_live": loop.n_live if loop else None,
        "acc_regs": acc,
        "n_acc_regs": len(acc),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max", type=int, default=10, help="highest accumulator count")
    ap.add_argument("--min", type=int, default=1, help="lowest accumulator count")
    ap.add_argument("--clang", default=CLANG, help="Peano clang++ for aie2")
    ap.add_argument("--include", default=INCLUDE, help="mlir-aie C++ header directory")
    ap.add_argument("--objdump", help="llvm-objdump that knows elf32-aie")
    ap.add_argument("--keep", help="directory to keep the objects in")
    args = ap.parse_args(argv)

    if not os.path.exists(args.clang):
        raise SystemExit(f"Peano clang++ not found at {args.clang}")
    objdump = aie_disasm.find_objdump(args.objdump)

    outdir = args.keep or tempfile.mkdtemp(prefix="acc_spill_")
    os.makedirs(outdir, exist_ok=True)

    print("Live aie::mmul<4,8,8,int8,int8,acc32> accumulators vs register pressure")
    print(f"  source   {SOURCE}")
    print(f"  clang    {args.clang}")
    print(f"  objects  {outdir}")
    print()
    header = (
        f"{'live acc':>8}  {'acc regs':>8}  {'loop cycles':>11}  {'cycles/MAC':>10}  "
        f"{'stack refs':>10}  {'frame B':>7}  {'bundles':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for n in range(args.min, args.max + 1):
        obj = compile_one(n, outdir, args.clang, args.include)
        if obj is None:
            continue
        r = analyse(obj, objdump)
        cyc = r["loop_cycles"]
        per = (cyc / n) if cyc else float("nan")
        rows.append((n, r, per))
        print(
            f"{n:>8}  {r['n_acc_regs']:>8}  {cyc if cyc is not None else '-':>11}  "
            f"{per:>10.2f}  {r['stack_refs']:>10}  "
            f"{r['frame'] if r['frame'] is not None else '-':>7}  "
            f"{r['bundles']:>7}"
        )

    clean = [n for n, r, _ in rows if r["stack_refs"] == 0]
    print()
    if clean:
        print(f"Highest count with zero stack traffic: {max(clean)}")
    spilled = [n for n, r, _ in rows if r["stack_refs"] > 0]
    if spilled:
        print(f"First count with any stack traffic:    {min(spilled)}")
    widest = max(rows, key=lambda t: t[1]["n_acc_regs"]) if rows else None
    if widest:
        names = " ".join(widest[1]["acc_regs"])
        print(
            f"Most accumulator registers allocated:  {widest[1]['n_acc_regs']} "
            f"at {widest[0]} live accumulators ({names})"
        )
    clean_rows = [t for t in rows if t[1]["stack_refs"] == 0 and t[2] == t[2]]
    if clean_rows:
        best = min(clean_rows, key=lambda t: t[2])
        print(
            f"Cheapest cycles per MAC without spill: {best[2]:.2f} "
            f"at {best[0]} accumulators"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
