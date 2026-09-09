"""Do a kernel's operands collide in a memory bank, and does its code care?

An AIE2 core tile has 64 KB of local data memory in **four** banks of 16 KB
(`getLocalMemorySize` 0x10000 and `getNumBanks` 4 in mlir-aie's own
`AIETargetModel.h`). A bundle can issue two loads, one on each memory port -- the
`a` and `b` suffixes of `vlda`/`vldb`. When both of those loads address the same
bank, the pair costs one extra cycle.

That penalty is measured, not assumed. `results/aie/memory_desktop2_20260909_*.log`
holds a controlled experiment: identical compiled function bytes, only the operand
addresses changed, fitted over a length sweep with r-squared 1.0.

    two loads, same bank        12.0 cycles/iteration
    two loads, separate banks   11.0
    one load,  same bank        11.0
    sub-bank offset 64/128/256  no effect -- the granularity is the bank

So the second load is free across banks and costs a cycle inside one. This tool
looks for that situation statically: it reads the allocated buffer addresses out of
a core ELF, assigns each to a bank, disassembles a kernel object, counts the bundles
that issue two loads, and reports where a collision meets a paired load. It also
says which banks are empty, because an empty bank is the fix.

Usage:
    python tools/aie_bank_check.py <build-cache-dir>
    python tools/aie_bank_check.py <build-cache-dir> --kernel matmul_i8_i32
    python tools/aie_bank_check.py --elf <core.elf> --obj <kernel.o>

A build cache directory is one of `~/.npu/cache/<key>/`, holding `elfs_main_core_*`
subdirectories and the kernel objects. Run from the ironenv so Peano's `llvm-nm` and
`llvm-objdump` are reachable.

This is a static check and reports a HAZARD, never a measured cost. Whether a given
paired load actually addresses two colliding buffers depends on which pointers it
uses, which this tool does not resolve; it reports the collision and the paired-load
count separately so the two are not confused.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools import aie_disasm as ad  # noqa: E402

# mlir-aie AIETargetModel.h: AIE2 core tile local memory 0x10000, getNumBanks() 4.
LOCAL_MEMORY = 0x10000
NUM_BANKS = 4
BANK_SIZE = LOCAL_MEMORY // NUM_BANKS

STACK_SYM = "_sp_start_value_DM_stack"
NM_RE = re.compile(r"^([0-9a-fA-F]+)\s+\S*\s*([A-Za-z])\s+(\S+)$")

# The two memory ports. A bundle naming one of each issues two loads in one cycle.
PORT_A = re.compile(r"^v?ld[a](?:\.|$)|^v?lda\b")
PORT_B = re.compile(r"^v?ld[b](?:\.|$)|^v?ldb\b")


def find_tool(name: str) -> str:
    for base in (
        os.path.expanduser("~/mlir-aie/ironenv/Lib/site-packages/llvm-aie/bin"),
        os.path.expanduser("~/mlir-aie/ironenv/lib/site-packages/llvm-aie/bin"),
    ):
        for cand in (os.path.join(base, name + ".exe"), os.path.join(base, name)):
            if os.path.exists(cand):
                return cand
    return name


def read_symbols(elf: str) -> list[tuple[str, int]]:
    out = subprocess.run([find_tool("llvm-nm"), "--numeric-sort", elf],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"llvm-nm failed on {elf}:\n{out.stderr[:600]}")
    syms = []
    for line in out.stdout.splitlines():
        m = NM_RE.match(line.strip())
        if not m:
            continue
        addr, kind, name = int(m.group(1), 16), m.group(2), m.group(3)
        if kind.upper() in ("T", "U", "W"):
            continue
        syms.append((name, addr))
    return syms


def own_window(syms: list[tuple[str, int]]) -> int | None:
    """The core's OWN 64 KB window, identified by the stack symbol living in it.

    A core also addresses its neighbours' memories, which appear in the same symbol
    table at other 64 KB windows. Only the core's own window has the stack.
    """
    for name, addr in syms:
        if name == STACK_SYM:
            return addr & ~(LOCAL_MEMORY - 1)
    return None


def paired_load_bundles(bundles) -> list:
    out = []
    for b in bundles:
        ops = [f.split()[0] for f in b.live]
        if any(PORT_A.match(o) for o in ops) and any(PORT_B.match(o) for o in ops):
            out.append(b)
    return out


def report_banks(elf: str) -> dict:
    syms = read_symbols(elf)
    base = own_window(syms)
    print(f"  {os.path.basename(elf)}")
    if base is None:
        print("    no stack symbol; cannot identify the core's own memory window")
        return {}
    print(f"    own local memory window 0x{base:05x}-0x{base + LOCAL_MEMORY - 1:05x}, "
          f"{NUM_BANKS} banks of {BANK_SIZE // 1024} KB")
    banks: dict[int, list[tuple[str, int]]] = {i: [] for i in range(NUM_BANKS)}
    for name, addr in syms:
        if not (base <= addr < base + LOCAL_MEMORY):
            continue
        if name == STACK_SYM:
            continue
        banks[(addr - base) // BANK_SIZE].append((name, addr))
    for i in range(NUM_BANKS):
        entries = banks[i]
        if not entries:
            print(f"    bank {i}: EMPTY")
            continue
        print(f"    bank {i}: {len(entries)}")
        for name, addr in entries:
            print(f"      0x{addr:05x}  {name}")
    return banks


def operand_collision(banks: dict) -> list[tuple[int, list[str]]]:
    """Banks holding two or more DISTINCT operands.

    The two halves of one double buffer do not count. A paired load reads two
    different operands; the halves are alternate incarnations of one operand and are
    never both read by the same iteration, so `X_buff_0` and `X_buff_1` sharing a
    bank is not a paired-load hazard.
    """
    hits = []
    for i, entries in banks.items():
        roots = {re.sub(r"_buff_\d+$", "", n) for n, _ in entries}
        if len(roots) > 1:
            hits.append((i, sorted(roots)))
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cache", nargs="?", help="a build cache directory")
    ap.add_argument("--elf", help="a single core ELF (instead of a cache directory)")
    ap.add_argument("--obj", action="append", default=[],
                    help="a kernel object to disassemble; repeatable")
    ap.add_argument("--kernel", help="only report this function")
    ap.add_argument("--objdump", default=None)
    args = ap.parse_args(argv)

    elfs, objs = [], list(args.obj)
    if args.cache:
        cache = os.path.expanduser(args.cache)
        elfs = sorted(glob.glob(os.path.join(cache, "elfs_main_core_*", "*.elf")))
        objs += sorted(glob.glob(os.path.join(cache, "*.o")))
    if args.elf:
        elfs.append(os.path.expanduser(args.elf))
    if not elfs and not objs:
        raise SystemExit("nothing to check: give a cache directory, or --elf/--obj")

    print("WHERE THE BUFFERS LANDED")
    banks = {}
    for elf in elfs[:1] if args.cache else elfs:
        banks = report_banks(elf)
    if len(elfs) > 1:
        print(f"  ({len(elfs)} core ELFs in this build; the first is shown, and the "
              f"allocation is per core)")
    print()

    hits = operand_collision(banks) if banks else []
    print("BANK SHARING")
    if not banks:
        print("  no bank map")
    elif not hits:
        print("  no bank holds more than one buffer")
    else:
        for i, roots in hits:
            print(f"  bank {i} holds {len(roots)} distinct buffers: {', '.join(roots)}")
    empty = [i for i, e in banks.items() if not e] if banks else []
    if empty:
        print(f"  banks with nothing in them: {empty}")
    print()

    print("BUNDLES THAT ISSUE TWO LOADS")
    total_paired = 0
    for obj in objs:
        sections = ad.parse(ad.disassemble(obj, ad.find_objdump(args.objdump)))
        for sec in sections:
            if args.kernel and not sec.name.endswith(args.kernel):
                continue
            loops = ad.find_loops(sec)
            pf = paired_load_bundles(sec.bundles)
            if not pf and not loops:
                continue
            print(f"  {sec.name}: {len(pf)} of {len(sec.bundles)} bundles")
            for lp in loops:
                lpp = paired_load_bundles(lp.bundles)
                total_paired += len(lpp)
                mark = "  <-- in the steady-state loop" if lpp else ""
                print(f"    loop {lp.name}: {len(lpp)} of {lp.n_bundles} bundles{mark}")
                for b in lpp:
                    print(f"      0x{b.addr:04x}  {' | '.join(b.live)}")
    print()

    print("VERDICT")
    if hits and total_paired:
        print(f"  HAZARD. {total_paired} paired-load bundle(s) inside hardware loops, and")
        print(f"  a bank holds more than one buffer. If a paired load reads two buffers in")
        print(f"  the same bank it costs one extra cycle per iteration, so a loop's bundle")
        print(f"  count understates its cycles by that much.")
        if empty:
            print(f"  Bank(s) {empty} are empty, so the collision is avoidable.")
    elif total_paired:
        print("  Paired loads exist but no bank holds two buffers; no collision to fix.")
    elif hits:
        print("  Buffers share a bank, but no loop issues two loads at once; no penalty.")
    else:
        print("  Neither condition holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
