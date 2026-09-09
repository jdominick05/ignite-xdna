"""Which special registers does Peano's AIE2 assembler accept as a `mov` source?

The disassembly of clock_kernels.o contains `mov r6, CORE_ID`, so the assembler does know
named special registers. If one of them is the tile timer, a module-level asm block can read
the cycle counter that get_cycles() and __builtin_readcyclecounter cannot.
"""
import os
import subprocess
import sys

CLANG = os.path.expanduser("~/mlir-aie/ironenv/Lib/site-packages/llvm-aie/bin/clang++.exe")
TMP = __import__("tempfile").mkdtemp(prefix="reg_probe_")

CANDIDATES = [
    # known-good control
    "CORE_ID",
    # timer guesses
    "TIMER", "TIMER_LOW", "TIMER_HIGH", "TM", "TM0", "TM1",
    "TIMER_L", "TIMER_H", "TIMERLOW", "TIMERHIGH",
    "CYCLE", "CYCLES", "CYCLE_LOW", "CYCLE_HIGH", "PERF0", "PERF_CNT0",
    # other plausible specials, to see the shape of the namespace
    "PC", "SP", "LR", "LC", "LS", "LE", "CR", "SR", "MC0", "MC1",
    "DC0", "DP", "S0", "S1", "DJ0", "DN0",
]


def try_reg(name):
    src = os.path.join(TMP, "rp.s")
    obj = os.path.join(TMP, "rp.o")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write(".text\n.globl t\nt:\n  mov r0, %s\n  ret lr\n  nop\n" % name)
    p = subprocess.run(
        [CLANG, "--target=aie2-none-unknown-elf", "-c", src, "-o", obj],
        capture_output=True, text=True,
    )
    return p.returncode == 0, (p.stderr or "").strip().splitlines()


ok, bad = [], []
for name in CANDIDATES:
    good, err = try_reg(name)
    (ok if good else bad).append(name)
    if good:
        print(f"  ACCEPTED  mov r0, {name}")
print()
print("accepted:", ", ".join(ok) if ok else "(none)")
print("rejected:", ", ".join(bad))
