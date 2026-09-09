"""Can hand-written AIE2 assembly be assembled and linked into a Peano kernel?

`docs/DECISIONS.md` records that Peano "rejects inline asm", which closed the idea of
hand-scheduling anything on this part. That is true of *statement-level* inline asm inside a
C++ function -- it dies in the IRTranslator, which is where S0 hit it -- but it says nothing
about a standalone assembly file, which never enters instruction selection at all and goes
straight to the integrated assembler.

This assembles `asm_core_id.s`, compiles a C++ caller, links the two, and asserts the symbol
resolves with nothing left undefined. Compile only, no NPU. If it passes, hand-scheduled AIE2
assembly is available on this machine and the closed door was only half closed.

`reg_probe.py` alongside answers the follow-on question: which special registers the assembler
will accept as a `mov` source. That is how the tile timer would be read from inside a kernel,
if it were exposed as one.

Usage:
    python kernels/asm_probe/check.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

BIN = os.path.expanduser("~/mlir-aie/ironenv/Lib/site-packages/llvm-aie/bin")
CLANG = os.path.join(BIN, "clang++.exe")
LLD = os.path.join(BIN, "ld.lld.exe")
NM = os.path.join(BIN, "llvm-nm.exe")
OBJDUMP = os.path.join(BIN, "llvm-objdump.exe")
HERE = os.path.dirname(os.path.abspath(__file__))
ASM = os.path.join(HERE, "asm_core_id.s")

CALLER = """extern "C" int asm_core_id();
extern "C" void caller(int *o) { o[0] = asm_core_id(); }
"""


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    for tool in (CLANG, LLD, NM, OBJDUMP):
        if not os.path.exists(tool):
            raise SystemExit(f"missing Peano tool: {tool}")

    tmp = tempfile.mkdtemp(prefix="asm_probe_")
    asm_o = os.path.join(tmp, "asm_core_id.o")
    cc = os.path.join(tmp, "caller.cc")
    cc_o = os.path.join(tmp, "caller.o")
    linked = os.path.join(tmp, "linked.o")
    with open(cc, "w", encoding="utf-8") as fh:
        fh.write(CALLER)

    failures = []

    rc, out = run([CLANG, "--target=aie2-none-unknown-elf", "-c", ASM, "-o", asm_o])
    print(f"assemble {os.path.basename(ASM)}: {'ok' if rc == 0 else 'FAILED'}")
    if rc != 0:
        print(out.strip()[:600])
        failures.append("assemble")

    if rc == 0:
        _, dis = run([OBJDUMP, "-d", "--no-show-raw-insn", asm_o])
        print("disassembly of the assembled object:")
        for line in dis.splitlines():
            if line.strip():
                print("   ", line)
        if "CORE_ID" not in dis:
            failures.append("disassembly lost the special-register read")

    rc, out = run([CLANG, "--target=aie2-none-unknown-elf", "-std=c++20", "-O2",
                   "-c", cc, "-o", cc_o])
    print(f"compile C++ caller: {'ok' if rc == 0 else 'FAILED'}")
    if rc != 0:
        print(out.strip()[:600])
        failures.append("compile caller")

    if not failures:
        rc, out = run([LLD, "-r", cc_o, asm_o, "-o", linked])
        print(f"link the two: {'ok' if rc == 0 else 'FAILED'}")
        if rc != 0:
            print(out.strip()[:600])
            failures.append("link")
        else:
            _, syms = run([NM, linked])
            print("symbols in the linked object:")
            for line in syms.splitlines():
                if line.strip():
                    print("   ", line)
            if " U " in syms:
                failures.append("undefined symbol after link")
            for want in ("asm_core_id", "caller"):
                if want not in syms:
                    failures.append(f"missing symbol {want}")

    print()
    if failures:
        print("RESULT: FAIL -- " + "; ".join(failures))
        return 1
    print("RESULT: PASS -- hand-written AIE2 assembly assembles and links with compiled C++.")
    print("Statement-level inline asm still fails in the IRTranslator; a standalone .s does not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
