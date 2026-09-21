"""How much of a core's 16 KB program memory does an engine kernel really occupy?

The per-opcode census (tools/engine_opcode_census.py) prices the KERNEL OBJECT's .text. Program
memory holds the LINKED core: the kernel object plus the core wrapper IRON generates around it
(the ObjectFifo acquire/release loop `core_<col>_<row>`, `_main_init`, `__start`). This tool
reads both out of an existing build and prints the difference, so a budget is stated against
what the device holds rather than against what the compiler emitted for one source file.

Static: it reads ELF and object files that already exist. It compiles nothing and never opens
the NPU. Run where Peano's llvm-size and llvm-objdump are reachable (the mlir-aie ironenv):

    bash scripts/research-iron.sh tools/engine_linked_size.py --build-root build/conv_engine \\
        --builds yolov8n_full yolov8n_rb2 modnet_cut_dense_20260921
    bash scripts/research-iron.sh tools/engine_linked_size.py --prj scratch/iron-cache/<key>

A build is a directory holding one kernel object and `elfs_main_core_<c>_<r>/*.elf`; with --build-root
that directory is `<root>/<build>/design.prj`. Paths are printed relative to the root given, so
a log carries no machine-local prefix.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.aie_bank_check import find_tool  # noqa: E402

# docs/SILICON.md, "Program memory | 16 KB | SPEC: device.yaml core_program_memory: 16".
PROGRAM_MEMORY = 16 * 1024


def text_sections(path: Path) -> list[tuple[str, int]]:
    out = subprocess.run([find_tool("llvm-size"), "-A", str(path)], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"llvm-size failed on {path}:\n{out.stderr[:600]}")
    return [(m.group(1), int(m.group(2)))
            for m in re.finditer(r"^(\.text\S*)\s+(\d+)", out.stdout, re.M) if int(m.group(2))]


def functions(path: Path) -> list[tuple[str, int]]:
    out = subprocess.run([find_tool("llvm-objdump"), "-t", str(path)], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"llvm-objdump failed on {path}:\n{out.stderr[:600]}")
    found = []
    for line in out.stdout.splitlines():
        m = re.match(r"^[0-9a-f]+\s+\S+\s+F\s+\S+\s+([0-9a-f]+)\s+(\S+)$", line)
        if m and int(m.group(1), 16):
            found.append((m.group(2), int(m.group(1), 16)))
    return sorted(found, key=lambda f: -f[1])


def report(label: str, prj: Path, object_name: str | None) -> int:
    # A project directory holds one kernel object at its top level; name it only to disambiguate.
    objs = [prj / object_name] if object_name else sorted(prj.glob("*.o"))
    elfs = sorted(prj.glob("elfs_main_core_*/*.elf"))
    if len(objs) != 1 or not objs[0].exists() or not elfs:
        print(f"{label}: expected one kernel object and at least one core ELF, found "
              f"{[o.name for o in objs]} and {len(elfs)} ELF(s); skipped")
        return 1
    obj = objs[0]
    obj_text = sum(size for _, size in text_sections(obj))
    linked = {sum(size for _, size in text_sections(e)) for e in elfs}
    print(f"{label}: {obj.name} .text {obj_text} B over {len(text_sections(obj))} sections; "
          f"{len(elfs)} core ELF(s), linked .text {sorted(linked)} B")
    if len(linked) != 1:
        print("  the cores do not agree on a linked size; no single figure to report")
        return 1
    total = linked.pop()
    wrapper = [(n, s) for n, s in functions(elfs[0]) if n not in {f for f, _ in functions(obj)}]
    print(f"  wrapper {total - obj_text} B = " + " + ".join(f"{n} {s}" for n, s in wrapper)
          + f" (sum {sum(s for _, s in wrapper)})")
    print(f"  program memory {PROGRAM_MEMORY} B: {total} used, {PROGRAM_MEMORY - total} B free; "
          f"an object may be at most {PROGRAM_MEMORY - (total - obj_text)} B under this wrapper")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-root", type=Path, help="directory of <build>/design.prj/ builds")
    ap.add_argument("--builds", nargs="+", default=[], help="build names under --build-root")
    ap.add_argument("--prj", type=Path, nargs="+", default=[], help="project directories given directly")
    ap.add_argument("--object", default=None,
                    help="kernel object file name, when a project directory holds more than one")
    args = ap.parse_args()
    if not (args.build_root and args.builds) and not args.prj:
        ap.error("give --build-root with --builds, or --prj")
    rc = 0
    for name in args.builds:
        rc |= report(name, args.build_root / name / "design.prj", args.object)
    for prj in args.prj:
        rc |= report(prj.name, prj, args.object)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
