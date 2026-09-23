# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static census of the conv-engine core program: how often it issues a vmac, per loop.

The graph engine runs one fixed core program (kernels/aie2/conv_engine/engine.cc) on every
tile; a model changes only its weight packets and DMA sequence. So the vector-issue rate of
that one ELF bounds every model the engine runs. AIE2 is statically scheduled (no
interlocks), so a hardware-loop body's bundle count IS its cycle count
(`tools/aie_disasm.py`, calibrated against results/aie/clock_probe_npu.log), and
vmac-bundles / bundles is the loop's issue rate against a ceiling of 1.0.

For every build under --root this tool:
  1. hashes each build's core ELF and groups builds that ship byte-identical core code,
  2. for each distinct ELF: .text size against the 16,384 B program memory, whether the
     ninth accumulator `cm8` is allocated, and for every hardware loop: bundles, vmac and
     vmul count, vmac/cycle, how many bundles also issue vshift / vmov / vlda / vldb,
     start/end 16-byte alignment, the size of the loop's last bundle, and the byte distance
     from the loop-setup writes (`movxm ls`, `movxm le`, `add.nc lc`/`mov lc`) to the loop
     start and end (the rules tnzr.org/xdna/xdna1_kernel.html states for hand assembly).

Static only: no NPU, no hardware context. A loop's issue rate is its steady-state ceiling;
the time spent outside loops, and stalls (memory, lock), are not in it.

    python tools/engine_core_census.py                       # every build, core 0_2
    python tools/engine_core_census.py --builds yolov8n_full sesr_m7 --out results/aie/x.log
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aie_disasm  # noqa: E402

PROGRAM_MEMORY = 16384
SETUP_RE = re.compile(r"\b(movxm\s+(ls|le)\b|add\.nc\s+lc\b|mov\s+lc\b|movxm\s+lc\b)")


def core_elf(build: Path, core: str) -> Path | None:
    p = build / "design.prj" / f"elfs_main_core_{core}" / f"elfs_main_core_{core}.elf"
    return p if p.exists() else None


def text_size(elf: Path, readelf: Path) -> int | None:
    out = subprocess.run([str(readelf), "-S", "-W", str(elf)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.search(r"\]\s+\.text\s+\S+\s+[0-9a-f]+\s+[0-9a-f]+\s+([0-9a-f]+)", line)
        if m:
            return int(m.group(1), 16)
    return None


def op_names(b: aie_disasm.Bundle) -> list[str]:
    return [f.split()[0] for f in b.live]


def analyse(elf: Path, objdump: str, readelf: Path) -> list[str]:
    text = aie_disasm.disassemble(str(elf), objdump)
    sections = aie_disasm.parse(text)
    out = []
    ts = text_size(elf, readelf)
    out.append(f"  .text {ts} B of {PROGRAM_MEMORY} B program memory "
               f"({PROGRAM_MEMORY - ts} B free)" if ts is not None else "  .text size: unread")
    cm = sorted(set(re.findall(r"\bcm\d+\b", text)), key=lambda s: int(s[2:]))
    # cm8's halves and quarters (bml8/bmh8, amll8..amhh8) are also uses of the ninth accumulator.
    n_cm8 = len(re.findall(r"\b(?:cm8|bm[lh]8|am[lh][lh]8)\b", text))
    out.append(f"  accumulators named: {', '.join(cm)}; uses of the ninth (cm8/bm?8/am??8): {n_cm8}")
    for sec in sections:
        bundles = sec.bundles
        func = None
        func_of = {}
        for b in bundles:
            if b.label and not b.label.startswith("."):
                func = b.label
            func_of[b.addr] = func
        idx = {b.addr: i for i, b in enumerate(bundles)}
        for lp in sec.loops:
            body = lp.bundles
            names = [op_names(b) for b in body]
            n = len(body)
            vmac = sum(1 for ns in names if any(x.startswith(("vmac", "vmul")) for x in ns))
            count = lambda pfx: sum(1 for ns in names if any(x.startswith(pfx) for x in ns))  # noqa: E731
            end_i = idx[lp.end]
            last_size = (bundles[end_i + 1].addr - lp.end) if end_i + 1 < len(bundles) else None
            setups = []
            for j in range(idx[lp.start] - 1, max(-1, idx[lp.start] - 40), -1):
                for f in bundles[j].fields:
                    if SETUP_RE.search(f):
                        setups.append((bundles[j].addr, re.sub(r"\s+", " ", f)))
            setup_txt = "; ".join(f"{f} @0x{a:x} ({lp.start - a} B before start, {lp.end - a} B before end)"
                                  for a, f in reversed(setups)) or "no setup write found within 40 bundles"
            out.append(f"  loop {lp.name} in <{func_of.get(lp.start)}> 0x{lp.start:x}-0x{lp.end:x}: "
                       f"{n} bundles, {vmac} vmac/vmul bundles = {vmac / n:.3f} vmac/cycle; "
                       f"vshift {count('vshift')}, vmov {count('vmov')}, vlda {count('vlda')}, vldb {count('vldb')}")
            out.append(f"      align: start%16={lp.start % 16} end%16={lp.end % 16}; "
                       f"last bundle {last_size} B; setup: {setup_txt}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("build/conv_engine"))
    ap.add_argument("--builds", nargs="*", help="build names under --root (default: all)")
    ap.add_argument("--core", default="0_2", help="core id, e.g. 0_2 (all 16 cores run the same program)")
    ap.add_argument("--objdump", default=None)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--bank-check", action="store_true",
                    help="also run tools/aie_bank_check.py on the newest ELF with its engine.o")
    args = ap.parse_args(argv)

    objdump = aie_disasm.find_objdump(args.objdump)
    readelf = Path(objdump).with_name("llvm-readelf.exe")
    builds = [args.root / b for b in args.builds] if args.builds else sorted(p for p in args.root.iterdir() if p.is_dir())

    groups: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
    elf_of: dict[str, Path] = {}
    for b in builds:
        elf = core_elf(b, args.core)
        if not elf:
            continue
        h = hashlib.sha256(elf.read_bytes()).hexdigest()[:12]
        groups[h].append((b.name, datetime.fromtimestamp(elf.stat().st_mtime)))
        elf_of.setdefault(h, elf)

    R = ["Conv-engine core census (static; bundles = cycles on this statically scheduled VLIW)",
         f"root: {args.root.as_posix()}  core: {args.core}  builds with a core ELF: {sum(len(v) for v in groups.values())}",
         f"distinct core ELFs: {len(groups)}", ""]
    order = sorted(groups, key=lambda h: max(t for _, t in groups[h]), reverse=True)
    for h in order:
        names = sorted(groups[h], key=lambda x: x[1])
        R.append(f"== ELF sha256 {h}: {len(names)} build(s), built {names[0][1]:%Y-%m-%d} .. {names[-1][1]:%Y-%m-%d} ==")
        R.append("  builds: " + ", ".join(n for n, _ in names))
        R.extend(analyse(elf_of[h], objdump, readelf))
        R.append("")
    if args.bank_check and order:
        # The newest ELF group is the one current engine.cc builds; check its first build.
        newest = sorted(groups[order[0]], key=lambda x: x[1])[-1][0]
        elf = core_elf(args.root / newest, args.core)
        obj = args.root / newest / "design.prj" / "engine.o"
        cmd = [sys.executable, str(Path(__file__).with_name("aie_bank_check.py")),
               "--elf", elf.as_posix(), "--obj", obj.as_posix(), "--kernel", "engine"]
        R.append(f"== Bank check of the newest ELF ({order[0]}, build {newest}, core {args.core}), static ==")
        R.append("command: python tools/aie_bank_check.py --elf " + elf.as_posix()
                 + " --obj " + obj.as_posix() + " --kernel engine")
        R.append(subprocess.run(cmd, capture_output=True, text=True).stdout.rstrip())
        R.append("")

    report = "\n".join(R) + "\n"
    sys.stdout.write(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
