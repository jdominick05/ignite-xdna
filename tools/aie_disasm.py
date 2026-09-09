"""Static AIE2 analysis: VLIW bundles, slot occupancy, hardware loops, stack frames.

Disassembles a core ELF or a kernel object with Peano's own llvm-objdump and reports
what the compiler actually scheduled, so a kernel's inner-loop cost can be read off the
machine code instead of inferred from a throughput fit.

Why this works on AIE2 and not on an out-of-order CPU: the core is a statically
scheduled VLIW with an exposed pipeline. The compiler covers every operand latency with
explicit nop bundles rather than leaving it to a hardware interlock, so the bundle count
of a zero-overhead hardware loop body IS its cycle count. Checked against
`results/aie/clock_probe_npu.log`: the scalar loop measures 9.000 cycles/iteration and
disassembles to 9 bundles; the vector loop measures 2.000 and disassembles to 2.

Slots. A full-width bundle is 16 bytes and prints its slots separated by ';'. The nop
mnemonics name them -- nopb, nopa, nops, nopx, nopm, nopv -- so there are six slots:
b and a are the TWO LOAD UNITS, s store, x scalar, m move, v vector. Read off 226
strictly six-field bundles across the build caches: slot b holds nopb/paddb/vldb, slot
a holds nopa/mova/vlda/lda, slot s holds nops/vst/st, slot x holds the scalar ALU and
control flow (add, lshl, or, event, ret), slot m holds mov/add.nc/vbcst/vshuffle, and
slot v holds the vector MAC. An earlier version of this file called slot b "branch",
which is wrong: it is the second load unit, `vldb` issues in it, and `ret` issues in
the scalar slot. That distinction is the whole reason a bank conflict is possible --
a bundle naming both a and b issues two loads in one cycle. `nopxm` is the fused
encoding printed when both x and m are empty, so a 5-field bundle still occupies six
slots.
Bundles using few slots are emitted in a compressed encoding shorter than 16 bytes;
they still issue in one cycle, so cycles are counted by bundle, never by byte.

Hardware loops. `add.nc lc, ...` sets the trip count and `movxm ls/le` the start and
end addresses. In the disassembly the last bundle of a loop body carries a `.L_LEnd*`
label. The body is taken as every bundle from the nearest preceding label through that
`.L_LEnd*` bundle inclusive, which reproduces both clock-probe loops exactly.

Usage:
    python tools/aie_disasm.py <file.elf|file.o> [--loops] [--slots] [--verbose]
    python tools/aie_disasm.py --cache-dir ~/.npu/cache/<hash> --core 0_2 --loops

Read-only. No NPU, no hardware context, runs in about a second.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

# Peano ships its own llvm-objdump; it is the only one here that knows elf32-aie.
DEFAULT_OBJDUMP = os.path.expanduser(
    "~/mlir-aie/ironenv/Lib/site-packages/llvm-aie/bin/llvm-objdump.exe"
)

# The six slot letters, in the order llvm-objdump prints them.
SLOTS_FULL = ("b", "a", "s", "x", "m", "v")
SLOTS_FUSED = ("b", "a", "s", "xm", "v")

NOP_RE = re.compile(r"^nop(b|a|s|x|m|v|xm)?$")
ADDR_RE = re.compile(r"^\s*([0-9a-f]+):\s*(.*)$")
LABEL_RE = re.compile(r"^([0-9a-f]+)\s+<([^>]+)>:$")
SECTION_RE = re.compile(r"^Disassembly of section (.+):$")
# Prologue frame adjust, e.g. "paddb [sp], #0x60".
FRAME_RE = re.compile(r"paddb\s+\[sp\],\s*#(-?0x[0-9a-f]+|-?\d+)")
STACK_REF_RE = re.compile(r"^(st|lda|ldb|vst|vlda|vldb)\b")
# Register files, as llvm-objdump names them for aie2.
REG_FILES = {
    "acc": re.compile(r"\bcm\d+\b"),      # accumulators, e.g. the target of vmac
    "vec": re.compile(r"\b[xy]\d+\b"),    # vector registers
    "wide": re.compile(r"\bw[cr]?\d+\b"),  # wide vector registers
    "ptr": re.compile(r"\bp\d+\b"),
    "scalar": re.compile(r"\br\d+\b"),
}


@dataclass
class Bundle:
    addr: int
    fields: list[str]
    label: str | None = None

    @property
    def slots(self) -> tuple[str, ...]:
        """Slot letters for this bundle's printed fields.

        Only a full-width bundle prints one field per slot. A compressed bundle prints
        just its live operations, so the slot identity of each field is not recoverable
        from the text and is reported as unknown.
        """
        if len(self.fields) == 6:
            return SLOTS_FULL
        if len(self.fields) == 5:
            return SLOTS_FUSED
        return tuple("?" for _ in self.fields)

    @property
    def is_full_width(self) -> bool:
        return len(self.fields) in (5, 6)

    @property
    def live(self) -> list[str]:
        """Fields that are real operations rather than nops."""
        out = []
        for f in self.fields:
            if not f:
                continue
            if NOP_RE.match(f.split()[0]):
                continue
            out.append(f)
        return out


@dataclass
class Loop:
    name: str
    start: int
    end: int
    bundles: list[Bundle] = field(default_factory=list)

    @property
    def n_bundles(self) -> int:
        return len(self.bundles)

    @property
    def n_live(self) -> int:
        return sum(len(b.live) for b in self.bundles)

    @property
    def density(self) -> float:
        """Live operations per bundle. The ceiling is 6, one per slot."""
        return self.n_live / self.n_bundles if self.n_bundles else 0.0


@dataclass
class Section:
    name: str
    bundles: list[Bundle] = field(default_factory=list)
    loops: list[Loop] = field(default_factory=list)

    @property
    def frame_bytes(self) -> int | None:
        """Stack frame from the first positive frame adjust, in bytes."""
        for b in self.bundles:
            for f in b.fields:
                m = FRAME_RE.search(f)
                if m:
                    value = int(m.group(1), 0)
                    if value > 0:
                        return value
        return None

    def registers(self, kind: str) -> list[str]:
        """Distinct registers of one file that this section names, in index order.

        Counting the accumulators a function actually allocates is a direct reading
        of register pressure, where a stack spill is only the consequence of running
        out. `cm0..cmN` are the accumulator targets of `vmac` and `vmul`.
        """
        pattern = REG_FILES[kind]
        seen: set[str] = set()
        for b in self.bundles:
            for f in b.fields:
                seen.update(pattern.findall(f))
        return sorted(seen, key=lambda s: (len(s), s))

    @property
    def stack_traffic(self) -> int:
        """Loads and stores based on the stack or frame pointer.

        A kernel that spills carries stack traffic outside its prologue and epilogue;
        one that keeps everything in registers carries almost none.
        """
        n = 0
        for b in self.bundles:
            for f in b.fields:
                if STACK_REF_RE.match(f) and re.search(r"\[(sp|p7)\b", f):
                    n += 1
        return n


def find_objdump(explicit: str | None) -> str:
    for candidate in (explicit, DEFAULT_OBJDUMP, shutil.which("llvm-objdump")):
        if candidate and os.path.exists(candidate):
            return candidate
    raise SystemExit(
        "llvm-objdump not found. Pass --objdump, or install mlir-aie's llvm-aie wheel."
    )


def disassemble(path: str, objdump: str) -> str:
    proc = subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", path],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"llvm-objdump failed on {path}:\n{proc.stderr.strip()}")
    return proc.stdout


def parse(text: str) -> list[Section]:
    sections: list[Section] = []
    current: Section | None = None
    pending_label: str | None = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        m = SECTION_RE.match(line.strip())
        if m:
            current = Section(name=m.group(1))
            sections.append(current)
            pending_label = None
            continue
        if current is None:
            continue
        m = LABEL_RE.match(line.strip())
        if m:
            pending_label = m.group(2)
            continue
        m = ADDR_RE.match(line)
        if not m:
            continue
        fields = [f.strip() for f in m.group(2).split(";")]
        fields = [f for f in fields if f]
        current.bundles.append(
            Bundle(addr=int(m.group(1), 16), fields=fields, label=pending_label)
        )
        pending_label = None

    for section in sections:
        section.loops = find_loops(section)
    return sections


def find_loops(section: Section) -> list[Loop]:
    """Hardware-loop bodies: nearest preceding label through the .L_LEnd* bundle."""
    loops: list[Loop] = []
    for i, b in enumerate(section.bundles):
        if not (b.label and b.label.startswith(".L_LEnd")):
            continue
        start_i = 0
        for j in range(i - 1, -1, -1):
            if section.bundles[j].label:
                start_i = j
                break
        body = section.bundles[start_i : i + 1]
        loops.append(Loop(name=b.label, start=body[0].addr, end=b.addr, bundles=body))
    return loops


def slot_histogram(bundles: list[Bundle]) -> dict[str, int]:
    """How often each named slot holds a real operation, over full-width bundles."""
    hist: dict[str, int] = {s: 0 for s in SLOTS_FULL}
    hist["xm(fused)"] = 0
    for b in bundles:
        if not b.is_full_width:
            continue
        for slot, f in zip(b.slots, b.fields):
            if NOP_RE.match(f.split()[0]):
                continue
            key = "xm(fused)" if slot == "xm" else slot
            hist[key] = hist.get(key, 0) + 1
    return hist


def report(path: str, sections: list[Section], args) -> dict:
    out: dict = {"file": path, "sections": []}
    print(f"== {path}")
    for s in sections:
        n_full = sum(1 for b in s.bundles if b.is_full_width)
        frame = s.frame_bytes
        rec: dict = {
            "name": s.name,
            "bundles": len(s.bundles),
            "full_width": n_full,
            "compressed": len(s.bundles) - n_full,
            "frame_bytes": frame,
            "stack_traffic": s.stack_traffic,
            "loops": [],
        }
        print(
            f"  section {s.name}: {len(s.bundles)} bundles "
            f"({n_full} full-width, {len(s.bundles) - n_full} compressed), "
            f"frame {frame if frame is not None else 'none'} B, "
            f"stack refs {s.stack_traffic}"
        )
        if args.regs:
            for kind in ("acc", "vec", "wide"):
                names = s.registers(kind)
                rec[f"regs_{kind}"] = names
                if names:
                    print(f"    {kind:>5} registers ({len(names)}): {' '.join(names)}")
        if args.slots:
            hist = slot_histogram(s.bundles)
            rec["slot_histogram"] = hist
            live = ", ".join(f"{k}={v}" for k, v in hist.items() if v)
            print(f"    slots used (full-width bundles only): {live or 'none'}")
        if args.loops:
            for lp in s.loops:
                rec["loops"].append(
                    {
                        "name": lp.name,
                        "start": f"0x{lp.start:x}",
                        "end": f"0x{lp.end:x}",
                        "bundles": lp.n_bundles,
                        "live_ops": lp.n_live,
                        "density": round(lp.density, 3),
                    }
                )
                print(
                    f"    loop {lp.name} 0x{lp.start:x}-0x{lp.end:x}: "
                    f"{lp.n_bundles} bundles = {lp.n_bundles} cycles/iteration, "
                    f"{lp.n_live} live ops, density {lp.density:.2f} of 6"
                )
                if args.verbose:
                    for b in lp.bundles:
                        ops = " | ".join(b.live) or "(all nop)"
                        print(f"        0x{b.addr:04x}  {ops}")
        out["sections"].append(rec)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", help="ELF or object files to analyse")
    ap.add_argument("--cache-dir", help="an IRON cache directory; picks core ELFs from it")
    ap.add_argument("--core", help="core id within --cache-dir, e.g. 0_2")
    ap.add_argument("--objdump", help="path to an llvm-objdump that knows elf32-aie")
    ap.add_argument("--loops", action="store_true", help="report hardware-loop bodies")
    ap.add_argument("--slots", action="store_true", help="report per-slot occupancy")
    ap.add_argument("--regs", action="store_true", help="report registers allocated")
    ap.add_argument("--verbose", action="store_true", help="print each loop bundle")
    ap.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    args = ap.parse_args(argv)

    targets = list(args.files)
    if args.cache_dir:
        root = os.path.expanduser(args.cache_dir)
        want = args.core
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if not f.endswith(".elf"):
                    continue
                if want and f"core_{want}" not in f:
                    continue
                targets.append(os.path.join(dirpath, f))
    if not targets:
        ap.error("give at least one file, or --cache-dir")

    objdump = find_objdump(args.objdump)
    reports = []
    for t in sorted(targets):
        reports.append(report(t, parse(disassemble(t, objdump)), args))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(reports, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
