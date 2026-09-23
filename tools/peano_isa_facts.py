# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read what Peano (llvm-aie) believes about the AIE2 core, from the compiler's own tablegen.

Every kernel in this repo is compiled by Peano, so its AIE2 machine model -- issue slots,
itinerary latencies, operand read cycles, the register file -- is the compiler-side ground
truth for scheduling questions. This tool pins that model to the exact llvm-aie commit that
is installed (from `clang --version`), downloads `llvm/lib/Target/AIE/aie2/*.td` at that
commit into a cache directory, and prints:

  1. the accumulator and vector register files (how many `cm`, `bml`/`bmh`, `amll`.., `x`..),
  2. the issue slots (`AIE2Slots.td`),
  3. for each audited mnemonic: the slot(s) it can issue in (the `_inst_<slot>` suffix of its
     format class), its itinerary, the result latency and the cycle each operand is read,
     and the functional units its stages reserve,
  4. a local cross-check that needs no network: small AIE-API snippets compiled with the
     installed Peano to assembly (`clang -S`), with the bundle distance the scheduler leaves
     between each dependent producer/consumer pair and the delay-slot count after `ret`.
     This is the inference method Steinert and Breuer use on tnzr.org/xdna/isa.html.

Reading an itinerary: `[6,3,1,1,1]` on `II_VMACf` means the result is written at cycle 6
and the accumulator input is read at cycle 3 (both counted from issue = 1), so a dependent
`vmac.f` on the same accumulator can issue 3 bundles after its producer.

What this is NOT: a measurement. A tablegen latency is what the scheduler assumes and is
tagged SPEC(Peano) wherever it is quoted. Silicon can disagree; only a probe with a log
under `results/aie/` settles that.

    python tools/peano_isa_facts.py                                   # auto-detect commit
    python tools/peano_isa_facts.py --out results/aie/peano_aie2_machine_model_a36c62b9.log
    python tools/peano_isa_facts.py --fetch-only                      # populate the cache
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

RAW = "https://raw.githubusercontent.com/Xilinx/llvm-aie/{commit}/llvm/lib/Target/AIE/aie2/{name}"
TD_FILES = (
    "AIE2Schedule.td",
    "AIE2Slots.td",
    "AIE2GenRegisterInfo.td",
    "AIE2GenInstrInfo.td",
    "AIE2GenFixupInstrInfo.td",
    "AIE2GenInstrFormats.td",
    "AIE2InstrInfo.td",
    "AIE2InstrFormats.td",
    "AIE2MultiSlotPseudoInstrInfo.td",
    "AIE2CompositeFormats.td",
    "AIE2InstrPatterns.td",
    "AIE2RegOperandDef.td",
    "AIE2.td",
)
DEFAULT_PEANO = Path.home() / "mlir-aie/ironenv/Lib/site-packages/llvm-aie"
DEFAULT_INCLUDE = Path.home() / "mlir-aie/ironenv/Lib/site-packages/mlir_aie/include"

# Mnemonics the audit asks about (tnzr.org Table 2 plus the ones this repo's docs discuss).
AUDITED = (
    "vmac", "vmac.f", "vmul", "vmul.f", "vlda", "vldb", "vst", "vlda.conv.fp32.bf16",
    "vst.conv.bf16.fp32", "vlda.ups.s32.s8", "vst.srs.s8.s32", "vshift", "vshift.align",
    "vshuffle", "vmov", "vbcst.8", "vbcst.16", "vbcst.32", "lda", "st", "mul", "add",
    "mova", "movx", "movxm", "padda", "paddb", "padds", "j", "jz", "jnz", "ret lr", "jl",
)

_BLOCK_COMMENT = re.compile(r"/\*(.*?)\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")


def detect_commit(peano: Path) -> str:
    """The llvm-aie commit of the installed Peano, from `clang --version`."""
    out = subprocess.run([str(peano / "bin" / "clang.exe"), "--version"],
                         capture_output=True, text=True, check=True).stdout
    m = re.search(r"llvm-aie ([0-9a-f]{40})", out)
    if not m:
        raise SystemExit(f"could not find an llvm-aie commit in clang --version:\n{out}")
    return m.group(1)


def fetch(commit: str, cache: Path) -> dict[str, Path]:
    cache = cache / commit
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in TD_FILES:
        dst = cache / name
        if not dst.exists():
            with urllib.request.urlopen(RAW.format(commit=commit, name=name), timeout=60) as r:
                dst.write_bytes(r.read())
        paths[name] = dst
    return paths


# --------------------------------------------------------------------------- tablegen parsing

def _balanced(text: str, start: int, open_ch: str = "<", close_ch: str = ">") -> tuple[str, int]:
    """Body between text[start] == open_ch and its matching close, and the index after it."""
    assert text[start] == open_ch
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError(f"unbalanced {open_ch} at {start}")


def _split_top(body: str) -> list[str]:
    """Split on commas that are not inside <>, [] or ()."""
    out, depth, cur = [], 0, []
    for ch in body:
        if ch in "<[(":
            depth += 1
        elif ch in ">])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def parse_itineraries(text: str) -> dict[str, dict]:
    """II_NAME -> {cycles: [...labelled...], units: [...], line: n} from AIE2Schedule.td."""
    # Keep block-comment labels inside operand lists (e.g. /*srFPFlags*/7 -> srFPFlags:7).
    labelled = _BLOCK_COMMENT.sub(lambda m: f"{m.group(1).strip()}:" if len(m.group(1)) < 40 else "", text)
    labelled = _LINE_COMMENT.sub("", labelled)
    line_of = {}
    for i, line in enumerate(text.splitlines(), 1):
        m = re.search(r"InstrItinData<\s*(II_\w+)", line)
        if m and m.group(1) not in line_of:
            line_of[m.group(1)] = i
    out = {}
    for m in re.finditer(r"\b(?:Mem)?InstrItinData<", labelled):
        body, _ = _balanced(labelled, m.end() - 1)
        fields = _split_top(body)
        if len(fields) < 2:
            continue
        name = fields[0].strip()
        stages = fields[1].strip()
        cycles = fields[2].strip() if len(fields) > 2 and fields[2].strip().startswith("[") else "[]"
        units = sorted(set(re.findall(r"\b([A-Z][A-Z0-9_]*(?:UNIT|SRS|UPS_UNIT)[A-Z0-9_]*)\b", stages)))
        ports = sorted(set(re.findall(r"\b([A-Z][A-Z0-9]*_(?:R|W)[A-Z]*_PORT)\b", stages)))
        out[name] = {
            "cycles": [c.strip() for c in cycles.strip("[]").split(",") if c.strip()],
            "units": units,
            "ports": ports,
            "line": line_of.get(name),
        }
    return out


def _slot_of(cls: str) -> str | None:
    m = re.search(r"_inst_([a-z]+)$", cls) or re.search(r"_(lng)$", cls)
    return m.group(1) if m else None


def parse_defs(text: str) -> list[dict]:
    """Every `def NAME : CLASS<...>` with the Itinerary of its enclosing `let` and its mnemonic."""
    clean = _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))
    tokens = re.finditer(r"\blet\s+(?P<let>[^{};]*?)\s+in\s*(?P<brace>\{)?"
                         r"|\bdef\s+(?P<def>\w+)\s*:\s*(?P<cls>\w+)\s*<"
                         r"|(?P<close>\})", clean)
    stack: list[tuple[str | None, bool]] = []  # (itinerary or None, is_block)
    pending: list[str | None] = []              # single-statement lets (no brace)
    out = []
    for t in tokens:
        if t.group("let") is not None:
            itin = re.search(r"Itinerary\s*=\s*(II_\w+)", t.group("let"))
            val = itin.group(1) if itin else None
            if t.group("brace"):
                stack.append((val, True))
            else:
                pending.append(val)
        elif t.group("def"):
            body, _ = _balanced(clean, t.end() - 1)
            mn = re.search(r'"([^"]*)"', body)
            itins = [v for v, _ in stack if v] + [v for v in pending if v]
            out.append({
                "name": t.group("def"),
                "cls": t.group("cls"),
                "mnemonic": mn.group(1) if mn else "",
                "itinerary": itins[-1] if itins else None,
                # Format classes end in `_inst_<slot>`, except the 48-bit long-immediate ones
                # (jumps, movxm), which end in `_lng`.
                "slot": _slot_of(t.group("cls")),
            })
            pending = []
        elif t.group("close"):
            if stack:
                stack.pop()
    return out


def parse_slots(text: str) -> list[tuple[str, str, int]]:
    return [(m.group(1), m.group(2), int(m.group(3)))
            for m in re.finditer(r"def\s+(\w+)\s*:\s*InstSlot<\"(\w+)\",\s*(\d+)>", text)]


def parse_registers(text: str) -> dict[str, list[str]]:
    fam = defaultdict(list)
    for m in re.finditer(r"^\s*def\s+([a-z]+)(\d+)\s*:", text, re.M):
        fam[m.group(1)].append(m.group(1) + m.group(2))
    return fam


# --------------------------------------------------------------------------- local cross-check

SNIPPETS = r"""
#include <aie_api/aie.hpp>
extern "C" {
// bf16 4x8x4: load accumulator, one mac, store (a single dependent chain).
void bf16_mac(const bfloat16 *__restrict a, const bfloat16 *__restrict b, float *__restrict c) {
  using MMUL = aie::mmul<4, 8, 4, bfloat16, bfloat16, accfloat>;
  MMUL m(aie::load_v<MMUL::size_C>(c));
  m.mac(aie::load_v<MMUL::size_A>(a), aie::load_v<MMUL::size_B>(b));
  aie::store_v(c, m.template to_vector<float>());
}
// bf16: two macs into the same accumulator, back to back (accumulator forwarding distance).
void bf16_mac2(const bfloat16 *__restrict a, const bfloat16 *__restrict b, float *__restrict c) {
  using MMUL = aie::mmul<4, 8, 4, bfloat16, bfloat16, accfloat>;
  MMUL m(aie::load_v<MMUL::size_C>(c));
  m.mac(aie::load_v<MMUL::size_A>(a), aie::load_v<MMUL::size_B>(b));
  m.mac(aie::load_v<MMUL::size_A>(a + MMUL::size_A), aie::load_v<MMUL::size_B>(b + MMUL::size_B));
  aie::store_v(c, m.template to_vector<float>());
}
// int8 4x8x8: mul then a dependent mac, then store (int32 out, no shift).
void i8_mac2(const int8 *__restrict a, const int8 *__restrict b, int32 *__restrict c) {
  using MMUL = aie::mmul<4, 8, 8, int8, int8, acc32>;
  MMUL m;
  m.mul(aie::load_v<MMUL::size_A>(a), aie::load_v<MMUL::size_B>(b));
  m.mac(aie::load_v<MMUL::size_A>(a + MMUL::size_A), aie::load_v<MMUL::size_B>(b + MMUL::size_B));
  aie::store_v(c, m.template to_vector<int32>());
}
}
"""

_OPS = re.compile(r"\b(vmac(?:\.f)?|vmul(?:\.f)?|vlda|vldb|vst|ret|mova|movx|nop\w*)\b")


def local_crosscheck(peano: Path, include: Path) -> list[str]:
    lines = []
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "snip.cc"
        asm = Path(td) / "snip.s"
        src.write_text(SNIPPETS)
        cmd = [str(peano / "bin" / "clang++.exe"), "--target=aie2-none-unknown-elf", "-std=c++20",
               "-O2", "-D__AIE_API_AIE_ADF_HPP__=1", "-I", str(include), "-S", str(src), "-o", str(asm)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            return ["local cross-check: compile FAILED", p.stderr.strip()[:2000]]
        text = asm.read_text()
    lines.append("command: clang++ --target=aie2-none-unknown-elf -std=c++20 -O2 "
                 "-D__AIE_API_AIE_ADF_HPP__=1 -I <mlir_aie>/include -S snip.cc -o snip.s "
                 "(snippets: SNIPPETS in tools/peano_isa_facts.py)")
    funcs = re.split(r"\n(?=\w+:\s*(?://.*)?\n)", text)
    for f in funcs:
        head = re.match(r"(\w+):", f)
        if not head or head.group(1) not in ("bf16_mac", "bf16_mac2", "i8_mac2"):
            continue
        bundles = []
        for ln in f.splitlines()[1:]:
            s = ln.split("//")[0].strip()
            if not s or s.startswith(".") or s.endswith(":"):
                continue
            bundles.append((s, ln))
        lines.append("")
        lines.append(f"[{head.group(1)}]  {len(bundles)} bundles")
        for i, (s, ln) in enumerate(bundles):
            note = ln.split("//", 1)[1].strip() if "//" in ln else ""
            lines.append(f"  {i:3d}  {re.sub(r'\s+', ' ', s)}" + (f"    // {note}" if note else ""))
        idx = {k: [i for i, (s, _) in enumerate(bundles) if re.search(k, s)]
               for k in (r"\bvmac\.f\b|\bvmac\b", r"\bvmul(\.f)?\b", r"\bvlda\s+(am|bm|cm)", r"\bvst\s+(am|bm|cm)|\bvst\b", r"\bret\b")}
        macs = idx[r"\bvmac\.f\b|\bvmac\b"] + idx[r"\bvmul(\.f)?\b"]
        macs.sort()
        accl = idx[r"\bvlda\s+(am|bm|cm)"]
        vst = idx[r"\bvst\s+(am|bm|cm)|\bvst\b"]
        if accl and macs:
            lines.append(f"  distance last accumulator load -> first mac: {macs[0] - max(a for a in accl if a < macs[0])} bundles"
                         if any(a < macs[0] for a in accl) else "  (no accumulator load before the first mac)")
        if len(macs) >= 2:
            lines.append(f"  distance mac -> dependent mac (same accumulator): {macs[1] - macs[0]} bundles")
        if macs and vst:
            after = [v for v in vst if v > macs[-1]]
            if after:
                lines.append(f"  distance last mac -> first store of its result: {after[0] - macs[-1]} bundles")
        if idx[r"\bret\b"]:
            r = idx[r"\bret\b"][0]
            lines.append(f"  bundles after ret (delay slots): {len(bundles) - 1 - r}")
    return lines


# --------------------------------------------------------------------------- report

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", help="llvm-aie commit (default: the installed Peano's)")
    ap.add_argument("--peano", type=Path, default=DEFAULT_PEANO, help="llvm-aie install dir")
    ap.add_argument("--include", type=Path, default=DEFAULT_INCLUDE, help="dir containing aie_api/")
    ap.add_argument("--cache", type=Path, default=Path("scratch/llvm-aie-td"), help="tablegen cache dir")
    ap.add_argument("--out", type=Path, help="also write the report here (UTF-8)")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--no-local", action="store_true", help="skip the clang -S cross-check")
    args = ap.parse_args(argv)

    installed = detect_commit(args.peano)
    commit = args.commit or installed
    paths = fetch(commit, args.cache)
    if args.fetch_only:
        for p in paths.values():
            print(f"{p.stat().st_size:9d}  {p.name}")
        return 0

    txt = {n: p.read_text(encoding="utf-8") for n, p in paths.items()}
    R = []
    R.append("Peano (llvm-aie) AIE2 machine model -- tablegen facts, tagged SPEC(Peano)")
    R.append(f"llvm-aie commit: {commit}  (installed Peano: {installed}{', MATCHES' if commit == installed else ', DIFFERS'})")
    R.append("source: https://github.com/Xilinx/llvm-aie/tree/<commit>/llvm/lib/Target/AIE/aie2")
    for n in TD_FILES:
        R.append(f"  sha256 {hashlib.sha256(paths[n].read_bytes()).hexdigest()[:16]}  {n}")
    R.append("NOT a measurement: these are the latencies and slots the scheduler assumes.")

    # 1. registers
    regs = parse_registers(txt["AIE2GenRegisterInfo.td"])
    R.append("")
    R.append("== 1. Register files (AIE2GenRegisterInfo.td) ==")
    for fam in ("cm", "bml", "bmh", "amll", "amlh", "amhl", "amhh", "x", "wl", "wh", "y", "r", "p", "m"):
        if regs.get(fam):
            R.append(f"  {fam:5s} {len(regs[fam]):3d}  {regs[fam][0]}..{regs[fam][-1]}")
    cm = re.findall(r"def (cm\d+)\s*:\s*\w+<[^,]*,\s*\"cm\d+\",\s*\[(\w+),\s*(\w+)\]", txt["AIE2GenRegisterInfo.td"])
    if cm:
        R.append("  1024-bit accumulators and their 512-bit halves: " + ", ".join(f"{c}=[{a},{b}]" for c, a, b in cm))

    # 2. slots
    R.append("")
    R.append("== 2. Issue slots (AIE2Slots.td) ==")
    for d, n, bits in parse_slots(txt["AIE2Slots.td"]):
        R.append(f"  {d:12s} \"{n}\"  {bits} encoding bits")

    # 3. audited mnemonics
    itins = parse_itineraries(txt["AIE2Schedule.td"])
    defs = parse_defs(txt["AIE2GenInstrInfo.td"]) + parse_defs(txt["AIE2GenFixupInstrInfo.td"]) \
        + parse_defs(txt["AIE2InstrInfo.td"])
    by_mn = defaultdict(list)
    for d in defs:
        by_mn[d["mnemonic"]].append(d)
    R.append("")
    R.append("== 3. Audited mnemonics: slot, itinerary, [result latency, operand read cycles...], units ==")
    R.append("   cycles are counted from issue = 1; the first entry is when the result is written.")
    for mn in AUDITED:
        ds = by_mn.get(mn, [])
        if not ds:
            R.append(f"  {mn:22s} (no def with this mnemonic in the fetched files)")
            continue
        slots = sorted({d["slot"] or "?" for d in ds})
        R.append(f"  {mn:22s} slots {'/'.join(slots)}  ({len(ds)} encodings)")
        seen = set()
        resolved = any(d["itinerary"] for d in ds)
        for d in ds:
            it = d["itinerary"]
            # Some immediate-offset encodings sit in a `let` this parser does not attribute;
            # they share their register-offset sibling's itinerary, so drop the unresolved row.
            if it in seen or (it is None and resolved):
                continue
            seen.add(it)
            if it is None:
                R.append(f"      (itinerary not attributed by this parser)  e.g. {d['name']} slot={d['slot']}")
                continue
            info = itins.get(it, {})
            cyc = ",".join(info.get("cycles", [])) or "-"
            units = " ".join(info.get("units", []))
            ln = info.get("line")
            R.append(f"      {str(it):24s} [{cyc}]  {units}  (Schedule.td:{ln})  e.g. {d['name']} slot={d['slot']}")

    R.append("")
    R.append("   Operand bypasses: II_VMAC's accumulator operand is VEC_Bypass (forwarded one cycle early),")
    R.append("   II_VMACf lists none. Delay slots are not in tablegen; section 4 reads them from `ret lr`.")

    # 4. local cross-check
    if not args.no_local:
        R.append("")
        R.append("== 4. Local cross-check: AIE-API snippets compiled with the installed Peano (clang -S) ==")
        R.append("   The bundle distance the scheduler leaves between dependent ops is its latency belief.")
        R.extend(local_crosscheck(args.peano, args.include))

    report = "\n".join(R) + "\n"
    sys.stdout.write(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
