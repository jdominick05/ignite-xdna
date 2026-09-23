# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate A of the INT4 study: which int4 operand pairs does the toolchain lower?

Compile only -- no NPU, no hardware context, no IRON. Two independent readings of the
installed toolchain (mlir-aie's aie_api headers + Peano/llvm-aie):

1. The MAC control word. Every Peano MAC intrinsic builds its configuration word through
   `<arch>_compute_control(sgn_x, sgn_y, amode, bmode, variant, ...)`, with a 2-bit `amode`
   field (bits 1-2) and a 2-bit `bmode` field (bits 3-4). Parsing every intrinsic in
   `<arch>_vmult.h` tabulates which (amode, bmode) codes the toolchain ever emits, and for
   which operand element types -- i.e. every operand-width pair the toolchain can issue.
2. aie::mmul. `isa_gate.cc` compiles one `aie::mmul<M,K,N,TA,TB,ACC>` per operand pair with
   IRON's exact Peano command (kernels/w4a8_probe/static_probe.py's IRON_FLAGS), plus a
   harness control per pair (-DNO_MMUL: same loads and casts, no mmul). A pair whose control
   compiles and whose mmul does not is a missing aie_api path, not a broken harness. For each
   mmul that compiles, the MAC instructions in the object are read back with llvm-objdump.

What this CANNOT establish: anything about the silicon. A code absent from the control-word
table is a code the toolchain never emits; whether the hardware decodes it is a different
question. The W4A8 probe is the precedent: device.yaml's cost table omitted int8 x int4 and
the silicon had it (docs/SILICON.md 1.2). So every verdict here is worded "no aie_api/Peano
path", never "the silicon cannot".

Usage (any Python; it only shells out to Peano):
    python kernels/int4_study/isa_gate.py
    python kernels/int4_study/isa_gate.py --log results/aie/int4_isa_gate_desktop2_20260923.log
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "kernels", "w4a8_probe"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import aie_disasm  # noqa: E402
from static_probe import CLANG, INCLUDE, IRON_FLAGS, SITE  # noqa: E402

SOURCE = os.path.join(_HERE, "isa_gate.cc")
PEANO_INC = os.path.join(SITE, "llvm-aie", "lib", "clang", "22", "include")
VMULT = {
    "aie2": os.path.join(PEANO_INC, "aiev2", "aiev2_vmult.h"),
    "aie2p": os.path.join(PEANO_INC, "aie2p", "aie2p_vmult.h"),
}

# (target, TA, TB, M, K, N, ACC, expected). "pass"/"fail" is the pre-registered
# expectation, written before the run from the header listing (aie2: detail/aie2/mmul_8_4.hpp
# is the only int4 mmul; no mmul_4_8, mmul_4_4 or mmul_16_4).
CASES = [
    # aie2 (Phoenix): the int4 pairs the headers declare
    ("aie2", "int8", "int4", 4, 16, 8, "acc32", "pass"),
    ("aie2", "uint8", "int4", 4, 16, 8, "acc32", "pass"),
    ("aie2", "uint8", "uint4", 4, 16, 8, "acc32", "pass"),
    ("aie2", "int8", "int4", 8, 16, 8, "acc32", "pass"),
    # aie2: the harness's own int8 control (the engine's shape, uint8 x int8)
    ("aie2", "uint8", "int8", 4, 8, 8, "acc32", "pass"),
    # aie2: int4 activations, W4A4, W4A16
    ("aie2", "int4", "int8", 4, 16, 8, "acc32", "fail"),
    ("aie2", "int4", "int8", 4, 8, 8, "acc32", "fail"),
    ("aie2", "int4", "int4", 4, 16, 8, "acc32", "fail"),
    ("aie2", "int4", "int4", 4, 32, 8, "acc32", "fail"),
    ("aie2", "int16", "int4", 4, 16, 8, "acc32", "fail"),
    ("aie2", "int16", "int4", 4, 16, 8, "acc64", "fail"),
    ("aie2", "int16", "int4", 4, 4, 8, "acc64", "fail"),
    # aie2p (Strix) controls: is a failure above aie2's, or aie_api's everywhere?
    ("aie2p", "int8", "int4", 4, 16, 16, "acc32", "pass"),
    ("aie2p", "int4", "int8", 4, 16, 8, "acc32", "fail"),
    ("aie2p", "int4", "int4", 4, 16, 8, "acc32", "fail"),
    ("aie2p", "int16", "int4", 4, 16, 8, "acc64", "fail"),
]

MAC_RE = re.compile(r"\b(vmac|vmul|vnegmac|vnegmul|vmsc|vaddmac|vsubmac|vnegmsc|vaddmsc|vsubmsc)(\.[\w.]+)?\b")
HOME = os.path.expanduser("~")


def scrub(text: str) -> str:
    for h in (HOME, HOME.replace("\\", "/")):
        text = text.replace(h, r"C:\Users\<user>")
    return text


# --- 1. control-word table -------------------------------------------------------------

ELEM_RE = re.compile(r"v\d+((?:u?int\d+)|bfloat16|float|cint\d+|cfloat|bfp16\w*)(_sparse)?")


def elem(arg: str) -> str:
    m = ELEM_RE.search(arg)
    return (m.group(1) + (m.group(2) or "")) if m else arg.strip()


INTRINSIC_RE = re.compile(r"INTRINSIC\([^)]*\)\s*(\w+)\s*\(([^)]*)\)\s*\{(.*?)\n\}", re.S)


def intrinsic_bodies(path: str) -> dict[str, list[str]]:
    """name -> the bodies of every overload of that intrinsic in a Peano vmult header."""
    out: dict[str, list[str]] = {}
    for m in INTRINSIC_RE.finditer(open(path, encoding="utf-8").read()):
        out.setdefault(m.group(1), []).append(m.group(3))
    return out


def four_bit_wrappers(path: str) -> list[str]:
    """Intrinsics with a 4-bit operand that build no control word of their own: software
    wrappers over other intrinsics. They are NOT in the control-word table."""
    lines = []
    for m in INTRINSIC_RE.finditer(open(path, encoding="utf-8").read()):
        name, args, body = m.group(1), m.group(2), m.group(3)
        if "_compute_control(" in body or not re.search(r"u?int4", args):
            continue
        lines.append(f"{name}({' '.join(args.split())}): "
                     f"{'unpacks to 8 bits' if 'unpack(' in body else 'no unpack'}")
    return sorted(set(lines))


def parse_controls(path: str) -> list[dict]:
    src = open(path, encoding="utf-8").read()
    out = []
    # INTRINSIC(ret) name(args) { ... compute_control(args); ... }
    for m in INTRINSIC_RE.finditer(src):
        name, args, body = m.group(1), m.group(2), m.group(3)
        c = re.search(r"_compute_control\(([^;]*)\);", body, re.S)
        if not c:
            continue
        cargs = [x.strip() for x in c.group(1).replace("\n", " ").split(",")]
        params = [x.strip() for x in args.split(",")]
        a = next((p for p in params if re.search(r"\ba$", p)), "")
        b = next((p for p in params if re.search(r"\bb$", p)), "")
        try:
            amode, bmode, variant = int(cargs[2]), int(cargs[3]), int(cargs[4])
        except ValueError:
            continue
        shape = re.sub(r"^(neg|add|sub)?(mul|mac|msc)_", "", name).replace("_conf", "")
        out.append(dict(name=name, a=elem(a), b=elem(b), amode=amode, bmode=bmode,
                        variant=variant, shape=shape))
    return out


def control_table(arch: str, rows: list[dict]) -> list[str]:
    lines = [f"{arch}: {len(rows)} MAC intrinsics parsed from {os.path.basename(VMULT[arch])}"]
    lines.append(f"  {'amode':>5} {'bmode':>5}  {'A element types':<34} {'B element types':<34} shapes")
    used = {}
    for r in rows:
        k = (r["amode"], r["bmode"])
        d = used.setdefault(k, dict(a=set(), b=set(), s=set()))
        d["a"].add(r["a"])
        d["b"].add(r["b"])
        d["s"].add(r["shape"])
    for (am, bm) in sorted(used):
        d = used[(am, bm)]
        lines.append(f"  {am:>5} {bm:>5}  {', '.join(sorted(d['a'])):<34} "
                     f"{', '.join(sorted(d['b'])):<34} {', '.join(sorted(d['s']))}")
    unused = [(am, bm) for am in range(4) for bm in range(4) if (am, bm) not in used]
    lines.append(f"  codes never emitted (amode, bmode): {unused}")
    four_a = sorted({r["a"] for r in rows if re.search(r"int4", r["a"])})
    four_b = sorted({(r["a"], r["b"]) for r in rows if re.search(r"int4", r["b"])})
    lines.append(f"  control words with a 4-bit A operand: {four_a or 'none'}")
    lines.append(f"  control words with a 4-bit B operand, as (A, B): {four_b or 'none'}")
    wr = four_bit_wrappers(VMULT[arch])
    lines.append(f"  4-bit-operand intrinsics with no control word of their own (wrappers): {len(wr)}")
    for w in wr[:6]:
        lines.append(f"    {w}")
    if len(wr) > 6:
        lines.append(f"    ... {len(wr) - 6} more; unpacking: "
                     f"{sum('unpacks' in w for w in wr)}/{len(wr)}")
    return lines


# --- 2. aie::mmul compiles -------------------------------------------------------------

def compile_case(target, ta, tb, m, k, n, acc, no_mmul, out):
    flags = [f for f in IRON_FLAGS if not f.startswith("--target=")]
    cmd = [CLANG, SOURCE, "-c", "-o", out, f"-I{INCLUDE}", *flags,
           f"--target={target}-none-unknown-elf",
           f"-DTA={ta}", f"-DTB={tb}", f"-DM_={m}", f"-DK_={k}", f"-DN_={n}", f"-DACC={acc}"]
    if no_mmul:
        cmd.append("-DNO_MMUL")
    p = subprocess.run(cmd, capture_output=True, text=True)
    first = ""
    if p.returncode != 0:
        errs = [ln for ln in p.stderr.splitlines() if "error:" in ln]
        mm = [ln for ln in errs if "mmul" in ln.lower()]
        first = (mm or errs or p.stderr.splitlines()[:1] or ["?"])[0]
        first = re.sub(r"^.*?error:\s*", "", first)
    return p.returncode == 0, first, cmd


UNPACK_RE = re.compile(r"\b[\w.]*unpack[\w.]*\b")


def mac_ops(obj: str, objdump: str) -> list[str]:
    text = aie_disasm.disassemble(obj, objdump)
    found = []
    for ln in text.splitlines():
        for rx in (MAC_RE, UNPACK_RE):
            for m in rx.finditer(ln):
                tok = m.group(0)
                if tok not in found:
                    found.append(tok)
    return found


def arch_macro(target: str) -> str:
    p = subprocess.run([CLANG, f"--target={target}-none-unknown-elf", "-dM", "-E", "-x", "c++",
                        os.devnull], capture_output=True, text=True)
    m = re.search(r"#define __AIE_ARCH__ (\d+)", p.stdout)
    return m.group(1) if m else "?"


def int4_branches(target: str) -> list[str]:
    """Which __AIE_ARCH__ branch of aie_api's mmul_8_4.hpp a target compiles, and whether it
    unpacks B to 8 bits before the MAC."""
    hdr = os.path.join(INCLUDE, "aie_api", "detail", target, "mmul_8_4.hpp")
    src = open(hdr, encoding="utf-8").read()
    arch = arch_macro(target)
    parts = re.split(r"^#(?:if|elif) __AIE_ARCH__ == (\d+)\s*$", src, flags=re.M)
    out = [f"{target}: the compiler defines __AIE_ARCH__ = {arch}; "
           f"{os.path.basename(hdr)} has {'no ' if len(parts) == 1 else ''}__AIE_ARCH__ branches"]
    branches = [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)] or [("(all)", src)]
    bodies = intrinsic_bodies(VMULT[target])
    for key, body in branches:
        body = re.split(r"^#(?:else|elif|endif)", body, flags=re.M)[0]
        macs = sorted(set(re.findall(r"::(\w*(?:mac|mul)\w*)\(", body)))
        if "unpack" in body:
            tag = "aie_api UNPACKS B to 8 bits, then an 8-bit MAC"
        elif any("unpack(" in b for mm in macs for b in bodies.get(mm, [])):
            tag = "the Peano intrinsic itself UNPACKS B to 8 bits, then 8-bit MACs"
        elif any("_compute_control(" in b for mm in macs for b in bodies.get(mm, [])):
            tag = "a 4-bit-B MAC control word (bmode 0) straight into the MAC"
        else:
            tag = "intrinsic definition not found in the vmult header"
        sel = "  <- this target" if key in (arch, "(all)") else ""
        out.append(f"  __AIE_ARCH__ == {key}: {tag}; calls {', '.join(macs)}{sel}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", default=None, help="also write the report here (UTF-8, scrubbed)")
    ap.add_argument("--objdump", default=None)
    args = ap.parse_args(argv)
    objdump = aie_disasm.find_objdump(args.objdump)

    rep = []
    rep.append("INT4 study, gate A: which int4 operand pairs does the toolchain lower?")
    rep.append(f"UTC: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    rep.append(f"MACHINE: {platform.node()}")
    try:
        commit = subprocess.run(["git", "-C", _ROOT, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "?"
    rep.append(f"COMMIT: {commit} (worktree-int4-study)")
    rep.append("COMMAND: python kernels/int4_study/isa_gate.py" + (f" --log {args.log}" if args.log else ""))
    dists = sorted(d for d in os.listdir(SITE) if d.endswith(".dist-info")
                   and (d.startswith("mlir_aie") or d.startswith("llvm_aie")))
    rep.append(f"TOOLCHAIN: {', '.join(d[:-10] for d in dists)}")
    rep.append("SCOPE: compile only. No NPU, no hardware context. Verdicts are about the")
    rep.append("  aie_api/Peano toolchain, never the silicon (see the docstring).")
    rep.append("")
    rep.append("PRE-REGISTERED (written before the run, from the header listing):")
    rep.append("  aie2: int8/uint8 x int4/uint4 compile; int4 x int8, int4 x int4 and int16 x int4 do not.")
    rep.append("  aie2 control word: no intrinsic with a 4-bit A; 4-bit B only against 8-bit A.")
    rep.append("  Gate A kills the aie_api/Peano DENSE path for int4 activations, W4A4 and W16A4 on aie2")
    rep.append("  if those mmuls fail while their harness controls compile.")
    rep.append("NOT PRE-REGISTERED: section 1's wrapper listing and section 3 were added after a dry")
    rep.append("  run showed aie2p's int8 x int4 object containing unpack instructions. They are")
    rep.append("  observations about how the toolchain lowers int4, not tested predictions.")
    rep.append("")

    rep.append("1. MAC CONTROL WORD: (amode, bmode) codes each target's intrinsics emit")
    for arch in ("aie2", "aie2p"):
        rows = parse_controls(VMULT[arch])
        rep.extend(control_table(arch, rows))
        rep.append("")

    rep.append("2. aie::mmul COMPILES (IRON's Peano flags; per-pair harness control = -DNO_MMUL)")
    hdr = f"  {'target':<6} {'A':<6} {'B':<6} {'MxKxN':<8} {'acc':<6} {'harness':<8} {'mmul':<6} {'expect':<7} {'match':<5} MAC ops / first error"
    rep.append(hdr)
    mism = 0
    with tempfile.TemporaryDirectory() as td:
        for i, (tgt, ta, tb, m, k, n, acc, exp) in enumerate(CASES):
            ctl_ok, ctl_err, _ = compile_case(tgt, ta, tb, m, k, n, acc, True, os.path.join(td, f"c{i}.o"))
            obj = os.path.join(td, f"m{i}.o")
            ok, err, _ = compile_case(tgt, ta, tb, m, k, n, acc, False, obj)
            got = "pass" if ok else "fail"
            match = "yes" if got == exp else "NO"
            if got != exp:
                mism += 1
            detail = ", ".join(mac_ops(obj, objdump)) if ok else err
            if not ctl_ok:
                detail = f"HARNESS FAILED ({ctl_err}) -- mmul verdict void; " + detail
            rep.append(f"  {tgt:<6} {ta:<6} {tb:<6} {f'{m}x{k}x{n}':<8} {acc:<6} "
                       f"{'ok' if ctl_ok else 'FAIL':<8} {got:<6} {exp:<7} {match:<5} {detail[:150]}")
    rep.append("")
    rep.append(f"expectation mismatches: {mism}")
    rep.append("")
    rep.append("3. HOW aie_api LOWERS int8 x int4 PER TARGET (aie_api detail/<target>/mmul_8_4.hpp)")
    for tgt in ("aie2", "aie2p"):
        rep.extend(int4_branches(tgt))
    text = scrub("\n".join(rep))
    print(text)
    if args.log:
        if os.path.exists(args.log):
            raise SystemExit(f"refusing to overwrite existing log {args.log}")
        with open(args.log, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
