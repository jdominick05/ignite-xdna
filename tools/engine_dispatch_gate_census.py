"""Price the compile-time dispatch gate through the mechanism production actually uses.

WHY THIS IS NOT engine_opcode_census.py --without. That tool measures what removing a dispatch
group frees by TEXT-STRIPPING the case out of a copy. Production instead compiles the case out
with `-DENGINE_OP_<NAME>=0` (kernels/aie2/conv_engine/design.py::gate_flags), which is a
different mechanism, so the price does not transfer by assertion. This tool compiles both and
compares the objects BYTE FOR BYTE. If they are identical, the earlier census's numbers are the
gated build's numbers and no separate measurement is owed - and, because every remaining
function compiled to the same bytes, no latency claim is owed either.

It also prints what each group costs on its own, through the -D path, so the per-group prices in
docs/BENCHMARKS.md have a production-mechanism reading beside the text-strip one.

Static: nothing is run on the device. This establishes object sizes, not a latency.

    bash scripts/research-lowlevel.sh --log results/aie/engine_dispatch_gate_census_<date>.log \\
        --checks-only -- bash scripts/research-iron.sh tools/engine_dispatch_gate_census.py

    # optionally, at a landing: prove the guards are inert against the kernel before them
    ... tools/engine_dispatch_gate_census.py --baseline <a pre-gate engine.cc>
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels" / "aie2" / "conv_engine" / "engine.cc"
OUT = ROOT / "scratch" / "dispatch_gate_census"

# Must match kernels/aie2/conv_engine/design.py::GATEABLE_OPS.
OPS = ["FUSED_CONV", "MUL", "SCALE", "POOL"]
CALL_OF = {
    "MUL": "mul_tile(d, apkt, psum, out)",
    "SCALE": "scale_tile(d, apkt, wpkt, psum, out)",
    "POOL": "avgpool_tile(d, apkt, psum, out)",
    "FUSED_CONV": "fused_conv_tile(hdr, apkt, psum, out, core_row)",
}
PROGRAM_MEMORY = 16384
CORE_WRAPPER = 880          # tools/engine_linked_size.py: IRON's core program beside the object
BUDGET = PROGRAM_MEMORY - CORE_WRAPPER


def compile_one(src: Path, obj: Path, defines):
    from aie.utils import config
    peano = Path(config.peano_install_dir()) / "bin"
    obj.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(peano / "clang++.exe"), "-O2", "-std=c++20", "--target=aie2-none-unknown-elf",
           "-nostdlib", "-DNDEBUG", "-D__AIE_API_AIE_ADF_HPP__"]
    cmd += [f"-D{d}" for d in defines]
    cmd += ["-I", config.cxx_header_path(), "-c", str(src), "-o", str(obj)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"compile failed ({obj.name}):\n{r.stderr[-3000:]}")
    size = subprocess.run([str(peano / "llvm-size.exe"), "-A", str(obj)],
                          capture_output=True, text=True).stdout
    text = sum(int(m.group(1)) for m in re.finditer(r"^\.text\S*\s+(\d+)", size, re.M))
    return text, sha256(obj.read_bytes()).hexdigest()


def strip_cases(text: str, eol: str) -> str:
    for op in OPS:
        block = f"    case OP_{op}:{eol}        {CALL_OF[op]};{eol}        break;{eol}"
        if text.count(block) != 1:
            raise SystemExit(f"case OP_{op} not found verbatim - the census anchors have moved")
        text = text.replace(block, "")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(SRC))
    ap.add_argument("--baseline", default=None,
                    help="a pre-gate engine.cc; checks the guards are inert when none is set")
    args = ap.parse_args()

    src = Path(args.source)
    gated = src.read_text(encoding="utf-8", newline="")
    if "#if ENGINE_OP_FUSED_CONV" not in gated:
        raise SystemExit(f"{src} carries no dispatch gate")
    work = OUT / "src"
    work.mkdir(parents=True, exist_ok=True)
    copy = work / "engine.cc"
    copy.write_text(gated, encoding="utf-8", newline="")

    ungated_text, ungated_sha = compile_one(copy, OUT / "obj" / "ungated.o", [])
    print(f"{'build':<34s} {'.text':>8s} {'freed':>7s} {'headroom':>9s}  object sha256")
    print(f"{'ungated (every case compiled in)':<34s} {ungated_text:8d} {'-':>7s} "
          f"{BUDGET - ungated_text:9d}  {ungated_sha[:16]}")

    for op in OPS:
        t, s = compile_one(copy, OUT / "obj" / f"minus_{op.lower()}.o", [f"ENGINE_OP_{op}=0"])
        print(f"{'  without ' + op:<34s} {t:8d} {ungated_text - t:7d} "
              f"{BUDGET - t:9d}  {s[:16]}")

    all_off = [f"ENGINE_OP_{op}=0" for op in OPS]
    gated_text, gated_sha = compile_one(copy, OUT / "obj" / "gated.o", all_off)
    print(f"{'  without all four':<34s} {gated_text:8d} {ungated_text - gated_text:7d} "
          f"{BUDGET - gated_text:9d}  {gated_sha[:16]}")
    print()
    print(f"ENGINE_DISPATCH_GATE_FREED {ungated_text - gated_text} B "
          f"({ungated_text} -> {gated_text}); headroom {BUDGET - ungated_text} -> "
          f"{BUDGET - gated_text} B of the {BUDGET} B object budget")
    print()

    strip_src = OUT / "strip" / "engine.cc"
    strip_src.parent.mkdir(parents=True, exist_ok=True)
    strip_src.write_text(strip_cases(gated, "\r\n" if "\r\n" in gated else "\n"),
                         encoding="utf-8", newline="")
    strip_text, strip_sha = compile_one(strip_src, OUT / "obj" / "stripped.o", [])
    ok = strip_sha == gated_sha
    print(f"text-stripped (what --without prices)  .text {strip_text:6d}  {strip_sha[:16]}")
    print(f"EQUIVALENCE  the -D gate is byte-identical to the text strip : "
          f"{'PASS' if ok else 'FAIL'}")

    inert = True
    if args.baseline:
        base = Path(args.baseline).read_text(encoding="utf-8", newline="")
        if "ENGINE_OP_" in base:
            raise SystemExit(f"{args.baseline} is already gated; pass the kernel from before it")
        b_src = OUT / "baseline" / "engine.cc"
        b_src.parent.mkdir(parents=True, exist_ok=True)
        b_src.write_text(base, encoding="utf-8", newline="")
        b_text, b_sha = compile_one(b_src, OUT / "obj" / "baseline.o", [])
        inert = b_sha == ungated_sha
        print(f"baseline (pre-gate kernel)             .text {b_text:6d}  {b_sha[:16]}")
        print(f"INERTNESS    the guards change nothing when none is set  : "
              f"{'PASS' if inert else 'FAIL'}")

    return 0 if (ok and inert) else 1


if __name__ == "__main__":
    sys.exit(main())
