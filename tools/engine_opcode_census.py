"""Census an engine core program once per opcode combination.

Each variant strips dispatch cases from a copy of the kernel source (the repository file is
never modified; unused inline functions are not emitted, and unreferenced functions of the
anonymous namespace are dropped, so a cut case costs nothing), compiles the copy with Peano and
prints the .text / accumulator-spill census. Two ways to ask:

  --ops A B ...      what each opcode ADDS: one row with none of them dispatched, one per opcode
                     dispatched alone, one with all of them. This is the original reading.
  --without A B ...  what removing a dispatch group FREES: the untouched program, one row per
                     group removed alone, one with all of them removed.

  --source FILE      census a different kernel as it stands. A source with none of the named
                     dispatch cases (the bf16 engine core has no switch) gets the single row.

Rows end with the object budget: the 16-core design links 880 B of IRON core program beside the
object (tools/engine_linked_size.py), so "under 16,384" is not the test - 15,504 is.

Run in the mlir-aie ironenv (bash scripts/research-iron.sh):

    bash scripts/research-iron.sh tools/engine_opcode_census.py [--ops MUL SCALE POOL]
    bash scripts/research-iron.sh tools/engine_opcode_census.py --without FUSED_CONV MUL SCALE POOL
    bash scripts/research-iron.sh tools/engine_opcode_census.py --source kernels/bf16_conv/engine_bf16.cc
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from engine_epilogue_variants import census  # noqa: E402

SRC = ROOT / "kernels" / "aie2" / "conv_engine" / "engine.cc"
OUT = ROOT / "scratch" / "census_perop"

CALL_OF = {
    "MUL": "mul_tile(d, apkt, psum, out)",
    "SCALE": "scale_tile(d, apkt, wpkt, psum, out)",
    "POOL": "avgpool_tile(d, apkt, psum, out)",
    "FUSED_CONV": "fused_conv_tile(hdr, apkt, psum, out, core_row)",
}


def case_block(op):
    return f"    case OP_{op}:\n        {CALL_OF[op]};\n        break;\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops", nargs="+", default=None, choices=sorted(CALL_OF))
    ap.add_argument("--without", nargs="+", default=None, choices=sorted(CALL_OF))
    ap.add_argument("--source", default=str(SRC), help="the kernel source to census (default: the int8 engine)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.ops and args.without:
        ap.error("--ops and --without ask different questions; give one")
    out, source = Path(args.out), Path(args.source)
    base = source.read_text(encoding="utf-8")
    named = args.without or args.ops or ["MUL", "SCALE", "POOL"]
    blocks = {op: case_block(op) for op in named if base.count(case_block(op)) == 1}
    missing = [op for op in named if op not in blocks]
    if missing and (source.resolve() == SRC.resolve() or blocks):
        raise SystemExit(f"{missing}: dispatch case not found verbatim in {source.name}")
    if not blocks:
        variants = [("as-is", base)]
    elif args.without:
        variants = [("all", base)]
        variants += [(f"-{op}", base.replace(blocks[op], "")) for op in blocks]
        variants.append(("-every", _strip(base, blocks, keep=None)))
    else:
        variants = [("none", _strip(base, blocks, keep=None))]
        variants += [(op, _strip(base, blocks, keep=op)) for op in blocks]
        variants.append(("all", base))
    for label, text in variants:
        d = out / ("minus_" + label[1:] if label.startswith("-") else label).lower()
        d.mkdir(parents=True, exist_ok=True)
        src = d / source.name
        src.write_text(text, encoding="utf-8")
        print(f"{label:12s} {census(src, d)}", flush=True)


def _strip(base, blocks, keep):
    text = base
    for op, block in blocks.items():
        if op != keep:
            text = text.replace(block, "")
    return text


if __name__ == "__main__":
    main()
