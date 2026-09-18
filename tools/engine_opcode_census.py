"""Census the persistent conv-engine core once per opcode combination.

Each variant strips the new-op dispatch cases not being priced out of a copy of
kernels/aie2/conv_engine/engine.cc (the repository file is never modified; unused
inline functions are not emitted, so a cut case costs nothing), compiles the copy
with Peano and prints the .text / accumulator-spill census. One row per new
opcode, one for the untouched baseline, one for all of them together.

Run in the mlir-aie ironenv (bash scripts/research-iron.sh):

    bash scripts/research-iron.sh tools/engine_opcode_census.py [--ops MUL SCALE POOL]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from engine_epilogue_variants import census  # noqa: E402

SRC = ROOT / "kernels" / "aie2" / "conv_engine" / "engine.cc"
OUT = ROOT / ".qwen" / "tmp" / "census_perop"

FN_OF = {"MUL": "mul_tile", "SCALE": "scale_tile", "POOL": "avgpool_tile"}


def case_block(op):
    args = {"MUL": "d, apkt, psum, out", "SCALE": "d, apkt, wpkt, psum, out", "POOL": "d, apkt, psum, out"}[op]
    return f"    case OP_{op}:\n        {FN_OF[op]}({args});\n        break;\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops", nargs="+", default=["MUL", "SCALE", "POOL"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    base = SRC.read_text(encoding="utf-8")
    blocks = {op: case_block(op) for op in args.ops}
    for op, block in blocks.items():
        assert base.count(block) == 1, f"{op}: dispatch case not found verbatim in engine.cc"
    variants = [("none", _strip(base, blocks, keep=None))]
    for op in args.ops:
        variants.append((op, _strip(base, blocks, keep=op)))
    variants.append(("all", base))
    for label, text in variants:
        d = out / label.lower()
        d.mkdir(parents=True, exist_ok=True)
        src = d / "engine.cc"
        src.write_text(text, encoding="utf-8")
        print(f"{label:6s} {census(src, d)}", flush=True)


def _strip(base, blocks, keep):
    text = base
    for op, block in blocks.items():
        if op != keep:
            text = text.replace(block, "")
    return text


if __name__ == "__main__":
    main()
