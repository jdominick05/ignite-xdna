"""Which kernel source each bf16 engine design in the jit cache was really built from.

The jit keys a design on the generator's bytecode and its CompileTime values before the generator
body runs. A kernel source chosen inside the body - a module global read at generation time - is
not in that key, so a second source silently reuses the first source's xclbin, and a harness that
prints the source it MEANT to build has printed nothing about the kernel that RAN. This tool reads
the cache itself: every entry keeps a copy of the source it compiled (`engine_bf16.cc`) and the
object it linked (`engine_bf16.o`), so the source is named by digest against the variant copies
under scratch/bf16_loop_variants and the object by its .text size.

Static: it opens no device and compiles nothing. Run where Peano's llvm-size is reachable:

    bash scripts/research-iron.sh tools/engine_bf16_cache_audit.py

The design's in-core repeat and packet count are read off its aie.mlir (the loop trip constants),
so a timing design (repeat 128) is told from a correctness design (repeat 1).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.engine_linked_size import text_sections  # noqa: E402


def variant_digests(variants_dir: Path) -> dict[str, str]:
    named = {"repository": ROOT / "kernels/bf16_conv/engine_bf16.cc"}
    if variants_dir.is_dir():
        for d in sorted(variants_dir.iterdir()):
            if (d / "engine_bf16.cc").exists():
                named[d.name] = d / "engine_bf16.cc"
    by_digest: dict[str, str] = {}
    for name, p in named.items():                   # an unmodified copy shares the repository's digest
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        by_digest[digest] = f"{by_digest[digest]}={name}" if digest in by_digest else name
    return by_digest


def trip_counts(mlir: str) -> list[int]:
    # The core's outer loop runs "forever" (an INT64_MAX trip count); only the finite trips - the
    # packet count and the in-core repeat - say what the design is.
    return sorted({int(x) for x in re.findall(r"arith\.constant (\d+) : index", mlir) if 1 < int(x) < 1 << 62})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "scratch/iron-cache"))
    ap.add_argument("--variants", default=str(ROOT / "scratch/bf16_loop_variants"))
    args = ap.parse_args()
    digests = variant_digests(Path(args.variants))
    rows = []
    for d in sorted(Path(args.cache).iterdir()):
        obj = d / "engine_bf16.o"
        if not obj.exists():
            continue
        src = d / "engine_bf16.cc"
        which = digests.get(hashlib.sha256(src.read_bytes()).hexdigest(), "unknown") if src.exists() else "no copy"
        mlir = (d / "aie.mlir").read_text(encoding="utf-8", errors="replace") if (d / "aie.mlir").exists() else ""
        trips = trip_counts(mlir)
        built = datetime.fromtimestamp(obj.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append((built, d.name[:8], sum(s for _, s in text_sections(obj)), which, trips))
    print(f"ENGINE_BF16_CACHE {len(rows)} engine designs under {Path(args.cache).name}; "
          f"sources known by digest: {sorted(set(digests.values()))}")
    for built, entry, text, which, trips in sorted(rows):
        print(f"  {built}  {entry}  object .text {text:5d} B  source={which:12s} finite loop trips {trips or '[1]'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
