"""Does the int8 engine's hot loop respond to the `ptr` pattern the way bf16's did?

Answers the static half of the port gate offline - object size, accumulator spills, hot-loop
bundle counts - before anything is built or run on a device. The tracked
kernels/aie2/conv_engine/engine.cc is never modified: variants are exact-text copies under
scratch/, the same way tools/engine_bf16_loop_variants.py treats the bf16 core.

The `ptr` transform is the one the bf16 sitting found and could not justify porting until it had
a measured bf16 win. It now has one, so this prices it on int8.

The transform is arithmetic-neutral by construction. `a + c * plane_bytes + aoff` and
`(a + aoff) + c * plane_bytes` are the same address, reached in the same order, feeding the same
multiply-accumulates, so a correct build is byte-exact and only the addressing code differs.

    bash scripts/research-lowlevel.sh --log results/aie/engine_int8_loop_variants_census_<date>.log \
        --checks-only -- bash scripts/research-iron.sh tools/engine_int8_loop_variants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from engine_epilogue_variants import census  # noqa: E402

SRC = ROOT / "kernels" / "aie2" / "conv_engine" / "engine.cc"
OUT = ROOT / "scratch" / "int8_loop_variants"

# --- stride-2 dual loop (engine.cc conv_pass) -------------------------------------------------
S2_OLD = """                for (int c = 0; c < ncin; ++c) {
                    const uint8_t *plane = a + c * d.plane_bytes;
                    auto [av0, od0] = aie::interleave_unzip(aie::load_unaligned_v<32>(plane + aoff0),
                                                            aie::load_unaligned_v<32>(plane + aoff0 + 32), 8);
                    V32u av1 = av0;
                    if (DUAL) {
                        auto [e1, o1] = aie::interleave_unzip(aie::load_unaligned_v<32>(plane + aoff1),
                                                              aie::load_unaligned_v<32>(plane + aoff1 + 32), 8);
                        av1 = e1;
                    }"""

S2_NEW = """                const uint8_t *p0 = a + aoff0;
                const uint8_t *p1 = a + aoff1;
                for (int c = 0; c < ncin; ++c, p0 += d.plane_bytes, p1 += d.plane_bytes) {
                    auto [av0, od0] = aie::interleave_unzip(aie::load_unaligned_v<32>(p0),
                                                            aie::load_unaligned_v<32>(p0 + 32), 8);
                    V32u av1 = av0;
                    if (DUAL) {
                        auto [e1, o1] = aie::interleave_unzip(aie::load_unaligned_v<32>(p1),
                                                              aie::load_unaligned_v<32>(p1 + 32), 8);
                        av1 = e1;
                    }"""

# --- stride-1 dual loop (engine.cc conv_pass) ------------------------------------------------
S1_OLD = """                for (int c = 0; c < ncin; ++c) {
                    const uint8_t *plane = a + c * d.plane_bytes;
                    V32u av0 = aie::load_unaligned_v<32>(plane + aoff0);
                    V32u av1 = av0;
                    if (DUAL)
                        av1 = aie::load_unaligned_v<32>(plane + aoff1);"""

S1_NEW = """                const uint8_t *p0 = a + aoff0;
                const uint8_t *p1 = a + aoff1;
                for (int c = 0; c < ncin; ++c, p0 += d.plane_bytes, p1 += d.plane_bytes) {
                    V32u av0 = aie::load_unaligned_v<32>(p0);
                    V32u av1 = av0;
                    if (DUAL)
                        av1 = aie::load_unaligned_v<32>(p1);"""

# --- fused_stage1's own loop over input blocks, a separate lever ------------------------------
FS1_OLD = """                    for (int c = 0; c < 2; ++c) {
                        const uint8_t *plane = apkt + c * plane_bytes;
                        V32u av = aie::load_unaligned_v<32>(plane + aoff);"""

FS1_NEW = """                    const uint8_t *pf = apkt + aoff;
                    for (int c = 0; c < 2; ++c, pf += plane_bytes) {
                        V32u av = aie::load_unaligned_v<32>(pf);"""


def apply(text: str, pairs) -> str:
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"anchor matched {text.count(old)} times, not 1:\n{old[:90]}...")
        text = text.replace(old, new)
    return text


VARIANTS = {
    "base": [],
    "ptr": [(S2_OLD, S2_NEW), (S1_OLD, S1_NEW)],
    "ptr_fused": [(S2_OLD, S2_NEW), (S1_OLD, S1_NEW), (FS1_OLD, FS1_NEW)],
}


def main() -> int:
    base = SRC.read_text(encoding="utf-8")
    for name, pairs in VARIANTS.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        src = d / "engine.cc"
        src.write_text(apply(base, pairs), encoding="utf-8")
        print(f"{name:10s} {census(src, d)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
