"""Write arithmetic-neutral variants of the bf16 engine core's hot loop and census each.

Every variant issues the SAME multiply-accumulates in the SAME order as
kernels/bf16_conv/engine_bf16.cc - only how the addresses are formed and how the loop nest is
shaped changes - so byte-exactness is preserved by construction and what is being searched for is
fewer bundles per hardware-loop iteration (a hardware-loop body's bundle count is its cycle count
on AIE2) and fewer hardware-loop entries. The repository's kernel is never modified: variants are
copies under --out, produced by exact-text replacement that fails loudly if the source has moved.

  base      the kernel as committed
  ptr       running plane pointers: `p += plane_elems` replaces `act + c * plane_elems + aoff`
  hint      load_unaligned_v(p, 8): every activation pointer is pixel-aligned, and says so
  ptr+hint  both
  flat      ONE hardware loop per pass over all k*k*ncin taps, offsets from a table built once per
            packet, instead of k*k loops of ncin trips each
  flat+hint both

Static only: it compiles objects and reads them. A variant that wins here still has to win on the
device (kernels/bf16_conv/engine_bf16.py --source <variant> --sweep / --bench).

THE SEARCH IS OVER AND `chunk` + `hoist` WON, so they are now IN engine_bf16.cc rather than beside
it. Every transform here except `base` therefore targets text the kernel no longer has, and `swap`
raises rather than producing a variant that silently is not the one named - which is the whole
reason it fails loudly. `base` still works and is the offline compile-and-census check for the
kernel as committed. Re-deriving the old table means checking out the commit before the loop landed;
the numbers themselves are in docs/BENCHMARKS.md and the logs they cite are not going anywhere.

    bash scripts/research-iron.sh tools/engine_bf16_loop_variants.py [--only ptr flat] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from engine_epilogue_variants import census  # noqa: E402

SRC = ROOT / "kernels" / "bf16_conv" / "engine_bf16.cc"
OUT = ROOT / "scratch" / "bf16_loop_variants"


def swap(text, old, new, count=1):
    assert text.count(old) == count, f"expected {count} of {old[:60]!r}, found {text.count(old)}"
    return text.replace(old, new)


def ptr(t):
    t = swap(t, "    const int ncin = d.ncin;\n", "    const int ncin = d.ncin;\n    const int plane_elems = d.plane_elems;\n")
    # stride 2
    t = swap(t, "                for (int c = 0; c < ncin; ++c) {\n"
                "                    const bfloat16 *plane = act + c * d.plane_elems;\n"
                "                    auto [av0, odd0] = aie::interleave_unzip(\n"
                "                        aie::load_unaligned_v<MMUL::size_A>(plane + aoff0),\n"
                "                        aie::load_unaligned_v<MMUL::size_A>(plane + aoff0 + MMUL::size_A), 8);\n",
             "                const bfloat16 *p0 = act + aoff0;\n"
             "                const bfloat16 *p1 = act + aoff1;\n"
             "                for (int c = 0; c < ncin; ++c, p0 += plane_elems, p1 += plane_elems) {\n"
             "                    auto [av0, odd0] = aie::interleave_unzip(\n"
             "                        aie::load_unaligned_v<MMUL::size_A>(p0),\n"
             "                        aie::load_unaligned_v<MMUL::size_A>(p0 + MMUL::size_A), 8);\n")
    t = swap(t, "                            aie::load_unaligned_v<MMUL::size_A>(plane + aoff1),\n"
                "                            aie::load_unaligned_v<MMUL::size_A>(plane + aoff1 + MMUL::size_A), 8);\n",
             "                            aie::load_unaligned_v<MMUL::size_A>(p1),\n"
             "                            aie::load_unaligned_v<MMUL::size_A>(p1 + MMUL::size_A), 8);\n")
    # stride 1
    t = swap(t, "                for (int c = 0; c < ncin; ++c) {\n"
                "                    const bfloat16 *plane = act + c * d.plane_elems;\n"
                "                    // Four consecutive output columns",
             "                const bfloat16 *p0 = act + aoff0;\n"
             "                const bfloat16 *p1 = act + aoff1;\n"
             "                for (int c = 0; c < ncin; ++c, p0 += plane_elems, p1 += plane_elems) {\n"
             "                    // Four consecutive output columns")
    t = swap(t, "                    const VA av0 = aie::load_unaligned_v<MMUL::size_A>(plane + aoff0);\n",
             "                    const VA av0 = aie::load_unaligned_v<MMUL::size_A>(p0);\n")
    t = swap(t, "                        av1 = aie::load_unaligned_v<MMUL::size_A>(plane + aoff1);\n",
             "                        av1 = aie::load_unaligned_v<MMUL::size_A>(p1);\n")
    return t


def hint(t):
    import re
    out, n = re.subn(r"(aie::load_unaligned_v<MMUL::size_A>\()([^;]*?)(\)[,;])", r"\1\2, 8\3", t)
    assert n >= 6, n
    return out


def flat(t):
    """One hardware loop per pass: the (ky, kx, c) nest becomes a walk of an offset table."""
    t = swap(t, "    int rows_in, cols_in, plane_elems;\n    int phase;\n",
             "    int rows_in, cols_in, plane_elems;\n    int phase;\n    int ntaps;\n    const int *tap;\n")
    start = t.index("    const int ncin = d.ncin;\n")
    end = t.index("    if (d.flags & (F_EMIT | F_HOLD)) {\n        bfloat16 *dst")
    body = (
        "    const int ntaps = d.ntaps;\n"
        "    const int *tap = d.tap;\n"
        "    const bfloat16 *wp = w;\n\n"
        "    if (d.stride == 2) {\n"
        "        const bfloat16 *b0 = act + (2 * r * d.cols_in + 8 * g0) * 8;\n"
        "        const bfloat16 *b1 = b0 + (g1 - g0) * 64;\n"
        "        for (int i = 0; i < ntaps; ++i) {\n"
        "            const int off = tap[i];\n"
        "            auto [av0, odd0] = aie::interleave_unzip(\n"
        "                aie::load_unaligned_v<MMUL::size_A>(b0 + off),\n"
        "                aie::load_unaligned_v<MMUL::size_A>(b0 + off + MMUL::size_A), 8);\n"
        "            VA av1 = av0;\n"
        "            if (DUAL) {\n"
        "                auto [e1, odd1] = aie::interleave_unzip(\n"
        "                    aie::load_unaligned_v<MMUL::size_A>(b1 + off),\n"
        "                    aie::load_unaligned_v<MMUL::size_A>(b1 + off + MMUL::size_A), 8);\n"
        "                av1 = e1;\n"
        "            }\n"
        "#pragma unroll\n"
        "            for (int b = 0; b < NCO; ++b) {\n"
        "                const VB wv = aie::load_v<MMUL::size_B>(wp);\n"
        "                wp += MMUL::size_B;\n"
        "                acc0[b].mac(av0, wv);\n"
        "                if (DUAL)\n"
        "                    acc1[b].mac(av1, wv);\n"
        "            }\n"
        "        }\n"
        "    } else {\n"
        "        const bfloat16 *b0 = act + (r * d.cols_in + 4 * g0) * 8;\n"
        "        const bfloat16 *b1 = b0 + (g1 - g0) * 32;\n"
        "        for (int i = 0; i < ntaps; ++i) {\n"
        "            const int off = tap[i];\n"
        "            const VA av0 = aie::load_unaligned_v<MMUL::size_A>(b0 + off);\n"
        "            VA av1 = av0;\n"
        "            if (DUAL)\n"
        "                av1 = aie::load_unaligned_v<MMUL::size_A>(b1 + off);\n"
        "#pragma unroll\n"
        "            for (int b = 0; b < NCO; ++b) {\n"
        "                const VB wv = aie::load_v<MMUL::size_B>(wp);\n"
        "                wp += MMUL::size_B;\n"
        "                acc0[b].mac(av0, wv);\n"
        "                if (DUAL)\n"
        "                    acc1[b].mac(av1, wv);\n"
        "            }\n"
        "        }\n"
        "    }\n\n"
    )
    t = t[:start] + body + t[end:]
    # Build the table once per packet, in the core's walk order: ky, kx, input channel block.
    t = swap(t, "    const Hdr d = read_header(wpkt, core_row);\n    if (d.op == OP_NOP)\n        return;\n",
             "    Hdr d = read_header(wpkt, core_row);\n    if (d.op == OP_NOP)\n        return;\n"
             "    int tap[64];\n    int n = 0;\n"
             "    for (int ky = 0; ky < d.k; ++ky)\n        for (int kx = 0; kx < d.k; ++kx)\n"
             "            for (int c = 0; c < d.ncin; ++c)\n"
             "                tap[n++] = (ky * d.cols_in + kx) * 8 + c * d.plane_elems;\n"
             "    d.ntaps = n;\n    d.tap = tap;\n")
    return t


def flatptr(t):
    """`flat` with running pointers: the table holds the DELTA to the next tap, added after the loads."""
    t = flat(t)
    t = swap(t, "                tap[n++] = (ky * d.cols_in + kx) * 8 + c * d.plane_elems;\n"
                "    d.ntaps = n;\n    d.tap = tap;\n",
             "                tap[n++] = (ky * d.cols_in + kx) * 8 + c * d.plane_elems;\n"
             "    tap[n] = tap[n - 1];\n"
             "    for (int i = 0; i < n; ++i)\n        tap[i] = tap[i + 1] - tap[i];   // now the step to the next tap\n"
             "    d.ntaps = n;\n    d.tap = tap;\n")
    t = swap(t, "        const bfloat16 *b0 = act + (2 * r * d.cols_in + 8 * g0) * 8;\n"
                "        const bfloat16 *b1 = b0 + (g1 - g0) * 64;\n",
             "        const bfloat16 *b0 = act + (2 * r * d.cols_in + 8 * g0) * 8;\n"
             "        const bfloat16 *b1 = b0 + (g1 - g0) * 64;\n        const int off = 0;\n")
    t = swap(t, "        const bfloat16 *b0 = act + (r * d.cols_in + 4 * g0) * 8;\n"
                "        const bfloat16 *b1 = b0 + (g1 - g0) * 32;\n",
             "        const bfloat16 *b0 = act + (r * d.cols_in + 4 * g0) * 8;\n"
             "        const bfloat16 *b1 = b0 + (g1 - g0) * 32;\n        const int off = 0;\n")
    t = swap(t, "            const int off = tap[i];\n", "            const int step = tap[i];\n", count=2)
    # advance both window pointers once the iteration's loads are issued
    t = swap(t, "#pragma unroll\n            for (int b = 0; b < NCO; ++b) {\n",
             "            b0 += step;\n            b1 += step;\n#pragma unroll\n            for (int b = 0; b < NCO; ++b) {\n", count=2)
    return t


TAPS = """// One input channel block's worth of one kernel tap: the activation windows of both column groups
// and the four output blocks' weights. Always inlined, so the accumulators stay in registers.
template <bool DUAL>
__attribute__((always_inline))
inline void tap_s1(MMUL *acc0, MMUL *acc1, const bfloat16 *p0, const bfloat16 *p1, const bfloat16 *&wp) {
    const VA av0 = aie::load_unaligned_v<MMUL::size_A>(p0);
    VA av1 = av0;
    if (DUAL)
        av1 = aie::load_unaligned_v<MMUL::size_A>(p1);
#pragma unroll
    for (int b = 0; b < NCO; ++b) {
        const VB wv = aie::load_v<MMUL::size_B>(wp);
        wp += MMUL::size_B;
        acc0[b].mac(av0, wv);
        if (DUAL)
            acc1[b].mac(av1, wv);
    }
}

// Stride 2: output pixel x reads input pixel 2x + kx, so load eight pixels as two 4-pixel vectors
// and keep the even ones. The unzip step is 8 ELEMENTS - one pixel's channels.
template <bool DUAL>
__attribute__((always_inline))
inline void tap_s2(MMUL *acc0, MMUL *acc1, const bfloat16 *p0, const bfloat16 *p1, const bfloat16 *&wp) {
    auto [av0, odd0] = aie::interleave_unzip(aie::load_unaligned_v<MMUL::size_A>(p0),
                                             aie::load_unaligned_v<MMUL::size_A>(p0 + MMUL::size_A), 8);
    VA av1 = av0;
    if (DUAL) {
        auto [e1, odd1] = aie::interleave_unzip(aie::load_unaligned_v<MMUL::size_A>(p1),
                                                aie::load_unaligned_v<MMUL::size_A>(p1 + MMUL::size_A), 8);
        av1 = e1;
    }
#pragma unroll
    for (int b = 0; b < NCO; ++b) {
        const VB wv = aie::load_v<MMUL::size_B>(wp);
        wp += MMUL::size_B;
        acc0[b].mac(av0, wv);
        if (DUAL)
            acc1[b].mac(av1, wv);
    }
}

// The input channel blocks of one tap, as CONSTANT-TRIP loops: 4s, then a 2, then a 1. The count is
// run-time, but a loop whose trip count the compiler can see is one it rotates, issuing the next
// iteration's loads under this iteration's multiply-accumulates (measured on the object: 16 bundles
// per 8 MACs against 24 for the run-time loop). The walk order is unchanged: block 0, 1, 2, ...
#define BLOCKS(TAP)                                                                        \\
    do {                                                                                   \\
        int n = ncin;                                                                      \\
        for (; n >= 4; n -= 4)                                                             \\
            for (int j = 0; j < 4; ++j, p0 += plane_elems, p1 += plane_elems)              \\
                TAP<DUAL>(acc0, acc1, p0, p1, wp);                                         \\
        if (n & 2)                                                                         \\
            for (int j = 0; j < 2; ++j, p0 += plane_elems, p1 += plane_elems)              \\
                TAP<DUAL>(acc0, acc1, p0, p1, wp);                                         \\
        if (n & 1)                                                                         \\
            TAP<DUAL>(acc0, acc1, p0, p1, wp);                                             \\
    } while (0)

"""


def chunk(t):
    """Running pointers AND constant-trip loops over the input channel blocks."""
    t = swap(t, "// One output row of one tile, for either one column group", TAPS + "// One output row of one tile, for either one column group")
    start = t.index("    const int ncin = d.ncin;\n")
    end = t.index("    if (d.flags & (F_EMIT | F_HOLD)) {\n        bfloat16 *dst")
    body = (
        "    const int ncin = d.ncin;\n"
        "    const int plane_elems = d.plane_elems;\n"
        "    const int k = d.k;\n"
        "    const bfloat16 *wp = w;\n\n"
        "    if (d.stride == 2) {\n"
        "        for (int ky = 0; ky < k; ++ky) {\n"
        "            for (int kx = 0; kx < k; ++kx) {\n"
        "                const bfloat16 *p0 = act + ((2 * r + ky) * d.cols_in + 8 * g0 + kx) * 8;\n"
        "                const bfloat16 *p1 = p0 + (g1 - g0) * 64;\n"
        "                BLOCKS(tap_s2);\n"
        "            }\n"
        "        }\n"
        "    } else {\n"
        "        for (int ky = 0; ky < k; ++ky) {\n"
        "            for (int kx = 0; kx < k; ++kx) {\n"
        "                const bfloat16 *p0 = act + ((r + ky) * d.cols_in + 4 * g0 + kx) * 8;\n"
        "                const bfloat16 *p1 = p0 + (g1 - g0) * 32;\n"
        "                BLOCKS(tap_s1);\n"
        "            }\n"
        "        }\n"
        "    }\n\n"
    )
    return t[:start] + body + t[end:]


def hoist(t):
    """The bias vectors formed once per packet instead of once per pass (milestone 1 does this)."""
    t = swap(t, "inline void conv_pass(const Hdr &d, const bfloat16 *act, const bfloat16 *w, const bfloat16 *bias,\n"
                "                      bfloat16 *out, float *psum, int r, int g0, int g1) {\n",
             "inline void conv_pass(const Hdr &d, const bfloat16 *act, const bfloat16 *w, const VC *bv,\n"
             "                      bfloat16 *out, float *psum, int r, int g0, int g1) {\n")
    t = swap(t, "        for (int b = 0; b < NCO; ++b) {\n"
                "            // The bias arrives pre-replicated to the accumulator's 4x4 shape, because bf16 has no\n"
                "            // 4-element load - aie::load_v<4> does not compile for bfloat16, 16 is the smallest.\n"
                "            aie::accum<accfloat, MMUL::size_C> ba;\n"
                "            ba.from_vector(aie::load_v<MMUL::size_C>(bias + b * MMUL::size_C));\n"
                "            const VC bv = ba.template to_vector<float>();\n"
                "            acc0[b] = MMUL(bv);\n"
                "            if (DUAL)\n"
                "                acc1[b] = MMUL(bv);\n"
                "        }\n",
             "        for (int b = 0; b < NCO; ++b) {\n"
             "            acc0[b] = MMUL(bv[b]);\n"
             "            if (DUAL)\n"
             "                acc1[b] = MMUL(bv[b]);\n"
             "        }\n")
    t = swap(t, "    const bfloat16 *w = base + W_OFFSET_ELEMS;\n",
             "    const bfloat16 *w = base + W_OFFSET_ELEMS;\n"
             "    // The bias arrives pre-replicated to the accumulator's 4x4 shape (bf16 has no 4-element\n"
             "    // load); it is widened once here rather than in every pass.\n"
             "    VC bv[NCO];\n"
             "#pragma unroll\n"
             "    for (int b = 0; b < NCO; ++b) {\n"
             "        aie::accum<accfloat, MMUL::size_C> ba;\n"
             "        ba.from_vector(aie::load_v<MMUL::size_C>(bias + b * MMUL::size_C));\n"
             "        bv[b] = ba.template to_vector<float>();\n"
             "    }\n")
    t = swap(t, "conv_pass<true>(d, apkt, w, bias, out, psum, r, g, g + 1);", "conv_pass<true>(d, apkt, w, bv, out, psum, r, g, g + 1);")
    t = swap(t, "conv_pass<false>(d, apkt, w, bias, out, psum, r, g, g);", "conv_pass<false>(d, apkt, w, bv, out, psum, r, g, g);")
    return t


def fixed(ncin=None, k=None):
    """`ptr` with a trip count the compiler can see: is that what lets it pipeline loads under MACs?"""
    def make(t):
        t = ptr(t)
        if ncin is not None:
            t = swap(t, "    const int ncin = d.ncin;\n", f"    const int ncin = {ncin};\n")
        if k is not None:
            t = swap(t, "    const int k = d.k;\n", f"    const int k = {k};\n")
        return t
    return make


VARIANTS = {
    "base": lambda t: t,
    "ptr": ptr,
    "hint": hint,
    "ptr+hint": lambda t: hint(ptr(t)),
    "flat": flat,
    "flat+hint": lambda t: hint(flat(t)),
    "flatptr": flatptr,
    "chunk": chunk,
    "hoist": hoist,
    "hoist+ptr": lambda t: hoist(ptr(t)),
    "ptr+ncin4": fixed(ncin=4),          # diagnostic only: not a general kernel
    "ptr+ncin4+k3": fixed(ncin=4, k=3),  # diagnostic only
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    ap.add_argument("--source", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    base = Path(args.source).read_text(encoding="utf-8")
    for name in args.only:
        d = Path(args.out) / name.replace("+", "_")
        d.mkdir(parents=True, exist_ok=True)
        src = d / "engine_bf16.cc"
        src.write_text(VARIANTS[name](base), encoding="utf-8")
        print(f"{name:10s} {census(src, d)}", flush=True)


if __name__ == "__main__":
    main()
