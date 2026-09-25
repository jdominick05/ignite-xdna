// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// F2-0b: F2's flush with fp32(I) built from I's 16-bit pieces by shuffles, with no srs on an acc64
// (the F2-0b addendum). F2-0's flush took fp32(I) with aie::to_float on the acc64 halves, whose four
// srs per 16 lanes set the saturation and rounding modes per call, inside the loop. Compile only
// (tools/hybrid_f2_0b.py); nothing here has run on a core.
//
// I comes as F2-0's f2_block leaves it in L1 (f2_epilogue.cc, not edited): 32 acc64 lanes per tile as
// 64 int32 words, lane k's low word at index 2k and its high word at 2k + 1. As 16-bit halves, lane k
// holds h0..h3, least significant first. For |I| < 2^47, I = h2 x 2^32 + h1 x 2^16 + h0 exactly, with
// h2 signed and h1, h0 unsigned. Each piece goes to fp32 exactly (aie::to_float's 16-bit path with a
// power-of-two shift), and the sum runs high to low. F2's bound is |I| < 2^39, so f2 + f1 is exact
// and only the last add rounds.
//
// Cases, one compile each:
//   -DF2_FLUSH_B3   fp32(I), then F2-0's flush: 3-piece splits, U6-E's ORDER[3]['P'] for P and for y
//   -DF2_FLUSH_B2   fp32(I), then 2-piece splits, U6-E's ORDER[2]['P'] for P and ORDER[2]['Y'] for y
//
// Lines taken from f2_epilogue.cc (this repo, AGPL): split3, the P_FIRST / P_TERM / Y_TERM macros,
// the tile loop's head, and the B3 branch (its lines 67-94, verbatim). split2 is split3's first two
// lines.
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernels/aie_kernel_utils.h"

#if !defined(F2_FLUSH_B3) && !defined(F2_FLUSH_B2)
#error "define one of F2_FLUSH_B3 or F2_FLUSH_B2"
#endif

using VBF = aie::vector<bfloat16, 32>;
using AF = aie::accum<accfloat, 32>;

// Pieces of x in bf16: each is the RNE bf16 of the running residual (crRnd, set before the loop), and
// the residual's subtraction is in accfloat, where it is exact.
__attribute__((always_inline)) static inline void split3(AF x, VBF &p1, VBF &p2,
                                                         VBF &p3) {
  p1 = x.template to_vector<bfloat16>();
  x = aie::sub(x, p1);
  p2 = x.template to_vector<bfloat16>();
  x = aie::sub(x, p2);
  p3 = x.template to_vector<bfloat16>();
}

__attribute__((always_inline)) static inline void split2(AF x, VBF &p1, VBF &p2) {
  p1 = x.template to_vector<bfloat16>();
  x = aie::sub(x, p1);
  p2 = x.template to_vector<bfloat16>();
}

// The fp32 add order, one term per call, smallest first (hybrid_u6e.py ORDER). a = Sx pieces, w = Dw
// pieces, h = fp32(I) pieces, p = P pieces; piece 1 is the largest. tools/hybrid_f2_0b.py reads these
// calls back from this file and checks them against U6-E's frozen ORDER.
#define P_FIRST(j, k) AF P = aie::mul(a##j, w##k)
#define P_TERM(j, k) P = aie::mac(P, a##j, w##k)
#define Y_TERM(u, v) Y = aie::mac(Y, h##u, p##v)

// fp32(I) for one tile's 32 lanes from its 64 int32 words: shuffles, then three exact conversions,
// then two fp32 adds, of which only the last rounds.
__attribute__((always_inline)) static inline AF fp32_of_I(const int32_t *__restrict Ib) {
  aie::vector<int32, 32> wa = aie::load_v<32>(Ib);      // lanes 0-15
  aie::vector<int32, 32> wb = aie::load_v<32>(Ib + 32); // lanes 16-31
  // Each lane's low word, (h0, h1), and high word, (h2, h3).
  aie::vector<uint16, 32> la = aie::filter_even(wa).template cast_to<uint16>();
  aie::vector<uint16, 32> lb = aie::filter_even(wb).template cast_to<uint16>();
  aie::vector<int16, 32> ha = aie::filter_odd(wa).template cast_to<int16>();
  aie::vector<int16, 32> hb = aie::filter_odd(wb).template cast_to<int16>();
  aie::vector<uint16, 32> H0 = aie::concat(aie::filter_even(la), aie::filter_even(lb));
  aie::vector<uint16, 32> H1 = aie::concat(aie::filter_odd(la), aie::filter_odd(lb));
  aie::vector<int16, 32> H2 = aie::concat(aie::filter_even(ha), aie::filter_even(hb));
  AF fa(aie::to_float<float>(H2, -32));             // h2 x 2^32, exact
  fa = aie::add(fa, aie::to_float<float>(H1, -16)); // + h1 x 2^16: exact, |I / 2^16| < 2^23
  fa = aie::add(fa, aie::to_float<float>(H0, 0));   // + h0: the one rounding
  return fa;
}

// I: 32 acc64 lanes as 64 int32 words per tile; sx: 4 fp32 (Sx, one per row); dw: 8 fp32 (Dw, one
// per column); y: 32 fp32. Every pointer advances per iteration.
extern "C" void f2_flush_b(const int32_t *__restrict Ib, const float *__restrict sx,
                           const float *__restrict dw, float *__restrict y, int32_t ntiles) {
  // Once, before the tile loop: RNE for the accfloat -> bf16 conversions. Never inside the loop.
  aie::set_rounding(aie::rounding_mode::conv_even);

  AIE_LOOP_NO_UNROLL
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int32_t t = 0; t < ntiles; ++t) {
    AF fa = fp32_of_I(Ib);
    Ib += 64;

#if defined(F2_FLUSH_B3)
    // Sx[r] to lanes r*8 + 0..7; Dw[c] to lanes 0..3 * 8 + c.
    aie::vector<float, 8> d = aie::load_v<8>(dw);
    AF sxa(aie::concat(aie::broadcast<float, 8>(sx[0]), aie::broadcast<float, 8>(sx[1]),
                       aie::broadcast<float, 8>(sx[2]), aie::broadcast<float, 8>(sx[3])));
    AF dwa(aie::concat(d, d, d, d));
    sx += 4;
    dw += 8;

    AF Y(aie::load_v<32>(y));
    VBF h1, h2, h3, a1, a2, a3, w1, w2, w3, p1, p2, p3;
    split3(fa, h1, h2, h3);
    split3(sxa, a1, a2, a3);
    split3(dwa, w1, w2, w3);
    P_FIRST(3, 1);
    P_TERM(2, 2);
    P_TERM(1, 3);
    P_TERM(2, 1);
    P_TERM(1, 2);
    P_TERM(1, 1);
    split3(P, p1, p2, p3);
    Y_TERM(3, 1);
    Y_TERM(2, 2);
    Y_TERM(1, 3);
    Y_TERM(2, 1);
    Y_TERM(1, 2);
    Y_TERM(1, 1);
    aie::store_v(y, Y.template to_vector<float>());
    y += 32;
#else
    // Sx[r] to lanes r*8 + 0..7; Dw[c] to lanes 0..3 * 8 + c.
    aie::vector<float, 8> d = aie::load_v<8>(dw);
    AF sxa(aie::concat(aie::broadcast<float, 8>(sx[0]), aie::broadcast<float, 8>(sx[1]),
                       aie::broadcast<float, 8>(sx[2]), aie::broadcast<float, 8>(sx[3])));
    AF dwa(aie::concat(d, d, d, d));
    sx += 4;
    dw += 8;

    AF Y(aie::load_v<32>(y));
    VBF h1, h2, a1, a2, w1, w2, p1, p2;
    split2(fa, h1, h2);
    split2(sxa, a1, a2);
    split2(dwa, w1, w2);
    P_FIRST(2, 1);
    P_TERM(1, 2);
    P_TERM(1, 1);
    split2(P, p1, p2);
    Y_TERM(2, 1);
    Y_TERM(1, 2);
    Y_TERM(1, 1);
    aie::store_v(y, Y.template to_vector<float>());
    y += 32;
#endif
  }
}
