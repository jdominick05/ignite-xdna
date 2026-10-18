// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// U6-0: the epilogue of U6's Q4_0 x int8-per-block GEMM on AIE2, for a bundle count. Compile only
// (tools/hybrid_u6_0.py); nothing here has run on a core.
//
// One iteration of u6_block's tile loop is one 32-block of one 4 x 8 output tile:
//   i     = A (4 x 32 int8) x B (32 x 8 signed int4), exact, in two int8 x int4 mmul steps
//   y    += float(i) x P, P = s_x x d_w, with fp32 products built from bf16 pieces
// in U6-E's form and fixed fp32 add order (tools/hybrid_u6e.py, frozen at 2505530). B holds (c - 8) as
// signed int4, repacked on the host, so there is no -8 x sum(q) term here. y is loaded, updated and
// stored every iteration at a new address: a round-trip through L1, never a register.
//
// Cases, one compile each:
//   -DU6_CONTROL             the loads of A and B, the two MACs, i stored as int32: the baseline
//   -DU6_SPLIT=3             the epilogue with 3 bf16 pieces per fp32 value (U6-E's split)
//   -DU6_SPLIT=2             the same with 2 pieces
//   -DU6_SPLIT=3 -DU6_I16    float(i) through int16 instead of int32
//   -DU6_TOFLOAT             u6_tofloat only: int32 -> fp32 with the call u6_block uses, alone
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernels/aie_kernel_utils.h"

#if defined(U6_TOFLOAT)

extern "C" void u6_tofloat(const int32_t *__restrict in, float *__restrict out,
                           int32_t n) {
  AIE_LOOP_NO_UNROLL
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int32_t t = 0; t < n; ++t) {
    aie::vector<int32, 32> v = aie::load_v<32>(in);
    in += 32;
    aie::store_v(out, aie::to_float<float>(v));
    out += 32;
  }
}

#else

#ifndef U6_SPLIT
#define U6_SPLIT 3
#endif
#if !defined(U6_CONTROL) && U6_SPLIT != 2 && U6_SPLIT != 3
#error "U6_SPLIT must be 2 or 3"
#endif

using MMUL = aie::mmul<4, 16, 8, int8, int4, acc32>; // C is 4 x 8, row-major: lane r*8 + c
using VBF = aie::vector<bfloat16, 32>;
using AF = aie::accum<accfloat, 32>;

// Pieces of x in bf16: each is the RNE bf16 of the running residual (crRnd, set before the loop), and
// the residual's subtraction is in accfloat, where it is exact.
__attribute__((always_inline)) static inline void split2(AF x, VBF &p1, VBF &p2) {
  p1 = x.template to_vector<bfloat16>();
  x = aie::sub(x, p1);
  p2 = x.template to_vector<bfloat16>();
}
__attribute__((always_inline)) static inline void split3(AF x, VBF &p1, VBF &p2,
                                                         VBF &p3) {
  p1 = x.template to_vector<bfloat16>();
  x = aie::sub(x, p1);
  p2 = x.template to_vector<bfloat16>();
  x = aie::sub(x, p2);
  p3 = x.template to_vector<bfloat16>();
}

// The fp32 add order, one term per call, smallest first (hybrid_u6e.py ORDER). a = s_x pieces,
// w = d_w pieces, h = float(i) pieces, p = P pieces; piece 1 is the largest. tools/hybrid_u6_0.py
// reads these calls back from this file and checks them against U6-E's frozen ORDER.
#define P_FIRST(j, k) AF P = aie::mul(a##j, w##k)
#define P_TERM(j, k) P = aie::mac(P, a##j, w##k)
#define Y_TERM(u, v) Y = aie::mac(Y, h##u, p##v)

// A and B: 128 B per tile each (two 64 B mmul operands); sx: 4 fp32 (one per row); dw: 8 fp32 (one
// per column); y: 32 fp32. Every pointer advances per iteration.
extern "C" void u6_block(const int8_t *__restrict A, const int8_t *__restrict B,
                         const float *__restrict sx, const float *__restrict dw,
                         float *__restrict y, int32_t ntiles) {
  // Once, before the tile loop: RNE for the accfloat -> bf16 conversions. Never inside the loop.
  aie::set_rounding(aie::rounding_mode::conv_even);

  AIE_LOOP_NO_UNROLL
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int32_t t = 0; t < ntiles; ++t) {
    MMUL C;
    C.mul(aie::load_v<64>(A), aie::load_v<64>(B).template cast_to<int4>());
    C.mac(aie::load_v<64>(A + 64), aie::load_v<64>(B + 64).template cast_to<int4>());
    A += 128;
    B += 128;

#if defined(U6_CONTROL)
    aie::store_v(reinterpret_cast<int32_t *>(y), C.template to_vector<int32>());
    y += 32;
#else
    // float(i), exact for |i| <= 32,512, then its two bf16 pieces (h2 is exact in bf16).
#if defined(U6_I16)
    AF fa(aie::to_float<float>(C.template to_vector<int16>()));
#else
    AF fa(aie::to_float<float>(C.template to_vector<int32>()));
#endif
    VBF h1 = fa.template to_vector<bfloat16>();
    VBF h2 = aie::sub(fa, h1).template to_vector<bfloat16>();

    // s_x[r] to lanes r*8 + 0..7; d_w[c] to lanes 0..3 * 8 + c.
    aie::vector<float, 8> d = aie::load_v<8>(dw);
    AF sxa(aie::concat(aie::broadcast<float, 8>(sx[0]), aie::broadcast<float, 8>(sx[1]),
                       aie::broadcast<float, 8>(sx[2]), aie::broadcast<float, 8>(sx[3])));
    AF dwa(aie::concat(d, d, d, d));
    sx += 4;
    dw += 8;

    AF Y(aie::load_v<32>(y));
#if U6_SPLIT == 3
    VBF a1, a2, a3, w1, w2, w3, p1, p2, p3;
    split3(sxa, a1, a2, a3);
    split3(dwa, w1, w2, w3);
    P_FIRST(3, 1);
    P_TERM(2, 2);
    P_TERM(1, 3);
    P_TERM(2, 1);
    P_TERM(1, 2);
    P_TERM(1, 1);
    split3(P, p1, p2, p3);
    Y_TERM(2, 2);
    Y_TERM(1, 3);
    Y_TERM(2, 1);
    Y_TERM(1, 2);
    Y_TERM(1, 1);
#else
    VBF a1, a2, w1, w2, p1, p2;
    split2(sxa, a1, a2);
    split2(dwa, w1, w2);
    P_FIRST(2, 1);
    P_TERM(1, 2);
    P_TERM(1, 1);
    split2(P, p1, p2);
    Y_TERM(2, 1);
    Y_TERM(1, 2);
    Y_TERM(1, 1);
#endif
    aie::store_v(y, Y.template to_vector<float>());
    y += 32;
#endif
  }
}

#endif
