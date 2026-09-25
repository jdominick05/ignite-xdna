// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// F2-0: the epilogue of F2 (integerized 8-bit scales) for route (i)'s Q4_0 x int8-per-block GEMM on
// AIE2, for a bundle count. Compile only (tools/hybrid_f2_0.py); nothing here has run on a core.
//
// f2_block: one iteration of the tile loop is one 32-block of one 4 x 8 output tile, as in U6-0's
// u6_block (kernels/u6_epilogue/u6_epilogue.cc):
//   i   = A (4 x 32 int8) x B (32 x 8 signed int4), exact, in two int8 x int4 mmul steps
//   P   = qx (x) qw: qx[r] a uint8 per row, qw[c] a signed int8 per column; exact in int16, |P| <= 32,385
//   I  += i x P, exact in a 64-bit accumulator; |i| <= 32,512, so i is exact in int16
// I is loaded, updated and stored every iteration at a new address: a round-trip through L1, never a
// register. It goes to and from memory bit for bit through aie::vector_cast (no shift, no saturation).
//
// f2_flush: once per superblock per tile, y += fp32(I) x (Sx (x) Dw), the fp32 product built from bf16
// pieces in U6-E's 3-term form (tools/hybrid_u6e.py, frozen at 2505530). fp32(I) has 24 significant
// bits, so it takes three pieces, and the y terms take the P order.
//
// Cases, one compile each:
//   -DF2_CONTROL     f2_block: the loads of A and B, the two MACs, i stored as int32 (U6-0's CONTROL)
//   -DF2_CORE        f2_block: i to int16, P, I += i x P in acc64
//   -DF2_CORE_I32    f2_block: the same with i kept as int32 (int32 x int16 into acc64)
//   -DF2_FLUSH       f2_flush only
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernels/aie_kernel_utils.h"

#if defined(F2_FLUSH)

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

// The fp32 add order, one term per call, smallest first (hybrid_u6e.py ORDER[3]). a = Sx pieces,
// w = Dw pieces, h = fp32(I) pieces, p = P pieces; piece 1 is the largest. tools/hybrid_f2_0.py reads
// these calls back from this file and checks them against U6-E's frozen ORDER.
#define P_FIRST(j, k) AF P = aie::mul(a##j, w##k)
#define P_TERM(j, k) P = aie::mac(P, a##j, w##k)
#define Y_TERM(u, v) Y = aie::mac(Y, h##u, p##v)

// I: 32 acc64 lanes as 64 int32 words per tile; sx: 4 fp32 (Sx, one per row); dw: 8 fp32 (Dw, one
// per column); y: 32 fp32. Every pointer advances per iteration.
extern "C" void f2_flush(const int32_t *__restrict Ib, const float *__restrict sx,
                         const float *__restrict dw, float *__restrict y, int32_t ntiles) {
  // Once, before the tile loop: RNE for the accfloat -> bf16 conversions. Never inside the loop.
  aie::set_rounding(aie::rounding_mode::conv_even);

  AIE_LOOP_NO_UNROLL
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int32_t t = 0; t < ntiles; ++t) {
    aie::accum<acc64, 16> I0 = aie::vector_cast<acc64>(aie::load_v<32>(Ib));
    aie::accum<acc64, 16> I1 = aie::vector_cast<acc64>(aie::load_v<32>(Ib + 32));
    Ib += 64;
    AF fa(aie::concat(aie::to_float<float>(I0), aie::to_float<float>(I1)));

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
  }
}

#else

#if !defined(F2_CONTROL) && !defined(F2_CORE) && !defined(F2_CORE_I32)
#error "define one of F2_CONTROL, F2_CORE, F2_CORE_I32 or F2_FLUSH"
#endif

using MMUL = aie::mmul<4, 16, 8, int8, int4, acc32>; // C is 4 x 8, row-major: lane r*8 + c

// A and B: 128 B per tile each (two 64 B mmul operands); qx: 4 uint8 (one per row); qw: a 16 B record
// per tile, of which the first 8 int8 are used (one per column); y: 32 int32 in CONTROL, 64 int32 (32
// acc64 lanes) otherwise. Every pointer advances per iteration.
extern "C" void f2_block(const int8_t *__restrict A, const int8_t *__restrict B,
                         const uint8_t *__restrict qx, const int8_t *__restrict qw,
                         int32_t *__restrict y, int32_t ntiles) {
  AIE_LOOP_NO_UNROLL
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int32_t t = 0; t < ntiles; ++t) {
    MMUL C;
    C.mul(aie::load_v<64>(A), aie::load_v<64>(B).template cast_to<int4>());
    C.mac(aie::load_v<64>(A + 64), aie::load_v<64>(B + 64).template cast_to<int4>());
    A += 128;
    B += 128;

#if defined(F2_CONTROL)
    aie::store_v(y, C.template to_vector<int32>());
    y += 32;
#else
    // P = qx (x) qw in int16: qx[r] to lanes r*8 + 0..7, qw[c] to lanes 0..3 * 8 + c.
    aie::vector<int16, 32> xr =
        aie::concat(aie::broadcast<int16, 8>(qx[0]), aie::broadcast<int16, 8>(qx[1]),
                    aie::broadcast<int16, 8>(qx[2]), aie::broadcast<int16, 8>(qx[3]));
    aie::vector<int16, 8> w = aie::unpack(aie::load_v<16>(qw)).template extract<8>(0);
    aie::vector<int16, 32> wr = aie::concat(w, w, w, w);
    qx += 4;
    qw += 16;
    aie::vector<int16, 32> P = aie::mul(xr, wr).template to_vector<int16>(0); // exact

    aie::accum<acc64, 16> I0 = aie::vector_cast<acc64>(aie::load_v<32>(y));
    aie::accum<acc64, 16> I1 = aie::vector_cast<acc64>(aie::load_v<32>(y + 32));
#if defined(F2_CORE_I32)
    aie::vector<int32, 32> i = C.template to_vector<int32>();
#else
    aie::vector<int16, 32> i = C.template to_vector<int16>(0); // exact: |i| <= 32,512
#endif
    I0 = aie::mac(I0, i.template extract<16>(0), P.template extract<16>(0));
    I1 = aie::mac(I1, i.template extract<16>(1), P.template extract<16>(1));
    aie::store_v(y, aie::vector_cast<int32>(I0));
    aie::store_v(y + 32, aie::vector_cast<int32>(I1));
    y += 64;
#endif
  }
}

#endif
