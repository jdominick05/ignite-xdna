// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// bf16 x bf16 -> f32 tile matmul for AIE2 (XDNA1), C += A B, a drop-in for upstream mm.cc's matmul_bf16_f32 and
// zero_f32 (same symbols, same operand layouts, same per-accumulator K order, so the output is meant to be
// bit-identical to mm.cc's).
//
// Layouts (mm.cc's, aie::mmul<4,8,4>): A is DIM_M/4 x DIM_K/8 blocks of 4x8, row-major over blocks; B is
// DIM_K/8 x DIM_N/4 blocks of 8x4; C is DIM_M/4 x DIM_N/4 blocks of 4x4. Each output block of 4 x 4 accumulators
// loads its C, runs K/8 steps of 16 MACs in increasing K, and stores.
//
// Why it exists (results/aie/notes_mm_bf16_schedule.md): mm.cc's K loop keeps eight post-incremented pointers and
// Peano spends ~30 of its 33 bundles moving them through the single move slot. Here A and B each have one base
// pointer per block and the four rows/columns are fixed offsets from it.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

namespace {
constexpr unsigned R = 4, S = 8, T = 4;
constexpr unsigned ROWA = DIM_M / R, COLA = DIM_K / S, COLB = DIM_N / T;
constexpr unsigned SA = R * S, SB = S * T, SC = R * T;   // elements per A, B, C block
static_assert(ROWA % 4 == 0 && COLB % 4 == 0, "4 x 4 accumulator blocks");
using MMUL = aie::mmul<R, S, T, bfloat16, bfloat16, accauto>;
}

extern "C" {

void matmul_bf16_f32(const bfloat16 *__restrict pA, const bfloat16 *__restrict pB, float *__restrict pC) {
  event0();
  for (unsigned z = 0; z < ROWA; z += 4) {
    for (unsigned j = 0; j < COLB; j += 4) {
      float *__restrict c0 = pC + (z * COLB + j) * SC;
      MMUL C00(aie::load_v<SC>(c0 + 0 * SC)), C01(aie::load_v<SC>(c0 + 1 * SC)),
           C02(aie::load_v<SC>(c0 + 2 * SC)), C03(aie::load_v<SC>(c0 + 3 * SC));
      MMUL C10(aie::load_v<SC>(c0 + (COLB + 0) * SC)), C11(aie::load_v<SC>(c0 + (COLB + 1) * SC)),
           C12(aie::load_v<SC>(c0 + (COLB + 2) * SC)), C13(aie::load_v<SC>(c0 + (COLB + 3) * SC));
      MMUL C20(aie::load_v<SC>(c0 + (2 * COLB + 0) * SC)), C21(aie::load_v<SC>(c0 + (2 * COLB + 1) * SC)),
           C22(aie::load_v<SC>(c0 + (2 * COLB + 2) * SC)), C23(aie::load_v<SC>(c0 + (2 * COLB + 3) * SC));
      MMUL C30(aie::load_v<SC>(c0 + (3 * COLB + 0) * SC)), C31(aie::load_v<SC>(c0 + (3 * COLB + 1) * SC)),
           C32(aie::load_v<SC>(c0 + (3 * COLB + 2) * SC)), C33(aie::load_v<SC>(c0 + (3 * COLB + 3) * SC));
      const bfloat16 *__restrict a = pA + z * COLA * SA;   // row block z; rows z+1..z+3 at +COLA*SA each
      const bfloat16 *__restrict b = pB + j * SB;          // column block j; columns j+1..j+3 at +SB each
      for (unsigned i = 0; i < COLA; ++i) chess_prepare_for_pipelining {
        auto A0 = aie::load_v<SA>(a), A1 = aie::load_v<SA>(a + COLA * SA), A2 = aie::load_v<SA>(a + 2 * COLA * SA),
             A3 = aie::load_v<SA>(a + 3 * COLA * SA);
        auto B0 = aie::load_v<SB>(b), B1 = aie::load_v<SB>(b + SB), B2 = aie::load_v<SB>(b + 2 * SB),
             B3 = aie::load_v<SB>(b + 3 * SB);
        a += SA;
        b += COLB * SB;
        C00.mac(A0, B0); C01.mac(A0, B1); C02.mac(A0, B2); C03.mac(A0, B3);
        C10.mac(A1, B0); C11.mac(A1, B1); C12.mac(A1, B2); C13.mac(A1, B3);
        C20.mac(A2, B0); C21.mac(A2, B1); C22.mac(A2, B2); C23.mac(A2, B3);
        C30.mac(A3, B0); C31.mac(A3, B1); C32.mac(A3, B2); C33.mac(A3, B3);
      }
      aie::store_v(c0 + 0 * SC, C00.template to_vector<float>()); aie::store_v(c0 + 1 * SC, C01.template to_vector<float>());
      aie::store_v(c0 + 2 * SC, C02.template to_vector<float>()); aie::store_v(c0 + 3 * SC, C03.template to_vector<float>());
      aie::store_v(c0 + (COLB + 0) * SC, C10.template to_vector<float>()); aie::store_v(c0 + (COLB + 1) * SC, C11.template to_vector<float>());
      aie::store_v(c0 + (COLB + 2) * SC, C12.template to_vector<float>()); aie::store_v(c0 + (COLB + 3) * SC, C13.template to_vector<float>());
      aie::store_v(c0 + (2 * COLB + 0) * SC, C20.template to_vector<float>()); aie::store_v(c0 + (2 * COLB + 1) * SC, C21.template to_vector<float>());
      aie::store_v(c0 + (2 * COLB + 2) * SC, C22.template to_vector<float>()); aie::store_v(c0 + (2 * COLB + 3) * SC, C23.template to_vector<float>());
      aie::store_v(c0 + (3 * COLB + 0) * SC, C30.template to_vector<float>()); aie::store_v(c0 + (3 * COLB + 1) * SC, C31.template to_vector<float>());
      aie::store_v(c0 + (3 * COLB + 2) * SC, C32.template to_vector<float>()); aie::store_v(c0 + (3 * COLB + 3) * SC, C33.template to_vector<float>());
    }
  }
  event1();
}

void zero_f32(float *__restrict pC) {
  const aie::vector<float, 16> z = aie::zeros<float, 16>();
  for (unsigned i = 0; i < DIM_M * DIM_N; i += 16) aie::store_v(pC + i, z);
}

}  // extern "C"
