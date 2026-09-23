// Copyright (C) 2026 The ignite-xdna contributors
// Portions adapted from mlir-aie v1.4.2 aie_kernels/aie2/mm.cc (matmul_vectorized_4x2_mmul),
//   Copyright (C) 2025 Advanced Micro Devices, Inc., Apache-2.0 WITH LLVM-exception.
// SPDX-License-Identifier: AGPL-3.0-or-later AND Apache-2.0 WITH LLVM-exception
//
// Gate C of the INT4 study: uint8 activations x int4 weights on one AIE2 core.
//
// The graph engine's activations are uint8 (zero point 128; kernels/aie2/conv_engine/engine.cc
// runs aie::mmul<4,8,8,uint8,int8>). The 2026-09-10 W4A8 probe tested int8 x int4 only
// (kernels/w4a8_probe/w4a8_kernels.cc, kept byte-identical because its logs describe it).
// This is that probe's 4x2 GEMM kernel with the A element type as a parameter:
//
//   mm_u8i8          uint8 x int8, aie::mmul<4,8,8,uint8,int8>   -- the engine's own pair (control)
//   mm_u8i4_native   uint8 x int4, aie::mmul<4,16,8,uint8,int4>  -- B packed two per byte
//
// Same layouts, same k-loop modes (-DINNER_NO_UNROLL / -DINNER_UNROLL2) and the same
// event0/event1 bracket as the W4A8 probe, so its cycle counts are comparable.

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernels/aie_kernel_utils.h"

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

#if defined(INNER_NO_UNROLL)
#define INNER_PRAGMA _Pragma("clang loop unroll(disable)")
#elif defined(INNER_UNROLL2)
#define INNER_PRAGMA _Pragma("clang loop unroll_count(2)")
#else
#define INNER_PRAGMA
#endif

enum { B_INT8 = 0, B_NATIVE = 2 };

template <int POLICY> struct shape;
template <> struct shape<B_INT8> {
  using MMUL = aie::mmul<4, 8, 8, uint8, int8, acc32>;
  static constexpr unsigned b_bytes = 64;
};
template <> struct shape<B_NATIVE> {
  using MMUL = aie::mmul<4, 16, 8, uint8, int4, acc32>;
  static constexpr unsigned b_bytes = 64;
};

template <int POLICY>
__attribute__((always_inline)) static inline auto load_b(const int8_t *p) {
  if constexpr (POLICY == B_INT8)
    return aie::load_v<64>(p);
  else
    return aie::load_v<64>(p).template cast_to<int4>();
}

// rowA, colA, colB count TILES (m/r, k/s, n/t), as upstream's template does.
template <unsigned rowA, unsigned colA, unsigned colB, int POLICY>
static inline void mm_4x2(const uint8_t *__restrict pA, const int8_t *__restrict pB,
                          int32_t *__restrict pC) {
  using MMUL = typename shape<POLICY>::MMUL;
  constexpr unsigned SA = MMUL::size_A; // uint8 elements = bytes
  constexpr unsigned SC = MMUL::size_C;
  constexpr unsigned BB = shape<POLICY>::b_bytes; // bytes per stored B tile

  auto outer_body = [&](unsigned z) [[gnu::always_inline]] {
    int32_t *__restrict pC1 = pC + (z * colB + 0) * SC;
    int32_t *__restrict pC2 = pC + ((z + 1) * colB + 0) * SC;
    int32_t *__restrict pC3 = pC + ((z + 2) * colB + 0) * SC;
    int32_t *__restrict pC4 = pC + ((z + 3) * colB + 0) * SC;

    for (unsigned j = 0; j < colB; j += 2) {
      const uint8_t *__restrict pA1 = pA + (z * colA + 0) * SA;
      const uint8_t *__restrict pA2 = pA + ((z + 1) * colA + 0) * SA;
      const uint8_t *__restrict pA3 = pA + ((z + 2) * colA + 0) * SA;
      const uint8_t *__restrict pA4 = pA + ((z + 3) * colA + 0) * SA;
      const int8_t *__restrict pB1 = pB + (j)*BB;
      const int8_t *__restrict pB2 = pB + (j + 1) * BB;

      MMUL C00(aie::load_v<SC>(pC1));
      MMUL C01(aie::load_v<SC>(pC1 + SC));
      MMUL C10(aie::load_v<SC>(pC2));
      MMUL C11(aie::load_v<SC>(pC2 + SC));
      MMUL C20(aie::load_v<SC>(pC3));
      MMUL C21(aie::load_v<SC>(pC3 + SC));
      MMUL C30(aie::load_v<SC>(pC4));
      MMUL C31(aie::load_v<SC>(pC4 + SC));

      INNER_PRAGMA
      for (unsigned i = 0; i < colA; i += 1) {
        auto A01 = aie::load_v<SA>(pA1);
        pA1 += SA;
        auto A11 = aie::load_v<SA>(pA2);
        pA2 += SA;
        auto A21 = aie::load_v<SA>(pA3);
        pA3 += SA;
        auto A31 = aie::load_v<SA>(pA4);
        pA4 += SA;
        auto B0 = load_b<POLICY>(pB1);
        pB1 += BB * colB;
        auto B1 = load_b<POLICY>(pB2);
        pB2 += BB * colB;

        C00.mac(A01, B0);
        C01.mac(A01, B1);
        C10.mac(A11, B0);
        C11.mac(A11, B1);
        C20.mac(A21, B0);
        C21.mac(A21, B1);
        C30.mac(A31, B0);
        C31.mac(A31, B1);
      }

      aie::store_v(pC1, C00.template to_vector<int32_t>());
      pC1 += SC;
      aie::store_v(pC1, C01.template to_vector<int32_t>());
      pC1 += SC;
      aie::store_v(pC2, C10.template to_vector<int32_t>());
      pC2 += SC;
      aie::store_v(pC2, C11.template to_vector<int32_t>());
      pC2 += SC;
      aie::store_v(pC3, C20.template to_vector<int32_t>());
      pC3 += SC;
      aie::store_v(pC3, C21.template to_vector<int32_t>());
      pC3 += SC;
      aie::store_v(pC4, C30.template to_vector<int32_t>());
      pC4 += SC;
      aie::store_v(pC4, C31.template to_vector<int32_t>());
      pC4 += SC;
    }
  };

  constexpr unsigned outer_iters = rowA / 4;
  if constexpr (outer_iters >= 4) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (unsigned z = 0; z < rowA; z += 4)
      outer_body(z);
  } else if constexpr (outer_iters >= 2) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (unsigned z = 0; z < rowA; z += 4)
      outer_body(z);
  } else {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 4)
      outer_body(z);
  }
}

static_assert(DIM_M % 16 == 0 && DIM_N % 16 == 0 && DIM_K % 16 == 0,
              "tile dims must divide the 4x2 expansion and the 16-deep int4 k");

#define U8I4_ENTRY extern "C" __attribute__((noinline))

// a: DIM_M x DIM_K uint8, pre-tiled 4 x s; b: DIM_K x DIM_N, pre-tiled s x 8 (int8, or
// int4 packed two per byte, low nibble first); c: DIM_M x DIM_N int32, pre-tiled 4 x 8.
U8I4_ENTRY void mm_u8i8(const uint8_t *a, const int8_t *b, int32_t *c) {
  mm_4x2<DIM_M / 4, DIM_K / 8, DIM_N / 8, B_INT8>(a, b, c);
}
U8I4_ENTRY void mm_u8i4_native(const uint8_t *a, const int8_t *b, int32_t *c) {
  mm_4x2<DIM_M / 4, DIM_K / 16, DIM_N / 8, B_NATIVE>(a, b, c);
}

// The W4A8 probe's hardware wrapper: zero C, then event0 / one kernel call / event1, then
// filler pairs so the trace packet reaches host memory (clock_probe's method).
#ifdef RUN_KERNEL
#ifndef FLUSH_PAIRS
#define FLUSH_PAIRS 256
#endif
extern "C" void u8i4_run(const uint8_t *a, const int8_t *b, int32_t *c) {
  const aie::vector<int32_t, 16> z = aie::zeros<int32_t, 16>();
  for (int i = 0; i < DIM_M * DIM_N; i += 16)
    aie::store_v(c + i, z);
  event0();
  RUN_KERNEL(a, b, c);
  event1();
  for (int k = 0; k < FLUSH_PAIRS; ++k) {
    event0();
    event1();
  }
}
#endif
