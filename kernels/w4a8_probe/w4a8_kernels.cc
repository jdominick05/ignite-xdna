// W4A8 on AIE2: two ways to multiply int8 activations by int4 weights.
//
//   unpack  store B packed (two int4 per byte), widen it to int8 in the core with
//           aie::unpack, then run the stock int8 x int8 4x8x8 mmul
//   native  hand the packed B straight to aie::mmul<4,16,8,int8,int4>, which lowers
//           to the same vmac builtin as int8 x int8 with a different B mode
//
// plus the controls each needs: a plain int8 copy loop against the standalone unpack
// loop, and a local copy of upstream's int8 GEMM kernel against the two int4 ones.
//
// The three GEMM kernels are one template. POLICY changes only how a B tile is loaded
// and which mmul shape consumes it; the A, C and loop structure are upstream
// aie_kernels/aie2/mm.cc matmul_vectorized_4x2_mmul (row-major B and C, the path
// whole_array compiles by default), line for line.
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

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

// Peano predefines __AIECC__, so AIE_LOOP_UNROLL(n) is this same clang pragma in an IRON
// build (same object); upstream's k loop has only AIE_LOOP_FLATTEN, empty under Peano.
// The k-loop modes, spelled as the pragmas themselves:
//   -DINNER_NO_UNROLL  keep the k loop a loop rather than unrolling it into the j loop
//   -DINNER_UNROLL2    unroll the k loop exactly twice, so the scheduler can overlap
//                      one iteration's loads with the previous one's vmacs
#if defined(INNER_NO_UNROLL)
#define INNER_PRAGMA _Pragma("clang loop unroll(disable)")
#elif defined(INNER_UNROLL2)
#define INNER_PRAGMA _Pragma("clang loop unroll_count(2)")
#else
#define INNER_PRAGMA
#endif

// ---------------------------------------------------------------- standalone loops

// 64 int8 in, 64 int8 out per iteration: the control for unpack_i4.
extern "C" void copy_i8(const int8_t *__restrict in, int8_t *__restrict out,
                        int32_t n) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int32_t i = 0; i < n; i += 64) {
    aie::vector<int8, 64> v = aie::load_v<64>(in);
    in += 64;
    aie::store_v(out, v);
    out += 64;
  }
}

// 64 packed int4 (32 bytes) in, 64 sign-extended int8 out per iteration.
extern "C" void unpack_i4(const int8_t *__restrict in, int8_t *__restrict out,
                          int32_t n) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int32_t i = 0; i < n; i += 64) {
    aie::vector<int4, 64> p = aie::load_v<32>(in).template cast_to<int4>();
    in += 32;
    aie::store_v(out, aie::unpack(p));
    out += 64;
  }
}

// ---------------------------------------------------------------- GEMM kernels

enum { B_INT8 = 0, B_UNPACK = 1, B_NATIVE = 2 };

template <int POLICY> struct shape;
template <> struct shape<B_INT8> {
  using MMUL = aie::mmul<4, 8, 8, int8, int8, acc32>;
  static constexpr unsigned s = 8, b_bytes = 64;
};
template <> struct shape<B_UNPACK> {
  using MMUL = aie::mmul<4, 8, 8, int8, int8, acc32>;
  static constexpr unsigned s = 8, b_bytes = 32;
};
template <> struct shape<B_NATIVE> {
  using MMUL = aie::mmul<4, 16, 8, int8, int4, acc32>;
  static constexpr unsigned s = 16, b_bytes = 64;
};

template <int POLICY>
__attribute__((always_inline)) static inline auto load_b(const int8_t *p) {
  if constexpr (POLICY == B_INT8)
    return aie::load_v<64>(p);
  else if constexpr (POLICY == B_UNPACK)
    return aie::unpack(aie::load_v<32>(p).template cast_to<int4>());
  else
    return aie::load_v<64>(p).template cast_to<int4>();
}

// rowA, colA, colB count TILES (m/r, k/s, n/t), as upstream's template does.
template <unsigned rowA, unsigned colA, unsigned colB, int POLICY>
static inline void mm_4x2(const int8_t *__restrict pA, const int8_t *__restrict pB,
                          int32_t *__restrict pC) {
  using MMUL = typename shape<POLICY>::MMUL;
  constexpr unsigned SA = MMUL::size_A; // int8 elements = bytes
  constexpr unsigned SC = MMUL::size_C;
  constexpr unsigned BB = shape<POLICY>::b_bytes; // bytes per stored B tile

  auto outer_body = [&](unsigned z) [[gnu::always_inline]] {
    int32_t *__restrict pC1 = pC + (z * colB + 0) * SC;
    int32_t *__restrict pC2 = pC + ((z + 1) * colB + 0) * SC;
    int32_t *__restrict pC3 = pC + ((z + 2) * colB + 0) * SC;
    int32_t *__restrict pC4 = pC + ((z + 3) * colB + 0) * SC;

    for (unsigned j = 0; j < colB; j += 2) {
      const int8_t *__restrict pA1 = pA + (z * colA + 0) * SA;
      const int8_t *__restrict pA2 = pA + ((z + 1) * colA + 0) * SA;
      const int8_t *__restrict pA3 = pA + ((z + 2) * colA + 0) * SA;
      const int8_t *__restrict pA4 = pA + ((z + 3) * colA + 0) * SA;
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

// The same kernel with A expanded twice instead of four times: upstream's
// matmul_vectorized_2x2_mmul. The native int4 shape's A operand is 512 bits (4x16
// int8) where int8's is 256 (4x8), so 4x2 holds six 512-bit operands per iteration
// against int8's four 256 + two 512; 2x2 halves that.
template <unsigned rowA, unsigned colA, unsigned colB, int POLICY>
static inline void mm_2x2(const int8_t *__restrict pA, const int8_t *__restrict pB,
                          int32_t *__restrict pC) {
  using MMUL = typename shape<POLICY>::MMUL;
  constexpr unsigned SA = MMUL::size_A;
  constexpr unsigned SC = MMUL::size_C;
  constexpr unsigned BB = shape<POLICY>::b_bytes;

  auto outer_body = [&](unsigned z) [[gnu::always_inline]] {
    int32_t *__restrict pC1 = pC + (z * colB) * SC;
    int32_t *__restrict pC2 = pC + ((z + 1) * colB) * SC;

    for (unsigned j = 0; j < colB; j += 2) {
      const int8_t *__restrict pA1 = pA + (z * colA) * SA;
      const int8_t *__restrict pA2 = pA + ((z + 1) * colA) * SA;
      const int8_t *__restrict pB1 = pB + (j)*BB;
      const int8_t *__restrict pB2 = pB + (j + 1) * BB;

      MMUL C00(aie::load_v<SC>(pC1));
      MMUL C01(aie::load_v<SC>(pC1 + SC));
      MMUL C10(aie::load_v<SC>(pC2));
      MMUL C11(aie::load_v<SC>(pC2 + SC));

      INNER_PRAGMA
      for (unsigned i = 0; i < colA; ++i) {
        auto A0 = aie::load_v<SA>(pA1);
        pA1 += SA;
        auto A1 = aie::load_v<SA>(pA2);
        pA2 += SA;
        auto B0 = load_b<POLICY>(pB1);
        pB1 += BB * colB;
        auto B1 = load_b<POLICY>(pB2);
        pB2 += BB * colB;

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }

      aie::store_v(pC1, C00.template to_vector<int32_t>());
      pC1 += SC;
      aie::store_v(pC1, C01.template to_vector<int32_t>());
      pC1 += SC;
      aie::store_v(pC2, C10.template to_vector<int32_t>());
      pC2 += SC;
      aie::store_v(pC2, C11.template to_vector<int32_t>());
      pC2 += SC;
    }
  };

  constexpr unsigned outer_iters = rowA / 2;
  if constexpr (outer_iters >= 4) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (unsigned z = 0; z < rowA; z += 2)
      outer_body(z);
  } else if constexpr (outer_iters >= 2) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (unsigned z = 0; z < rowA; z += 2)
      outer_body(z);
  } else {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 2)
      outer_body(z);
  }
}

static_assert(DIM_M % 16 == 0 && DIM_N % 16 == 0 && DIM_K % 16 == 0,
              "tile dims must divide the 4x2 expansion and the 16-deep int4 k");

// noinline: the hardware wrapper below calls one of these, and each must run as the
// same standalone function the static probe disassembles, not as a copy inlined into
// the wrapper and scheduled afresh.
#define W4A8_ENTRY extern "C" __attribute__((noinline))

// a: DIM_M x DIM_K int8, pre-tiled 4 x s; b: DIM_K x DIM_N, pre-tiled s x 8 (int8, or
// int4 packed two per byte); c: DIM_M x DIM_N int32, pre-tiled 4 x 8, accumulated in
// place exactly as upstream's matmul_i8_i32 does.
W4A8_ENTRY void mm_i8i8_local(const int8_t *a, const int8_t *b, int32_t *c) {
  mm_4x2<DIM_M / 4, DIM_K / 8, DIM_N / 8, B_INT8>(a, b, c);
}
W4A8_ENTRY void mm_i8i4_unpack(const int8_t *a, const int8_t *b, int32_t *c) {
  mm_4x2<DIM_M / 4, DIM_K / 8, DIM_N / 8, B_UNPACK>(a, b, c);
}
W4A8_ENTRY void mm_i8i4_native(const int8_t *a, const int8_t *b, int32_t *c) {
  mm_4x2<DIM_M / 4, DIM_K / 16, DIM_N / 8, B_NATIVE>(a, b, c);
}
W4A8_ENTRY void mm_i8i8_local_2x2(const int8_t *a, const int8_t *b, int32_t *c) {
  mm_2x2<DIM_M / 4, DIM_K / 8, DIM_N / 8, B_INT8>(a, b, c);
}
W4A8_ENTRY void mm_i8i4_unpack_2x2(const int8_t *a, const int8_t *b, int32_t *c) {
  mm_2x2<DIM_M / 4, DIM_K / 8, DIM_N / 8, B_UNPACK>(a, b, c);
}
W4A8_ENTRY void mm_i8i4_native_2x2(const int8_t *a, const int8_t *b, int32_t *c) {
  mm_2x2<DIM_M / 4, DIM_K / 16, DIM_N / 8, B_NATIVE>(a, b, c);
}

// ---------------------------------------------------------------- hardware wrapper
//
// What w4a8_probe.py runs on one core: zero C, then event0 / one call to RUN_KERNEL /
// event1, so the tile's trace unit stamps the cycles of exactly one kernel call and
// nothing else -- no DMA and no lock sits inside the bracket. The filler pairs after it
// are clock_probe's: one event pair alone never fills a trace packet, so it would never
// reach host memory; the host reads the FIRST event0 and the FIRST event1.
#ifdef RUN_KERNEL
#ifndef FLUSH_PAIRS
#define FLUSH_PAIRS 256
#endif
extern "C" void w4a8_run(const int8_t *a, const int8_t *b, int32_t *c) {
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
