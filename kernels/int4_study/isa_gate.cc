// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Gate A of the INT4 study: which int4 operand pairs does the aie_api/Peano toolchain
// lower for a target? One aie::mmul, typed by compile defines, nothing else.
//
//   -DTA=<A type> -DTB=<B type> -DM_=<m> -DK_=<k> -DN_=<n> -DACC=<acc32|acc64>
//   -DNO_MMUL   the harness control: the same loads and casts, no mmul. If this compiles
//               and the mmul build does not, the failure is the mmul's, not the harness's.
//
// Compile only. Nothing here runs on the NPU, and a compile failure is a statement about
// the toolchain (aie_api + Peano), never about the silicon: the W4A8 probe already found
// the device.yaml cost table wrong once (docs/SILICON.md 1.2's int8xint4 row).

#include <aie_api/aie.hpp>
#include <stdint.h>

template <typename T> struct bits;
template <> struct bits<int4> { static constexpr unsigned v = 4; };
template <> struct bits<uint4> { static constexpr unsigned v = 4; };
template <> struct bits<int8> { static constexpr unsigned v = 8; };
template <> struct bits<uint8> { static constexpr unsigned v = 8; };
template <> struct bits<int16> { static constexpr unsigned v = 16; };

// N elements of T from raw bytes: load N*bits/8 int8, reinterpret as T.
template <typename T, unsigned N>
static inline aie::vector<T, N> load_as(const int8_t *__restrict p) {
  constexpr unsigned BYTES = N * bits<T>::v / 8;
  return aie::load_v<BYTES>(p).template cast_to<T>();
}

extern "C" void gate(const int8_t *__restrict a, const int8_t *__restrict b,
                     int32_t *__restrict c) {
  auto va = load_as<TA, M_ * K_>(a);
  auto vb = load_as<TB, K_ * N_>(b);
#ifdef NO_MMUL
  // Keep both operands live so the loads and casts are instantiated and emitted.
  aie::store_v(reinterpret_cast<int8_t *>(c), va.template cast_to<int8>());
  aie::store_v(reinterpret_cast<int8_t *>(c) + 256, vb.template cast_to<int8>());
#else
  aie::mmul<M_, K_, N_, TA, TB, ACC> m;
  m.mul(va, vb);
  aie::store_v(c, m.template to_vector<int32>());
#endif
}
