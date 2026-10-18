// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// EmbeddingGemma 2 local attention, one core: 16 query rows of one head (head_dim 256) against a window of
// 32-key blocks, flash order (online softmax), scaling 1.0, band |i - j| <= W built on the core from indices.
//
// Per key block the host streams eight 4 KiB chunks: four K^T chunks (head dims 64c..64c+63 x 32 keys, mmul B
// tiles [8][8] of 8x4) then four V chunks (32 keys x columns 64c..64c+63, B tiles [4][16] of 8x4).
//   egattn_score(qpkt, kt, state, c, kb)  S += Q[:, 64c:64c+64] @ Kt_c            (c = 0..3)
//   egattn_softmax(qpkt, state, kb)       mask, m' = max(m, rowmax S), alpha = exp(m - m'), O *= alpha,
//                                         P = bf16(exp(S - m')) (0 where masked), l = alpha l + rowsum(P)
//   egattn_pv(v, state, c)                O[:, 64c:64c+64] += P @ V_c                (c = 0..3)
//   egattn_finish(state, out)             out = bf16(O / l), row-major [16][256]
// A block no row can see is skipped by all three (its chunks are still consumed). A row that has seen no
// visible key keeps l = 0, and its alpha is forced to 0 (no exp(-inf - -inf)).
//
// Q packet (int32 words): [0] q0 = first query row's position, [1] s = position of the window's first key,
// [2] W = window radius, [3..15] reserved; then Q as bf16 mmul A tiles [4][32] of 4x8 (16 x 256).
// State (fp32 words): O tiles [4][64] of 4x4 (4096) | S tiles [4][8] (512) | scores row-major [16][32] (512) |
// m (16) | l (16) | skip flag (16) | P bf16 A tiles [4][4] of 4x8 (256 words).
#include <aie_api/aie.hpp>
#include <stdint.h>

using F = aie::vector<float, 16>;
using MM = aie::mmul<4, 8, 4, bfloat16, bfloat16, accfloat>;
static inline __attribute__((always_inline)) F full(float x) { return aie::broadcast<float, 16>(x); }
static inline __attribute__((always_inline)) F mul(F x, F y) { return aie::mul(x, y).to_vector<float>(); }
static inline __attribute__((always_inline)) F fp(aie::vector<bfloat16, 16> x) {
  return aie::accum<accfloat, 16>(x).to_vector<float>();
}
static inline __attribute__((always_inline)) aie::vector<bfloat16, 16> bf(F x) {
  return aie::accum<accfloat, 16>(x).to_vector<bfloat16>();
}
#include "exp_fp32.h"

static inline __attribute__((always_inline)) F reciprocal16(F x) {
  auto zero = aie::eq(x, full(0));
  x = aie::select(x, full(1), zero);
  auto seed = aie::sub(aie::broadcast<int32, 16>(0x7ef311c3), x.cast_to<int32>());
  F y = seed.cast_to<float>();
  for (int i = 0; i < 3; ++i) y = mul(y, aie::sub(full(2), mul(x, y)));
  return aie::select(y, full(0), zero);
}
static inline __attribute__((always_inline)) F row_factors(F factors, int group) {
  F ret;
  for (int i = 0; i < 4; ++i) ret.insert<4>(i, aie::broadcast<float, 4>(factors.get(group * 4 + i)));
  return ret;
}

constexpr int O_OFF = 0, S_OFF = 4096, SC_OFF = 4608, M_OFF = 5120, L_OFF = 5136, SKIP_OFF = 5152, P_OFF = 5168;
constexpr float NEG = -3.402823466e38f;

// 0: no row sees any key of block kb, 1: every row sees every key, 2: partial.
static inline int visibility(const int32_t *hdr, int kb) {
  int q0 = hdr[0], k0 = hdr[1] + 32 * kb, W = hdr[2];
  if (k0 - (q0 + 15) > W || q0 - (k0 + 31) > W) return 0;
  if ((k0 + 31) - q0 <= W && (q0 + 15) - k0 <= W) return 1;
  return 2;
}

extern "C" void egattn_score(const int32_t *qpkt, const bfloat16 *kt, float *state, int32_t c, int32_t kb) {
  if (c == 0 && kb == 0) {
    for (int i = 0; i < 4096; i += 16) aie::store_v(state + O_OFF + i, full(0));
    aie::store_v(state + M_OFF, full(NEG));
    aie::store_v(state + L_OFF, full(0));
  }
  if (visibility(qpkt, kb) == 0) return;
  const bfloat16 *q = reinterpret_cast<const bfloat16 *>(qpkt + 16);
  float *s = state + S_OFF;
  for (int a = 0; a < 4; a += 2)
    for (int b = 0; b < 8; b += 4) {
      MM c00, c01, c02, c03, c10, c11, c12, c13;
      if (c == 0) {
        c00 = MM(full(0)); c01 = MM(full(0)); c02 = MM(full(0)); c03 = MM(full(0));
        c10 = MM(full(0)); c11 = MM(full(0)); c12 = MM(full(0)); c13 = MM(full(0));
      } else {
        c00 = MM(aie::load_v<16>(s + (a * 8 + b) * 16)); c01 = MM(aie::load_v<16>(s + (a * 8 + b + 1) * 16));
        c02 = MM(aie::load_v<16>(s + (a * 8 + b + 2) * 16)); c03 = MM(aie::load_v<16>(s + (a * 8 + b + 3) * 16));
        c10 = MM(aie::load_v<16>(s + ((a + 1) * 8 + b) * 16)); c11 = MM(aie::load_v<16>(s + ((a + 1) * 8 + b + 1) * 16));
        c12 = MM(aie::load_v<16>(s + ((a + 1) * 8 + b + 2) * 16)); c13 = MM(aie::load_v<16>(s + ((a + 1) * 8 + b + 3) * 16));
      }
      for (int k = 0; k < 8; ++k) {
        auto a0 = aie::load_v<32>(q + (a * 32 + c * 8 + k) * 32), a1 = aie::load_v<32>(q + ((a + 1) * 32 + c * 8 + k) * 32);
        auto b0 = aie::load_v<32>(kt + (k * 8 + b) * 32), b1 = aie::load_v<32>(kt + (k * 8 + b + 1) * 32);
        auto b2 = aie::load_v<32>(kt + (k * 8 + b + 2) * 32), b3 = aie::load_v<32>(kt + (k * 8 + b + 3) * 32);
        c00.mac(a0, b0); c01.mac(a0, b1); c02.mac(a0, b2); c03.mac(a0, b3);
        c10.mac(a1, b0); c11.mac(a1, b1); c12.mac(a1, b2); c13.mac(a1, b3);
      }
      aie::store_v(s + (a * 8 + b) * 16, c00.to_vector<float>()); aie::store_v(s + (a * 8 + b + 1) * 16, c01.to_vector<float>());
      aie::store_v(s + (a * 8 + b + 2) * 16, c02.to_vector<float>()); aie::store_v(s + (a * 8 + b + 3) * 16, c03.to_vector<float>());
      aie::store_v(s + ((a + 1) * 8 + b) * 16, c10.to_vector<float>()); aie::store_v(s + ((a + 1) * 8 + b + 1) * 16, c11.to_vector<float>());
      aie::store_v(s + ((a + 1) * 8 + b + 2) * 16, c12.to_vector<float>()); aie::store_v(s + ((a + 1) * 8 + b + 3) * 16, c13.to_vector<float>());
    }
}

extern "C" void egattn_softmax(const int32_t *qpkt, float *state, int32_t kb) {
  // fp32 -> bf16 must round to nearest even; another kernel may have left crRnd elsewhere. exp16's to_fixed split
  // is exact under any integer rounding (its remainder absorbs the choice), so one mode serves both.
  aie::set_rounding(aie::rounding_mode::conv_even);
  int vis = visibility(qpkt, kb);
  reinterpret_cast<int32_t *>(state)[SKIP_OFF] = vis == 0;   // an int flag: AIE2 has no scalar float compare
  if (vis == 0) return;
  const float *s = state + S_OFF;
  float *scores = state + SC_OFF;
  // S tiles -> row-major [16][32]: tile (a, b) lane r*4+col is row a*4+r, key b*4+col.
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 8; ++b) {
      F t = aie::load_v<16>(s + (a * 8 + b) * 16);
      for (int r = 0; r < 4; ++r) aie::store_v(scores + (a * 4 + r) * 32 + b * 4, t.extract<4>(r));
    }
  int q0 = qpkt[0], k0 = qpkt[1] + 32 * kb, W = qpkt[2];
  aie::vector<int32, 16> lane;
  for (int j = 0; j < 16; ++j) lane.set(j, j);
  F maxima;
  for (int i = 0; i < 16; ++i) {
    F s0 = aie::load_v<16>(scores + i * 32), s1 = aie::load_v<16>(scores + i * 32 + 16);
    if (vis == 2) {
      // key - row for lanes 0..15 and 16..31; visible iff -W <= d <= W
      auto d0 = aie::add(lane, aie::broadcast<int32, 16>(k0 - (q0 + i)));
      auto d1 = aie::add(lane, aie::broadcast<int32, 16>(k0 + 16 - (q0 + i)));
      auto bad0 = aie::gt(d0, aie::broadcast<int32, 16>(W)) | aie::lt(d0, aie::broadcast<int32, 16>(-W));
      auto bad1 = aie::gt(d1, aie::broadcast<int32, 16>(W)) | aie::lt(d1, aie::broadcast<int32, 16>(-W));
      s0 = aie::select(s0, full(NEG), bad0);
      s1 = aie::select(s1, full(NEG), bad1);
      aie::store_v(scores + i * 32, s0);
      aie::store_v(scores + i * 32 + 16, s1);
    }
    maxima.set(aie::reduce_max(aie::max(s0, s1)), i);
  }
  F oldm = aie::load_v<16>(state + M_OFF), oldl = aie::load_v<16>(state + L_OFF);
  F newm = aie::max(oldm, maxima);
  F alpha = aie::select(exp16(aie::sub(oldm, newm)), full(0), aie::eq(oldl, full(0)));
  for (int a = 0; a < 4; ++a) {
    F scale = row_factors(alpha, a);
    if (aie::eq(scale, full(1)).full()) continue;
    for (int b = 0; b < 64; ++b) {
      float *o = state + O_OFF + (a * 64 + b) * 16;
      aie::store_v(o, mul(aie::load_v<16>(o), scale));
    }
  }
  bfloat16 *pr = reinterpret_cast<bfloat16 *>(state + SC_OFF);   // row-major bf16 P overwrites the consumed scores
  F sums;
  for (int i = 0; i < 16; ++i) {
    F mx = full(newm.get(i));
    F x0 = aie::load_v<16>(scores + i * 32), x1 = aie::load_v<16>(scores + i * 32 + 16);
    auto e0 = bf(exp16(aie::sub(x0, mx))), e1 = bf(exp16(aie::sub(x1, mx)));
    e0 = aie::select(e0, aie::zeros<bfloat16, 16>(), aie::eq(x0, full(NEG)));
    e1 = aie::select(e1, aie::zeros<bfloat16, 16>(), aie::eq(x1, full(NEG)));
    sums.set(aie::reduce_add(aie::add(fp(e0), fp(e1))), i);
    // row i's 32 bf16 occupy fp32 words 16i..16i+15, inside scores row i/2 <= i: already loaded, never re-read
    aie::store_v(pr + i * 32, e0);
    aie::store_v(pr + i * 32 + 16, e1);
  }
  aie::store_v(state + M_OFF, newm);
  aie::store_v(state + L_OFF, aie::add(mul(oldl, alpha), sums));
  // row-major P [16][32] -> A tiles [4][4] of 4x8
  bfloat16 *p = reinterpret_cast<bfloat16 *>(state + P_OFF);
  for (int a = 0; a < 4; ++a)
    for (int k = 0; k < 4; ++k) {
      auto blk = aie::concat(aie::load_v<8>(pr + (a * 4) * 32 + k * 8), aie::load_v<8>(pr + (a * 4 + 1) * 32 + k * 8),
                             aie::load_v<8>(pr + (a * 4 + 2) * 32 + k * 8), aie::load_v<8>(pr + (a * 4 + 3) * 32 + k * 8));
      aie::store_v(p + (a * 4 + k) * 32, blk);
    }
}

extern "C" void egattn_pv(const bfloat16 *v, float *state, int32_t c) {
  if (reinterpret_cast<const int32_t *>(state)[SKIP_OFF]) return;
  const bfloat16 *p = reinterpret_cast<const bfloat16 *>(state + P_OFF);
  float *o = state + O_OFF;
  for (int a = 0; a < 4; a += 2)
    for (int b = 0; b < 16; b += 4) {
      int n = c * 16 + b;
      MM c00(aie::load_v<16>(o + (a * 64 + n) * 16)), c01(aie::load_v<16>(o + (a * 64 + n + 1) * 16));
      MM c02(aie::load_v<16>(o + (a * 64 + n + 2) * 16)), c03(aie::load_v<16>(o + (a * 64 + n + 3) * 16));
      MM c10(aie::load_v<16>(o + ((a + 1) * 64 + n) * 16)), c11(aie::load_v<16>(o + ((a + 1) * 64 + n + 1) * 16));
      MM c12(aie::load_v<16>(o + ((a + 1) * 64 + n + 2) * 16)), c13(aie::load_v<16>(o + ((a + 1) * 64 + n + 3) * 16));
      for (int k = 0; k < 4; ++k) {
        auto a0 = aie::load_v<32>(p + (a * 4 + k) * 32), a1 = aie::load_v<32>(p + ((a + 1) * 4 + k) * 32);
        auto b0 = aie::load_v<32>(v + (k * 16 + b) * 32), b1 = aie::load_v<32>(v + (k * 16 + b + 1) * 32);
        auto b2 = aie::load_v<32>(v + (k * 16 + b + 2) * 32), b3 = aie::load_v<32>(v + (k * 16 + b + 3) * 32);
        c00.mac(a0, b0); c01.mac(a0, b1); c02.mac(a0, b2); c03.mac(a0, b3);
        c10.mac(a1, b0); c11.mac(a1, b1); c12.mac(a1, b2); c13.mac(a1, b3);
      }
      aie::store_v(o + (a * 64 + n) * 16, c00.to_vector<float>()); aie::store_v(o + (a * 64 + n + 1) * 16, c01.to_vector<float>());
      aie::store_v(o + (a * 64 + n + 2) * 16, c02.to_vector<float>()); aie::store_v(o + (a * 64 + n + 3) * 16, c03.to_vector<float>());
      aie::store_v(o + ((a + 1) * 64 + n) * 16, c10.to_vector<float>()); aie::store_v(o + ((a + 1) * 64 + n + 1) * 16, c11.to_vector<float>());
      aie::store_v(o + ((a + 1) * 64 + n + 2) * 16, c12.to_vector<float>()); aie::store_v(o + ((a + 1) * 64 + n + 3) * 16, c13.to_vector<float>());
    }
}

extern "C" void egattn_finish(const float *state, bfloat16 *out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  F inv = reciprocal16(aie::load_v<16>(state + L_OFF));
  for (int a = 0; a < 4; ++a) {
    F scale = row_factors(inv, a);
    for (int b = 0; b < 64; b += 4) {
      // four adjacent O tiles hold rows a*4..a*4+3, columns b*4..b*4+15: emit each row's 16 columns as one bf16 store
      F t0 = mul(aie::load_v<16>(state + O_OFF + (a * 64 + b) * 16), scale);
      F t1 = mul(aie::load_v<16>(state + O_OFF + (a * 64 + b + 1) * 16), scale);
      F t2 = mul(aie::load_v<16>(state + O_OFF + (a * 64 + b + 2) * 16), scale);
      F t3 = mul(aie::load_v<16>(state + O_OFF + (a * 64 + b + 3) * 16), scale);
      for (int r = 0; r < 4; ++r) {
        F row = aie::concat(t0.extract<4>(r), t1.extract<4>(r), t2.extract<4>(r), t3.extract<4>(r));
        aie::store_v(out + (a * 4 + r) * 256 + b * 4, bf(row));
      }
    }
  }
}
