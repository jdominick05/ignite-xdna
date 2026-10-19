// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// EmbeddingGemma 2 local attention, one core: 16 query rows of one head (head_dim 256) against a window of
// 32-key blocks, flash order (online softmax), scaling 1.0, band |i - j| <= W built on the core from indices.
//
// Per key block the host streams eight 4 KiB chunks: four K^T chunks (head dims 64c..64c+63 x 32 keys, mmul B
// tiles [8][8] of 8x4) then four V chunks (32 keys x columns 64c..64c+63, B tiles [4][16] of 8x4).
//   egattn_score(qpkt, kt, state, c, kb)  S += Q[:, 64c:64c+64] @ Kt_c            (c = 0..3)
//   egattn_softmax(qpkt, state, kb)       (integer, fixed-point S and m) mask; lazy max: m' = rowmax S if it
//                                         exceeds m by more than TAU = 8, else m;
//                                         alpha = exp(m - m'), O *= alpha only where m moved,
//                                         P = bf16(exp(S - m')) <= e^TAU (0 where masked), l = alpha l + rowsum(P)
// Softmax stays in the S tile layout (no row-major pass, no per-row scalar loops).
//   egattn_pv(v, state, c)                O[:, 64c:64c+64] += P @ V_c                (c = 0..3)
//   egattn_finish(state, out)             out = bf16(O / l), row-major [16][256]
// A block no row can see is skipped by all three (its chunks are still consumed). A row that has seen no
// visible key keeps l = 0, and its alpha is forced to 0 (no exp(-inf - -inf)).
//
// Q packet (int32 words): [0] q0 = first query row's position, [1] s = position of the window's first key,
// [2] W = window radius, [3..15] reserved; then Q as bf16 mmul A tiles [4][32] of 4x8 (16 x 256).
// State (fp32 words): O tiles [4][64] of 4x4 (4096) | S tiles [4][8] (512) | unused (512) |
// m (16) | l (16) | skip flag (16) | P bf16 A tiles [4][4] of 4x8 (256 words).
//
// EGATTN_HD (default 256) is the head dim; ROWS = 4096 / HD query rows per call keep every buffer above at its
// HD = 256 size: Q 2048 words, O 4096, out 2048 (the model's global layers, HD = 512, run 8 rows a call). The
// chunks per block are HD / 64 K^T then HD / 64 V; S, P, m and l use the first ROWS rows (lanes 0..ROWS-1).
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef EGATTN_HD
#define EGATTN_HD 256
#endif
constexpr int HD = EGATTN_HD, ROWS = 4096 / HD, R4 = ROWS / 4, NC = HD / 64, OT = HD / 4, QT = HD / 8;
static_assert(HD == 256 || HD == 512, "head dim 256 (16 rows) or 512 (8 rows)");

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
// Fixed point (8 fraction bits) for scores and the running max, so softmax runs on native int32 vector ops (AIE2
// emulates most fp32 vector ops; v4's fp32 softmax spent ~10 us a block, see the BENCHMARKS profile).
using I = aie::vector<int32, 16>;
static inline __attribute__((always_inline)) I ifull(int32_t x) { return aie::broadcast<int32, 16>(x); }
constexpr int32_t TAUF = 8 * 256, LOF = -80 * 256, MASKV = -(1 << 29), MINIT = -(1 << 28);
// Sentinels: a masked score is MASKV, a row's initial max MINIT; no difference formed below can wrap.
extern "C" bfloat16 egexp_ilut_ab[512], egexp_ilut_cd[512], egexp_flut_ab[512], egexp_flut_cd[512];
// exp(d / 256) for d clamped to [LOF, TAUF]: e^(d >> 8) * e^((d & 255) / 256) from two BF16 tables, product exact in
// accfloat. The split is an arithmetic shift and a mask, and both lookups use step 0, so no rounding mode enters
// (a lookup's own index shift follows crRnd: measured, expprobe 20261018T215844Z).
static inline __attribute__((always_inline)) aie::accum<accfloat, 16> expfix(I d) {
  d = aie::min(ifull(TAUF), aie::max(ifull(LOF), d));
  using LT = aie::lut<4, bfloat16, bfloat16>;
  LT li(256, (bfloat16 __aie_dm_resource_a *)egexp_ilut_ab, (bfloat16 __aie_dm_resource_b *)egexp_ilut_cd);
  LT lf(256, (bfloat16 __aie_dm_resource_a *)egexp_flut_ab, (bfloat16 __aie_dm_resource_b *)egexp_flut_cd);
  aie::parallel_lookup<uint16, LT, aie::lut_oor_policy::truncate> il(li, 0), fl(lf, 0);
  auto hi = aie::bit_and(aie::downshift(d, 8), ifull(255)).pack<int16>().cast_to<uint16>();
  auto lo = aie::bit_and(d, ifull(255)).pack<int16>().cast_to<uint16>();
  return aie::mul(il.fetch(hi), fl.fetch(lo));
}

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
static inline __attribute__((always_inline)) I irow_factors(I factors, int group) {
  I ret;
  for (int i = 0; i < 4; ++i) ret.insert<4>(i, aie::broadcast<int32, 4>(factors.get(group * 4 + i)));
  return ret;
}

constexpr int O_OFF = 0, S_OFF = 4096, SC_OFF = 4608, M_OFF = 5120, L_OFF = 5136, SKIP_OFF = 5152, P_OFF = 5168;
constexpr float NEG = -3.402823466e38f;

// 0: no row sees any key of block kb, 1: every row sees every key, 2: partial.
static inline int visibility(const int32_t *hdr, int kb) {
  int q0 = hdr[0], k0 = hdr[1] + 32 * kb, W = hdr[2];
  if (k0 - (q0 + ROWS - 1) > W || q0 - (k0 + 31) > W) return 0;
  if ((k0 + 31) - q0 <= W && (q0 + ROWS - 1) - k0 <= W) return 1;
  return 2;
}

extern "C" void egattn_score(const int32_t *qpkt, const bfloat16 *kt, float *state, int32_t c, int32_t kb) {
  aie::set_rounding(aie::rounding_mode::conv_even);   // to_fixed below: the probe checks which rounding the device used
  if (c == 0 && kb == 0) {
    for (int i = 0; i < 4096; i += 16) aie::store_v(state + O_OFF + i, full(0));
    aie::store_v(reinterpret_cast<int32_t *>(state + M_OFF), ifull(MINIT));
    aie::store_v(state + L_OFF, full(0));
  }
  if (visibility(qpkt, kb) == 0) return;
#ifdef EGATTN_HOLLOW_SCORE   // profiling variant only: wrong output by construction
  return;
#endif
  const bfloat16 *q = reinterpret_cast<const bfloat16 *>(qpkt + 16);
  float *s = state + S_OFF;
  for (int a = 0; a < R4; a += 2)
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
        auto a0 = aie::load_v<32>(q + (a * QT + c * 8 + k) * 32), a1 = aie::load_v<32>(q + ((a + 1) * QT + c * 8 + k) * 32);
        auto b0 = aie::load_v<32>(kt + (k * 8 + b) * 32), b1 = aie::load_v<32>(kt + (k * 8 + b + 1) * 32);
        auto b2 = aie::load_v<32>(kt + (k * 8 + b + 2) * 32), b3 = aie::load_v<32>(kt + (k * 8 + b + 3) * 32);
        c00.mac(a0, b0); c01.mac(a0, b1); c02.mac(a0, b2); c03.mac(a0, b3);
        c10.mac(a1, b0); c11.mac(a1, b1); c12.mac(a1, b2); c13.mac(a1, b3);
      }
      // the last chunk writes S as fixed point (8 fraction bits) for the integer softmax
      auto put = [&](int tile, MM &acc) {
        if (c == NC - 1) aie::store_v(reinterpret_cast<int32_t *>(s) + tile * 16, aie::to_fixed<int32>(acc.to_vector<float>(), 8));
        else aie::store_v(s + tile * 16, acc.to_vector<float>());
      };
      put(a * 8 + b, c00); put(a * 8 + b + 1, c01); put(a * 8 + b + 2, c02); put(a * 8 + b + 3, c03);
      put((a + 1) * 8 + b, c10); put((a + 1) * 8 + b + 1, c11); put((a + 1) * 8 + b + 2, c12); put((a + 1) * 8 + b + 3, c13);
    }
}

extern "C" void egattn_softmax(const int32_t *qpkt, float *state, int32_t kb) {
  aie::set_rounding(aie::rounding_mode::conv_even);   // accfloat -> bf16 (P) rounds to nearest even
  int vis = visibility(qpkt, kb);
  reinterpret_cast<int32_t *>(state)[SKIP_OFF] = vis == 0;   // an int flag: AIE2 has no scalar float compare
  if (vis == 0) return;
#ifdef EGATTN_HOLLOW_SOFTMAX
  return;
#endif
  // Integer domain: S tiles hold to_fixed(S, 8) (written by egattn_score's last chunk), m is fixed point too.
  // Tile (a, b) lane r*4+c is row a*4+r, key b*4+c.
  int32_t *s = reinterpret_cast<int32_t *>(state + S_OFF);
  if (vis == 2) {
    int base = qpkt[1] + 32 * kb - qpkt[0], W = qpkt[2];   // key - row at a = b = 0, r = c = 0
    I off;                                                 // c - r
    for (int j = 0; j < 16; ++j) off.set((j & 3) - (j >> 2), j);
    for (int a = 0; a < R4; ++a)
      for (int b = 0; b < 8; ++b) {
        auto d = aie::add(off, ifull(base + 4 * (b - a)));
        auto bad = aie::gt(d, ifull(W)) | aie::lt(d, ifull(-W));
        int32_t *t = s + (a * 8 + b) * 16;
        aie::store_v(t, aie::select(aie::load_v<16>(t), ifull(MASKV), bad));
      }
  }
  auto group8 = [&](int a) {
    I mx = aie::load_v<16>(s + a * 128);
    for (int b = 1; b < 8; ++b) mx = aie::max(mx, aie::load_v<16>(s + (a * 8 + b) * 16));
    return aie::max(aie::filter_even(mx, 1), aie::filter_odd(mx, 1));   // lane r*2 + c'
  };
  I v01 = aie::concat(group8(0), group8(1));
  I v23 = R4 == 4 ? aie::concat(group8(R4 - 2), group8(R4 - 1)) : v01;   // ROWS = 8: lanes 8..15 repeat rows 0..7
  I rowmax = aie::concat(aie::max(aie::filter_even(v01, 1), aie::filter_odd(v01, 1)),
                         aie::max(aie::filter_even(v23, 1), aie::filter_odd(v23, 1)));   // lane i: row i
  int32_t *mi = reinterpret_cast<int32_t *>(state + M_OFF);
  I oldm = aie::load_v<16>(mi);
  F oldl = aie::load_v<16>(state + L_OFF);
  // Lazy rescale: m moves only where a row max exceeds it by more than TAU (fixed TAUF), so P <= e^8. A row with no
  // visible key keeps m = MINIT (its masked max MASKV is below it).
  auto up = aie::gt(rowmax, aie::add(oldm, ifull(TAUF)));
  I newm = aie::select(oldm, rowmax, up);
  F alpha = aie::select(full(1), expfix(aie::sub(oldm, newm)).to_vector<float>(), up);
  alpha = aie::select(alpha, full(0), aie::eq(oldl.cast_to<int32>(), aie::zeros<int32, 16>()));   // l == +0
#ifndef EGATTN_HOLLOW_RESCALE
  if (!up.empty()) {
    for (int a = 0; a < R4; ++a) {
      F scale = row_factors(alpha, a);
      if (aie::eq(scale, full(1)).full()) continue;
      for (int b = 0; b < OT; ++b) {
        float *o = state + O_OFF + (a * OT + b) * 16;
        aie::store_v(o, mul(aie::load_v<16>(o), scale));
      }
    }
  }
#endif
  bfloat16 *p = reinterpret_cast<bfloat16 *>(state + P_OFF);
  const auto ones = aie::broadcast<bfloat16, 16>(1.0f);
  aie::vector<float, 8> sum8[4] = {aie::zeros<float, 8>(), aie::zeros<float, 8>(), aie::zeros<float, 8>(),
                                   aie::zeros<float, 8>()};
#ifndef EGATTN_HOLLOW_EXP
  for (int a = 0; a < R4; ++a) {
    I mx = irow_factors(newm, a);
    aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
    for (int k = 0; k < 4; ++k) {
      I x0 = aie::load_v<16>(s + (a * 8 + 2 * k) * 16), x1 = aie::load_v<16>(s + (a * 8 + 2 * k + 1) * 16);
      // masked lanes are exactly 0 (an exp of the clamp would be e^-80, nonzero, and would break the l == 0 guard)
      auto e0 = aie::select(expfix(aie::sub(x0, mx)).to_vector<bfloat16>(), aie::zeros<bfloat16, 16>(),
                            aie::eq(x0, ifull(MASKV)));
      auto e1 = aie::select(expfix(aie::sub(x1, mx)).to_vector<bfloat16>(), aie::zeros<bfloat16, 16>(),
                            aie::eq(x1, ifull(MASKV)));
      acc = aie::mac(aie::mac(acc, e0, ones), e1, ones);
      // P A tile (a, k) is 4 rows x 8 keys: each row = 4 keys of tile 2k then 4 keys of tile 2k+1
      auto [lo, hi] = aie::interleave_zip(e0, e1, 4);
      aie::store_v(p + (a * 4 + k) * 32, lo);
      aie::store_v(p + (a * 4 + k) * 32 + 16, hi);
    }
    F t = acc.to_vector<float>();
    sum8[a] = aie::add(aie::filter_even(t, 1), aie::filter_odd(t, 1));   // lane r*2 + c'
  }
  F w01 = aie::concat(sum8[0], sum8[1]), w23 = R4 == 4 ? aie::concat(sum8[2], sum8[3]) : w01;
  F sums = aie::concat(aie::add(aie::filter_even(w01, 1), aie::filter_odd(w01, 1)),
                       aie::add(aie::filter_even(w23, 1), aie::filter_odd(w23, 1)));   // lane i: row i
#else
  F sums = full(1);
#endif
  aie::store_v(mi, newm);
  aie::store_v(state + L_OFF, aie::add(mul(oldl, alpha), sums));
}

extern "C" void egattn_pv(const bfloat16 *v, float *state, int32_t c) {
  if (reinterpret_cast<const int32_t *>(state)[SKIP_OFF]) return;
#ifdef EGATTN_HOLLOW_PV
  return;
#endif
  const bfloat16 *p = reinterpret_cast<const bfloat16 *>(state + P_OFF);
  float *o = state + O_OFF;
  for (int a = 0; a < R4; a += 2)
    for (int b = 0; b < 16; b += 4) {
      int n = c * 16 + b;
      MM c00(aie::load_v<16>(o + (a * OT + n) * 16)), c01(aie::load_v<16>(o + (a * OT + n + 1) * 16));
      MM c02(aie::load_v<16>(o + (a * OT + n + 2) * 16)), c03(aie::load_v<16>(o + (a * OT + n + 3) * 16));
      MM c10(aie::load_v<16>(o + ((a + 1) * OT + n) * 16)), c11(aie::load_v<16>(o + ((a + 1) * OT + n + 1) * 16));
      MM c12(aie::load_v<16>(o + ((a + 1) * OT + n + 2) * 16)), c13(aie::load_v<16>(o + ((a + 1) * OT + n + 3) * 16));
      for (int k = 0; k < 4; ++k) {
        auto a0 = aie::load_v<32>(p + (a * 4 + k) * 32), a1 = aie::load_v<32>(p + ((a + 1) * 4 + k) * 32);
        auto b0 = aie::load_v<32>(v + (k * 16 + b) * 32), b1 = aie::load_v<32>(v + (k * 16 + b + 1) * 32);
        auto b2 = aie::load_v<32>(v + (k * 16 + b + 2) * 32), b3 = aie::load_v<32>(v + (k * 16 + b + 3) * 32);
        c00.mac(a0, b0); c01.mac(a0, b1); c02.mac(a0, b2); c03.mac(a0, b3);
        c10.mac(a1, b0); c11.mac(a1, b1); c12.mac(a1, b2); c13.mac(a1, b3);
      }
      aie::store_v(o + (a * OT + n) * 16, c00.to_vector<float>()); aie::store_v(o + (a * OT + n + 1) * 16, c01.to_vector<float>());
      aie::store_v(o + (a * OT + n + 2) * 16, c02.to_vector<float>()); aie::store_v(o + (a * OT + n + 3) * 16, c03.to_vector<float>());
      aie::store_v(o + ((a + 1) * OT + n) * 16, c10.to_vector<float>()); aie::store_v(o + ((a + 1) * OT + n + 1) * 16, c11.to_vector<float>());
      aie::store_v(o + ((a + 1) * OT + n + 2) * 16, c12.to_vector<float>()); aie::store_v(o + ((a + 1) * OT + n + 3) * 16, c13.to_vector<float>());
    }
}

extern "C" void egattn_finish(const float *state, bfloat16 *out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#ifdef EGATTN_EXP_PROBE   // diagnostic: out[j] = bf16(expfix(LOF + (11 j) / 2)), j = 0..4095, nothing else
  for (int j = 0; j < 4096; j += 16) {
    I d;
    for (int i = 0; i < 16; ++i) d.set(LOF + (11 * (j + i)) / 2, i);
    aie::store_v(out + j, expfix(d).to_vector<bfloat16>());
  }
  return;
#endif
  F inv = reciprocal16(aie::load_v<16>(state + L_OFF));
  for (int a = 0; a < R4; ++a) {
    F scale = row_factors(inv, a);
    for (int b = 0; b < OT; b += 4) {
      // four adjacent O tiles hold rows a*4..a*4+3, columns b*4..b*4+15: emit each row's 16 columns as one bf16 store
      F t0 = mul(aie::load_v<16>(state + O_OFF + (a * OT + b) * 16), scale);
      F t1 = mul(aie::load_v<16>(state + O_OFF + (a * OT + b + 1) * 16), scale);
      F t2 = mul(aie::load_v<16>(state + O_OFF + (a * OT + b + 2) * 16), scale);
      F t3 = mul(aie::load_v<16>(state + O_OFF + (a * OT + b + 3) * 16), scale);
      for (int r = 0; r < 4; ++r) {
        F row = aie::concat(t0.extract<4>(r), t1.extract<4>(r), t2.extract<4>(r), t3.extract<4>(r));
        aie::store_v(out + (a * 4 + r) * HD + b * 4, bf(row));
      }
    }
  }
}
