// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Output-tile epilogue for kernels/gemm_generic/whole_array_generic.py --epilogue-src (m 32, n 64, f32 C),
// called by every core after a tile's last K step with the mode its runtime sequence wrote:
//   mode 0: nothing (every plain weight product).
//   mode 1: GeGLU. The host interleaves gate|up's columns at weight upload so output tile p holds gate columns
//           32p..32p+31 then up columns 32p..32p+31. The C tile is stored in 4x4 blocks, block (z, j) at
//           (z * 16 + j) * 16 (the design's c_dims), so gate block j and up block j + 8 hold the same rows and
//           intermediate columns: gate block j <- gelu_tanh(gate) * up, in fp32. The host reads columns 0..31.
// gelu_tanh(x) = x * sigmoid(2 sqrt(2/pi) (x + 0.044715 x^3)), the identity behind HF's gelu_pytorch_tanh, not an
// approximation of it. exp(-|z|) = 2^n 2^f with n = round(t), |f| <= 1/2 and 2^f by a degree-7 Taylor polynomial
// in f ln 2 (error <= 0.347^8 / 8! = 5.2e-9); 1 / (1 + e) by the Newton reciprocal attn_local.cc uses. No tables.
#include <aie_api/aie.hpp>
#include <stdint.h>

using F = aie::vector<float, 16>;

static inline __attribute__((always_inline)) F full(float x) { return aie::broadcast<float, 16>(x); }
static inline __attribute__((always_inline)) F mul(F a, F b) { return aie::mul(a, b).to_vector<float>(); }

// exp(-y) for y >= 0 (clamped at 80: e^-80 = 1.8e-35 is still normal)
static inline __attribute__((always_inline)) F expneg(F y) {
  F t = mul(aie::min(y, full(80.0f)), full(-1.4426950408889634f));   // -y log2 e, in [-115.5, 0]
  auto n = aie::to_fixed<int32>(t, 0);                                // conv_even: round half even
  F u = mul(aie::sub(t, aie::to_float(n)), full(0.6931471805599453f));
  F p = full(1.0f / 5040.0f);
  p = aie::add(mul(p, u), full(1.0f / 720.0f));
  p = aie::add(mul(p, u), full(1.0f / 120.0f));
  p = aie::add(mul(p, u), full(1.0f / 24.0f));
  p = aie::add(mul(p, u), full(1.0f / 6.0f));
  p = aie::add(mul(p, u), full(0.5f));
  p = aie::add(mul(p, u), full(1.0f));
  p = aie::add(mul(p, u), full(1.0f));
  return aie::add(p.cast_to<int32>(), aie::upshift(n, 23)).cast_to<float>();   // times 2^n
}

// 1 / x for x in [1, 2]: magic seed, three Newton steps
static inline __attribute__((always_inline)) F reciprocal16(F x) {
  F y = aie::sub(aie::broadcast<int32, 16>(0x7ef311c3), x.cast_to<int32>()).cast_to<float>();
  for (int i = 0; i < 3; ++i) y = mul(y, aie::sub(full(2.0f), mul(x, y)));
  return y;
}

static inline __attribute__((always_inline)) F gelu_tanh(F x) {
  F z = mul(mul(x, aie::add(full(1.0f), mul(mul(x, x), full(0.044715f)))), full(1.5957691216057308f));
  // |z| by clearing the sign bit: aie::abs on a float vector took the integer abs of its bits (measured,
  // results/aie/embedgemma_gemm_epilogue_geglu_diag_desktop2_20261018T232039Z: wrong exactly where z < 0)
  F e = expneg(aie::bit_and(z.cast_to<int32>(), aie::broadcast<int32, 16>(0x7fffffff)).cast_to<float>());
  F r = reciprocal16(aie::add(full(1.0f), e));
  F s = aie::select(mul(e, r), r, aie::ge(z, full(0.0f)));   // sigmoid(z): r for z >= 0, e r below
  return mul(x, s);
}

// mode 2: the same GeGLU from native bf16 products (fp32 accumulation), each rounded where geglu_model.py's mode-2
// model rounds it; exp(-|z|) from attn_local.cc v5's two BF16 tables on a fixed-point z (no rounding mode enters
// the lookup: shift and mask, step 0). Mode 1's fp32 arithmetic is emulated on AIE2 and cost ~1,540 cycles a vector.
#include "geglu_tables.h"
using B16 = aie::vector<bfloat16, 16>;
using I = aie::vector<int32, 16>;
using A = aie::accum<accfloat, 16>;
static inline __attribute__((always_inline)) I ifull(int32_t x) { return aie::broadcast<int32, 16>(x); }
static inline __attribute__((always_inline)) B16 bfull(float x) { return aie::broadcast<bfloat16, 16>(x); }
static inline __attribute__((always_inline)) A acc(F x) { A a; a.from_vector(x); return a; }

static inline __attribute__((always_inline)) F geglu_bf16(F g, F u) {
  using LT = aie::lut<4, bfloat16, bfloat16>;
  LT li(256, (bfloat16 __aie_dm_resource_a *)eggl_ilut_ab, (bfloat16 __aie_dm_resource_b *)eggl_ilut_cd);
  LT lf(256, (bfloat16 __aie_dm_resource_a *)eggl_flut_ab, (bfloat16 __aie_dm_resource_b *)eggl_flut_cd);
  aie::parallel_lookup<uint16, LT, aie::lut_oor_policy::truncate> il(li, 0), fl(lf, 0);
  B16 gb = acc(g).to_vector<bfloat16>();
  B16 c = aie::mul(aie::mul(gb, gb).to_vector<bfloat16>(), bfull(0.044715f)).to_vector<bfloat16>();
  B16 wb = aie::mac(acc(g), gb, c).to_vector<bfloat16>();                      // g + 0.044715 g^3
  I zq = aie::to_fixed<int32>(aie::mul(wb, bfull(408.51689f)).to_vector<float>(), 0);   // 256 * 2 sqrt(2/pi) w
  I d = aie::max(aie::neg(aie::abs(zq)), ifull(-80 * 256));                   // integer abs: exact
  auto hi = aie::bit_and(aie::downshift(d, 8), ifull(255)).pack<int16>().cast_to<uint16>();
  auto lo = aie::bit_and(d, ifull(255)).pack<int16>().cast_to<uint16>();
  B16 e = aie::mul(il.fetch(hi), fl.fetch(lo)).to_vector<bfloat16>();          // e^-|z|
  B16 den = aie::mac(acc(full(1.0f)), e, bfull(1.0f)).to_vector<bfloat16>();  // 1 + e
  F denf = aie::mul(den, bfull(1.0f)).to_vector<float>();
  B16 y = acc(aie::sub(ifull(0x7ef311c3), denf.cast_to<int32>()).cast_to<float>()).to_vector<bfloat16>();
  for (int i = 0; i < 2; ++i)
    y = aie::mul(y, aie::msc(acc(full(2.0f)), den, y).to_vector<bfloat16>()).to_vector<bfloat16>();
  B16 s = aie::select(aie::mul(e, y).to_vector<bfloat16>(), y, aie::ge(zq, ifull(0)));   // sigmoid(z)
  return aie::mul(aie::mul(gb, s).to_vector<bfloat16>(), acc(u).to_vector<bfloat16>()).to_vector<float>();
}

// Timing probes (wrong output by construction): mode 3 = the loop's loads and stores only; mode 4 = mode 2 with
// the two table lookups replaced by a constant.
static inline __attribute__((always_inline)) F geglu_nolut(F g, F u) {
  B16 gb = acc(g).to_vector<bfloat16>();
  B16 c = aie::mul(aie::mul(gb, gb).to_vector<bfloat16>(), bfull(0.044715f)).to_vector<bfloat16>();
  B16 wb = aie::mac(acc(g), gb, c).to_vector<bfloat16>();
  I zq = aie::to_fixed<int32>(aie::mul(wb, bfull(408.51689f)).to_vector<float>(), 0);
  B16 e = bfull(0.5f);
  B16 den = aie::mac(acc(full(1.0f)), e, bfull(1.0f)).to_vector<bfloat16>();
  F denf = aie::mul(den, bfull(1.0f)).to_vector<float>();
  B16 y = acc(aie::sub(ifull(0x7ef311c3), denf.cast_to<int32>()).cast_to<float>()).to_vector<bfloat16>();
  for (int i = 0; i < 2; ++i)
    y = aie::mul(y, aie::msc(acc(full(2.0f)), den, y).to_vector<bfloat16>()).to_vector<bfloat16>();
  B16 s = aie::select(aie::mul(e, y).to_vector<bfloat16>(), y, aie::ge(zq, ifull(0)));
  return aie::mul(aie::mul(gb, s).to_vector<bfloat16>(), acc(u).to_vector<bfloat16>()).to_vector<float>();
}

// mode 5: mode 2 without tables. The probes put the two lookups at ~4.4 of mode 2's 5.9 ms per 2048x4096 call
// (results/aie/embedgemma_gemm_epilogue_profile_desktop2_20261018T235109Z; both table halves sit in one L1 bank,
// which IRON's placement does not let this source change). e^-|z| = 2^n 2^f instead: d = -|round(256 z log2 e)|,
// n = d >> 8, f = (d & 255) / 256 (exact in bf16), 2^f by a degree-3 minimax polynomial in bf16 MACs, n added to
// the fp32 exponent. geglu_model.py mode 5 is the model.
static inline __attribute__((always_inline)) F geglu_poly(F g, F u) {
  B16 gb = acc(g).to_vector<bfloat16>();
  B16 c = aie::mul(aie::mul(gb, gb).to_vector<bfloat16>(), bfull(0.044715f)).to_vector<bfloat16>();
  B16 wb = aie::mac(acc(g), gb, c).to_vector<bfloat16>();                        // g + 0.044715 g^3
  I tq = aie::to_fixed<int32>(aie::mul(wb, bfull(589.3713f)).to_vector<float>(), 0);   // 256 z log2 e
  I d = aie::max(aie::neg(aie::abs(tq)), ifull(-100 * 256));
  I n = aie::downshift(d, 8);                                                    // floor
  B16 f = acc(aie::to_float(aie::bit_and(d, ifull(255)), 8)).to_vector<bfloat16>();   // exact
  B16 p = aie::mac(acc(full(0.2260760f)), f, bfull(0.0781021f)).to_vector<bfloat16>();
  p = aie::mac(acc(full(0.6958160f)), p, f).to_vector<bfloat16>();
  F pf = aie::mac(acc(full(0.9999925f)), p, f).to_vector<float>();               // 2^f
  B16 e = acc(aie::add(pf.cast_to<int32>(), aie::upshift(n, 23)).cast_to<float>()).to_vector<bfloat16>();
  B16 den = aie::mac(acc(full(1.0f)), e, bfull(1.0f)).to_vector<bfloat16>();
  F denf = aie::mul(den, bfull(1.0f)).to_vector<float>();
  B16 y = acc(aie::sub(ifull(0x7ef311c3), denf.cast_to<int32>()).cast_to<float>()).to_vector<bfloat16>();
  for (int i = 0; i < 2; ++i)
    y = aie::mul(y, aie::msc(acc(full(2.0f)), den, y).to_vector<bfloat16>()).to_vector<bfloat16>();
  B16 s = aie::select(aie::mul(e, y).to_vector<bfloat16>(), y, aie::ge(tq, ifull(0)));
  return aie::mul(aie::mul(gb, s).to_vector<bfloat16>(), acc(u).to_vector<bfloat16>()).to_vector<float>();
}

extern "C" void egemm_epilogue(float *c, int32_t mode) {
  if (mode < 1 || mode > 6) return;
  // (A cycle-counter probe reading 0x340F8 through read_tm hung the core: dispatch timeout,
  // results/aie/embedgemma_gemm_epilogue_cycles_desktop2_20261019T000938Z. Removed.)
  if (mode == 6) {   // mode 5, two independent vectors per iteration (its loop was 52% NOP-only bundles)
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int z = 0; z < 8; ++z)
      for (int j = 0; j < 8; j += 2) {
        float *g0 = c + (z * 16 + j) * 16, *g1 = g0 + 16;
        F a0 = aie::load_v<16>(g0), b0 = aie::load_v<16>(g0 + 8 * 16);
        F a1 = aie::load_v<16>(g1), b1 = aie::load_v<16>(g1 + 8 * 16);
        F r0 = geglu_poly(a0, b0), r1 = geglu_poly(a1, b1);
        aie::store_v(g0, r0);
        aie::store_v(g1, r1);
      }
    return;
  }
  if (mode == 5) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int z = 0; z < 8; ++z)
      for (int j = 0; j < 8; ++j) {
        float *g = c + (z * 16 + j) * 16;
        aie::store_v(g, geglu_poly(aie::load_v<16>(g), aie::load_v<16>(g + 8 * 16)));
      }
    return;
  }
  aie::set_rounding(aie::rounding_mode::conv_even);
  if (mode == 3 || mode == 4) {
    for (int z = 0; z < 8; ++z)
      for (int j = 0; j < 8; ++j) {
        float *g = c + (z * 16 + j) * 16;
        F gv = aie::load_v<16>(g), uv = aie::load_v<16>(g + 8 * 16);
        aie::store_v(g, mode == 3 ? uv : geglu_nolut(gv, uv));
      }
    return;
  }
  if (mode == 2) {
    for (int z = 0; z < 8; ++z)
      for (int j = 0; j < 8; ++j) {
        float *g = c + (z * 16 + j) * 16;
        aie::store_v(g, geglu_bf16(aie::load_v<16>(g), aie::load_v<16>(g + 8 * 16)));
      }
    return;
  }
  for (int z = 0; z < 8; ++z)
    for (int j = 0; j < 8; ++j) {
      float *g = c + (z * 16 + j) * 16;
      aie::store_v(g, mul(gelu_tanh(aie::load_v<16>(g)), aie::load_v<16>(g + 8 * 16)));
    }
}
