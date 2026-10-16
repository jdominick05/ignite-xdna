// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// A 256-entry FP32 table of 2^(i/256). Duplicate groups of four entries for
// the lookup banks: [v0,v1,v2,v3,v0,v1,v2,v3], rather than adjacent duplicates.
// Generate both physical table copies at build time. Keep this function out
// of line: three copies would exceed the linked program-memory budget.
extern "C" float fp_exp_ab[512],fp_exp_cd[512];
static __attribute__((noinline)) F exp16(F x) {
  auto tiny=aie::lt(x,full(-80.0f));
  x=aie::min(full(0),aie::max(full(-80.0f),x));
  auto scaled=aie::to_fixed<int32>(mul(x,full(1.4426950408889634f)),8);
  auto exponent=aie::downshift(scaled,8);
  auto address=aie::bit_and(scaled,aie::broadcast<int32,16>(255));
  F remainder=aie::sub(x,mul(aie::to_float(scaled),full(.0027076061740622863f)));
  using LT=aie::lut<4,float>;
  LT lut(256,(float __aie_dm_resource_a *)fp_exp_ab,(float __aie_dm_resource_b *)fp_exp_cd);
  aie::parallel_lookup<uint16,LT,aie::lut_oor_policy::truncate> lookup(lut,0);
  F base=lookup.fetch(address.pack<int16>().cast_to<uint16>());
  F mantissa=mul(base,aie::add(full(1),remainder));
  // On this clamped domain both input and result are normal FP32. Adjust
  // exponent bits exactly; a floating multiply loses accuracy near exp(-80)
  // because its bf16 low products can underflow.
  F result=aie::add(mantissa.cast_to<int32>(),aie::upshift(exponent,23)).cast_to<float>();
  return aie::select(result,full(0),tiny);
}
