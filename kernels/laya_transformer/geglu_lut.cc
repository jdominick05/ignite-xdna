// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Separate approximation candidate. The registered polynomial source is untouched.
#define chain_geglu polynomial_chain_geglu
#include "mlp.cc"
#undef chain_geglu

extern "C" float gelu_tail_ab[1024],gelu_tail_cd[1024];

static __attribute__((noinline)) F gelu_lookup(F x) {
  F magnitude=aie::bit_and(x.cast_to<uint32>(),aie::broadcast<uint32,16>(0x7fffffff)).cast_to<float>();
  // 512 tail-CDF samples spaced by 1/32. Clamp the lookup, then supply the
  // asymptote outside it. Fixed-point fractional indexing uses floor crRnd,
  // restored by the existing command_norm fence before each encoder MLP.
  auto outside=aie::ge(magnitude,full(15.9375f));
  F bounded=aie::min(magnitude,full(15.9375f));
  auto fixed=aie::to_fixed<int32>(bounded,16);
  auto index=aie::downshift(fixed,11);
  F fraction=aie::to_float(aie::bit_and(fixed,aie::broadcast<int32,16>(2047)),11);
  using LT=aie::lut<4,float>;
  LT lut(512,(float __aie_dm_resource_a *)gelu_tail_ab,(float __aie_dm_resource_b *)gelu_tail_cd);
  aie::parallel_lookup<uint16,LT,aie::lut_oor_policy::truncate> lookup(lut,0);
  F low=lookup.fetch(index.pack<int16>().cast_to<uint16>());
  F high=lookup.fetch(aie::add(index,aie::broadcast<int32,16>(1)).pack<int16>().cast_to<uint16>());
  F tail=aie::add(low,mul(fraction,aie::sub(high,low)));
  tail=aie::select(tail,full(0),outside);
  F cdf=aie::select(aie::sub(full(1),tail),tail,aie::lt(x,full(0)));
  return mul(x,cdf);
}

extern "C" void chain_geglu(const int32_t *q,const int32_t *v,int32_t *out,int poison) {
  const float *gate=reinterpret_cast<const float *>(q+16),*value=reinterpret_cast<const float *>(v+16);
  for(int i=0;i<2080;i+=16) aie::store_v(out+i,aie::broadcast<int32,16>(poison));
  bfloat16 *packed=reinterpret_cast<bfloat16 *>(out+16);
  for(int row=0;row<32;++row) for(int col=0;col<64;col+=16) {
    F y=mul(gelu_lookup(aie::load_v<16>(gate+row*64+col)),aie::load_v<16>(value+row*64+col));
    auto bits=y.cast_to<uint32>();
    auto rounded=aie::add(bits,aie::add(aie::broadcast<uint32,16>(0x7fff),aie::bit_and(aie::downshift(bits,16),aie::broadcast<uint32,16>(1))));
    auto short_bits=aie::filter_even(aie::downshift(rounded,16).cast_to<uint16>()).cast_to<bfloat16>();
    const int offset=(row/4)*256+(col/8)*32+(row%4)*8;
    aie::store_v(packed+offset,short_bits.extract<8>(0));
    aie::store_v(packed+offset+32,short_bits.extract<8>(1));
  }
}
