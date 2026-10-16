// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Device-only GeGLU and operand conversion for a composed MLP.
#include <aie_api/aie.hpp>
#include <stdint.h>
using F=aie::vector<float,16>;
static inline F full(float x) { return aie::broadcast<float,16>(x); }
static inline F mul(F x,F y) { return aie::mul(x,y).to_vector<float>(); }
static inline F reciprocal(F x) {
  F y=aie::sub(aie::broadcast<int32,16>(0x7ef311c3),x.cast_to<int32>()).cast_to<float>();
  for(int i=0;i<3;++i) y=mul(y,aie::sub(full(2),mul(x,y)));
  return y;
}
static inline F exp_negative(F x) {
  auto tiny=aie::lt(x,full(-80));
  x=aie::max(full(-80),x);
  auto n=aie::to_fixed<int32>(mul(x,full(1.4426950408889634f)));
  F r=aie::sub(x,mul(aie::to_float(n),full(.6931471805599453f)));
  F p=full(.0013888888888889f);
  p=aie::add(full(.0083333333333333f),mul(r,p));
  p=aie::add(full(.0416666666666667f),mul(r,p));
  p=aie::add(full(.1666666666666667f),mul(r,p));
  p=aie::add(full(.5f),mul(r,p));
  p=aie::add(full(1),mul(r,p));
  p=aie::add(full(1),mul(r,p));
  auto power=aie::upshift(aie::add(n,aie::broadcast<int32,16>(127)),23).cast_to<float>();
  return aie::select(mul(p,power),full(0),tiny);
}
static __attribute__((noinline)) F gelu(F x) {
  // This AIE API lowers abs(float-vector) to integer vabs.s32. Clear
  // the IEEE sign bit explicitly so negative inputs retain their magnitude.
  F magnitude=aie::bit_and(x.cast_to<uint32>(),aie::broadcast<uint32,16>(0x7fffffff)).cast_to<float>();
  F z=mul(magnitude,full(.7071067811865475f));
  F t=reciprocal(aie::add(full(1),mul(full(.3275911f),z)));
  F p=full(1.061405429f);
  p=aie::add(full(-1.453152027f),mul(t,p));
  p=aie::add(full(1.421413741f),mul(t,p));
  p=aie::add(full(-.284496736f),mul(t,p));
  p=aie::add(full(.254829592f),mul(t,p));
  F tail=mul(mul(p,t),exp_negative(aie::sub(full(0),mul(z,z))));
  F cdf=aie::select(aie::sub(full(1),mul(full(.5f),tail)),mul(full(.5f),tail),aie::lt(x,full(0)));
  return mul(x,cdf);
}
extern "C" void transformer_matmul(const bfloat16 *,const bfloat16 *,float *,int,int,int);
extern "C" void chain_gemm_step(const int32_t *q,const int32_t *b,float *state,int reset) {
  transformer_matmul(reinterpret_cast<const bfloat16 *>(q+16),reinterpret_cast<const bfloat16 *>(b),state,reset,8,16);
}
extern "C" void chain_geglu(const int32_t *q,const int32_t *v,int32_t *out,int poison) {
  const float *gate=reinterpret_cast<const float *>(q+16),*value=reinterpret_cast<const float *>(v+16);
  for(int i=0;i<2080;i+=16) aie::store_v(out+i,aie::broadcast<int32,16>(poison));
  // Explicit integer RNE avoids changing crRnd, which attention owns.
  bfloat16 *packed=reinterpret_cast<bfloat16 *>(out+16);
  for(int row=0;row<32;++row) for(int col=0;col<64;col+=16) {
    F y=mul(gelu(aie::load_v<16>(gate+row*64+col)),aie::load_v<16>(value+row*64+col));
    auto bits=y.cast_to<uint32>();
    auto rounded=aie::add(bits,aie::add(aie::broadcast<uint32,16>(0x7fff),aie::bit_and(aie::downshift(bits,16),aie::broadcast<uint32,16>(1))));
    auto short_bits=aie::filter_even(aie::downshift(rounded,16).cast_to<uint16>()).cast_to<bfloat16>();
    const int offset=(row/4)*256+(col/8)*32+(row%4)*8;
    aie::store_v(packed+offset,short_bits.extract<8>(0));
    aie::store_v(packed+offset+32,short_bits.extract<8>(1));
  }
}
