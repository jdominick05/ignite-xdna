// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// FP32 residual + LayerNorm(1024), epsilon=1e-5. Affine parameters resident
// for 32 rows. Emit both residual sum and normalized activation for reuse.
#include <aie_api/aie.hpp>
#include <stdint.h>
using F=aie::vector<float,16>;
static inline F full(float x) { return aie::broadcast<float,16>(x); }
static inline F mul(F x,F y) { return aie::mul(x,y).to_vector<float>(); }
using U=aie::vector<uint32,16>;
static inline U ui(uint32 x) { return aie::broadcast<uint32,16>(x); }
template<unsigned N> static inline U jam(U v) {
  U sticky=aie::select(ui(1),ui(0),aie::eq(aie::bit_and(v,ui((1u<<N)-1)),ui(0)));
  return aie::bit_or(aie::downshift(v,N),sticky);
}
template<unsigned N> static inline void align_part(U &v,U d) {
  auto use=~aie::eq(aie::bit_and(d,ui(N)),ui(0));
  v=aie::select(v,jam<N>(v),use);
}
template<unsigned N> static inline void normalize_part(U &v,U &e) {
  auto use=aie::lt(v,ui(1u<<(27-N)))&aie::gt(e,ui(N));
  v=aie::select(v,aie::upshift(v,N),use);
  e=aie::select(e,aie::sub(e,ui(N)),use);
}
static inline F residual_add(F x,F r) {
  // Finite FP32 addition with guard/round/sticky bits and ties-to-even.
  // Native vadd.f did not reproduce CPU residual sums in the qualification.
  U a=x.cast_to<uint32>(),b=r.cast_to<uint32>();
  U ma=aie::bit_and(a,ui(0x7fffffff)),mb=aie::bit_and(b,ui(0x7fffffff));
  auto swap=aie::lt(ma,mb);
  U big=aie::select(a,b,swap),small=aie::select(b,a,swap);
  U eb=aie::bit_and(aie::downshift(big,23),ui(255));
  U es=aie::bit_and(aie::downshift(small,23),ui(255));
  U sb=aie::bit_or(aie::bit_and(big,ui(0x7fffff)),aie::select(ui(0x800000),ui(0),aie::eq(eb,ui(0))));
  U ss=aie::bit_or(aie::bit_and(small,ui(0x7fffff)),aie::select(ui(0x800000),ui(0),aie::eq(es,ui(0))));
  eb=aie::max(eb,ui(1)); es=aie::max(es,ui(1));
  U d=aie::sub(eb,es); sb=aie::upshift(sb,3); ss=aie::upshift(ss,3);
  ss=aie::select(ss,aie::select(ui(1),ui(0),aie::eq(ss,ui(0))),aie::ge(d,ui(32)));
  align_part<1>(ss,d); align_part<2>(ss,d); align_part<4>(ss,d); align_part<8>(ss,d); align_part<16>(ss,d);
  auto same=aie::eq(aie::bit_and(aie::bit_xor(big,small),ui(0x80000000)),ui(0));
  U z=aie::select(aie::sub(sb,ss),aie::add(sb,ss),same);
  auto carry=aie::ge(z,ui(1u<<27));
  z=aie::select(z,jam<1>(z),carry); eb=aie::select(eb,aie::add(eb,ui(1)),carry);
  normalize_part<16>(z,eb); normalize_part<8>(z,eb); normalize_part<4>(z,eb); normalize_part<2>(z,eb); normalize_part<1>(z,eb);
  U low=aie::bit_and(z,ui(7)),sig=aie::downshift(z,3);
  auto up=aie::gt(low,ui(4))|(aie::eq(low,ui(4))&~aie::eq(aie::bit_and(sig,ui(1)),ui(0)));
  sig=aie::add(sig,aie::select(ui(0),ui(1),up));
  auto rounded_carry=aie::ge(sig,ui(1u<<24));
  sig=aie::select(sig,aie::downshift(sig,1),rounded_carry);
  eb=aie::select(eb,aie::add(eb,ui(1)),rounded_carry);
  eb=aie::select(eb,ui(0),aie::lt(sig,ui(1u<<23)));
  U sign=aie::bit_and(big,ui(0x80000000));
  sign=aie::select(sign,ui(0),aie::eq(sig,ui(0))&~same);
  U result=aie::bit_or(sign,aie::bit_or(aie::upshift(eb,23),aie::bit_and(sig,ui(0x7fffff))));
  result=aie::select(result,aie::bit_or(sign,ui(0x7f800000)),aie::ge(eb,ui(255)));
  return result.cast_to<float>();
}
extern "C" void residual_norm(const int32_t *in,const float *affine,int32_t *out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *x=reinterpret_cast<const float *>(in+16),*r=x+1024;
  float *sum=reinterpret_cast<float *>(out+16),*y=sum+1024;
  for(int i=0;i<1024;i+=16) {
    F z=residual_add(aie::load_v<16>(x+i),aie::load_v<16>(r+i));
    aie::store_v(sum+i,z);
  }
  F total=full(0),anchor=full(sum[0]);
  for(int i=0;i<1024;i+=16) total=aie::add(total,aie::sub(aie::load_v<16>(sum+i),anchor));
  F mean=mul(full(aie::reduce_add(total)),full(0.0009765625f));
  F squares=full(0);
  for(int i=0;i<1024;i+=16) {
    F z=aie::sub(aie::sub(aie::load_v<16>(sum+i),anchor),mean);
    squares=aie::add(squares,mul(z,z));
  }
  F variance=aie::add(mul(full(aie::reduce_add(squares)),full(0.0009765625f)),full(1e-5f));
  // Vector Newton refinement avoids scalar software floating-point helpers.
  auto halfbits=aie::downshift(variance.cast_to<uint32>(),1);
  F inv=aie::sub(aie::broadcast<uint32,16>(0x5f375a86),halfbits).cast_to<float>();
  for(int j=0;j<4;++j)
    inv=mul(inv,aie::sub(full(1.5f),mul(mul(full(0.5f),variance),mul(inv,inv))));
  for(int i=0;i<1024;i+=16) {
    F centered=aie::sub(aie::sub(aie::load_v<16>(sum+i),anchor),mean);
    F z=mul(mul(centered,inv),aie::load_v<16>(affine+i));
    aie::store_v(y+i,aie::add(z,aie::load_v<16>(affine+1024+i)));
  }
  for(int i=0;i<16;++i) { out[i]=in[0]; out[2064+i]=in[0]; }
}
