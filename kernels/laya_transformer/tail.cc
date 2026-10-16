// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Tail-only image: the program drops unused norm/stem/attention/GeGLU entry points.
#include "vector.cc"

static inline aie::vector<int32,16> indices(int first) {
  aie::vector<int32,16> result;
  for(int lane=0;lane<16;++lane) result.set(first+lane,lane);
  return result;
}
static F logarithm(F x) {
  auto u=x.cast_to<uint32>();
  auto exponent=aie::sub(aie::downshift(u,23).cast_to<int32>(),aie::broadcast<int32,16>(127));
  F mantissa=aie::bit_or(aie::bit_and(u,ui(0x7fffff)),ui(0x3f800000)).cast_to<float>();
  F z=mul(aie::sub(mantissa,full(1)),activation_impl::reciprocal(aie::add(mantissa,full(1))));
  F square=mul(z,z),poly=full(1.0f/15.0f);
  poly=aie::add(full(1.0f/13.0f),mul(square,poly));
  poly=aie::add(full(1.0f/11.0f),mul(square,poly));
  poly=aie::add(full(1.0f/9.0f),mul(square,poly));
  poly=aie::add(full(1.0f/7.0f),mul(square,poly));
  poly=aie::add(full(1.0f/5.0f),mul(square,poly));
  poly=aie::add(full(1.0f/3.0f),mul(square,poly));
  poly=aie::add(full(1),mul(square,poly));
  return aie::add(mul(mul(full(2),z),poly),mul(aie::to_float(exponent),full(.6931471805599453f)));
}
extern "C" void tail_collect(const int32_t *kv,int32_t *state,int packet) {
  for(int row=0;row<32;++row) state[packet*32+row]=kv[16+row*64];
}
extern "C" void tail_finish(const int32_t *q,const int32_t *state,int32_t *out,int poison) {
  for(int i=0;i<2080;i+=16) aie::store_v(out+i,aie::broadcast<int32,16>(poison));
  const float *scores=reinterpret_cast<const float *>(state); float *values=reinterpret_cast<float *>(out);
  int count=q[0],valid_count=0; F maximum=full(-3.402823466e38f);
  for(int i=0;i<128;i+=16) {
    auto real=aie::lt(indices(i),aie::broadcast<int32,16>(count));
    auto flags=aie::load_v<16>(q+16+i);
    auto allowed=~aie::eq(flags,aie::zeros<int32,16>());
    F value=aie::select(full(-10000),aie::load_v<16>(scores+i),allowed);
    aie::store_v(values+16+i,value);
    maximum=aie::max(maximum,aie::select(full(-3.402823466e38f),value,real));
    auto counted=aie::select(aie::zeros<int32,16>(),aie::broadcast<int32,16>(1),real&allowed);
    valid_count+=aie::reduce_add(counted);
  }
  maximum=full(aie::reduce_max(maximum)); F sum=full(0);
  for(int i=0;i<128;i+=16) {
    auto real=aie::lt(indices(i),aie::broadcast<int32,16>(count));
    F shift=aie::select(full(-80),aie::sub(aie::load_v<16>(values+16+i),maximum),real);
    F value=activation_impl::exp_negative(shift);
    value=aie::select(full(0),value,real); aie::store_v(values+160+i,value); sum=aie::add(sum,value);
  }
  F inverse=activation_impl::reciprocal(full(aie::reduce_add(sum))),entropy=full(0);
  for(int i=0;i<128;i+=16) {
    F probability=mul(aie::load_v<16>(values+160+i),inverse);
    F logp=logarithm(aie::max(full(1e-9f),probability));
    aie::store_v(values+160+i,probability); aie::store_v(values+288+i,logp);
    entropy=aie::add(entropy,mul(probability,logp));
  }
  uint32_t first=0,second=0;
  for(int i=0;i<count;++i) {
    uint32_t bits=static_cast<uint32_t>(out[160+i]);
    if(bits>=first) { second=first; first=bits; } else if(bits>second) second=bits;
  }
  F top=aie::broadcast<uint32,16>(first).cast_to<float>(),runner=aie::broadcast<uint32,16>(second).cast_to<float>();
  F n=aie::to_float(aie::broadcast<int32,16>(valid_count<2 ? 2 : valid_count));
  F normalized=mul(aie::sub(full(0),full(aie::reduce_add(entropy))),activation_impl::reciprocal(logarithm(n)));
  out[144]=static_cast<int32_t>(first); values[145]=aie::sub(top,runner).get(0);
  values[146]=normalized.get(0); values[147]=mul(n,full(.00392156862745098f)).get(0);
}
extern "C" void tail_vector(const int32_t *q,const int32_t *kv,int32_t *out,int poison,int mode,int feature,int core_row) {
  if(mode==14 || mode==15) { vector_bias(q,kv,out,poison,mode); return; }
  for(int i=0;i<2080;i+=16) aie::store_v(out+i,aie::broadcast<int32,16>(poison));
  if(mode==20) {
    for(int i=16;i<1040;i+=16) aie::store_v(out+i,aie::zeros<int32,16>());
    if(core_row!=0) return;
    bfloat16 *packed=reinterpret_cast<bfloat16 *>(out+16);
    for(int col=0;col<64;col+=16) {
      F value=full(0);
      if(feature<16) value=aie::load_v<16>(reinterpret_cast<const float *>(q)+feature*64+col);
      else if(col==0) {
        value=aie::load_v<16>(reinterpret_cast<const float *>(kv)+144);
        value=aie::select(full(0),value,aie::lt(indices(0),aie::broadcast<int32,16>(4)));
      }
      auto bits=rounded(value); int offset=(col/8)*32;
      aie::store_v(packed+offset,bits.extract<8>(0)); aie::store_v(packed+offset+32,bits.extract<8>(1));
    }
  } else if(mode==21) {
    auto real=aie::lt(indices(0),aie::broadcast<int32,16>(2));
    F value=aie::select(full(-3.402823466e38f),aie::load_v<16>(reinterpret_cast<const float *>(q)+16),real);
    F weight=activation_impl::exp_negative(aie::sub(value,full(aie::reduce_max(value))));
    weight=aie::select(full(0),weight,real);
    F probability=mul(weight,activation_impl::reciprocal(full(aie::reduce_add(weight))));
    float *target=reinterpret_cast<float *>(out+16); target[0]=probability.get(0); target[1]=probability.get(1);
  }
}
