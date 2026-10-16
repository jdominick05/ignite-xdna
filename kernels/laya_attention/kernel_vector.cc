// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// One-core streamed attention. FP arithmetic stays on the vector datapath.
#include <aie_api/aie.hpp>
#include <stdint.h>
using F = aie::vector<float,16>;
using MM = aie::mmul<4,8,4,bfloat16,bfloat16,accfloat>;
static inline __attribute__((always_inline)) F full(float x) { return aie::broadcast<float,16>(x); }
static inline __attribute__((always_inline)) F mul(F x,F y) { return aie::mul(x,y).to_vector<float>(); }
static inline __attribute__((always_inline)) F fp(aie::vector<bfloat16,16> x) { return aie::accum<accfloat,16>(x).to_vector<float>(); }
static inline __attribute__((always_inline)) aie::vector<bfloat16,16> bf(F x) { return aie::accum<accfloat,16>(x).to_vector<bfloat16>(); }
// Nearest-integer range reduction; no scalar floating-point conversions or math.
static inline __attribute__((always_inline)) F exp16(F x) {
  auto tiny=aie::lt(x,full(-80.0f));
  x=aie::min(full(0),aie::max(full(-80.0f),x));
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
static inline __attribute__((always_inline)) F reciprocal16(F x) {
  auto zero=aie::eq(x,full(0));
  x=aie::select(x,full(1),zero);
  auto seed=aie::sub(aie::broadcast<int32,16>(0x7ef311c3),x.cast_to<int32>());
  F y=seed.cast_to<float>();
  for(int i=0;i<3;++i) y=mul(y,aie::sub(full(2),mul(x,y)));
  return aie::select(y,full(0),zero);
}
static inline __attribute__((always_inline)) F row_factors(F factors,int group) {
  F ret;
  for(int i=0;i<4;++i) ret.insert<4>(i,aie::broadcast<float,4>(factors.get(group*4+i)));
  return ret;
}
static inline __attribute__((always_inline)) void score_tile(F s,float *scores,int a,int b) {
  s=mul(s,full(.125f));
  for(int i=0;i<4;++i) aie::store_v(scores+(a*4+i)*32+b*4,s.extract<4>(i));
}
// Unchanged input/output packet ABI. O is tiled [4,16,4,4]. Scratch is 4096 B:
// scores[16,32] f32, row probabilities[16,32] bf16, tiled probabilities[4,4,4,8].
extern "C" void attention_block(const int32_t *packet,float *state,int32_t *scratch) {
  const bfloat16 *q=reinterpret_cast<const bfloat16 *>(packet+16);
  const bfloat16 *kt=q+1024,*v=kt+2048,*mask=v+2048;
  float *scores=reinterpret_cast<float *>(scratch);
  bfloat16 *pr=reinterpret_cast<bfloat16 *>(scratch+512),*p=pr+512;
  float *m=state+1024,*l=m+16;
  if(packet[0]) {
    for(int i=0;i<1024;i+=16) aie::store_v(state+i,full(0));
    aie::store_v(m,full(-3.402823466e38f)); aie::store_v(l,full(0));
  }
  // As in the bf16 convolution/GEMM kernels, keep independent accumulators
  // live and reuse each loaded operand across two products (2x2 expansion).
  for(int a=0;a<4;a+=2) for(int b=0;b<8;b+=2) {
    MM c00(full(0)),c01(full(0)),c10(full(0)),c11(full(0));
    for(int k=0;k<8;++k) {
      auto a0=aie::load_v<32>(q+a*256+k*32),a1=aie::load_v<32>(q+(a+1)*256+k*32);
      auto b0=aie::load_v<32>(kt+k*256+b*32),b1=aie::load_v<32>(kt+k*256+(b+1)*32);
      c00.mac(a0,b0); c01.mac(a0,b1); c10.mac(a1,b0); c11.mac(a1,b1);
    }
    score_tile(c00.to_vector<float>(),scores,a,b); score_tile(c01.to_vector<float>(),scores,a,b+1);
    score_tile(c10.to_vector<float>(),scores,a+1,b); score_tile(c11.to_vector<float>(),scores,a+1,b+1);
  }
  F maxima;
  for(int i=0;i<16;++i) {
    F s0=aie::load_v<16>(scores+i*32),s1=aie::load_v<16>(scores+i*32+16);
    auto z0=aie::eq(fp(aie::load_v<16>(mask+i*32)),full(0));
    auto z1=aie::eq(fp(aie::load_v<16>(mask+i*32+16)),full(0));
    s0=aie::select(s0,full(-3.402823466e38f),z0);
    s1=aie::select(s1,full(-3.402823466e38f),z1);
    aie::store_v(scores+i*32,s0); aie::store_v(scores+i*32+16,s1);
    maxima.set(aie::reduce_max(aie::max(s0,s1)),i);
  }
  F oldl=aie::load_v<16>(l);
  F newm=aie::max(aie::load_v<16>(m),maxima);
  F alpha=exp16(aie::sub(aie::load_v<16>(m),newm));
  alpha=aie::select(alpha,full(0),aie::eq(oldl,full(0)));
  for(int a=0;a<4;++a) {
    F scale=row_factors(alpha,a);
    for(int b=0;b<16;++b) aie::store_v(state+(a*16+b)*16,mul(aie::load_v<16>(state+(a*16+b)*16),scale));
  }
  F sums;
  for(int i=0;i<16;++i) {
    F mx=full(newm.get(i));
    auto e0=bf(exp16(aie::sub(aie::load_v<16>(scores+i*32),mx)));
    auto e1=bf(exp16(aie::sub(aie::load_v<16>(scores+i*32+16),mx)));
    auto z0=aie::eq(fp(aie::load_v<16>(mask+i*32)),full(0));
    auto z1=aie::eq(fp(aie::load_v<16>(mask+i*32+16)),full(0));
    e0=aie::select(e0,aie::zeros<bfloat16,16>(),z0);
    e1=aie::select(e1,aie::zeros<bfloat16,16>(),z1);
    aie::store_v(pr+i*32,e0); aie::store_v(pr+i*32+16,e1);
    sums.set(aie::reduce_add(aie::add(fp(e0),fp(e1))),i);
  }
  aie::store_v(m,newm); aie::store_v(l,aie::add(mul(oldl,alpha),sums));
  for(int a=0;a<4;++a) for(int k=0;k<4;++k) {
    auto block=aie::concat(aie::load_v<8>(pr+(a*4)*32+k*8),aie::load_v<8>(pr+(a*4+1)*32+k*8),
                           aie::load_v<8>(pr+(a*4+2)*32+k*8),aie::load_v<8>(pr+(a*4+3)*32+k*8));
    aie::store_v(p+(a*4+k)*32,block);
  }
  for(int a=0;a<4;a+=2) for(int b=0;b<16;b+=2) {
    MM c00(full(0)),c01(full(0)),c10(full(0)),c11(full(0));
    for(int k=0;k<4;++k) {
      auto a0=aie::load_v<32>(p+a*128+k*32),a1=aie::load_v<32>(p+(a+1)*128+k*32);
      auto b0=aie::load_v<32>(v+k*512+b*32),b1=aie::load_v<32>(v+k*512+(b+1)*32);
      c00.mac(a0,b0); c01.mac(a0,b1); c10.mac(a1,b0); c11.mac(a1,b1);
    }
    aie::store_v(state+(a*16+b)*16,aie::add(aie::load_v<16>(state+(a*16+b)*16),c00.to_vector<float>()));
    aie::store_v(state+(a*16+b+1)*16,aie::add(aie::load_v<16>(state+(a*16+b+1)*16),c01.to_vector<float>()));
    aie::store_v(state+((a+1)*16+b)*16,aie::add(aie::load_v<16>(state+((a+1)*16+b)*16),c10.to_vector<float>()));
    aie::store_v(state+((a+1)*16+b+1)*16,aie::add(aie::load_v<16>(state+((a+1)*16+b+1)*16),c11.to_vector<float>()));
  }
}
extern "C" void attention_finish(const float *state,float *out) {
  out+=16;
  F inv=reciprocal16(aie::load_v<16>(state+1040));
  for(int a=0;a<4;++a) {
    F scale=row_factors(inv,a);
    for(int b=0;b<16;++b) {
      F tile=mul(aie::load_v<16>(state+(a*16+b)*16),scale);
      for(int i=0;i<4;++i) aie::store_v(out+(a*4+i)*64+b*4,tile.extract<4>(i));
    }
  }
}
extern "C" void attention_empty(const int32_t *,float *,int32_t *) {}
extern "C" void attention_zero(const float *,float *out) {
  for(int i=16;i<1040;i+=16) aie::store_v(out+i,full(0));
}
