// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// One-core streamed attention. FP arithmetic stays on the vector datapath.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef ATTENTION_FP32_EXP
#include "lut_based_ops.h"
#endif
using F = aie::vector<float,16>;
using MM = aie::mmul<4,8,4,bfloat16,bfloat16,accfloat>;
#ifdef ATTENTION_SHARED_MATMUL
extern "C" void transformer_matmul(const bfloat16 *,const bfloat16 *,float *,int,int,int);
#endif
#ifdef ATTENTION_FP32_PROBS
extern "C" void transformer_pv(const bfloat16 *,const bfloat16 *,float *);
#endif
static inline __attribute__((always_inline)) F full(float x) { return aie::broadcast<float,16>(x); }
static inline __attribute__((always_inline)) F mul(F x,F y) { return aie::mul(x,y).to_vector<float>(); }
static inline __attribute__((always_inline)) F fp(aie::vector<bfloat16,16> x) { return aie::accum<accfloat,16>(x).to_vector<float>(); }
static inline __attribute__((always_inline)) aie::vector<bfloat16,16> bf(F x) { return aie::accum<accfloat,16>(x).to_vector<bfloat16>(); }
// Three bf16 products preserve both numerator and scale residuals in fp32.
// Dropping the low*low term contributes only the product of two bf16 errors.
static inline __attribute__((always_inline)) F rescale(F x,F y) {
  auto hi=bf(x),lo=bf(aie::sub(x,fp(hi))),factor=bf(y);
  auto residual=bf(aie::sub(y,fp(factor)));
  auto acc=aie::mac(aie::mul(hi,factor),lo,factor);
  return aie::mac(acc,hi,residual).to_vector<float>();
}
// Index the upstream exp tables from FP32 scores directly. Converting scores
// to bf16 before range reduction loses significant bits on peaked inputs.
#ifdef ATTENTION_FP32_EXP
#include "exp_fp32.h"
#else
static inline __attribute__((always_inline)) F exp16(F x) {
  auto tiny=aie::lt(x,full(-80.0f));
  x=aie::min(full(0),aie::max(full(-80.0f),x));
  using LT = aie::lut<4,bfloat16,bfloat16>;
  LT li(256,(bfloat16 __aie_dm_resource_a *)exp_ilut_ab,
            (bfloat16 __aie_dm_resource_b *)exp_ilut_cd);
  LT lf(256,(bfloat16 __aie_dm_resource_a *)exp_flut_ab,
            (bfloat16 __aie_dm_resource_b *)exp_flut_cd);
  aie::parallel_lookup<uint16,LT,aie::lut_oor_policy::truncate> il(li,8),fl(lf,0);
  auto index=aie::to_fixed<int32>(x,8).pack<int16>().cast_to<uint16>();
  F e=aie::mul(il.fetch(index),fl.fetch(index)).to_vector<float>();
  return aie::select(e,full(0),tiny);
}
#endif
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
  // Division by eight is an exponent adjustment for normal FP32 lanes.
  // Preserve zero and use the general path for subnormal or nonfinite lanes.
  auto u=s.cast_to<uint32>();
  auto magnitude=aie::bit_and(u,aie::broadcast<uint32,16>(0x7fffffff));
  auto normal=aie::ge(magnitude,aie::broadcast<uint32,16>(0x02000000))
             & aie::lt(magnitude,aie::broadcast<uint32,16>(0x7f800000));
  if(normal.full()) s=aie::sub(u,aie::broadcast<uint32,16>(3u<<23)).cast_to<float>();
  else if(!aie::eq(magnitude,aie::zeros<uint32,16>()).full()) s=mul(s,full(.125f));
  for(int i=0;i<4;++i) aie::store_v(scores+(a*4+i)*32+b*4,s.extract<4>(i));
}
// Unchanged input/output packet ABI. O is tiled [4,16,4,4]. Scratch is 4096 B:
// scores[16,32] f32, row probabilities[16,32] bf16, tiled probabilities[4,4,4,8].
static inline __attribute__((always_inline)) void attention_compute(
    const bfloat16 *q,const bfloat16 *kt,const bfloat16 *v,const bfloat16 *mask,
    float *state,int32_t *scratch,bool reset,bool unmasked=false) {
  float *scores=reinterpret_cast<float *>(scratch);
  bfloat16 *pr=reinterpret_cast<bfloat16 *>(scratch+512),*p=pr+512;
  float *m=state+1024,*l=m+16;
  if(reset) {
    for(int i=0;i<1024;i+=16) aie::store_v(state+i,full(0));
    aie::store_v(m,full(-3.402823466e38f)); aie::store_v(l,full(0));
  }
  // The persistent transformer program shares this product with its generic
  // GEMM command. Scratch's second half is free until probabilities are formed.
#ifdef ATTENTION_SHARED_MATMUL
  float *product=reinterpret_cast<float *>(scratch+512);
  transformer_matmul(q,kt,product,1,4,8);
  for(int a=0;a<4;++a) for(int b=0;b<8;++b)
    score_tile(aie::load_v<16>(product+(a*8+b)*16),scores,a,b);
#else
  // As in the bf16 convolution/GEMM kernels, keep independent accumulators
  // live and reuse each loaded operand across four products (2x4 expansion).
  for(int a=0;a<4;a+=2) for(int b=0;b<8;b+=4) {
    MM c00(full(0)),c01(full(0)),c02(full(0)),c03(full(0));
    MM c10(full(0)),c11(full(0)),c12(full(0)),c13(full(0));
#ifdef ATTENTION_UNROLL_2
#pragma clang loop unroll_count(2)
#endif
    for(int k=0;k<8;++k) {
      auto a0=aie::load_v<32>(q+a*256+k*32),a1=aie::load_v<32>(q+(a+1)*256+k*32);
      auto b0=aie::load_v<32>(kt+k*256+b*32),b1=aie::load_v<32>(kt+k*256+(b+1)*32);
      auto b2=aie::load_v<32>(kt+k*256+(b+2)*32),b3=aie::load_v<32>(kt+k*256+(b+3)*32);
      c00.mac(a0,b0); c01.mac(a0,b1); c02.mac(a0,b2); c03.mac(a0,b3);
      c10.mac(a1,b0); c11.mac(a1,b1); c12.mac(a1,b2); c13.mac(a1,b3);
    }
    score_tile(c00.to_vector<float>(),scores,a,b); score_tile(c01.to_vector<float>(),scores,a,b+1);
    score_tile(c02.to_vector<float>(),scores,a,b+2); score_tile(c03.to_vector<float>(),scores,a,b+3);
    score_tile(c10.to_vector<float>(),scores,a+1,b); score_tile(c11.to_vector<float>(),scores,a+1,b+1);
    score_tile(c12.to_vector<float>(),scores,a+1,b+2); score_tile(c13.to_vector<float>(),scores,a+1,b+3);
  }
#endif
  F maxima;
  for(int i=0;i<16;++i) {
    F s0=aie::load_v<16>(scores+i*32),s1=aie::load_v<16>(scores+i*32+16);
    if(!unmasked) {
    auto z0=aie::eq(aie::load_v<16>(mask+i*32).cast_to<int16>(),aie::zeros<int16,16>());
    auto z1=aie::eq(aie::load_v<16>(mask+i*32+16).cast_to<int16>(),aie::zeros<int16,16>());
    s0=aie::select(s0,full(-3.402823466e38f),z0);
    s1=aie::select(s1,full(-3.402823466e38f),z1);
    }
    aie::store_v(scores+i*32,s0); aie::store_v(scores+i*32+16,s1);
    maxima.set(aie::reduce_max(aie::max(s0,s1)),i);
  }
  F oldl=aie::load_v<16>(l);
  F newm=aie::max(aie::load_v<16>(m),maxima);
  F alpha=exp16(aie::sub(aie::load_v<16>(m),newm));
  alpha=aie::select(alpha,full(0),aie::eq(oldl,full(0)));
  for(int a=0;a<4;++a) {
    F scale=row_factors(alpha,a);
    if(!aie::eq(scale,full(1)).full()) {
      if(aie::eq(scale,full(0)).full()) {
        for(int b=0;b<16;++b) aie::store_v(state+(a*16+b)*16,full(0));
      } else {
        for(int b=0;b<16;++b) aie::store_v(state+(a*16+b)*16,rescale(aie::load_v<16>(state+(a*16+b)*16),scale));
      }
    }
  }
  F sums;
  for(int i=0;i<16;++i) {
    F mx=full(newm.get(i));
#ifdef ATTENTION_FP32_PROBS
    F e0=exp16(aie::sub(aie::load_v<16>(scores+i*32),mx));
    F e1=exp16(aie::sub(aie::load_v<16>(scores+i*32+16),mx));
#else
    auto e0=bf(exp16(aie::sub(aie::load_v<16>(scores+i*32),mx)));
    auto e1=bf(exp16(aie::sub(aie::load_v<16>(scores+i*32+16),mx)));
#endif
    if(!unmasked) {
    auto z0=aie::eq(aie::load_v<16>(mask+i*32).cast_to<int16>(),aie::zeros<int16,16>());
    auto z1=aie::eq(aie::load_v<16>(mask+i*32+16).cast_to<int16>(),aie::zeros<int16,16>());
#ifdef ATTENTION_FP32_PROBS
    e0=aie::select(e0,full(0),z0); e1=aie::select(e1,full(0),z1);
#else
    e0=aie::select(e0,aie::zeros<bfloat16,16>(),z0);
    e1=aie::select(e1,aie::zeros<bfloat16,16>(),z1);
#endif
    }
#ifdef ATTENTION_FP32_PROBS
    auto hi0=bf(e0),hi1=bf(e1);
    aie::store_v(pr+i*32,hi0); aie::store_v(pr+i*32+16,hi1);
    // Compact low parts into the already-consumed first half of scores.
    // A row's two score vectors have both been loaded before either store.
    auto low=reinterpret_cast<bfloat16 *>(scores);
    aie::store_v(low+i*32,bf(aie::sub(e0,fp(hi0))));
    aie::store_v(low+i*32+16,bf(aie::sub(e1,fp(hi1))));
    sums.set(aie::reduce_add(aie::add(e0,e1)),i);
#else
    aie::store_v(pr+i*32,e0); aie::store_v(pr+i*32+16,e1);
    sums.set(aie::reduce_add(aie::add(fp(e0),fp(e1))),i);
#endif
  }
  aie::store_v(m,newm); aie::store_v(l,aie::add(mul(oldl,alpha),sums));
#ifdef ATTENTION_FP32_PROBS
  for(int part=0;part<2;++part) {
    const bfloat16 *row=part ? reinterpret_cast<const bfloat16 *>(scores) : pr;
    for(int a=0;a<4;++a) for(int k=0;k<4;++k) {
      auto block=aie::concat(aie::load_v<8>(row+(a*4)*32+k*8),aie::load_v<8>(row+(a*4+1)*32+k*8),
                            aie::load_v<8>(row+(a*4+2)*32+k*8),aie::load_v<8>(row+(a*4+3)*32+k*8));
      aie::store_v(p+(a*4+k)*32,block);
    }
    transformer_pv(p,v,state);
  }
#else
  for(int a=0;a<4;++a) for(int k=0;k<4;++k) {
    auto block=aie::concat(aie::load_v<8>(pr+(a*4)*32+k*8),aie::load_v<8>(pr+(a*4+1)*32+k*8),
                           aie::load_v<8>(pr+(a*4+2)*32+k*8),aie::load_v<8>(pr+(a*4+3)*32+k*8));
    aie::store_v(p+(a*4+k)*32,block);
  }
  for(int a=0;a<4;a+=2) for(int b=0;b<16;b+=4) {
    MM c00(aie::load_v<16>(state+((a+0)*16+b+0)*16)),c01(aie::load_v<16>(state+((a+0)*16+b+1)*16)),c02(aie::load_v<16>(state+((a+0)*16+b+2)*16)),c03(aie::load_v<16>(state+((a+0)*16+b+3)*16));
    MM c10(aie::load_v<16>(state+((a+1)*16+b+0)*16)),c11(aie::load_v<16>(state+((a+1)*16+b+1)*16)),c12(aie::load_v<16>(state+((a+1)*16+b+2)*16)),c13(aie::load_v<16>(state+((a+1)*16+b+3)*16));
#ifdef ATTENTION_UNROLL_2
#pragma clang loop unroll_count(2)
#endif
    for(int k=0;k<4;++k) {
      auto a0=aie::load_v<32>(p+a*128+k*32),a1=aie::load_v<32>(p+(a+1)*128+k*32);
      auto b0=aie::load_v<32>(v+k*512+(b+0)*32);
      auto b1=aie::load_v<32>(v+k*512+(b+1)*32);
      auto b2=aie::load_v<32>(v+k*512+(b+2)*32);
      auto b3=aie::load_v<32>(v+k*512+(b+3)*32);
      c00.mac(a0,b0); c01.mac(a0,b1); c02.mac(a0,b2); c03.mac(a0,b3);
      c10.mac(a1,b0); c11.mac(a1,b1); c12.mac(a1,b2); c13.mac(a1,b3);
    }
    aie::store_v(state+((a+0)*16+b+0)*16,c00.to_vector<float>());
    aie::store_v(state+((a+0)*16+b+1)*16,c01.to_vector<float>());
    aie::store_v(state+((a+0)*16+b+2)*16,c02.to_vector<float>());
    aie::store_v(state+((a+0)*16+b+3)*16,c03.to_vector<float>());
    aie::store_v(state+((a+1)*16+b+0)*16,c10.to_vector<float>());
    aie::store_v(state+((a+1)*16+b+1)*16,c11.to_vector<float>());
    aie::store_v(state+((a+1)*16+b+2)*16,c12.to_vector<float>());
    aie::store_v(state+((a+1)*16+b+3)*16,c13.to_vector<float>());
  }
#endif
}
extern "C" void attention_block(const int32_t *packet,float *state,int32_t *scratch) {
  const bfloat16 *q=reinterpret_cast<const bfloat16 *>(packet+16);
  const bfloat16 *kt=q+1024,*v=kt+2048,*mask=v+2048;
  attention_compute(q,kt,v,mask,state,scratch,packet[0]);
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
