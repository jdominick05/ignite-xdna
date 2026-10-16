// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Rotary and Q/K/V conversion between the persistent GEMM and attention ABIs.
#include <aie_api/aie.hpp>
#include <stdint.h>
using F=aie::vector<float,16>;
static inline F mul(F x,F y) { return aie::mul(x,y).to_vector<float>(); }
static inline aie::vector<bfloat16,16> rounded(F x) {
  auto bits=x.cast_to<uint32>();
  auto u=aie::add(bits,aie::add(aie::broadcast<uint32,16>(0x7fff),aie::bit_and(aie::downshift(bits,16),aie::broadcast<uint32,16>(1))));
  return aie::filter_even(aie::downshift(u,16).cast_to<uint16>()).cast_to<bfloat16>();
}
extern "C" void transformer_matmul(const bfloat16 *,const bfloat16 *,float *,int,int,int);
extern "C" void chain_gemm_step(const int32_t *q,const int32_t *b,float *state,int reset) {
  transformer_matmul(reinterpret_cast<const bfloat16 *>(q+16),reinterpret_cast<const bfloat16 *>(b),state,reset,8,16);
}
extern "C" void chain_finish(const float *state,float *out,int tiled) {
  int stride=tiled ? 65 : 64;
  if(!tiled) out+=16;
  // 32 rows of 64 channels plus a guard fill the unchanged 2080-word
  // output packet. Shim DMA scatters rows into norm's feature-tile layout.
  for(int a=0;a<8;++a) for(int b=0;b<16;++b) {
    F value=aie::load_v<16>(state+(a*16+b)*16);
    for(int row=0;row<4;++row) aie::store_unaligned_v(out+(a*4+row)*stride+b*4,value.extract<4>(row));
  }
}
extern "C" void norm_pack_step(const int32_t *kv,int32_t *state,int packet,int core_row,int feature) {
  int row=packet-core_row*32;
  if(row<0 || row>=32) return;
  const float *input=reinterpret_cast<const float *>(kv)+feature*64;
  bfloat16 *packed=reinterpret_cast<bfloat16 *>(state);
  for(int col=0;col<64;col+=16) {
    auto value=rounded(aie::load_v<16>(input+col));
    int offset=(row/4)*256+(col/8)*32+(row%4)*8;
    aie::store_v(packed+offset,value.extract<8>(0));
    aie::store_v(packed+offset+32,value.extract<8>(1));
  }
}
extern "C" void norm_pack_finish(const int32_t *state,int32_t *out,int poison) {
  for(int i=0;i<2080;i+=16) aie::store_v(out+i,aie::broadcast<int32,16>(poison));
  for(int i=0;i<1024;i+=16) aie::store_v(out+16+i,aie::load_v<16>(state+i));
}
extern "C" void chain_elementwise(const int32_t *q,const int32_t *kv,int32_t *out,int poison,
                                  int mode,int position,int valid,int local,int32_t *state) {
  for(int i=0;i<2080;i+=16) aie::store_v(out+i,aie::broadcast<int32,16>(poison));
  bfloat16 *tmp=reinterpret_cast<bfloat16 *>(state);
  const float *x=reinterpret_cast<const float *>(q+16),*coef=reinterpret_cast<const float *>(kv);
  int half=mode==5 ? 1 : 0;
  int begin=mode<6 ? half*16 : 0,end=mode<6 ? begin+16 : 32;
  // Mode 7 merges already-transposed K with unrotated V. Other modes rotate
  // FP32 projections, then perform the required bf16 operand conversion.
  for(int row=begin;row<end;++row) for(int col=0;col<32;col+=16) {
    if(mode==7 || mode==8) {
      // Mode 8 joins two attention outputs, each FP32[16,64], into
      // one bf16 GEMM operand tile[32,64] entirely on the device.
      const float *v=reinterpret_cast<const float *>((mode==8 && row<16 ? q : kv)+16);
      int source_row=mode==8 && row>=16 ? row-16 : row;
      aie::store_v(tmp+row*64+col,rounded(aie::load_v<16>(v+source_row*64+col)));
      aie::store_v(tmp+row*64+32+col,rounded(aie::load_v<16>(v+source_row*64+32+col)));
    } else {
      F lo=aie::load_v<16>(x+row*64+col),hi=aie::load_v<16>(x+row*64+32+col);
      F c=aie::load_v<16>(coef+row*32+col),s=aie::load_v<16>(coef+1024+row*32+col);
      F a=aie::sub(mul(lo,c),mul(hi,s)),b=aie::add(mul(hi,c),mul(lo,s));
      if(mode<6 && (valid==0 || (local && position+row-64>=valid))) { a=aie::zeros<float,16>(); b=a; }
      aie::store_v(tmp+row*64+col,rounded(a));
      aie::store_v(tmp+row*64+32+col,rounded(b));
    }
  }
  const uint16_t *bits=reinterpret_cast<const uint16_t *>(tmp);
  uint16_t *dst=reinterpret_cast<uint16_t *>(out+16);
  if(mode<6 || mode==8) {
    // Q[16,64], tiled 4x8, with an attention-owned in-band header.
    if(mode<6) { out[0]=position+begin; out[1]=valid; out[2]=poison; out[3]=local; out[4]=0; }
    int row_blocks=mode==8 ? 8 : 4;
    for(int a=0;a<row_blocks;++a) for(int b=0;b<8;++b) for(int r=0;r<4;++r) for(int c=0;c<8;c+=2) {
      int source=(begin+a*4+r)*64+b*8+c;
      reinterpret_cast<uint32_t *>(dst)[((a*8+b)*32+r*8+c)/2]=uint32_t(bits[source])|(uint32_t(bits[source+1])<<16);
    }
  } else if(mode==6) {
    // K transpose[64,32], tiled 8x4.
    for(int a=0;a<8;++a) for(int b=0;b<8;++b) for(int r=0;r<8;++r) for(int c=0;c<4;c+=2) {
      int feature=a*8+r,token=b*4+c;
      reinterpret_cast<uint32_t *>(dst)[((a*8+b)*32+r*4+c)/2]=uint32_t(bits[token*64+feature])|(uint32_t(bits[(token+1)*64+feature])<<16);
    }
  } else {
    for(int i=0;i<1024;i+=16) aie::store_v(out+16+i,aie::load_v<16>(q+16+i));
    // V[32,64], tiled 8x4, follows transposed K in the KV packet.
    for(int a=0;a<4;++a) for(int b=0;b<16;++b) for(int r=0;r<8;++r) for(int c=0;c<4;c+=2) {
      int source=(a*8+r)*64+b*4+c;
      reinterpret_cast<uint32_t *>(dst)[1024+((a*16+b)*32+r*4+c)/2]=uint32_t(bits[source])|(uint32_t(bits[source+1])<<16);
    }
    for(int i=0;i<32;i+=2) out[2064+i/2]=i|((i+1)<<16);
  }
}
