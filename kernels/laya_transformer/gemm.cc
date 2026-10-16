// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Shared product: attention16x32 or GEMM32x64, K64 bf16 blocks.
#include <aie_api/aie.hpp>
#include <stdint.h>
using MM=aie::mmul<4,8,4,bfloat16,bfloat16,accfloat>;
using F=aie::vector<float,16>;
#ifdef ATTENTION_FP32_PROBS
static __attribute__((noinline)) void transformer_product(const bfloat16 *q,const bfloat16 *b,float *state,int reset,int rows,int cols,int kblocks) {
  const int qstride=kblocks*32;
#else
extern "C" void transformer_matmul(const bfloat16 *q,const bfloat16 *b,float *state,int reset,int rows,int cols) {
  const int kblocks=8,qstride=256;
#endif
  if(reset) for(int i=0;i<rows*cols*16;i+=16) aie::store_v(state+i,aie::zeros<float,16>());
  for(int a=0;a<rows;a+=2) for(int n=0;n<cols;n+=4) {
    MM c00(aie::load_v<16>(state+(a*cols+n)*16)),c01(aie::load_v<16>(state+(a*cols+n+1)*16));
    MM c02(aie::load_v<16>(state+(a*cols+n+2)*16)),c03(aie::load_v<16>(state+(a*cols+n+3)*16));
    MM c10(aie::load_v<16>(state+((a+1)*cols+n)*16)),c11(aie::load_v<16>(state+((a+1)*cols+n+1)*16));
    MM c12(aie::load_v<16>(state+((a+1)*cols+n+2)*16)),c13(aie::load_v<16>(state+((a+1)*cols+n+3)*16));
    for(int k=0;k<kblocks;++k) {
      auto a0=aie::load_v<32>(q+a*qstride+k*32),a1=aie::load_v<32>(q+(a+1)*qstride+k*32);
      auto b0=aie::load_v<32>(b+k*cols*32+n*32),b1=aie::load_v<32>(b+k*cols*32+(n+1)*32);
      auto b2=aie::load_v<32>(b+k*cols*32+(n+2)*32),b3=aie::load_v<32>(b+k*cols*32+(n+3)*32);
      c00.mac(a0,b0); c01.mac(a0,b1); c02.mac(a0,b2); c03.mac(a0,b3);
      c10.mac(a1,b0); c11.mac(a1,b1); c12.mac(a1,b2); c13.mac(a1,b3);
    }
    aie::store_v(state+(a*cols+n)*16,c00.to_vector<float>());
    aie::store_v(state+(a*cols+n+1)*16,c01.to_vector<float>());
    aie::store_v(state+(a*cols+n+2)*16,c02.to_vector<float>());
    aie::store_v(state+(a*cols+n+3)*16,c03.to_vector<float>());
    aie::store_v(state+((a+1)*cols+n)*16,c10.to_vector<float>());
    aie::store_v(state+((a+1)*cols+n+1)*16,c11.to_vector<float>());
    aie::store_v(state+((a+1)*cols+n+2)*16,c12.to_vector<float>());
    aie::store_v(state+((a+1)*cols+n+3)*16,c13.to_vector<float>());
  }
}
#ifdef ATTENTION_FP32_PROBS
extern "C" void transformer_matmul(const bfloat16 *q,const bfloat16 *b,float *state,int reset,int rows,int cols) {
  transformer_product(q,b,state,reset,rows,cols,8);
}
extern "C" void transformer_pv(const bfloat16 *q,const bfloat16 *v,float *state) {
  transformer_product(q,v,state,0,4,16,4);
}
#endif
extern "C" void gemm_step(const int32_t *packet,const int32_t *weights,float *state) {
  transformer_matmul(reinterpret_cast<const bfloat16 *>(packet+16),
    reinterpret_cast<const bfloat16 *>(weights),state,packet[6],8,16);
}
extern "C" void gemm_finish(const float *state,float *output) {
  output+=16;
  for(int a=0;a<8;++a) for(int b=0;b<16;++b) {
    F value=aie::load_v<16>(state+(a*16+b)*16);
    for(int r=0;r<4;++r) aie::store_v(output+(a*4+r)*64+b*4,value.extract<4>(r));
  }
}
