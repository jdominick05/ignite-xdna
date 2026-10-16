// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Reuse the qualified exact adder, GELU and packing implementations without
// modifying their default programs. Unused entry points are discarded at link.
#define residual_norm vector_unused_norm
#include "../laya_norm/kernel.cc"
#undef residual_norm
namespace activation_impl {
#include "mlp.cc"
}
namespace packing_impl {
#define chain_gemm_step vector_unused_rotary_gemm
#define chain_finish vector_unused_rotary_finish
#include "rotary.cc"
#undef chain_gemm_step
#undef chain_finish
}

static inline aie::vector<bfloat16,16> rounded(F x) {
  auto u=x.cast_to<uint32>();
  auto value=aie::add(u,aie::add(aie::broadcast<uint32,16>(0x7fff),aie::bit_and(aie::downshift(u,16),aie::broadcast<uint32,16>(1))));
  return aie::filter_even(aie::downshift(value,16).cast_to<uint16>()).cast_to<bfloat16>();
}
extern "C" void vector_bias(const int32_t *q,const int32_t *kv,int32_t *out,int poison,int mode) {
  for(int i=0;i<2080;i+=16) aie::store_v(out+i,aie::broadcast<int32,16>(poison));
  const float *input=reinterpret_cast<const float *>(q+16),*bias=reinterpret_cast<const float *>(kv);
  for(int row=0;row<32;++row) for(int col=0;col<64;col+=16) {
    F value=residual_add(aie::load_v<16>(input+row*64+col),aie::load_v<16>(bias+col));
    if(mode==13) value=aie::max(full(0),value);
    if(mode==14) value=activation_impl::gelu(value);
    if(mode<15) {
      auto bits=rounded(value); auto packed=reinterpret_cast<bfloat16 *>(out+16);
      int offset=(row/4)*256+(col/8)*32+(row%4)*8;
      aie::store_v(packed+offset,bits.extract<8>(0)); aie::store_v(packed+offset+32,bits.extract<8>(1));
    } else if(mode==16) {
      aie::store_unaligned_v(reinterpret_cast<float *>(out)+row*65+col,value);
    } else aie::store_v(reinterpret_cast<float *>(out+16)+row*64+col,value);
  }
}
extern "C" void vector_seed(const int32_t *q,const int32_t *kv,int32_t *out,int poison) {
  for(int i=0;i<2080;i+=16) aie::store_v(out+i,aie::broadcast<int32,16>(poison));
  for(int i=0;i<1024;i+=16) {
    aie::store_unaligned_v(out+i+i/64,aie::load_v<16>(q+i));
    aie::store_v(out+1040+i,aie::load_v<16>(kv+i));
  }
}
extern "C" void vector_sum(const int32_t *q,int32_t *out,int poison) {
  for(int i=0;i<2080;i+=16) aie::store_v(out+i,aie::broadcast<int32,16>(poison));
  const float *input=reinterpret_cast<const float *>(q);
  for(int i=0;i<1024;i+=16) {
    F value=residual_add(aie::load_unaligned_v<16>(input+i+i/64),aie::load_v<16>(input+1040+i));
    aie::store_v(reinterpret_cast<float *>(out)+i,value);
    aie::store_v(reinterpret_cast<float *>(out)+1040+i,value);
  }
}
extern "C" void vector_carry(const int32_t *q,int32_t *out,int poison) {
  // The first encoder Add consumes normalized embeddings (graph node 93).
  for(int i=0;i<2080;i+=16) aie::store_v(out+i,aie::broadcast<int32,16>(poison));
  for(int i=0;i<1024;i+=16) {
    auto value=aie::load_v<16>(q+i);
    aie::store_v(out+i,value); aie::store_v(out+1040+i,value);
  }
}
