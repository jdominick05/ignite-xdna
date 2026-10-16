// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "kernel_lut.cc"
#ifndef ATTENTION_BYPASS_MASK
#define ATTENTION_BYPASS_MASK 0
#endif

// Q stays resident for all sixteen key blocks. K/V is multicast to four
// independent query tiles. Mask construction and all attention work are on NPU.
extern "C" void attention_shared(const int32_t *qp,const int32_t *kv,
                                float *state,int32_t *scratch,bfloat16 *mask,int key_block) {
  const auto *q=reinterpret_cast<const bfloat16 *>(qp+16);
  const auto *kt=reinterpret_cast<const bfloat16 *>(kv);
  const auto *v=kt+2048;
  const int query=qp[0],valid=qp[1],local=qp[3],key=key_block*32;
  if(key_block==0) {
    for(int i=0;i<1024;i+=16) aie::store_v(state+i,full(0));
    aie::store_v(state+1024,full(-3.402823466e38f));
    aie::store_v(state+1040,full(0));
  }
  // Padded query rows with no allowed keys have uniform source semantics;
  // their pre-zeroed Q must still consume every V block.
  bool uniform=valid==0 || (local && query+15-64>=valid);
  if(!uniform && local && (key>query+15+64 || key+31<query-64)) return;
  if(!uniform && key>=valid) return;
  const bool unmasked=ATTENTION_BYPASS_MASK && !local && valid==512;
  if(!unmasked) {
  auto lane=aie::load_v<32>((const int16_t *)(kv+2048));
  // The last 64 B hold a constant 0..31 ramp; no per-block host mask packet.
  auto positions=aie::add(lane,(int16_t)key);
  auto ones=aie::broadcast<int16,32>(0x3f80),zeros=aie::zeros<int16,32>();
  for(int i=0;i<16;++i) {
    bool row_uniform=valid==0 || (local && query+i-64>=valid);
    auto allowed=aie::lt(positions,(int16_t)valid);
    if(local) allowed=allowed & aie::ge(positions,(int16_t)(query+i-64))
                               & aie::le(positions,(int16_t)(query+i+64));
    auto values=row_uniform ? ones : aie::select(zeros,ones,allowed);
    aie::store_v(mask+i*32,values.cast_to<bfloat16>());
  }
  }
  attention_compute(q,kt,v,mask,state,scratch,false,unmasked);
}
