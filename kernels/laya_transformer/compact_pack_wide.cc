// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// One complete KV object is FP32[32,66]; each row contributes64 features.
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" void norm_pack_compact_step(const int32_t *kv,int32_t *state,int packet,int core_row,int feature) {
  if(packet!=core_row) return;
  const float *input=reinterpret_cast<const float *>(kv);
  bfloat16 *packed=reinterpret_cast<bfloat16 *>(state);
  for(int row=0;row<32;++row) for(int col=0;col<64;col+=16) {
    auto value=aie::load_unaligned_v<16>(input+row*66+col);
    auto bits=value.cast_to<uint32>();
    auto rounded=aie::add(bits,aie::add(aie::broadcast<uint32,16>(0x7fff),aie::bit_and(aie::downshift(bits,16),aie::broadcast<uint32,16>(1))));
    auto result=aie::filter_even(aie::downshift(rounded,16).cast_to<uint16>()).cast_to<bfloat16>();
    int offset=(row/4)*256+(col/8)*32+(row%4)*8;
    aie::store_v(packed+offset,result.extract<8>(0));
    aie::store_v(packed+offset+32,result.extract<8>(1));
  }
}
