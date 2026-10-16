// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <aie_api/aie.hpp>
#include <stdint.h>
// Q begins at the selected FP32 embedding row, without a command header.
// Preserve E as the residual. The next norm consumes tiled X=0 and R=E.
extern "C" void stem_copy(const int32_t *q,int32_t *out,int poison) {
  for(int i=0;i<2080;i+=16) aie::store_v(out+i,aie::broadcast<int32,16>(poison));
  for(int i=0;i<1040;i+=16) aie::store_v(out+i,aie::zeros<int32,16>());
  for(int feature=0;feature<16;++feature) out[feature*65+64]=poison;
  for(int i=0;i<1024;i+=16) aie::store_v(out+1040+i,aie::load_v<16>(q+i));
}
extern "C" void command_norm(const int32_t *,const int32_t *,int32_t *);
extern "C" void stem_norm(const int32_t *q,const int32_t *affine,int32_t *out) {
  command_norm(q,affine,out);
  // Embedding LayerNorm has no residual addition. Keep the exact original
  // E bits, including negative zero, for the following encoder residual.
  for(int i=1040;i<2064;i+=16) aie::store_v(out+i,aie::load_v<16>(q+i));
}
