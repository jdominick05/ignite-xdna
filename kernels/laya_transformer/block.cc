// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Output ABI shared with the attention image; this image retains GeGLU.
#include <aie_api/aie.hpp>
extern "C" void chain_finish(const float *state,float *out,int tiled) {
  int stride=tiled ? 65 : 64;
  if(!tiled) out+=16;
  for(int a=0;a<8;++a) for(int b=0;b<16;++b) {
    auto value=aie::load_v<16>(state+(a*16+b)*16);
    for(int row=0;row<4;++row) aie::store_unaligned_v(out+(a*4+row)*stride+b*4,value.extract<4>(row));
  }
}
