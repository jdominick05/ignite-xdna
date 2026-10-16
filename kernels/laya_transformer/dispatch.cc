// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" void residual_norm(const int32_t *,const float *,int32_t *);
extern "C" void command_norm(const int32_t *q,const int32_t *affine,int32_t *out) {
#ifndef TRANSFORMER_NORM_TILED
  if(q[5]) {
    // Diagnostic control: same traffic and linked image, no norm arithmetic.
    for(int i=0;i<2048;i+=16) aie::store_v(out+16+i,aie::load_v<16>(q+16+i));
    for(int i=0;i<16;++i) { out[i]=q[0]; out[2064+i]=q[0]; }
    return;
  }
#endif
  residual_norm(q,reinterpret_cast<const float *>(affine),out);
  // Peano elides a get_rounding()/set_rounding(saved) pair across this
  // external call. The norm kernel leaves crRnd=12. Attention's float-to-
  // fixed LUT indexing requires the original floor mode (crRnd=0).
  aie::set_rounding(aie::rounding_mode::floor);
  // Keep a rounding-sensitive operation live: an otherwise unused mode
  // change is also eliminated by this compiler. This volatile stack store
  // is a control-state fence; it does not supply a neural-network output.
  auto rounded=aie::to_fixed<int32>(aie::load_v<16>(reinterpret_cast<const float *>(q+16)),0);
  volatile int32_t mode_fence=rounded.get(0);
}
