// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <aie_api/aie.hpp>
#include <stdint.h>
using F = aie::vector<float,16>;
static inline F widen(aie::vector<bfloat16,16> x) {
    return aie::upshift(x.cast_to<uint16>().unpack(),16).cast_to<float>();
}
extern "C" void draft_projection_precise(const int32_t *packet,int32_t *output,float *state) {
    const auto *x=reinterpret_cast<const uint16_t *>(packet+16);
    const auto *w=reinterpret_cast<const bfloat16 *>(packet+144);
    F sum=packet[1] ? aie::zeros<float,16>() : aie::load_v<16>(state);
    F correction=packet[1] ? aie::zeros<float,16>() : aie::load_v<16>(state+16);
    // A BF16 product fits exactly in FP32. Compensate the accumulation before
    // the single BF16 output rounding; weights remain BF16 throughout storage.
    for(int k=0;k<256;++k) {
        F value=aie::broadcast<uint32,16>(uint32_t(x[k])<<16).cast_to<float>();
        F product=aie::mul(value,widen(aie::load_v<16>(w+k*16))).to_vector<float>();
        F corrected=aie::sub(product,correction);
        F next=aie::add(sum,corrected);
        correction=aie::sub(aie::sub(next,sum),corrected);
        sum=next;
    }
    aie::store_v(state,sum);aie::store_v(state+16,correction);
    for(int i=0;i<48;i+=16)
        aie::store_v(output+i,aie::broadcast<int32,16>(packet[0]));
    auto bits=sum.cast_to<uint32>();
    auto rounded=aie::add(bits,aie::add(aie::broadcast<uint32,16>(0x7fff),
        aie::bit_and(aie::downshift(bits,16),aie::broadcast<uint32,16>(1))));
    aie::store_v(reinterpret_cast<bfloat16 *>(output+16),
        aie::filter_even(aie::downshift(rounded,16).cast_to<uint16>()).cast_to<bfloat16>());
}
