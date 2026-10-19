// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <aie_api/aie.hpp>
#include <stdint.h>
using B=aie::vector<bfloat16,16>;
using F=aie::vector<float,16>;
#ifndef BF16_HELPER_INLINE
#define BF16_HELPER_INLINE inline
#endif
static BF16_HELPER_INLINE F widen(B x) {
    return aie::upshift(x.cast_to<uint16>().unpack(),16).cast_to<float>();
}
static BF16_HELPER_INLINE B rounded(F x) {
    auto bits=x.cast_to<uint32>();
    auto r=aie::add(bits,aie::add(aie::broadcast<uint32,16>(0x7fff),
        aie::bit_and(aie::downshift(bits,16),aie::broadcast<uint32,16>(1))));
    return aie::filter_even(aie::downshift(r,16).cast_to<uint16>()).cast_to<bfloat16>();
}
extern "C" void draft_residual(const int32_t *packet,int32_t *output) {
    const int count=packet[1];
    const auto *x=reinterpret_cast<const bfloat16 *>(packet+16);
    const auto *residual=reinterpret_cast<const bfloat16 *>(packet+1040);
    B scale=aie::broadcast<uint16,16>(uint16_t(packet[2])).cast_to<bfloat16>();
    auto *y=reinterpret_cast<bfloat16 *>(output+16);
    for(int i=0;i<1056;i+=16)
        aie::store_v(output+i,aie::broadcast<int32,16>(packet[0]));
    for(int i=0;i<count;i+=16) {
        B a=aie::load_v<16>(x+i),b=aie::load_v<16>(residual+i);
        B sum=rounded(aie::add(widen(a),widen(b)));
        aie::store_v(y+i,rounded(aie::mul(sum,scale).to_vector<float>()));
    }
}
