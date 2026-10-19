// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <aie_api/aie.hpp>
#include <stdint.h>
using B = aie::vector<bfloat16,16>;
using F = aie::vector<float,16>;
#ifndef BF16_HELPER_INLINE
#define BF16_HELPER_INLINE inline
#endif
static BF16_HELPER_INLINE F widen(B x) {
    return aie::upshift(x.cast_to<uint16>().unpack(),16).cast_to<float>();
}
static BF16_HELPER_INLINE B rounded(F x) {
    auto u=x.cast_to<uint32>();
    auto r=aie::add(u,aie::add(aie::broadcast<uint32,16>(0x7fff),
        aie::bit_and(aie::downshift(u,16),aie::broadcast<uint32,16>(1))));
    return aie::filter_even(aie::downshift(r,16).cast_to<uint16>()).cast_to<bfloat16>();
}
extern "C" void draft_rope(const int32_t *packet,int32_t *output) {
    const int width=packet[1], heads=packet[2];
    const auto *q=reinterpret_cast<const bfloat16 *>(packet+16);
    const auto *cosine=reinterpret_cast<const bfloat16 *>(packet+1040);
    const auto *sine=reinterpret_cast<const bfloat16 *>(packet+1296);
    auto *y=reinterpret_cast<bfloat16 *>(output+16);
    for (int i=0; i<1056; i+=16)
        aie::store_v(output+i,aie::broadcast<int32,16>(packet[0]));
    for (int head=0; head<heads; ++head) for (int i=0; i<width; i+=16) {
        auto x=aie::load_v<16>(q+head*width+i);
        B rotated=aie::load_v<16>(q+head*width+(i<width/2 ? i+width/2 : i-width/2));
        if (i<width/2)
            rotated=aie::bit_xor(rotated.cast_to<uint16>(),aie::broadcast<uint16,16>(0x8000)).cast_to<bfloat16>();
        // Preserve both intermediate BF16 roundings used by Transformers.
        B a=rounded(aie::mul(x,aie::load_v<16>(cosine+i)).to_vector<float>());
        B b=rounded(aie::mul(rotated,aie::load_v<16>(sine+i)).to_vector<float>());
        aie::store_v(y+head*width+i,rounded(aie::add(widen(a),widen(b))));
    }
}
