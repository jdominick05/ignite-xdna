// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <aie_api/aie.hpp>
#include <stdint.h>
using F=aie::vector<float,16>;
static inline F full(float x) { return aie::broadcast<float,16>(x); }
static inline F mul(F x,F y) { return aie::mul(x,y).to_vector<float>(); }
static inline F widen(aie::vector<bfloat16,16> x) {
    return aie::upshift(x.cast_to<uint16>().unpack(),16).cast_to<float>();
}
static inline aie::vector<bfloat16,16> rounded(F x) {
    auto u=x.cast_to<uint32>();
    auto r=aie::add(u,aie::add(aie::broadcast<uint32,16>(0x7fff),
        aie::bit_and(aie::downshift(u,16),aie::broadcast<uint32,16>(1))));
    return aie::filter_even(aie::downshift(r,16).cast_to<uint16>()).cast_to<bfloat16>();
}
static inline F reciprocal(F x) {
    F y=aie::sub(aie::broadcast<int32,16>(0x7ef311c3),x.cast_to<int32>()).cast_to<float>();
    for(int i=0;i<4;++i) y=mul(y,aie::sub(full(2),mul(x,y)));
    return y;
}
#ifndef BF16_EXP_ATTRIBUTES
#define BF16_EXP_ATTRIBUTES inline
#endif
static BF16_EXP_ATTRIBUTES F exp_negative(F x) {
    auto tiny=aie::lt(x,full(-80));
    x=aie::max(full(-80),x);
    auto n=aie::to_fixed<int32>(mul(x,full(1.4426950408889634f)));
    F r=aie::sub(x,mul(aie::to_float(n),full(.6931471805599453f)));
    F p=full(1.0f/40320.0f);
    p=aie::add(full(1.0f/5040.0f),mul(r,p));
    p=aie::add(full(1.0f/720.0f),mul(r,p));
    p=aie::add(full(1.0f/120.0f),mul(r,p));
    p=aie::add(full(1.0f/24.0f),mul(r,p));
    p=aie::add(full(1.0f/6.0f),mul(r,p));
    p=aie::add(full(.5f),mul(r,p));
    p=aie::add(full(1),mul(r,p));
    p=aie::add(full(1),mul(r,p));
    auto power=aie::upshift(aie::add(n,aie::broadcast<int32,16>(127)),23).cast_to<float>();
    return aie::select(mul(p,power),full(0),tiny);
}
extern "C" void draft_softmax(const int32_t *packet,int32_t *output,float *state) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int reset=packet[1], emit=packet[2], valid=packet[3];
    const auto *x=reinterpret_cast<const bfloat16 *>(packet+16);
    auto *y=reinterpret_cast<bfloat16 *>(output+16);
    alignas(64) const int32_t lane_ids[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    auto lanes=aie::load_v<16>(lane_ids);
    F maximum=reset ? full(-3.402823466e38f) : aie::load_v<16>(state);
    F total=reset ? full(0) : aie::load_v<16>(state+16);
    for(int i=0;i<288;i+=16) aie::store_v(output+i,aie::broadcast<int32,16>(packet[0]));
    if(!emit) {
        F block_max=full(-3.402823466e38f);
        for(int i=0;i<512;i+=16) {
            auto keep=aie::lt(aie::add(lanes,aie::broadcast<int32,16>(i)),aie::broadcast<int32,16>(valid));
            F value=aie::select(full(-3.402823466e38f),widen(aie::load_v<16>(x+i)),keep);
            block_max=aie::max(block_max,value);
        }
        F next=aie::max(maximum,full(aie::reduce_max(block_max)));
        F block_sum=full(0);
        for(int i=0;i<512;i+=16) {
            auto keep=aie::lt(aie::add(lanes,aie::broadcast<int32,16>(i)),aie::broadcast<int32,16>(valid));
            F value=exp_negative(aie::sub(widen(aie::load_v<16>(x+i)),next));
            block_sum=aie::add(block_sum,aie::select(full(0),value,keep));
        }
        total=aie::add(mul(total,exp_negative(aie::sub(maximum,next))),full(aie::reduce_add(block_sum)));
        aie::store_v(state,next);aie::store_v(state+16,total);
    } else {
        F inverse=reciprocal(total);
        for(int i=0;i<512;i+=16) {
            auto keep=aie::lt(aie::add(lanes,aie::broadcast<int32,16>(i)),aie::broadcast<int32,16>(valid));
            F p=mul(exp_negative(aie::sub(widen(aie::load_v<16>(x+i)),maximum)),inverse);
            aie::store_v(y+i,rounded(aie::select(full(0),p,keep)));
        }
    }
}
