// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" void draft_projection(const int32_t *packet, int32_t *output, float *state) {
    using MM = aie::mmul<4,8,4,bfloat16,bfloat16,accfloat>;
    const auto *x = reinterpret_cast<const bfloat16 *>(packet+16);
    const auto *w = reinterpret_cast<const bfloat16 *>(packet+144);
    const bool reset = packet[1] != 0;
    MM c0(reset ? aie::zeros<float,16>() : aie::load_v<16>(state));
    MM c1(reset ? aie::zeros<float,16>() : aie::load_v<16>(state+16));
    MM c2(reset ? aie::zeros<float,16>() : aie::load_v<16>(state+32));
    MM c3(reset ? aie::zeros<float,16>() : aie::load_v<16>(state+48));
    for (int k=0; k<32; ++k) {
        auto row = aie::load_v<8>(x+k*8);
        aie::vector<bfloat16,32> a;
        a.insert<8>(0,row); a.insert<8>(1,row);
        a.insert<8>(2,row); a.insert<8>(3,row);
        c0.mac(a,aie::load_v<32>(w+k*128));
        c1.mac(a,aie::load_v<32>(w+k*128+32));
        c2.mac(a,aie::load_v<32>(w+k*128+64));
        c3.mac(a,aie::load_v<32>(w+k*128+96));
    }
    aie::store_v(state,c0.to_vector<float>());
    aie::store_v(state+16,c1.to_vector<float>());
    aie::store_v(state+32,c2.to_vector<float>());
    aie::store_v(state+48,c3.to_vector<float>());
    for (int i=0; i<48; i+=16)
        aie::store_v(output+i,aie::broadcast<int32,16>(packet[0]));
    aie::vector<float,16> value;
    value.insert<4>(0,c0.to_vector<float>().extract<4>(0));
    value.insert<4>(1,c1.to_vector<float>().extract<4>(0));
    value.insert<4>(2,c2.to_vector<float>().extract<4>(0));
    value.insert<4>(3,c3.to_vector<float>().extract<4>(0));
    auto bits = value.cast_to<uint32>();
    auto rounded = aie::add(bits,aie::add(aie::broadcast<uint32,16>(0x7fff),
        aie::bit_and(aie::downshift(bits,16),aie::broadcast<uint32,16>(1))));
    auto result = aie::filter_even(aie::downshift(rounded,16).cast_to<uint16>()).cast_to<bfloat16>();
    aie::store_v(reinterpret_cast<bfloat16 *>(output+16),result);
}
