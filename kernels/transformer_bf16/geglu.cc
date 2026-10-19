// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <aie_api/aie.hpp>
#include <stdint.h>
static inline uint16_t gelu_bits(uint16_t bits, const uint16_t *table) {
    const uint16_t magnitude = bits & 0x7fff, sign = bits & 0x8000;
    if (magnitude >= 0x4180) return sign ? 0x8000 : bits;
    if (magnitude >= 0x3980)
        return table[(sign ? 2048 : 0) + magnitude - 0x3980];
    const uint16_t half = magnitude >= 0x100 ? magnitude - 0x80
        : (magnitude >> 1) + ((magnitude & 3) == 3);
    return sign | half;
}
static inline void store_bf16(bfloat16 *p, aie::vector<float,16> x) {
    auto bits = x.cast_to<uint32>();
    auto rounded = aie::add(bits, aie::add(aie::broadcast<uint32,16>(0x7fff),
        aie::bit_and(aie::downshift(bits,16),aie::broadcast<uint32,16>(1))));
    aie::store_v(p,aie::filter_even(aie::downshift(rounded,16).cast_to<uint16>()).cast_to<bfloat16>());
}
#ifdef BF16_EXTERNAL_GELU_TABLE
extern "C" void draft_geglu_resident(const int32_t *packet, int32_t *output,const uint16_t *table) {
#else
extern "C" void draft_geglu(const int32_t *packet, int32_t *output) {
    const auto *table = reinterpret_cast<const uint16_t *>(packet + 2064);
#endif
    const auto *gate = reinterpret_cast<const uint16_t *>(packet + 16);
    const auto *up = reinterpret_cast<const bfloat16 *>(packet + 1040);
    auto *out = reinterpret_cast<bfloat16 *>(output + 16);
    for (int i = 0; i < 1056; i += 16)
        aie::store_v(output+i,aie::broadcast<int32,16>(packet[0]));
    for (int i = 0; i < 2048; i += 16) {
        alignas(32) uint16_t selected[16];
        for (int j = 0; j < 16; ++j) selected[j] = gelu_bits(gate[i+j],table);
        auto activation = aie::load_v<16>(selected).cast_to<bfloat16>();
        auto product = aie::mul(activation,aie::load_v<16>(up+i)).to_vector<float>();
        store_bf16(out+i,product);
    }
}
