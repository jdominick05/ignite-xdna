// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <aie_api/aie.hpp>
#include <stdint.h>
using F = aie::vector<float, 16>;
static inline F full(float x) { return aie::broadcast<float, 16>(x); }
static inline F mul(F a, F b) { return aie::mul(a, b).to_vector<float>(); }
static inline F load_bf16(const bfloat16 *p) {
    auto bits = aie::load_v<16>(p).cast_to<uint16>().unpack();
    return aie::upshift(bits, 16).cast_to<float>();
}
static inline void store_bf16(bfloat16 *p, F x) {
    auto bits = x.cast_to<uint32>();
    auto rounded = aie::add(bits, aie::add(aie::broadcast<uint32,16>(0x7fff),
        aie::bit_and(aie::downshift(bits,16), aie::broadcast<uint32,16>(1))));
    aie::store_v(p, aie::filter_even(aie::downshift(rounded,16).cast_to<uint16>()).cast_to<bfloat16>());
}
extern "C" void draft_norm(const int32_t *packet, int32_t *output) {
    const int width = packet[1], rows = packet[2];
    const auto *x = reinterpret_cast<const bfloat16 *>(packet + 16);
    const auto *weight = reinterpret_cast<const bfloat16 *>(packet + 1040);
    auto *y = reinterpret_cast<bfloat16 *>(output + 16);
    for (int i = 0; i < 1056; i += 16)
        aie::store_v(output + i, aie::broadcast<int32,16>(packet[0]));
    const float inverse_width = reinterpret_cast<const float *>(packet)[3];
    for (int row = 0; row < rows; ++row) {
        F squares = full(0);
        for (int i = 0; i < width; i += 16) {
            F value = load_bf16(x + row*width + i);
            squares = aie::add(squares, mul(value,value));
        }
        F variance = aie::add(mul(full(aie::reduce_add(squares)),full(inverse_width)),full(1e-6f));
        F inverse = aie::sub(aie::broadcast<uint32,16>(0x5f375a86),
                            aie::downshift(variance.cast_to<uint32>(),1)).cast_to<float>();
        for (int j = 0; j < 4; ++j)
            inverse = mul(inverse, aie::sub(full(1.5f), mul(mul(full(0.5f),variance), mul(inverse,inverse))));
        for (int i = 0; i < width; i += 16)
            store_bf16(y + row*width + i,
                       mul(mul(load_bf16(x + row*width + i), inverse),load_bf16(weight + i)));
    }
}
