// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// One core's share of an M=1 bf16 GEMV, y = W x, for the whole-array decoder projection (see design.py).
//
// Every object the core receives is OBJ_WORDS long. The first of a call carries a header and x; each one after it is a
// weight chunk of 12 outputs x 192 inputs in mmul<4,8,4> order, [kstep 24][acc 3][kk 8][n 4]. The 12 outputs are
// three accumulators of 4; x is replicated into the four A rows, so every row of C is the same dot product and row 0
// is read out. Accumulation is fp32 across every chunk of a group, then one round-to-nearest-even to bf16.
#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HDR_WORDS = 16;
constexpr int KCHUNK = 192;
constexpr int KSTEPS = KCHUNK / 8;
constexpr int TILE = 32;              // one 8x4 B tile of bf16

extern "C" {

// Header words: [0] groups of 12 outputs this core makes, [1] K chunks per group, [2] K.
void gemv_begin(const int32_t *obj, int32_t *x_local, int32_t *ctl) {
    for (int i = 0; i < HDR_WORDS; i += 16)
        aie::store_v(ctl + i, aie::load_v<16>(obj + i));
    const int words = obj[2] / 2;
    for (int i = 0; i < words; i += 16)
        aie::store_v(x_local + i, aie::load_v<16>(obj + HDR_WORDS + i));
}

void gemv_chunk(const int32_t *obj, const int32_t *x_local, float *acc, int32_t q) {
    using MM = aie::mmul<4, 8, 4, bfloat16, bfloat16, accfloat>;
    const auto *x = reinterpret_cast<const bfloat16 *>(x_local) + q * KCHUNK;
    const auto *w = reinterpret_cast<const bfloat16 *>(obj);
    const bool reset = q == 0;
    MM c0(reset ? aie::zeros<float, 16>() : aie::load_v<16>(acc));
    MM c1(reset ? aie::zeros<float, 16>() : aie::load_v<16>(acc + 16));
    MM c2(reset ? aie::zeros<float, 16>() : aie::load_v<16>(acc + 32));
    for (int k = 0; k < KSTEPS; ++k) {
        auto row = aie::load_v<8>(x + k * 8);
        aie::vector<bfloat16, 32> a;
        a.insert<8>(0, row); a.insert<8>(1, row);
        a.insert<8>(2, row); a.insert<8>(3, row);
        c0.mac(a, aie::load_v<TILE>(w + k * 3 * TILE));
        c1.mac(a, aie::load_v<TILE>(w + k * 3 * TILE + TILE));
        c2.mac(a, aie::load_v<TILE>(w + k * 3 * TILE + 2 * TILE));
    }
    aie::store_v(acc, c0.to_vector<float>());
    aie::store_v(acc + 16, c1.to_vector<float>());
    aie::store_v(acc + 32, c2.to_vector<float>());
}

// Row 0 of each accumulator, rounded to bf16 on the bits (ties to even) and packed two to a word.
void gemv_emit(const float *acc, int32_t *out) {
    const auto *bits = reinterpret_cast<const uint32_t *>(acc);
    for (int i = 0; i < 12; i += 2) {
        uint32_t lo = bits[(i / 4) * 16 + i % 4], hi = bits[((i + 1) / 4) * 16 + (i + 1) % 4];
        lo = (lo + 0x7fffu + ((lo >> 16) & 1u)) >> 16;
        hi = (hi + 0x7fffu + ((hi >> 16) & 1u)) >> 16;
        out[i / 2] = static_cast<int32_t>(lo | (hi << 16));
    }
}

}
