// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <aie_api/aie.hpp>
#include <stdint.h>
using F=aie::vector<float,16>;
static inline F mul(F a,F b) {return aie::mul(a,b).to_vector<float>();}
#include "../bf16_product.hpp"
extern "C" void bf16_product_probe(const int32_t *packet,int32_t *output) {
    aie::store_v(output,aie::broadcast<int32,16>(packet[0]));
    F a=aie::load_v<16>(reinterpret_cast<const float *>(packet+16));
    F b=aie::load_v<16>(reinterpret_cast<const float *>(packet+32));
    aie::store_v(reinterpret_cast<float *>(output+16),bf16_operand_product(a,b));
}
