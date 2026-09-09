// A fixed external object: address permutations belong only to the IRON wrapper.
// All input DMA, operand initialization, output stores and filler packets are
// outside the event pair. Inputs are small enough that the int32 checksum cannot
// overflow over the accepted iteration range (<= 2^20).
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void memory_probe(int32_t *params, int32_t *out,
                             int16_t *__restrict a, int16_t *__restrict b) {
    const uint32_t iterations = (uint32_t)params[0];
    const uint32_t mode = (uint32_t)params[1];
    const uint32_t seed = (uint32_t)params[2];
    for (uint32_t j = 0; j < 512; ++j) {
        a[j] = (int16_t)((j + seed) % 7) - 3;
        b[j] = (int16_t)((j * 3 + seed + 1) % 7) - 3;
    }
    aie::accum<acc64, 16> acc = aie::zeros<acc64, 16>();
    if (mode == 0) {
        event0();
        for (uint32_t i = 0; i < iterations; ++i) {
            const uint32_t offset = (i & 31) * 16;
            auto av = aie::load_v<16>(a + offset);
            auto bv = aie::load_v<16>(b + offset);
            acc = aie::mac(acc, av, bv);
        }
        event1();
    } else {
        auto ones = aie::broadcast<int16_t, 16>(1);
        event0();
        for (uint32_t i = 0; i < iterations; ++i) {
            const uint32_t offset = (i & 31) * 16;
            auto av = aie::load_v<16>(a + offset);
            acc = aie::mac(acc, av, ones);
        }
        event1();
    }
    const int32_t checksum = aie::reduce_add(acc.to_vector<int32_t>());
    for (int k = 0; k < 256; ++k) { event0(); event1(); }
    for (int j = 0; j < 64; ++j) out[j] = 0;
    out[0] = (int32_t)iterations;
    out[1] = (int32_t)mode;
    out[2] = checksum;
    const auto tile = aie::tile::current().global_id();
    out[3] = tile.row;
    out[4] = tile.col;
    out[5] = (int32_t)seed;
}
