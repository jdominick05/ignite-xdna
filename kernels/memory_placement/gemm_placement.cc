// Placement intervention on a 64 x (64*panels) x 64 tiled INT8 GEMM.
// Two operand panels are resident; alternating them tests local ping-pong
// addressing, without claiming overlap with DMA or whole-array GEMM performance.
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void gemm_placement(int32_t *params, int32_t *out,
        int8_t *__restrict a, int8_t *__restrict b, int8_t *reserve) {
    const uint32_t panels = params[0], alternate = params[1], seed = params[2];
    reserve[0] = (int8_t)seed; // Keep the reserved bank region live in both layouts.
    for (uint32_t j = 0; j < 8192; ++j) {
        a[j] = (int8_t)((j + seed) % 7) - 3;
        b[j] = (int8_t)((3*j + seed + 1) % 7) - 3;
    }
    using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, acc32>;
    event0();
    for (unsigned mb = 0; mb < 16; ++mb) {
        for (unsigned nb = 0; nb < 8; ++nb) {
            MMUL acc;
            for (unsigned p = 0; p < panels; ++p) {
                const unsigned base = alternate ? (p & 1)*4096 : 0;
                for (unsigned kb = 0; kb < 8; ++kb) {
                    auto av = aie::load_v<32>(a + base + (mb*8+kb)*32);
                    auto bv = aie::load_v<64>(b + base + (kb*8+nb)*64);
                    if (p == 0 && kb == 0) acc.mul(av, bv);
                    else acc.mac(av, bv);
                }
            }
            aie::store_v(out+(mb*8+nb)*32, acc.to_vector<int32_t>());
        }
    }
    event1();
    for (int k = 0; k < 256; ++k) { event0(); event1(); }
}
