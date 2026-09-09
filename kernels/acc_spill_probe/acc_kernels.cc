// Find the AIE2 accumulator register count by compiling K live mmul accumulators
// and watching for the first stack spill. Compile only; never runs on hardware.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef NACC
#define NACC 4
#endif

// The shape upstream mlir-aie's conv2dk3 uses: 4x8x8 int8 with an int32 accumulator.
using MMUL = aie::mmul<4, 8, 8, int8_t, int8_t, acc32>;

extern "C" void acc_probe(const int8_t *__restrict A, const int8_t *__restrict B,
                          int8_t *__restrict C) {
    MMUL acc[NACC];

#pragma clang loop unroll(full)
    for (int i = 0; i < NACC; ++i)
        acc[i].mul(aie::load_v<32>(A + 32 * i), aie::load_v<64>(B + 64 * i));

    // The k loop keeps every accumulator live across iterations, so the register
    // allocator cannot retire one early.
    for (int k = 1; k < 8; ++k) {
#pragma clang loop unroll(full)
        for (int i = 0; i < NACC; ++i)
            acc[i].mac(aie::load_v<32>(A + 256 * k + 32 * i),
                       aie::load_v<64>(B + 512 * k + 64 * i));
    }

#pragma clang loop unroll(full)
    for (int i = 0; i < NACC; ++i)
        aie::store_v(C + 32 * i, acc[i].template to_vector<int8_t>());
}
