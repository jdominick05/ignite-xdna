// clock_probe: bracket a DMA-free loop with the two core instruction events
// (event0 / event1) so the tile's trace unit stamps both with its 64-bit
// timer. The host fits hardware time against the stamped cycle delta to
// recover the core clock (docs/SILICON.md objective S0).
//
// Why events and not the counter directly: Peano (llvm-aie 22) declares
// get_cycles() in its aie_api compat header but never defines it (ld.lld:
// undefined symbol), does not lower __builtin_readcyclecounter for aie2, and
// rejects inline asm. The trace unit is the one path to the tile timer that
// the open toolchain exposes.
//
// Two loop modes, selected per call from the input buffer so one xclbin
// serves every point of the sweep (no per-target recompile):
//
//   mode 0 (scalar): `target` iterations of a dependent add on a volatile
//          local -- a load, add, store and loop step per iteration.
//   mode 1 (vector): `target` iterations of a dependent 16-lane int32
//          vector add. Cycles per iteration should be a small stable
//          constant for both; the two modes must fit to the SAME frequency.
//
// Input  (int32): [0] mode, [1] target (iterations)
// Output (int32): [0] mode echo   [1] target echo
//                 [2] global tile row   [3] global tile col   (from get_coreid())
//                 [4] checksum: mode 0  sum_{i<target} i        mod 2^32
//                               mode 1  16 * sum_{i<target} i   mod 2^32
//                 [5..] zero
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef N_OUT
#error "N_OUT must be defined"
#endif
#ifndef FLUSH_PAIRS
#define FLUSH_PAIRS 256
#endif

extern "C" {

void clock_probe(int32_t *in, int32_t *out) {
    const uint32_t mode = (uint32_t)in[0];
    const uint32_t target = (uint32_t)in[1];

    int32_t check = 0;
    if (mode == 0) {
        volatile int32_t sink = 0;
        event0();
        for (uint32_t i = 0; i < target; ++i) {
            sink = sink + (int32_t)i;
        }
        event1();
        check = sink;
    } else {
        aie::vector<int32_t, 16> acc = aie::zeros<int32_t, 16>();
        event0();
        for (uint32_t i = 0; i < target; ++i) {
            acc = aie::add(acc, aie::broadcast<int32_t, 16>((int32_t)i));
        }
        event1();
        check = aie::reduce_add(acc);
    }

    // Flush. The trace unit packs event frames into 32-byte packets and the
    // shim S2MM DMA writes 64-byte bursts, so one event0/event1 pair alone
    // never reaches host memory (mlir-aie programming_guide section-4b: "a
    // simple core may have too few events to create a valid trace packet").
    // Trailing filler pairs push the real pair out; the host reads the FIRST
    // event0 and the FIRST event1 only.
    for (int k = 0; k < FLUSH_PAIRS; ++k) {
        event0();
        event1();
    }

    const aie::tile_id gid = aie::tile::current().global_id();

    for (int i = 0; i < N_OUT; ++i)
        out[i] = 0;
    out[0] = (int32_t)mode;
    out[1] = (int32_t)target;
    out[2] = (int32_t)gid.row;
    out[3] = (int32_t)gid.col;
    out[4] = check;
}

} // extern "C"
