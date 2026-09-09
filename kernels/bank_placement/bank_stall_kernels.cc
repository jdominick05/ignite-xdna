// bank_stall_probe: a POSITIVE CONTROL for whether the AIE2 trace unit can see a
// same-bank dual-load stall at all.
//
// WHY THIS EXISTS
// ---------------
// results/aie/bank_ab_h12_npu.log moved the int8 GEMM's operands out of a shared 16 KB
// bank exactly as designed and could not resolve the effect: wall time on a shared
// machine cannot see a ~3% core-side change (96 cycles of a 3,274-cycle call). Its own
// closing paragraph names the fix -- read the trace unit's stall taxonomy inside the
// dispatch, where host contention cannot reach.
//
// But results/aie/pmu_probe_npu.log established that only LOCK_STALL has ever read
// nonzero on this machine, and said plainly that MEMORY_STALL, STREAM_STALL and
// CASCADE_STALL "are not shown to work by this run. A kernel that uses them is needed
// before any of those three can be trusted." A same-bank dual-load conflict should
// surface as MEMORY_STALL. So pointing the trace at the GEMM's two arms FIRST would be
// uninterpretable: MEMORY_STALL = 0 in both arms cannot distinguish "no conflict" from
// "this event does not capture bank conflicts", and there is no BANK_CONFLICT event in
// the CoreEvent enum this toolchain ships (checked: the nearest are GROUP_STALL 22,
// MEMORY_STALL 23, DM_ACCESS_TO_UNAVAILABLE 66).
//
// HOW THE TWO ARMS ARE MADE, AND WHY IT IS NOT A MODE FLAG
// --------------------------------------------------------
// This kernel has NO arm selector. It always issues the same two load streams from the
// same two offsets in the same array. The arms are produced by RELOCATING the array:
// bank_ab_h12 established that the allocator lays a core's local buffers immediately
// above its stack, so raising the worker's stack_size shifts every buffer up by the same
// amount and changes nothing else. Sliding the array across a 16 KB bank boundary moves
// the second stream into or out of the first stream's bank while leaving the instruction
// schedule, the trip count and the kernel object identical. That is a stronger control
// than a mode branch, which would compile two different code paths.
//
// Local memory on an AIE2 core tile is 64 KB in four 16 KB banks, so two addresses share
// a bank iff (addr >> 14) matches. The two streams sit 8 KB apart inside a 12 KB array,
// so they share a bank iff (base mod 16384) < 8192 -- flipped by a stack_size sweep.
// The kernel reports both absolute addresses and the HOST decides which arm each run was;
// nothing here assumes the intervention landed.
//
// SIZE CONSTRAINT, MEASURED NOT GUESSED
// -------------------------------------
// The core's .bss region is about 16 KB on this design, not the full 64 KB. A 32 KB
// array was tried first and ld.lld rejected it ("section '.bss' will not fit in region
// 'data': overflowed by 16640 bytes"), which is what fixes the array at 12 KB and the
// stream separation at 8 KB rather than a full bank.
//
// Input  (int32): [0] unused, [1] target (iterations)
// Output (int32): [0] 0             [1] target echo
//                 [2] tile row      [3] tile col
//                 [4] checksum      [5] input fifo buffer address
//                 [6] stream A address   [7] stream B address
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef N_OUT
#error "N_OUT must be defined"
#endif
#ifndef FLUSH_PAIRS
#define FLUSH_PAIRS 256
#endif

#define VEC 16
// PAD_WORDS slides both streams up inside one array. It is the RELOCATION LEVER, and it
// is an index offset rather than a link-time one on purpose: stack_size was tried first
// and does NOT move this array. Measured -- at stack 0x400 and 0x1400 the streams landed
// at 0x74100/0x76100 both times. The allocator lays ObjectFifo buffers above the stack,
// which is what bank_ab_h12 shifted, but a kernel's own .bss is placed by the linker and
// the stack does not touch it. An index offset cannot depend on any of that.
#ifndef PAD_WORDS
#define PAD_WORDS 0
#endif
#ifndef A_FROM_FIFO
#define A_FROM_FIFO 0
#endif
#define ARRAY_WORDS (10240 / 4)      // 10 KB: what is left of .bss once the two 4 KB
                                     // ObjectFifo buffers are placed (15.25 KB overflowed by 3328)
#define STREAM_B_WORDS (2048 / 4)    // 2 KB apart
#define IN_STREAM_WORDS 256          // stream B starts 1 KB into the input fifo buffer
#define WINDOW_WORDS 256             // 1 KB window per stream
#define WINDOW_MASK (WINDOW_WORDS - 1)

static int32_t probe_buf[ARRAY_WORDS];

extern "C" {

void bank_stall_probe(int32_t *in, int32_t *out) {
    const uint32_t target = (uint32_t)in[1];

    // Touch the buffer so it is resident and cannot be optimised away.
    for (int i = 0; i < ARRAY_WORDS; ++i)
        probe_buf[i] = i;

    // Stream A comes from the kernel's own .bss array; stream B from the INPUT
    // OBJECTFIFO BUFFER. Both are core-local memory and both are dual-issuable, but the
    // allocator places them independently, which is the only way to get two streams into
    // different banks on this design. Keeping both in .bss was tried and is IMPOSSIBLE
    // here: the core's .bss region is about 16 KB and starts 256 bytes into a bank, so it
    // lies entirely inside ONE 16 KB bank and no static array can straddle a boundary.
    // THE ARM SELECTOR, and it changes exactly one thing: where stream A's base is.
    // The loop below, its trip count and its two loads are identical either way.
    //   A_FROM_FIFO=0  A from .bss (bank 29), B from the fifo buffer (bank 30)  SEPARATED
    //   A_FROM_FIFO=1  both from the fifo buffer, adjacent 1 KB windows          COLLIDING
    // It has to be done this way. The core's .bss lies entirely inside ONE 16 KB bank
    // (measured: it spans 0x75000-0x77C00, bank 29) and the input fifo buffer begins
    // exactly on the next bank boundary (0x78000, bank 30), so no offset within .bss can
    // ever reach the fifo's bank and no static array can straddle a boundary. Putting
    // both streams inside the fifo buffer is the only way to make them collide.
#if A_FROM_FIFO
    int32_t *pa = in + IN_STREAM_WORDS + WINDOW_WORDS;
#else
    int32_t *pa = probe_buf + PAD_WORDS;
#endif
    int32_t *pb = in + IN_STREAM_WORDS;

    aie::vector<int32_t, VEC> acc = aie::zeros<int32_t, VEC>();
    uint32_t idx = 0;

    event0();
    for (uint32_t i = 0; i < target; ++i) {
        // Two INDEPENDENT loads: nothing here forces an order, so the scheduler is free
        // to pair them into one cycle. That pairing is exactly what a same-bank conflict
        // would have to break.
        aie::vector<int32_t, VEC> va = aie::load_v<VEC>(pa + idx);
        aie::vector<int32_t, VEC> vb = aie::load_v<VEC>(pb + idx);
        acc = aie::add(acc, aie::add(va, vb));
        idx = (idx + VEC) & WINDOW_MASK;
    }
    event1();

    // Flush: the trace unit packs frames into 32-byte packets and the shim S2MM writes
    // 64-byte bursts, so one event pair never reaches host memory on its own.
    for (int k = 0; k < FLUSH_PAIRS; ++k) {
        event0();
        event1();
    }

    const aie::tile_id gid = aie::tile::current().global_id();

    for (int i = 0; i < N_OUT; ++i)
        out[i] = 0;
    out[0] = 0;
    out[1] = (int32_t)target;
    out[2] = (int32_t)gid.row;
    out[3] = (int32_t)gid.col;
    out[4] = aie::reduce_add(acc);
    out[5] = (int32_t)(uintptr_t)in;
    out[6] = (int32_t)(uintptr_t)pa;
    out[7] = (int32_t)(uintptr_t)pb;
}

} // extern "C"
