// A bf16 convolution on AIE2, built on the native bfloat16 mmul.
//
// WHY THIS FILE EXISTS. The silicon does bf16 at 128 MACs/cycle (AMD's own device.yaml for AIE2,
// read in results/aie/notes_aie2_device_dtypes.log), and this repo has already run bf16 GEMM,
// GroupNorm and elementwise kernels on it. What has never existed anywhere - not here, not in
// mlir-aie's aie_kernels tree, not in the programming_examples - is a bf16 CONVOLUTION. Upstream's
// conv2dk1.cc and conv2dk3.cc are int8/uint8 only, and this repo's closest artifact,
// kernels/dispatch_floor/splice_conv.cc, is a broadcast scale-and-bias rather than a convolution.
// So there is no reference to copy and this is written from the arithmetic.
//
// SHAPE. AIE2's bf16 matrix intrinsic is mmul<4, 8, 4>: 4 pixels by 8 input channels by 4 output
// channels, accumulating in fp32. That N of 4 is the one structural difference from the int8 engine
// (kernels/aie2/conv_engine/engine.cc), whose mmul<4, 8, 8> is why its tiles are [block][row][col][8].
// Here the input keeps 8 channels per block, because K is still 8, and the OUTPUT blocks by 4.
//
//   act  [cin_block][row][col][8]   bf16, halo rows and columns included in row/col
//   wts  [ky][kx][cin_block][cout_block][32]   bf16, 32 = 8 input by 4 output, walked in this order
//        followed by [cout_block][16] bf16 of bias, ALREADY REPLICATED to the accumulator's shape:
//        mmul's C is 4 pixels by 4 channels row major, so element m*4+n wants bias[n], which is the
//        four bias values repeated four times. Replicating on the host rather than on the core also
//        avoids a sub-native load: aie::load_v<4> does not exist for bf16, and 16 is the smallest
//        bf16 vector the core can load.
//   out  [cout_block][row][col][4]  bf16
//
// The weight layout is defined to match the walk order below exactly, so the kernel reads it as one
// contiguous stream and never computes a weight address.
//
// WHY THE BIAS RIDES IN THE WEIGHT STREAM, and why it is bf16. An AIE2 core tile has two input and
// two output DMA channels, no more; a third input fifo is refused at placement with "tile (0, 2)
// requires 3 input/1 output DMA channels, but only 2 input/2 output available". Activations and
// weights take the two, so the bias is appended to the weight buffer. Rounding it to bf16 costs
// almost nothing here - it is one addend against ncin*8*k*k products, and the accumulation itself
// stays fp32 - but it IS a rounding the reference has to match, so the reference rounds it too.
//
// ROUNDING. conv_even is not optional: the accumulator-to-bf16 store only matches a host
// round-to-nearest-even pack with it set, which this repo learned once already in the bf16
// GroupNorm work (docs/DECISIONS.md, Peano notes).
#include <aie_api/aie.hpp>

#ifndef KDIM
#define KDIM 3
#endif
#ifndef ROWS_OUT
#define ROWS_OUT 4
#endif
#ifndef COLS_OUT
#define COLS_OUT 16
#endif
#ifndef COLS_IN
#define COLS_IN (COLS_OUT + KDIM - 1)
#endif
#ifndef ROWS_IN
#define ROWS_IN (ROWS_OUT + KDIM - 1)
#endif
#ifndef NCIN
#define NCIN 1
#endif
#ifndef NCOUT
#define NCOUT 1
#endif

using MMUL = aie::mmul<4, 8, 4, bfloat16, bfloat16>;

// 4 pixels per mmul, so a row of outputs is walked in groups of four.
static constexpr int GROUPS = COLS_OUT / 4;
static constexpr int PLANE = ROWS_IN * COLS_IN * 8;      // one input channel block, in elements
static constexpr int OPLANE = ROWS_OUT * COLS_OUT * 4;   // one output channel block, in elements

extern "C" void conv_bf16(bfloat16 *act, bfloat16 *wts, bfloat16 *out) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);

    const bfloat16 *bias = wts + KDIM * KDIM * NCIN * NCOUT * MMUL::size_B;

    // Bias for every output block, loaded once.
    aie::vector<float, MMUL::size_C> bv[NCOUT];
#pragma unroll
    for (int ob = 0; ob < NCOUT; ++ob) {
        aie::accum<accfloat, MMUL::size_C> bacc;
        bacc.from_vector(aie::load_v<MMUL::size_C>(bias + ob * MMUL::size_C));
        bv[ob] = bacc.to_vector<float>();
    }

    for (int r = 0; r < ROWS_OUT; ++r) {
        for (int g = 0; g < GROUPS; ++g) {
            // NCOUT accumulators in flight. A single accumulator would serialise on the vector
            // MAC's latency - every mac waiting on the one before it - which measured 27.18 GFLOPS,
            // 5.9% of this core's bf16 ceiling and only 1.09x one CPU thread. The int8 engine keeps
            // four in flight for the same reason (kernels/aie2/conv_engine/engine.cc).
            MMUL acc[NCOUT];
#pragma unroll
            for (int ob = 0; ob < NCOUT; ++ob)
                acc[ob] = MMUL(bv[ob]);

            const bfloat16 *wp = wts;
            for (int ky = 0; ky < KDIM; ++ky) {
                for (int kx = 0; kx < KDIM; ++kx) {
                    // 4 consecutive output columns read 4 consecutive input columns at this kernel
                    // offset, and [col][8] laid end to end is exactly mmul's 4x8 A.
                    const int aoff = ((r + ky) * COLS_IN + 4 * g + kx) * 8;
                    for (int cb = 0; cb < NCIN; ++cb) {
                        const aie::vector<bfloat16, MMUL::size_A> a =
                            aie::load_unaligned_v<MMUL::size_A>(act + cb * PLANE + aoff);
                        // One activation load feeds every output block, which is why the weight
                        // layout puts the output block innermost: this walk stays contiguous.
#pragma unroll
                        for (int ob = 0; ob < NCOUT; ++ob) {
                            acc[ob].mac(a, aie::load_v<MMUL::size_B>(wp + ob * MMUL::size_B));
                        }
                        wp += NCOUT * MMUL::size_B;
                    }
                }
            }
#pragma unroll
            for (int ob = 0; ob < NCOUT; ++ob)
                aie::store_v(out + ob * OPLANE + (r * COLS_OUT + 4 * g) * 4,
                             acc[ob].template to_vector<bfloat16>());
        }
    }
}
