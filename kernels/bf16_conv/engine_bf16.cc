// A bf16 convolution engine core: ONE program that serves every layer of a model.
//
// WHY THIS EXISTS BESIDE conv_bf16.cc. That file is milestone 1 - a single convolution whose every
// dimension is a compile-time constant, kept intact because a landed result
// (results/aie/bf16_conv_npu_20260921.log) cites it. It cannot run a model: a model has 71
// convolutions of differing shape, one compiled kernel each would be 71 programs, and program
// memory is 16 KB per core. Upstream's chained-ObjectFifo ResNet
// (programming_examples/ml/resnet/layers_conv2_x) has exactly that limitation and is why it stops
// at three blocks.
//
// The int8 engine (kernels/aie2/conv_engine/engine.cc) solved this and this file copies the
// solution: per-layer parameters arrive in the weight packet's header and are read on the core, so
// the program is fixed and only the instruction stream changes.
//
// WHAT STAYS COMPILE-TIME, AND WHY IT MUST. NCO - the number of output blocks, and therefore the
// number of accumulators alive at once. engine.cc's header comment is explicit that a run-time trip
// count over the output blocks spills the accumulators out of the vector registers and made the
// 66-layer frame 2.6 to 3 times slower. Every loop over NCO here is `#pragma unroll` with a
// constant bound for that reason. The tile shape is compile-time for the same reason.
//
// ACCUMULATORS IN FLIGHT. Measured on this silicon
// (results/aie/bf16_conv_bench_npu_20260921.log): 1 accumulator 24.44 GFLOPS, 2 -> 31.13,
// 4 -> 36.64, 8 -> 45.35, on identical work. Eight is the target, but eight OUTPUT BLOCKS would
// double both the weight packet and the output tile and overrun L1. So this takes engine.cc's
// trick instead - two column groups computed at once, `acc0` and `acc1`, sharing one loaded weight
// vector. Eight accumulators, four blocks' worth of weights, no extra L1.
//
// SHAPE. AIE2's bf16 matrix intrinsic is mmul<4, 8, 4>: 4 pixels by 8 input channels by 4 output
// channels, accumulating in fp32. The N of 4 is the structural difference from int8's
// mmul<4, 8, 8>, so the OUTPUT blocks by 4 where the input still blocks by 8. Usefully, the output
// tile is then the same SIZE as int8's - half the channels per block, twice the bytes each.
//
//   act   [cin_block][row][col][8]                   bf16, halo included, cin_block pitch in header
//   wts   [ky][kx][cin_block][cout_block][32]        bf16, 32 = 8 in by 4 out, in walk order
//   bias  [cout_block][16]                           bf16, pre-replicated to mmul's 4x4 C
//   out   [cout_block][row][col][4]                  bf16
//   psum  [cout_block][row][col][4]                  fp32 partial sums, then a held bf16 tile
//
// NO INTEGER EPILOGUE. int8's requantization - the shifts, the zero point of 128, the HardSwish and
// sigmoid constant fits - is deleted rather than widened, because bf16 needs none of it. What
// remains is ReLU and ReLU6, which are a max and a min.
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

// Header word indices (int32) inside the W packet. Deliberately far smaller than int8's 32 words:
// every requantization field is gone.
enum {
    H_OP = 0, H_K = 1, H_STRIDE = 2, H_NCIN = 3, H_FLAGS = 4,
    H_COUNT_OUT = 5, H_COUNT_ACC = 6,
    H_ROWS_IN = 7, H_COLS_IN = 8, H_PLANE_ELEMS = 9,
    H_PHASE0 = 12,   // 12..15, one per core row
};
enum { OP_NOP = 0, OP_CONV = 1, OP_RESIDUAL = 2 };
// F_LOAD_PSUM: continue a partial sum rather than starting from the bias, for a layer whose input
//   channels do not fit one packet. F_EMIT: write the activated tile out. F_HOLD: keep it for a
//   residual packet to add to. Without either, the fp32 partial sums stay in psum.
enum { F_LOAD_PSUM = 1, F_EMIT = 2, F_RELU = 4, F_RELU6 = 8, F_HOLD = 16 };

constexpr int HDR_BYTES = 128;
constexpr int TILE_ROWS = 5;
constexpr int TILE_COLS = 20;
constexpr int NCO = 4;                  // output blocks of 4 channels; also accumulators per group
constexpr int GROUPS = TILE_COLS / 4;   // 5 groups of 4 pixels across the tile

using MMUL = aie::mmul<4, 8, 4, bfloat16, bfloat16>;
using VA = aie::vector<bfloat16, MMUL::size_A>;   // 32 = 4 pixels x 8 channels
using VB = aie::vector<bfloat16, MMUL::size_B>;   // 32 = 8 in x 4 out
using VC = aie::vector<float, MMUL::size_C>;      // 16 = 4 pixels x 4 out

constexpr int BIAS_ELEMS = NCO * MMUL::size_C;               // 64 bf16 = 128 B
constexpr int W_OFFSET_ELEMS = HDR_BYTES / 2 + BIAS_ELEMS;   // header and bias, counted in bf16
constexpr int OUT_BLOCK_ELEMS = TILE_ROWS * TILE_COLS * 4;   // 400 bf16 per output block
constexpr int PSUM_BLOCK_ELEMS = TILE_ROWS * TILE_COLS * 4;  // 400 float per output block
constexpr int HOLD_OFFSET_ELEMS = NCO * PSUM_BLOCK_ELEMS;    // held bf16 tile lives past the sums

struct Hdr {
    int op, k, stride, ncin, flags;
    int rows_in, cols_in, plane_elems;
    int phase;
};

inline Hdr read_header(const int32_t *h, int core_row) {
    Hdr d;
    d.op = h[H_OP]; d.k = h[H_K]; d.stride = h[H_STRIDE]; d.ncin = h[H_NCIN];
    d.flags = h[H_FLAGS];
    d.rows_in = h[H_ROWS_IN]; d.cols_in = h[H_COLS_IN]; d.plane_elems = h[H_PLANE_ELEMS];
    d.phase = h[H_PHASE0 + (core_row & 3)];
    return d;
}

// ReLU and ReLU6 are the whole epilogue. MODNet-Cut's 35 Clip nodes are ReLU6; a plain ReLU is the
// same with the ceiling left off.
__attribute__((always_inline))
inline aie::vector<bfloat16, MMUL::size_C> epilogue(MMUL &acc, const Hdr &d) {
    aie::vector<bfloat16, MMUL::size_C> v = acc.template to_vector<bfloat16>();
    if (d.flags & F_RELU)
        v = aie::max(v, bfloat16(0.0f));
    if (d.flags & F_RELU6)
        v = aie::min(v, bfloat16(6.0f));
    return v;
}

// One output row of one tile, for either one column group (DUAL false) or two (DUAL true).
// DUAL is a template parameter, not a flag, so neither variant pays a branch in the inner loop and
// the accumulator arrays stay register-resident.
template <bool DUAL>
__attribute__((always_inline))
inline void conv_pass(const Hdr &d, const bfloat16 *act, const bfloat16 *w, const bfloat16 *bias,
                      bfloat16 *out, float *psum, int r, int g0, int g1) {
    MMUL acc0[NCO], acc1[NCO];
    const int off0 = (r * TILE_COLS + 4 * g0) * 4;
    const int off1 = (r * TILE_COLS + 4 * g1) * 4;

    if (d.flags & F_LOAD_PSUM) {
#pragma unroll
        for (int b = 0; b < NCO; ++b) {
            acc0[b] = MMUL(aie::load_v<MMUL::size_C>(psum + b * PSUM_BLOCK_ELEMS + off0));
            if (DUAL)
                acc1[b] = MMUL(aie::load_v<MMUL::size_C>(psum + b * PSUM_BLOCK_ELEMS + off1));
        }
    } else {
#pragma unroll
        for (int b = 0; b < NCO; ++b) {
            // The bias arrives pre-replicated to the accumulator's 4x4 shape, because bf16 has no
            // 4-element load - aie::load_v<4> does not compile for bfloat16, 16 is the smallest.
            aie::accum<accfloat, MMUL::size_C> ba;
            ba.from_vector(aie::load_v<MMUL::size_C>(bias + b * MMUL::size_C));
            const VC bv = ba.template to_vector<float>();
            acc0[b] = MMUL(bv);
            if (DUAL)
                acc1[b] = MMUL(bv);
        }
    }

    const int ncin = d.ncin;
    const int k = d.k;
    const bfloat16 *wp = w;

    if (d.stride == 2) {
        // Output pixel x reads input pixel 2x + kx, so four consecutive outputs are not four
        // consecutive inputs. Load eight pixels as two 4-pixel vectors and keep the even ones.
        // The unzip step is 8 ELEMENTS - one pixel's channels - exactly as in the int8 engine,
        // where 8 elements happened also to be 8 bytes.
        for (int ky = 0; ky < k; ++ky) {
            for (int kx = 0; kx < k; ++kx) {
                const int aoff0 = ((2 * r + ky) * d.cols_in + 8 * g0 + kx) * 8;
                const int aoff1 = aoff0 + (g1 - g0) * 64;
                for (int c = 0; c < ncin; ++c) {
                    const bfloat16 *plane = act + c * d.plane_elems;
                    auto [av0, odd0] = aie::interleave_unzip(
                        aie::load_unaligned_v<MMUL::size_A>(plane + aoff0),
                        aie::load_unaligned_v<MMUL::size_A>(plane + aoff0 + MMUL::size_A), 8);
                    VA av1 = av0;
                    if (DUAL) {
                        auto [e1, odd1] = aie::interleave_unzip(
                            aie::load_unaligned_v<MMUL::size_A>(plane + aoff1),
                            aie::load_unaligned_v<MMUL::size_A>(plane + aoff1 + MMUL::size_A), 8);
                        av1 = e1;
                    }
#pragma unroll
                    for (int b = 0; b < NCO; ++b) {
                        const VB wv = aie::load_v<MMUL::size_B>(wp);
                        wp += MMUL::size_B;
                        acc0[b].mac(av0, wv);
                        if (DUAL)
                            acc1[b].mac(av1, wv);
                    }
                }
            }
        }
    } else {
        for (int ky = 0; ky < k; ++ky) {
            for (int kx = 0; kx < k; ++kx) {
                const int aoff0 = ((r + ky) * d.cols_in + 4 * g0 + kx) * 8;
                const int aoff1 = aoff0 + (g1 - g0) * 32;
                for (int c = 0; c < ncin; ++c) {
                    const bfloat16 *plane = act + c * d.plane_elems;
                    // Four consecutive output columns read four consecutive input columns at this
                    // kernel offset, and [col][8] laid end to end is exactly mmul's 4x8 A.
                    const VA av0 = aie::load_unaligned_v<MMUL::size_A>(plane + aoff0);
                    VA av1 = av0;
                    if (DUAL)
                        av1 = aie::load_unaligned_v<MMUL::size_A>(plane + aoff1);
#pragma unroll
                    for (int b = 0; b < NCO; ++b) {
                        // One weight vector, both column groups. This is what makes eight
                        // accumulators cost four blocks of weights.
                        const VB wv = aie::load_v<MMUL::size_B>(wp);
                        wp += MMUL::size_B;
                        acc0[b].mac(av0, wv);
                        if (DUAL)
                            acc1[b].mac(av1, wv);
                    }
                }
            }
        }
    }

    if (d.flags & (F_EMIT | F_HOLD)) {
        bfloat16 *dst = (d.flags & F_EMIT)
                            ? out
                            : reinterpret_cast<bfloat16 *>(psum + HOLD_OFFSET_ELEMS);
#pragma unroll
        for (int b = 0; b < NCO; ++b) {
            aie::store_v(dst + b * OUT_BLOCK_ELEMS + off0, epilogue(acc0[b], d));
            if (DUAL)
                aie::store_v(dst + b * OUT_BLOCK_ELEMS + off1, epilogue(acc1[b], d));
        }
    } else {
#pragma unroll
        for (int b = 0; b < NCO; ++b) {
            aie::store_v(psum + b * PSUM_BLOCK_ELEMS + off0, acc0[b].template to_vector<float>());
            if (DUAL)
                aie::store_v(psum + b * PSUM_BLOCK_ELEMS + off1, acc1[b].template to_vector<float>());
        }
    }
}

}  // namespace

// wpkt: [header 128 B][bias 128 B][weights ...], all one buffer because a core tile has two input
// DMA channels and the activations take the other.
extern "C" void engine_bf16(int32_t *wpkt, bfloat16 *apkt, bfloat16 *out, float *psum,
                            int32_t core_row) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);

    const Hdr d = read_header(wpkt, core_row);
    if (d.op == OP_NOP)
        return;

    const bfloat16 *base = reinterpret_cast<const bfloat16 *>(wpkt);
    const bfloat16 *bias = base + HDR_BYTES / 2;
    const bfloat16 *w = base + W_OFFSET_ELEMS;

    for (int r = 0; r < TILE_ROWS; ++r) {
        int g = 0;
        // Two column groups at a time for eight live accumulators; GROUPS is odd, so the last
        // group runs the single-group instantiation.
        for (; g + 1 < GROUPS; g += 2)
            conv_pass<true>(d, apkt, w, bias, out, psum, r, g, g + 1);
        for (; g < GROUPS; ++g)
            conv_pass<false>(d, apkt, w, bias, out, psum, r, g, g);
    }
}
