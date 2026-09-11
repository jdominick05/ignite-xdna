//===- conv_im2col_kernel.cc --------------------------------------*- C++ -*-===//
//
// Zero-Realignment Vectorized 3x3 Conv Im2Col Compute Kernel for Tile(0, 2)
// AMD Phoenix AIE2 (XDNA1, npu1_4col)
//
// Consumes 288-byte receptive field patches directly from L1 ping-pong buffers
// (%core_ping, %core_pong) populated by the 4-D Strided MemTile DMA.
//
// Zero-Realignment Contract:
// 1. Zero Software Sliding Window: No aie::sliding_mul, no vshift, no vmov.
// 2. Direct Contiguous Vector Loads: The 3x3x32 patch (288 bytes) is loaded in
//    exact 32-byte (256-bit) contiguous vector chunks via aie::load_v<32>.
// 3. Hardware VMAC Saturation: Uses aie::mmul<4, 8, 8, int8, int8> lowering to
//    native vmac instructions on Slot [v].
// 4. Register Stationarity: Stationary L1 weights (C_out=32) are multiplied
//    against input activations into 4 INT32 accumulators (cm0..cm3).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <aie_api/aie.hpp>
#include <aie_kernels/aie_kernel_utils.h>

using MMUL = aie::mmul<4, 8, 8, int8, int8>;

extern "C" {

/// Compute a single 288-byte receptive field patch against stationary L1 weights.
///
/// Parameters:
///   patch   : Pointer to 288 contiguous INT8 bytes in L1 (%core_ping or %core_pong)
///             representing 9 spatial taps * 32 input channels.
///   weights : Pointer to stationary L1 weights (9 taps * 4 blocks * 64 bytes = 2304 bytes).
///   out     : Pointer to destination L1 buffer for 128 INT32 accumulators
///             (4 blocks of 32 elements).
void conv_im2col_kernel(
    const int8_t *__restrict patch,
    const int8_t *__restrict weights,
    int32_t *__restrict out)
{
    // Accumulator registers cm0, cm1, cm2, cm3 (4 x 1024-bit accumulators)
    MMUL c0 = aie::zeros<acc32, 32>();
    MMUL c1 = aie::zeros<acc32, 32>();
    MMUL c2 = aie::zeros<acc32, 32>();
    MMUL c3 = aie::zeros<acc32, 32>();

    const int8_t *w_ptr = weights;
    const int8_t *a_ptr = patch;

    // 9 reduction iterations corresponding to the 9 taps (3x3 receptive field).
    // Each tap contains 32 channels loaded as a contiguous vector chunk.
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(9)
    for (int k = 0; k < 9; ++k) {
        // Direct aligned 32-byte vector load on Slot [a] (Load 1 unit)
        aie::vector<int8, 32> va = aie::load_v<32>(a_ptr);
        a_ptr += 32;

        // Stationary weight block loads on Slot [b] (Load 2 unit)
        // Each block is an 8x8 INT8 matrix (64 bytes = 2 x 32B loads)
        aie::vector<int8, 64> vb0 = aie::load_v<64>(w_ptr);
        aie::vector<int8, 64> vb1 = aie::load_v<64>(w_ptr + 64);
        aie::vector<int8, 64> vb2 = aie::load_v<64>(w_ptr + 128);
        aie::vector<int8, 64> vb3 = aie::load_v<64>(w_ptr + 192);
        w_ptr += 256;

        // 4 VMAC instructions on Slot [v] (Vector unit)
        // Zero vshift, zero vmov register packing
        c0.mac(va, vb0);
        c1.mac(va, vb1);
        c2.mac(va, vb2);
        c3.mac(va, vb3);
    }

    // Direct 256-bit vector stores to L1 output memory on Slot [s]
    aie::store_v(out,       c0.to_vector<int32>());
    aie::store_v(out + 32,  c1.to_vector<int32>());
    aie::store_v(out + 64,  c2.to_vector<int32>());
    aie::store_v(out + 96,  c3.to_vector<int32>());
}

/// Ping-pong buffer consumer pipeline driver.
///
/// Drives execution over alternating %core_ping and %core_pong buffers for
/// n_iterations rounds (e.g. 3 iterations = 6 spatial patches, matching
/// the 4-D strided MemTile DMA configuration in im2col_4d.mlir).
///
/// Parameters:
///   ping_buf     : Pointer to %core_ping (288 bytes)
///   pong_buf     : Pointer to %core_pong (288 bytes)
///   weights      : Pointer to stationary weights in L1
///   out_ping     : Destination buffer for ping results
///   out_pong     : Destination buffer for pong results
///   n_iterations : Number of ping-pong rounds
void conv_im2col_ping_pong(
    const int8_t *__restrict ping_buf,
    const int8_t *__restrict pong_buf,
    const int8_t *__restrict weights,
    int32_t *__restrict out_ping,
    int32_t *__restrict out_pong,
    int n_iterations)
{
    for (int iter = 0; iter < n_iterations; ++iter) {
        conv_im2col_kernel(ping_buf, weights, out_ping + iter * 128);
        conv_im2col_kernel(pong_buf, weights, out_pong + iter * 128);
    }
}

} // extern "C"
