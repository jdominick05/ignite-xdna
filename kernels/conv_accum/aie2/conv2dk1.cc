//===- conv2dk1.cc -------------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LOCAL COPY, ryzen-ai-xdna1-quantization. Byte-identical to the installed
// mlir-aie kernel at
//   ironenv/Lib/site-packages/mlir_aie/include/aie_kernels/aie2/conv2dk1.cc
// as of 2026-09-09, plus ONE change: conv2dk1_i8_vector's `n == 4` case is
// peeled so its four accumulators are named locals rather than a
// runtime-indexed array, making them register-resident. See the long comment
// at that site.
//
// It is copied rather than edited in place because that installed file is
// shared with every other session on this machine. Nothing outside this repo
// is modified; kernels/conv_accum/build_conv_accum.py redirects the one
// source lookup in-process.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

#ifdef SCALAR

const int32_t UMAX = 255;

#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_i8_scalar(int8_t *input, int8_t *kernels, uint8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale) {
  event0();

  int x, ic, oc, ic8, oc8;
  // scale=-17;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (x = 0; x < input_width; x++) { // col of output image
      for (oc8 = 0; oc8 < 8; oc8++) {
        int sum = 0;
        int sum_srs = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        // sum_srs=sum>>scale;
        sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}

#else // UINT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: uint8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_ui8_scalar(uint8_t *input, int8_t *kernels, uint8_t *output,
                         const int32_t input_width,
                         const int32_t input_channels,
                         const int32_t output_channels, const int scale) {
  event0();

  int x, ic, oc, ic8, oc8;
  // scale=-17;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (x = 0; x < input_width; x++) { // col of output image
      for (oc8 = 0; oc8 < 8; oc8++) {
        int sum = 0;
        int sum_srs = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            uint8_t val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int8_t k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                               (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        // sum_srs=sum>>scale;
        sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}

#endif // UINT8_ACT

#else // Vector

#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - vector
// act: int8, wts: int8, out: uint8
//
// Assume IC >= 16 as that gives ideal inner loop schedule
//
// TODO - Restricting input_width is mutiple of 32
// Because each VMAC works on 4 inputs at a time and we store intermediate
// results in 8 accumulators, having input_width be a multiple of 4*8=32 is
// ideal. However, we should be able to support input_width that is only a
// multiple of 4 but there is some strange scheduling happening now so for
// now, we do not.
//*****************************************************************************
// Local fix (2026-09-07, this checkout only): upstream restricted input_width
// to a multiple of 32 (assert, compiled out in release) and hardcoded
// iw_32_rem=0, so the remainder columns of any other width were silently
// never computed -- not a numeric bug, dead code. Rewritten to walk the
// width in blocks of N<=4 chunks (4 pixels/chunk) addressed directly from
// the input/output base pointers (no incremental pointer state to drift),
// keeping live accumulators within AIE2's 6 hardware accumulator registers
// (upstream used 8 concurrently). See conv2dk3.cc's own fix for the same
// pattern and its empirical verification.
void conv2dk1_i8_vector(int8_t *input, int8_t *kernels, uint8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale) {
  event0();

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;
  ::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
  ::aie::set_rounding(
      aie::rounding_mode::positive_inf); // Needed to saturate properly to uint8

  const int scaleT = scale;
#ifdef INPUT_WIDTH
  const int iw = INPUT_WIDTH;
#else
  const int iw = input_width;
#endif
  const int ics8 = input_channels / 8;
  const int ocs8 = output_channels / 8;
  assert(ics8 > 2); // Assume IC >= 16
  assert((iw % 4) == 0);
  const int total_chunks = iw / 4;

  for (int oc = 0; oc < ocs8; oc++) {
    int chunk = 0;
    while (chunk < total_chunks) {
      int n = total_chunks - chunk;
      if (n > 4)
        n = 4;

      // ACCUMULATOR RESIDENCY FIX (this repo's copy only; see the header comment
      // at the top of this file). `acc_tmp` below is an array indexed by `x`,
      // whose trip count `n` is a RUNTIME value, so the `x` loop cannot be
      // unrolled and the array cannot live in registers -- registers are not
      // dynamically addressable. The result is that every `.mac()` becomes
      // load-four-quarters / mac / store-four-quarters, and the hot loop issues
      // 0.045 MACs per cycle against the int8 GEMM's 0.889
      // (results/aie/conv_issue_rate_decomposed.log).
      //
      // n == 4 is the common case -- `total_chunks` is iw/4 and every width this
      // design uses is a multiple of 4 -- so it is peeled here into four NAMED
      // accumulators, which is exactly the pattern mm.cc's
      // matmul_vectorized_2x2_mmul uses (C00, C01, C10, C11). Four live 4x8x8
      // acc32 accumulators is one full 1024-bit register each, under this
      // machine's measured five-register spill-free ceiling
      // (results/aie/accumulator_width_vs_count.log).
      //
      // The array loop is KEPT as the tail and is genuinely reached: widths whose
      // total_chunks is not a multiple of 4 end with a short block. iw=56 gives
      // 14 = 4+4+4+2, so the 56x56 shape in kernels/bottleneck_sweep runs the tail
      // at n=2 and verifies. It is correctness, not dead code.
      if (n == 4) {
        MMUL4x8x8 C0 = aie::zeros<acc32, 32>();
        MMUL4x8x8 C1 = aie::zeros<acc32, 32>();
        MMUL4x8x8 C2 = aie::zeros<acc32, 32>();
        MMUL4x8x8 C3 = aie::zeros<acc32, 32>();

        int8_t *k_ptr = kernels + (size_t)oc * ics8 * 64;
        for (int ic = 0; ic < ics8; ic++) {
          aie::vector<int8, 64> in_b = aie::load_v<64>(k_ptr);
          k_ptr += 64;
          const int8_t *a_ptr =
              input + (size_t)ic * iw * 8 + (size_t)chunk * 32;
          C0.mac(aie::load_v<32>(a_ptr + 0 * 32), in_b);
          C1.mac(aie::load_v<32>(a_ptr + 1 * 32), in_b);
          C2.mac(aie::load_v<32>(a_ptr + 2 * 32), in_b);
          C3.mac(aie::load_v<32>(a_ptr + 3 * 32), in_b);
        }

        uint8_t *o_ptr = output + (size_t)oc * iw * 8 + (size_t)chunk * 32;
        aie::store_v(o_ptr + 0 * 32, C0.to_vector<uint8>(scaleT));
        aie::store_v(o_ptr + 1 * 32, C1.to_vector<uint8>(scaleT));
        aie::store_v(o_ptr + 2 * 32, C2.to_vector<uint8>(scaleT));
        aie::store_v(o_ptr + 3 * 32, C3.to_vector<uint8>(scaleT));
        chunk += 4;
        continue;
      }

      MMUL4x8x8 acc_tmp[4];
      for (int x = 0; x < n; x++)
        acc_tmp[x] = aie::zeros<acc32, 32>();

      int8_t *k_ptr = kernels + (size_t)oc * ics8 * 64;
      for (int ic = 0; ic < ics8; ic++) {
        aie::vector<int8, 64> in_b = aie::load_v<64>(k_ptr);
        k_ptr += 64;
        for (int x = 0; x < n; x++) {
          const int8_t *a_ptr =
              input + (size_t)ic * iw * 8 + (size_t)(chunk + x) * 32;
          aie::vector<int8, 32> in_a = aie::load_v<32>(a_ptr);
          acc_tmp[x].mac(in_a, in_b);
        }
      }

      for (int x = 0; x < n; x++) {
        aie::vector<uint8, 32> o1 = acc_tmp[x].to_vector<uint8>(scaleT);
        uint8_t *o_ptr =
            output + (size_t)oc * iw * 8 + (size_t)(chunk + x) * 32;
        aie::store_v(o_ptr, o1);
      }
      chunk += n;
    }
  }

  event1();
}

#else // UINT8_ACT

//*****************************************************************************
// conv2d 1x1 - vector
// act: uint8, wts: int8, out: uint8
//
// Assume IC >= 16 as that gives ideal inner loop schedule
//
// TODO - Restricting input_width is mutiple of 32
// Because each VMAC works on 4 inputs at a time and we store intermediate
// results in 8 accumulators, having input_width be a multiple of 4*8=32 is
// ideal. However, we should be able to support input_width that is only a
// multiple of 4 but there is some strange scheduling happening now so for
// now, we do not.
//*****************************************************************************
// Local fix (2026-09-07, this checkout only): see conv2dk1_i8_vector above --
// same dead-code remainder, same fix.
void conv2dk1_ui8_vector(uint8_t *input, int8_t *kernels, uint8_t *output,
                         const int32_t input_width,
                         const int32_t input_channels,
                         const int32_t output_channels, const int scale) {
  event0();

  using MMUL4x8x8 = aie::mmul<4, 8, 8, uint8, int8>;
  ::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
  ::aie::set_rounding(
      aie::rounding_mode::positive_inf); // Needed to saturate properly to uint8

  const int scaleT = scale;
#ifdef INPUT_WIDTH
  const int iw = INPUT_WIDTH;
#else
  const int iw = input_width;
#endif
  const int ics8 = input_channels / 8;
  const int ocs8 = output_channels / 8;
  assert(ics8 > 2); // Assume IC >= 16
  assert((iw % 4) == 0);
  const int total_chunks = iw / 4;

  for (int oc = 0; oc < ocs8; oc++) {
    int chunk = 0;
    while (chunk < total_chunks) {
      int n = total_chunks - chunk;
      if (n > 4)
        n = 4;

      MMUL4x8x8 acc_tmp[4];
      for (int x = 0; x < n; x++)
        acc_tmp[x] = aie::zeros<acc32, 32>();

      int8_t *k_ptr = kernels + (size_t)oc * ics8 * 64;
      for (int ic = 0; ic < ics8; ic++) {
        aie::vector<int8, 64> in_b = aie::load_v<64>(k_ptr);
        k_ptr += 64;
        for (int x = 0; x < n; x++) {
          const uint8_t *a_ptr =
              input + (size_t)ic * iw * 8 + (size_t)(chunk + x) * 32;
          aie::vector<uint8, 32> in_a = aie::load_v<32>(a_ptr);
          acc_tmp[x].mac(in_a, in_b);
        }
      }

      for (int x = 0; x < n; x++) {
        aie::vector<uint8, 32> o1 = acc_tmp[x].to_vector<uint8>(scaleT);
        uint8_t *o_ptr =
            output + (size_t)oc * iw * 8 + (size_t)(chunk + x) * 32;
        aie::store_v(o_ptr, o1);
      }
      chunk += n;
    }
  }

  event1();
}

#endif // UINT8_ACT

#endif // Vector

//*****************************************************************************
// conv2d 1x1 wrappers
//*****************************************************************************
extern "C" {

#ifdef SCALAR

#ifdef INT8_ACT

void conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
}

#else // UINT8_ACT

void conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
                  const int32_t input_width, const int32_t input_channels,
                  const int32_t output_channels, const int scale) {
  conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                      output_channels, scale);
}

#endif // UINT8_ACT

#else // Vector

#ifdef INT8_ACT

void conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  conv2dk1_i8_vector(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
}

#else // UINT8_ACT

void conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
                  const int32_t input_width, const int32_t input_channels,
                  const int32_t output_channels, const int scale) {
  conv2dk1_ui8_vector(input, kernels, output, input_width, input_channels,
                      output_channels, scale);
}

#endif // UINT8_ACT

#endif // Vector

} // extern "C"