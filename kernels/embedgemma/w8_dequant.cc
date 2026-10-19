// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// W8 B-tile dequantization for kernels/gemm_generic/whole_array_generic.py --b-w8 (k 64, n 64, bf16 4x8x4 mmul).
// The host (tools/embedgemma_gemm_probe.py w8_tiles) writes each 64x64 weight tile already in the L1 order the bf16
// matmul reads, 8x4 blocks (block (kb, nb) at (kb * 16 + nb) * 32, row-major inside), as one byte per weight in
// sign-magnitude form, b = sign << 7 | |q| with q in [-127, 127]; then, per 4-column block nb, 8 bf16 scales
// [s0 s1 s2 s3 s0 s1 s2 s3] for columns 4nb..4nb+3 (one scale per column per 64 K rows: the group is the K tile).
// Output: bf16(q * s), the value kernels/embedgemma/w4.py's dequant() gives.
// The bf16 with bits sign | 0x4300 | |q| is +-(128 + |q|) and the one with bits sign | 0x4300 is +-128, both exact.
// Both come from the byte without a shift: sign-extended to 16 bits, a negative byte is 0xff80 | |q|, so
// (x & 0xc37f) | 0x4300 = sign | 0x4300 | |q| and that & 0xff80 = sign | 0x4300 (the first build shifted the sign
// bit up, which Peano lowers through the accumulator: 55 bundles per 128 weights);
// (128 + |q|) s - 128 s = q s exactly in the fp32 accumulator (each product has at most 16 significant bits), and
// the store rounds it to bf16 once, ties to even.
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr int K = 64, N = 64, KB = K / 8, NB = N / 4;

using I16 = aie::vector<int16, 32>;
using B16 = aie::vector<bfloat16, 16>;

extern "C" void egemm_w8_dequant(const uint8_t *__restrict in, bfloat16 *__restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *sc = reinterpret_cast<const bfloat16 *>(in + K * N);
  const I16 keep = aie::broadcast<int16, 32>((int16_t)0xc37f), base = aie::broadcast<int16, 32>(0x4300),
            hi = aie::broadcast<int16, 32>((int16_t)0xff80);
  for (int nb = 0; nb < NB; ++nb) {
    aie::vector<bfloat16, 8> h = aie::load_v<8>(sc + nb * 8);
    B16 s = aie::concat(h, h);
    for (int kb = 0; kb < KB; ++kb) chess_prepare_for_pipelining {
      const int idx = (kb * NB + nb) * 32;
      I16 w = aie::load_v<32>(reinterpret_cast<const int8_t *>(in) + idx).unpack();   // sign-extending
      I16 fb = aie::bit_or(aie::bit_and(w, keep), base);
      I16 cb = aie::bit_and(fb, hi);
      for (int half = 0; half < 2; ++half) {
        B16 f = fb.extract<16>(half).cast_to<bfloat16>();
        B16 c = cb.extract<16>(half).cast_to<bfloat16>();
        aie::store_v(out + idx + half * 16, aie::msc(aie::mul(f, s), c, s).to_vector<bfloat16>());
      }
    }
  }
}
