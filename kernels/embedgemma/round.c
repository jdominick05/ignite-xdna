/* Copyright (C) 2026 The ignite-xdna contributors
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * fp32 -> bf16 (round to nearest, ties to even) for EmbeddingGemma 2's NPU inputs, AVX2 + OpenMP.
 *
 * round_bf16_rows(src, src_stride, dst, dst_stride, rows, cols): dst[r*dst_stride + c] = bf16(src[r*src_stride + c])
 * for r < rows, c < cols (strides in elements). Bit-identical to the NumPy reference
 * ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) for finite inputs, which is what model.bf16_bits computes; NaN and Inf are
 * not handled specially (an EmbeddingGemma activation is finite or the gate has already failed).
 * Build: cl /nologo /LD /O2 /arch:AVX2 /openmp round.c /Fe:embedgemma_round.dll
 */
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static void round_row(const float *src, uint16_t *dst, int64_t cols) {
    const __m256i bias = _mm256_set1_epi32(0x7FFF), one = _mm256_set1_epi32(1);
    int64_t c = 0;
    for (; c + 16 <= cols; c += 16) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(src + c));
        __m256i b = _mm256_loadu_si256((const __m256i *)(src + c + 8));
        a = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(a, bias), _mm256_and_si256(_mm256_srli_epi32(a, 16), one)), 16);
        b = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(b, bias), _mm256_and_si256(_mm256_srli_epi32(b, 16), one)), 16);
        /* packus works per 128-bit lane: [a0 b0 a1 b1] -> reorder to [a0 a1 b0 b1] */
        __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xD8);
        _mm256_storeu_si256((__m256i *)(dst + c), p);
    }
    for (; c < cols; ++c) {
        uint32_t u;
        memcpy(&u, src + c, 4);
        dst[c] = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    }
}

EXPORT void round_bf16_rows(const float *src, int64_t src_stride, uint16_t *dst, int64_t dst_stride,
                            int64_t rows, int64_t cols) {
    int64_t r;
#pragma omp parallel for schedule(static) if (rows * cols > (1 << 16))
    for (r = 0; r < rows; ++r) round_row(src + r * src_stride, dst + r * dst_stride, cols);
}
