// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/**
 * src/ignite_xdna/pipelines/preprocess_simd.c
 *
 * High-Performance Fused Ingress Preprocessing Kernel for YOLOv8 on AMD Phoenix.
 * Combines in a single memory pass:
 *   1. Letterbox aspect-ratio preserving bounding box calculation
 *   2. Bit-exact Q11 fixed-point bilinear spatial interpolation (matches OpenCV INTER_LINEAR)
 *   3. BGR -> RGB planar channel transposition (HWC -> CHW)
 *   4. Zero-copy uint8 -> int8 scale quantization: (val - 128)
 *   5. Direct write into DMA-pinned bo_in host memory buffer
 *
 * Parallelized with OpenMP across CPU cores with zero Python GIL locking.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_MSC_VER) || defined(__AVX2__)
#include <immintrin.h>
#define HAVE_AVX2 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define PAD_VALUE_INT8 ((int8_t)-14) // 114 - 128 = -14 (0xF2)

typedef struct {
    int x0;
    int x1;
    int bx0;
    int bx1;
    int off0;
    int off1;
} XCoordTable;

// Export symbol for Windows DLL
#if defined(_WIN32) || defined(__CYGWIN__)
#define PREPROCESS_API __declspec(dllexport)
#else
#define PREPROCESS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fused Letterbox + Bilinear Resize + BGR-to-RGB + Quantization into CHW int8 destination.
 *
 * @param src_bgr      Pointer to source BGR image (HWC uint8)
 * @param src_w        Source image width in pixels
 * @param src_h        Source image height in pixels
 * @param src_stride   Source image stride in bytes (typically src_w * 3)
 * @param dst_chw      Pointer to destination buffer (CHW int8, size 3 * dst_w * dst_h)
 * @param dst_w        Target network width (e.g., 640)
 * @param dst_h        Target network height (e.g., 640)
 * @param out_pad_top  Output pointer for top padding pixels
 * @param out_pad_left Output pointer for left padding pixels
 * @param out_scale    Output pointer for aspect ratio scale factor
 * @return 0 on success; -1 null pointer or non-positive size; -2 coordinate
 *         table allocation failed (targets wider than 1024 px use the heap);
 *         -3 src_stride shorter than src_w * 3.
 */
PREPROCESS_API int fused_preprocess_bgr_to_chw_int8(
    const uint8_t* __restrict src_bgr,
    int src_w,
    int src_h,
    int src_stride,
    int8_t* __restrict dst_chw,
    int dst_w,
    int dst_h,
    int* out_pad_top,
    int* out_pad_left,
    float* out_scale
) {
    if (!src_bgr || !dst_chw || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return -1;
    }
    // A row is read up to byte (src_w - 1) * 3 + 2; a shorter stride would
    // pull the tail of every row from the next row's memory.
    if (src_stride < src_w * 3) {
        return -3;
    }

    // 1. Calculate letterbox dimensions
    float scale_w = (float)dst_w / (float)src_w;
    float scale_h = (float)dst_h / (float)src_h;
    float scale = (scale_w < scale_h) ? scale_w : scale_h;

    int nw = (int)floorf(src_w * scale + 0.5f);
    int nh = (int)floorf(src_h * scale + 0.5f);
    if (nw > dst_w) nw = dst_w;
    if (nh > dst_h) nh = dst_h;

    int pad_top = (dst_h - nh) / 2;
    int pad_left = (dst_w - nw) / 2;

    if (out_pad_top) *out_pad_top = pad_top;
    if (out_pad_left) *out_pad_left = pad_left;
    if (out_scale) *out_scale = scale;

    const int plane_size = dst_w * dst_h;
    int8_t* __restrict dst_r_plane = dst_chw;
    int8_t* __restrict dst_g_plane = dst_chw + plane_size;
    int8_t* __restrict dst_b_plane = dst_chw + (plane_size * 2);

    // 2. Initialize entire canvas to pad value (-14) across parallel threads
#ifdef _OPENMP
    #pragma omp parallel sections
    {
        #pragma omp section
        memset(dst_r_plane, PAD_VALUE_INT8, (size_t)plane_size);
        #pragma omp section
        memset(dst_g_plane, PAD_VALUE_INT8, (size_t)plane_size);
        #pragma omp section
        memset(dst_b_plane, PAD_VALUE_INT8, (size_t)plane_size);
    }
#else
    memset(dst_chw, PAD_VALUE_INT8, (size_t)plane_size * 3);
#endif

    // 3. Precompute 1D horizontal interpolation table. nw never exceeds dst_w,
    //    so stack storage covers every target width up to 1024 px; wider
    //    targets use a heap table instead of being refused.
    XCoordTable x_tab_stack[1024];
    XCoordTable* x_tab = x_tab_stack;
    XCoordTable* x_tab_heap = NULL;
    if (nw > 1024) {
        x_tab_heap = (XCoordTable*)malloc((size_t)nw * sizeof(XCoordTable));
        if (!x_tab_heap) return -2;
        x_tab = x_tab_heap;
    }

    float fx = (float)src_w / (float)nw;
    for (int x = 0; x < nw; ++x) {
        float sx = (x + 0.5f) * fx - 0.5f;
        int x0 = (int)floorf(sx);
        if (x0 < 0) x0 = 0;
        int x1 = x0 + 1;
        if (x1 >= src_w) x1 = src_w - 1;

        float alpha = sx - (float)x0;
        if (alpha < 0.0f) alpha = 0.0f;
        if (alpha > 1.0f) alpha = 1.0f;

        int bx1 = (int)floorf(alpha * 2048.0f + 0.5f);
        int bx0 = 2048 - bx1;

        x_tab[x].x0 = x0;
        x_tab[x].x1 = x1;
        x_tab[x].bx0 = bx0;
        x_tab[x].bx1 = bx1;
        x_tab[x].off0 = x0 * 3;
        x_tab[x].off1 = x1 * 3;
    }

#if defined(HAVE_AVX2)
    int n_pairs = nw / 2;
    __declspec(align(32)) __m256i w_bx0_stack[512];
    __declspec(align(32)) __m256i w_bx1_stack[512];
    __m256i* w_bx0 = w_bx0_stack;
    __m256i* w_bx1 = w_bx1_stack;
    __m256i* w_bx0_heap = NULL;
    __m256i* w_bx1_heap = NULL;

    if (n_pairs > 512) {
        w_bx0_heap = (__m256i*)_mm_malloc((size_t)(n_pairs + 1) * sizeof(__m256i), 32);
        w_bx1_heap = (__m256i*)_mm_malloc((size_t)(n_pairs + 1) * sizeof(__m256i), 32);
        if (w_bx0_heap && w_bx1_heap) {
            w_bx0 = w_bx0_heap;
            w_bx1 = w_bx1_heap;
        }
    }

    for (int p = 0; p < n_pairs; ++p) {
        int xA = p * 2;
        int xB = xA + 1;
        w_bx0[p] = _mm256_set_epi32(0, x_tab[xB].bx0, x_tab[xB].bx0, x_tab[xB].bx0,
                                    0, x_tab[xA].bx0, x_tab[xA].bx0, x_tab[xA].bx0);
        w_bx1[p] = _mm256_set_epi32(0, x_tab[xB].bx1, x_tab[xB].bx1, x_tab[xB].bx1,
                                    0, x_tab[xA].bx1, x_tab[xA].bx1, x_tab[xA].bx1);
    }
    const __m256i v_1024 = _mm256_set1_epi32(1024);
#endif

    // 4. Precompute 1D vertical interpolation table (Y coords and Q11 weights)
    float fy = (float)src_h / (float)nh;

    // 5. Parallel row processing across CPU threads via OpenMP
    int y;
#pragma omp parallel for schedule(static)
    for (y = 0; y < nh; ++y) {
        float sy = (y + 0.5f) * fy - 0.5f;
        int y0 = (int)floorf(sy);
        if (y0 < 0) y0 = 0;
        int y1 = y0 + 1;
        if (y1 >= src_h) y1 = src_h - 1;

        float beta = sy - (float)y0;
        if (beta < 0.0f) beta = 0.0f;
        if (beta > 1.0f) beta = 1.0f;

        int by1 = (int)floorf(beta * 2048.0f + 0.5f);
        int by0 = 2048 - by1;

        const uint8_t* __restrict row0 = src_bgr + ((size_t)y0 * src_stride);
        const uint8_t* __restrict row1 = src_bgr + ((size_t)y1 * src_stride);

        int out_y = pad_top + y;
        size_t row_offset = (size_t)out_y * dst_w + pad_left;
        int8_t* __restrict out_r = dst_r_plane + row_offset;
        int8_t* __restrict out_g = dst_g_plane + row_offset;
        int8_t* __restrict out_b = dst_b_plane + row_offset;

#if defined(HAVE_AVX2)
        __m256i v_by0 = _mm256_set1_epi32(by0);
        __m256i v_by1 = _mm256_set1_epi32(by1);

        int p = 0;
        for (; p < n_pairs; ++p) {
            int xA = p * 2;
            int xB = xA + 1;

            int off0_A = x_tab[xA].off0;
            int off1_A = x_tab[xA].off1;
            int off0_B = x_tab[xB].off0;
            int off1_B = x_tab[xB].off1;

            uint32_t raw_p00_A = *(const uint32_t*)(row0 + off0_A);
            uint32_t raw_p00_B = *(const uint32_t*)(row0 + off0_B);
            uint32_t raw_p01_A = *(const uint32_t*)(row0 + off1_A);
            uint32_t raw_p01_B = *(const uint32_t*)(row0 + off1_B);

            uint32_t raw_p10_A = *(const uint32_t*)(row1 + off0_A);
            uint32_t raw_p10_B = *(const uint32_t*)(row1 + off0_B);
            uint32_t raw_p11_A = *(const uint32_t*)(row1 + off1_A);
            uint32_t raw_p11_B = *(const uint32_t*)(row1 + off1_B);

            __m256i P00 = _mm256_set_m128i(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p00_B)),
                                          _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p00_A)));
            __m256i P01 = _mm256_set_m128i(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p01_B)),
                                          _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p01_A)));
            __m256i P10 = _mm256_set_m128i(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p10_B)),
                                          _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p10_A)));
            __m256i P11 = _mm256_set_m128i(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p11_B)),
                                          _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p11_A)));

            __m256i bx0 = w_bx0[p];
            __m256i bx1 = w_bx1[p];

            __m256i r0 = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(P00, bx0),
                                                  _mm256_mullo_epi32(P01, bx1)), v_1024), 11);
            __m256i r1 = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(P10, bx0),
                                                  _mm256_mullo_epi32(P11, bx1)), v_1024), 11);

            __m256i v = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(r0, v_by0),
                                                  _mm256_mullo_epi32(r1, v_by1)), v_1024), 11);

            int32_t buf[8];
            _mm256_storeu_si256((__m256i*)buf, v);

            out_r[xA] = (int8_t)(buf[2] - 128);
            out_g[xA] = (int8_t)(buf[1] - 128);
            out_b[xA] = (int8_t)(buf[0] - 128);

            out_r[xB] = (int8_t)(buf[6] - 128);
            out_g[xB] = (int8_t)(buf[5] - 128);
            out_b[xB] = (int8_t)(buf[4] - 128);
        }
        int x_start = p * 2;
#else
        int x_start = 0;
#endif
        for (int x = x_start; x < nw; ++x) {
            const XCoordTable* t = &x_tab[x];
            const uint8_t* p00 = row0 + t->off0;
            const uint8_t* p01 = row0 + t->off1;
            const uint8_t* p10 = row1 + t->off0;
            const uint8_t* p11 = row1 + t->off1;

            int r0_b = (p00[0] * t->bx0 + p01[0] * t->bx1 + 1024) >> 11;
            int r1_b = (p10[0] * t->bx0 + p11[0] * t->bx1 + 1024) >> 11;
            int v_b = (r0_b * by0 + r1_b * by1 + 1024) >> 11;

            int r0_g = (p00[1] * t->bx0 + p01[1] * t->bx1 + 1024) >> 11;
            int r1_g = (p10[1] * t->bx0 + p11[1] * t->bx1 + 1024) >> 11;
            int v_g = (r0_g * by0 + r1_g * by1 + 1024) >> 11;

            int r0_r = (p00[2] * t->bx0 + p01[2] * t->bx1 + 1024) >> 11;
            int r1_r = (p10[2] * t->bx0 + p11[2] * t->bx1 + 1024) >> 11;
            int v_r = (r0_r * by0 + r1_r * by1 + 1024) >> 11;

            out_r[x] = (int8_t)(v_r - 128);
            out_g[x] = (int8_t)(v_g - 128);
            out_b[x] = (int8_t)(v_b - 128);
        }
    }

#if defined(HAVE_AVX2)
    if (w_bx0_heap) _mm_free(w_bx0_heap);
    if (w_bx1_heap) _mm_free(w_bx1_heap);
#endif
    free(x_tab_heap);
    return 0;
}

/**
 * Fused letterbox + bilinear resize + model input quantization straight into the
 * graph engine's input plane (the DMA-visible workspace buffer).
 *
 * The plane is the channel-blocked layout [dst_h + 2 halo][dst_w + 2 halo][8]
 * uint8 with R, G, B in channels 0, 1, 2. Only the interior of channels 0..2 is
 * written; the halo ring and channels 3..7 keep what the caller placed there
 * once (the zero point). Every interior value is lut[pixel], where pixel is the
 * same Q11 bilinear value fused_preprocess_bgr_to_chw_int8 produces (its int8
 * output plus 128) and letterbox padding is pixel 114, so the plane is
 * byte-identical to quantizing that function's output with the same table.
 *
 * @return 0 on success; -1 null pointer or non-positive size; -2 coordinate
 *         table allocation failed; -3 src_stride shorter than src_w * 3.
 */
PREPROCESS_API int fused_preprocess_bgr_to_c8_plane(
    const uint8_t* __restrict src_bgr,
    int src_w,
    int src_h,
    int src_stride,
    uint8_t* __restrict dst_plane,
    int dst_w,
    int dst_h,
    int halo,
    const uint8_t* __restrict lut,
    int* out_pad_top,
    int* out_pad_left,
    float* out_scale
) {
    if (!src_bgr || !dst_plane || !lut || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 || halo < 0) {
        return -1;
    }
    if (src_stride < src_w * 3) {
        return -3;
    }

    float scale_w = (float)dst_w / (float)src_w;
    float scale_h = (float)dst_h / (float)src_h;
    float scale = (scale_w < scale_h) ? scale_w : scale_h;
    int nw = (int)floorf(src_w * scale + 0.5f);
    int nh = (int)floorf(src_h * scale + 0.5f);
    if (nw > dst_w) nw = dst_w;
    if (nh > dst_h) nh = dst_h;
    int pad_top = (dst_h - nh) / 2;
    int pad_left = (dst_w - nw) / 2;
    if (out_pad_top) *out_pad_top = pad_top;
    if (out_pad_left) *out_pad_left = pad_left;
    if (out_scale) *out_scale = scale;

    const size_t pitch = (size_t)(dst_w + 2 * halo) * 8;
    const uint8_t pad_q = lut[114];

    if (nw == src_w && nh == src_h) {
        // Identity resize (a 640x480 camera into a 640 input): the Q11 weights are (2048, 0)
        // on both axes, so every bilinear value is the source pixel and a lookup copy
        // produces the same bytes as the general path.
        int yi;
#pragma omp parallel for schedule(static)
        for (yi = 0; yi < dst_h; ++yi) {
            uint8_t* __restrict row = dst_plane + (size_t)(yi + halo) * pitch + (size_t)halo * 8;
            int iy = yi - pad_top;
            if (iy < 0 || iy >= nh) {
                for (int x = 0; x < dst_w; ++x) {
                    uint8_t* p = row + (size_t)x * 8;
                    p[0] = pad_q; p[1] = pad_q; p[2] = pad_q;
                }
                continue;
            }
            for (int x = 0; x < pad_left; ++x) {
                uint8_t* p = row + (size_t)x * 8;
                p[0] = pad_q; p[1] = pad_q; p[2] = pad_q;
            }
            for (int x = pad_left + nw; x < dst_w; ++x) {
                uint8_t* p = row + (size_t)x * 8;
                p[0] = pad_q; p[1] = pad_q; p[2] = pad_q;
            }
            const uint8_t* __restrict s = src_bgr + (size_t)iy * src_stride;
            uint8_t* __restrict out = row + (size_t)pad_left * 8;
            for (int x = 0; x < nw; ++x) {
                const uint8_t* q = s + (size_t)x * 3;
                uint8_t* p = out + (size_t)x * 8;
                p[0] = lut[q[2]];
                p[1] = lut[q[1]];
                p[2] = lut[q[0]];
            }
        }
        return 0;
    }

    XCoordTable x_tab_stack[1024];
    XCoordTable* x_tab = x_tab_stack;
    XCoordTable* x_tab_heap = NULL;
    if (nw > 1024) {
        x_tab_heap = (XCoordTable*)malloc((size_t)nw * sizeof(XCoordTable));
        if (!x_tab_heap) return -2;
        x_tab = x_tab_heap;
    }
    float fx = (float)src_w / (float)nw;
    for (int x = 0; x < nw; ++x) {
        float sx = (x + 0.5f) * fx - 0.5f;
        int x0 = (int)floorf(sx);
        if (x0 < 0) x0 = 0;
        int x1 = x0 + 1;
        if (x1 >= src_w) x1 = src_w - 1;
        float alpha = sx - (float)x0;
        if (alpha < 0.0f) alpha = 0.0f;
        if (alpha > 1.0f) alpha = 1.0f;
        int bx1 = (int)floorf(alpha * 2048.0f + 0.5f);
        x_tab[x].x0 = x0;
        x_tab[x].x1 = x1;
        x_tab[x].bx0 = 2048 - bx1;
        x_tab[x].bx1 = bx1;
        x_tab[x].off0 = x0 * 3;
        x_tab[x].off1 = x1 * 3;
    }

#if defined(HAVE_AVX2)
    int n_pairs = nw / 2;
    __declspec(align(32)) __m256i w_bx0_stack[512];
    __declspec(align(32)) __m256i w_bx1_stack[512];
    __m256i* w_bx0 = w_bx0_stack;
    __m256i* w_bx1 = w_bx1_stack;
    __m256i* w_bx0_heap = NULL;
    __m256i* w_bx1_heap = NULL;

    if (n_pairs > 512) {
        w_bx0_heap = (__m256i*)_mm_malloc((size_t)(n_pairs + 1) * sizeof(__m256i), 32);
        w_bx1_heap = (__m256i*)_mm_malloc((size_t)(n_pairs + 1) * sizeof(__m256i), 32);
        if (w_bx0_heap && w_bx1_heap) {
            w_bx0 = w_bx0_heap;
            w_bx1 = w_bx1_heap;
        }
    }

    for (int p = 0; p < n_pairs; ++p) {
        int xA = p * 2;
        int xB = xA + 1;
        w_bx0[p] = _mm256_set_epi32(0, x_tab[xB].bx0, x_tab[xB].bx0, x_tab[xB].bx0,
                                    0, x_tab[xA].bx0, x_tab[xA].bx0, x_tab[xA].bx0);
        w_bx1[p] = _mm256_set_epi32(0, x_tab[xB].bx1, x_tab[xB].bx1, x_tab[xB].bx1,
                                    0, x_tab[xA].bx1, x_tab[xA].bx1, x_tab[xA].bx1);
    }

    const __m256i v_1024 = _mm256_set1_epi32(1024);
    const __m256i v_pad_val = _mm256_set1_epi8((char)pad_q);
    const __m256i v_pad_mask = _mm256_setr_epi8(
        (char)0xFF, (char)0xFF, (char)0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
        (char)0xFF, (char)0xFF, (char)0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
        (char)0xFF, (char)0xFF, (char)0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
        (char)0xFF, (char)0xFF, (char)0xFF, 0x00, 0x00, 0x00, 0x00, 0x00
    );
#endif

    float fy = (float)src_h / (float)nh;

    int y;
#pragma omp parallel for schedule(static)
    for (y = 0; y < dst_h; ++y) {
        uint8_t* __restrict row = dst_plane + (size_t)(y + halo) * pitch + (size_t)halo * 8;
        int iy = y - pad_top;
        if (iy < 0 || iy >= nh) {
#if defined(HAVE_AVX2)
            int x = 0;
            for (; x + 3 < dst_w; x += 4) {
                uint8_t* p = row + (size_t)x * 8;
                __m256i orig = _mm256_loadu_si256((const __m256i*)p);
                __m256i blended = _mm256_blendv_epi8(orig, v_pad_val, v_pad_mask);
                _mm256_storeu_si256((__m256i*)p, blended);
            }
            for (; x < dst_w; ++x) {
                uint8_t* p = row + (size_t)x * 8;
                p[0] = pad_q; p[1] = pad_q; p[2] = pad_q;
            }
#else
            for (int x = 0; x < dst_w; ++x) {
                uint8_t* p = row + (size_t)x * 8;
                p[0] = pad_q; p[1] = pad_q; p[2] = pad_q;
            }
#endif
            continue;
        }
#if defined(HAVE_AVX2)
        int xl = 0;
        for (; xl + 3 < pad_left; xl += 4) {
            uint8_t* p = row + (size_t)xl * 8;
            __m256i orig = _mm256_loadu_si256((const __m256i*)p);
            __m256i blended = _mm256_blendv_epi8(orig, v_pad_val, v_pad_mask);
            _mm256_storeu_si256((__m256i*)p, blended);
        }
        for (; xl < pad_left; ++xl) {
            uint8_t* p = row + (size_t)xl * 8;
            p[0] = pad_q; p[1] = pad_q; p[2] = pad_q;
        }
        int xr = pad_left + nw;
        for (; xr + 3 < dst_w; xr += 4) {
            uint8_t* p = row + (size_t)xr * 8;
            __m256i orig = _mm256_loadu_si256((const __m256i*)p);
            __m256i blended = _mm256_blendv_epi8(orig, v_pad_val, v_pad_mask);
            _mm256_storeu_si256((__m256i*)p, blended);
        }
        for (; xr < dst_w; ++xr) {
            uint8_t* p = row + (size_t)xr * 8;
            p[0] = pad_q; p[1] = pad_q; p[2] = pad_q;
        }
#else
        for (int x = 0; x < pad_left; ++x) {
            uint8_t* p = row + (size_t)x * 8;
            p[0] = pad_q; p[1] = pad_q; p[2] = pad_q;
        }
        for (int x = pad_left + nw; x < dst_w; ++x) {
            uint8_t* p = row + (size_t)x * 8;
            p[0] = pad_q; p[1] = pad_q; p[2] = pad_q;
        }
#endif
        float sy = (iy + 0.5f) * fy - 0.5f;
        int y0 = (int)floorf(sy);
        if (y0 < 0) y0 = 0;
        int y1 = y0 + 1;
        if (y1 >= src_h) y1 = src_h - 1;
        float beta = sy - (float)y0;
        if (beta < 0.0f) beta = 0.0f;
        if (beta > 1.0f) beta = 1.0f;
        int by1 = (int)floorf(beta * 2048.0f + 0.5f);
        int by0 = 2048 - by1;
        const uint8_t* __restrict row0 = src_bgr + ((size_t)y0 * src_stride);
        const uint8_t* __restrict row1 = src_bgr + ((size_t)y1 * src_stride);
        uint8_t* __restrict out = row + (size_t)pad_left * 8;

#if defined(HAVE_AVX2)
        __m256i v_by0 = _mm256_set1_epi32(by0);
        __m256i v_by1 = _mm256_set1_epi32(by1);

        int p = 0;
        for (; p < n_pairs; ++p) {
            int xA = p * 2;
            int xB = xA + 1;

            int off0_A = x_tab[xA].off0;
            int off1_A = x_tab[xA].off1;
            int off0_B = x_tab[xB].off0;
            int off1_B = x_tab[xB].off1;

            uint32_t raw_p00_A = *(const uint32_t*)(row0 + off0_A);
            uint32_t raw_p00_B = *(const uint32_t*)(row0 + off0_B);
            uint32_t raw_p01_A = *(const uint32_t*)(row0 + off1_A);
            uint32_t raw_p01_B = *(const uint32_t*)(row0 + off1_B);

            uint32_t raw_p10_A = *(const uint32_t*)(row1 + off0_A);
            uint32_t raw_p10_B = *(const uint32_t*)(row1 + off0_B);
            uint32_t raw_p11_A = *(const uint32_t*)(row1 + off1_A);
            uint32_t raw_p11_B = *(const uint32_t*)(row1 + off1_B);

            __m256i P00 = _mm256_set_m128i(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p00_B)),
                                          _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p00_A)));
            __m256i P01 = _mm256_set_m128i(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p01_B)),
                                          _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p01_A)));
            __m256i P10 = _mm256_set_m128i(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p10_B)),
                                          _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p10_A)));
            __m256i P11 = _mm256_set_m128i(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p11_B)),
                                          _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p11_A)));

            __m256i bx0 = w_bx0[p];
            __m256i bx1 = w_bx1[p];

            __m256i r0 = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(P00, bx0),
                                                  _mm256_mullo_epi32(P01, bx1)), v_1024), 11);
            __m256i r1 = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(P10, bx0),
                                                  _mm256_mullo_epi32(P11, bx1)), v_1024), 11);

            __m256i v = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(r0, v_by0),
                                                  _mm256_mullo_epi32(r1, v_by1)), v_1024), 11);

            int32_t buf[8];
            _mm256_storeu_si256((__m256i*)buf, v);

            uint8_t* pA = out + (size_t)xA * 8;
            pA[0] = lut[buf[2]];
            pA[1] = lut[buf[1]];
            pA[2] = lut[buf[0]];

            uint8_t* pB = out + (size_t)xB * 8;
            pB[0] = lut[buf[6]];
            pB[1] = lut[buf[5]];
            pB[2] = lut[buf[4]];
        }
        int x_start = p * 2;
#else
        int x_start = 0;
#endif
        for (int x = x_start; x < nw; ++x) {
            const XCoordTable* t = &x_tab[x];
            const uint8_t* p00 = row0 + t->off0;
            const uint8_t* p01 = row0 + t->off1;
            const uint8_t* p10 = row1 + t->off0;
            const uint8_t* p11 = row1 + t->off1;
            int v_b = ((((p00[0] * t->bx0 + p01[0] * t->bx1 + 1024) >> 11) * by0 +
                        ((p10[0] * t->bx0 + p11[0] * t->bx1 + 1024) >> 11) * by1 + 1024) >> 11);
            int v_g = ((((p00[1] * t->bx0 + p01[1] * t->bx1 + 1024) >> 11) * by0 +
                        ((p10[1] * t->bx0 + p11[1] * t->bx1 + 1024) >> 11) * by1 + 1024) >> 11);
            int v_r = ((((p00[2] * t->bx0 + p01[2] * t->bx1 + 1024) >> 11) * by0 +
                        ((p10[2] * t->bx0 + p11[2] * t->bx1 + 1024) >> 11) * by1 + 1024) >> 11);
            uint8_t* p = out + (size_t)x * 8;
            p[0] = lut[v_r];
            p[1] = lut[v_g];
            p[2] = lut[v_b];
        }
    }

#if defined(HAVE_AVX2)
    if (w_bx0_heap) _mm_free(w_bx0_heap);
    if (w_bx1_heap) _mm_free(w_bx1_heap);
#endif
    free(x_tab_heap);
    return 0;
}

/**
 * Channel-blocked uint8 tensor [blocks][h][w][8] (zero point 128) -> int8 NCHW
 * [channels][h][w] (zero point 0) by flipping the top bit, channel c read from
 * block c / 8, byte c % 8 of every pixel. One pass per channel, parallel over
 * channels.
 *
 * @return 0 on success; -1 null pointer, non-positive size or channels > 8 * blocks.
 */
PREPROCESS_API int c8_blocks_to_nchw_int8(
    const uint8_t* __restrict src,
    int blocks,
    int h,
    int w,
    int channels,
    int8_t* __restrict dst
) {
    if (!src || !dst || blocks <= 0 || h <= 0 || w <= 0 || channels <= 0 || channels > blocks * 8) {
        return -1;
    }
    const size_t hw = (size_t)h * (size_t)w;
    int c;
#pragma omp parallel for schedule(static)
    for (c = 0; c < channels; ++c) {
        const uint8_t* __restrict s = src + (size_t)(c / 8) * hw * 8 + (size_t)(c % 8);
        int8_t* __restrict d = dst + (size_t)c * hw;
        for (size_t i = 0; i < hw; ++i) {
            d[i] = (int8_t)(s[i * 8] ^ 0x80);
        }
    }
    return 0;
}

/**
 * Per-pixel maximum over the first ``channels`` channels of a channel-blocked uint8
 * tensor [blocks][h][w][8] (zero point 128), as int8 with zero point 0: max_out[y * w + x]
 * equals the maximum over c of (src channel c at (y, x)) ^ 0x80, the value the int8 NCHW
 * view of the same tensor holds. A detect head's class-logit maximum per anchor, computed
 * row by row in the worker threads that just wrote the egress, so the decoder's
 * confidence prune reads h * w bytes instead of the whole class tensor.
 *
 * @return 0 on success; -1 null pointer, non-positive size or channels > 8 * blocks.
 */
PREPROCESS_API int c8_blocks_class_max_int8(
    const uint8_t* __restrict src,
    int blocks,
    int h,
    int w,
    int channels,
    int8_t* __restrict max_out
) {
    if (!src || !max_out || blocks <= 0 || h <= 0 || w <= 0 || channels <= 0 || channels > blocks * 8) {
        return -1;
    }
    const size_t hw = (size_t)h * (size_t)w;
    int y;
#pragma omp parallel for schedule(static)
    for (y = 0; y < h; ++y) {
        int8_t* __restrict m = max_out + (size_t)y * w;
        for (int x = 0; x < w; ++x) m[x] = -128;
        for (int b = 0; b < blocks; ++b) {
            int nk = channels - 8 * b;
            if (nk <= 0) break;
            if (nk > 8) nk = 8;
            const uint8_t* __restrict row = src + (size_t)b * hw * 8 + (size_t)y * w * 8;
            for (int x = 0; x < w; ++x) {
                const uint8_t* p = row + (size_t)x * 8;
                int best = m[x];
                for (int k = 0; k < nk; ++k) {
                    int v = (int)(int8_t)(p[k] ^ 0x80);
                    if (v > best) best = v;
                }
                m[x] = (int8_t)best;
            }
        }
    }
    return 0;
}

/**
 * Fast DepthToSpace (CRD mode, bs=2) + LUT dequantization for super-resolution (SESR M7):
 * Input is channel-blocked uint8 [2][h][w][8], 12 active channels.
 * Output is upscaled BGR image uint8 [2*h][2*w][3].
 */
PREPROCESS_API int depth_to_space_crd_bgr(
    const uint8_t* __restrict src,
    int h,
    int w,
    const uint8_t* __restrict lut,
    uint8_t* __restrict dst
) {
    if (!src || !lut || !dst || h <= 0 || w <= 0) {
        return -1;
    }
    const size_t hw = (size_t)h * (size_t)w;
    const uint8_t* __restrict b0 = src;
    const uint8_t* __restrict b1 = src + hw * 8;
    const int out_stride = w * 2 * 3;

    for (int y = 0; y < h; ++y) {
        uint8_t* __restrict row_top = dst + (size_t)(2 * y) * out_stride;
        uint8_t* __restrict row_bot = dst + (size_t)(2 * y + 1) * out_stride;
        const uint8_t* __restrict p0 = b0 + (size_t)y * w * 8;
        const uint8_t* __restrict p1 = b1 + (size_t)y * w * 8;

        for (int x = 0; x < w; ++x) {
            uint8_t p0_0 = p0[0], p0_1 = p0[1], p0_2 = p0[2], p0_3 = p0[3];
            uint8_t p0_4 = p0[4], p0_5 = p0[5], p0_6 = p0[6], p0_7 = p0[7];
            uint8_t p1_0 = p1[0], p1_1 = p1[1], p1_2 = p1[2], p1_3 = p1[3];
            p0 += 8;
            p1 += 8;

            row_top[0] = lut[p1_0];
            row_top[1] = lut[p0_4];
            row_top[2] = lut[p0_0];
            row_top[3] = lut[p1_1];
            row_top[4] = lut[p0_5];
            row_top[5] = lut[p0_1];
            row_top += 6;

            row_bot[0] = lut[p1_2];
            row_bot[1] = lut[p0_6];
            row_bot[2] = lut[p0_2];
            row_bot[3] = lut[p1_3];
            row_bot[4] = lut[p0_7];
            row_bot[5] = lut[p0_3];
            row_bot += 6;
        }
    }
    return 0;
}

/**
 * Fused direct bilinear resize + BGR-to-RGB + input quantization straight into
 * the dense graph engine's input plane (e.g. for SESR super-resolution).
 * No letterboxing or aspect-ratio padding is performed; the image is scaled to (dst_w, dst_h).
 *
 * @param src_bgr      Pointer to source BGR image (HWC uint8)
 * @param src_w        Source image width in pixels
 * @param src_h        Source image height in pixels
 * @param src_stride   Source image stride in bytes (typically src_w * 3)
 * @param dst_plane    Pointer to destination input plane [dst_h + 2*halo][dst_w + 2*halo][8] uint8
 * @param dst_w        Target network width (e.g. 256)
 * @param dst_h        Target network height (e.g. 256)
 * @param halo         Halo border size in pixels (e.g. 1)
 * @param lut          Optional 256-byte LUT (uint8); if NULL, pixel values are written unmapped
 * @return 0 on success; negative error code on failure.
 */
PREPROCESS_API int fused_resize_bgr_to_c8_plane(
    const uint8_t* __restrict src_bgr,
    int src_w,
    int src_h,
    int src_stride,
    uint8_t* __restrict dst_plane,
    int dst_w,
    int dst_h,
    int halo,
    const uint8_t* __restrict lut
) {
    if (!src_bgr || !dst_plane || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 || halo < 0) {
        return -1;
    }
    if (src_stride < src_w * 3) {
        return -3;
    }

    const size_t pitch = (size_t)(dst_w + 2 * halo) * 8;

    if (src_w == dst_w && src_h == dst_h) {
        int y;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (y = 0; y < dst_h; ++y) {
            const uint8_t* __restrict src_row = src_bgr + ((size_t)y * src_stride);
            uint8_t* __restrict dst_row = dst_plane + (size_t)(y + halo) * pitch + (size_t)halo * 8;
            for (int x = 0; x < dst_w; ++x) {
                uint8_t b = src_row[x * 3 + 0];
                uint8_t g = src_row[x * 3 + 1];
                uint8_t r = src_row[x * 3 + 2];
                uint8_t* p = dst_row + (size_t)x * 8;
                p[0] = lut ? lut[r] : r;
                p[1] = lut ? lut[g] : g;
                p[2] = lut ? lut[b] : b;
            }
        }
        return 0;
    }

    // Precompute 1D horizontal table
    XCoordTable x_tab_stack[1024];
    XCoordTable* x_tab = x_tab_stack;
    XCoordTable* x_tab_heap = NULL;
    if (dst_w > 1024) {
        x_tab_heap = (XCoordTable*)malloc((size_t)dst_w * sizeof(XCoordTable));
        if (!x_tab_heap) return -2;
        x_tab = x_tab_heap;
    }

    float fx = (float)src_w / (float)dst_w;
    for (int x = 0; x < dst_w; ++x) {
        float sx = (x + 0.5f) * fx - 0.5f;
        int x0 = (int)floorf(sx);
        if (x0 < 0) x0 = 0;
        int x1 = x0 + 1;
        if (x1 >= src_w) x1 = src_w - 1;

        float alpha = sx - (float)x0;
        if (alpha < 0.0f) alpha = 0.0f;
        if (alpha > 1.0f) alpha = 1.0f;

        int bx1 = (int)floorf(alpha * 2048.0f + 0.5f);
        int bx0 = 2048 - bx1;

        x_tab[x].x0 = x0;
        x_tab[x].x1 = x1;
        x_tab[x].bx0 = bx0;
        x_tab[x].bx1 = bx1;
        x_tab[x].off0 = x0 * 3;
        x_tab[x].off1 = x1 * 3;
    }

#if defined(HAVE_AVX2)
    int n_pairs = dst_w / 2;
    __declspec(align(32)) __m256i w_bx0_stack[512];
    __declspec(align(32)) __m256i w_bx1_stack[512];
    __m256i* w_bx0 = w_bx0_stack;
    __m256i* w_bx1 = w_bx1_stack;
    __m256i* w_bx0_heap = NULL;
    __m256i* w_bx1_heap = NULL;

    if (n_pairs > 512) {
        w_bx0_heap = (__m256i*)_mm_malloc((size_t)(n_pairs + 1) * sizeof(__m256i), 32);
        w_bx1_heap = (__m256i*)_mm_malloc((size_t)(n_pairs + 1) * sizeof(__m256i), 32);
        if (w_bx0_heap && w_bx1_heap) {
            w_bx0 = w_bx0_heap;
            w_bx1 = w_bx1_heap;
        }
    }

    for (int p = 0; p < n_pairs; ++p) {
        int xA = p * 2;
        int xB = xA + 1;
        w_bx0[p] = _mm256_set_epi32(0, x_tab[xB].bx0, x_tab[xB].bx0, x_tab[xB].bx0,
                                    0, x_tab[xA].bx0, x_tab[xA].bx0, x_tab[xA].bx0);
        w_bx1[p] = _mm256_set_epi32(0, x_tab[xB].bx1, x_tab[xB].bx1, x_tab[xB].bx1,
                                    0, x_tab[xA].bx1, x_tab[xA].bx1, x_tab[xA].bx1);
    }
    const __m256i v_1024 = _mm256_set1_epi32(1024);
#endif

    float fy = (float)src_h / (float)dst_h;
    int y;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (y = 0; y < dst_h; ++y) {
        float sy = (y + 0.5f) * fy - 0.5f;
        int y0 = (int)floorf(sy);
        if (y0 < 0) y0 = 0;
        int y1 = y0 + 1;
        if (y1 >= src_h) y1 = src_h - 1;

        float beta = sy - (float)y0;
        if (beta < 0.0f) beta = 0.0f;
        if (beta > 1.0f) beta = 1.0f;

        int by1 = (int)floorf(beta * 2048.0f + 0.5f);
        int by0 = 2048 - by1;

        const uint8_t* __restrict row0 = src_bgr + ((size_t)y0 * src_stride);
        const uint8_t* __restrict row1 = src_bgr + ((size_t)y1 * src_stride);
        uint8_t* __restrict dst_row = dst_plane + (size_t)(y + halo) * pitch + (size_t)halo * 8;

#if defined(HAVE_AVX2)
        __m256i v_by0 = _mm256_set1_epi32(by0);
        __m256i v_by1 = _mm256_set1_epi32(by1);

        int p = 0;
        for (; p < n_pairs; ++p) {
            int xA = p * 2;
            int xB = xA + 1;

            int off0_A = x_tab[xA].off0;
            int off1_A = x_tab[xA].off1;
            int off0_B = x_tab[xB].off0;
            int off1_B = x_tab[xB].off1;

            uint32_t raw_p00_A = *(const uint32_t*)(row0 + off0_A);
            uint32_t raw_p00_B = *(const uint32_t*)(row0 + off0_B);
            uint32_t raw_p01_A = *(const uint32_t*)(row0 + off1_A);
            uint32_t raw_p01_B = *(const uint32_t*)(row0 + off1_B);

            uint32_t raw_p10_A = *(const uint32_t*)(row1 + off0_A);
            uint32_t raw_p10_B = *(const uint32_t*)(row1 + off0_B);
            uint32_t raw_p11_A = *(const uint32_t*)(row1 + off1_A);
            uint32_t raw_p11_B = *(const uint32_t*)(row1 + off1_B);

            __m256i P00 = _mm256_set_m128i(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p00_B)),
                                          _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p00_A)));
            __m256i P01 = _mm256_set_m128i(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p01_B)),
                                          _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p01_A)));
            __m256i P10 = _mm256_set_m128i(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p10_B)),
                                          _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p10_A)));
            __m256i P11 = _mm256_set_m128i(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p11_B)),
                                          _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw_p11_A)));

            __m256i bx0 = w_bx0[p];
            __m256i bx1 = w_bx1[p];

            __m256i r0 = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(P00, bx0),
                                                  _mm256_mullo_epi32(P01, bx1)), v_1024), 11);
            __m256i r1 = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(P10, bx0),
                                                  _mm256_mullo_epi32(P11, bx1)), v_1024), 11);

            __m256i v = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(r0, v_by0),
                                                  _mm256_mullo_epi32(r1, v_by1)), v_1024), 11);

            int32_t buf[8];
            _mm256_storeu_si256((__m256i*)buf, v);

            uint8_t* pA = dst_row + (size_t)xA * 8;
            pA[0] = lut ? lut[buf[2]] : (uint8_t)buf[2];
            pA[1] = lut ? lut[buf[1]] : (uint8_t)buf[1];
            pA[2] = lut ? lut[buf[0]] : (uint8_t)buf[0];

            uint8_t* pB = dst_row + (size_t)xB * 8;
            pB[0] = lut ? lut[buf[6]] : (uint8_t)buf[6];
            pB[1] = lut ? lut[buf[5]] : (uint8_t)buf[5];
            pB[2] = lut ? lut[buf[4]] : (uint8_t)buf[4];
        }
        int x_start = p * 2;
#else
        int x_start = 0;
#endif
        for (int x = x_start; x < dst_w; ++x) {
            const XCoordTable* t = &x_tab[x];
            const uint8_t* p00 = row0 + t->off0;
            const uint8_t* p01 = row0 + t->off1;
            const uint8_t* p10 = row1 + t->off0;
            const uint8_t* p11 = row1 + t->off1;

            int r0_b = (p00[0] * t->bx0 + p01[0] * t->bx1 + 1024) >> 11;
            int r1_b = (p10[0] * t->bx0 + p11[0] * t->bx1 + 1024) >> 11;
            int v_b = (r0_b * by0 + r1_b * by1 + 1024) >> 11;

            int r0_g = (p00[1] * t->bx0 + p01[1] * t->bx1 + 1024) >> 11;
            int r1_g = (p10[1] * t->bx0 + p11[1] * t->bx1 + 1024) >> 11;
            int v_g = (r0_g * by0 + r1_g * by1 + 1024) >> 11;

            int r0_r = (p00[2] * t->bx0 + p01[2] * t->bx1 + 1024) >> 11;
            int r1_r = (p10[2] * t->bx0 + p11[2] * t->bx1 + 1024) >> 11;
            int v_r = (r0_r * by0 + r1_r * by1 + 1024) >> 11;

            uint8_t* p = dst_row + (size_t)x * 8;
            p[0] = lut ? lut[v_r] : (uint8_t)v_r;
            p[1] = lut ? lut[v_g] : (uint8_t)v_g;
            p[2] = lut ? lut[v_b] : (uint8_t)v_b;
        }
    }

#if defined(HAVE_AVX2)
    if (w_bx0_heap) _mm_free(w_bx0_heap);
    if (w_bx1_heap) _mm_free(w_bx1_heap);
#endif
    free(x_tab_heap);
    return 0;
}

/**
 * A BGR image into lanes 0..2 of the bf16 engine's input plane, RGB order, through a uint16 table of
 * bf16 patterns: the bf16 twin of fused_resize_bgr_to_c8_plane's table write, WITHOUT the resize.
 * dst_plane is [h + 2*halo][w + 2*halo][8] uint16; interior pixel (y, x) gets lanes 0, 1, 2 =
 * lut[R], lut[G], lut[B]. The halo ring and lanes 3..7 are not touched: the session zeroes them once,
 * and they must stay +0.0 (a zero weight times a NaN is NaN).
 *
 * No resize, on purpose. The native bilinear resize above is within one code of OpenCV's, not equal to
 * it, and a frame already at the network size needs no interpolation; a caller with another size
 * resizes with cv2 first, exactly as the numpy path does.
 *
 * @return 0 on success; -1 null pointer, non-positive size or negative halo; -3 src_stride shorter
 *         than w * 3.
 */
PREPROCESS_API int bgr_to_c8_plane_bf16(
    const uint8_t* __restrict src_bgr,
    int w,
    int h,
    int src_stride,
    uint16_t* __restrict dst_plane,
    int halo,
    const uint16_t* __restrict lut
) {
    if (!src_bgr || !dst_plane || !lut || w <= 0 || h <= 0 || halo < 0) {
        return -1;
    }
    if (src_stride < w * 3) {
        return -3;
    }
    const size_t plane_w = (size_t)w + 2 * (size_t)halo;
    for (int y = 0; y < h; ++y) {
        const uint8_t* __restrict s = src_bgr + (size_t)y * (size_t)src_stride;
        uint16_t* __restrict d = dst_plane + (((size_t)y + halo) * plane_w + (size_t)halo) * 8;
        for (int x = 0; x < w; ++x) {
            d[0] = lut[s[2]];
            d[1] = lut[s[1]];
            d[2] = lut[s[0]];
            s += 3;
            d += 8;
        }
    }
    return 0;
}

// Exponent all ones: +-Inf or NaN. On the bits, because this file builds with /fp:fast, under which
// the compiler may assume no float is ever NaN and fold an isfinite() test away.
#define BF16_NONFINITE(b) (((b) & 0x7F80u) == 0x7F80u)

static inline uint8_t bf16_pixel(uint16_t bits, float mean) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)bits << 16;  // bf16 is the top half of a float32: widening is exact
    float x = v.f + mean;        // one float32 add, as numpy's float32 image + float32(mean)
    if (x < 0.0f) x = 0.0f;
    if (x > 255.0f) x = 255.0f;
    return (uint8_t)(int)x;      // truncation, as astype(uint8) does on [0, 255]
}

/**
 * DepthToSpace (CRD, bs=2) + bf16 egress for super-resolution on the bf16 engine (SESR M7 at W8A16):
 * the bf16 twin of depth_to_space_crd_bgr. Input is channel-blocked bf16 patterns uint16 [2][h][w][8]
 * in real units, 12 active channels; output is the upscaled BGR image uint8 [2*h][2*w][3].
 *
 * Each value is the float pipeline's own postprocess (npu/sesr.py, mirrored by bf16_dense_image in
 * runtime/graph_session.py): the pattern widened to float32, plus mean in float32, clipped to [0, 255]
 * and TRUNCATED to uint8. The channel map is depth_to_space_crd_bgr's, the same 12 lanes and no
 * others; lanes 4..7 of the second block are padding and are never read.
 *
 * A non-finite value in an active channel refuses the whole tile before any pixel is written. The bf16
 * engine produces one only when a multiply-accumulate read memory nothing wrote, or when the input
 * lanes past the image's channels were not +0.0; a pixel made from it would hide that fault.
 *
 * @param out_nonfinite  Optional; receives the number of non-finite active values (0 on success).
 * @return 0 on success; -1 null pointer or non-positive size; -2 a non-finite active value (dst
 *         untouched).
 */
PREPROCESS_API int depth_to_space_crd_bgr_bf16(
    const uint16_t* __restrict src,
    int h,
    int w,
    float mean,
    uint8_t* __restrict dst,
    int64_t* out_nonfinite
) {
    if (!src || !dst || h <= 0 || w <= 0) {
        return -1;
    }
    const size_t hw = (size_t)h * (size_t)w;
    const uint16_t* __restrict b0 = src;
    const uint16_t* __restrict b1 = src + hw * 8;

    int64_t bad = 0;
    for (size_t i = 0; i < hw; ++i) {
        const uint16_t* p0 = b0 + i * 8;
        const uint16_t* p1 = b1 + i * 8;
        for (int k = 0; k < 8; ++k) bad += BF16_NONFINITE(p0[k]);
        for (int k = 0; k < 4; ++k) bad += BF16_NONFINITE(p1[k]);
    }
    if (out_nonfinite) *out_nonfinite = bad;
    if (bad) {
        return -2;
    }

    const int out_stride = w * 2 * 3;
    for (int y = 0; y < h; ++y) {
        uint8_t* __restrict row_top = dst + (size_t)(2 * y) * out_stride;
        uint8_t* __restrict row_bot = dst + (size_t)(2 * y + 1) * out_stride;
        const uint16_t* __restrict p0 = b0 + (size_t)y * w * 8;
        const uint16_t* __restrict p1 = b1 + (size_t)y * w * 8;

        for (int x = 0; x < w; ++x) {
            row_top[0] = bf16_pixel(p1[0], mean);
            row_top[1] = bf16_pixel(p0[4], mean);
            row_top[2] = bf16_pixel(p0[0], mean);
            row_top[3] = bf16_pixel(p1[1], mean);
            row_top[4] = bf16_pixel(p0[5], mean);
            row_top[5] = bf16_pixel(p0[1], mean);
            row_top += 6;

            row_bot[0] = bf16_pixel(p1[2], mean);
            row_bot[1] = bf16_pixel(p0[6], mean);
            row_bot[2] = bf16_pixel(p0[2], mean);
            row_bot[3] = bf16_pixel(p1[3], mean);
            row_bot[4] = bf16_pixel(p0[7], mean);
            row_bot[5] = bf16_pixel(p0[3], mean);
            row_bot += 6;

            p0 += 8;
            p1 += 8;
        }
    }
    return 0;
}

#ifdef __cplusplus
}
#endif
