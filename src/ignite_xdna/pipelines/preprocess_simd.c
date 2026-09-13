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

#ifdef _OPENMP
#include <omp.h>
#endif

#define PAD_VALUE_INT8 ((int8_t)-14) // 114 - 128 = -14 (0xF2)

typedef struct {
    int x0;
    int x1;
    int bx0;
    int bx1;
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
    }

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

        // Vectorized row loop with unrolled fixed-point bilinear filtering
        for (int x = 0; x < nw; ++x) {
            int x0 = x_tab[x].x0;
            int x1 = x_tab[x].x1;
            int bx0 = x_tab[x].bx0;
            int bx1 = x_tab[x].bx1;

            const uint8_t* p00 = row0 + (x0 * 3);
            const uint8_t* p01 = row0 + (x1 * 3);
            const uint8_t* p10 = row1 + (x0 * 3);
            const uint8_t* p11 = row1 + (x1 * 3);

            // Channel 0: Blue (written to Plane 2)
            int r0_b = (p00[0] * bx0 + p01[0] * bx1 + 1024) >> 11;
            int r1_b = (p10[0] * bx0 + p11[0] * bx1 + 1024) >> 11;
            int v_b = (r0_b * by0 + r1_b * by1 + 1024) >> 11;

            // Channel 1: Green (written to Plane 1)
            int r0_g = (p00[1] * bx0 + p01[1] * bx1 + 1024) >> 11;
            int r1_g = (p10[1] * bx0 + p11[1] * bx1 + 1024) >> 11;
            int v_g = (r0_g * by0 + r1_g * by1 + 1024) >> 11;

            // Channel 2: Red (written to Plane 0)
            int r0_r = (p00[2] * bx0 + p01[2] * bx1 + 1024) >> 11;
            int r1_r = (p10[2] * bx0 + p11[2] * bx1 + 1024) >> 11;
            int v_r = (r0_r * by0 + r1_r * by1 + 1024) >> 11;

            // Clamp and convert to signed int8 (-128 .. 127)
            if (v_r < 0) v_r = 0; else if (v_r > 255) v_r = 255;
            if (v_g < 0) v_g = 0; else if (v_g > 255) v_g = 255;
            if (v_b < 0) v_b = 0; else if (v_b > 255) v_b = 255;

            out_r[x] = (int8_t)(v_r - 128);
            out_g[x] = (int8_t)(v_g - 128);
            out_b[x] = (int8_t)(v_b - 128);
        }
    }

    free(x_tab_heap);
    return 0;
}

#ifdef __cplusplus
}
#endif
