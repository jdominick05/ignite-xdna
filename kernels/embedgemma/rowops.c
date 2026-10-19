/* Copyright (C) 2026 The ignite-xdna contributors
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * Native host row ops for EmbeddingGemma 2's remaining host work (model.py's native hook, --native-rowops):
 *   eg_rms      out = x / sqrt(mean(x^2) + eps) [* w]                      (model._rms; w may be NULL)
 *   eg_add_rms  out = (res + rms(a) * w) * scalar                           (every post-norm and residual add,
 *                                                                            the layer scalar folded in)
 *   eg_gelu_mul out = gelu_tanh(a) * b                                      (the per-layer-input gate)
 * fp32 throughout, sums of squares accumulated in double, OpenMP over rows. `out` may alias `res` or `x`.
 * Build: native.py build --rowops  (cl /O2 /arch:AVX2 /openmp, as the others)
 */
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static float inv_rms(const float *x, int64_t d, float eps) {
    double ss = 0.0;
    for (int64_t j = 0; j < d; ++j) ss += (double)x[j] * x[j];
    return (float)(1.0 / sqrt(ss / (double)d + eps));
}

EXPORT void eg_rms(const float *x, int64_t rows, int64_t d, const float *w, float eps, float *out) {
    int64_t r;
#pragma omp parallel for schedule(static)
    for (r = 0; r < rows; ++r) {
        const float *xr = x + r * d;
        float *o = out + r * d, inv = inv_rms(xr, d, eps);
        if (w)
            for (int64_t j = 0; j < d; ++j) o[j] = xr[j] * inv * w[j];
        else
            for (int64_t j = 0; j < d; ++j) o[j] = xr[j] * inv;
    }
}

EXPORT void eg_add_rms(const float *res, const float *a, int64_t rows, int64_t d, const float *w, float eps,
                       float scalar, float *out) {
    int64_t r;
#pragma omp parallel for schedule(static)
    for (r = 0; r < rows; ++r) {
        const float *ar = a + r * d, *rr = res + r * d;
        float *o = out + r * d, inv = inv_rms(ar, d, eps);
        for (int64_t j = 0; j < d; ++j) o[j] = (rr[j] + ar[j] * inv * w[j]) * scalar;
    }
}

EXPORT void eg_gelu_mul(const float *a, const float *b, int64_t n, float *out) {
    int64_t i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < n; ++i) {
        float x = a[i];
        out[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))) * b[i];
    }
}
