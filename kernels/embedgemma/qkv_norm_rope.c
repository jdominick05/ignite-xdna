/* Copyright (C) 2026 The ignite-xdna contributors
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * The host comparison point for fusing EmbeddingGemma 2's post-QKV ops: one pass over the QKV GEMM's output row
 * that does what model.py does with three _rms calls and two _rope calls:
 *   q_h = rope(rms(q_h) * q_norm), k_h = rope(rms(k_h) * k_norm), v_h = rms(v_h)   for every head h,
 * rms(x) = x / sqrt(mean(x^2) + eps), rope(y)[j] = y[j] cos[j] - y[j + hd/2] sin[j] (j < hd/2) and
 * y[j] cos[j] + y[j - hd/2] sin[j] above, cos/sin [L, hd] as model.py builds them. fp32 throughout, the sum of
 * squares accumulated in double (numpy's fp32 pairwise sum differs from it in the last bits). OpenMP over rows.
 * Build: native.py build --qkv  (cl /O2 /arch:AVX2 /openmp)
 */
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static void head(const float *x, int hd, const float *w, const float *c, const float *s, float eps, float *o) {
    double ss = 0.0;
    for (int j = 0; j < hd; ++j) ss += (double)x[j] * x[j];
    float inv = (float)(1.0 / sqrt(ss / hd + eps));
    int h2 = hd / 2;
    if (!c) {   /* v: no gain, no rotation */
        for (int j = 0; j < hd; ++j) o[j] = x[j] * inv;
        return;
    }
    for (int j = 0; j < h2; ++j) {
        float a = x[j] * inv * w[j], b = x[j + h2] * inv * w[j + h2];
        o[j] = a * c[j] - b * s[j];
        o[j + h2] = b * c[j + h2] + a * s[j + h2];
    }
}

/* qkv [L, (H + 2 kv) hd] -> q [L, H, hd], k [L, kv, hd], v [L, kv, hd] */
EXPORT void eg_qkv_norm_rope(const float *qkv, int64_t L, int H, int kv, int hd, const float *qw, const float *kw,
                             const float *cos_, const float *sin_, float eps, float *q, float *k, float *v) {
    int64_t r;
    const int64_t width = (int64_t)(H + 2 * kv) * hd;
#pragma omp parallel for schedule(static)
    for (r = 0; r < L; ++r) {
        const float *x = qkv + r * width, *c = cos_ + r * hd, *s = sin_ + r * hd;
        for (int h = 0; h < H; ++h) head(x + h * hd, hd, qw, c, s, eps, q + (r * H + h) * hd);
        for (int h = 0; h < kv; ++h) head(x + (H + h) * hd, hd, kw, c, s, eps, k + (r * kv + h) * hd);
        for (int h = 0; h < kv; ++h) head(x + (H + kv + h) * hd, hd, 0, 0, 0, eps, v + (r * kv + h) * hd);
    }
}
