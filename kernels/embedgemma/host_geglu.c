/* Copyright (C) 2026 The ignite-xdna contributors
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * A native host GeGLU, the comparison point for geglu_epilogue.cc (tools/embedgemma_host_geglu_bench.py):
 * ff[r, j] = gelu_tanh(gu[r, j]) * gu[r, F + j] for gu [rows, 2F] fp32, in fp32, OpenMP over rows, the inner loop
 * left to MSVC's /arch:AVX2 vectoriser (tanhf has a vectorised SVML form there).
 * Build: native.py build --host-geglu  (cl /nologo /LD /O2 /arch:AVX2 /openmp /fp:fast)
 */
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void host_geglu(const float *gu, int64_t rows, int64_t F, float *ff) {
    int64_t r;
#pragma omp parallel for schedule(static)
    for (r = 0; r < rows; ++r) {
        const float *g = gu + r * 2 * F, *u = g + F;
        float *o = ff + r * F;
        for (int64_t j = 0; j < F; ++j) {
            float x = g[j];
            o[j] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))) * u[j];
        }
    }
}
