// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Matrix projections and BF16-aware normalization; no compensated GEMV opcode.
#define BF16_MATRIX_ONLY
#define BF16_EXACT_NORM_DISPATCH
#define BF16_PRODUCT_ATTRIBUTES __attribute__((noinline))
#define BF16_ERROR_ATTRIBUTES __attribute__((noinline))
#define BF16_HELPER_INLINE __attribute__((always_inline)) inline
#include "state_dispatch.cc"
