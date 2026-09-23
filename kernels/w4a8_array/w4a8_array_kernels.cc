// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// W4A8 on the whole array: the kernels whole_array_w4a8.py links into every core.
//
// The GEMM kernels are the one-core probe's, unchanged -- this file includes
// kernels/w4a8_probe/w4a8_kernels.cc rather than copying it, so the loop each array
// build runs is the loop results/aie/w4a8_probe_npu.log timed (same template, same
// POLICY, same INNER_* k-loop modes). The zero kernel is upstream's own: mm.cc gets
// zero_i32 from aie_kernels/aie2/zero.cc's zero_vectorized<int32, DIM_M, DIM_N>, and
// so does this file, so every arm zeroes its C tile with the same code.
//
// Symbols a design binds: mm_i8i8_local, mm_i8i4_unpack, mm_i8i4_native (C += A x B,
// A pre-tiled 4 x s int8, B pre-tiled s x 8 as int8 or int4 packed two per byte, low
// nibble first, C pre-tiled 4 x 8 int32) and zero_i32. RUN_KERNEL must stay undefined:
// its trace wrapper is the one-core probe's, not the array's.
#ifdef RUN_KERNEL
#error "RUN_KERNEL belongs to the one-core probe (w4a8_probe.py), not the array design"
#endif

#include "w4a8_kernels.cc"

#include "aie_kernels/aie2/zero.cc"

extern "C" void zero_i32(int32_t *c_out) {
  zero_vectorized<int32_t, DIM_M, DIM_N>(c_out);
}
