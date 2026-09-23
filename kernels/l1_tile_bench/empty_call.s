// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 The ignite-xdna contributors
//
// The control for tools/l1_tile_bench.py: a function with the GEMM kernels' signature
// (three pointers, ignored) that returns at once. `ret lr` has five delay slots on AIE2
// (Peano's machine model, results/aie/peano_aie2_machine_model_a36c62b9.log), so the
// smallest legal body is the return plus five nops. Timing it in the same harness gives the
// harness's own cost per call, which the GEMM kernels' cycles are then read against.
	.text
	.globl	empty_call
	.p2align	4
	.type	empty_call,@function
empty_call:
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end0:
	.size	empty_call, .Lfunc_end0-empty_call
