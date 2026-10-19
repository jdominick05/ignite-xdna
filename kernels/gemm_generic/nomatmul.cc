// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Timing probe for whole_array_generic.py --hollow-matmul: takes the matmul kernel's place and does nothing, so a
// call keeps every DMA, lock and object-FIFO step and loses only the MACs. Its output is wrong by construction.
#include <stdint.h>

extern "C" void egemm_nomatmul(void *a, void *b, void *c) { (void)a, (void)b, (void)c; }
