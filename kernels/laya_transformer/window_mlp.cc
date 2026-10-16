// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Independent output states share one Q, preserving each tile's original K order.
#include <stdint.h>
extern "C" void chain_gemm_step(const int32_t *,const int32_t *,float *,int);
extern "C" void chain_finish(const float *,float *,int);
extern "C" void window_gemm_step(const int32_t *q,const int32_t *kv,float *states,int reset,int slot) {
  chain_gemm_step(q,kv,states+slot*2048,reset);
}
extern "C" void window_finish(const float *states,int32_t *out,int tiled,int slot) {
  chain_finish(states+slot*2048,reinterpret_cast<float *>(out),tiled);
}
