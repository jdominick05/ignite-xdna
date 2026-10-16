// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Streamed 16-query x 32-key attention. All state belongs to the caller.
#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>
#include <string.h>

// Stable exp for non-positive max-subtracted scores. Range reduction keeps the
// sixth-degree polynomial on [-ln(2),0]; construct 2^n as an fp32 exponent.
// Avoid rounding the argument to bf16 before exponentiation.
static inline float stable_exp(float x) {
  if (x < -80.0f) return 0;
  int n = (int)(x*1.4426950408889634f);
  float r=x-(float)n*.6931471805599453f;
  float poly=1.0f+r*(1.0f+r*(.5f+r*(.1666666666666667f+r*(.0416666666666667f+r*(.0083333333333333f+r*.0013888888888889f)))));
  uint32_t u=(uint32_t)(n+127)<<23;
  float power; memcpy(&power,&u,sizeof(power));
  return poly*power;
}

using MM = aie::mmul<4, 8, 4, bfloat16, bfloat16, accfloat>;
// Packet: 16 i32 header; Q[4,8,4,8], Kt[8,8,8,4], V[4,16,8,4], mask[16,32].
// State: unnormalized O[16,64], m[16], l[16]. Scratch: S[16,32], P[4,4,4,8].
extern "C" void attention_block(const int32_t *packet, float *state, int32_t *scratch) {
  const bfloat16 *q = reinterpret_cast<const bfloat16 *>(packet + 16);
  const bfloat16 *kt = q + 1024;
  const bfloat16 *v = kt + 2048;
  const bfloat16 *mask = v + 2048;
  float *scores = reinterpret_cast<float *>(scratch);
  bfloat16 *p = reinterpret_cast<bfloat16 *>(scratch + 512);
  float *m = state + 1024, *l = m + 16;
  if (packet[0]) {
    for (int i = 0; i < 1024; i += 16) aie::store_v(state + i, aie::zeros<float,16>());
    for (int i = 0; i < 16; ++i) { m[i] = -3.402823466e38f; l[i] = 0; }
  }
  alignas(32) float tile[16];
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 8; ++b) {
      MM c;
      c.mul(aie::load_v<32>(q + a*256), aie::load_v<32>(kt + b*32));
      for (int k = 1; k < 8; ++k)
        c.mac(aie::load_v<32>(q + a*256+k*32), aie::load_v<32>(kt + k*256+b*32));
      aie::store_v(tile, c.to_vector<float>());
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) scores[(a*4+i)*32+b*4+j] = tile[i*4+j] * 0.125f;
    }
  }
  for (int i = 0; i < 16; ++i) {
    float mx = m[i];
    bool any = false;
    for (int j = 0; j < 32; ++j) if ((float)mask[i*32+j] != 0) {
      float s = scores[i*32+j]; mx = s > mx ? s : mx; any = true;
    }
    float alpha = 1.0f;
    if (any) {
      alpha = l[i] == 0 ? 0.0f : stable_exp(m[i]-mx);
    }
    for (int d = 0; d < 64; ++d) state[i*64+d] *= alpha;
    float sum = 0;
      for (int col = 0; col < 32; ++col) {
        bfloat16 e = (float)mask[i*32+col] != 0 ? (bfloat16)stable_exp(scores[i*32+col]-mx) : (bfloat16)0;
        p[((i/4)*4+col/8)*32+(i%4)*8+col%8] = e;
        sum += (float)e;
      }
    m[i] = mx; l[i] = l[i]*alpha+sum;
  }
  // P is tiled 4x8, V tiled 8x4. The running numerator stays fp32.
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 16; ++b) {
      MM c;
      c.mul(aie::load_v<32>(p+a*128), aie::load_v<32>(v+b*32));
      for (int k = 1; k < 4; ++k)
        c.mac(aie::load_v<32>(p+a*128+k*32), aie::load_v<32>(v+k*512+b*32));
      aie::store_v(tile, c.to_vector<float>());
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) state[(a*4+i)*64+b*4+j] += tile[i*4+j];
    }
  }
}

extern "C" void attention_finish(const float *state, float *out) {
  out += 16; // Keep the prefix guard intact; MLIR passes the full FIFO buffer.
  const float *l = state + 1040;
  // The graph's all-masked rows are handled by the host's extracted effective mask.
  for (int i = 0; i < 16; ++i) {
    float inv = l[i] == 0 ? 0 : aie::inv(l[i]);
    for (int d = 0; d < 64; ++d) out[i*64+d] = state[i*64+d]*inv;
  }
}

extern "C" void attention_empty(const int32_t *, float *, int32_t *) {
}
extern "C" void attention_zero(const float *, float *out) {
  for (int i = 16; i < 1040; i += 16) aie::store_v(out+i,aie::zeros<float,16>());
}
