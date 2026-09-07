//===- attention_kernels.cc -------------------------------------*- C++ -*-===//
//
// Fully Vectorized Fused BF16 Self-Attention kernel on XDNA1 (AIE2 / Phoenix).
//
// Computes:
//   1. Scores[N, N] = (Q[N, D_pad] @ K[N, D_pad]^T) * scale
//   2. Attn[N, N]   = Softmax(Scores[N, N], dim=-1)
//   3. Out[N, D_pad]= Attn[N, N] @ V[N, D_pad]
//
// Intermediate matrices stay in 64 KB tile data memory.
// All loads and stores are 32-byte aligned using native AIE2 SIMD instructions.
//
// Compile-time parameters (passed as -D flags by attention.py):
//   N_TOKENS      Sequence length (e.g. 64, 16, 256)
//   HEAD_DIM      Actual head dimension (e.g. 20, 16, 24)
//   HEAD_DIM_PAD  Aligned head dimension (16 or 32)
//   SCALE_VAL     Attention scale factor (e.g. 1 / sqrt(HEAD_DIM))
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef N_TOKENS
#define N_TOKENS 64
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 20
#endif

#ifndef HEAD_DIM_PAD
#define HEAD_DIM_PAD 32
#endif

#ifndef SCALE_VAL
#define SCALE_VAL 0.2236068f
#endif

namespace {

// Fast, stable exp approximation for AIE2 scalar float pipe.
// Valid for x <= 0 (in Softmax, x = val - max_val <= 0).
// Uses 4 immediate multiplies for 2^(-m), avoiding uninitialized .rodata memory in AIE tile.
inline float fast_exp(float x) {
  if (x < -15.0f) return 0.0f;
  if (x > 0.0f) x = 0.0f;

  float z = -x * 1.4426950408889634f;
  int32_t m = (int32_t)z;
  if (m > 15) return 0.0f;

  float r = x + (float)m * 0.6931471805599453f; // r in [-ln(2), 0]
  float exp_r = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * 0.04166667f)));

  float p = 1.0f;
  if (m & 1) p *= 0.5f;
  if (m & 2) p *= 0.25f;
  if (m & 4) p *= 0.0625f;
  if (m & 8) p *= 0.00390625f;

  return p * exp_r;
}

} // namespace

extern "C" {

void attention_single_head(
    const bfloat16 *__restrict qkv,    // [3 * N * HEAD_DIM_PAD]: q, then k, then v
    bfloat16 *__restrict out,          // [N * HEAD_DIM_PAD]
    bfloat16 *__restrict scores_row    // [N] row scratchpad via IRON Buffer (512 bytes for N=256)
) {
  event0();

  const float scale = (float)SCALE_VAL;
  const bfloat16 *q = qkv;
  const bfloat16 *k = qkv + (N_TOKENS * HEAD_DIM_PAD);
  const bfloat16 *v = qkv + (2 * N_TOKENS * HEAD_DIM_PAD);

  for (int i = 0; i < N_TOKENS; ++i) {
    const bfloat16 *q_row = q + (i * HEAD_DIM_PAD);
    bfloat16 *out_row = out + (i * HEAD_DIM_PAD);

    // 1. Q[i] @ K^T with scaling: scores_row[j] = scale * (Q[i] @ K[j])
#if HEAD_DIM_PAD == 16
    const aie::vector<bfloat16, 16> q0 = aie::load_v<16>(q_row);
    for (int j = 0; j < N_TOKENS; ++j) {
      const aie::vector<bfloat16, 16> k0 = aie::load_v<16>(k + j * 16);
      aie::accum<accfloat, 16> acc = aie::mul(q0, k0);
      float dot = aie::reduce_add<float>(acc);
      scores_row[j] = (bfloat16)(dot * scale);
    }
#elif HEAD_DIM_PAD == 32
    const aie::vector<bfloat16, 16> q0 = aie::load_v<16>(q_row);
    const aie::vector<bfloat16, 16> q1 = aie::load_v<16>(q_row + 16);

    for (int j = 0; j < N_TOKENS; ++j) {
      const aie::vector<bfloat16, 16> k0 = aie::load_v<16>(k + j * 32);
      const aie::vector<bfloat16, 16> k1 = aie::load_v<16>(k + j * 32 + 16);
      aie::accum<accfloat, 16> acc = aie::mul(q0, k0);
      acc = aie::mac(acc, q1, k1);
      float dot = aie::reduce_add<float>(acc);
      scores_row[j] = (bfloat16)(dot * scale);
    }
#endif

    // 2. Softmax over scores_row[0 .. N_TOKENS-1]
    aie::vector<bfloat16, 16> v_max = aie::load_v<16>(scores_row);
    for (int j = 16; j < N_TOKENS; j += 16) {
      v_max = aie::max(v_max, aie::load_v<16>(scores_row + j));
    }
    float max_val = (float)aie::reduce_max(v_max);

    // Pass 2: compute exp once, accumulate sum, store exp in-place in scores_row
    float sum = 0.0f;
    for (int j = 0; j < N_TOKENS; ++j) {
      float ev = fast_exp((float)scores_row[j] - max_val);
      sum += ev;
      scores_row[j] = (bfloat16)ev;
    }

    // Pass 3: vectorized normalization (16 elements per vector op)
    const float inv_sum = 1.0f / (sum + 1e-7f);
    const aie::vector<bfloat16, 16> inv_v = aie::broadcast<bfloat16, 16>((bfloat16)inv_sum);
    for (int j = 0; j < N_TOKENS; j += 16) {
      aie::vector<bfloat16, 16> ev_v = aie::load_v<16>(scores_row + j);
      aie::accum<accfloat, 16> acc = aie::mul(ev_v, inv_v);
      aie::store_v(scores_row + j, acc.template to_vector<bfloat16>());
    }

    // 3. Attn[i] @ V: out[i, :] = sum_j (scores_row[j] * v[j, :])
#if HEAD_DIM_PAD == 16
    aie::accum<accfloat, 16> acc0;
    acc0.from_vector(aie::zeros<float, 16>());
    for (int j = 0; j < N_TOKENS; ++j) {
      bfloat16 a = scores_row[j];
      const aie::vector<bfloat16, 16> v0 = aie::load_v<16>(v + j * 16);
      acc0 = aie::mac(acc0, v0, a);
    }
    aie::store_v(out_row, acc0.template to_vector<bfloat16>());

#elif HEAD_DIM_PAD == 32
    aie::accum<accfloat, 16> acc0;
    acc0.from_vector(aie::zeros<float, 16>());
    aie::accum<accfloat, 16> acc1;
    acc1.from_vector(aie::zeros<float, 16>());

    for (int j = 0; j < N_TOKENS; ++j) {
      bfloat16 a = scores_row[j];
      const aie::vector<bfloat16, 16> v0 = aie::load_v<16>(v + j * 32);
      const aie::vector<bfloat16, 16> v1 = aie::load_v<16>(v + j * 32 + 16);
      acc0 = aie::mac(acc0, v0, a);
      acc1 = aie::mac(acc1, v1, a);
    }
    aie::store_v(out_row, acc0.template to_vector<bfloat16>());
    aie::store_v(out_row + 16, acc1.template to_vector<bfloat16>());
#endif
  }

  event1();
}

} // extern "C"
