//===- groupnorm_kernels.cc -------------------------------------*- C++ -*-===//
//
// Per-core compute for a BF16 GroupNorm(32) on XDNA1 (AIE2 / npu1), driven by
// ../groupnorm.py. One core owns GROUPS_PER_CORE consecutive groups of
// GROUP_LEN elements each and sees them twice through one input ObjectFifo, in
// CHUNK-element objects:
//
//   object 0                 : the core's parameters, written as raw float32
//                              bits inside a bf16 chunk (gn_init)
//   pass 1, N chunks / group : per-lane fp32 sum and sum-of-squares (gn_stats),
//                              then (sum, sumsq) -> (a, b) per group (gn_finalize)
//   pass 2, N chunks / group : y = a * x + b, written to the output fifo (gn_affine)
//
// with a = scale[g] / sqrt(var[g] + eps) and b = bias[g] - mean[g] * a, so the
// second pass is one fused multiply-add per element. Everything that touches
// the data is vectorised 16 lanes wide with the bf16 x bf16 -> fp32 MAC path;
// the scalar float arithmetic is confined to gn_finalize (4 groups per call).
//
// Compile-time parameters (all passed as -D flags by groupnorm.py):
//   CHUNK            elements per fifo object (multiple of 16)
//   GROUP_LEN        elements per group (L of the [1, 32, L] tensor)
//   GROUPS_PER_CORE  groups handled by this core
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef CHUNK
#error "CHUNK must be defined"
#endif
#ifndef GROUP_LEN
#error "GROUP_LEN must be defined"
#endif
#ifndef GROUPS_PER_CORE
#error "GROUPS_PER_CORE must be defined"
#endif

static_assert(CHUNK % 16 == 0, "CHUNK must be a multiple of the vector width");

namespace {

constexpr int VEC = 16;
// Per group: 16 lanes of running sum followed by 16 lanes of running sumsq,
// kept in fp32 and only reduced across lanes once, in gn_finalize.
constexpr int ACC_PER_GROUP = 2 * VEC;
constexpr int N_PARAMS = 2 * GROUPS_PER_CORE; // scale[GROUPS_PER_CORE], bias[...]
// ONNX InstanceNormalization epsilon of every node in resnetv2_50x3_xint8.onnx
// (9.999999747378752e-06 is 1e-5 rounded to fp32).
constexpr float EPS = 1e-5f;

// 1/sqrt(x) in software: Peano's AIE libc has no float sqrtf, and this runs
// four times per core per call, so cost is irrelevant. Bit-hack seed (max rel
// error ~0.18%) plus three Newton steps (error squares each step), which is
// fp32-exact.
inline float rsqrt_sw(float x) {
  union {
    float f;
    uint32_t u;
  } c;
  c.f = x;
  c.u = 0x5f375a86u - (c.u >> 1);
  float y = c.f;
  for (int i = 0; i < 3; i++)
    y = y * (1.5f - 0.5f * x * y * y);
  return y;
}

} // namespace

extern "C" {

// Object 0 of the input stream: N_PARAMS float32 values (scale then bias for
// this core's groups) stored as raw bits in the bf16 chunk buffer. Copies them
// out, zeroes the accumulators, and pins the accumulator->bf16 conversion to
// round-to-nearest-even so the output matches a host-side fp32->bf16 pack.
void gn_init(bfloat16 *raw, float *params, float *acc) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *src = reinterpret_cast<const float *>(raw);
  for (int i = 0; i < N_PARAMS; i++)
    params[i] = src[i];
  const aie::vector<float, VEC> z = aie::zeros<float, VEC>();
  for (int i = 0; i < GROUPS_PER_CORE * ACC_PER_GROUP; i += VEC)
    aie::store_v(acc + i, z);
}

// Pass 1: accumulate x and x*x for one CHUNK of group g into that group's
// per-lane fp32 accumulators. Products are exact (bf16 x bf16 fits fp32) and
// the fp32 lane sums only ever see GROUP_LEN / 16 addends each.
void gn_stats(bfloat16 *in, float *acc, int32_t g) {
  float *a = acc + g * ACC_PER_GROUP;
  aie::accum<accfloat, VEC> acc_s;
  aie::accum<accfloat, VEC> acc_q;
  acc_s.from_vector(aie::load_v<VEC>(a));
  acc_q.from_vector(aie::load_v<VEC>(a + VEC));
  const aie::vector<bfloat16, VEC> ones = aie::broadcast<bfloat16, VEC>(1.0f);
  for (int i = 0; i < CHUNK; i += VEC) {
    aie::vector<bfloat16, VEC> v = aie::load_v<VEC>(in + i);
    acc_s = aie::mac(acc_s, v, ones);
    acc_q = aie::mac(acc_q, v, v);
  }
  aie::store_v(a, acc_s.template to_vector<float>());
  aie::store_v(a + VEC, acc_q.template to_vector<float>());
}

// Between the passes: fold (sum, sumsq, scale, bias) into the affine pair
// (a, b) per group. Variance is E[x^2] - mean^2 in fp32; the harness reports
// how far that sits from the centred two-pass value on the real tensor.
void gn_finalize(float *acc, float *params, float *coef) {
  const float inv_n = 1.0f / (float)GROUP_LEN;
  for (int g = 0; g < GROUPS_PER_CORE; g++) {
    const float *a = acc + g * ACC_PER_GROUP;
    float s = aie::reduce_add(aie::load_v<VEC>(a));
    float q = aie::reduce_add(aie::load_v<VEC>(a + VEC));
    float mean = s * inv_n;
    float var = q * inv_n - mean * mean;
    if (var < 0.0f)
      var = 0.0f;
    float rstd = rsqrt_sw(var + EPS);
    float scale = params[g];
    float bias = params[GROUPS_PER_CORE + g];
    float ca = scale * rstd;
    coef[2 * g] = ca;
    coef[2 * g + 1] = bias - mean * ca;
  }
}

// Pass 2: y = a * x + b for one CHUNK of group g. `a` is split into a bf16
// hi part and a bf16 residual so the product keeps ~16 mantissa bits instead
// of 8; `b` seeds the fp32 accumulator exactly. Two MACs per vector.
void gn_affine(bfloat16 *in, bfloat16 *out, float *coef, int32_t g) {
  const float ca = coef[2 * g];
  const float cb = coef[2 * g + 1];
  const bfloat16 a_hi = (bfloat16)ca;
  const bfloat16 a_lo = (bfloat16)(ca - (float)a_hi);
  const aie::vector<bfloat16, VEC> va_hi = aie::broadcast<bfloat16, VEC>(a_hi);
  const aie::vector<bfloat16, VEC> va_lo = aie::broadcast<bfloat16, VEC>(a_lo);
  aie::accum<accfloat, VEC> acc_b;
  acc_b.from_vector(aie::broadcast<float, VEC>(cb));
  for (int i = 0; i < CHUNK; i += VEC) {
    aie::vector<bfloat16, VEC> v = aie::load_v<VEC>(in + i);
    aie::accum<accfloat, VEC> r = aie::mac(acc_b, v, va_hi);
    r = aie::mac(r, v, va_lo);
    aie::store_v(out + i, r.template to_vector<bfloat16>());
  }
}

} // extern "C"
