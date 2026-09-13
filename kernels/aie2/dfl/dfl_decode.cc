// Phoenix AIE2 DFL decode micro-kernel.
//
// Wire format for one anchor:
//   input[0:64]   four groups of sixteen signed INT8 Q4 DFL logits
//   input[64:144] eighty signed INT8 Q4 class logits
//
// The core wire output is anchor-major and interleaves the two result fields:
//   wire[chunk, 84] = [x1, y1, x2, y2, score_0 ... score_79]
// The MemTile egress uses two strided DMA descriptors to deinterleave this
// stream into contiguous host BOs for boxes[8400,4] and scores[8400,80].
//
// The 16-bin expectation is kept in fixed point until the final scalar pixel
// conversion.  The exp polynomial and reciprocal normalization operate on
// 512-bit AIE2 vectors (32 int16 lanes); the upper half is padding because one
// DFL distribution has sixteen bins.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int kBins = 16;
constexpr int kDflDims = 4;
constexpr int kClasses = 80;
constexpr int kAnchorInputBytes = 144;
constexpr int kQ15One = 32767;

using I16x32 = aie::vector<int16_t, 32>;

static inline __attribute__((always_inline)) I16x32 mul_q15(
    I16x32 a, I16x32 b) {
  return aie::mul(a, b).template to_vector<int16_t>(15);
}

// Restoring division for a fixed-point reciprocal.  Keeping this integer and
// bounded avoids a compiler runtime call for scalar floating-point division in
// the core ELF.  The result is floor((1 << numerator_bits) / denominator).
static inline __attribute__((always_inline)) int32_t reciprocal_fixed(
    uint32_t denominator, int numerator_bits) {
  if (denominator == 0)
    return 0;
  const uint32_t numerator = 1u << numerator_bits;
  uint32_t quotient = 0;
  uint32_t remainder = 0;
  for (int bit = numerator_bits; bit >= 0; --bit) {
    remainder = (remainder << 1) | ((numerator >> bit) & 1u);
    if (remainder >= denominator) {
      remainder -= denominator;
      quotient |= 1u << bit;
    }
  }
  return static_cast<int32_t>(quotient);
}

// exp(-z/16), where z is a non-negative INT8-Q4 distance from the maximum
// logit.  Range reduction uses 2^-q and a degree-four polynomial for the
// residual on approximately [-ln(2), 0].  The residual is evaluated in Q15 by
// the vector ALU; q and the final power-of-two scale are lane-local integer
// bookkeeping.
static inline __attribute__((always_inline)) void exp_negative_q15(
    const int8_t *logits, int8_t max_logit, int16_t *out_q15) {
  int16_t residual_q15[32] __attribute__((aligned(64))) = {};
  int exponent[16] = {};

  for (int lane = 0; lane < kBins; ++lane) {
    int z = static_cast<int>(max_logit) - static_cast<int>(logits[lane]);
    if (z < 0)
      z = 0;

    // Work in Q8 for the range reduction: ln(2) ~= 177/256 and z/16 maps to
    // z*16 in that scale.  The bounded corrections keep the residual in
    // [-ln(2), 0] without a scalar division instruction.
    const int z_q8 = z * 16;
    int q = (z * 23 + 128) >> 8;
    while (q * 177 > z_q8)
      --q;
    while ((q + 1) * 177 <= z_q8)
      ++q;
    if (q < 0)
      q = 0;
    exponent[lane] = q;
    const int residual_q8 = q * 177 - z_q8;
    residual_q15[lane] = static_cast<int16_t>(residual_q8 * 128);
  }

  const I16x32 residual = aie::load_v<32>(residual_q15);
  const I16x32 one = aie::broadcast<int16_t, 32>(kQ15One);
  const I16x32 half = aie::broadcast<int16_t, 32>(16384);
  const I16x32 sixth = aie::broadcast<int16_t, 32>(5461);
  const I16x32 twenty_fourth = aie::broadcast<int16_t, 32>(1365);

  const I16x32 r2 = mul_q15(residual, residual);
  const I16x32 r3 = mul_q15(r2, residual);
  const I16x32 r4 = mul_q15(r2, r2);
  const I16x32 term2 = mul_q15(r2, half);
  const I16x32 term3 = mul_q15(r3, sixth);
  const I16x32 term4 = mul_q15(r4, twenty_fourth);
  const I16x32 polynomial = aie::add(
      aie::add(aie::add(one, residual), term2), aie::add(term3, term4));

  int16_t polynomial_q15[32] __attribute__((aligned(64))) = {};
  aie::store_v(polynomial_q15, polynomial);
  for (int lane = 0; lane < kBins; ++lane) {
    const int q = exponent[lane];
    out_q15[lane] = q >= 15 ? 0 : static_cast<int16_t>(
        static_cast<int32_t>(polynomial_q15[lane]) >> q);
  }
  for (int lane = kBins; lane < 32; ++lane)
    out_q15[lane] = 0;
}

static inline __attribute__((always_inline)) float dfl_expectation(
    const int8_t *logits) {
  int8_t max_logit = logits[0];
  for (int lane = 1; lane < kBins; ++lane)
    if (logits[lane] > max_logit)
      max_logit = logits[lane];

  int16_t exp_q15[32] __attribute__((aligned(64))) = {};
  exp_negative_q15(logits, max_logit, exp_q15);

  int32_t sum = 0;
  for (int lane = 0; lane < kBins; ++lane)
    sum += exp_q15[lane];

  // Divide the Q15 sum by sixteen before taking the reciprocal.  The scale
  // numerator is increased by the same factor, retaining the full Q15 output
  // range while keeping the reciprocal in int16.
  const uint32_t reduced_sum = static_cast<uint32_t>(sum > 16 ? sum >> 4 : 1);
  const int32_t reciprocal = reciprocal_fixed(reduced_sum, 26);
  const I16x32 exp_vec = aie::load_v<32>(exp_q15);
  const I16x32 reciprocal_vec =
      aie::broadcast<int16_t, 32>(static_cast<int16_t>(reciprocal));
  const I16x32 probability_vec = mul_q15(exp_vec, reciprocal_vec);

  int16_t probabilities[32] __attribute__((aligned(64))) = {};
  int16_t projection[32] __attribute__((aligned(64))) = {};
  for (int lane = 0; lane < kBins; ++lane)
    projection[lane] = static_cast<int16_t>(lane);
  aie::store_v(probabilities, probability_vec);

  // This is the vector dot product against [0, ..., 15].  The accumulator is
  // widened before the scalar reduction so the expectation cannot overflow.
  const I16x32 probability_input = aie::load_v<32>(probabilities);
  const I16x32 projection_input = aie::load_v<32>(projection);
  const auto products = aie::mul(probability_input, projection_input)
                            .template to_vector<int32_t>(0);
  int32_t dot = 0;
  for (int lane = 0; lane < kBins; ++lane)
    dot += products[lane];
  return static_cast<float>(dot) / 32768.0f;
}

static inline __attribute__((always_inline)) int16_t sigmoid_q15(
    int8_t logit) {
  int8_t negative_abs[32] __attribute__((aligned(64))) = {};
  const int magnitude = logit < 0 ? -static_cast<int>(logit)
                                  : static_cast<int>(logit);
  negative_abs[0] = static_cast<int8_t>(-magnitude);

  int16_t exp_q15[32] __attribute__((aligned(64))) = {};
  exp_negative_q15(negative_abs, 0, exp_q15);
  const int32_t denominator = 32768 + exp_q15[0];
  const int32_t inverse = reciprocal_fixed(denominator, 30);
  if (logit >= 0)
    return static_cast<int16_t>(inverse);
  const int32_t product = static_cast<int32_t>(exp_q15[0]) * inverse;
  return static_cast<int16_t>((product + (1 << 14)) >> 15);
}

static inline __attribute__((always_inline)) void anchor_geometry(
    int anchor, int *grid_x, int *grid_y, int *stride) {
  if (anchor < 6400) {
    *stride = 8;
    *grid_x = anchor % 80;
    *grid_y = anchor / 80;
  } else if (anchor < 8000) {
    const int cell = anchor - 6400;
    *stride = 16;
    *grid_x = cell % 40;
    *grid_y = cell / 40;
  } else {
    const int cell = anchor - 8000;
    *stride = 32;
    *grid_x = cell % 20;
    *grid_y = cell / 20;
  }
}

} // namespace

extern "C" void dfl_decode_chunk(const int8_t *__restrict input,
                                   float *__restrict wire,
                                   int32_t anchor_base) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::set_saturation(aie::saturation_mode::saturate);
  event0();

  for (int local_anchor = 0; local_anchor < 15; ++local_anchor) {
    const int8_t *anchor = input + local_anchor * kAnchorInputBytes;
    float distances[kDflDims];
    for (int dim = 0; dim < kDflDims; ++dim)
      distances[dim] = dfl_expectation(anchor + dim * kBins);

    int grid_x, grid_y, stride;
    anchor_geometry(anchor_base + local_anchor, &grid_x, &grid_y, &stride);
    const float center_x = (static_cast<float>(grid_x) + 0.5f) * stride;
    const float center_y = (static_cast<float>(grid_y) + 0.5f) * stride;
    float *record = wire + local_anchor * 84;
    float *box = record;
    box[0] = center_x - distances[0] * stride;
    box[1] = center_y - distances[1] * stride;
    box[2] = center_x + distances[2] * stride;
    box[3] = center_y + distances[3] * stride;

    float *class_scores = record + 4;
    for (int cls = 0; cls < kClasses; ++cls)
      class_scores[cls] = static_cast<float>(
          sigmoid_q15(anchor[64 + cls])) / 32768.0f;
  }

  __builtin_aiev2_sched_barrier();
  event1();
}
