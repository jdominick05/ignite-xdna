// Synthetic native Conv2D fixture: NCHW bf16, groups=C=32, kernel=1x1.
// Constant weights and bias make the independent CPU oracle unambiguous.
#include <aie_api/aie.hpp>

extern "C" void splice_conv(bfloat16 *input, bfloat16 *output) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const auto weights = aie::broadcast<bfloat16, 16>((float)WEIGHT);
  aie::accum<accfloat, 16> bias;
  bias.from_vector(aie::broadcast<float, 16>((float)BIAS));
  for (int i = 0; i < CHUNK; i += 16) {
    const auto x = aie::load_v<16>(input + i);
    const auto y = aie::mac(bias, x, weights);
    aie::store_v(output + i, y.to_vector<bfloat16>());
  }
}
