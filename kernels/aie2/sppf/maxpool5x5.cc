// Phoenix ABI: input/output/scratch [20][20][16] INT8. Sixteen channel shards.
// Each 512-bit register holds four pixels. No input/output aliasing is allowed.
#include <aie_api/aie.hpp>
#include <stdint.h>

using V = aie::vector<int8_t, 64>;

extern "C" void maxpool5x5(const int8_t *__restrict input,
                           int8_t *__restrict output,
                           int8_t *__restrict horizontal) {
  const V neg = aie::broadcast<int8_t, 64>(-128);
  // Load each source vector once; shuffles retain channel identity across pixels.
  for (int y = 0; y < 20; ++y) {
    V prev = neg;
    V cur = aie::load_v<64>(input + y * 320);
    for (int g = 0; g < 5; ++g) {
      V next = g < 4 ? aie::load_v<64>(input + y * 320 + (g + 1) * 64) : neg;
      V left2 = aie::shuffle_down_fill(prev, cur, 32);
      V left1 = aie::shuffle_down_fill(prev, cur, 48);
      V right1 = aie::shuffle_down_fill(cur, next, 16);
      V right2 = aie::shuffle_down_fill(cur, next, 32);
      aie::store_v(horizontal + y * 320 + g * 64,
                   aie::max(aie::max(left2, left1), aie::max(cur, aie::max(right1, right2))));
      prev = cur; cur = next;
    }
  }
  // Each horizontal vector is loaded once and reused across five vertical windows.
  for (int g = 0; g < 5; ++g) {
    V a = neg, b = neg;
    V c = aie::load_v<64>(horizontal + g * 64);
    V d = aie::load_v<64>(horizontal + 320 + g * 64);
    for (int y = 0; y < 20; ++y) {
      V e = y < 18 ? aie::load_v<64>(horizontal + (y + 2) * 320 + g * 64) : neg;
      aie::store_v(output + y * 320 + g * 64,
                   aie::max(aie::max(a, b), aie::max(c, aie::max(d, e))));
      a = b; b = c; c = d; d = e;
    }
  }
}
