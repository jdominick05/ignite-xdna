#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void onchip_stream(int32_t *params, int32_t *out) {
  uint32_t count = params[0];
  uint32_t last = 0;
  event0();
  for (uint32_t i = 0; i < count / 16; ++i) {
#pragma unroll 16
    for (int j = 0; j < 16; ++j)
      last = get_ss_uint();
  }
  event1();
  for (int i = 0; i < 256; ++i) {
    event0();
    event1();
  }
  out[0] = last;
  out[1] = count;
}
