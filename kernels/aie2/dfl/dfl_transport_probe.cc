// Transport-floor probe for the Phoenix DFL stage.
//
// Same symbol and ABI as dfl_decode.cc, with the decode math removed.  Each of
// the 84 floats in an anchor's wire record is anchor_index * 256 plus that
// anchor's input byte at the same field index, so every box and score stays
// exactly checkable (all values are integers below 2^24, exact in float32)
// while the core does only the work of forwarding bytes it has read.
//
// Linked by tests/test_dfl_transport_floor.py in place of dfl_decode.cc; the
// MLIR transport from dfl_stage.py is unchanged.

#include <stdint.h>

extern "C" void dfl_decode_chunk(const int8_t *__restrict input,
                                 float *__restrict wire,
                                 int32_t anchor_base) {
  for (int local_anchor = 0; local_anchor < 15; ++local_anchor) {
    const int8_t *anchor = input + local_anchor * 144;
    const float base = static_cast<float>(anchor_base + local_anchor) * 256.0f;
    float *record = wire + local_anchor * 84;
    for (int field = 0; field < 84; ++field)
      record[field] = base + static_cast<float>(anchor[field]);
  }
}
