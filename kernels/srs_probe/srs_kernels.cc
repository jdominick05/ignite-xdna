// srs_probe: run the AIE2 shift-round-saturate (vsrs) on known accumulator lanes with a
// RUNTIME shift, so the shift reaches the hardware through a scalar s register exactly as
// the VitisAI DPU overlay's requant handlers do (mov sN, rX; vsrs ..., sN). The host
// compares every lane against exact oracles for several shifter widths (shift used as is,
// shift & 31, shift & 63, clamped), which settles how many bits of sN the shifter reads.
//
// Rounding is forced to floor and saturation to saturate, so the oracle is exact.
//
// Input (int32 x 64):
//   [0] mode: 0 = v32acc32 -> v32int16 (lsrs)      lanes: in[32..63] as int32
//             1 = v32acc32 -> v32int8  (ssrs)      lanes: in[32..63] as int32
//             2 = v16acc64 -> v16int32 (lsrs)      lanes: in[32..47] as int32, upshifted by in[2]
//             3 = v32int8 -> v32acc32 (ups by shift) -> int32, the DPU's bias path
//                                                lanes: low bytes of in[32..63] as int8
//   [1] shift (any int; passed unmodified to the intrinsic)
//   [2] upshift for mode 2 (0..32), applied with ups so acc64 lanes span 64 bits
// Output (int32 x 64): [0] mode echo, [1] shift echo, [2] rounding read back,
//   [3] saturation read back, [32..] the lanes, sign-extended to int32
#include <aie_api/aie.hpp>
#include <stdint.h>

// Lanes start at word LANE (128 B): AIE vector loads drop the low address bits, so an
// unaligned in + 4 silently reads from in + 0 (measured: the first run echoed mode/shift as lanes).
#define LANE 32

extern "C" {

void srs_probe(int32_t *in, int32_t *out) {
    const int32_t mode = in[0];
    const int32_t shift = in[1];
    const int32_t up = in[2];

    aie::tile::current().set_rounding(aie::rounding_mode::floor);
    aie::tile::current().set_saturation(aie::saturation_mode::saturate);

    out[0] = mode;
    out[1] = shift;
    out[2] = (int32_t)aie::tile::current().get_rounding();
    out[3] = (int32_t)aie::tile::current().get_saturation();

    if (mode == 0 || mode == 1) {
        aie::vector<int32_t, 32> v = aie::load_v<32>(in + LANE);
        aie::accum<acc32, 32> acc;
        acc.from_vector(v, 0);
        if (mode == 0) {
            v32int16 r = lsrs((v32acc32)acc, shift, 1);
            aie::vector<int16_t, 32> rv(r);
            for (int i = 0; i < 32; ++i) out[LANE + i] = rv[i];
        } else {
            v32int8 r = ssrs((v32acc32)acc, shift, 1);
            aie::vector<int8_t, 32> rv(r);
            for (int i = 0; i < 32; ++i) out[LANE + i] = rv[i];
        }
    } else if (mode == 4) {
        // Same as mode 3, but the int8 lanes come straight from memory (packed 4 per word at
        // in[LANE..LANE+7]) so the load and the upshift can fuse into vlda.ups.s32.s8, the form
        // the DPU overlay uses. Check the disassembly: the fusion is the compiler's choice.
        aie::vector<int8_t, 32> v8 = aie::load_v<32>((const int8_t *)(in + LANE));
        aie::accum<acc32, 32> acc;
        acc.from_vector(v8, shift);
        aie::vector<int32_t, 32> rv = acc.template to_vector<int32_t>(0);
        for (int i = 0; i < 32; ++i) out[LANE + i] = rv[i];
    } else if (mode == 3) {
        // The DPU overlay's bias path: int8 lanes upshifted into acc32 by a RUNTIME scalar
        // (vlda.ups.s32.s8 / vups.s32.s8 ..., sN), read back unshifted as int32.
        aie::vector<int8_t, 32> v8;
        for (int i = 0; i < 32; ++i) v8[i] = (int8_t)in[LANE + i];
        aie::accum<acc32, 32> acc;
        acc.from_vector(v8, shift);
        aie::vector<int32_t, 32> rv = acc.template to_vector<int32_t>(0);
        for (int i = 0; i < 32; ++i) out[LANE + i] = rv[i];
    } else {
        aie::vector<int32_t, 16> v = aie::load_v<16>(in + LANE);
        aie::accum<acc64, 16> acc;
        acc.from_vector(v, up);
        v16int32 r = lsrs((v16acc64)acc, shift, 1);
        aie::vector<int32_t, 16> rv(r);
        for (int i = 0; i < 16; ++i) out[LANE + i] = rv[i];
        for (int i = 16; i < 32; ++i) out[LANE + i] = 0;
    }
    for (int i = 4; i < LANE; ++i) out[i] = 0;
}

}  // extern "C"
