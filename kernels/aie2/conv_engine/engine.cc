// Phoenix AIE2 packet-driven convolution engine core program.
//
// One program serves every YOLOv8n layer. A weight packet (W) carries a
// 128-byte header, 32 int32 biases and up to 9,216 bytes of int8 weights; an
// activation packet (A) is always 6,400 bytes of uint8 activations in the
// channel-blocked layout [block][row][col][8]; the output object (O) is always
// [4 blocks][5 rows][20 cols][8] uint8 (3,200 bytes). Partial sums for one
// tile stay in a core-local int32 buffer between input-channel chunks.
//
// Arithmetic contract (mirrored bit for bit by
// src/ignite_xdna/compiler/engine_emulator.py):
//   acc  = bias[co] + sum(w * q)             int32, q uint8, w int8
//   q1   = sat_u8(rne(acc >> SHIFT_OUT))     bias already holds 128 << SHIFT_OUT
//   t    = q1 - 128
//   hs   = clip(rne((t * A1 + B1) >> S1), 0, QMAX)
//   qh   = min(rne((hs * K2) >> S2), 127)
//   y    = rne((t * qh) >> YSH)
//   q2   = sat_u8(y + 128)                   (act = hswish) else q2 = q1
//   F_SIGMOID (F_HSWISH clear, the passes emit q1): SiLU through a four-line sigmoid, applied by a separate
//   loop over the emitted or held tile after every pass (inside a pass it spills the pass accumulators):
//     u = |t|;  g = clip(min_i rne((u * A_i + B_i) >> S1), 0, 64)    i = 1..4, A_i int16, B_i int32
//     q2 = sat_u8(rne(((t << 6) + u * g) >> YSH) + 128)              = t * (64 + sign(t) * g) >> YSH
//   residual packet: q = sat_u8(rne(((qm - 128) << LSH_M) + ((qr - 128) << LSH_R)) >> RSH) + 128)
//                    with LSH_M = 0 and LSH_R = RSH unless the header sets F_RES_SHIFTS;
//                    with F_HSWISH the residual packet's own HardSwish constants then act on q (an
//                    activation after the add: YOLO-World's split C2fAttn output convolutions)
//   OP_MUL (general elementwise product, held tile x A tile):
//                    y = sat_u8(sat16(rne((tm * tr) >> YSH)) + 128)   tm = held - 128, tr = A - 128
//   OP_SCALE (per-channel scalar gain on the A tile, no conv passes): coefficients are 4 blocks x
//                    32 int16 lanes (8 channel values replicated 4x per block) at W_OFFSET in the W
//                    packet; y = sat_u8(sat16(rne((t * co) >> YSH)) + 128); F_HOLD writes the hold
//                    buffer instead of the output object, F_EMIT emits the tile.
//   OP_POOL (k x k stride-1 average pool, k <= 5, same two-packet hold/emit pair and halo-in-packet
//                    geometry as the max pool): q = sat_u8(rne((sum * K2) >> S2)), sum over the
//                    window of the unsigned operand bytes; K2/S2 is the reciprocal of the divisor.
// rne = round half to even (AIE conv_even rounding), sat_u8 = clamp to [0, 255].
//
// Program memory is 16 KB. The eight accumulators of a pass must stay in vector
// registers: every loop over the four output blocks has a constant trip count and is
// unrolled, conv_pass is always inlined, and nothing inside those loops depends on a
// run-time value. An outlined pass (1.4 KB and 384 B stack frames) or a run-time
// guard per output block spills them to memory and made the 66-layer frame's core
// time 2.6 and 3 times longer on Phoenix. A row's five pixel groups are computed as
// two dual passes (groups 0-1, 2-3) and one single pass (group 4).
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

// Header word indices (int32) inside the W packet.
enum {
    H_OP = 0, H_K = 1, H_STRIDE = 2, H_NCIN = 3, H_NCO = 4, H_FLAGS = 5,
    H_SHIFT_OUT = 6, H_A1 = 7, H_B1 = 8, H_S1 = 9, H_QMAX = 10, H_K2 = 11,
    H_S2 = 12, H_YSH = 13, H_RSH = 14, H_COUNT_OUT = 15, H_COUNT_ACC = 16,
    H_PHASE0 = 17, H_ROWS_IN = 21, H_COLS_IN = 22, H_PLANE_BYTES = 23,
    H_RLSH_M = 24, H_RLSH_R = 25, H_A2 = 26, H_B2 = 27, H_A3 = 28, H_B3 = 29, H_A4 = 30, H_B4 = 31,
};
enum { OP_NOP = 0, OP_CONV = 1, OP_MAXPOOL = 2, OP_RESIDUAL = 3, OP_FUSED_CONV = 4,
       OP_MUL = 5, OP_SCALE = 6, OP_POOL = 7 };
// Dispatch groups a container does not reach are gated out at compile time. The compiler
// scans the opcode of every packet it is about to store and passes -DENGINE_OP_<NAME>=0 for
// each group none of them uses; the case goes, and with it every anonymous-namespace function
// only that case referenced. No container ignite-compile produces reaches any of the four -
// match_stencil_fusion runs in a test and the verifier but never in the compile path, and
// engine_schedule.py emits no Mul/Scale/Pool packet - which is 5,456 B of a 15,184 B object.
// Each defaults to 1, so a build passing no flags (the census, the synthetic silicon sequence,
// any direct build_program caller) still gets the whole kernel. The ENUM IS NEVER GATED: a
// header word keeps its meaning whatever is compiled in, and the emulator is the contract.
#ifndef ENGINE_OP_FUSED_CONV
#define ENGINE_OP_FUSED_CONV 1
#endif
#ifndef ENGINE_OP_MUL
#define ENGINE_OP_MUL 1
#endif
#ifndef ENGINE_OP_SCALE
#define ENGINE_OP_SCALE 1
#endif
#ifndef ENGINE_OP_POOL
#define ENGINE_OP_POOL 1
#endif
// F_RES_SHIFTS: a residual packet takes the left shifts of both operands from
// H_RLSH_M / H_RLSH_R; without it the held tile is unshifted and the residual
// tile is shifted by RSH (the rule every YOLOv8n residual uses).
// F_SIGMOID: after the passes, the sigmoid SiLU acts on the emitted or held tile at the
// header's line constants (A1/B1, A2/B2..A4/B4, S1, YSH).
enum { F_LOAD_PSUM = 1, F_EMIT = 2, F_HSWISH = 4, F_UP2 = 8, F_HOLD = 16, F_RES_SHIFTS = 32, F_SIGMOID = 64, F_RESIDUAL = 128 };
constexpr int HDR_BYTES = 128;
constexpr int BIAS_BYTES = 128;
constexpr int W_OFFSET = HDR_BYTES + BIAS_BYTES;
constexpr int TILE_ROWS = 5;
constexpr int TILE_COLS = 20;
constexpr int NCO = 4;                                        // output blocks per tile
constexpr int OUT_BLOCK_BYTES = TILE_ROWS * TILE_COLS * 8;    // 800
constexpr int PSUM_BLOCK_WORDS = TILE_ROWS * TILE_COLS * 8;   // 800 int32
constexpr int HOLD_OFFSET_BYTES = NCO * PSUM_BLOCK_WORDS * 4; // 12,800
constexpr int MID_ROWS = 7;
constexpr int MID_COLS_PITCH = 24;                            // 24 cols x 8 bytes = 192 bytes/row
constexpr int MID_BLOCK_BYTES = MID_ROWS * MID_COLS_PITCH * 8; // 1344 bytes
constexpr int UP2_SRC_ROWS = 4;
constexpr int UP2_SRC_COLS = 20;
constexpr int UP2_SRC_BLOCK_BYTES = UP2_SRC_ROWS * UP2_SRC_COLS * 8;  // 640

using MMUL = aie::mmul<4, 8, 8, uint8, int8>;
using V32u = aie::vector<uint8, 32>;
using V64s = aie::vector<int8, 64>;
using V32i16 = aie::vector<int16, 32>;
using Acc = aie::accum<acc32, 32>;

struct Hdr {
    int op, k, stride, ncin, flags, shift_out;
    int a1, b1, s1, qmax, k2, s2, ysh, rsh;
    int rlsh_m, rlsh_r;
    int rows_in, cols_in, plane_bytes;
    int phase;
};

inline Hdr read_header(const int32_t *h, int core_row) {
    Hdr d;
    d.op = h[H_OP]; d.k = h[H_K]; d.stride = h[H_STRIDE]; d.ncin = h[H_NCIN];
    d.flags = h[H_FLAGS]; d.shift_out = h[H_SHIFT_OUT];
    d.a1 = h[H_A1]; d.b1 = h[H_B1]; d.s1 = h[H_S1]; d.qmax = h[H_QMAX];
    d.k2 = h[H_K2]; d.s2 = h[H_S2]; d.ysh = h[H_YSH]; d.rsh = h[H_RSH];
    d.rlsh_m = (d.flags & F_RES_SHIFTS) ? h[H_RLSH_M] : 0;
    d.rlsh_r = (d.flags & F_RES_SHIFTS) ? h[H_RLSH_R] : d.rsh;
    d.rows_in = h[H_ROWS_IN]; d.cols_in = h[H_COLS_IN]; d.plane_bytes = h[H_PLANE_BYTES];
    d.phase = h[H_PHASE0 + (core_row & 3)];
    return d;
}

inline V32i16 unpack_centered(V32u q) {
    return aie::sub(q.template unpack().template cast_to<int16>(), int16_t(128));
}

inline V32u sat_u8_from_i16(V32i16 y) {
    Acc ya;
    ya.from_vector(y);
    return ya.template to_vector<uint8>(0);
}

// HardSwish on 32 uint8 values at the header's constants: q1 -> q2.
__attribute__((always_inline))
inline V32u hswish_u8(V32u q1, const Hdr &d) {
    V32i16 t = unpack_centered(q1);
    Acc accb;
    accb.from_vector(aie::broadcast<int32, 32>(d.b1));
    Acc a1 = aie::mac(accb, t, int16_t(d.a1));
    V32i16 hs = a1.template to_vector<int16>(d.s1);
    hs = aie::max(hs, int16_t(0));
    hs = aie::min(hs, int16_t(d.qmax));
    V32i16 qh = aie::mul(hs, int16_t(d.k2)).template to_vector<int16>(d.s2);
    qh = aie::min(qh, int16_t(127));
    V32i16 y = aie::mul(t, qh).template to_vector<int16>(d.ysh);
    return sat_u8_from_i16(aie::add(y, int16_t(128)));
}

// Piecewise-linear sigmoid SiLU on 32 uint8 values: q1 -> q2. One accumulator at a time: each
// line is reduced to int16 before the next is formed.
struct SigmoidK {
    int32_t b[4];
    int16_t a[4];
    int s, ysh;
};

__attribute__((always_inline))
inline V32u silu_pl_u8(V32u q1, const SigmoidK &k) {
    V32i16 t = unpack_centered(q1);
    V32i16 u = aie::abs(t);
    Acc l;
    l.from_vector(aie::broadcast<int32, 32>(k.b[0]));
    V32i16 g = aie::mac(l, u, k.a[0]).template to_vector<int16>(k.s);
    l.from_vector(aie::broadcast<int32, 32>(k.b[1]));
    g = aie::min(g, aie::mac(l, u, k.a[1]).template to_vector<int16>(k.s));
    l.from_vector(aie::broadcast<int32, 32>(k.b[2]));
    g = aie::min(g, aie::mac(l, u, k.a[2]).template to_vector<int16>(k.s));
    l.from_vector(aie::broadcast<int32, 32>(k.b[3]));
    g = aie::min(g, aie::mac(l, u, k.a[3]).template to_vector<int16>(k.s));
    g = aie::max(g, int16_t(0));
    g = aie::min(g, int16_t(64));
    l.from_vector(t, 6);
    V32i16 y = aie::mac(l, u, g).template to_vector<int16>(k.ysh);
    return sat_u8_from_i16(aie::add(y, int16_t(128)));
}

// HardSwish epilogue on one 4x8 accumulator: acc -> 32 uint8 outputs.
__attribute__((always_inline))
inline V32u epilogue(MMUL &acc, const Hdr &d) {
    V32u q1 = acc.template to_vector<uint8>(d.shift_out);
    if (!(d.flags & F_HSWISH))
        return q1;
    return hswish_u8(q1, d);
}

// One pass over output row r: pixel groups g0 and g0 + 1 (DUAL) or group g0 alone,
// all four output blocks.
template <bool DUAL>
__attribute__((always_inline))
inline void conv_pass(const Hdr &d, const uint8_t *a, const int8_t *w, const int32_t *bias,
                      int32_t *psum, uint8_t *out, int r, int g0) {
    const int g1 = DUAL ? g0 + 1 : g0;
    const int off0 = (r * TILE_COLS + g0 * 4) * 8;
    const int off1 = (r * TILE_COLS + g1 * 4) * 8;
    MMUL acc0[NCO], acc1[NCO];
    if (d.flags & F_LOAD_PSUM) {
#pragma unroll
        for (int b = 0; b < NCO; ++b) {
            acc0[b] = MMUL(aie::load_v<32>(psum + b * PSUM_BLOCK_WORDS + off0));
            if (DUAL)
                acc1[b] = MMUL(aie::load_v<32>(psum + b * PSUM_BLOCK_WORDS + off1));
        }
    } else {
#pragma unroll
        for (int b = 0; b < NCO; ++b) {
            aie::vector<int32, 8> bv = aie::load_v<8>(bias + b * 8);
            acc0[b] = MMUL(bv.template grow_replicate<32>());
            if (DUAL)
                acc1[b] = MMUL(bv.template grow_replicate<32>());
        }
    }
    const int ncin = d.ncin;
    const int k = d.k;
    const int8_t *wp = w;
    if (d.stride == 2) {
        // Input pixel j = 2x + kx: load eight consecutive pixels as two 4-pixel
        // vectors and keep the even pixels (8-byte chunks) with an unzip.
        // Both loops over input channel blocks walk RUNNING POINTERS rather than rebuilding
        // `a + c * d.plane_bytes + aoff` each iteration. Arithmetic-neutral - same addresses, same
        // order, same MACs, byte-exact - but it replaces a multiply per iteration with an add, and
        // that is worth more than it looks: the stride-1 dual loop goes 23 bundles per 8 MACs to
        // 18 and the stride-2 dual 32 to 27, the object SHRINKS by 96 B, and on silicon it is
        // -2.8% on a YOLOv8n frame and -4.4% on YOLOv8s. Do not "simplify" this back to indexing
        // inside the loop. fused_stage1 deliberately still indexes: its trip count is the constant
        // 2, so the compiler already strength-reduces it and the pointer form compiles identically.
        for (int ky = 0; ky < k; ++ky) {
            for (int kx = 0; kx < k; ++kx) {
                const int aoff0 = ((2 * r + ky) * d.cols_in + 8 * g0 + kx) * 8;
                const int aoff1 = aoff0 + (g1 - g0) * 64;
                const uint8_t *p0 = a + aoff0;
                const uint8_t *p1 = a + aoff1;
                for (int c = 0; c < ncin; ++c, p0 += d.plane_bytes, p1 += d.plane_bytes) {
                    auto [av0, od0] = aie::interleave_unzip(aie::load_unaligned_v<32>(p0),
                                                            aie::load_unaligned_v<32>(p0 + 32), 8);
                    V32u av1 = av0;
                    if (DUAL) {
                        auto [e1, o1] = aie::interleave_unzip(aie::load_unaligned_v<32>(p1),
                                                              aie::load_unaligned_v<32>(p1 + 32), 8);
                        av1 = e1;
                    }
#pragma unroll
                    for (int b = 0; b < NCO; ++b) {
                        V64s wv = aie::load_v<64>(wp);
                        wp += 64;
                        acc0[b].mac(av0, wv);
                        if (DUAL)
                            acc1[b].mac(av1, wv);
                    }
                }
            }
        }
    } else {
        for (int ky = 0; ky < k; ++ky) {
            for (int kx = 0; kx < k; ++kx) {
                const int aoff0 = ((r + ky) * d.cols_in + 4 * g0 + kx) * 8;
                const int aoff1 = aoff0 + (g1 - g0) * 32;
                const uint8_t *p0 = a + aoff0;
                const uint8_t *p1 = a + aoff1;
                for (int c = 0; c < ncin; ++c, p0 += d.plane_bytes, p1 += d.plane_bytes) {
                    V32u av0 = aie::load_unaligned_v<32>(p0);
                    V32u av1 = av0;
                    if (DUAL)
                        av1 = aie::load_unaligned_v<32>(p1);
#pragma unroll
                    for (int b = 0; b < NCO; ++b) {
                        V64s wv = aie::load_v<64>(wp);
                        wp += 64;
                        acc0[b].mac(av0, wv);
                        if (DUAL)
                            acc1[b].mac(av1, wv);
                    }
                }
            }
        }
    }
    // Retire: emit the activated tile, hold it for a residual packet, or keep partial sums.
    if (d.flags & (F_EMIT | F_HOLD)) {
        uint8_t *dst = (d.flags & F_EMIT) ? out : reinterpret_cast<uint8_t *>(psum) + HOLD_OFFSET_BYTES;
#pragma unroll
        for (int b = 0; b < NCO; ++b) {
            aie::store_v(dst + b * OUT_BLOCK_BYTES + off0, epilogue(acc0[b], d));
            if (DUAL)
                aie::store_v(dst + b * OUT_BLOCK_BYTES + off1, epilogue(acc1[b], d));
        }
    } else {
#pragma unroll
        for (int b = 0; b < NCO; ++b) {
            aie::store_v(psum + b * PSUM_BLOCK_WORDS + off0, acc0[b].template to_vector<int32>());
            if (DUAL)
                aie::store_v(psum + b * PSUM_BLOCK_WORDS + off1, acc1[b].template to_vector<int32>());
        }
    }
}

// The sigmoid SiLU over a whole emitted or held tile, in place. It runs after every pass of
// the packet, so no pass accumulator is live: inside the pass epilogue the same arithmetic
// spills them to the stack (tools/engine_epilogue_variants.py). Out of line, so run() and its
// pass code are laid out as before; the constants are copied from the raw header into locals
// first, because a store through the uint8 tile pointer may alias the header and would force a
// reload of every constant after each store.
__attribute__((noinline))
void silu_pl_tile(const int32_t *h, uint8_t *dst) {
    const SigmoidK k = {{h[H_B1], h[H_B2], h[H_B3], h[H_B4]},
                        {int16_t(h[H_A1]), int16_t(h[H_A2]), int16_t(h[H_A3]), int16_t(h[H_A4])},
                        h[H_S1], h[H_YSH]};
    for (int off = 0; off < NCO * OUT_BLOCK_BYTES; off += 32)
        aie::store_v(dst + off, silu_pl_u8(aie::load_v<32>(dst + off), k));
}

// Nearest 2x upsampling: expand the [10 blocks][4][20][8] source packet in place
// into a [8 blocks][5][20][8] full-resolution tile. Output row r reads source row
// (r + phase) >> 1; output col x reads source col x >> 1. Blocks are expanded from
// the last to the first; the first four overlap their source and go through tmp.
inline void up2_expand(uint8_t *a, int phase, uint8_t *tmp) {
    for (int b = 7; b >= 0; --b) {
        const uint8_t *src = a + b * UP2_SRC_BLOCK_BYTES;
        if (b < 4) {
            for (int i = 0; i < UP2_SRC_BLOCK_BYTES; i += 64)
                aie::store_v(tmp + i, aie::load_v<64>(src + i));
            src = tmp;
        }
        uint8_t *dst = a + b * OUT_BLOCK_BYTES;
        for (int r = 0; r < TILE_ROWS; ++r) {
            const uint8_t *srow = src + (((r + phase) >> 1) * UP2_SRC_COLS) * 8;
            uint8_t *drow = dst + (r * TILE_COLS) * 8;
            for (int g = 0; g < TILE_COLS / 8; ++g) {  // 8 output pixels from 4 source pixels
                V32u s = aie::load_v<32>(srow + g * 32);
                auto [lo, hi] = aie::interleave_zip(s, s, 8);
                aie::store_v(drow + g * 64, lo);
                aie::store_v(drow + g * 64 + 32, hi);
            }
            // Pixels 16..19 from source pixels 8, 9.
            aie::vector<uint8, 16> s2 = aie::load_v<16>(srow + 64);
            auto [lo2, hi2] = aie::interleave_zip(aie::concat(s2, s2), aie::concat(s2, s2), 8);
            aie::store_v(drow + 128, lo2);
        }
    }
}

// Residual add: the conv tile (activated by its own packet, or not) was held as uint8 in
// the psum buffer by the previous packet; this packet's A holds the residual tile
// [block][5][20][8]. With F_HSWISH the sum is activated here, at this packet's constants.
inline void residual_tile(const Hdr &d, const uint8_t *a, const int32_t *psum, uint8_t *out) {
    const uint8_t *held = reinterpret_cast<const uint8_t *>(psum) + HOLD_OFFSET_BYTES;
    const int rsh = d.rsh;
    const int lsh_m = d.rlsh_m;
    const int16_t rmul = int16_t(1 << d.rlsh_r);
    const bool act = (d.flags & F_HSWISH) != 0;
    for (int off = 0; off < NCO * OUT_BLOCK_BYTES; off += 32) {
        V32i16 tm = unpack_centered(aie::load_v<32>(held + off));
        V32i16 tr = unpack_centered(aie::load_v<32>(a + off));
        Acc sa;
        sa.from_vector(tm, lsh_m);
        sa = aie::mac(sa, tr, rmul);
        V32i16 y = sa.template to_vector<int16>(rsh);
        V32u q = sat_u8_from_i16(aie::add(y, int16_t(128)));
        if (act)
            q = hswish_u8(q, d);
        aie::store_v(out + off, q);
    }
}

// 5x5 stride-1 max pool with the two-pixel halo already inside the packet: A
// holds two blocks of [rows_in][cols_in][8]; output row r, col x is the max
// over the 5x5 window starting at packet row r, col x. A tile takes two
// packets: the first (F_HOLD) keeps its two blocks in the hold area, the
// second (F_EMIT) copies them to output blocks 0-1 and computes blocks 2-3.
inline void maxpool_tile(const Hdr &d, const uint8_t *a, int32_t *psum, uint8_t *out) {
    uint8_t *hold = reinterpret_cast<uint8_t *>(psum) + HOLD_OFFSET_BYTES;
    uint8_t *dst = (d.flags & F_EMIT) ? out + 2 * OUT_BLOCK_BYTES : hold;
    if (d.flags & F_EMIT) {
        for (int i = 0; i < 2 * OUT_BLOCK_BYTES; i += 32)
            aie::store_v(out + i, aie::load_v<32>(hold + i));
    }
    // Unsigned bytes are compared as signed bytes after flipping the top bit
    // (q ^ 0x80 preserves the unsigned order), so the native int8 max applies.
    using V32s = aie::vector<int8, 32>;
    const V32s flip = aie::broadcast<int8, 32>(int8_t(-128));
    for (int b = 0; b < 2; ++b) {
        const uint8_t *plane = a + b * d.plane_bytes;
        for (int r = 0; r < TILE_ROWS; ++r) {
            for (int g = 0; g < TILE_COLS / 4; ++g) {
                V32s m = aie::broadcast<int8, 32>(int8_t(-128));
                for (int dy = 0; dy < 5; ++dy) {
                    const uint8_t *row = plane + ((r + dy) * d.cols_in + 4 * g) * 8;
                    for (int dx = 0; dx < 5; ++dx) {
                        V32s v = aie::load_unaligned_v<32>(row + dx * 8).template cast_to<int8>();
                        m = aie::max(m, aie::bit_xor(v, flip));
                    }
                }
                aie::store_v(dst + b * OUT_BLOCK_BYTES + (r * TILE_COLS + 4 * g) * 8,
                             aie::bit_xor(m, flip).template cast_to<uint8>());
            }
        }
    }
}

// ----------------------------------------------------------------------------
// General elementwise product, per-channel scalar gain, k x k average pool
// ----------------------------------------------------------------------------

// Elementwise multiply: the held tile (first operand, left by a producing packet) times the
// A packet tile (second). YSH is the requantization shift; the product of two centered bytes
// needs no accumulator beyond the int32 lanes the mac already forms.
inline void mul_tile(const Hdr &d, const uint8_t *a, const int32_t *psum, uint8_t *out) {
    const uint8_t *held = reinterpret_cast<const uint8_t *>(psum) + HOLD_OFFSET_BYTES;
    const int ysh = d.ysh;
#pragma clang loop unroll(disable)
    for (int off = 0; off < NCO * OUT_BLOCK_BYTES; off += 32) {
        V32i16 tm = unpack_centered(aie::load_v<32>(held + off));
        V32i16 tr = unpack_centered(aie::load_v<32>(a + off));
        Acc sa;
        sa.from_vector(aie::broadcast<int32, 32>(0));
        sa = aie::mac(sa, tm, tr);
        aie::store_v(out + off, sat_u8_from_i16(aie::add(sa.template to_vector<int16>(ysh), int16_t(128))));
    }
}

// Per-channel scalar gain on the A tile, no conv passes: 4 blocks x 32 int16 coefficients at
// W_OFFSET (8 channel values replicated 4x per block, matching the 4 pixels x 8 channels a
// 32-byte vector covers). F_HOLD writes the hold buffer for a later packet, F_EMIT the object.
inline void scale_tile(const Hdr &d, const uint8_t *a, const uint8_t *wpkt, int32_t *psum, uint8_t *out) {
    const int16_t *coef = reinterpret_cast<const int16_t *>(wpkt + W_OFFSET);
    uint8_t *dst = (d.flags & F_EMIT) ? out : reinterpret_cast<uint8_t *>(psum) + HOLD_OFFSET_BYTES;
    const int ysh = d.ysh;
    for (int b = 0; b < NCO; ++b) {
        const V32i16 cv = aie::load_v<32>(coef + b * 32);
#pragma clang loop unroll(disable)
        for (int o = 0; o < OUT_BLOCK_BYTES; o += 32) {
            const int off = b * OUT_BLOCK_BYTES + o;
            V32i16 t = unpack_centered(aie::load_v<32>(a + off));
            Acc sa;
            sa.from_vector(aie::broadcast<int32, 32>(0));
            sa = aie::mac(sa, t, cv);
            aie::store_v(dst + off, sat_u8_from_i16(aie::add(sa.template to_vector<int16>(ysh), int16_t(128))));
        }
    }
}

// k x k stride-1 average pool (k <= 5) on the max pool's geometry: the window halo is inside
// the packet, two blocks per packet, the first packet holds blocks 0-1, the second emits them
// and its own pair. The window sum of unsigned bytes fits int16 (25 x 255 = 6,375); the
// reciprocal K2/S2 divides by any window size whose rounded mean stays in [0, 255].
inline void avgpool_tile(const Hdr &d, const uint8_t *a, int32_t *psum, uint8_t *out) {
    uint8_t *hold = reinterpret_cast<uint8_t *>(psum) + HOLD_OFFSET_BYTES;
    uint8_t *dst = (d.flags & F_EMIT) ? out + 2 * OUT_BLOCK_BYTES : hold;
    if (d.flags & F_EMIT) {
        for (int i = 0; i < 2 * OUT_BLOCK_BYTES; i += 32)
            aie::store_v(out + i, aie::load_v<32>(hold + i));
    }
    const int k = d.k;
    const int16_t rmul = int16_t(d.k2);
    const int rsh = d.s2;
    for (int b = 0; b < 2; ++b) {
        const uint8_t *plane = a + b * d.plane_bytes;
#pragma clang loop unroll(disable)
        for (int r = 0; r < TILE_ROWS; ++r) {
#pragma clang loop unroll(disable)
            for (int g = 0; g < TILE_COLS / 4; ++g) {
                V32i16 s = aie::zeros<int16, 32>();
                for (int dy = 0; dy < k; ++dy) {
                    const uint8_t *row = plane + ((r + dy) * d.cols_in + 4 * g) * 8;
                    for (int dx = 0; dx < k; ++dx)
                        s = aie::add(s, aie::load_unaligned_v<32>(row + dx * 8).template unpack().template cast_to<int16>());
                }
                aie::store_v(dst + b * OUT_BLOCK_BYTES + (r * TILE_COLS + 4 * g) * 8,
                             aie::mul(s, rmul).template to_vector<uint8>(rsh));
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Fused Spatial Stencil Convolution (Stage 1 Conv3x3 + Stage 2 Conv3x3 + Residual)
// ----------------------------------------------------------------------------
__attribute__((noinline))
void fused_stage1(const int32_t *h, const uint8_t *apkt, uint8_t *mid, int core_row) {
    const int y_quad = h[H_PHASE0];
    const int x0 = h[H_PHASE0 + 1];
    const int H = h[H_PHASE0 + 2];
    const int W = h[H_PHASE0 + 3];
    const int y0 = y_quad + TILE_ROWS * core_row;

    const int r_start = (y0 == 0) ? 1 : 0;
    const int r_end = (y0 + TILE_ROWS >= H) ? 6 : 7;
    const int c_start = (x0 == 0) ? 1 : 0;
    const int c_end = (x0 + TILE_COLS >= W) ? 21 : 22;

    const int w1_off = h[H_A3]; // byte offset to Stage 1 block
    const uint8_t *w1_base = reinterpret_cast<const uint8_t *>(h) + w1_off;
    const int32_t *s1_params = reinterpret_cast<const int32_t *>(w1_base);
    const int32_t *b1 = reinterpret_cast<const int32_t *>(w1_base + 32);
    const int8_t *w1 = reinterpret_cast<const int8_t *>(w1_base + 96);

    Hdr d1;
    d1.shift_out = s1_params[7];
    d1.a1 = s1_params[0];
    d1.b1 = s1_params[1];
    d1.s1 = s1_params[2];
    d1.qmax = s1_params[3];
    d1.k2 = s1_params[4];
    d1.s2 = s1_params[5];
    d1.ysh = s1_params[6];
    d1.flags = F_HSWISH;

    const int cols_in = h[H_COLS_IN];
    const int plane_bytes = h[H_PLANE_BYTES];
    const V32u zp_vec = aie::broadcast<uint8, 32>(128);

    for (int rm = 0; rm < MID_ROWS; ++rm) {
        if (rm < r_start || rm >= r_end) {
            for (int b = 0; b < 2; ++b) {
                uint8_t *row_dst = mid + b * MID_BLOCK_BYTES + rm * MID_COLS_PITCH * 8;
                for (int v = 0; v < 6; ++v)
                    aie::store_v(row_dst + v * 32, zp_vec);
            }
            continue;
        }
        for (int g = 0; g < 6; ++g) {
            const int cm0 = (g == 5) ? 18 : g * 4;
            MMUL acc0 = MMUL(aie::load_v<8>(b1).template grow_replicate<32>());
            MMUL acc1 = MMUL(aie::load_v<8>(b1 + 8).template grow_replicate<32>());

            const int8_t *wp = w1;
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    const int aoff = ((rm + ky) * cols_in + cm0 + kx) * 8;
                    for (int c = 0; c < 2; ++c) {
                        const uint8_t *plane = apkt + c * plane_bytes;
                        V32u av = aie::load_unaligned_v<32>(plane + aoff);
                        V64s wv0 = aie::load_v<64>(wp);
                        V64s wv1 = aie::load_v<64>(wp + 64);
                        wp += 128;
                        acc0.mac(av, wv0);
                        acc1.mac(av, wv1);
                    }
                }
            }
            V32u q0 = epilogue(acc0, d1);
            V32u q1 = epilogue(acc1, d1);
            aie::store_v(mid + 0 * MID_BLOCK_BYTES + (rm * MID_COLS_PITCH + cm0) * 8, q0);
            aie::store_v(mid + 1 * MID_BLOCK_BYTES + (rm * MID_COLS_PITCH + cm0) * 8, q1);
        }
        if (c_start == 1) {
            *reinterpret_cast<uint64_t *>(mid + 0 * MID_BLOCK_BYTES + rm * MID_COLS_PITCH * 8) = 0x8080808080808080ULL;
            *reinterpret_cast<uint64_t *>(mid + 1 * MID_BLOCK_BYTES + rm * MID_COLS_PITCH * 8) = 0x8080808080808080ULL;
        }
        if (c_end == 21) {
            *reinterpret_cast<uint64_t *>(mid + 0 * MID_BLOCK_BYTES + (rm * MID_COLS_PITCH + 21) * 8) = 0x8080808080808080ULL;
            *reinterpret_cast<uint64_t *>(mid + 1 * MID_BLOCK_BYTES + (rm * MID_COLS_PITCH + 21) * 8) = 0x8080808080808080ULL;
        }
    }
}

__attribute__((noinline))
void fused_stage2(const Hdr &d2, const uint8_t *mid, const uint8_t *apkt, const int8_t *w2,
                  const int32_t *bias2, uint8_t *out) {
    for (int r = 0; r < TILE_ROWS; ++r) {
        for (int g = 0; g < 5; ++g) {
            const int g0 = g * 4;
            const int off0 = (r * TILE_COLS + g0) * 8;
            MMUL acc[NCO];
#pragma unroll
            for (int b = 0; b < NCO; ++b) {
                aie::vector<int32, 8> bv = aie::load_v<8>(bias2 + b * 8);
                acc[b] = MMUL(bv.template grow_replicate<32>());
            }
            const int8_t *wp = w2;
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    const int mid_off = ((r + ky) * MID_COLS_PITCH + g0 + kx) * 8;
                    for (int c = 0; c < 2; ++c) {
                        const uint8_t *plane = mid + c * MID_BLOCK_BYTES;
                        V32u av = aie::load_unaligned_v<32>(plane + mid_off);
#pragma unroll
                        for (int b = 0; b < NCO; ++b) {
                            V64s wv = aie::load_v<64>(wp);
                            wp += 64;
                            acc[b].mac(av, wv);
                        }
                    }
                }
            }
#pragma unroll
            for (int b = 0; b < NCO; ++b) {
                V32u q = epilogue(acc[b], d2);
                if ((d2.flags & F_RESIDUAL) && b < 2) {
                    const int res_off = ((2 + r) * d2.cols_in + 2 + g0) * 8;
                    V32u qres = aie::load_unaligned_v<32>(apkt + b * d2.plane_bytes + res_off);
                    V32i16 tm = unpack_centered(q);
                    V32i16 tr = unpack_centered(qres);
                    Acc sa;
                    sa.from_vector(tm, d2.rlsh_m);
                    sa = aie::mac(sa, tr, int16_t(1 << d2.rlsh_r));
                    V32i16 y = sa.template to_vector<int16>(d2.rsh);
                    q = sat_u8_from_i16(aie::add(y, int16_t(128)));
                }
                aie::store_v(out + b * OUT_BLOCK_BYTES + off0, q);
            }
        }
    }
}

inline void fused_conv_tile(const int32_t *hdr, const uint8_t *apkt, int32_t *psum, uint8_t *out, int core_row) {
    uint8_t *mid = reinterpret_cast<uint8_t *>(psum) + HOLD_OFFSET_BYTES;
    fused_stage1(hdr, apkt, mid, core_row);

    const Hdr d2 = read_header(hdr, core_row);
    const int w2_off = hdr[H_A4];
    const uint8_t *wpkt = reinterpret_cast<const uint8_t *>(hdr);
    const int32_t *bias2 = reinterpret_cast<const int32_t *>(wpkt + HDR_BYTES);
    const int8_t *w2 = reinterpret_cast<const int8_t *>(wpkt + w2_off);

    fused_stage2(d2, mid, apkt, w2, bias2, out);
}

inline void run(int32_t *hdr, uint8_t *apkt, uint8_t *out, int32_t *psum, int core_row) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);
    const Hdr d = read_header(hdr, core_row);
    const uint8_t *wpkt = reinterpret_cast<const uint8_t *>(hdr);
    const int32_t *bias = reinterpret_cast<const int32_t *>(wpkt + HDR_BYTES);
    const int8_t *w = reinterpret_cast<const int8_t *>(wpkt + W_OFFSET);
    switch (d.op) {
    case OP_CONV:
        if (d.flags & F_UP2)
            up2_expand(apkt, d.phase, out);  // accumulate-only packets: out is scratch
        for (int r = 0; r < TILE_ROWS; ++r) {  // five pixel groups per row: 0-1, 2-3, 4
            conv_pass<true>(d, apkt, w, bias, psum, out, r, 0);
            conv_pass<true>(d, apkt, w, bias, psum, out, r, 2);
            conv_pass<false>(d, apkt, w, bias, psum, out, r, 4);
        }
        if ((d.flags & F_SIGMOID) && (d.flags & (F_EMIT | F_HOLD)))
            silu_pl_tile(hdr, (d.flags & F_EMIT) ? out : reinterpret_cast<uint8_t *>(psum) + HOLD_OFFSET_BYTES);
        break;
    case OP_MAXPOOL:
        maxpool_tile(d, apkt, psum, out);
        break;
    case OP_RESIDUAL:
        residual_tile(d, apkt, psum, out);
        break;
#if ENGINE_OP_MUL
    case OP_MUL:
        mul_tile(d, apkt, psum, out);
        break;
#endif
#if ENGINE_OP_SCALE
    case OP_SCALE:
        scale_tile(d, apkt, wpkt, psum, out);
        break;
#endif
#if ENGINE_OP_POOL
    case OP_POOL:
        avgpool_tile(d, apkt, psum, out);
        break;
#endif
#if ENGINE_OP_FUSED_CONV
    case OP_FUSED_CONV:
        fused_conv_tile(hdr, apkt, psum, out, core_row);
        break;
#endif
    default:
        break;
    }
}

}  // namespace

extern "C" {

// wpkt: W packet viewed as int32 words (the header is read from it); apkt:
// 6,400-byte A packet; out: 3,200-byte output object, written only when the
// header sets F_EMIT (accumulate-only packets receive a scratch object, which
// an F_UP2 packet also uses as expansion scratch); psum: 16,000-byte
// core-local scratch; core_row: 0..3 inside the column.
void engine(int32_t *wpkt, uint8_t *apkt, uint8_t *out, int32_t *psum, int32_t core_row) {
    run(wpkt, apkt, out, psum, core_row);
}

}  // extern "C"
