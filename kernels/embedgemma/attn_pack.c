/* Copyright (C) 2026 The ignite-xdna contributors
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * Host packing for kernels/embedgemma/attn_array.py's fused local attention, OpenMP: the same words as its NumPy
 * pack() (checked bit for bit by ``attn_array.py check-native``), written straight into mapped BOs.
 *
 * hd is the head dim (256 or 512); a tile is TR = 4096 / hd query rows, a column group 4 TR rows, a key block
 * NCH = hd / 64 K^T chunks then NCH V chunks.
 * egattn_pack_q(q [L, H, hd] fp32, L, H, W, nkb, hd, items [n][3] = (head, q0 of the group, window start), n,
 *               out [n * 4][QWORDS] u32): per tile a header (q0, s, W, nkb, 12 zeros) then the tile as
 *               bf16 4x8 tiles (attn_probe.tiled(bits(q), 4, 8)).
 * egattn_pack_kv(k, v [L, KV, hd] fp32, L, KV, hd, out [KV][L/32][2 NCH][CHUNK] u32): per 32-key block NCH K^T
 *               chunks (64 dims x 32 keys as 8x4 tiles) then NCH V chunks (32 keys x 64 dims as 8x4 tiles).
 * egattn_unpack(raw [n * 4][TR][hd] bf16, items, n, H, hd, out [L, H, hd] fp32): the device's tiles back in place.
 * bf16 is round to nearest, ties to even, as attn_probe.bf16 (finite inputs).
 * Build: cl /nologo /LD /O2 /arch:AVX2 /openmp attn_pack.c /Fe:egattn_pack.dll
 */
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

enum { QWORDS = 2064, CHUNK = 1024 };

static inline uint16_t bf(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

EXPORT void egattn_pack_q(const float *q, int64_t L, int64_t H, int32_t W, int32_t nkb, int32_t HD,
                          const int32_t *items, int64_t n, uint32_t *out) {
    const int TR = 4096 / HD, QT = HD / 8;
    (void)L;
    int64_t t, x;
    (void)t, (void)x;
#pragma omp parallel for schedule(static)
    for (t = 0; t < n * 4; ++t) {
        int64_t i = t / 4, r = t % 4;
        int32_t h = items[3 * i], q0 = items[3 * i + 1] + TR * (int32_t)r, s = items[3 * i + 2];
        uint32_t *pk = out + t * QWORDS;
        pk[0] = (uint32_t)q0, pk[1] = (uint32_t)s, pk[2] = (uint32_t)W, pk[3] = (uint32_t)nkb;
        memset(pk + 4, 0, 12 * 4);
        uint16_t *dst = (uint16_t *)(pk + 16);
        for (int rb = 0; rb < TR / 4; ++rb)
            for (int rr = 0; rr < 4; ++rr) {
                const float *row = q + ((int64_t)(q0 + rb * 4 + rr) * H + h) * HD;
                for (int cb = 0; cb < QT; ++cb)
                    for (int c = 0; c < 8; ++c) dst[((rb * QT + cb) * 4 + rr) * 8 + c] = bf(row[cb * 8 + c]);
            }
    }
}

EXPORT void egattn_pack_kv(const float *k, const float *v, int64_t L, int64_t KV, int32_t HD, uint32_t *out) {
    const int NCH = HD / 64;
    int64_t nb = L / 32;
    int64_t t, x;
    (void)t, (void)x;
#pragma omp parallel for schedule(static)
    for (x = 0; x < KV * nb; ++x) {
        int64_t kvh = x / nb, b = x % nb;
        uint16_t *base = (uint16_t *)(out + x * 2 * NCH * CHUNK);
        for (int c = 0; c < NCH; ++c) {
            uint16_t *kt = base + c * 2 * CHUNK, *vt = base + (NCH + c) * 2 * CHUNK;
            for (int key = 0; key < 32; ++key) {
                const float *kr = k + ((b * 32 + key) * KV + kvh) * HD + c * 64;
                const float *vr = v + ((b * 32 + key) * KV + kvh) * HD + c * 64;
                int kb = key / 4, kk = key % 4;      /* K^T: tile (dim / 8, key / 4), lane (dim % 8) * 4 + key % 4 */
                for (int dim = 0; dim < 64; ++dim)
                    kt[(((dim / 8) * 8 + kb) * 8 + dim % 8) * 4 + kk] = bf(kr[dim]);
                int vb = key / 8, vk = key % 8;      /* V: tile (key / 8, dim / 4), lane (key % 8) * 4 + dim % 4 */
                for (int dim = 0; dim < 64; ++dim)
                    vt[((vb * 16 + dim / 4) * 8 + vk) * 4 + dim % 4] = bf(vr[dim]);
            }
        }
    }
}

EXPORT void egattn_unpack(const uint16_t *raw, const int32_t *items, int64_t n, int64_t H, int32_t HD, float *out) {
    const int TR = 4096 / HD;
    int64_t t, x;
    (void)t, (void)x;
#pragma omp parallel for schedule(static)
    for (t = 0; t < n * 4; ++t) {
        int64_t i = t / 4, r = t % 4;
        int32_t h = items[3 * i], q0 = items[3 * i + 1] + TR * (int32_t)r;
        for (int row = 0; row < TR; ++row) {
            const uint16_t *src = raw + (t * TR + row) * HD;
            uint32_t *dst = (uint32_t *)(out + ((int64_t)(q0 + row) * H + h) * HD);
            for (int d = 0; d < HD; ++d) dst[d] = (uint32_t)src[d] << 16;
        }
    }
}
