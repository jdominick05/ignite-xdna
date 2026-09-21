"""The bf16 engine emulator, offline (no NPU, no compiler).

  python tests/test_engine_bf16_emulator_offline.py

The emulator is the only reference a bf16 packet has, so it is checked here against a NAIVE LOOP
that shares none of its code: one scalar multiply-accumulate at a time, with the addresses computed
the way kernels/bf16_conv/engine_bf16.cc computes them (the activation offset from row, kernel tap
and column; the weight pointer advanced 32 elements per output block in walk order; element
m*4+n of a 4x4 result being pixel m, channel n). A layout the two disagree on fails here instead of
being blamed on the core.

- `to_bf16` rounds to nearest even on the bits, keeps the sign of zero and NaN, and carries into
  infinity; `bf16_bits` / `from_bf16_bits` are inverses and tell -0.0 from +0.0;
- `run_packet` equals the naive loop bit for bit for every multiply-accumulate model, over k = 1, 3, 5,
  stride 1 and 2, one and several input channel blocks;
- a two-packet F_LOAD_PSUM chain equals the naive loop run in the same two phases, and is NOT the
  single packet over all the blocks - the order differs, which is the reason the order is modelled;
- F_HOLD leaves the emitted tile's bit patterns in the 800 float32 slots past the partial sums;
- F_RELU and F_RELU6 floor at +0.0 and cap at 6.0;
- the three multiply-accumulate models give three different answers on the cancellation vectors
  the silicon probe sends, so that probe can tell them apart.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.compiler import engine_bf16_emulator as em  # noqa: E402

F32 = np.float32


def scalar_mac(acc, a8, w8, model):
    """One output lane of one multiply-accumulate, by the model's definition and nothing else."""
    if model == "aligned":
        ops = [float(acc)] + [float(a) * float(w) for a, w in zip(a8, w8)]
        peak = max(abs(o) for o in ops)
        if peak == 0.0:
            return F32(0.0)
        quantum = 2.0 ** (math.frexp(peak)[1] - 24)          # a 24-bit grid under the largest operand
        return F32(sum(round(o / quantum) for o in ops) * quantum)   # round() is ties-to-even
    if model == "wide":
        return F32(math.fsum([float(acc)] + [float(a) * float(w) for a, w in zip(a8, w8)]))
    if model == "sequential":
        for a, w in zip(a8, w8):
            acc = F32(acc + F32(a * w))
        return acc
    dot = F32(0.0)
    for a, w in zip(a8, w8):
        dot = F32(dot + F32(a * w))
    return F32(acc + dot)


def naive_accumulate(header, act, wts, start, model, every=1):
    """Partial sums after one packet, from `start` ([NCO][R][C][4] float32), in the core's walk."""
    k, stride, ncin = (int(header[i]) for i in (em.H_K, em.H_STRIDE, em.H_NCIN))
    cols_in, plane = int(header[em.H_COLS_IN]), int(header[em.H_PLANE_ELEMS])
    acc = start.copy()
    lanes = [(b, r, x, n) for b in range(em.NCO) for r in range(em.TILE_ROWS)
             for x in range(em.TILE_COLS) for n in range(4)][::every]
    for b, r, x, n in lanes:
        v, wp = acc[b, r, x, n], 0
        for ky in range(k):
            for kx in range(k):
                for c in range(ncin):
                    base = c * plane + ((r * stride + ky) * cols_in + x * stride + kx) * 8
                    w32 = wts[wp + b * 32: wp + b * 32 + 32]
                    v = scalar_mac(v, act[base:base + 8], w32[n::4], model)
                    wp += em.NCO * 32
        acc[b, r, x, n] = v
    return acc, lanes


def packet(k, stride, ncin, flags, seed):
    rows_in = (em.TILE_ROWS - 1) * stride + k
    cols_in = (em.TILE_COLS - 1) * stride + k
    plane = rows_in * cols_in * 8
    rng = np.random.default_rng(seed)
    act = em.to_bf16(rng.normal(size=ncin * plane).astype(F32))
    wts = em.to_bf16(rng.normal(scale=0.25, size=k * k * ncin * em.NCO * 32).astype(F32))
    per_ch = em.to_bf16(rng.normal(scale=0.1, size=em.NCO * 4).astype(F32)).reshape(em.NCO, 4)
    bias = np.repeat(per_ch[:, None, :], 4, axis=1).reshape(-1)
    header = em.make_header(k=k, stride=stride, ncin=ncin, flags=flags,
                            rows_in=rows_in, cols_in=cols_in, plane_elems=plane)
    return header, act, wts, bias


def bias_start(bias):
    start = np.empty((em.NCO, em.TILE_ROWS, em.TILE_COLS, 4), F32)
    start[:] = bias.reshape(em.NCO, 4, 4)[:, 0, :].reshape(em.NCO, 1, 1, 4)
    return start


class Bf16Codec(unittest.TestCase):
    def test_round_to_nearest_even_on_the_bits(self):
        one, ulp = F32(1.0), F32(2.0 ** -7)               # bf16 keeps 7 fraction bits
        half = F32(2.0 ** -8)
        self.assertEqual(em.to_bf16(np.array([one + half]))[0], one)               # tie -> even (down)
        self.assertEqual(em.to_bf16(np.array([one + ulp + half]))[0], one + 2 * ulp)  # tie -> even (up)
        self.assertEqual(em.to_bf16(np.array([one + half + F32(2.0 ** -20)]))[0], one + ulp)
        self.assertTrue(np.isinf(em.to_bf16(np.array([np.finfo(F32).max]))[0]))
        self.assertTrue(np.isnan(em.to_bf16(np.array([np.nan], F32))[0]))

    def test_sign_of_zero_survives_and_is_visible_in_the_bits(self):
        z = em.to_bf16(np.array([0.0, -0.0], F32))
        self.assertTrue(np.array_equal(z, np.array([0.0, -0.0], F32)))   # value-equal, which proves nothing
        self.assertEqual(em.bf16_bits(z).tolist(), [0x0000, 0x8000])

    def test_bits_round_trip(self):
        v = em.to_bf16(np.random.default_rng(3).normal(scale=50, size=4096).astype(F32))
        self.assertTrue(np.array_equal(em.from_bf16_bits(em.bf16_bits(v)).view(np.uint32), v.view(np.uint32)))

    def test_agrees_with_ml_dtypes_when_it_is_installed(self):
        try:
            from ml_dtypes import bfloat16
        except ImportError:
            self.skipTest("ml_dtypes is not installed")
        v = np.random.default_rng(4).normal(scale=1e3, size=20000).astype(F32)
        self.assertTrue(np.array_equal(em.bf16_bits(em.to_bf16(v)), v.astype(bfloat16).view(np.uint16)))


class PacketAgainstNaiveLoop(unittest.TestCase):
    SHAPES = [dict(k=1, stride=1, ncin=3, every=1), dict(k=3, stride=1, ncin=2, every=3),
              dict(k=3, stride=2, ncin=1, every=3), dict(k=5, stride=1, ncin=1, every=7)]

    def test_every_model_every_shape(self):
        for shape in self.SHAPES:
            every = shape["every"]
            header, act, wts, bias = packet(shape["k"], shape["stride"], shape["ncin"], em.F_EMIT, seed=11)
            for model in em.MAC_MODELS:
                with self.subTest(model=model, **shape):
                    want, lanes = naive_accumulate(header, act, wts, bias_start(bias), model, every)
                    psum, out = np.zeros(em.PSUM_FLOATS, F32), np.zeros(em.NCO * em.OUT_BLOCK_ELEMS, F32)
                    em.run_packet(header, act, wts, bias, psum, out, mac_model=model)
                    got = out.reshape(em.NCO, em.TILE_ROWS, em.TILE_COLS, 4)
                    for lane in lanes:
                        self.assertEqual(em.bf16_bits(got[lane]), em.bf16_bits(em.to_bf16(want[lane])), lane)

    def test_partial_sums_stay_fp32_and_match_bit_for_bit(self):
        header, act, wts, bias = packet(3, 1, 2, 0, seed=12)       # no F_EMIT, no F_HOLD: sums stay in psum
        for model in em.MAC_MODELS:
            want, lanes = naive_accumulate(header, act, wts, bias_start(bias), model, every=5)
            psum, out = np.zeros(em.PSUM_FLOATS, F32), np.zeros(em.NCO * em.OUT_BLOCK_ELEMS, F32)
            em.run_packet(header, act, wts, bias, psum, out, mac_model=model)
            got = psum[:em.HOLD_OFFSET_ELEMS].reshape(want.shape)
            for lane in lanes:
                self.assertEqual(got[lane].view(np.uint32), want[lane].view(np.uint32), (model, lane))
            self.assertFalse(out.any(), "an accumulate-only packet must not write the output tile")

    def test_load_psum_chain_is_two_phases_not_one_packet(self):
        h1, act1, w1, bias = packet(3, 1, 2, 0, seed=13)
        h2, act2, w2, _ = packet(3, 1, 2, em.F_LOAD_PSUM | em.F_EMIT, seed=14)
        model = "sequential"
        mid, _ = naive_accumulate(h1, act1, w1, bias_start(bias), model, every=1)
        want, lanes = naive_accumulate(h2, act2, w2, mid, model, every=9)
        psum, out = np.zeros(em.PSUM_FLOATS, F32), np.zeros(em.NCO * em.OUT_BLOCK_ELEMS, F32)
        em.run_packet(h1, act1, w1, bias, psum, out, mac_model=model)
        em.run_packet(h2, act2, w2, bias, psum, out, mac_model=model)
        got = out.reshape(want.shape)
        for lane in lanes:
            self.assertEqual(em.bf16_bits(got[lane]), em.bf16_bits(em.to_bf16(want[lane])), lane)

        # The same four blocks as ONE packet walk tap-outer, block-inner: a different order of the
        # same additions. In fp32 the partial sums differ, which is why the order is modelled at all.
        k, plane = 3, int(h1[em.H_PLANE_ELEMS])
        both_w = np.concatenate([w1.reshape(k, k, 2, -1), w2.reshape(k, k, 2, -1)], axis=2).reshape(-1)
        one = em.make_header(k=3, stride=1, ncin=4, flags=0, rows_in=7, cols_in=22, plane_elems=plane)
        p1, p2 = np.zeros(em.PSUM_FLOATS, F32), np.zeros(em.PSUM_FLOATS, F32)
        em.run_packet(one, np.concatenate([act1, act2]), both_w, bias, p1, out.copy(), mac_model=model)
        em.run_packet(h1, act1, w1, bias, p2, out.copy(), mac_model=model)
        chained = em.make_header(k=3, stride=1, ncin=2, flags=em.F_LOAD_PSUM, rows_in=7, cols_in=22, plane_elems=plane)
        em.run_packet(chained, act2, w2, bias, p2, out.copy(), mac_model=model)
        sums = slice(0, em.HOLD_OFFSET_ELEMS)
        self.assertTrue(np.allclose(p1[sums], p2[sums], rtol=1e-4, atol=1e-4))
        self.assertFalse(np.array_equal(p1[sums].view(np.uint32), p2[sums].view(np.uint32)))

    def test_hold_stores_the_emitted_bits_at_the_cores_width(self):
        header, act, wts, bias = packet(3, 1, 1, em.F_EMIT | em.F_RELU, seed=15)
        psum, out = np.zeros(em.PSUM_FLOATS, F32), np.zeros(em.NCO * em.OUT_BLOCK_ELEMS, F32)
        em.run_packet(header, act, wts, bias, psum, out)
        held_hdr = header.copy()
        held_hdr[em.H_FLAGS] = em.F_HOLD | em.F_RELU
        psum2, out2 = np.zeros(em.PSUM_FLOATS, F32), np.zeros(em.NCO * em.OUT_BLOCK_ELEMS, F32)
        em.run_packet(held_hdr, act, wts, bias, psum2, out2)             # 2,400 floats: used to raise
        self.assertEqual(em.PSUM_FLOATS, 2400)
        self.assertFalse(out2.any(), "a held tile is not emitted")
        self.assertTrue(np.array_equal(em.bf16_bits(em.held_tile(psum2)), em.bf16_bits(out)))
        tail = psum2[em.HOLD_OFFSET_ELEMS:].view(np.uint8)
        self.assertTrue(np.array_equal(tail, em.bf16_bits(out).view(np.uint8)), "same bytes, same offsets")

    def test_relu_floor_and_relu6_ceiling(self):
        header, act, wts, bias = packet(3, 1, 2, em.F_EMIT, seed=16)
        raw, relu, relu6 = (np.zeros(em.NCO * em.OUT_BLOCK_ELEMS, F32) for _ in range(3))
        for flags, out in ((em.F_EMIT, raw), (em.F_EMIT | em.F_RELU, relu),
                           (em.F_EMIT | em.F_RELU | em.F_RELU6, relu6)):
            header[em.H_FLAGS] = flags
            em.run_packet(header, act * F32(4), wts, bias, np.zeros(em.PSUM_FLOATS, F32), out)
        self.assertTrue((raw < 0).any() and (raw > 6).any(), "the data must exercise both clamps")
        self.assertTrue(np.array_equal(em.bf16_bits(relu), em.bf16_bits(np.where(raw > 0, raw, F32(0)))))
        self.assertTrue(np.array_equal(em.bf16_bits(relu6), em.bf16_bits(np.clip(np.where(raw > 0, raw, F32(0)), 0, 6))))
        self.assertFalse((em.bf16_bits(relu) == 0x8000).any(), "the floor is +0.0, never -0.0")


class ModelsAreDistinguishable(unittest.TestCase):
    """The vectors the silicon probe sends: one lane, k = 1, one input block, exact powers of two."""

    @staticmethod
    def lane(products, bias):
        header = em.make_header(k=1, stride=1, ncin=1, flags=em.F_EMIT, rows_in=5, cols_in=20)
        act = np.zeros(5 * 20 * 8, F32)
        wts = np.zeros(em.NCO * 32, F32)
        for kk, (a, w) in enumerate(products):
            act[kk], wts[kk * 4] = a, w                          # pixel 0, block 0, output channel 0
        b = np.zeros(em.NCO * 16, F32)
        b[0::4][:4] = bias                                       # channel 0 of block 0, replicated
        got = {}
        for model in em.MAC_MODELS:
            out = np.zeros(em.NCO * em.OUT_BLOCK_ELEMS, F32)
            em.run_packet(header, act, wts, b, np.zeros(em.PSUM_FLOATS, F32), out, mac_model=model)
            got[model] = float(out[0])
        return got

    def test_a_small_product_between_two_cancelling_large_ones(self):
        big = (F32(2 ** 13), F32(2 ** 12))
        got = self.lane([big, (F32(1), F32(1)), (big[0], -big[1])], bias=0.0)
        self.assertEqual(got, {"aligned": 0.0, "wide": 1.0, "sequential": 0.0, "dot_first": 0.0})

    def test_a_small_product_after_the_large_ones_have_cancelled(self):
        # The one that separates the silicon from every ordered model: in order, 2^25 - 2^25 is 0 and
        # the +1 survives; aligned to 2^25 first, the +1 is a quarter of a grid step and is gone.
        big = (F32(2 ** 13), F32(2 ** 12))
        got = self.lane([big, (big[0], -big[1]), (F32(1), F32(1))], bias=0.0)
        self.assertEqual(got, {"aligned": 0.0, "wide": 1.0, "sequential": 1.0, "dot_first": 1.0})

    def test_small_products_against_a_large_accumulator(self):
        ones = [(F32(1), F32(1))] * 4
        got = self.lane(ones + [(F32(2 ** 13), -F32(2 ** 12))], bias=2.0 ** 25)
        self.assertEqual(got, {"aligned": 0.0, "wide": 4.0, "sequential": 0.0, "dot_first": 4.0})

    def test_the_grid_rounds_ties_to_even(self):
        # What the silicon returned for (2^24, +s, -2^24): s = 1 -> 0, 3 -> 4, 5 -> 4. Truncation
        # would give 0, 2, 4 and round-half-up 2, 4, 6.
        big = (F32(2 ** 12), F32(2 ** 12))
        for small, want in ((1, 0.0), (3, 4.0), (5, 4.0)):
            got = self.lane([big, (F32(small), F32(1)), (big[0], -big[1])], bias=0.0)
            self.assertEqual(got["aligned"], want, small)


class Header(unittest.TestCase):
    def test_words_land_where_the_core_reads_them(self):
        h = em.make_header(op=em.OP_CONV, k=5, stride=2, ncin=3, flags=em.F_EMIT | em.F_RELU6,
                           count_out=2, count_acc=1, rows_in=9, cols_in=24, phases=(4, 5, 6, 7))
        self.assertEqual(h.dtype, np.int32)
        self.assertEqual(h.size * 4, em.HDR_BYTES)
        self.assertEqual([int(h[i]) for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)], [1, 5, 2, 3, 10, 2, 1, 9, 24, 9 * 24 * 8])
        self.assertEqual(h[12:16].tolist(), [4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main(verbosity=2)
