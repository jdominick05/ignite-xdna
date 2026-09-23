"""Offline checks of the bf16 packer fork (engine_schedule_bf16) and its reference (graph_reference_bf16).

Synthetic graphs, so these run on any machine. The packet layout is checked against a float64
convolution written here from scratch - the one check that shares nothing with the packer - at a
tolerance derived from bf16's rounding rather than picked. Everything else pins a refusal or a
contract the packer must enforce.
"""
import unittest

import numpy as np

from ignite_xdna.compiler import engine_bf16_emulator as em
from ignite_xdna.compiler import engine_schedule_bf16 as eb
from ignite_xdna.compiler import graph_ir
from ignite_xdna.compiler import graph_reference_bf16 as gr16
from ignite_xdna.compiler.engine_schedule import layer_chunks
from ignite_xdna.compiler.graph_ir import ConvLayer, GraphIR, Segment, TensorInfo

W_SCALE = 2.0 ** -6
B_SCALE = 2.0 ** -10


def blocks(c):
    return -(-c // 8)


def conv(name, src, cin, cout, k, out, stride=1, act=None, seed=0, residual=None, w_scale=W_SCALE):
    rng = np.random.default_rng(seed)
    layer = ConvLayer(name=name, index=0, inputs=[Segment(src, 0, blocks(cin))], in_scale=1.0, k=k,
                      stride=stride, pad=k // 2, weights=rng.integers(-127, 128, (cout, cin, k, k)).astype(np.int8),
                      bias_q=rng.integers(-3000, 3000, cout).astype(np.int32), bias_scale=B_SCALE,
                      weight_scale=w_scale, conv_scale=1.0, output=out, act=act)
    if residual is not None:
        layer.residual = Segment(residual, 0, blocks(cout))
    return layer


def graph(layers, shapes, inp="x"):
    """``shapes``: tensor name -> (channels, height, width)."""
    tensors = {n: TensorInfo(n, c, h, w, 1.0, 128, producer="input" if n == inp else n)
               for n, (c, h, w) in shapes.items()}
    for i, L in enumerate(layers):
        L.index = i
    return GraphIR(tensors=tensors, layers=layers, input=inp, outputs=[("y", layers[-1].output)])


def bf16_tensor(rng, c, h, w):
    """Channel-blocked [blocks*8][H][W] of bf16 values, zero in the channels past ``c``."""
    x = np.zeros((blocks(c) * 8, h, w), np.float32)
    x[:c] = em.to_bf16(rng.normal(scale=2.0, size=(c, h, w)).astype(np.float32))
    return x


def conv_f64(x, layer):
    """float64 convolution of real values, and the sum of |terms| per output (the rounding scale)."""
    w = layer.weights.astype(np.float64) * layer.weight_scale
    b = em.to_bf16(eb.dequantized_bias(layer)).astype(np.float64)
    k, s, p = layer.k, layer.stride, layer.pad
    xp = np.pad(x[:layer.cin].astype(np.float64), ((0, 0), (p, p), (p, p)))
    ho = (x.shape[1] + 2 * p - k) // s + 1
    wo = (x.shape[2] + 2 * p - k) // s + 1
    acc = np.repeat(b[:, None, None], ho * wo, axis=1).reshape(-1, ho, wo)
    mag = np.abs(acc).copy()
    for ky in range(k):
        for kx in range(k):
            patch = xp[:, ky:ky + s * ho:s, kx:kx + s * wo:s]
            acc += np.tensordot(w[:, :, ky, kx], patch, axes=([1], [0]))
            mag += np.tensordot(np.abs(w[:, :, ky, kx]), np.abs(patch), axes=([1], [0]))
    return acc, mag


class PacketLayout(unittest.TestCase):
    """A packet's weights, bias, flags and chunking against a float64 convolution.

    The tolerance is bf16's, not a fudge: the store rounds to 8 significant bits, a relative error
    of at most 2**-8 per rounding, and a residual layer rounds twice (held tile, then sum). The
    accumulator's aligned 24-bit grid adds at most a few units of 2**-24 of the terms' magnitude per
    multiply-accumulate, bounded here by 2**-16 of their sum. A packet with a channel in the wrong
    place is wrong by the size of a value, far outside both.
    """

    def check(self, ir, layer, x, res=None):
        tensors = {ir.input: x}
        if res is not None:
            tensors[layer.residual.tensor] = res
        got = gr16.direct_layer(ir, layer, tensors)[:layer.cout]
        acc, mag = conv_f64(x, layer)
        want = np.maximum(acc, 0.0) if layer.act == "relu" else acc
        scale = np.abs(want)
        if res is not None:
            scale = scale + np.abs(want + res[:layer.cout])
            want = want + res[:layer.cout]
        err = np.abs(got.astype(np.float64) - want)
        bound = 2.0 ** -7 * scale + 2.0 ** -16 * mag + 1e-30
        self.assertTrue(np.all(err <= bound), f"{layer.name}: worst err/bound {np.max(err / bound):.3f}")
        self.assertTrue(np.all(np.isfinite(got)))
        return got

    def test_k1_two_groups_and_an_overlapping_edge_tile(self):
        rng = np.random.default_rng(1)
        L = conv("k1", "x", 20, 24, 1, "y", act="relu", seed=1)
        ir = graph([L], {"x": (20, 20, 36), "y": (24, 20, 36)})
        self.check(ir, L, bf16_tensor(rng, 20, 20, 36))

    def test_k3s1_two_chunks_accumulate_through_psum(self):
        rng = np.random.default_rng(2)
        L = conv("k3", "x", 40, 16, 3, "y", act="relu", seed=2)
        ir = graph([L], {"x": (40, 20, 20), "y": (16, 20, 20)})
        self.assertEqual([c.kind for c in layer_chunks(ir, L)], ["k3s1", "k3s1"])
        self.check(ir, L, bf16_tensor(rng, 40, 20, 20))

    def test_k3s2_downsamples(self):
        rng = np.random.default_rng(3)
        L = conv("k3s2", "x", 8, 16, 3, "y", stride=2, seed=3)
        ir = graph([L], {"x": (8, 40, 40), "y": (16, 20, 20)})
        self.check(ir, L, bf16_tensor(rng, 8, 40, 40))

    def test_k5s1_on_a_three_channel_image(self):
        rng = np.random.default_rng(4)
        L = conv("k5", "x", 3, 16, 5, "y", seed=4)
        ir = graph([L], {"x": (3, 20, 40), "y": (16, 20, 40)})
        self.check(ir, L, bf16_tensor(rng, 3, 20, 40))

    def test_k5s1_two_input_blocks_and_a_partial_output_group(self):
        rng = np.random.default_rng(5)
        L = conv("tail", "x", 16, 12, 5, "y", seed=5)
        ir = graph([L], {"x": (16, 20, 20), "y": (12, 20, 20)})
        self.assertEqual([c.kind for c in layer_chunks(ir, L)], ["k5s1", "k5s1"])
        got = self.check(ir, L, bf16_tensor(rng, 16, 20, 20))
        self.assertEqual(got.shape[0], 12)

    def test_relu_then_residual_add(self):
        rng = np.random.default_rng(6)
        L = conv("body", "x", 16, 16, 3, "y", act="relu", seed=6, residual="r")
        ir = graph([L], {"x": (16, 20, 20), "r": (16, 20, 20), "y": (16, 20, 20)})
        self.assertEqual([c.kind for c in layer_chunks(ir, L)], ["k3s1", "res"])
        self.check(ir, L, bf16_tensor(rng, 16, 20, 20), res=bf16_tensor(rng, 16, 20, 20))


class PacketContract(unittest.TestCase):
    def setUp(self):
        self.L = conv("k3", "x", 40, 16, 3, "y", act="relu", seed=7)
        self.ir = graph([self.L], {"x": (40, 20, 20), "y": (16, 20, 20)})
        self.first, self.last = layer_chunks(self.ir, self.L)

    def test_every_chunk_packet_satisfies_the_contract(self):
        for g_pkts in eb.layer_packets(self.ir, self.L):
            for p in g_pkts:
                eb.check_counts(eb.unpack_w(p)[0])

    def test_both_counts_set_is_refused(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            eb.conv_packet(self.L, 0, self.last, count_out=1, count_acc=1)

    def test_an_accumulating_chunk_counted_as_output_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unwritten"):
            eb.conv_packet(self.L, 0, self.first, count_out=1, count_acc=0)

    def test_an_emitting_chunk_counted_as_accumulate_is_refused(self):
        with self.assertRaisesRegex(ValueError, "overwrite the held tile"):
            eb.conv_packet(self.L, 0, self.last, count_out=0, count_acc=1)

    def test_a_holding_residual_counted_as_output_is_refused(self):
        h = em.make_header(op=em.OP_RESIDUAL, flags=em.F_HOLD, count_out=1, count_acc=0)
        with self.assertRaisesRegex(ValueError, "unwritten"):
            eb.pack_w(h, np.zeros(em.BIAS_ELEMS, np.float32), np.zeros(0, np.float32))

    def test_the_held_chunk_of_a_residual_layer_accumulates(self):
        L = conv("body", "x", 16, 16, 3, "y", act="relu", residual="r")
        ir = graph([L], {"x": (16, 20, 20), "r": (16, 20, 20), "y": (16, 20, 20)})
        hold, res = [eb.unpack_w(p)[0] for p in eb.layer_packets(ir, L)[0]]
        self.assertEqual(int(hold[em.H_FLAGS]), em.F_HOLD | em.F_RELU)
        self.assertEqual((int(hold[em.H_COUNT_OUT]), int(hold[em.H_COUNT_ACC])), (0, 1))
        self.assertEqual((int(res[em.H_OP]), int(res[em.H_FLAGS])), (em.OP_RESIDUAL, em.F_EMIT))

    def test_ncin_is_trimmed_to_the_real_blocks(self):
        L = conv("k3", "x", 16, 16, 3, "y")
        ir = graph([L], {"x": (16, 20, 20), "y": (16, 20, 20)})
        (chunk,) = layer_chunks(ir, L)
        self.assertEqual(chunk.ncin, 4)
        self.assertEqual(int(eb.unpack_w(eb.chunk_packet(L, 0, chunk))[0][em.H_NCIN]), 2)

    def test_plane_elems_is_the_element_count_not_doubled(self):
        h = eb.unpack_w(eb.chunk_packet(self.L, 0, self.first))[0]
        self.assertEqual(int(h[em.H_PLANE_ELEMS]), 8 * 25 * 8)

    def test_bias_is_replicated_to_the_4x4_accumulator_shape(self):
        _, bias, _ = eb.unpack_w(eb.chunk_packet(self.L, 0, self.first))
        per_ch = em.to_bf16(eb.dequantized_bias(self.L))
        b = bias.reshape(em.NCO, 4, 4)
        for m in range(4):
            np.testing.assert_array_equal(b[:, m, :].reshape(-1), per_ch)

    def test_a_non_power_of_two_weight_scale_is_refused(self):
        L = conv("k3", "x", 16, 16, 3, "y", w_scale=0.3)
        ir = graph([L], {"x": (16, 20, 20), "y": (16, 20, 20)})
        with self.assertRaisesRegex(ValueError, "not exact in bf16"):
            eb.layer_packets(ir, L)

    def test_a_chunk_with_no_real_channel_is_refused(self):
        L = conv("k5", "x", 8, 16, 5, "y")
        L.inputs = [Segment("x", 0, 2)]      # two blocks planned, eight real channels
        ir = graph([L], {"x": (16, 20, 20), "y": (16, 20, 20)})
        with self.assertRaisesRegex(ValueError, "no real input channel"):
            eb.layer_packets(ir, L)


class Admission(unittest.TestCase):
    def ir_with(self, layer):
        return graph([layer], {"x": (16, 20, 20), "y": (16, 20, 20)})

    def test_a_plain_relu_graph_is_admitted(self):
        eb.check_ir(self.ir_with(conv("ok", "x", 16, 16, 3, "y", act="relu")))

    def test_hardswish_is_refused(self):
        L = conv("hs", "x", 16, 16, 3, "y", act="hswish")
        with self.assertRaisesRegex(ValueError, "hs: activation 'hswish'"):
            eb.check_ir(self.ir_with(L))

    def test_an_activation_after_the_residual_add_is_refused(self):
        L = conv("post", "x", 16, 16, 3, "y", residual="x")
        L.post_hswish = graph_ir.relu_epilogue()
        with self.assertRaisesRegex(ValueError, "post: sigmoid or post-residual"):
            eb.check_ir(self.ir_with(L))

    def test_an_upsampled_input_is_refused(self):
        L = conv("up", "x", 16, 16, 1, "y")
        L.inputs = [Segment("x", 0, 2, up2=True)]
        with self.assertRaisesRegex(ValueError, "up: upsampled"):
            eb.check_ir(self.ir_with(L))

    def test_a_pool_is_refused(self):
        ir = self.ir_with(conv("ok", "x", 16, 16, 3, "y"))
        ir.layers.append(graph_ir.PoolLayer("pool", 1, Segment("y", 0, 2), "z"))
        with self.assertRaisesRegex(ValueError, "pool: max pool"):
            eb.check_ir(ir)

    def test_a_host_layer_is_refused(self):
        ir = self.ir_with(conv("ok", "x", 16, 16, 3, "y"))
        ir.layers.append(graph_ir.HostLayer("host", 1, Segment("y", 0, 2), "z", b""))
        with self.assertRaisesRegex(ValueError, "host: a host layer"):
            eb.check_ir(ir)

    def test_a_kernel_size_with_no_packet_geometry_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no packet geometry"):
            eb.check_ir(self.ir_with(conv("k7", "x", 16, 16, 7, "y")))


if __name__ == "__main__":
    unittest.main()
