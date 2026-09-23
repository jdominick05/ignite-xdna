"""Offline checks of the bf16 packer fork (engine_schedule_bf16) and its reference (graph_reference_bf16).

Synthetic graphs, so these run on any machine. The packet layout is checked against a float64
convolution written here from scratch - the one check that shares nothing with the packer - at a
tolerance derived from bf16's rounding rather than picked. Everything else pins a refusal or a
contract the packer must enforce.
"""
import unittest
from pathlib import Path

import numpy as np

from ignite_xdna.compiler import engine_bf16_emulator as em
from ignite_xdna.compiler import engine_schedule_bf16 as eb
from ignite_xdna.compiler import graph_ir
from ignite_xdna.compiler import graph_reference_bf16 as gr16
from ignite_xdna.compiler.engine_schedule import layer_chunks, tile_origins
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


def chain_ir():
    """Five layers covering every bf16 chunk kind the fork admits, on maps that overlap their edge tiles.

    x(3) -k5-> a(16) -k3 relu, + a-> b(16) -k1 relu-> c(40: 5 blocks, 3 groups, a junk plane)
         -k3, two chunks-> d(16) -k3 stride 2-> e(24: 3 blocks, 2 groups).
    40 x 44 tiles as rows 0, 20 and columns 0, 20, 24; e at 20 x 22 as columns 0 and 2.
    """
    layers = [conv("head", "x", 3, 16, 5, "a", seed=10),
              conv("body", "a", 16, 16, 3, "b", act="relu", seed=11, residual="a"),
              conv("wide", "b", 16, 40, 1, "c", act="relu", seed=12),
              conv("deep", "c", 40, 16, 3, "d", seed=13),
              conv("down", "d", 16, 24, 3, "e", stride=2, seed=14)]
    return graph(layers, {"x": (3, 40, 44), "a": (16, 40, 44), "b": (16, 40, 44), "c": (40, 40, 44),
                          "d": (16, 40, 44), "e": (24, 20, 22)})


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


class WorkspaceAndPatterns(unittest.TestCase):
    """Placements, the halo, and every DMA pattern a chain's layers can issue - checked per pattern.

    The window check is the direct test of the trap the int8 byte literals set: it reads each fill
    through its DMA pattern from a workspace holding distinct random values and compares the
    positions the core multiplies against the tensor, indexed here from scratch.
    """

    @classmethod
    def setUpClass(cls):
        cls.ir = chain_ir()
        cls.ws = eb.plan_workspace(cls.ir)
        rng = np.random.default_rng(20)
        cls.values = {n: bf16_tensor(rng, t.channels, t.height, t.width) for n, t in cls.ir.tensors.items()}

    def tiles(self, L):
        t = self.ir.tensors[L.output]
        return [(y, x) for y in tile_origins(t.height, 20) for x in tile_origins(t.width, 20)]

    def test_placements_are_bf16_storage_with_zero_halos_and_even_planes(self):
        for name, p in self.ws.placements.items():
            self.assertEqual((p.dtype, p.halo_value, p.band_rows), ("uint16", 0, 0), name)
            if name != self.ir.input:
                self.assertEqual(p.planes % eb.OUT_BLOCKS_8, 0, name)
        self.assertEqual(self.ws.placements["c"].planes, 6)
        self.assertEqual(self.ws.placements["x"].planes, 1)
        self.assertEqual({n: p.halo for n, p in self.ws.placements.items()},
                         {"x": 2, "a": 1, "b": 0, "c": 1, "d": 1, "e": 0})

    def test_reuse_shares_a_slot_and_never_in_space_and_time_at_once(self):
        life = {"x": (0, 0), "a": (0, 1), "b": (1, 2), "c": (2, 3), "d": (3, 4), "e": (4, 6)}
        ps = list(self.ws.placements.values())
        self.assertEqual(self.ws.placements["a"].base, self.ws.placements["c"].base)
        for i, a in enumerate(ps):
            for b in ps[i + 1:]:
                space = max(a.base, b.base) < min(a.base + a.nbytes, b.base + b.nbytes)
                time_ = max(life[a.name][0], life[b.name][0]) <= min(life[a.name][1], life[b.name][1])
                self.assertFalse(space and time_, f"{a.name} and {b.name}")
            self.assertEqual(a.base % 64, 0)

    def test_without_reuse_every_tensor_owns_its_slot(self):
        ws = eb.plan_workspace(self.ir, reuse=False)
        spans = sorted((p.base, p.base + p.nbytes) for p in ws.placements.values())
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            self.assertLessEqual(a1, b0)

    def test_every_fill_is_one_whole_packet_inside_the_workspace(self):
        for L in self.ir.layers:
            t = self.ir.tensors[L.output]
            for ch in layer_chunks(self.ir, L):
                for g in range(eb.output_groups(t.blocks)):
                    for y, x in self.tiles(L):
                        p = eb.a_pattern(self.ws, L, ch, y, x, g)
                        idx = p.indices()
                        self.assertEqual(p.nbytes, em.A_BYTES, (L.name, ch.kind))
                        self.assertEqual(p.offset % 4, 0)
                        self.assertTrue(all(s <= eb.MAX_STRIDE_BYTES for s in p.strides))
                        self.assertGreaterEqual(int(idx.min()), 0)
                        self.assertLess(int(idx.max()), self.ws.nbytes)

    def test_every_drain_moves_whole_joined_objects(self):
        for L in self.ir.layers:
            t = self.ir.tensors[L.output]
            for g in range(eb.output_groups(t.blocks)):
                self.assertEqual(eb.o_pattern(self.ws, L, g, 0, 0).nbytes, 4 * em.O_BYTES)
                self.assertEqual(eb.run_drain(self.ws, L, g, [(0, 0)]).nbytes, 4 * em.O_BYTES)
        L = self.ir.layers[0]
        self.assertEqual(eb.run_drain(self.ws, L, 0, [(0, 0), (20, 0)]).nbytes, 8 * em.O_BYTES)

    def test_every_fill_delivers_the_right_pixels_where_the_core_reads(self):
        self.check_fills(self.ir, self.ws, self.values)

    def test_a_two_group_residual_reads_each_groups_own_blocks(self):
        # The chain's residual layer has one output group, which cannot tell a group stride of two
        # blocks from int8's four. This one has two.
        L = conv("res32", "x", 32, 32, 3, "y", act="relu", seed=21, residual="x")
        ir = graph([L], {"x": (32, 20, 40), "y": (32, 20, 40)})
        rng = np.random.default_rng(21)
        values = {n: bf16_tensor(rng, t.channels, t.height, t.width) for n, t in ir.tensors.items()}
        self.check_fills(ir, eb.plan_workspace(ir), values)

    def check_fills(self, ir, ws, values):
        for L in ir.layers:
            t = ir.tensors[L.output]
            arr = ws.halo_fill()
            for n in [s.tensor for s in L.inputs] + ([L.residual.tensor] if L.residual else []):
                ws.write_values(arr, n, values[n])
            for ch in layer_chunks(ir, L):
                for g in range(eb.output_groups(t.blocks)):
                    for y in tile_origins(t.height, 20):
                        for x in tile_origins(t.width, 20):
                            for r in range(4):
                                y0 = min(y + 5 * r, t.height - 5)
                                got = eb.a_pattern(ws, L, ch, y0, x, g).read(arr).view(np.uint16)
                                got = got.reshape(ch.read_blocks, ch.rows_in, ch.cols_in, 8)
                                self.check_window(values, L, ch, g, y0, x, got)

    def check_window(self, values, L, ch, g, y0, x0, got):
        if ch.kind == "res":
            src = values[L.residual.tensor]
            c0 = (L.residual.block_offset + 2 * g) * 8
            want = src[c0:c0 + 16, y0:y0 + 5, x0:x0 + 20].reshape(2, 8, 5, 20).transpose(0, 2, 3, 1)
            np.testing.assert_array_equal(got[:2], em.bf16_bits(want), str((L.name, "res", g, y0, x0)))
            return
        seg = L.inputs[ch.seg_index]
        src = values[seg.tensor]
        m = 8
        xp = np.pad(src, ((0, 0), (m, m), (m, m)))
        rows = L.stride * 4 + L.k
        cols = L.stride * 19 + L.k
        oy, ox = L.stride * y0 - L.pad + m, L.stride * x0 - L.pad + m
        ncin = -(-min(ch.ncin * 8, L.cin - ch.block_start * 8) // 8)
        b0 = seg.block_offset + ch.block_start
        want = xp[b0 * 8:(b0 + ncin) * 8, oy:oy + rows, ox:ox + cols].reshape(ncin, 8, rows, cols).transpose(0, 2, 3, 1)
        np.testing.assert_array_equal(got[:ncin, :rows, :cols], em.bf16_bits(want), str((L.name, ch.kind, y0, x0)))

    def test_halo_fill_zeroes_every_ring_and_leaves_the_background_elsewhere(self):
        nan = 0x7FC0
        arr = self.ws.halo_fill(background=nan)
        for name, p in self.ws.placements.items():
            planes = arr[p.base:p.base + p.planes * p.plane_bytes].view(np.uint16).reshape(
                p.planes, p.height + 2 * p.halo, p.width + 2 * p.halo, 8)
            interior = planes[:, p.halo:p.halo + p.height, p.halo:p.halo + p.width]
            if p.halo:
                ring = planes.copy()
                ring[:, p.halo:p.halo + p.height, p.halo:p.halo + p.width] = 0
                self.assertFalse(ring.any(), name)
            self.assertTrue(np.all(interior == nan), name)

    def test_write_tensor_refuses_float_values(self):
        with self.assertRaisesRegex(TypeError, "bf16_bits"):
            self.ws.write_tensor(self.ws.halo_fill(), "a", self.values["a"])

    def test_a_plane_past_the_descriptor_step_is_refused_at_planning(self):
        L = conv("big", "x", 16, 16, 3, "y")
        ir = graph([L], {"x": (16, 640, 640), "y": (16, 640, 640)})
        with self.assertRaisesRegex(ValueError, "step field"):
            eb.plan_workspace(ir)


def check_streams(case, ir, scheds, store):
    """Stream invariants every column program must satisfy, whatever the graph."""
    from ignite_xdna.compiler.engine_sequence import MAX_REPEAT
    for s in scheds:
        for prog in s.programs:
            a_items = sum(1 for it in prog if it[0] in ("a", "A"))
            w_served = sum(it[3] if it[0] == "w" else it[2] for it in prog if it[0] in ("w", "W"))
            case.assertEqual(w_served, a_items, s.name)
            fill_objects = sum(it[1].nbytes for it in prog if it[0] == "A") // (4 * em.A_BYTES)
            weight_uses = 0
            for it in prog:
                if it[0] == "w":
                    pkts = store.packets_at(it[1], it[2])
                elif it[0] == "W":
                    data = it[1].read(store.blob())
                    pkts = [data[k:k + em.W_BYTES] for k in range(0, data.size, em.W_BYTES)]
                else:
                    pkts = []
                for p in pkts:
                    h = p[:em.HDR_BYTES].view(np.int32)
                    weight_uses += int(h[em.H_COUNT_OUT]) + int(h[em.H_COUNT_ACC])
            # Every activation object the fills deliver is paired with exactly one weight-object use.
            case.assertEqual(weight_uses, fill_objects, s.name)
            for it in prog:
                pats = [it[1]] if it[0] in ("A", "o", "W") else []
                for p in pats:
                    case.assertEqual(p.offset % 4, 0)
                    case.assertLessEqual(len(p.sizes), 4)
                    if len(p.sizes) == 4:
                        case.assertLessEqual(p.sizes[0], MAX_REPEAT, s.name)
                    if p.buffer == "ws":
                        case.assertTrue(all(st <= eb.MAX_STRIDE_BYTES for st in p.strides), s.name)
                if it[0] == "A":
                    case.assertEqual(it[1].nbytes % em.A_BYTES, 0, s.name)
                elif it[0] == "o":
                    case.assertEqual(it[1].nbytes % (4 * em.O_BYTES), 0, s.name)
                elif it[0] == "w":
                    case.assertEqual(it[2] % em.W_BYTES, 0)
    blob = store.blob()
    for k in range(0, blob.size, em.W_BYTES):
        eb.check_counts(blob[k:k + em.HDR_BYTES].view(np.int32))
    case.assertLessEqual(store.opcodes(), {em.OP_CONV, em.OP_RESIDUAL})


class Schedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ir = chain_ir()
        cls.ws = eb.plan_workspace(cls.ir)
        cls.scheds, cls.store = eb.schedule_graph(cls.ir, cls.ws)

    def test_streams_pair_every_activation_object_with_a_weight_use(self):
        check_streams(self, self.ir, self.scheds, self.store)

    def test_rounds_and_packets_follow_the_16_channel_group(self):
        from ignite_xdna.compiler.engine_schedule import tile_origins as to
        for s, L in zip(self.scheds, self.ir.layers):
            t = self.ir.tensors[L.output]
            rounds = eb.output_groups(t.blocks) * len(to(t.height, 20)) * len(to(t.width, 20))
            self.assertEqual((s.rounds, s.packets), (rounds, 4 * rounds * len(layer_chunks(self.ir, L))), L.name)
        self.assertEqual([eb.output_groups(self.ir.tensors[L.output].blocks) for L in self.ir.layers],
                         [1, 1, 3, 1, 2])

    def test_the_flag_variants_keep_the_invariants(self):
        for flags in ({"weight_repeat": False}, {"merge_group_weights": False}, {"balance_columns": False}):
            scheds, store = eb.schedule_graph(self.ir, self.ws, **flags)
            check_streams(self, self.ir, scheds, store)

    def test_the_int8_transport_experiments_are_refused(self):
        with self.assertRaisesRegex(ValueError, "no activation ring"):
            eb.schedule_graph(self.ir, self.ws, activation_ring=4)
        with self.assertRaisesRegex(ValueError, "no activation ring"):
            eb.schedule_graph(self.ir, self.ws, weight_buffer=True)


NAN = 0x7FC0


def halo_ring_is_zero(ws, arr, name):
    p = ws.placements[name]
    if not p.halo:
        return True
    planes = arr[p.base:p.base + p.planes * p.plane_bytes].view(np.uint16).reshape(
        p.planes, p.height + 2 * p.halo, p.width + 2 * p.halo, 8).copy()
    planes[:, p.halo:p.halo + p.height, p.halo:p.halo + p.width] = 0
    return not planes.any()


def emulate_chain(ir, ws, scheds, store, x, background=0, mac_model=None):
    """Run every layer's schedule on one workspace; each output is read back right after its layer,
    because slot reuse may hand its bytes to a later tensor."""
    arr = ws.halo_fill(background=background)
    ws.write_values(arr, ir.input, x)
    outs, rings = {}, {}
    for s, L in zip(scheds, ir.layers):
        eb.emulate_layer(s, store, arr, mac_model=mac_model)
        outs[L.output] = ws.read_tensor(arr, L.output).copy()
        rings[L.output] = halo_ring_is_zero(ws, arr, L.output)
    return outs, rings


class Emulation(unittest.TestCase):
    """The schedule, emulated through its DMA patterns and streams, against the direct reference.

    Both run the same packets through the same core emulator, so they must agree to the bit; what
    differs is everything between DDR and the core. A disagreement is a moved byte.
    """

    @classmethod
    def setUpClass(cls):
        cls.ir = chain_ir()
        cls.ws = eb.plan_workspace(cls.ir)
        cls.scheds, cls.store = eb.schedule_graph(cls.ir, cls.ws)
        cls.x = bf16_tensor(np.random.default_rng(30), 3, 40, 44)
        cls.ref = gr16.run_direct(cls.ir, cls.x)

    def test_each_layer_seeded_from_the_reference_is_bit_exact(self):
        for s, L in zip(self.scheds, self.ir.layers):
            arr = self.ws.halo_fill()
            for n in [g.tensor for g in L.inputs] + ([L.residual.tensor] if L.residual else []):
                self.ws.write_values(arr, n, self.ref[n])
            eb.emulate_layer(s, self.store, arr)
            np.testing.assert_array_equal(self.ws.read_tensor(arr, L.output), em.bf16_bits(self.ref[L.output]),
                                          L.name)

    def test_the_whole_chain_on_one_workspace_is_bit_exact_and_keeps_its_halos(self):
        outs, rings = emulate_chain(self.ir, self.ws, self.scheds, self.store, self.x)
        for L in self.ir.layers:
            np.testing.assert_array_equal(outs[L.output], em.bf16_bits(self.ref[L.output]), L.name)
            self.assertTrue(rings[L.output], f"{L.name} wrote into its halo ring")

    def test_a_nan_background_changes_not_one_bit(self):
        # Everything no layer writes - over-read planes, the space past a tensor's rows, unused slots -
        # holds a NaN. If any multiply-accumulate touched it, even by a zero weight, the output would.
        clean, _ = emulate_chain(self.ir, self.ws, self.scheds, self.store, self.x)
        poisoned, rings = emulate_chain(self.ir, self.ws, self.scheds, self.store, self.x, background=NAN)
        for L in self.ir.layers:
            np.testing.assert_array_equal(poisoned[L.output], clean[L.output], L.name)
            self.assertTrue(np.all(np.isfinite(em.from_bf16_bits(poisoned[L.output]))), L.name)
            self.assertTrue(rings[L.output], L.name)

    def test_the_reference_is_nontrivial(self):
        # Guard against a vacuous pass: outputs are finite, mostly non-zero and differ between layers.
        for L in self.ir.layers:
            y = self.ref[L.output][:self.ir.tensors[L.output].channels]
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertGreater(np.count_nonzero(y) / y.size, 0.3, L.name)


SESR = Path(__file__).resolve().parents[1] / "models" / "sesr_m7_xint8.onnx"


@unittest.skipUnless(SESR.exists(), "sesr_m7 model not present")
class SesrSchedule(unittest.TestCase):
    """SESR-M7 through the fork. These counts are DERIVED - a pin against regression, not a result."""

    @classmethod
    def setUpClass(cls):
        cls.ir = graph_ir.lower_yolov8n(SESR)
        cls.ws = eb.plan_workspace(cls.ir)
        cls.scheds, cls.store = eb.schedule_graph(cls.ir, cls.ws)

    def test_the_counts_match_the_phase_0_recount(self):
        # results/aie/bf16_packet_recount_20260922.log: every SESR layer is 16 channels or fewer, so one
        # 16-channel group serves it and the bf16 counts equal int8's.
        self.assertEqual([(s.rounds, s.packets) for s in self.scheds],
                         [(169, 676)] * 7 + [(169, 1352)] * 2)
        self.assertEqual((sum(s.rounds for s in self.scheds), sum(s.packets for s in self.scheds)), (1521, 7436))

    def test_the_counts_equal_the_int8_schedule_of_the_same_ir(self):
        from ignite_xdna.compiler import engine_schedule as es
        ws8 = es.plan_workspace(self.ir)
        scheds8, _ = es.schedule_graph(self.ir, ws8)
        self.assertEqual([(s.rounds, s.packets) for s in self.scheds], [(s.rounds, s.packets) for s in scheds8])

    def test_the_streams_keep_their_invariants(self):
        check_streams(self, self.ir, self.scheds, self.store)
        self.assertEqual(self.store.opcodes(), {em.OP_CONV, em.OP_RESIDUAL})

    def test_head_residual_and_tail_emulate_bit_exactly(self):
        # "wide" is the fast MAC model; both sides use it, so exactness is unaffected by the choice.
        rng = np.random.default_rng(31)
        values = {n: bf16_tensor(rng, t.channels, t.height, t.width) for n, t in self.ir.tensors.items()}
        values[self.ir.input] = em.to_bf16(np.pad(
            rng.integers(-128, 128, (3, 256, 256)).astype(np.float32), ((0, 5), (0, 0), (0, 0))))
        for i in (0, 7, 8):
            L = self.ir.layers[i]
            arr = self.ws.halo_fill(background=NAN)
            for n in [g.tensor for g in L.inputs] + ([L.residual.tensor] if L.residual else []):
                self.ws.write_values(arr, n, values[n])
            eb.emulate_layer(self.scheds[i], self.store, arr, mac_model="wide")
            want = gr16.direct_layer(self.ir, L, values, mac_model="wide")
            np.testing.assert_array_equal(self.ws.read_tensor(arr, L.output), em.bf16_bits(want), L.name)

    def test_placements_are_bf16_with_the_halos_their_readers_need(self):
        for p in self.ws.placements.values():
            self.assertEqual((p.dtype, p.halo_value), ("uint16", 0))
        # The 5x5 head and tail read at halo 2; head and body.0-5 feed a 3x3; the tail's output none.
        self.assertEqual({n: p.halo for n, p in self.ws.placements.items() if p.halo},
                         {self.ir.input: 2, **{L.output: 1 for L in self.ir.layers[:7]},
                          self.ir.layers[7].output: 2})


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
