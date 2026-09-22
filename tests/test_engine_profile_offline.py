"""The engine profile: one bundle for what differs between the int8 and the bf16 engine.

The test that earns its keep here is the opcode-collision one. The two engines number their opcodes
from zero independently, so 2 is MAXPOOL to one and RESIDUAL to the other, and the packet STRIDE
survives the difference by coincidence - W_BYTES is 9,472 and HDR_BYTES 128 in both, and the opcode
is word 0 in both. That coincidence is what makes the bug quiet: a bf16 blob walks correctly and
comes back named wrongly, and the container is then refused for reaching a case it never reaches.

Offline, no device, no IRON: the design module is named rather than imported, so everything except
``load_design`` resolves without a toolchain.
"""
import unittest

import numpy as np

from ignite_xdna.compiler import engine_bf16_emulator as bf16
from ignite_xdna.compiler import engine_emulator as int8
from ignite_xdna.compiler.engine_compile import (BF16_PROFILE, ENGINE_PROFILES, INT8_PROFILE,
                                                 check_kernel_covers_packets, profile_for_manifest)
from ignite_xdna.compiler.serializer import ELEM_BF16, ELEM_INT, ENGINE_CONV_BF16, ENGINE_CONV_INT8


def wpackets(opcodes, emu) -> bytes:
    """A weight-packet blob carrying these opcodes, in the packet layout ``emu`` describes."""
    out = bytearray()
    for op in opcodes:
        pkt = np.zeros(emu.W_BYTES, dtype=np.uint8)
        header = np.zeros(emu.HDR_BYTES // 4, dtype=np.int32)
        header[emu.H_OP] = op
        pkt[:emu.HDR_BYTES] = header.view(np.uint8)
        out += pkt.tobytes()
    return bytes(out)


class TestProfileShape(unittest.TestCase):

    def test_int8_profile_resolves_its_modules(self):
        self.assertIs(INT8_PROFILE.emulator, int8)
        self.assertEqual(INT8_PROFILE.name, ENGINE_CONV_INT8)
        self.assertEqual(INT8_PROFILE.elem, ELEM_INT)
        self.assertEqual(INT8_PROFILE.object_file, "engine.o")
        self.assertTrue(INT8_PROFILE.transport_options)
        self.assertTrue(hasattr(INT8_PROFILE.schedule, "plan_workspace"))

    def test_bf16_profile_resolves_its_emulator(self):
        self.assertIs(BF16_PROFILE.emulator, bf16)
        self.assertEqual(BF16_PROFILE.name, ENGINE_CONV_BF16)
        self.assertEqual(BF16_PROFILE.elem, ELEM_BF16)
        self.assertEqual(BF16_PROFILE.object_file, "engine_bf16.o")
        self.assertFalse(BF16_PROFILE.transport_options)

    def test_the_bf16_scheduler_refuses_and_says_why(self):
        # The socket is wired; the packer that plugs into it is the fork. The refusal has to say so,
        # not read as a missing module.
        with self.assertRaises(NotImplementedError) as cm:
            _ = BF16_PROFILE.schedule
        self.assertIn("no scheduler yet", str(cm.exception))

    def test_profile_for_manifest_maps_both_engines(self):
        self.assertIs(profile_for_manifest({"engine": ENGINE_CONV_INT8}), INT8_PROFILE)
        self.assertIs(profile_for_manifest({"engine": ENGINE_CONV_BF16}), BF16_PROFILE)

    def test_profile_for_manifest_refuses_an_engine_it_does_not_know(self):
        with self.assertRaises(ValueError) as cm:
            profile_for_manifest({"engine": "some_other_engine_v9"})
        self.assertIn("some_other_engine_v9", str(cm.exception))

    def test_every_registered_profile_has_a_distinct_engine_name(self):
        names = [p.name for p in ENGINE_PROFILES.values()]
        self.assertEqual(len(names), len(set(names)))


class TestTheOpcodeCollision(unittest.TestCase):
    """The reason the name table has to travel with its emulator."""

    def test_the_two_engines_disagree_about_what_opcode_two_means(self):
        self.assertEqual(int8.OP_NAMES[2], "MAXPOOL")
        self.assertEqual(bf16.OP_NAMES[2], "RESIDUAL")
        self.assertEqual(bf16.OP_RESIDUAL, 2)

    def test_the_packet_stride_survives_the_difference_by_coincidence(self):
        # This is precisely why the bug is quiet rather than loud: the walk is correct and only the
        # naming is wrong. If either of these ever stops holding, the failure becomes a misparse.
        self.assertEqual(int8.W_BYTES, bf16.W_BYTES)
        self.assertEqual(int8.HDR_BYTES, bf16.HDR_BYTES)
        self.assertEqual(int8.H_OP, bf16.H_OP)
        self.assertEqual((int8.TILE_ROWS, int8.TILE_COLS), (bf16.TILE_ROWS, bf16.TILE_COLS))

    def test_a_bf16_residual_container_is_accepted_under_its_own_profile(self):
        blob = wpackets([bf16.OP_CONV, bf16.OP_RESIDUAL], bf16)
        check_kernel_covers_packets({"kernel_ops": ["CONV", "RESIDUAL"]}, blob, profile=BF16_PROFILE)

    def test_the_same_container_would_be_falsely_refused_under_the_int8_profile(self):
        # The bug the profile fixes, demonstrated rather than asserted. The int8 table reads opcode 2
        # as MAXPOOL, decides the packets reach a case the kernel was built without, and refuses a
        # container that is perfectly sound.
        blob = wpackets([bf16.OP_CONV, bf16.OP_RESIDUAL], bf16)
        with self.assertRaises(ValueError) as cm:
            check_kernel_covers_packets({"kernel_ops": ["CONV", "RESIDUAL"]}, blob, profile=INT8_PROFILE)
        self.assertIn("MAXPOOL", str(cm.exception))

    def test_the_int8_path_still_refuses_what_it_should(self):
        blob = wpackets([int8.OP_POOL], int8)
        with self.assertRaises(ValueError):
            check_kernel_covers_packets({"kernel_ops": ["CONV"]}, blob)

    def test_the_default_profile_is_int8_so_existing_callers_are_unchanged(self):
        blob = wpackets([int8.OP_CONV], int8)
        check_kernel_covers_packets({"kernel_ops": ["CONV"]}, blob)


class TestBf16ByteBudgets(unittest.TestCase):
    """The manifest's tile block is read from the emulator, so these are container-visible."""

    def test_the_activation_object_doubles_and_the_output_object_does_not(self):
        self.assertEqual(bf16.A_BYTES, 12800)
        self.assertEqual(int8.A_BYTES, 6400)
        # The same 3,200 B object either way - 32 channels at one byte, or 16 at two. That halving is
        # what tools/bf16_packet_recount.py counts per layer.
        self.assertEqual(bf16.O_BYTES, int8.O_BYTES)
        self.assertEqual(bf16.OUT_ELEMS * 2, bf16.O_BYTES)

    def test_the_byte_budgets_are_derived_from_the_element_counts(self):
        self.assertEqual(bf16.A_BYTES, bf16.A_ELEMS * 2)
        self.assertEqual(bf16.O_BYTES, bf16.OUT_ELEMS * 2)


if __name__ == "__main__":
    unittest.main()
