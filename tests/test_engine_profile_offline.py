"""The engine profile: one bundle for what differs between the int8 and the bf16 engine.

The test that earns its keep here is the opcode-collision one. The two engines number their opcodes
from zero independently, so 2 is MAXPOOL to one and RESIDUAL to the other, and the packet STRIDE
survives the difference by coincidence - W_BYTES is 9,472 and HDR_BYTES 128 in both, and the opcode
is word 0 in both. That coincidence is what makes the bug quiet: a bf16 blob walks correctly and
comes back named wrongly, and the container is then refused for reaching a case it never reaches.

Offline, no device, no IRON: the design module is named rather than imported, so everything except
``load_design`` resolves without a toolchain. The real-container class reads build/sesr_m7_bf16.ignite
when it has been built and skips otherwise; reading it opens no device either.
"""
import unittest
from pathlib import Path

import numpy as np

from ignite_xdna.compiler import engine_bf16_emulator as bf16
from ignite_xdna.compiler import engine_emulator as int8
from ignite_xdna.compiler.engine_compile import (BF16_PROFILE, ENGINE_PROFILES, INT8_PROFILE,
                                                 check_kernel_covers_packets, profile_for_manifest)
from ignite_xdna.compiler.serializer import (ELEM_BF16, ELEM_INT, ENGINE_CONV_BF16, ENGINE_CONV_INT8,
                                             IgniteModelReader)


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

    def test_the_bf16_profile_resolves_its_own_scheduler(self):
        # The fork plugs into the socket, and it is not the int8 scheduler under another name: that
        # one is untouched, and handing a bf16 compile to it would pack int8 packets.
        from ignite_xdna.compiler import engine_schedule, engine_schedule_bf16
        self.assertIs(BF16_PROFILE.schedule, engine_schedule_bf16)
        self.assertIsNot(BF16_PROFILE.schedule, engine_schedule)
        self.assertIs(INT8_PROFILE.schedule, engine_schedule)

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


ROOT = Path(__file__).resolve().parents[1]
BF16_SESR = ROOT / "build" / "sesr_m7_bf16.ignite"
INT8_SESR = ROOT / "build" / "sesr_m7.ignite"
SESR_QDQ = ROOT / "models" / "sesr_m7_xint8.onnx"


@unittest.skipUnless(BF16_SESR.exists() and SESR_QDQ.exists(), "build/sesr_m7_bf16.ignite not built")
class TestTheFirstBf16Container(unittest.TestCase):
    """``ignite-compile --engine graph --datapath bf16`` on SESR-M7, read back. No device is opened.

    The real-container versions of the checks above: what it declares, that its packets are the
    fork's, and that its real packets meet the opcode collision the profile exists for.
    """

    @classmethod
    def setUpClass(cls):
        with IgniteModelReader(str(BF16_SESR)) as reader:
            cls.m = reader.manifest
            cls.wp = reader.get_blob_bytes("wpackets.bin")
            cls.insts = reader.get_blob_bytes("insts.bin")

    def test_it_declares_the_bf16_engine_and_width(self):
        ge = self.m["graph_engine"]
        self.assertIs(profile_for_manifest(self.m), BF16_PROFILE)
        self.assertEqual({(p["dtype"], p.get("elem")) for p in ge["placements"].values()}, {("uint16", ELEM_BF16)})
        self.assertEqual(ge["kernel_ops"], ["CONV", "RESIDUAL"])
        self.assertEqual(ge["tile"]["a_bytes"], bf16.A_BYTES)
        # 12 channels of 256 x 256 at two bytes: egress is counted in bytes, so the width counts.
        self.assertEqual(self.m["egress_bytes"], 12 * 256 * 256 * 2)

    def test_its_packets_are_the_forks_packets(self):
        from ignite_xdna.compiler import engine_schedule_bf16 as eb
        from ignite_xdna.compiler.graph_ir import lower_yolov8n
        ir = lower_yolov8n(str(SESR_QDQ))
        scheds, store = eb.schedule_graph(ir, eb.plan_workspace(ir))
        self.assertEqual(self.wp, store.blob().tobytes())
        self.assertEqual([(L["rounds"], L["packets"]) for L in self.m["graph_engine"]["layers"]],
                         [(s.rounds, s.packets) for s in scheds])

    def test_its_real_packets_pass_under_bf16_and_would_be_falsely_refused_under_int8(self):
        check_kernel_covers_packets(self.m["graph_engine"], self.wp, profile=BF16_PROFILE)
        with self.assertRaises(ValueError) as cm:
            check_kernel_covers_packets(self.m["graph_engine"], self.wp, profile=INT8_PROFILE)
        self.assertIn("MAXPOOL", str(cm.exception))

    @unittest.skipUnless(INT8_SESR.exists(), "build/sesr_m7.ignite not built")
    def test_its_instruction_stream_is_not_the_int8_one(self):
        # Same task count, so the same length; the byte counts the descriptors carry are what doubled.
        with IgniteModelReader(str(INT8_SESR)) as reader:
            insts8 = reader.get_blob_bytes("insts.bin")
        self.assertEqual(len(self.insts), len(insts8))
        self.assertNotEqual(self.insts, insts8)


class TestTheDatapathFlag(unittest.TestCase):
    """``ignite-compile --datapath`` picks a registered profile, and its default leaves every existing
    invocation on int8."""

    ARGS = ["--engine", "graph", "--input", "m.onnx", "--output", "o.ignite"]

    def test_the_default_is_int8(self):
        from ignite_xdna.compiler.cli import parse_args
        self.assertIs(ENGINE_PROFILES[parse_args(self.ARGS).datapath], INT8_PROFILE)

    def test_bf16_names_the_bf16_profile(self):
        from ignite_xdna.compiler.cli import parse_args
        self.assertIs(ENGINE_PROFILES[parse_args(self.ARGS + ["--datapath", "bf16"]).datapath], BF16_PROFILE)

    def test_an_unregistered_datapath_is_refused(self):
        from ignite_xdna.compiler.cli import parse_args
        with self.assertRaises(SystemExit):
            parse_args(self.ARGS + ["--datapath", "fp16"])


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
