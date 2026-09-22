"""Region arithmetic: bytes, not elements, and the two are equal only at one byte.

Every region length in the runtime used to be an element count spent as a byte count. That is exact
at one byte an element and covers exactly half the region at two - and because the halved byte count
EQUALS the element count, the subsequent reshape succeeds and the tensor comes back plausible rather
than absent. A half-length sync is worse still: it leaves the back half of the region holding the
previous frame, which reads as ghosting, not as corruption.

These tests pin both halves of the fix: that the new helpers reproduce the old arithmetic exactly at
one byte, placement by placement, against a real committed container; and that they actually scale at
two, so the readiness is not merely asserted.
"""
import unittest
from pathlib import Path

import numpy as np

from ignite_xdna.runtime.graph_session import (_boundary_region, _plane_view, _region, halo_fill_image,
                                               placement_dtype, plane_bytes, region_bytes)

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = ROOT / "build" / "sesr_m7.ignite"


def old_region_bytes(p):
    """Exactly what the runtime computed before: an element count, spent as bytes."""
    h, w, halo = int(p["height"]), int(p["width"]), int(p["halo"])
    return int(p["blocks"]) * (h + 2 * halo) * (w + 2 * halo) * 8


def placement(dtype="uint8", **kw):
    p = {"base": 0, "halo": 1, "height": 8, "width": 12, "blocks": 3, "planes": 4, "channels": 20,
         "band_rows": 0, "dtype": dtype}
    p.update(kw)
    return p


class TestTheHelpersAgreeWithTheOldArithmeticAtOneByte(unittest.TestCase):
    """The no-op half. If this fails, the refactor changed shipping behaviour."""

    def test_synthetic_placements_reproduce_the_old_expression(self):
        for halo in (0, 1, 2):
            for blocks in (1, 3, 8):
                p = placement(halo=halo, blocks=blocks)
                self.assertEqual(region_bytes(p), old_region_bytes(p), f"halo={halo} blocks={blocks}")

    @unittest.skipUnless(CONTAINER.exists(), "sesr_m7 container not present")
    def test_every_placement_of_a_real_container_is_unchanged(self):
        from ignite_xdna.compiler.serializer import IgniteModelReader
        with IgniteModelReader(str(CONTAINER)) as reader:
            ge = reader.manifest["graph_engine"]
        self.assertTrue(ge["placements"], "the container has no placements to check")
        for name, p in ge["placements"].items():
            self.assertEqual(placement_dtype(p).itemsize, 1, f"{name} is not a one-byte placement")
            self.assertEqual(region_bytes(p), old_region_bytes(p), name)
            self.assertEqual(_region(p), (int(p["base"]), old_region_bytes(p)), name)

    @unittest.skipUnless(CONTAINER.exists(), "sesr_m7 container not present")
    def test_halo_fill_image_is_the_declared_workspace_size_and_sets_its_rings(self):
        from ignite_xdna.compiler.serializer import IgniteModelReader
        with IgniteModelReader(str(CONTAINER)) as reader:
            ge = reader.manifest["graph_engine"]
        ws = halo_fill_image(ge)
        self.assertEqual(ws.size, int(ge["workspace_bytes"]))
        self.assertEqual(ws.dtype, np.uint8)
        haloed = [p for p in ge["placements"].values() if p["halo"]]
        self.assertTrue(haloed, "nothing in this container has a halo, so this proves nothing")
        for p in haloed:
            planes = _plane_view(ws, p)
            self.assertTrue((planes[:, :p["halo"], :, :] == p.get("halo_value", 128)).all())


class TestTheHelpersActuallyScale(unittest.TestCase):
    """The readiness half. A no-op that is not also correct at two bytes has bought nothing."""

    def test_a_two_byte_placement_spans_twice_the_region(self):
        narrow, wide = placement("uint8"), placement("uint16")
        self.assertEqual(region_bytes(wide), 2 * region_bytes(narrow))
        self.assertEqual(plane_bytes(wide), 2 * plane_bytes(narrow))

    def test_region_bytes_is_planes_times_plane_bytes(self):
        p = placement("uint16")
        self.assertEqual(region_bytes(p), p["blocks"] * plane_bytes(p))
        self.assertEqual(region_bytes(p, p["planes"]), p["planes"] * plane_bytes(p))

    def test_a_plane_view_of_a_two_byte_placement_is_two_byte_and_correctly_shaped(self):
        p = placement("uint16", base=0)
        buf = np.zeros(region_bytes(p, p["planes"]), dtype=np.uint8)
        view = _plane_view(buf, p)
        self.assertEqual(view.dtype, np.uint16)
        h, w, halo = p["height"], p["width"], p["halo"]
        self.assertEqual(view.shape, (p["planes"], h + 2 * halo, w + 2 * halo, 8))
        # The view has to be a WINDOW on the buffer, not a copy, or staging writes nothing.
        view[0, 0, 0, 0] = 0xBEEF
        self.assertEqual(buf[:2].view(np.uint16)[0], 0xBEEF)

    def test_the_boundary_helper_and_the_region_helper_now_agree(self):
        # _boundary_region threaded itemsize and _region did not, though both claimed to return
        # bytes for the same span. They were the same arithmetic all along.
        for dtype in ("uint8", "uint16"):
            p = placement(dtype)
            self.assertEqual(_region(p), _boundary_region(p), dtype)


class TestTheHeadEgressContract(unittest.TestCase):
    """``HeadSpec.nbytes`` named bytes and returned elements. Ignition consumes this module through
    the [npu] wheel, so the int8 numbers must be identical to the byte - and they are, because
    int8's itemsize is 1. The field exists so a wider head could never land at half its offset.
    """

    def test_nbytes_is_unchanged_for_int8_heads(self):
        from ignite_xdna.runtime.heads import HeadSpec
        shape = (1, 64, 80, 80)
        h = HeadSpec(name="p3_box", shape=shape, offset=0, scale=0.05, zero_point=0)
        self.assertEqual(h.itemsize, 1)
        self.assertEqual(h.nbytes, int(np.prod(shape)))
        self.assertEqual(h.end, h.offset + int(np.prod(shape)))

    def test_nbytes_would_scale_for_a_wider_head(self):
        from ignite_xdna.runtime.heads import HeadSpec
        shape = (1, 64, 80, 80)
        narrow = HeadSpec(name="p3_box", shape=shape, offset=0, scale=0.05, zero_point=0)
        wide = HeadSpec(name="p3_box", shape=shape, offset=0, scale=0.05, zero_point=0, dtype="int16")
        self.assertEqual(wide.nbytes, 2 * narrow.nbytes)

    def test_unpack_still_refuses_an_egress_that_is_not_int8(self):
        from ignite_xdna.runtime.heads import DetectHeadLayout, HeadSpec
        layout = DetectHeadLayout(heads=(HeadSpec("p3_box", (1, 2, 2, 2), 0, 0.05, 0),))
        with self.assertRaises(TypeError):
            layout.unpack(np.zeros(8, dtype=np.uint8))
        views = layout.unpack(np.zeros(8, dtype=np.int8))
        self.assertEqual(views["p3_box"].shape, (1, 2, 2, 2))


class TestAMissingDtypeIsRefused(unittest.TestCase):
    """The default that made the whole class silent."""

    def test_a_placement_with_no_dtype_raises_and_names_the_key(self):
        p = placement()
        del p["dtype"]
        for fn in (placement_dtype, plane_bytes, region_bytes):
            with self.assertRaises(KeyError, msg=fn.__name__) as cm:
                fn(p)
            self.assertIn("dtype", str(cm.exception))

    def test_an_empty_dtype_is_refused_too(self):
        with self.assertRaises(KeyError):
            region_bytes(placement(dtype=""))


if __name__ == "__main__":
    unittest.main()
