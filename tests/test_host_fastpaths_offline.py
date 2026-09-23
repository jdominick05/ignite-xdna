"""Offline checks of the host fast paths added with the consolidated head readback (no device).

- ``yolo_decode_c8_blocks``: detections decoded straight from channel-blocked uint8 heads equal the int8 NCHW decode,
  native and numpy, on synthetic heads, at both box forms the library implements (reg_max 16 and 1). At reg_max 1 the
  box head is one block wide and its lanes 4-7 are padding, so this is what proves the decode reads the lane rather
  than the block pair DFL needs.
- ``depth_to_space_crd_bgr``: SESR M7's native DepthToSpace + lookup equals ``DenseGraphSession.postprocess``'s numpy
  loop byte for byte.
- ``resize_bgr_to_c8_plane``: the native bilinear resize is not OpenCV's fixed-point INTER_LINEAR; it stays within
  one code of ``cv2.resize`` on every byte (measured about 20 % of bytes one code apart).
- ``depth_to_space_crd_bgr_bf16`` and ``bgr_to_c8_plane_bf16``: the bf16 engine's super-resolution egress and
  ingress equal ``graph_session``'s numpy paths byte for byte - the egress on every finite bf16 pattern - and the
  egress refuses a non-finite value exactly where numpy does. The library is checked against its source, so a
  DLL left unbuilt after the C changed fails rather than skipping these.
"""
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.compiler.engine_bf16_emulator import bf16_bits, to_bf16  # noqa: E402
from ignite_xdna.pipelines import preprocess as pp  # noqa: E402
from ignite_xdna.pipelines.yolo_pipeline import YoloDecoder, reg_max_from_manifest  # noqa: E402
from ignite_xdna.runtime.graph_session import bf16_dense_image, bf16_input_lut  # noqa: E402

GRIDS = (("p3", 80), ("p4", 40), ("p5", 20))
SCALES = {"p3_box": (0.0625, 0), "p4_box": (0.0625, 0), "p5_box": (0.125, 0),
          "p3_cls": (0.125, 0), "p4_cls": (0.125, 0), "p5_cls": (0.25, 0)}


def _synthetic_heads(rng, confident=8, reg_max=16):
    """int8 NCHW heads [C][H][W] with a few confident class logits per grid.

    The box head is ``4 * reg_max`` channels: 64 for YOLOv8's DFL, 4 for a DFL-free head
    like YOLO26's, where the four channels are the four distances directly.
    """
    heads = {}
    for g, hw in GRIDS:
        box = rng.integers(-60, 60, (4 * reg_max, hw, hw), dtype=np.int8)
        cls = rng.integers(-128, -50, (80, hw, hw), dtype=np.int8)
        for _ in range(confident):
            cls[rng.integers(0, 80), rng.integers(0, hw), rng.integers(0, hw)] = rng.integers(20, 127)
        heads[f"{g}_box"], heads[f"{g}_cls"] = box, cls
    return heads


def _to_c8(nchw, junk):
    """int8 [C][H][W] -> the graph session's raw view: uint8 [blocks][H*W][8], zero point 128, junk in padding."""
    c, h, w = nchw.shape
    blocks = (c + 7) // 8
    full = np.full((blocks * 8, h, w), junk, dtype=np.uint8)
    full[:c] = nchw.view(np.uint8) ^ 0x80
    return np.ascontiguousarray(np.transpose(full.reshape(blocks, 8, h, w), (0, 2, 3, 1)).reshape(blocks, h * w, 8))


def _detections(dets):
    return [(d.x0, d.y0, d.w, d.h, d.score, d.class_id) for d in dets]


class HostFastPathsOffline(unittest.TestCase):
    def _c8_equals_nchw(self, reg_max):
        native = YoloDecoder(imgsz=640, native_decode=True)
        native.set_reg_max(reg_max)
        if not native.uses_native_decode or not hasattr(native._native._lib, "yolo_decode_c8_blocks"):
            self.skipTest(f"native decode library without yolo_decode_c8_blocks at reg_max {reg_max}")
        numpy_dec = YoloDecoder(imgsz=640, native_decode=False)
        numpy_dec.set_reg_max(reg_max)
        checked = 0
        for seed in range(12):
            rng = np.random.default_rng(seed)
            nchw = _synthetic_heads(rng, reg_max=reg_max)
            cls_max = {f"{g}_cls": nchw[f"{g}_cls"].reshape(80, -1).max(0) for g, _ in GRIDS}
            ref_heads = dict(nchw, scales=SCALES, cls_max=cls_max)
            c8_heads = {k: _to_c8(v, junk=int(rng.integers(0, 256))) for k, v in nchw.items()}
            c8_heads.update(scales=SCALES, cls_max=cls_max)
            for conf in (0.001, 0.25, 0.6):
                pad, scale = (int(rng.integers(0, 80)), int(rng.integers(0, 80))), float(rng.uniform(0.3, 1.5))
                ref = _detections(numpy_dec.postprocess(ref_heads, pad, scale, conf_thres=conf, iou_thres=0.7))
                nchw_native = _detections(native.postprocess(ref_heads, pad, scale, conf_thres=conf, iou_thres=0.7))
                c8_native = _detections(native.postprocess(c8_heads, pad, scale, conf_thres=conf, iou_thres=0.7))
                no_max = {k: v for k, v in c8_heads.items() if k != "cls_max"}
                c8_numpy = _detections(native.postprocess(no_max, pad, scale, conf_thres=conf, iou_thres=0.7))
                self.assertEqual(nchw_native, ref, (reg_max, seed, conf))
                self.assertEqual(c8_native, ref, (reg_max, seed, conf))
                self.assertEqual(c8_numpy, ref, (reg_max, seed, conf))
                checked += len(ref)
        self.assertGreater(checked, 0)
        return checked

    def test_c8_decode_equals_nchw_decode(self):
        """YOLOv8's 16-bin DFL box head, the shipped path."""
        self._c8_equals_nchw(16)

    def test_c8_decode_equals_nchw_decode_dfl_free(self):
        """A DFL-free box head: four channels in one block, lanes 4-7 padding."""
        self._c8_equals_nchw(1)

    def test_depth_to_space_matches_numpy(self):
        rng = np.random.default_rng(7)
        for oh, ow in ((256, 256), (64, 40), (1, 1)):
            blocks = rng.integers(0, 256, (2, oh, ow, 8), dtype=np.uint8)
            lut = rng.integers(0, 256, 256, dtype=np.uint8)
            got = np.zeros((2 * oh, 2 * ow, 3), dtype=np.uint8)
            if not pp.depth_to_space_crd_bgr(blocks, oh, ow, lut, got):
                self.skipTest("native preprocessor without depth_to_space_crd_bgr")
            ref = np.empty_like(got)
            bs, oc = 2, 3
            for k in range(oc):
                for i in range(bs):
                    for j in range(bs):
                        ch = (k * bs + i) * bs + j
                        ref[i::bs, j::bs, oc - 1 - k] = lut[blocks[ch // 8, :, :, ch % 8]]
            self.assertTrue(np.array_equal(got, ref), (oh, ow))

    def test_resize_is_within_one_code_of_opencv(self):
        import cv2
        rng = np.random.default_rng(9)
        for sh, sw, dh, dw in ((1080, 810, 256, 256), (300, 500, 640, 640), (256, 256, 256, 256), (97, 131, 256, 256)):
            img = rng.integers(0, 256, (sh, sw, 3), dtype=np.uint8)
            halo = 1
            plane = np.zeros((dh + 2 * halo, dw + 2 * halo, 8), dtype=np.uint8)
            if not pp.resize_bgr_to_c8_plane(img, plane, dw, dh, halo, None):
                self.skipTest("native preprocessor without fused_resize_bgr_to_c8_plane")
            ref = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
            diff = np.abs(plane[halo:halo + dh, halo:halo + dw, :3].astype(np.int16) - ref.astype(np.int16))
            self.assertLessEqual(int(diff.max()), 1, (sh, sw, dh, dw))
            if (sh, sw) == (dh, dw):
                self.assertEqual(int(diff.max()), 0)
            self.assertTrue(np.all(plane[:halo] == 0) and np.all(plane[:, :, 3:] == 0))


NAN, INF, NEG_INF = 0x7FC0, 0x7F80, 0xFF80


def _every_finite_bf16_pattern(h=64, w=86) -> np.ndarray:
    """uint16 [2][h][w][8]: the finite bf16 patterns in order across the 12 active lanes - every one of them at
    the default size - and NaN in the four padding lanes of the second block, which neither path may read."""
    pats = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    finite = pats[(pats & 0x7F80) != 0x7F80]
    active = np.resize(finite, (h, w, 12))
    blocks = np.empty((2, h, w, 8), np.uint16)
    blocks[0] = active[:, :, :8]
    blocks[1, :, :, :4] = active[:, :, 8:]
    blocks[1, :, :, 4:] = NAN
    return blocks


class Bf16SrHostPathsOffline(unittest.TestCase):
    """The bf16 SR host ends, native against ``graph_session``'s numpy, which ``test_bf16_dense_session_offline``
    checks in turn against the compiler's layout and ``npu/sesr.py``."""

    def setUp(self):
        for name in ("depth_to_space_crd_bgr_bf16", "bgr_to_c8_plane_bf16"):
            if not pp.has_native(name):
                self.skipTest(f"native preprocessor without {name}")

    def test_egress_equals_numpy_on_every_finite_pattern(self):
        blocks = _every_finite_bf16_pattern()
        _, h, w, _ = blocks.shape
        active = np.concatenate([blocks[0].reshape(-1, 8), blocks[1].reshape(-1, 8)[:, :4]], axis=1)
        self.assertEqual(np.unique(active).size, 65536 - 256)   # all but exponent 0xFF: 2 signs x 128 mantissas
        for mean in (128.0, 0.0, 127.5, 1e-3):
            got = np.zeros((2 * h, 2 * w, 3), np.uint8)
            self.assertTrue(pp.depth_to_space_crd_bgr_bf16(blocks, h, w, mean, got))
            np.testing.assert_array_equal(got, bf16_dense_image(blocks, 12, 2, mean), err_msg=f"mean {mean}")

    def test_egress_refuses_a_non_finite_active_value_as_numpy_does_and_writes_nothing(self):
        clean = _every_finite_bf16_pattern(16, 24)
        for block, lane in ((0, 0), (0, 7), (1, 0), (1, 3)):
            for pattern in (NAN, INF, NEG_INF):
                blocks = clean.copy()
                blocks[block, 5, 9, lane] = pattern
                blocks[block, 11, 2, lane] = 0xFFC1          # a second NaN, negative, with payload
                got = np.full((32, 48, 3), 42, np.uint8)
                with self.assertRaises(RuntimeError) as native:
                    pp.depth_to_space_crd_bgr_bf16(blocks, 16, 24, 128.0, got)
                with self.assertRaises(RuntimeError) as numpy_path:
                    bf16_dense_image(blocks, 12, 2, 128.0)
                self.assertEqual(str(native.exception), str(numpy_path.exception))
                self.assertTrue(str(native.exception).startswith("2 non-finite"), str(native.exception))
                self.assertTrue(np.all(got == 42), (block, lane, pattern))

    def test_egress_declines_what_it_cannot_serve_and_leaves_dst_alone(self):
        blocks = _every_finite_bf16_pattern(16, 24)
        cases = {
            "uint8 source": (blocks.view(np.uint8), 16, 24, np.zeros((32, 48, 3), np.uint8)),
            "one block": (np.ascontiguousarray(blocks[:1]), 16, 24, np.zeros((32, 48, 3), np.uint8)),
            "strided source": (blocks[:, :, ::2], 16, 12, np.zeros((32, 24, 3), np.uint8)),
            "wrong image": (blocks, 16, 24, np.zeros((32, 48, 4), np.uint8)),
            "float image": (blocks, 16, 24, np.zeros((32, 48, 3), np.float32)),
        }
        for label, (src, h, w, dst) in cases.items():
            before = dst.copy()
            self.assertFalse(pp.depth_to_space_crd_bgr_bf16(src, h, w, 128.0, dst), label)
            np.testing.assert_array_equal(dst, before, err_msg=label)

    def test_ingress_equals_numpy_and_touches_nothing_else(self):
        lut = bf16_input_lut({"mean": 128.0, "divisor": 1.0})
        rng = np.random.default_rng(11)
        for h, w, halo in ((256, 256, 2), (12, 10, 2), (7, 5, 0), (3, 9, 1)):
            img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            got = np.full((h + 2 * halo, w + 2 * halo, 8), 0x1234, np.uint16)
            want = got.copy()
            want[halo:halo + h, halo:halo + w, :3] = lut[img[:, :, ::-1]]
            self.assertTrue(pp.bgr_to_c8_plane_bf16(img, got, halo, lut), (h, w, halo))
            np.testing.assert_array_equal(got, want, err_msg=f"{(h, w, halo)}")

    def test_ingress_reads_a_row_strided_view_as_it_is(self):
        lut = bf16_input_lut({"mean": 128.0, "divisor": 1.0})
        big = np.random.default_rng(12).integers(0, 256, (40, 50, 3), dtype=np.uint8)
        view = big[5:17, 7:17]                               # 12 x 10, rows 150 bytes apart
        self.assertFalse(view.flags["C_CONTIGUOUS"])
        got = np.zeros((16, 14, 8), np.uint16)
        self.assertTrue(pp.bgr_to_c8_plane_bf16(view, got, 2, lut))
        want = np.zeros_like(got)
        want[2:14, 2:12, :3] = lut[view[:, :, ::-1]]
        np.testing.assert_array_equal(got, want)

    def test_ingress_declines_what_it_cannot_serve_and_leaves_the_plane_alone(self):
        lut = bf16_input_lut({"mean": 128.0, "divisor": 1.0})
        img = np.random.default_rng(13).integers(0, 256, (12, 10, 3), dtype=np.uint8)
        plane = np.zeros((16, 14, 8), np.uint16)
        cases = {
            "another size (no resize here)": (img[:11], plane, lut),
            "uint8 plane": (img, np.zeros((16, 14, 8), np.uint8), lut),
            "float frame": (img.astype(np.float32), plane, lut),
            "column-strided frame": (img[:, ::-1], plane, lut),
            "uint8 table": (img, plane, lut.astype(np.uint8)),
            "short table": (img, plane, lut[:128]),
        }
        for label, (src, dst, table) in cases.items():
            before = dst.copy()
            self.assertFalse(pp.bgr_to_c8_plane_bf16(src, dst, 2, table), label)
            np.testing.assert_array_equal(dst, before, err_msg=label)


class ShippedLibraryMatchesItsSource(unittest.TestCase):
    """Every function ``preprocess_simd.c`` exports is in the library that loaded. The DLL is committed beside the
    source; a C change without a rebuild would otherwise leave every native test above skipping, not failing."""

    def test_every_exported_function_is_in_the_loaded_library(self):
        if pp._LIB is None:
            self.skipTest("no native preprocessor loaded")
        source = (ROOT / "src" / "ignite_xdna" / "pipelines" / "preprocess_simd.c").read_text(encoding="utf-8")
        exported = re.findall(r"^PREPROCESS_API\s+\w+\s+(\w+)\s*\(", source, re.M)
        self.assertGreaterEqual(len(exported), 8)
        missing = [name for name in exported if not hasattr(pp._LIB, name)]
        self.assertEqual(missing, [], f"{pp._LIB._name} predates its source")



class RegMaxFromManifest(unittest.TestCase):
    """A container's declared reg_max must agree with the box head it describes.

    The compiler derives one from the other, so they agree by construction - but containers compiled
    before it did that declare 16 beside a 4-channel box head. That combination does not raise when
    decoded, it silently reduces over DFL bins that are not there and returns wrong boxes, which is
    the worst available failure mode. build/yolo26n.ignite and two siblings are real examples.
    """

    @staticmethod
    def _manifest(reg_max, box_channels):
        return {"reg_max": reg_max,
                "output_shapes": {"p3_box": [1, box_channels, 80, 80],
                                  "p3_cls": [1, 80, 80, 80]}}

    def test_dfl_and_dfl_free_heads_are_accepted(self):
        self.assertEqual(reg_max_from_manifest(self._manifest(16, 64)), 16)   # YOLOv8
        self.assertEqual(reg_max_from_manifest(self._manifest(1, 4)), 1)      # YOLO26

    def test_a_stale_manifest_is_refused_naming_both_numbers(self):
        with self.assertRaises(ValueError) as caught:
            reg_max_from_manifest(self._manifest(16, 4))
        message = str(caught.exception)
        self.assertIn("64", message)   # what 16 bins would require
        self.assertIn("4", message)    # what the head actually carries

    def test_nothing_declared_leaves_the_caller_default(self):
        self.assertIsNone(reg_max_from_manifest(None))
        self.assertIsNone(reg_max_from_manifest({}))

    def test_a_manifest_without_shapes_is_taken_at_its_word(self):
        # Older containers carry reg_max but no output_shapes; there is nothing to check against,
        # and refusing them would break containers that decode correctly today.
        self.assertEqual(reg_max_from_manifest({"reg_max": 16}), 16)

if __name__ == "__main__":
    unittest.main()
