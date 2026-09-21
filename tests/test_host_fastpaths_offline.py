"""Offline checks of the host fast paths added with the consolidated head readback (no device).

- ``yolo_decode_c8_blocks``: detections decoded straight from channel-blocked uint8 heads equal the int8 NCHW decode,
  native and numpy, on synthetic heads, at both box forms the library implements (reg_max 16 and 1). At reg_max 1 the
  box head is one block wide and its lanes 4-7 are padding, so this is what proves the decode reads the lane rather
  than the block pair DFL needs.
- ``depth_to_space_crd_bgr``: SESR M7's native DepthToSpace + lookup equals ``DenseGraphSession.postprocess``'s numpy
  loop byte for byte.
- ``resize_bgr_to_c8_plane``: the native bilinear resize is not OpenCV's fixed-point INTER_LINEAR; it stays within
  one code of ``cv2.resize`` on every byte (measured about 20 % of bytes one code apart).
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.pipelines import preprocess as pp  # noqa: E402
from ignite_xdna.pipelines.yolo_pipeline import YoloDecoder  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
