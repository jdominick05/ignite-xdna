"""YOLO-World v2 at run time, offline (no NPU): the text side and the decode.

  python tests/test_yolow_pipeline_offline.py

Uses models/yolow_text_encoder.onnx and models/yolow_text.npz (``pipelines/yolow/6_text_encoder.py``) and
models/yolov8s-worldv2_cut.onnx (gitignored: tests are skipped when absent), onnxruntime and opencv.

- the tokenizer gives the token ids ``clip.tokenize`` gave for fixed phrases, and refuses non-ASCII names;
- the guides computed from the bundle for COCO's 80 names equal the exported model's text-guide initializers
  (within 1e-4), and a five-name vocabulary gives guides of class dimension 5;
- ``YoloWorldDecoder`` returns the same detections, bit for bit, as ``npu/yolow.py``'s ``decode_yolow`` followed by
  ``npu/yolo.py``'s ``postprocess`` on random heads with the same constants; decoding int8 views with their scales
  equals decoding the dequantized heads; and pruning anchors below the threshold leaves the detections unchanged.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.pipelines import yolow_text as yt  # noqa: E402
from ignite_xdna.pipelines.yolow_pipeline import HEAD_ORDER, YoloWorldDecoder  # noqa: E402

MODELS = ROOT / "models"
ENCODER = MODELS / "yolow_text_encoder.onnx"
BUNDLE = MODELS / "yolow_text.npz"
CUT = MODELS / "yolov8s-worldv2_cut.onnx"
# clip.tokenize (the ultralytics CLIP fork, truncate=True) on 2026-09-16: start token, text, end token.
CLIP_TOKENS = {
    "person": [49406, 2533, 49407],
    "road sign": [49406, 1759, 2292, 49407],
    "don't walk sign": [49406, 847, 713, 2374, 2292, 49407],
    "rock &amp; roll": [49406, 2172, 261, 3341, 49407],
    "3d printer": [49406, 274, 323, 14521, 49407],
    "hot-dog": [49406, 2069, 268, 1929, 49407],
}


@unittest.skipUnless(BUNDLE.exists(), f"{BUNDLE.name} not present")
class Tokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(BUNDLE) as z:
            cls.tok = yt.ClipTokenizer([str(m) for m in z["bpe_merges"]])

    def test_token_ids_match_clip(self):
        ids = self.tok(list(CLIP_TOKENS))
        self.assertEqual(ids.shape, (len(CLIP_TOKENS), yt.CONTEXT_LENGTH))
        for row, (text, want) in zip(ids, CLIP_TOKENS.items()):
            self.assertEqual(row[:len(want)].tolist(), want, text)
            self.assertFalse(row[len(want):].any(), text)

    def test_long_text_is_cut_and_ends_in_the_end_token(self):
        row = self.tok(["word " * 100])[0]
        self.assertEqual(row[-1], self.tok.eot)
        self.assertTrue(row.all())

    def test_non_ascii_names_are_refused(self):
        with self.assertRaisesRegex(ValueError, "ASCII"):
            self.tok(["café"])


@unittest.skipUnless(ENCODER.exists() and BUNDLE.exists() and CUT.exists(), "text bundle or cut model not present")
class Guides(unittest.TestCase):
    def test_coco_guides_equal_the_exported_initializers(self):
        import onnx
        from onnx import numpy_helper
        from npu.yolo import COCO_CLASSES
        text = yt.YoloWorldText(ENCODER, BUNDLE)
        guides = text.host_constants(text.embed(list(COCO_CLASSES)))
        inits = {t.name: numpy_helper.to_array(t) for t in onnx.load(str(CUT)).graph.initializer}
        self.assertEqual(sorted(guides), sorted(yt.guide_name(b) for b in yt.GUIDE_BLOCKS))
        for name, g in guides.items():
            self.assertEqual(g.shape, inits[name].shape, name)
            self.assertLess(float(np.abs(g - inits[name]).max()), 1e-4, name)
        five = text.host_constants(text.embed(["person", "bus", "window", "road sign", "shoe"]))
        self.assertEqual([five[yt.guide_name(b)].shape[1] for b in yt.GUIDE_BLOCKS], [5, 5, 5, 5])


class Decoder(unittest.TestCase):
    @staticmethod
    def random_heads(rng, k=512):
        heads = []
        for g in (80, 40, 20):
            heads.append(rng.normal(0, 3, (1, 64, g, g)).astype(np.float32))
        for g in (80, 40, 20):
            heads.append(rng.normal(0, 1, (1, k, g, g)).astype(np.float32))
        return heads

    def setUp(self):
        import npu.yolow as yw
        self.yw = yw
        rng = np.random.default_rng(3)
        self.names = ["person", "bus", "shoe", "road sign", "window", "dog", "cat"]
        e = rng.normal(0, 1, (len(self.names), 512)).astype(np.float32)
        self.emb = e / np.linalg.norm(e, axis=1, keepdims=True)
        self.heads = self.random_heads(rng)
        # A few anchors made confident so detections exist.
        for level, g in enumerate((80, 40, 20)):
            for a in rng.choice(g * g, 5, replace=False):
                self.heads[3 + level][0, :, a // g, a % g] = self.emb[rng.integers(len(self.names))] * 12
        self.dec = YoloWorldDecoder(self.emb, self.names, yw.SCALES, yw.BIASES, conf_thres=0.25, iou_thres=0.5)

    def test_detections_equal_the_numpy_reference(self):
        from npu.yolo import postprocess
        pad, scale = (80, 0), 0.5
        ref = postprocess(self.yw.decode_yolow(self.heads, self.emb, imgsz=640), pad, scale, 0.25, 0.5)
        got = self.dec.postprocess(self.dec.decode(self.heads), pad, scale)
        self.assertGreater(len(ref), 0)
        self.assertEqual([(d.x0, d.y0, d.w, d.h, d.score, d.class_id) for d in got],
                         [(float(x), float(y), float(w), float(h), s, c) for x, y, w, h, s, c in ref])
        self.assertEqual([d.name for d in got], [self.names[c] for *_, c in ref])

    def test_int8_views_with_scales_equal_the_dequantized_heads(self):
        rng = np.random.default_rng(4)
        q = {n: rng.integers(-128, 128, h.shape, dtype=np.int8) for n, h in zip(HEAD_ORDER, self.heads)}
        scales = {n: (0.0625 if n.endswith("box") else 0.015625, 0) for n in HEAD_ORDER}
        floats = [(q[n].astype(np.float32) - np.float32(0)) * np.float32(scales[n][0]) for n in HEAD_ORDER]
        a = self.dec.decode(q, scales)
        b = self.dec.decode(floats)
        self.assertTrue(np.array_equal(a, b))

    def test_int8_fast_path_is_bit_exact_with_pruning_and_falls_back_off_power_of_two(self):
        """The int8 decode (scale folded into the contrastive product, per-level prune, kept boxes only) equals the
        float reference bit for bit, with and without a threshold; a scale that is not a power of two takes the
        reference path."""
        rng = np.random.default_rng(8)
        q = {n: rng.integers(-128, 128, h.shape, dtype=np.int8) for n, h in zip(HEAD_ORDER, self.heads)}
        for level, g in enumerate((80, 40, 20)):   # confident anchors: visual features along a class embedding
            for a in rng.choice(g * g, 4, replace=False):
                q[HEAD_ORDER[3 + level]][0, :, a // g, a % g] = np.clip(np.round(self.emb[level] * 900), -128, 127)
        scales = {n: (0.25 if n.endswith("box") else 0.125, 0) for n in HEAD_ORDER}
        floats = [(q[n].astype(np.float32) - np.float32(0)) * np.float32(scales[n][0]) for n in HEAD_ORDER]
        for conf in (None, 0.25, 0.001):
            a = self.dec.decode(q, scales, conf)
            b = self.dec.decode(floats, None, conf)
            self.assertEqual(a.shape, b.shape, conf)
            self.assertTrue(np.array_equal(a, b), conf)
        self.assertGreater(self.dec.decode(q, scales, 0.25).shape[2], 0)
        odd = dict(scales, p4_cls=(0.1, 0))
        floats_odd = [(q[n].astype(np.float32) - np.float32(0)) * np.float32(odd[n][0]) for n in HEAD_ORDER]
        self.assertTrue(np.array_equal(self.dec.decode(q, odd, 0.25), self.dec.decode(floats_odd, None, 0.25)))

    def test_pruned_decode_gives_the_same_detections(self):
        pad, scale = (0, 0), 1.0
        full = self.dec.postprocess(self.dec.decode(self.heads), pad, scale)
        pruned = self.dec.postprocess(self.dec.decode(self.heads, conf_thres=0.25), pad, scale)
        self.assertGreater(len(full), 0)
        self.assertEqual(full, pruned)


if __name__ == "__main__":
    unittest.main()
