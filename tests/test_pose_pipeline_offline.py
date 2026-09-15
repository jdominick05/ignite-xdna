"""Offline (no device): a pose container's heads decode to exactly the people ONNX Runtime's numpy tail finds.

For assets/bus.jpg, and three data/coco128 images when that directory exists:

* ``pose_pipeline.letterbox`` equals ``npu.yolo.letterbox`` byte for byte;
* the nine uint8 head tensors ``graph_reference.run_direct`` computes from the quantized input (the tensors
  the NPU writes, which tests/test_graph_engine_offline.py emulates and tools/verify_engine_container.py
  reads back from the device) are packed into the pose manifest's int8 egress, unpacked through
  ``resolve_head_layout`` and decoded by ``PoseDecoder``;
* the result equals ``npu.yolo_pose_decode.decode_heads`` + ``npu.yolo_pose.postprocess`` on ONNX Runtime's
  float heads for the same image, bit for bit (boxes, scores, 17 keypoints each), at the COCO evaluation
  settings of pipelines/yolov8n-pose/5_eval_map.py (conf 0.001, IoU 0.7, 300 people) and at the demo
  settings (conf 0.25, IoU 0.5).

    python -m pytest tests/test_pose_pipeline_offline.py      (resnet_env17)
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

POSE = ROOT / "models" / "yolov8n-pose_cut_xint8.onnx"
COCO128 = ROOT / "data" / "coco128"
SETTINGS = ((0.001, 0.7, 300), (0.25, 0.5, 300))


@unittest.skipUnless(POSE.exists(), "yolov8n-pose model not present")
class PoseDecodeMatchesReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2
        import onnxruntime as ort

        import npu.yolo as yolo
        import npu.yolo_pose as yp
        from ignite_xdna.compiler import engine_schedule as es
        from ignite_xdna.compiler import graph_ir
        from ignite_xdna.compiler.engine_compile import build_manifest, head_name_for
        from ignite_xdna.runtime.heads import resolve_head_layout
        cls.cv2, cls.yolo, cls.yp = cv2, yolo, yp
        cls.ir = graph_ir.lower_yolov8n(POSE)
        ws = es.plan_workspace(cls.ir)
        scheds, store = es.schedule_graph(cls.ir, ws)
        manifest = build_manifest(cls.ir, ws, scheds, store, POSE.stem, 0, "x", "y", 0.0)
        cls.status = resolve_head_layout(manifest, int(manifest["egress_bytes"]))
        cls.head_of = {tensor: head_name_for(onnx_name) for onnx_name, tensor in cls.ir.outputs}
        cls.sess = ort.InferenceSession(str(POSE), providers=["CPUExecutionProvider"])
        cls.images = [ROOT / "assets" / "bus.jpg"]
        if COCO128.is_dir():
            cls.images += sorted(COCO128.rglob("*.jpg"))[:3]

    def test_decode_of_container_heads_equals_onnx_runtime_tail(self):
        from ignite_xdna.compiler import graph_reference as gr
        from ignite_xdna.pipelines.pose_pipeline import PoseDecoder, letterbox
        from npu.yolo_pose_decode import decode_heads
        self.assertTrue(self.status.present, self.status.reason)
        decoder = PoseDecoder()
        t_in = self.ir.tensors[self.ir.input]
        people_total = 0
        for path in self.images:
            img = self.cv2.imread(str(path))
            x, pad, scale = self.yolo.letterbox(img)
            x2, pad2, scale2 = letterbox(img)
            self.assertTrue(np.array_equal(x, x2) and pad == pad2 and scale == scale2, path.name)

            float_heads = self.sess.run(self.yp.HEAD_OUTS, {self.sess.get_inputs()[0].name: x})
            direct = gr.run_direct(self.ir, gr.quantize_input(x[0], t_in.scale, t_in.zero_point))
            int8 = {}
            for tensor, name in self.head_of.items():
                t = self.ir.tensors[tensor]
                int8[name] = (direct[tensor][:t.channels] ^ 0x80).view(np.int8)[None]
            egress = self.status.layout.pack(int8)
            views = self.status.layout.unpack(egress)
            scales = self.status.layout.scales()

            for conf, iou, max_det in SETTINGS:
                ref = self.yp.postprocess(decode_heads(float_heads, conf_thres=conf), pad, scale, conf, iou, max_det)
                got = decoder.postprocess(decoder.decode(views, scales, conf), pad, scale, conf, iou, max_det)
                from_float = decoder.postprocess(decoder.decode(float_heads, None, conf), pad, scale, conf, iou,
                                                 max_det)
                for result in (got, from_float):
                    self.assertEqual(len(result), len(ref), f"{path.name} conf {conf}")
                    for d, (x0, y0, w, h, s, kpts) in zip(result, ref):
                        self.assertEqual((np.float32(d.x0), np.float32(d.y0), np.float32(d.w), np.float32(d.h)),
                                         (x0, y0, w, h), path.name)
                        self.assertEqual(np.float32(d.score), s, path.name)
                        self.assertTrue(np.array_equal(d.keypoints, kpts), path.name)
                if conf == 0.25:
                    people_total += len(ref)
        self.assertGreater(people_total, 0, "no person found on any image at conf 0.25")


if __name__ == "__main__":
    unittest.main()
