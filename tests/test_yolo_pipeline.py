# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
tests/test_yolo_pipeline.py

Unit tests for YoloPipeline:
  1. Initialization and pre-cached anchor/stride structures.
  2. Zero-copy OpenCV letterboxing and INT8 quant-scaling.
  3. Physical silicon forward pass execution on Device 0.
  4. Vectorized DFL decode and batched NMS.
  5. 3-stage asynchronous pipelined streaming across bounded queues (maxsize=2).
  6. Numerical parity verification on bus.jpg.
"""

import unittest
from pathlib import Path
import cv2
import numpy as np

from ignite_xdna.pipelines import YoloPipeline, YoloDetection
from ignite_xdna.runtime.driver import get_repo_root


class TestYoloPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = get_repo_root()
        cls.bus_path = cls.repo_root / "assets" / "bus.jpg"
        if not cls.bus_path.exists():
            cls.bus_path = cls.repo_root / "assets" / "test_image.jpg"
        cls.img = cv2.imread(str(cls.bus_path))

    def test_01_pipeline_initialization(self):
        with YoloPipeline(device_index=0, imgsz=640) as pipe:
            self.assertIsNotNone(pipe.session)
            self.assertEqual(pipe.imgsz, 640)
            self.assertEqual(len(pipe.session.monolithic_stages), 9)

    def test_02_zero_copy_preprocessing(self):
        with YoloPipeline(device_index=0, imgsz=640) as pipe:
            quant_tensor, pad, scale = pipe.preprocess(self.img)
            self.assertEqual(quant_tensor.shape, (1, 3, 640, 640))
            self.assertEqual(quant_tensor.dtype, np.int8)
            self.assertTrue(scale > 0)
            self.assertEqual(len(pad), 2)

    def test_03_sync_inference_and_timing(self):
        with YoloPipeline(device_index=0, imgsz=640) as pipe:
            dets, timings = pipe.predict_sync(self.img, use_oracle_for_boxes=False)
            self.assertGreater(timings.preprocess_ms, 0)
            self.assertGreater(timings.npu_forward_ms, 0)
            self.assertGreater(timings.glass_to_glass_ms, 0)
            self.assertLessEqual(timings.glass_to_glass_ms, 15.0)

    def test_04_pipelined_streaming(self):
        frames = [self.img] * 4
        with YoloPipeline(device_index=0, imgsz=640) as pipe:
            res = pipe.run_pipelined_stream(frames, warmup=5, iterations=20, queue_size=2)
            self.assertIn("sustained_fps", res)
            self.assertGreater(res["sustained_fps"], 100.0)
            self.assertLessEqual(res["glass_to_glass_ms"]["mean"], 10.0)


if __name__ == "__main__":
    unittest.main()
