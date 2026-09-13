# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/runtime/loader.py

Zero-Copy Runtime Loader for .ignite binary containers on AMD Phoenix silicon.
Provides high-level one-liner callable execution interface:
  engine = ignite_xdna.load("yolov8n.ignite")
  results = engine(image)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ignite_xdna.compiler.serializer import IgniteModelReader
from ignite_xdna.runtime.session import InferenceSession


class IgniteEngine:
    """
    High-level bare-metal inference engine wrapping a memory-mapped .ignite container.
    Binds 64-byte aligned transaction buffers directly to PyXRT instruction objects with 0 memcpy.
    """

    def __init__(
        self,
        ignite_path: Union[str, Path],
        device_index: int = 0,
        conf_thres: float = 0.25,
        iou_thres: float = 0.50,
        **kwargs,
    ):
        self.path = Path(ignite_path)
        if not self.path.exists():
            raise FileNotFoundError(f".ignite model container not found: {self.path}")

        # Initialize bare-metal session with zero-copy memory-mapped container
        self.session = InferenceSession.from_file(self.path, device_index=device_index, **kwargs)
        self.manifest: Dict[str, Any] = self.session.ignite_manifest or {}
        self.model_name: str = self.manifest.get("model_name", self.path.stem)
        self.stages: List[str] = list(self.session.monolithic_stages.keys())
        self.quant_scales: Dict[str, Any] = self.manifest.get("quant_scales", {})

        # If YOLO model metadata is present, wire the streaming detection pipeline
        self.pipeline = None
        if "strides" in self.manifest or "yolo" in self.model_name.lower():
            from ignite_xdna.pipelines.yolo_pipeline import YoloPipeline
            self.pipeline = YoloPipeline(
                model_path_or_bundle=self.path,
                device_index=device_index,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
            )
            # Share active session
            self.pipeline.session = self.session

    def __call__(self, input_data: Any) -> Any:
        """
        Executes end-to-end inference:
        - If passed a 3D OpenCV image (HWC uint8 BGR/RGB), runs full detection pipeline.
        - If passed a 4D tensor (1, 3, 640, 640), runs monolithic NPU forward pass.
        """
        if self.pipeline is not None:
            if isinstance(input_data, np.ndarray) and input_data.ndim == 3:
                dets, _ = self.pipeline.predict_sync(input_data, use_oracle_for_boxes=False)
                return dets
            elif isinstance(input_data, np.ndarray) and input_data.ndim == 4:
                return self.session.run_yolo_monolithic(input_data)
        return self.session.run(input_data)

    def predict(self, input_data: Any) -> Any:
        """Alias for __call__."""
        return self.__call__(input_data)

    def close(self):
        """Releases hardware context, memory mappings, and buffers."""
        if self.pipeline is not None:
            self.pipeline = None
        if self.session is not None:
            self.session.close()
            self.session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def load(
    model_path: Union[str, Path],
    device_index: int = 0,
    **kwargs,
) -> IgniteEngine:
    """
    One-liner entry point to load a compiled .ignite model container on AMD Phoenix silicon.

    Usage:
        import ignite_xdna
        engine = ignite_xdna.load("yolov8n.ignite")
        detections = engine(cv2.imread("image.jpg"))
    """
    return IgniteEngine(model_path, device_index=device_index, **kwargs)
