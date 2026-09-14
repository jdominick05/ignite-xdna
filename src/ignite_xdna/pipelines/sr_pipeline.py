# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/pipelines/sr_pipeline.py

Single-image super-resolution (SESR M7, 2x) on the Phoenix NPU from a graph-engine
``.ignite`` container (``task == "super_resolution"``). One dispatch per frame runs
every convolution on the NPU; the host resizes the frame into the input plane,
reads the dense tail tensor back and applies the model's DepthToSpace.

    with SuperResolutionPipeline("build/sesr_m7.ignite") as sr:
        image_bgr, timings = sr.predict_sync(frame_bgr)      # 512x512 for a 256x256 network input
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from ignite_xdna.runtime.graph_session import DenseGraphSession


@dataclass
class SrTimings:
    """Stage latencies of one frame in milliseconds; ``npu_forward_ms`` = dispatch + readback."""
    preprocess_ms: float
    npu_forward_ms: float
    dispatch_ms: float
    readback_ms: float
    postprocess_ms: float
    glass_to_glass_ms: float
    source: str = "npu"


class SuperResolutionPipeline:
    """Owns one ``DenseGraphSession`` (one XRT hardware context) for its lifetime."""

    def __init__(self, model_path: Union[str, Path], device_index: int = 0):
        self.model_path = Path(model_path)
        self.device_index = device_index
        self.session = DenseGraphSession(self.model_path, device_index=device_index)
        self.input_hw: Tuple[int, int] = self.session.input_hw
        self.scale: int = self.session.scale

    def predict_sync(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, SrTimings]:
        """BGR uint8 frame (any size; resized to the network input) -> upscaled BGR uint8 image."""
        session = self.session
        if session is None:
            raise RuntimeError("SuperResolutionPipeline is closed")
        t0 = time.perf_counter()
        session.stage_image(img_bgr)
        t1 = time.perf_counter()
        session.dispatch()
        t2 = time.perf_counter()
        raw = session.read_output()
        t3 = time.perf_counter()
        image = session.postprocess(raw)
        t4 = time.perf_counter()
        return image, SrTimings(preprocess_ms=(t1 - t0) * 1e3, npu_forward_ms=(t3 - t1) * 1e3,
                                dispatch_ms=(t2 - t1) * 1e3, readback_ms=(t3 - t2) * 1e3,
                                postprocess_ms=(t4 - t3) * 1e3, glass_to_glass_ms=(t4 - t0) * 1e3)

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
