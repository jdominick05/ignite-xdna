# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignite_xdna/pipelines/classification_pipeline.py

Image classification on the Phoenix NPU from a graph-engine ``.ignite`` container
(``task == "classify"``). The Gemm and the convolutions feeding it run natively on NPU
Device 0 -- the Gemm as a single 1x1 convolution.

Whether the container declares a host segment depends on where the average happens. A
head-only model, whose graph input already holds pooled features, declares none. A whole
classifier must declare one: nothing in the engine computes a global average, so
``lower_yolov8n`` carves the pooling into a host region, which makes that container a hybrid
in the manifest's own terms -- as YOLO11n's attention core is -- and not a network wholly on
the device. ``ClassificationPipeline`` runs either shape unchanged.

    with ClassificationPipeline("build/resnet50_head.ignite") as pipe:
        probs, timings = pipe.predict_sync(pooled_features)
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ignite_xdna.runtime.graph_session import ClassificationSession


@dataclass
class ClassificationTimings:
    """Stage latencies of one forward pass in milliseconds."""
    preprocess_ms: float
    npu_forward_ms: float
    dispatch_ms: float
    readback_ms: float
    postprocess_ms: float
    glass_to_glass_ms: float
    source: str = "npu"


class ClassificationPipeline:
    """Owns one ``ClassificationSession`` (one XRT hardware context) for its lifetime."""

    def __init__(self, model_path: Union[str, Path], device_index: int = 0):
        self.model_path = Path(model_path)
        self.device_index = device_index
        self.session = ClassificationSession(self.model_path, device_index=device_index)
        self.num_classes: int = self.session.num_classes
        self.input_hw: Tuple[int, int] = self.session.input_hw
        self.in_channels: int = self.session.in_channels

    def predict_sync(self, input_data: np.ndarray) -> Tuple[np.ndarray, ClassificationTimings]:
        """Input data [Cin] or [C, H, W] -> probabilities [Cout] and stage latencies."""
        session = self.session
        if session is None:
            raise RuntimeError("ClassificationPipeline is closed")
        t0 = time.perf_counter()
        session.stage_input(input_data)
        t1 = time.perf_counter()
        session.dispatch()
        t2 = time.perf_counter()
        logits = session.read_logits()
        t3 = time.perf_counter()
        # Softmax
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        t4 = time.perf_counter()
        return probs, ClassificationTimings(
            preprocess_ms=(t1 - t0) * 1e3,
            npu_forward_ms=(t3 - t1) * 1e3,
            dispatch_ms=(t2 - t1) * 1e3,
            readback_ms=(t3 - t2) * 1e3,
            postprocess_ms=(t4 - t3) * 1e3,
            glass_to_glass_ms=(t4 - t0) * 1e3,
        )

    def predict(self, pooled_features: np.ndarray, topk: int = 5) -> Tuple[List[Tuple[int, float]], ClassificationTimings]:
        """Return top-k (class_id, probability) pairs and timings."""
        probs, timings = self.predict_sync(pooled_features)
        k = min(int(topk), probs.size)
        idx = np.argpartition(-probs, k - 1)[:k]
        idx = idx[np.argsort(-probs[idx])]
        top = [(int(i), float(probs[i])) for i in idx]
        return top, timings

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
