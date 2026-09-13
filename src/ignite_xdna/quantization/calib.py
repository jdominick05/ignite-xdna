#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/quantization/calib.py

Calibration dataset loader and activation collection for Post-Training Quantization (PTQ).
Loads images from COCO calibration datasets (e.g. data/coco128/ or data/coco_calib/),
applies standard YOLOv8 letterbox preprocessing, and runs forward passes to record
intermediate layer activation statistics for AdaRound and scale determination.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union
import cv2
import numpy as np
import onnx
from onnx import helper


def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Standard YOLOv8 letterbox preprocessing resizing with padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess_image(
    image_path: Union[str, Path],
    img_size: int = 640,
) -> np.ndarray:
    """Reads image from disk and formats into [1, 3, img_size, img_size] float32 tensor."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")

    img, _, _ = letterbox(img_bgr, (img_size, img_size), auto=False)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)


class CalibrationDataset:
    """Loads and preprocesses calibration images from a directory."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        max_samples: int = 128,
        img_size: int = 640,
    ):
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.img_size = img_size
        self.image_paths = self._discover_images()

    def _discover_images(self) -> List[Path]:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = [
            p for p in sorted(self.data_dir.iterdir())
            if p.is_file() and p.suffix.lower() in valid_exts
        ]
        return paths[:self.max_samples]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self) -> Iterator[np.ndarray]:
        for path in self.image_paths:
            yield preprocess_image(path, img_size=self.img_size)

    def load_all_batches(self, batch_size: int = 1) -> List[np.ndarray]:
        """Loads and stacks calibration images into batches."""
        images = [preprocess_image(p, img_size=self.img_size) for p in self.image_paths]
        if not images:
            return []
        batches = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            batches.append(np.concatenate(chunk, axis=0))
        return batches


def collect_calibration_activations(
    model: onnx.ModelProto,
    calib_dataset: CalibrationDataset,
    target_tensor_names: Optional[List[str]] = None,
    max_samples: int = 16,
) -> Dict[str, np.ndarray]:
    """
    Runs calibration images through the model using ONNX Runtime to collect
    intermediate activation tensors for AdaRound optimization and scale selection.
    """
    import onnxruntime as ort

    # Clone model and expose intermediate tensors as graph outputs
    probe_model = onnx.ModelProto.FromString(model.SerializeToString())
    graph = probe_model.graph

    existing_outputs = {o.name for o in graph.output}
    if target_tensor_names is None:
        # Collect inputs to all Conv nodes
        target_tensor_names = []
        for node in graph.node:
            if node.op_type == "Conv" and len(node.input) > 0:
                target_tensor_names.append(node.input[0])
        target_tensor_names = sorted(set(target_tensor_names))

    for name in target_tensor_names:
        if name not in existing_outputs:
            graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))

    sess = ort.InferenceSession(
        probe_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    inp_name = sess.get_inputs()[0].name
    collected: Dict[str, List[np.ndarray]] = {name: [] for name in target_tensor_names}

    samples_run = 0
    for img_tensor in calib_dataset:
        if samples_run >= max_samples:
            break
        outputs = sess.run(target_tensor_names, {inp_name: img_tensor})
        for name, val in zip(target_tensor_names, outputs):
            collected[name].append(val)
        samples_run += 1

    # Concatenate across samples: [N, C, H, W]
    merged = {
        name: np.concatenate(vals, axis=0) if vals else np.zeros((1, 1), dtype=np.float32)
        for name, vals in collected.items()
    }
    return merged
