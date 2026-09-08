"""MiDaS monocular depth estimation preprocess/postprocess, in one place.

Quark-free by the `npu/` invariant, and cv2-only so it imports in `resnet_env17`
(which has no torch and no Pillow) as well as in `resnet_env`.

Preprocesses images using standard ImageNet mean/std normalization and provides
depth normalization and colormap visualization utilities for evaluation and demos.
"""
import cv2
import numpy as np

INPUT_SIZE = 256

# Standard ImageNet normalization parameters used by MiDaS v2.1 Small
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr, target_size=INPUT_SIZE):
    """BGR uint8 HWC -> NCHW float32 normalized, plus original (orig_h, orig_w).

    Resizes via cv2.INTER_LINEAR, converts BGR to RGB, normalizes by ImageNet
    mean and std, and returns a contiguous (1, 3, target_size, target_size) array.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)
    return np.ascontiguousarray(tensor, dtype=np.float32), (orig_h, orig_w)


def postprocess_depth(raw_depth, orig_shape=None, normalize=True):
    """Network output -> relative inverse depth map.

    If orig_shape is provided as (orig_h, orig_w), resizes back to the original aspect ratio.
    If normalize is True, performs min-max normalization to uint8 [0, 255].
    """
    depth = np.squeeze(raw_depth)
    if orig_shape is not None:
        orig_h, orig_w = orig_shape
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    if not normalize:
        return depth

    d_min = float(np.min(depth))
    d_max = float(np.max(depth))
    if d_max - d_min > 1e-6:
        depth_u8 = ((depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    else:
        depth_u8 = np.zeros_like(depth, dtype=np.uint8)
    return depth_u8


def colorize_depth(depth_u8, colormap=cv2.COLORMAP_INFERNO):
    """Applies a colormap to a uint8 [0, 255] depth map (default: INFERNO)."""
    return cv2.applyColorMap(depth_u8, colormap)


def input_size(dims, where="model"):
    """Validate an NCHW input shape and return its square side length."""
    if len(dims) != 4:
        raise SystemExit(f"{where}: expected a 4D NCHW input, got shape {dims}")
    h, w = dims[2], dims[3]
    if not isinstance(h, int) or not isinstance(w, int) or h <= 0 or w <= 0:
        raise SystemExit(
            f"{where}: input spatial dims are not static ({dims}). Export with "
            "static batch=1 and fixed spatial dimensions.")
    if h != w:
        raise SystemExit(f"{where}: expected square input, got {h}x{w}")
    return h
