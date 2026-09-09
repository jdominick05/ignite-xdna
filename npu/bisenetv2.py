"""BiSeNetV2 bilateral semantic segmentation preprocess/postprocess, in one place.

Quark-free by the `npu/` invariant, and cv2-only so it imports in `resnet_env17`
(which has no torch and no Pillow) as well as in `resnet_env`.

Preprocesses images using standard ImageNet mean/std normalization and provides
segmentation palette colorization, class argmax, and alpha overlay utilities.
"""
import cv2
import numpy as np

INPUT_SIZE = 512

# Standard 19 evaluation classes for Cityscapes
CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

# Official Cityscapes color palette (RGB)
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],   # 0: road
    [244, 35, 232],   # 1: sidewalk
    [70, 70, 70],     # 2: building
    [102, 102, 156],  # 3: wall
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170, 30],   # 6: traffic light
    [220, 220, 0],    # 7: traffic sign
    [107, 142, 35],   # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [70, 130, 180],   # 10: sky
    [220, 20, 60],    # 11: person
    [255, 0, 0],      # 12: rider
    [0, 0, 142],      # 13: car
    [0, 0, 70],       # 14: truck
    [0, 60, 100],     # 15: bus
    [0, 80, 100],     # 16: train
    [0, 0, 230],      # 17: motorcycle
    [119, 11, 32],    # 18: bicycle
], dtype=np.uint8)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess(img_bgr, target_size=INPUT_SIZE):
    """BGR uint8 HWC -> NCHW float32 normalized, plus original (orig_h, orig_w).

    Resizes via cv2.INTER_LINEAR, converts BGR to RGB, scales to [0, 1],
    normalizes with ImageNet mean/std, and returns a contiguous
    (1, 3, target_size, target_size) float32 array.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)
    return np.ascontiguousarray(tensor, dtype=np.float32), (orig_h, orig_w)


def postprocess_mask(raw_logits, orig_shape=None):
    """Network output (1, 19, H, W) or (19, H, W) -> uint8 class mask (H, W).

    If orig_shape is provided as (orig_h, orig_w), resizes the mask back
    using nearest-neighbor interpolation to preserve integer class indices.
    """
    logits = np.squeeze(raw_logits)
    if logits.ndim != 3 or logits.shape[0] != len(CITYSCAPES_CLASSES):
        raise ValueError(f"Expected logits of shape (19, H, W), got {logits.shape}")
    mask = np.argmax(logits, axis=0).astype(np.uint8)

    if orig_shape is not None:
        orig_h, orig_w = orig_shape
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return mask


def colorize_mask(class_mask):
    """Maps uint8 class indices [0, 18] to a BGR image using CITYSCAPES_PALETTE."""
    clamped = np.clip(class_mask, 0, len(CITYSCAPES_CLASSES) - 1)
    rgb_colored = CITYSCAPES_PALETTE[clamped]
    return cv2.cvtColor(rgb_colored, cv2.COLOR_RGB2BGR)


def overlay_segmentation(img_bgr, class_mask, alpha=0.5):
    """Alpha-blends the colorized segmentation mask onto the original BGR image."""
    orig_h, orig_w = img_bgr.shape[:2]
    if class_mask.shape[:2] != (orig_h, orig_w):
        class_mask = cv2.resize(class_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    color_mask = colorize_mask(class_mask)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, color_mask, alpha, 0)


def input_size(dims, where="model"):
    """Validate an NCHW input shape and return its square side length."""
    if len(dims) != 4:
        raise SystemExit(f"{where}: expected a 4D NCHW input, got shape {dims}")
    n, c, h, w = dims
    if n != 1:
        raise SystemExit(f"{where}: expected batch size 1, got {n}")
    if c != 3:
        raise SystemExit(f"{where}: expected 3 color channels, got {c}")
    if h != w:
        raise SystemExit(f"{where}: expected square input, got {h}x{w}")
    return int(h)
