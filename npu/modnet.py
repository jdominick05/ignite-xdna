"""MODNet portrait-matting preprocess/postprocess, in one place.

Quark-free by the `npu/` invariant, and cv2-only so it imports in `resnet_env17`
(which has no torch and no Pillow) as well as in `resnet_env`.

The transform was copied five times when the pipeline landed -- `3_quantize.py`,
`4_matte.py`, `5_eval.py`, `demos/portrait_matting_demo.py` and
`tools/train_on_user.py` each carried their own -- and the calibration copy resized
through `PIL.Image.BILINEAR` while every inference copy used `cv2.INTER_LINEAR`.
Pillow antialiases on downscale and OpenCV does not, so those two do not agree
pixel-for-pixel, which is exactly the calibration/inference drift the repo forbids.
"""

import cv2
import numpy as np

INPUT_SIZE = 512


def preprocess(img_bgr, target_size=INPUT_SIZE):
    """BGR uint8 HWC -> NCHW float32 in [-1, 1], plus the original (h, w).

    Returns the source shape because the matte is resized back to it, never to the
    square the network ran at.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    arr = (resized.astype(np.float32) - 127.5) / 127.5
    tensor = np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)
    return np.ascontiguousarray(tensor), (orig_h, orig_w)


def postprocess_matte(raw_matte, orig_shape):
    """Network output -> single-channel alpha in [0, 1] at the original resolution.

    The head-cut variants emit raw logits where the stock graph emits a sigmoid, so the
    sigmoid is applied only when the values fall outside [0, 1].
    """
    orig_h, orig_w = orig_shape
    matte = np.squeeze(raw_matte)
    if np.min(matte) < -0.01 or np.max(matte) > 1.01:
        matte = 1.0 / (1.0 + np.exp(-np.clip(matte, -30.0, 30.0)))
    matte = np.clip(matte, 0.0, 1.0)
    return cv2.resize(matte, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
