"""SESR super-resolution preprocess/postprocess and fidelity metrics, in one place.

Quark-free by the `npu/` invariant, and cv2-only so it imports in `resnet_env17`
(which has no torch and no Pillow) as well as in `resnet_env`.

Preprocesses images using SESR's 128.0 pixel centering and provides tiling,
reconstruction, PSNR (dB), and SSIM evaluation utilities for SISR models.
"""
import math
import cv2
import numpy as np

INPUT_SIZE = 256
SCALE = 2
MEAN_RGB = 128.0


def preprocess(img_bgr, target_size=INPUT_SIZE):
    """BGR uint8 HWC -> NCHW float32 centered by 128.0, plus original (orig_h, orig_w).

    Resizes to (target_size, target_size) via cv2.INTER_LINEAR, converts BGR to RGB,
    subtracts 128.0 to match SESR's trained activation space [-128.0, 127.0],
    and returns a contiguous (1, 3, target_size, target_size) float32 array.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if (orig_h, orig_w) != (target_size, target_size):
        img_rgb = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    arr = img_rgb.astype(np.float32) - MEAN_RGB
    tensor = np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)
    return np.ascontiguousarray(tensor, dtype=np.float32), (orig_h, orig_w)


def postprocess(pred_chw):
    """Network output (NCHW or CHW float32) -> BGR uint8 image.

    Adds back 128.0 centering offset, clips to [0, 255], and converts RGB to BGR.
    """
    pred = np.squeeze(pred_chw)
    if pred.ndim != 3:
        raise ValueError(f"Expected 3D (C, H, W) tensor after squeeze, got shape {pred.shape}")
    pred_rgb = np.transpose(pred, (1, 2, 0)) + MEAN_RGB
    pred_u8 = np.clip(pred_rgb, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(pred_u8, cv2.COLOR_RGB2BGR)


def bgr2y(img_bgr):
    """Extract standard ITU-R BT.601 luminance (Y channel) from BGR uint8 image."""
    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return ycbcr[:, :, 0]


def compute_psnr(img1, img2, y_channel=True):
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB between two BGR uint8 images.

    If y_channel is True, evaluates on the luminance (Y) channel per SISR standard.
    """
    if y_channel and img1.ndim == 3:
        i1 = bgr2y(img1)
        i2 = bgr2y(img2)
    else:
        i1 = img1
        i2 = img2
    mse = np.mean((i1.astype(np.float64) - i2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1, img2, y_channel=True):
    """Compute Structural Similarity Index (SSIM) between two BGR uint8 images.

    If y_channel is True, evaluates on the luminance (Y) channel per SISR standard.
    Uses standard 11x11 Gaussian window (sigma=1.5, K1=0.01, K2=0.03).
    """
    if y_channel and img1.ndim == 3:
        i1 = bgr2y(img1).astype(np.float64)
        i2 = bgr2y(img2).astype(np.float64)
    else:
        i1 = img1.astype(np.float64)
        i2 = img2.astype(np.float64)

    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(i1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(i2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(i1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(i2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(i1 * i2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(np.mean(ssim_map))


def split_into_tiles(img_bgr, patch_size=(INPUT_SIZE, INPUT_SIZE), overlap=8):
    """Split arbitrary-size BGR image into overlapping tiles with reflect-padding."""
    h, w = img_bgr.shape[:2]
    ph, pw = patch_size
    core_h = ph - 2 * overlap
    core_w = pw - 2 * overlap
    n_h = math.ceil(h / core_h)
    n_w = math.ceil(w / core_w)
    h_pad = n_h * core_h
    w_pad = n_w * core_w

    pad_h = h_pad - h
    pad_w = w_pad - w
    img_pad = np.pad(img_bgr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    big_pad = np.pad(img_pad, ((overlap, overlap), (overlap, overlap), (0, 0)), mode="reflect")

    tiles = []
    for iy in range(n_h):
        for ix in range(n_w):
            cy = iy * core_h
            cx = ix * core_w
            tile = big_pad[cy : cy + ph, cx : cx + pw]
            tiles.append(tile)

    return tiles, (h, w), (h_pad, w_pad), (n_h, n_w)


def merge_tiles(sr_tiles, orig_hw, padded_hw, grid_hw, scale=SCALE, overlap=8):
    """Merge upscaled tiles into full-resolution image, stripping reflect overlap."""
    h, w = orig_hw
    h_pad, w_pad = padded_hw
    n_h, n_w = grid_hw

    ph, pw = sr_tiles[0].shape[:2]
    sr_overlap = overlap * scale
    core_h = ph - 2 * sr_overlap
    core_w = pw - 2 * sr_overlap

    sr_h_pad = h_pad * scale
    sr_w_pad = w_pad * scale
    recon = np.zeros((sr_h_pad, sr_w_pad, 3), dtype=sr_tiles[0].dtype)

    idx = 0
    for iy in range(n_h):
        for ix in range(n_w):
            cy = iy * core_h
            cx = ix * core_w
            core = sr_tiles[idx][sr_overlap : sr_overlap + core_h, sr_overlap : sr_overlap + core_w]
            recon[cy : cy + core_h, cx : cx + core_w] = core
            idx += 1

    return np.ascontiguousarray(recon[: h * scale, : w * scale])


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
