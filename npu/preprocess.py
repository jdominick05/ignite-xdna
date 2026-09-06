"""Shared preprocessing for the classification pipeline. Kept quark-free so the
inference script can import it without triggering Quark's custom-op build."""
import numpy as np
from PIL import Image

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPEG")


def build_transform(cfg):
    c, h, w = cfg["input_size"]
    mean = np.array(cfg["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(cfg["std"], dtype=np.float32).reshape(3, 1, 1)
    crop_pct = cfg.get("crop_pct", 0.875)
    resize_to = int(h / crop_pct)  # timm uses floor

    def transform(path):
        img = Image.open(path).convert("RGB")
        iw, ih = img.size
        if iw < ih:
            new_w, new_h = resize_to, int(round(ih * resize_to / iw))
        else:
            new_h, new_w = resize_to, int(round(iw * resize_to / ih))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        img = img.crop((left, top, left + w, top + h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        arr = (arr - mean) / std
        return arr[np.newaxis, ...].astype(np.float32)

    return transform
