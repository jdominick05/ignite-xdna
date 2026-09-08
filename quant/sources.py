"""Replay each pipeline's exact preprocessing and file order for calibration."""
import glob
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from npu.preprocess import IMG_EXTS, build_transform
from npu.yolo import letterbox


class ImageFolderSource:
    """The classification pipeline's reader: recursive IMG_EXTS glob, sorted, timm transform."""

    def __init__(self, folder: Path, cfg: dict, limit: int | None, input_name: str = "input"):
        if limit is not None and limit <= 0:
            raise ValueError("Calibration limit must be positive")
        files = []
        for extension in IMG_EXTS:
            files.extend(glob.glob(str(Path(folder) / "**" / extension), recursive=True))
        self.files = [Path(p) for p in sorted(files)[:limit]]
        if not self.files:
            raise ValueError(f"No calibration images under {folder}")
        self.transform = build_transform(cfg)
        self.input_name = input_name
        self.cfg = cfg

    def listing(self) -> list[Path]:
        return list(self.files)

    def preprocess(self) -> dict:
        return dict(self.cfg)

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[np.ndarray]:
        for path in self.files:
            yield self.transform(path)


class CocoSource:
    """pipelines/yolov8n/3b_quantize_cut.py's CocoCalibReader: sorted *.jpg, npu.yolo.letterbox.

    The letterbox size comes from the graph input, never a flag, so calibration and
    inference read the same number the same way.
    """

    def __init__(self, folder: Path, limit: int | None, imgsz: int, input_name: str = "images"):
        if limit is not None and limit <= 0:
            raise ValueError("Calibration limit must be positive")
        self.files = [Path(p) for p in sorted(glob.glob(str(Path(folder) / "*.jpg")))[:limit]]
        if not self.files:
            raise ValueError(f"No jpgs in {folder}")
        self.imgsz = int(imgsz)
        self.input_name = input_name

    def listing(self) -> list[Path]:
        return list(self.files)

    def preprocess(self) -> dict:
        return {"family": "yolo_cut", "letterbox": self.imgsz, "input_name": self.input_name}

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[np.ndarray]:
        for path in self.files:
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"cv2 could not read {path}")
            yield letterbox(img, self.imgsz)[0]


def as_reader(source):
    """Quark's calibration API is duck-typed; this adapter imports no quantizer."""
    class Reader:
        def __init__(self):
            self.rewind()

        def get_next(self):
            value = next(self.iterator, None)
            return None if value is None else {source.input_name: value}

        def rewind(self):
            self.iterator = iter(source)

        def __len__(self):
            return len(source)

    return Reader()
