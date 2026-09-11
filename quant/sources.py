"""Replay each pipeline's exact preprocessing and file order for calibration."""
import glob
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from npu.modnet import preprocess as modnet_preprocess
from npu.preprocess import IMG_EXTS, build_transform
from npu.yolo import letterbox

MODNET_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPEG")  # 3_quantize.py PortraitCalibReader, in its order


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


class ModnetSource:
    """pipelines/modnet/3_quantize.py's PortraitCalibReader, through npu.modnet.preprocess.

    This is the point of the owned source: the matting pipeline's calibration and its
    inference now read the same function rather than two copies of one transform, so the
    PIL/OpenCV divergence that section 5 of Category B measured cannot recur silently.
    The listing rule is the reader's: a recursive glob per extension in the source's own
    order, then one sort over the union, then the limit. The size comes from the graph,
    never a flag, so it cannot drift either. Where the vendor reader silently skips an
    unreadable file, this raises, as the sibling sources do.
    """

    def __init__(self, folder: Path, limit: int | None, size: int, input_name: str = "input"):
        if limit is not None and limit <= 0:
            raise ValueError("Calibration limit must be positive")
        files = []
        for extension in MODNET_EXTS:
            files.extend(glob.glob(str(Path(folder) / "**" / extension), recursive=True))
        self.files = [Path(p) for p in sorted(files)[:limit]]
        if not self.files:
            raise ValueError(f"No calibration images under {folder}")
        self.size = int(size)
        self.input_name = input_name

    def listing(self) -> list[Path]:
        return list(self.files)

    def preprocess(self) -> dict:
        return {"family": "modnet", "size": self.size, "input_name": self.input_name,
                "transform": "npu.modnet.preprocess"}

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[np.ndarray]:
        for path in self.files:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"cv2 could not read {path}")
            yield modnet_preprocess(img, self.size)[0]


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


class StreamingHistogramAccumulator:
    """Online uniform histogram accumulator for in-memory streaming activations.

    Accumulates per-tensor and per-channel activation histograms in-memory during
    forward passes, eliminating disk spooling of intermediate float16 activations
    (reducing I/O overhead from O(samples * elements * 2 B) to O(bins * tensors)).
    """

    def __init__(self, num_bins: int = 2048, per_channel: bool = False, channel_axis: int = 1):
        if num_bins < 16:
            raise ValueError("Histogram bin count must be at least 16")
        self.num_bins = int(num_bins)
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.ranges: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.counts: dict[str, np.ndarray] = {}
        self.total_elements: dict[str, int] = {}

    def update_range(self, name: str, x: np.ndarray) -> None:
        """Update global dynamic range [vmin, vmax] per tensor or per channel."""
        arr = np.asarray(x, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Nonfinite values encountered during range collection for {name}")
        if self.per_channel and arr.ndim > self.channel_axis:
            axes = tuple(i for i in range(arr.ndim) if i != self.channel_axis)
            c_min = arr.min(axis=axes)
            c_max = arr.max(axis=axes)
            if name not in self.ranges:
                self.ranges[name] = (c_min.copy(), c_max.copy())
            else:
                curr_min, curr_max = self.ranges[name]
                np.minimum(curr_min, c_min, out=curr_min)
                np.maximum(curr_max, c_max, out=curr_max)
        else:
            vmin = float(arr.min())
            vmax = float(arr.max())
            if name not in self.ranges:
                self.ranges[name] = (np.array(vmin, dtype=np.float32), np.array(vmax, dtype=np.float32))
            else:
                curr_min, curr_max = self.ranges[name]
                if vmin < float(curr_min):
                    self.ranges[name] = (np.array(vmin, dtype=np.float32), curr_max)
                if vmax > float(curr_max):
                    self.ranges[name] = (self.ranges[name][0], np.array(vmax, dtype=np.float32))

    def init_histograms(self) -> None:
        """Preallocate histogram count arrays after ranges are known."""
        for name, (vmin, vmax) in self.ranges.items():
            if self.per_channel and vmin.ndim > 0:
                num_channels = vmin.shape[0]
                self.counts[name] = np.zeros((num_channels, self.num_bins), dtype=np.int64)
            else:
                self.counts[name] = np.zeros(self.num_bins, dtype=np.int64)
            self.total_elements[name] = 0

    def accumulate(self, name: str, x: np.ndarray) -> None:
        """Stream forward activations into uniform histogram bins in-memory."""
        arr = np.asarray(x, dtype=np.float32)
        if name not in self.ranges:
            raise KeyError(f"Range not initialized for {name}; call update_range and init_histograms first")
        if name not in self.counts:
            self.init_histograms()

        vmin, vmax = self.ranges[name]
        self.total_elements[name] = self.total_elements.get(name, 0) + arr.size

        if self.per_channel and vmin.ndim > 0:
            num_channels = vmin.shape[0]
            transposed = np.moveaxis(arr, self.channel_axis, 0)
            reshaped = transposed.reshape((num_channels, -1))
            for c in range(num_channels):
                c_vmin = float(vmin[c])
                c_vmax = float(vmax[c])
                c_data = reshaped[c]
                if c_vmax == c_vmin:
                    self.counts[name][c, 0] += c_data.size
                else:
                    delta = (c_vmax - c_vmin) / self.num_bins
                    bins = np.clip(((c_data - c_vmin) / delta).astype(np.int32), 0, self.num_bins - 1)
                    self.counts[name][c] += np.bincount(bins, minlength=self.num_bins)
        else:
            s_vmin = float(vmin)
            s_vmax = float(vmax)
            if s_vmax == s_vmin:
                self.counts[name][0] += arr.size
            else:
                delta = (s_vmax - s_vmin) / self.num_bins
                bins = np.clip(((arr - s_vmin) / delta).astype(np.int32), 0, self.num_bins - 1)
                self.counts[name] += np.bincount(bins.ravel(), minlength=self.num_bins)

    def get_histogram(self, name: str) -> tuple[float | np.ndarray, float | np.ndarray, np.ndarray]:
        """Return (vmin, vmax, counts) for tensor name."""
        vmin, vmax = self.ranges[name]
        return (vmin if vmin.ndim > 0 else float(vmin),
                vmax if vmax.ndim > 0 else float(vmax),
                self.counts[name])

