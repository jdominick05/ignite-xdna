"""Replay the classification pipeline's exact preprocessing and file order."""
import glob
from pathlib import Path
from typing import Iterator

import numpy as np

from npu.preprocess import IMG_EXTS, build_transform


class ImageFolderSource:
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

    def listing(self) -> list[Path]:
        return list(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[np.ndarray]:
        for path in self.files:
            yield self.transform(path)


def as_reader(source: ImageFolderSource):
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
