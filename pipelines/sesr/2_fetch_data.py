"""
SESR step 2: Fetch validation benchmark datasets and prepare calibration data.

Downloads standard SISR benchmark datasets (Set5 and Set14) from Hugging Face:
  - Set5:  5 standard benchmark images (baby, bird, butterfly, head, woman)
  - Set14: 14 standard benchmark images (baboon, barbara, bridge, comic, lenna...)
Organized into pairs of HR (ground truth) and LR_x2 (low-resolution bicubic input).

Also extracts 100-200 256x256 calibration crops into data/sesr_calib/ from
existing data/calib/ or data/coco/val2017/.

    conda activate resnet_env
    python pipelines/sesr/2_fetch_data.py
"""
import glob
import io
import os
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from npu.paths import DATA
from npu.sesr import INPUT_SIZE

SET5_HR_URL = "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_HR.tar.gz"
SET5_LR_URL = "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_LR_x2.tar.gz"
SET14_HR_URL = "https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz"
SET14_LR_URL = "https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_LR_x2.tar.gz"

VAL_DIR = DATA / "sesr_val"
CALIB_DIR = DATA / "sesr_calib"


def fetch_and_extract_tar(url: str, dest_dir: Path, subfolder_name: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = dest_dir / subfolder_name
    if target_path.exists() and any(target_path.iterdir()):
        print(f"  {target_path} already exists, skipping download.")
        return

    print(f"  Downloading {url} ...")
    resp = urlopen(url)
    data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(path=dest_dir)
    print(f"  Extracted to {target_path}")


def prepare_calib_crops(n_crops: int = 100):
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(CALIB_DIR.glob("*.jpg")) + list(CALIB_DIR.glob("*.png"))
    if len(existing) >= n_crops:
        print(f"  Calibration directory {CALIB_DIR} already has {len(existing)} images.")
        return

    # Look for source images in data/calib or data/coco/val2017
    sources = []
    for candidate in [DATA / "calib", DATA / "coco" / "val2017", DATA / "midas_val"]:
        if candidate.exists():
            sources.extend(sorted(candidate.glob("*.jpg")) + sorted(candidate.glob("*.png")))
            if len(sources) >= n_crops:
                break

    if not sources:
        print("  No local image pool found; extracting crops from validation sets...")
        sources = list(VAL_DIR.rglob("*_HR/*.png"))

    print(f"  Extracting {n_crops} crops of {INPUT_SIZE}x{INPUT_SIZE} into {CALIB_DIR} ...")
    count = 0
    for img_path in sources:
        if count >= n_crops:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < INPUT_SIZE or w < INPUT_SIZE:
            img = cv2.resize(img, (max(h, INPUT_SIZE), max(w, INPUT_SIZE)))
            h, w = img.shape[:2]

        # Center crop
        top = (h - INPUT_SIZE) // 2
        left = (w - INPUT_SIZE) // 2
        crop = img[top : top + INPUT_SIZE, left : left + INPUT_SIZE]
        out_name = CALIB_DIR / f"crop_{count:04d}.png"
        cv2.imwrite(str(out_name), crop)
        count += 1

    print(f"  Prepared {count} calibration crops in {CALIB_DIR}.")


def main():
    print("=== Step 2: Fetch SESR Validation and Calibration Data ===")
    print("\nFetching Set5 benchmark...")
    fetch_and_extract_tar(SET5_HR_URL, VAL_DIR, "Set5_HR")
    fetch_and_extract_tar(SET5_LR_URL, VAL_DIR, "Set5_LR_x2")

    print("\nFetching Set14 benchmark...")
    fetch_and_extract_tar(SET14_HR_URL, VAL_DIR, "Set14_HR")
    fetch_and_extract_tar(SET14_LR_URL, VAL_DIR, "Set14_LR_x2")

    print("\nPreparing calibration crops...")
    prepare_calib_crops(n_crops=100)

    print("\nData preparation complete!")
    set5_count = len(list((VAL_DIR / "Set5_HR").glob("*.png")))
    set14_count = len(list((VAL_DIR / "Set14_HR").glob("*.png")))
    calib_count = len(list(CALIB_DIR.glob("*.png")))
    print(f"  Set5 pairs:  {set5_count} images")
    print(f"  Set14 pairs: {set14_count} images")
    print(f"  Calib crops: {calib_count} images ({INPUT_SIZE}x{INPUT_SIZE})")


if __name__ == "__main__":
    main()
