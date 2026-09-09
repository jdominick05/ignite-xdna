"""Real-ESRGAN step 2: Fetch validation benchmark datasets and prepare calibration data.

Downloads standard SISR 4x benchmark datasets (Set5 and Set14) from Hugging Face:
  - Set5:  5 standard benchmark images (baby, bird, butterfly, head, woman)
  - Set14: 14 standard benchmark images (baboon, barbara, bridge, comic, lenna...)
Organized into pairs of HR (ground truth) and LR_x4 (low-resolution bicubic 4x downscaled input).

Also extracts 100 calibration crops of size target_size (default: 64x64) into data/realesrgan_calib/.

    conda activate resnet_env
    python pipelines/realesrgan/2_fetch_data.py
"""
import argparse
import io
import os
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from npu.paths import DATA
from npu.realesrgan import DEFAULT_INPUT_SIZE

SET5_HR_URL = "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_HR.tar.gz"
SET5_LR_URL = "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_LR_x4.tar.gz"
SET14_HR_URL = "https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz"
SET14_LR_URL = "https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_LR_x4.tar.gz"

VAL_DIR = DATA / "realesrgan_val"
CALIB_DIR = DATA / "realesrgan_calib"


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


def prepare_calib_crops(crop_size: int = DEFAULT_INPUT_SIZE, n_crops: int = 100):
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(CALIB_DIR.glob("*.jpg")) + list(CALIB_DIR.glob("*.png"))
    if len(existing) >= n_crops:
        print(f"  Calibration directory {CALIB_DIR} already has {len(existing)} images.")
        return

    # Look for source images in data pools
    sources = []
    for candidate in [
        DATA / "sesr_calib",
        DATA / "calib",
        DATA / "coco" / "val2017",
        DATA / "midas_val",
    ]:
        if candidate.exists():
            sources.extend(sorted(candidate.glob("*.jpg")) + sorted(candidate.glob("*.png")))
            if len(sources) >= n_crops:
                break

    if not sources:
        print("  Extracting crops from validation sets...")
        sources = list(VAL_DIR.rglob("*_HR/*.png"))

    if not sources:
        raise SystemExit(f"No source images found to generate calibration crops in {DATA}")

    print(f"  Extracting {n_crops} {crop_size}x{crop_size} crops into {CALIB_DIR} ...")
    count = 0
    for src_path in sources:
        img = cv2.imread(str(src_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < crop_size or w < crop_size:
            continue

        # Center crop or grid crop
        start_y = (h - crop_size) // 2
        start_x = (w - crop_size) // 2
        crop = img[start_y : start_y + crop_size, start_x : start_x + crop_size]

        out_name = f"crop_{count:04d}_{src_path.stem}.png"
        cv2.imwrite(str(CALIB_DIR / out_name), crop)
        count += 1
        if count >= n_crops:
            break

    print(f"  Extracted {count} crops to {CALIB_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Real-ESRGAN benchmark datasets and build calibration pool.")
    parser.add_argument(
        "--crop-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help=f"Calibration crop size (default: {DEFAULT_INPUT_SIZE}).",
    )
    parser.add_argument("--n-crops", type=int, default=100, help="Number of calibration crops (default: 100).")
    args = parser.parse_args()

    print("Fetching Set5 4x benchmark...")
    fetch_and_extract_tar(SET5_HR_URL, VAL_DIR, "Set5_HR")
    fetch_and_extract_tar(SET5_LR_URL, VAL_DIR, "Set5_LR_x4")

    print("Fetching Set14 4x benchmark...")
    fetch_and_extract_tar(SET14_HR_URL, VAL_DIR, "Set14_HR")
    fetch_and_extract_tar(SET14_LR_URL, VAL_DIR, "Set14_LR_x4")

    print("Preparing calibration crops...")
    prepare_calib_crops(crop_size=args.crop_size, n_crops=args.n_crops)
    print("Done.")


if __name__ == "__main__":
    main()
