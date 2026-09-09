"""Stage calibration and validation datasets for BiSeNetV2.

Links or copies 300 calibration images to data/bisenetv2_calib/ and
50 validation scenes to data/bisenetv2_val/ from data/coco/val2017.
"""
import glob
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = REPO_ROOT / "data" / "bisenetv2_calib"
VAL_DIR = REPO_ROOT / "data" / "bisenetv2_val"
COCO_VAL = REPO_ROOT / "data" / "coco" / "val2017"
CALIB_LIMIT = 300
VAL_LIMIT = 50


def main():
    if not COCO_VAL.is_dir():
        raise SystemExit(f"COCO val2017 directory not found at {COCO_VAL}")

    images = sorted(glob.glob(str(COCO_VAL / "*.jpg")))
    if len(images) < CALIB_LIMIT + VAL_LIMIT:
        raise SystemExit(f"Need at least {CALIB_LIMIT + VAL_LIMIT} images, found {len(images)}")

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Staging {CALIB_LIMIT} calibration images in {CALIB_DIR}...")
    for img in images[:CALIB_LIMIT]:
        dst = CALIB_DIR / os.path.basename(img)
        if not dst.exists():
            shutil.copy2(img, dst)

    print(f"Staging {VAL_LIMIT} validation images in {VAL_DIR}...")
    for img in images[CALIB_LIMIT:CALIB_LIMIT + VAL_LIMIT]:
        dst = VAL_DIR / os.path.basename(img)
        if not dst.exists():
            shutil.copy2(img, dst)

    print(f"Data staging complete: {len(list(CALIB_DIR.glob('*.jpg')))} calib, {len(list(VAL_DIR.glob('*.jpg')))} val")


if __name__ == "__main__":
    main()
