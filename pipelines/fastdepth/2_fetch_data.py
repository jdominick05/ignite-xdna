"""
Step 2: Prepare calibration and evaluation images for FastDepth monocular depth estimation.

Selects images from data/midas_calib and data/midas_val (or data/coco/val2017),
setting up data/fastdepth_calib/ and data/fastdepth_val/.

Run in resnet_env or resnet_env17:
    python pipelines/fastdepth/2_fetch_data.py --calib-count 300 --val-count 100
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIDAS_CALIB = ROOT / "data" / "midas_calib"
MIDAS_VAL = ROOT / "data" / "midas_val"
COCO_VAL = ROOT / "data" / "coco" / "val2017"
CALIB_SRC = ROOT / "data" / "calib"

FASTDEPTH_CALIB = ROOT / "data" / "fastdepth_calib"
FASTDEPTH_VAL = ROOT / "data" / "fastdepth_val"


def main():
    ap = argparse.ArgumentParser(description="Prepare FastDepth calibration and validation images.")
    ap.add_argument("--calib-count", type=int, default=300, help="Number of calibration images (default: 300)")
    ap.add_argument("--val-count", type=int, default=100, help="Number of validation images (default: 100)")
    args = ap.parse_args()

    FASTDEPTH_CALIB.mkdir(parents=True, exist_ok=True)
    FASTDEPTH_VAL.mkdir(parents=True, exist_ok=True)

    # 1. Calib sources
    calib_srcs = []
    if MIDAS_CALIB.is_dir():
        calib_srcs = sorted(MIDAS_CALIB.glob("*.jpg"))
    elif COCO_VAL.is_dir():
        calib_srcs = sorted(COCO_VAL.glob("*.jpg"))
    elif CALIB_SRC.is_dir():
        calib_srcs = sorted(CALIB_SRC.glob("*.jpg"))

    if not calib_srcs:
        raise SystemExit("No candidate images found in data/midas_calib, data/coco/val2017, or data/calib.")

    calib_files = calib_srcs[:args.calib_count]
    print(f"Preparing {len(calib_files)} calibration images in {FASTDEPTH_CALIB}...")
    for src in calib_files:
        dst = FASTDEPTH_CALIB / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    # 2. Val sources
    val_srcs = []
    if MIDAS_VAL.is_dir():
        val_srcs = sorted(MIDAS_VAL.glob("*.jpg"))
    elif COCO_VAL.is_dir():
        val_srcs = sorted(COCO_VAL.glob("*.jpg"))[args.calib_count:]

    if not val_srcs:
        val_srcs = calib_srcs[args.calib_count:]

    val_files = val_srcs[:args.val_count]
    print(f"Preparing {len(val_files)} validation images in {FASTDEPTH_VAL}...")
    for src in val_files:
        dst = FASTDEPTH_VAL / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
