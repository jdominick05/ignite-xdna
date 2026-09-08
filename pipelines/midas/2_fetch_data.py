"""
Step 2: Prepare calibration and evaluation images for MiDaS monocular depth estimation.

Selects diverse indoor and outdoor scenes from data/coco/val2017 or data/calib,
copying them to data/midas_calib/ (for Quark XINT8 calibration) and
data/midas_val/ (for quantitative depth evaluation).

Run in resnet_env or resnet_env17:
    python pipelines/midas/2_fetch_data.py --calib-count 300 --val-count 100
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COCO_VAL = ROOT / "data" / "coco" / "val2017"
CALIB_SRC = ROOT / "data" / "calib"
MIDAS_CALIB = ROOT / "data" / "midas_calib"
MIDAS_VAL = ROOT / "data" / "midas_val"


def main():
    ap = argparse.ArgumentParser(description="Prepare MiDaS calibration and validation images.")
    ap.add_argument("--calib-count", type=int, default=300, help="Number of calibration images (default: 300)")
    ap.add_argument("--val-count", type=int, default=100, help="Number of validation images (default: 100)")
    args = ap.parse_args()

    MIDAS_CALIB.mkdir(parents=True, exist_ok=True)
    MIDAS_VAL.mkdir(parents=True, exist_ok=True)

    sources = []
    if COCO_VAL.is_dir():
        sources = sorted(COCO_VAL.glob("*.jpg"))
    elif CALIB_SRC.is_dir():
        sources = sorted(CALIB_SRC.glob("*.jpg"))

    if not sources:
        raise SystemExit("No candidate images found in data/coco/val2017 or data/calib.")

    total_needed = args.calib_count + args.val_count
    selected = sources[:total_needed]

    calib_files = selected[:args.calib_count]
    val_files = selected[args.calib_count:args.calib_count + args.val_count]

    print(f"Preparing {len(calib_files)} calibration images in {MIDAS_CALIB}...")
    for src in calib_files:
        dst = MIDAS_CALIB / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    print(f"Preparing {len(val_files)} evaluation images in {MIDAS_VAL}...")
    for src in val_files:
        dst = MIDAS_VAL / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
