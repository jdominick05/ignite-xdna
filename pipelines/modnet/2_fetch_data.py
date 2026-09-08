"""
Step 2: Prepare portrait calibration and evaluation images for MODNet.

Filters prominent person images from existing COCO annotations, copying or
linking them to data/modnet_calib/ and data/modnet_val/.

Run in resnet_env:
    python pipelines/modnet/2_fetch_data.py --count 200
"""

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN_PATH = ROOT / "data" / "coco" / "annotations" / "person_keypoints_val2017.json"
IMG_DIR = ROOT / "data" / "coco" / "val2017"
CALIB_DIR = ROOT / "data" / "modnet_calib"
VAL_DIR = ROOT / "data" / "modnet_val"


def main():
    ap = argparse.ArgumentParser(description="Prepare MODNet calibration and evaluation data.")
    ap.add_argument("--count", type=int, default=200, help="Number of calibration images (default: 200)")
    ap.add_argument("--val-count", type=int, default=50, help="Number of validation images (default: 50)")
    args = ap.parse_args()

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    if not ANN_PATH.exists():
        print(f"[2_fetch_data] Annotation file not found: {ANN_PATH}")
        print(f"[2_fetch_data] Falling back to data/coco_calib...")
        src_images = list((ROOT / "data" / "coco_calib").glob("*.jpg"))[:args.count + args.val_count]
        person_img_files = src_images
    else:
        print(f"[2_fetch_data] Reading annotations from {ANN_PATH}...")
        with open(ANN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Collect image IDs with at least 1 reasonably sized person annotation
        img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
        person_img_ids = set()
        for ann in data["annotations"]:
            if ann.get("category_id") == 1 and ann.get("area", 0) > 10000:
                person_img_ids.add(ann["image_id"])

        person_img_files = [IMG_DIR / img_id_to_file[iid] for iid in person_img_ids if (IMG_DIR / img_id_to_file[iid]).exists()]
        print(f"[2_fetch_data] Found {len(person_img_files)} portrait candidate images in COCO val2017.")

    calib_files = person_img_files[:args.count]
    val_files = person_img_files[args.count:args.count + args.val_count]

    print(f"[2_fetch_data] Copying {len(calib_files)} images to {CALIB_DIR}...")
    for p in calib_files:
        dst = CALIB_DIR / p.name
        if not dst.exists():
            shutil.copy2(p, dst)

    print(f"[2_fetch_data] Copying {len(val_files)} images to {VAL_DIR}...")
    for p in val_files:
        dst = VAL_DIR / p.name
        if not dst.exists():
            shutil.copy2(p, dst)

    print(f"[2_fetch_data] Complete: {len(calib_files)} calib, {len(val_files)} val.")


if __name__ == "__main__":
    main()
