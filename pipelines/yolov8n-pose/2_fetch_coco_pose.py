"""
YOLO-pose step 2: extract COCO's keypoint annotations for val2017.

Images, the calibration subset and instances_val2017.json are already fetched
by pipelines/yolov8n/2_fetch_coco.py -- pose calibration reuses data/coco_calib
as-is (calibration never looks at labels). This step only pulls the one file
that pipeline didn't: person_keypoints_val2017.json, needed for OKS mAP.

    python pipelines/yolov8n-pose/2_fetch_coco_pose.py

If data/coco/annotations.zip is missing, run pipelines/yolov8n/2_fetch_coco.py
first (or --n-calib 0 there to skip the calibration copy step).
"""
import argparse
import os
import zipfile

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import DATA

MEMBER = "annotations/person_keypoints_val2017.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-dir", default=str(DATA / "coco"))
    args = ap.parse_args()

    anz = os.path.join(args.coco_dir, "annotations.zip")
    dst = os.path.join(args.coco_dir, MEMBER)
    if os.path.isfile(dst):
        print(f"have {dst}")
        return
    if not os.path.isfile(anz):
        raise SystemExit(f"{anz} missing -- run pipelines/yolov8n/2_fetch_coco.py first "
                         "(it downloads the same zip this file lives in)")
    print(f"extracting {MEMBER} from {anz}...")
    with zipfile.ZipFile(anz) as z:
        z.extract(MEMBER, args.coco_dir)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
