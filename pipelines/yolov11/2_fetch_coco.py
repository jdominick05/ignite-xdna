"""
YOLOv11 step 2: download or verify COCO val2017 and calibration subset.

Delegates to or mirrors the shared COCO dataset at data/coco and data/coco_calib.

    python pipelines/yolov11/2_fetch_coco.py --n-calib 300
"""
import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import DATA

IMG_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def fetch(url, dest):
    if os.path.exists(dest):
        print(f"have {dest}")
        return
    print(f"downloading {url}")

    def hook(b, bs, total):
        if total > 0 and b % 200 == 0:
            print(f"  {100 * b * bs / total:5.1f}%", end="\r")

    urllib.request.urlretrieve(url, dest, hook)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "coco"))
    ap.add_argument("--n-calib", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    imz = os.path.join(args.out, "val2017.zip")
    anz = os.path.join(args.out, "annotations.zip")
    fetch(IMG_URL, imz)
    fetch(ANN_URL, anz)

    if not os.path.isdir(os.path.join(args.out, "val2017")):
        print("extracting images...")
        with zipfile.ZipFile(imz) as z:
            z.extractall(args.out)
    if not os.path.isfile(os.path.join(args.out, "annotations", "instances_val2017.json")):
        print("extracting annotations...")
        with zipfile.ZipFile(anz) as z:
            z.extractall(args.out, members=[m for m in z.namelist()
                                            if m.endswith("instances_val2017.json")])

    imgs = sorted(os.listdir(os.path.join(args.out, "val2017")))
    stride = max(1, len(imgs) // args.n_calib)
    calib_dir = os.path.join(os.path.dirname(args.out), "coco_calib")
    os.makedirs(calib_dir, exist_ok=True)
    picked = imgs[::stride][: args.n_calib]
    for f in picked:
        shutil.copy(os.path.join(args.out, "val2017", f), os.path.join(calib_dir, f))
    print(f"{len(imgs)} val images; {len(picked)} copied to {calib_dir} for calibration")


if __name__ == "__main__":
    main()
