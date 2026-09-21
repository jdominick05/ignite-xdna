"""Score the dense models against real COCO labels, rather than against each other.

Everything measured for these two families so far is AGREEMENT - with the quantized CPU reference,
or with the FP32 model. `pipelines/bisenetv2/5_eval.py` is the same: its `--ref` is
`bisenetv2_fp32.onnx`, so the mIoU it prints is mIoU against the float model's own output. None of
it answers "which stack is closer to the truth", because none of it has ever seen a label.

This does, using the COCO instance annotations already on disk. Two metrics, one per family:

* **bisenetv2** predicts 19 Cityscapes classes; COCO annotates 80 "thing" classes. Only the overlap
  can be scored, so a pixel is labelled only where a mapped class has an annotation and every other
  pixel is IGNORED. Intersection and union are both taken over labelled pixels, so predicting the
  wrong mapped class is punished but predicting anything over unlabelled sky or road is neither
  rewarded nor punished. Two caveats travel with every number: this is a Cityscapes-trained model on
  COCO photographs, and only 7 of its 19 classes are scoreable, so ABSOLUTE mIoU is low for every
  arm including FP32. The comparison between arms is the result; the absolute value is not.
* **modnet_cut** produces an alpha matte, thresholded at 0.5 to a person mask. COCO annotates every
  person instance, so here the ground truth covers the WHOLE image and binary IoU is honest.
  Caveat: COCO masks are polygons, so this scores gross person segmentation and says nothing about
  the hair-level detail a matting model exists for.

Crowd regions (`iscrowd`) are ignored in both, because a crowd RLE marks "people somewhere in here"
and cannot fairly be scored per pixel.

Run each backend as its own process through the research wrapper, and interleave them:

    ./scripts/research-lowlevel.sh --log results/dense/acc_<family>_<backend>_<tag>.log \\
        --checks-only --npu -- python benchmarks/dense_accuracy.py \\
        --family bisenetv2 --backend ignite --container build/bisenetv2_dense.ignite
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "Ignition" / "src")]

import cv2
import numpy as np
import onnxruntime as ort
from pycocotools import mask as coco_mask

FAMILIES = {
    "bisenetv2": ("models/bisenetv2_fp32_xint8.onnx", "models/bisenetv2_fp32.onnx", "segment"),
    "modnet_cut": ("models/modnet/modnet_cut_xint8_calibfix.onnx",
                   "models/modnet/modnet_cut_fp32.onnx", "matte"),
}
ANNOTATIONS = "data/coco/annotations/instances_val2017.json"
IMAGE_DIR = "data/coco/val2017"
IGNORE = 255

# Cityscapes class index -> COCO category name, for classes that genuinely correspond. Cityscapes
# "rider" has no COCO counterpart and is folded into person; "traffic sign" is nearest to stop sign.
CITY_TO_COCO = {
    6: "traffic light", 7: "stop sign", 11: "person", 12: "person", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(tag, **payload) -> None:
    print(tag, json.dumps(payload, sort_keys=True), flush=True)


def preprocess(family, image, shape):
    from npu import bisenetv2, modnet
    if len(shape) != 4 or shape[0:2] != [1, 3] or shape[2] != shape[3]:
        raise ValueError(f"unsupported image shape {shape}")
    return (bisenetv2 if family == "bisenetv2" else modnet).preprocess(image, target_size=shape[2])


def postprocess(family, value, shape):
    from npu import bisenetv2, modnet
    return (bisenetv2.postprocess_mask(value, shape) if family == "bisenetv2"
            else modnet.postprocess_matte(value, shape))


def ort_session(path, optimize=True):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if not optimize:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def decode(ann, height, width):
    """One annotation's binary mask at the image's own resolution."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = coco_mask.frPyObjects(seg, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(seg["counts"], list):
        rle = coco_mask.frPyObjects(seg, height, width)
    else:
        rle = seg
    return coco_mask.decode(rle).astype(bool)


def ground_truth(family, anns, cats, height, width):
    """Label map for one image, or None when it carries nothing scoreable.

    bisenetv2: Cityscapes indices where a mapped class is annotated, IGNORE everywhere else.
    modnet_cut: 1 for person, 0 for not-person, IGNORE over crowd regions.
    """
    if family == "modnet_cut":
        gt = np.zeros((height, width), np.uint8)
        found = False
        for a in anns:
            if cats[a["category_id"]] != "person":
                continue
            m = decode(a, height, width)
            if a.get("iscrowd"):
                gt[m & (gt == 0)] = IGNORE
            else:
                gt[m] = 1
                found = True
        return gt if found else None

    coco_to_city = {}
    for city, name in CITY_TO_COCO.items():
        coco_to_city.setdefault(name, city)
    gt = np.full((height, width), IGNORE, np.uint8)
    found = False
    # Largest first, so a small object painted later wins its own pixels.
    for a in sorted(anns, key=lambda a: -float(a.get("area", 0.0))):
        name = cats[a["category_id"]]
        if name not in coco_to_city:
            continue
        m = decode(a, height, width)
        if a.get("iscrowd"):
            gt[m & (gt == IGNORE)] = IGNORE
            continue
        gt[m] = coco_to_city[name]
        found = True
    return gt if found else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--backend", choices=["ignite", "amd", "cpu", "fp32"], required=True)
    ap.add_argument("--container", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="0 is every scoreable image; anything else is a SLICE")
    ap.add_argument("--threshold", type=float, default=0.5, help="matte -> person mask, modnet_cut only")
    ap.add_argument("--progress", type=int, default=250)
    args = ap.parse_args()

    model, fp32, task = FAMILIES[args.family]
    coco = json.loads((ROOT / ANNOTATIONS).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    images = sorted(coco["images"], key=lambda i: i["id"])

    classes = sorted(set(CITY_TO_COCO)) if args.family == "bisenetv2" else [1]
    emit("IDENTITY", family=args.family, backend=args.backend, task=task,
         model_sha256=sha(ROOT / model), annotations_sha256=sha(ROOT / ANNOTATIONS),
         container_sha256=sha(args.container) if args.container else None,
         scored_classes=classes, threshold=args.threshold if args.family == "modnet_cut" else None,
         numpy=np.__version__, ort=ort.__version__, full_set=not args.limit)

    shape = ort_session(ROOT / (fp32 if args.backend == "fp32" else model)).get_inputs()[0].shape
    target = native = None
    if args.backend == "amd":
        from npu.session import build_session
        from npu.paths import BISENETV2_CACHE_KEY, modnet_cache_key
        key = BISENETV2_CACHE_KEY if args.family == "bisenetv2" else modnet_cache_key(model)
        target = build_session(ROOT / model, "npu", cache_key=key)
    elif args.backend == "ignite":
        from ignite_xdna.runtime.dense_session import DenseTensorSession
        native = DenseTensorSession(str(args.container))
    else:
        target = ort_session(ROOT / (fp32 if args.backend == "fp32" else model),
                             optimize=args.backend != "cpu")
    name = target.get_inputs()[0].name if target is not None else None

    inter = np.zeros(max(classes) + 1, np.int64)
    union = np.zeros(max(classes) + 1, np.int64)
    correct = total = scored = skipped = 0
    per_image = []
    t0 = time.perf_counter()
    try:
        for img in images:
            if args.limit and scored >= args.limit:
                break
            path = ROOT / IMAGE_DIR / img["file_name"]
            if not path.exists():
                skipped += 1
                continue
            anns = anns_by_img.get(img["id"], [])
            frame = cv2.imread(str(path))
            if frame is None:
                skipped += 1
                continue
            gt = ground_truth(args.family, anns, cats, frame.shape[0], frame.shape[1])
            if gt is None:
                skipped += 1
                continue

            x, original = preprocess(args.family, frame, shape)
            raw = native.run(x)[0] if native is not None else target.run(None, {name: x})[0]
            out = postprocess(args.family, raw, original)
            pred = (np.asarray(out) >= args.threshold).astype(np.uint8) if args.family == "modnet_cut" \
                else np.asarray(out).astype(np.uint8)

            valid = gt != IGNORE
            if not valid.any():
                skipped += 1
                continue
            correct += int((pred[valid] == gt[valid]).sum())
            total += int(valid.sum())
            img_i = img_u = 0
            for c in classes:
                p, g = (pred == c) & valid, (gt == c) & valid
                i, u = int((p & g).sum()), int((p | g).sum())
                inter[c] += i
                union[c] += u
                img_i += i
                img_u += u
            per_image.append(img_i / img_u if img_u else 0.0)
            scored += 1
            if args.progress and scored % args.progress == 0:
                emit("PROGRESS", scored=scored, skipped=skipped,
                     running_miou=float(np.mean([inter[c] / union[c] for c in classes if union[c]])),
                     seconds=round(time.perf_counter() - t0, 1))
    finally:
        close = getattr(native, "close", None)
        if callable(close):
            close()

    ious = {int(c): (float(inter[c] / union[c]) if union[c] else None) for c in classes}
    present = [v for v in ious.values() if v is not None]
    emit("ACCURACY", family=args.family, backend=args.backend, images_scored=scored,
         images_skipped=skipped, full_set=not args.limit,
         miou=float(np.mean(present)) if present else None,
         pixel_accuracy=float(correct / total) if total else None,
         mean_per_image_iou=float(np.mean(per_image)) if per_image else None,
         labelled_pixels=int(total), per_class_iou=ious,
         seconds=round(time.perf_counter() - t0, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
