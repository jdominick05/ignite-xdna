"""
YOLO-pose step 5: OKS keypoint mAP on val2017, so "more accurate" is a number.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'

    python pipelines/yolov8n-pose/5_eval_map.py --model models/yolov8n-pose_cut.onnx --ep cpu
    python pipelines/yolov8n-pose/5_eval_map.py --model models/yolov8n-pose_cut_xint8.onnx --ep npu

Head-cut only (see 4_pose.py for why the full graph isn't wired up here).

Evaluation settings follow ultralytics/COCO convention, NOT the demo defaults:
conf 0.001, class-agnostic NMS (there is only one class) at IoU 0.7, keeping up
to 300 boxes -- see pipelines/yolov8n/5_eval_map.py for why a high conf
threshold understates mAP.

pycocotools scores OKS (object keypoint similarity) from the per-keypoint
sigmas COCO defines for its 17-point skeleton, via iouType="keypoints" --
different metric from the detect pipeline's iouType="bbox", so this cannot
reuse that script's COCOeval call unmodified, only its structure. A
prediction's "keypoints" visibility flag is not scored by OKS itself (COCOeval
only reads x/y for matched, non-zero-visibility GT points) but pycocotools
still requires the field: pass 2 (visible) for every predicted point, never 0,
so a low-confidence keypoint isn't accidentally treated as absent.

Writes results/dets_kpts_<model>_<ep>.json (COCO keypoint-detection format).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

import npu.yolo_pose as yp
from npu.paths import DATA, RESULTS, YOLO_POSE_CUT_CACHE_KEY
from npu.session import build_session, clear_cache
from npu.yolo import input_size
from npu.yolo_pose_decode import decode_heads, head_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ep", choices=["cpu", "npu"], default="npu")
    ap.add_argument("--images", default=str(DATA / "coco" / "val2017"))
    ap.add_argument("--ann", default=str(DATA / "coco" / "annotations" /
                                         "person_keypoints_val2017.json"))
    ap.add_argument("--n", type=int, default=0, help="0 = all 5000")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--cache-key", default=None)
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--log", type=int, default=2)
    ap.add_argument("--dets", default=None, help="where to write detections json")
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    for p in (args.images, args.ann):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run pipelines/yolov8n-pose/2_fetch_coco_pose.py "
                             "(and pipelines/yolov8n/2_fetch_coco.py for the images)")

    stem = os.path.splitext(os.path.basename(args.model))[0]
    cache_key = args.cache_key or YOLO_POSE_CUT_CACHE_KEY
    if args.fresh:
        clear_cache(cache_key)

    coco = COCO(args.ann)
    img_ids = sorted(coco.getImgIds())
    if args.n:
        img_ids = img_ids[: args.n]

    sess = build_session(args.model, args.ep, cache_key, args.xclbin,
                         log_severity=args.log)
    inp = sess.get_inputs()[0].name
    imgsz = input_size(sess.get_inputs()[0].shape, args.model)
    n_out = len(sess.get_outputs())
    if n_out != len(yp.HEAD_OUTS):
        raise SystemExit(f"expected {len(yp.HEAD_OUTS)} head outputs, got {n_out}: "
                         f"{[o.name for o in sess.get_outputs()]}. Full-graph pose "
                         "models are not supported by this script.")
    order = head_order(sess, imgsz)
    print(f"model: head-cut ({n_out} outputs), numpy decode, input {imgsz}x{imgsz}")
    print(f"eval : {len(img_ids)} images, conf {args.conf}, iou {args.iou}, "
          f"max_det {args.max_det}, class-agnostic NMS (single class)")

    dets, times = [], []
    t_start = time.perf_counter()
    for k, iid in enumerate(img_ids):
        info = coco.loadImgs(iid)[0]
        img = cv2.imread(os.path.join(args.images, info["file_name"]))
        if img is None:
            print(f"  skip unreadable {info['file_name']}")
            continue
        x, pad, scale = yp.letterbox(img, imgsz)
        t0 = time.perf_counter()
        outs = sess.run(None, {inp: x})
        out = decode_heads([outs[i] for i in order], imgsz=imgsz, conf_thres=args.conf)
        times.append(time.perf_counter() - t0)
        for x0, y0, w, h, s, kpts in yp.postprocess(out, pad, scale, args.conf,
                                                    args.iou, args.max_det):
            flat = kpts.copy()
            flat[:, 2] = 2  # COCOeval scores keypoint OKS from x/y only; see docstring
            dets.append({"image_id": iid,
                         "category_id": 1,  # COCO "person"
                         "keypoints": [round(float(v), 2) for v in flat.reshape(-1)],
                         "score": round(float(s), 5)})
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(img_ids)}  ({len(dets)} detections so far)")

    wall = time.perf_counter() - t_start
    times = np.array(times) * 1000
    print(f"\n{len(dets)} detections over {len(times)} images in {wall:.0f}s")
    print(f"inference  mean {times.mean():.2f} ms   median {np.median(times):.2f} ms")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = args.dets or str(RESULTS / f"dets_kpts_{stem}_{args.ep}.json")
    with open(out_json, "w") as f:
        json.dump(dets, f)
    print(f"wrote {out_json}")

    if not dets:
        raise SystemExit("no detections at all -- something is broken upstream")

    ev = COCOeval(coco, coco.loadRes(out_json), "keypoints")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()

    print(f"\n=== {stem} | {args.ep.upper()} | {imgsz}px | {len(img_ids)} images ===")
    print(f"OKS mAP@50-95  {ev.stats[0] * 100:.2f}")
    print(f"OKS mAP@50     {ev.stats[1] * 100:.2f}")
    print(f"latency        {times.mean():.2f} ms")


if __name__ == "__main__":
    main()
