"""
YOLO-World v2 step 5: COCO mAP on val2017.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'

    python pipelines/yolow/5_eval_map.py --model models/yolov8s-worldv2.onnx --ep cpu
    python pipelines/yolow/5_eval_map.py --model models/yolov8s-worldv2_cut_xint8.onnx --ep npu --fresh

Evaluation settings follow ultralytics/COCO convention:
conf 0.001 and per-class NMS at IoU 0.7, keeping up to 300 boxes.
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

import npu.yolow as yw
from npu.paths import DATA, RESULTS, yolow_cache_key
from npu.session import build_session, clear_cache


def build_forward(sess, conf, imgsz, txt_feats):
    """-> (run_fn, decode_fn, description). Pure sess.run is timed alone as 'infer'."""
    inp = sess.get_inputs()[0].name
    n_out = len(sess.get_outputs())
    if n_out == len(yw.HEAD_OUTS):
        order = yw.head_order(sess, imgsz)
        return (lambda x: sess.run(None, {inp: x}),
                lambda r: yw.decode_yolow([r[i] for i in order], txt_feats, imgsz=imgsz, conf_thres=conf),
                f"head-cut ({n_out} outputs), contrastive + numpy decode")
    if n_out == 1:
        return (lambda x: sess.run(None, {inp: x}),
                lambda r: r[0],
                "full graph (1 output), ONNX decode")
    raise SystemExit(f"don't know what to do with {n_out} outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ep", choices=["cpu", "dml", "npu"], default="npu")
    ap.add_argument("--images", default=str(DATA / "coco" / "val2017"))
    ap.add_argument("--ann", default=str(DATA / "coco" / "annotations" /
                                         "instances_val2017.json"))
    ap.add_argument("--n", type=int, default=0, help="0 = all 5000")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--agnostic", action="store_true")
    ap.add_argument("--txt-feats", default=None)
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
            raise SystemExit(f"missing {p} -- run pipelines/yolow/2_fetch_coco.py")

    stem = os.path.splitext(os.path.basename(args.model))[0]
    cache_key = args.cache_key or yolow_cache_key(args.model)
    if args.fresh:
        clear_cache(cache_key)

    coco = COCO(args.ann)
    img_ids = sorted(coco.getImgIds())
    if args.n:
        img_ids = img_ids[: args.n]

    sess = build_session(args.model, args.ep, cache_key, args.xclbin,
                         log_severity=args.log)
    imgsz = yw.input_size(sess.get_inputs()[0].shape, args.model)
    txt_feats = np.load(args.txt_feats) if args.txt_feats else yw.load_coco_txt_feats()

    run_fn, decode_fn, desc = build_forward(sess, args.conf, imgsz, txt_feats)
    print(f"model: {desc}, input {imgsz}x{imgsz}")
    print(f"eval : {len(img_ids)} images, conf {args.conf}, iou {args.iou}, "
          f"max_det {args.max_det}, "
          f"{'class-agnostic' if args.agnostic else 'per-class'} NMS")

    dets, times = [], []
    t_start = time.perf_counter()
    for k, iid in enumerate(img_ids):
        info = coco.loadImgs(iid)[0]
        img = cv2.imread(os.path.join(args.images, info["file_name"]))
        if img is None:
            print(f"  skip unreadable {info['file_name']}")
            continue
        x, pad, scale = yw.letterbox(img, imgsz)
        t0 = time.perf_counter()
        raw = run_fn(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

        preds = decode_fn(raw)
        boxes = yw.postprocess(preds, pad, scale,
                               conf_thres=args.conf, iou_thres=args.iou,
                               agnostic=args.agnostic, max_det=args.max_det)
        for x0, y0, w, h, s, c in boxes:
            dets.append({
                "image_id": iid,
                "category_id": yw.COCO_IDS[c],
                "bbox": [round(float(x0), 2), round(float(y0), 2),
                         round(float(w), 2), round(float(h), 2)],
                "score": round(float(s), 5),
            })
        if (k + 1) % 500 == 0 or (k + 1) == len(img_ids):
            elapsed = time.perf_counter() - t_start
            cur_ms = np.mean(times[-500:]) if len(times) >= 500 else np.mean(times)
            print(f"  [{k + 1:4d}/{len(img_ids)}]  infer {cur_ms:5.2f} ms  "
                  f"elapsed {elapsed:5.1f}s")

    ts = np.array(times)
    print(f"\nTiming ({len(times)} images):")
    print(f"  mean   : {ts.mean():.2f} ms  ({1000 / ts.mean():.1f} fps)")
    print(f"  median : {np.median(ts):.2f} ms")
    print(f"  p95    : {np.percentile(ts, 95):.2f} ms")

    if not dets:
        print("no detections above threshold")
        return

    if args.dets:
        dets_path = args.dets
    else:
        dets_path = str(RESULTS / f"eval_{stem}_{args.ep}_dets.json")
    os.makedirs(os.path.dirname(dets_path) or ".", exist_ok=True)
    with open(dets_path, "w") as f:
        json.dump(dets, f)
    print(f"wrote {len(dets)} detections to {dets_path}")

    coco_dt = coco.loadRes(dets_path)
    ev = COCOeval(coco, coco_dt, "bbox")
    if args.n:
        ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()


if __name__ == "__main__":
    main()
