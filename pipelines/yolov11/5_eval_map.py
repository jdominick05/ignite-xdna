"""
YOLOv11 step 5: COCO mAP on val2017.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'

    python pipelines/yolov11/5_eval_map.py --model models/yolo11n.onnx --ep cpu
    python pipelines/yolov11/5_eval_map.py --model models/yolo11n_cut_xint8.onnx --ep npu --fresh

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

import npu.yolov11 as y11
from npu.paths import DATA, RESULTS, yolo11_cache_key
from npu.session import build_session, clear_cache


def build_forward(sess, conf, imgsz):
    """-> (run_fn, decode_fn, description). Pure sess.run is timed alone as 'infer'."""
    inp = sess.get_inputs()[0].name
    n_out = len(sess.get_outputs())
    if n_out == len(y11.HEAD_OUTS):
        order = y11.head_order(sess, imgsz)
        return (lambda x: sess.run(None, {inp: x}),
                lambda r: y11.decode_heads([r[i] for i in order], imgsz=imgsz, conf_thres=conf),
                f"head-cut ({n_out} outputs), numpy decode")
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
            raise SystemExit(f"missing {p} -- run pipelines/yolov11/2_fetch_coco.py")

    stem = os.path.splitext(os.path.basename(args.model))[0]
    cache_key = args.cache_key or yolo11_cache_key(args.model)
    if args.fresh:
        clear_cache(cache_key)

    coco = COCO(args.ann)
    img_ids = sorted(coco.getImgIds())
    if args.n:
        img_ids = img_ids[: args.n]

    sess = build_session(args.model, args.ep, cache_key, args.xclbin,
                         log_severity=args.log)
    imgsz = y11.input_size(sess.get_inputs()[0].shape, args.model)
    run_fn, decode_fn, desc = build_forward(sess, args.conf, imgsz)
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
        x, pad, scale = y11.letterbox(img, imgsz)
        t0 = time.perf_counter()
        raw = run_fn(x)
        times.append(time.perf_counter() - t0)
        out = decode_fn(raw)
        for x0, y0, w, h, s, c in y11.postprocess(out, pad, scale, args.conf,
                                                 args.iou, args.agnostic,
                                                 args.max_det):
            dets.append({"image_id": iid,
                         "category_id": y11.COCO_IDS[c],
                         "bbox": [round(float(x0), 2), round(float(y0), 2),
                                  round(float(w), 2), round(float(h), 2)],
                         "score": round(float(s), 5)})
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(img_ids)}  ({len(dets)} detections so far)")

    wall = time.perf_counter() - t_start
    times = np.array(times) * 1000
    print(f"\n{len(dets)} detections over {len(times)} images in {wall:.0f}s")
    print(f"inference  mean {times.mean():.2f} ms   median {np.median(times):.2f} ms")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = args.dets or str(RESULTS / f"dets_{stem}_{args.ep}.json")
    with open(out_json, "w") as f:
        json.dump(dets, f)
    print(f"wrote {out_json}")

    if not dets:
        raise SystemExit("no detections at all -- something is broken upstream")

    ev = COCOeval(coco, coco.loadRes(out_json), "bbox")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()

    print(f"\n=== {stem} | {args.ep.upper()} | {imgsz}px | {len(img_ids)} images ===")
    print(f"mAP@50-95  {ev.stats[0] * 100:.2f}")
    print(f"mAP@50     {ev.stats[1] * 100:.2f}")
    print(f"mAP small / medium / large   {ev.stats[3] * 100:.2f} / "
          f"{ev.stats[4] * 100:.2f} / {ev.stats[5] * 100:.2f}")
    print(f"latency    {times.mean():.2f} ms")


if __name__ == "__main__":
    main()
