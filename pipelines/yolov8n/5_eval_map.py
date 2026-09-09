"""
YOLO step 5: COCO mAP on val2017, so "more accurate" is a number.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'

    python pipelines/yolov8n/5_eval_map.py --model models/yolov8n.onnx --ep cpu
    python pipelines/yolov8n/5_eval_map.py --model models/yolov8n_cut_xint8.onnx --ep npu

Or ./scripts/yolo-eval.sh, which runs the whole comparison table.

Handles both graph shapes, like 4_detect.py: 1 output = full graph decoding in
ONNX, 6 outputs = head-cut with the numpy decode.

Evaluation settings follow ultralytics/COCO convention, NOT the demo defaults:
conf 0.001 (not 0.25) and per-class NMS at IoU 0.7, keeping up to 300 boxes.
Evaluating at conf 0.25 understates mAP badly, because average precision is
computed over the whole precision/recall curve and a high threshold simply
deletes the low-confidence tail that the metric wants to integrate over.

Writes results/dets_<model>_<ep>.json (COCO detection format), which is a
durable artifact: tools can re-score it without re-running inference.
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

import npu.yolo as yc
from npu.paths import DATA, RESULTS, YOLO_CACHE_KEY, YOLO_CUT_CACHE_KEY
from npu.session import build_session, clear_cache
from npu.yolo_decode import decode_heads, head_order


def build_forward(sess, conf, imgsz):
    """-> (run_fn, decode_fn, description). Same dispatch as 4_detect.py.

    Split so a caller can time run_fn (pure sess.run) separately from
    decode_fn (DFL/anchor decode for a cut model, a numpy cost that scales
    with how many candidates survive `conf` -- folding it into "inference"
    makes a cut model's latency depend on the eval threshold instead of the
    EP, and at this script's conf 0.001 that cost is not small: the NPU cut
    model's mean moved from 8.94ms at demo conf 0.25 to 15.01ms here on a
    single image, before even reaching the 5000-image average."""
    inp = sess.get_inputs()[0].name
    n_out = len(sess.get_outputs())
    if n_out == len(yc.HEAD_OUTS):
        order = head_order(sess, imgsz)
        return (lambda x: sess.run(None, {inp: x}),
                lambda r: decode_heads([r[i] for i in order], imgsz=imgsz, conf_thres=conf),
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
    ap.add_argument("--progress-every", type=int, default=500,
                    help="print progress every N images (default 500). Lower it "
                         "when a run is being watched for a mid-run hardware "
                         "hang -- it sets how precisely a crash can be located")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--agnostic", action="store_true",
                    help="class-agnostic NMS (worse mAP; here to measure the cost)")
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
            raise SystemExit(f"missing {p} -- run pipelines/yolov8n/2_fetch_coco.py")

    stem = os.path.splitext(os.path.basename(args.model))[0]
    cache_key = args.cache_key or (
        YOLO_CUT_CACHE_KEY if "cut" in stem.lower() else YOLO_CACHE_KEY)
    if args.fresh:
        clear_cache(cache_key)

    coco = COCO(args.ann)
    img_ids = sorted(coco.getImgIds())
    if args.n:
        img_ids = img_ids[: args.n]

    sess = build_session(args.model, args.ep, cache_key, args.xclbin,
                         log_severity=args.log)
    # Letterbox size comes from the model, not from a flag -- see 4_detect.py.
    imgsz = yc.input_size(sess.get_inputs()[0].shape, args.model)
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
            print(f"  skip unreadable {info['file_name']}", flush=True)
            continue
        x, pad, scale = yc.letterbox(img, imgsz)
        t0 = time.perf_counter()
        raw = run_fn(x)
        times.append(time.perf_counter() - t0)
        out = decode_fn(raw)
        for x0, y0, w, h, s, c in yc.postprocess(out, pad, scale, args.conf,
                                                 args.iou, args.agnostic,
                                                 args.max_det):
            dets.append({"image_id": iid,
                         "category_id": yc.COCO_IDS[c],
                         "bbox": [round(float(x0), 2), round(float(y0), 2),
                                  round(float(w), 2), round(float(h), 2)],
                         "score": round(float(s), 5)})
        # flush=True is load-bearing, not tidiness. This runs under run_logged's
        # tee, so stdout is a pipe and Python block-buffers it -- and a run that
        # dies on a hardware DPU timeout never flushes, taking the last progress
        # lines with it. That is exactly the run whose position you need. It is
        # also why --progress-every exists: at the default 500 a crash localises
        # only to a 500-image window, which is too coarse to say which image or
        # subgraph was in flight.
        if (k + 1) % args.progress_every == 0:
            print(f"  {k + 1}/{len(img_ids)}  ({len(dets)} detections so far)",
                  flush=True)

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
