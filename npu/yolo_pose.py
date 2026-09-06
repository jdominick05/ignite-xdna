"""
Letterboxing, decoding and drawing for the YOLOv8n-pose pipeline. Quark-free,
like npu.yolo, which this borrows letterbox() from directly -- calibration and
inference must use byte-identical preprocessing, and duplicating that
function would risk the two copies drifting apart.

Pose differs from detect (npu.yolo) in three ways that matter here:
  * cls has 1 channel (person only), not 80.
  * a third branch, cv4, emits 51 raw keypoint channels per anchor
    (17 COCO keypoints x (x, y, visibility)) alongside the existing box (cv2)
    and cls (cv3) branches.
  * the graph output is (1, 56, N) = 4 box + 1 cls + 51 kpt, not (1, 84, N).
"""
import cv2
import numpy as np

from .yolo import letterbox  # noqa: F401  (re-exported for pipeline scripts)

INPUT_SIZE = 640

# Detect-head geometry, mirrored from npu.yolo.
STRIDES = (8, 16, 32)
REG_MAX = 16       # DFL bins per box side
NUM_CLASSES = 1     # yolov8-pose has one class: "person"
NUM_KPTS = 17        # COCO keypoints
KPT_DIM = 3         # x, y, visibility per keypoint

# The nine raw conv outputs of the yolov8-pose head, in canonical order: box
# distributions (64ch) at strides 8/16/32, class logit (1ch) at the same three
# strides, then raw keypoints (51ch) at the same three strides.
# pipelines/yolov8n-pose/1b_cut_head.py makes these the graph outputs, and
# npu.yolo_pose_decode expects them in exactly this order.
BOX_OUTS = [f"/model.22/cv2.{i}/cv2.{i}.2/Conv_output_0" for i in range(3)]
CLS_OUTS = [f"/model.22/cv3.{i}/cv3.{i}.2/Conv_output_0" for i in range(3)]
KPT_OUTS = [f"/model.22/cv4.{i}/cv4.{i}.2/Conv_output_0" for i in range(3)]
HEAD_OUTS = BOX_OUTS + CLS_OUTS + KPT_OUTS


def head_shapes(imgsz=INPUT_SIZE, strides=STRIDES, nc=NUM_CLASSES,
                nk=NUM_KPTS, kdim=KPT_DIM):
    """Expected shapes of HEAD_OUTS for a square input of `imgsz`. See
    npu.yolo.head_shapes -- same derivation, extended with the kpt branch."""
    grids = [imgsz // s for s in strides]
    return ([(1, 4 * REG_MAX, g, g) for g in grids]
            + [(1, nc, g, g) for g in grids]
            + [(1, nk * kdim, g, g) for g in grids])


HEAD_SHAPES = head_shapes()

# COCO 17-point skeleton, 0-indexed, matching COCO_KEYPOINTS order below.
# Same edge list ultralytics draws, converted from its 1-indexed form.
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def postprocess(output, pad, scale, conf_thres=0.25, iou_thres=0.5, max_det=300):
    """output: (1, 56, N) from npu.yolo_pose_decode.decode_heads.

    Returns list of (x0, y0, w, h, score, kpts) in ORIGINAL image pixels,
    where kpts is an (17, 3) array of (x, y, visibility), x/y also mapped
    back to original image pixels.

    One class only, so NMS is class-agnostic -- there is no "agnostic" split
    like npu.yolo.postprocess, because agnostic and per-class NMS are the
    same thing when there is exactly one class.
    """
    out = output[0].T                      # (N, 56)
    boxes = out[:, :4].copy()              # cx, cy, w, h in letterboxed pixels
    scores = out[:, 4]                     # already sigmoid'd, single class
    kpts = out[:, 5:].reshape(-1, NUM_KPTS, KPT_DIM).copy()

    keep = scores >= conf_thres
    boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]
    if len(boxes) == 0:
        return []

    if len(scores) > 10 * max_det:
        top = np.argpartition(-scores, 10 * max_det)[:10 * max_det]
        boxes, scores, kpts = boxes[top], scores[top], kpts[top]

    boxes[:, 0] -= pad[1]
    boxes[:, 1] -= pad[0]
    boxes /= scale
    kpts[:, :, 0] -= pad[1]
    kpts[:, :, 1] -= pad[0]
    kpts[:, :, :2] /= scale

    xywh = np.stack([boxes[:, 0] - boxes[:, 2] / 2,
                     boxes[:, 1] - boxes[:, 3] / 2,
                     boxes[:, 2], boxes[:, 3]], axis=1)

    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf_thres, iou_thres)
    if len(idx) == 0:
        return []
    idx = np.array(idx).reshape(-1)
    if len(idx) > max_det:
        idx = idx[np.argsort(-scores[idx])[:max_det]]

    return [(*xywh[i], float(scores[i]), kpts[i]) for i in idx]


def draw(img_bgr, dets, kpt_conf_thres=0.5):
    for x, y, w, h, s, kpts in dets:
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(img_bgr, p1, p2, (0, 255, 0), 2)
        cv2.putText(img_bgr, f"person {s:.2f}", (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        visible = kpts[:, 2] >= kpt_conf_thres
        for i, (px, py, pv) in enumerate(kpts):
            if visible[i]:
                cv2.circle(img_bgr, (int(px), int(py)), 3, (0, 0, 255), -1)
        for a, b in SKELETON:
            if visible[a] and visible[b]:
                pa = (int(kpts[a, 0]), int(kpts[a, 1]))
                pb = (int(kpts[b, 0]), int(kpts[b, 1]))
                cv2.line(img_bgr, pa, pb, (255, 128, 0), 2)
    return img_bgr
