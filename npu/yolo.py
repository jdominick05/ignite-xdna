"""
Letterboxing, decoding and drawing for the YOLOv8n pipeline. Quark-free.

Session construction lives in npu.session; the COCO class list and the
YOLO-index-to-COCO-category-id map live here because both the detector and a
future mAP evaluation need them.
"""
import cv2
import numpy as np

INPUT_SIZE = 640

# Detect-head geometry. These live here rather than in npu.yolo_decode because
# head_shapes() needs them and yolo_decode imports from this module, not the
# other way round; yolo_decode re-exports them so existing imports still work.
STRIDES = (8, 16, 32)
REG_MAX = 16        # DFL bins per box side
NUM_CLASSES = 80    # COCO

# The six raw detection convs of the yolov8 head, in canonical order:
# box distributions (64ch) at strides 8/16/32, then class logits (80ch) at the
# same three strides. pipelines/yolov8n/1b_cut_head.py makes these the graph
# outputs, and npu.yolo_decode expects them in exactly this order.
BOX_OUTS = [f"/model.22/cv2.{i}/cv2.{i}.2/Conv_output_0" for i in range(3)]
CLS_OUTS = [f"/model.22/cv3.{i}/cv3.{i}.2/Conv_output_0" for i in range(3)]
HEAD_OUTS = BOX_OUTS + CLS_OUTS


def head_shapes(imgsz=INPUT_SIZE, strides=STRIDES, nc=NUM_CLASSES):
    """Expected shapes of HEAD_OUTS for a square input of `imgsz`.

    Derived, not hardcoded: the grids are imgsz/stride, so 640 gives 80/40/20
    and 512 gives 64/32/16. This is the ONLY thing in the repo that depended on
    the input resolution, which is why a resolution sweep needed it computed.
    Channel counts do not move with resolution -- 4*REG_MAX box bins and nc
    class logits per anchor regardless -- and they do not move with model width
    either, which is why every yolov8 variant cuts to the same six shapes.
    """
    grids = [imgsz // s for s in strides]
    return ([(1, 4 * REG_MAX, g, g) for g in grids]
            + [(1, nc, g, g) for g in grids])


HEAD_SHAPES = head_shapes()   # the 640 case, kept as a name for convenience


def input_size(dims, where="model"):
    """Validate an NCHW input shape and return its square side length.

    `dims` is whatever the source hands over: onnxruntime gives ints for a
    static export and strings ('batch') for a dynamic one, and onnx's dim_value
    gives 0 for a symbolic dim. Reject both rather than letting a symbolic
    dimension reach letterbox(), where it would fail much further downstream.
    """
    if len(dims) != 4:
        raise SystemExit(f"{where}: expected a 4D NCHW input, got shape {dims}")
    h, w = dims[2], dims[3]
    if not isinstance(h, int) or not isinstance(w, int) or h <= 0 or w <= 0:
        raise SystemExit(
            f"{where}: input spatial dims are not static ({dims}). Export with "
            "dynamic=False; a dynamic axis means the letterbox size is unknowable.")
    if h != w:
        raise SystemExit(f"{where}: input is {h}x{w}; this pipeline letterboxes "
                         "to a square, so a non-square model is not supported")
    if h % max(STRIDES) != 0:
        raise SystemExit(f"{where}: input {h} is not a multiple of {max(STRIDES)}; "
                         "the stride-32 head grid would not be an integer")
    return h


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# YOLO class index -> COCO category id (COCO ids have gaps)
COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50,
            51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
            76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def letterbox(img_bgr, size=INPUT_SIZE):
    """Resize keeping aspect, pad to size x size. Returns NCHW float32 RGB [0,1],
    plus (pad_top, pad_left) and scale for mapping boxes back."""
    h, w = img_bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(x), (top, left), scale


def postprocess(output, pad, scale, conf_thres=0.25, iou_thres=0.5,
                agnostic=False, max_det=300):
    """output: (1, 84, N) from the full-graph model, or from npu.yolo_decode.

    Returns list of (x0, y0, w, h, score, class_idx) in ORIGINAL image pixels.

    NMS is per-class by default, which is what ultralytics does and what COCO
    mAP assumes: class-agnostic NMS lets an overlapping person and chair
    suppress each other and silently costs real detections. Pass agnostic=True
    only if you deliberately want one box per region regardless of class.
    """
    out = output[0].T                     # (N, 84)
    boxes = out[:, :4].copy()             # cx, cy, w, h in letterboxed pixels
    scores_all = out[:, 4:]               # already sigmoid'd
    cls = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(cls)), cls]

    keep = scores >= conf_thres
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    if len(boxes) == 0:
        return []

    # Cheap guard for evaluation thresholds (conf ~0.001), where tens of
    # thousands of boxes would otherwise reach NMS.
    if len(scores) > 10 * max_det:
        top = np.argpartition(-scores, 10 * max_det)[:10 * max_det]
        boxes, scores, cls = boxes[top], scores[top], cls[top]

    boxes[:, 0] -= pad[1]
    boxes[:, 1] -= pad[0]
    boxes /= scale
    xywh = np.stack([boxes[:, 0] - boxes[:, 2] / 2,
                     boxes[:, 1] - boxes[:, 3] / 2,
                     boxes[:, 2], boxes[:, 3]], axis=1)

    bl, sl = xywh.tolist(), scores.tolist()
    if agnostic:
        idx = cv2.dnn.NMSBoxes(bl, sl, conf_thres, iou_thres)
    else:
        idx = cv2.dnn.NMSBoxesBatched(bl, sl, cls.tolist(), conf_thres, iou_thres)
    if len(idx) == 0:
        return []
    idx = np.array(idx).reshape(-1)
    if len(idx) > max_det:
        idx = idx[np.argsort(-scores[idx])[:max_det]]
    return [(*xywh[i], float(scores[i]), int(cls[i])) for i in idx]


def draw(img_bgr, dets):
    for x, y, w, h, s, c in dets:
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(img_bgr, p1, p2, (0, 255, 0), 2)
        label = f"{COCO_CLASSES[c]} {s:.2f}"
        cv2.putText(img_bgr, label, (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img_bgr
