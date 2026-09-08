"""
YOLOv6n constants and head-cut node names. Quark-free, like the rest of npu/.

Structurally close to npu.yolo (same 640/8-16-32/80-class anchor-free
detect head), with one real difference: yolov6n ships use_dfl=False,
reg_max=0 (configs/yolov6n.py), so the box head is 4 raw ltrb channels per
anchor straight out of a Conv -- no DFL weighted-sum decode, no Softmax
anywhere in the exported graph. letterbox/postprocess/draw are unchanged
from npu.yolo and re-exported here rather than duplicated -- postprocess()
only assumes (1, 4+nc, N) with cx,cy,w,h pixels + sigmoid'd scores, which is
exactly what npu.yolov6_decode.decode_heads produces.
"""
from .yolo import (INPUT_SIZE, STRIDES, NUM_CLASSES, COCO_CLASSES, COCO_IDS,  # noqa: F401
                    letterbox, postprocess, draw, input_size)

REG_OUTS = [f"/detect/reg_preds.{i}/Conv_output_0" for i in range(3)]
CLS_OUTS = [f"/detect/cls_preds.{i}/Conv_output_0" for i in range(3)]
HEAD_OUTS = REG_OUTS + CLS_OUTS


def head_shapes(imgsz=INPUT_SIZE, strides=STRIDES, nc=NUM_CLASSES):
    """Expected shapes of HEAD_OUTS for a square input of `imgsz`, in
    REG_OUTS-then-CLS_OUTS order -- mirrors npu.yolo.head_shapes but with 4
    raw box channels per level instead of 4*REG_MAX (no DFL bins here)."""
    shapes = []
    for s in strides:
        g = imgsz // s
        shapes.append((1, 4, g, g))
    for s in strides:
        g = imgsz // s
        shapes.append((1, nc, g, g))
    return shapes
