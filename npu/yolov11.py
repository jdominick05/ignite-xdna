"""
YOLOv11 constants and head-cut node names. Quark-free, like the rest of npu/.

Structurally identical detect-head geometry to YOLOv8 (640 input, 8-16-32 strides,
REG_MAX=16 DFL distribution bins -> 64 box channels, and 80 COCO class logits),
but the detect head is moved to `/model.23/` (from `/model.22/`) and the classification
branch uses decoupled Depthwise Separable Convolutions (DWConv).

The backbone also introduces the C2PSA spatial self-attention block at layer 10
(4D MatMul + Softmax + Transpose).
"""
import numpy as np

from .yolo import (
    COCO_CLASSES,
    COCO_IDS,
    INPUT_SIZE,
    NUM_CLASSES,
    REG_MAX,
    STRIDES,
    draw,
    head_shapes,
    input_size,
    letterbox,
    postprocess,
)
from .yolo_decode import decode_heads

__all__ = [
    "BOX_OUTS",
    "CLS_OUTS",
    "COCO_CLASSES",
    "COCO_IDS",
    "HEAD_OUTS",
    "INPUT_SIZE",
    "NUM_CLASSES",
    "REG_MAX",
    "STRIDES",
    "decode_heads",
    "draw",
    "head_order",
    "head_shapes",
    "input_size",
    "letterbox",
    "postprocess",
]

# The six raw detection convs of the YOLOv11 head, in canonical order:
# box distributions (64ch) at strides 8/16/32, then class logits (80ch) at the
# same three strides. pipelines/yolov11/1b_cut_head.py makes these the graph
# outputs, and npu.yolo_decode / npu.yolov11 expects them in exactly this order.
BOX_OUTS = [f"/model.23/cv2.{i}/cv2.{i}.2/Conv_output_0" for i in range(3)]
CLS_OUTS = [f"/model.23/cv3.{i}/cv3.{i}.2/Conv_output_0" for i in range(3)]
HEAD_OUTS = BOX_OUTS + CLS_OUTS


def head_order(session, imgsz=INPUT_SIZE):
    """Indices that reorder a session's outputs into HEAD_OUTS order.

    Quantization renames tensors (e.g. Conv_output_0 ->
    Conv_output_0_DequantizeLinear_Output), so fall back to matching by shape.
    """
    expected = head_shapes(imgsz)
    outs = session.get_outputs()
    shapes = [tuple(o.shape) for o in outs]
    if sorted(shapes) != sorted(expected):
        raise SystemExit(
            f"unexpected head output shapes: {shapes}\n"
            f"expected (in any order, for imgsz={imgsz}) {expected}"
        )
    if [o.name for o in outs] == HEAD_OUTS:
        return list(range(len(HEAD_OUTS)))
    return [shapes.index(s) for s in expected]
