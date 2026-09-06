"""
Numpy reimplementation of the yolov8-pose detect-head tail that
pipelines/yolov8n-pose/1b_cut_head.py removes from the ONNX graph. Mirrors
npu.yolo_decode (box + cls decode is identical DFL/sigmoid math); adds the
keypoint branch ultralytics' Pose.kpts_decode does in torch:

    x = (raw_x * 2 + (anchor_x - 0.5)) * stride
    y = (raw_y * 2 + (anchor_y - 0.5)) * stride
    v = sigmoid(raw_v)

anchor_x/anchor_y here are the same grid centres (already +0.5) that box
decode uses, via npu.yolo_decode.anchors_and_strides -- so `anchor - 0.5`
recovers the raw grid coordinate ultralytics' formula wants, and the two
decode paths cannot drift out of sync with each other.
"""
import numpy as np

from .yolo_decode import _sigmoid, _softmax, anchors_and_strides
from .yolo_pose import HEAD_OUTS, INPUT_SIZE, KPT_DIM, NUM_KPTS, REG_MAX, STRIDES, head_shapes

__all__ = ["decode_heads", "head_order"]


def decode_heads(outs, imgsz=INPUT_SIZE, strides=STRIDES, conf_thres=None):
    """
    outs: nine arrays in npu.yolo_pose.HEAD_OUTS order -- box (1,64,g,g),
          cls (1,1,g,g), kpt (1,51,g,g), for g = imgsz/8, imgsz/16, imgsz/32.

    Returns (1, 56, N) = concat([xywh, cls prob, kpt x/y/vis x17]) in
    letterboxed pixels, drop-in for npu.yolo_pose.postprocess.
    """
    box_f, cls_f, kpt_f = outs[:3], outs[3:6], outs[6:]
    nc = cls_f[0].shape[1]

    box = np.concatenate([b.reshape(1, 4 * REG_MAX, -1) for b in box_f], 2)
    cls = np.concatenate([c.reshape(1, nc, -1) for c in cls_f], 2)
    kpt = np.concatenate([k.reshape(1, NUM_KPTS * KPT_DIM, -1) for k in kpt_f], 2)
    anc, st = anchors_and_strides(imgsz, strides)

    if conf_thres is not None:
        c = min(max(float(conf_thres), 1e-12), 1.0 - 1e-12)
        t = np.log(c / (1.0 - c))
        keep = np.flatnonzero(cls[0].max(0) > t)
        if keep.size == 0:
            return np.zeros((1, 4 + nc + NUM_KPTS * KPT_DIM, 0), np.float32)
        box, cls, kpt = box[:, :, keep], cls[:, :, keep], kpt[:, :, keep]
        anc, st = anc[:, :, keep], st[:, keep]

    box = box.astype(np.float32, copy=False)
    n = box.shape[2]

    d = _softmax(box.reshape(1, 4, REG_MAX, n).transpose(0, 2, 1, 3).copy(), axis=1)
    bins = np.arange(REG_MAX, dtype=np.float32).reshape(1, REG_MAX, 1, 1)
    ltrb = (d * bins).sum(1)

    x1y1 = anc - ltrb[:, 0:2]
    x2y2 = anc + ltrb[:, 2:4]
    cxcy = (x1y1 + x2y2) * 0.5
    wh = x2y2 - x1y1
    xywh = np.concatenate([cxcy, wh], 1) * st[:, None]

    kpt = kpt.astype(np.float32, copy=False).reshape(1, NUM_KPTS, KPT_DIM, n)
    grid_xy = (anc - 0.5)[:, None, :, :]                    # (1, 1, 2, n), un-offset grid coords
    kxy = (kpt[:, :, :2, :] * 2.0 + grid_xy) * st[:, None, None, :]
    kv = _sigmoid(kpt[:, :, 2:3, :])
    kpt_decoded = np.concatenate([kxy, kv], 2).reshape(1, NUM_KPTS * KPT_DIM, n)

    return np.concatenate([xywh, _sigmoid(cls.astype(np.float32, copy=False)),
                           kpt_decoded], 1)


def head_order(session, imgsz=INPUT_SIZE):
    """Indices that reorder a session's outputs into HEAD_OUTS order. See
    npu.yolo_decode.head_order -- identical logic, pose's own shapes."""
    expected = head_shapes(imgsz)
    outs = session.get_outputs()
    shapes = [tuple(o.shape) for o in outs]
    if sorted(shapes) != sorted(expected):
        raise SystemExit(f"unexpected head output shapes: {shapes}\n"
                         f"expected (in any order, for imgsz={imgsz}) {expected}")
    if [o.name for o in outs] == HEAD_OUTS:
        return list(range(len(HEAD_OUTS)))
    return [shapes.index(s) for s in expected]
