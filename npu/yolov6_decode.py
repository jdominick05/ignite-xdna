"""
Numpy reimplementation of the yolov6n detect-head tail that
pipelines/yolov6n/1b_cut_head.py removes from the ONNX graph. Quark-free,
like the rest of npu/.

Given the six raw conv outputs (npu.yolov6.HEAD_OUTS order), produces the
same (1, 84, N) tensor npu.yolo.postprocess expects.

Simpler than npu.yolo_decode: yolov6n has reg_max=0 / use_dfl=False
(configs/yolov6n.py), so the box head already IS the ltrb distance, straight
out of a Conv -- no DFL softmax / expected-value step. The dist2bbox formula
(anchor_points +/- ltrb, then * stride) was matched against
yolov6/utils/general.py::dist2bbox and yolov6/assigners/anchor_generator.py's
generate_anchors(is_eval=True, mode='af') in the upstream repo.
"""
import numpy as np

from .yolov6 import HEAD_OUTS, INPUT_SIZE, STRIDES, head_shapes

__all__ = ["STRIDES", "anchors_and_strides", "decode_heads", "head_order"]

_cache: dict = {}


def anchors_and_strides(imgsz=INPUT_SIZE, strides=STRIDES):
    """(1,2,N) anchor centres in grid units, (1,N) stride per anchor. Same
    grid_cell_offset=0.5 convention as upstream generate_anchors(mode='af')."""
    key = (imgsz, strides)
    if key in _cache:
        return _cache[key]
    pts, sts = [], []
    for s in strides:
        g = imgsz // s
        c = np.arange(g, dtype=np.float32) + 0.5
        yy, xx = np.meshgrid(c, c, indexing="ij")
        pts.append(np.stack([xx.ravel(), yy.ravel()], 0))
        sts.append(np.full(g * g, float(s), np.float32))
    out = (np.concatenate(pts, 1)[None], np.concatenate(sts)[None])
    _cache[key] = out
    return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def decode_heads(outs, imgsz=INPUT_SIZE, strides=STRIDES, conf_thres=None):
    """
    outs: six arrays in npu.yolov6.HEAD_OUTS order -- reg (1,4,g,g) then
          cls (1,80,g,g), for g = imgsz/8, imgsz/16, imgsz/32.

    imgsz MUST match the size the input was letterboxed to, read from the
    model itself via npu.yolov6.input_size -- see npu.yolo_decode.decode_heads'
    docstring for why a stale value fails silently instead of raising.

    Returns (1, 84, N) = concat([xywh in letterboxed pixels, class probs]),
    drop-in for npu.yolo.postprocess.

    conf_thres: if given, only anchors whose best class logit clears
        logit(conf_thres) are decoded, same early-exit trick as
        npu.yolo_decode.decode_heads (sigmoid is monotonic, so this drops
        exactly the anchors postprocess() would have dropped anyway).
    """
    reg_f, cls_f = outs[:3], outs[3:]
    nc = cls_f[0].shape[1]

    reg = np.concatenate([r.reshape(1, 4, -1) for r in reg_f], 2).astype(np.float32, copy=False)
    cls = np.concatenate([c.reshape(1, nc, -1) for c in cls_f], 2)
    anc, st = anchors_and_strides(imgsz, strides)

    if conf_thres is not None:
        c = min(max(float(conf_thres), 1e-12), 1.0 - 1e-12)
        t = np.log(c / (1.0 - c))  # inverse sigmoid
        keep = np.flatnonzero(cls[0].max(0) > t)
        if keep.size == 0:
            return np.zeros((1, 4 + nc, 0), np.float32)
        reg = reg[:, :, keep]
        cls = cls[:, :, keep]
        anc = anc[:, :, keep]
        st = st[:, keep]

    # reg is already ltrb in grid units -- no DFL decode needed (reg_max=0).
    x1y1 = anc - reg[:, 0:2]
    x2y2 = anc + reg[:, 2:4]
    cxcy = (x1y1 + x2y2) * 0.5
    wh = x2y2 - x1y1
    xywh = np.concatenate([cxcy, wh], 1) * st[:, None]

    return np.concatenate([xywh, _sigmoid(cls.astype(np.float32, copy=False))], 1)


def head_order(session, imgsz=INPUT_SIZE):
    """Indices that reorder a session's outputs into HEAD_OUTS order. Same
    shape-fallback contract as npu.yolo_decode.head_order."""
    expected = head_shapes(imgsz)
    outs = session.get_outputs()
    shapes = [tuple(o.shape) for o in outs]
    if sorted(shapes) != sorted(expected):
        raise SystemExit(f"unexpected head output shapes: {shapes}\n"
                         f"expected (in any order, for imgsz={imgsz}) {expected}")
    if [o.name for o in outs] == HEAD_OUTS:
        return list(range(len(HEAD_OUTS)))
    return [shapes.index(s) for s in expected]
