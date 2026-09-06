"""
Numpy reimplementation of the yolov8 detect-head tail that
pipelines/yolov8n/1b_cut_head.py removes from the ONNX graph. Quark-free, like
the rest of npu/.

Given the six raw conv outputs (npu.yolo.HEAD_OUTS order), produces the same
(1, 84, N) tensor `output0` carried, so npu.yolo.postprocess needs no changes.

The anchor grid and stride vector built here were verified bit-identical to the
constants ultralytics bakes into the exported graph (/model.22/Constant_12,
_13, _15).
"""
import numpy as np

from .yolo import HEAD_OUTS, INPUT_SIZE, REG_MAX, STRIDES, head_shapes

# REG_MAX and STRIDES now live in npu.yolo, because head_shapes() there needs
# them and this module already imports from that one. Re-exported here so
# `from npu.yolo_decode import STRIDES` keeps working.
__all__ = ["REG_MAX", "STRIDES", "anchors_and_strides", "decode_heads", "head_order"]

_cache: dict = {}


def anchors_and_strides(imgsz=INPUT_SIZE, strides=STRIDES):
    """(1,2,N) anchor centres in grid units, (1,N) stride per anchor."""
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


def _softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=axis, keepdims=True)
    return x


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def decode_heads(outs, imgsz=INPUT_SIZE, strides=STRIDES, conf_thres=None):
    """
    outs: six arrays in npu.yolo.HEAD_OUTS order -- box (1,64,g,g) then
          cls (1,80,g,g), for g = imgsz/8, imgsz/16, imgsz/32.

    imgsz MUST match the size the input was letterboxed to: it sets the anchor
    grid, and an anchor grid built for a different resolution silently decodes
    every box to the wrong place instead of raising. Callers get it from the
    model itself via npu.yolo.input_size rather than assuming 640.

    Returns (1, 84, N) = concat([xywh in letterboxed pixels, class probs]),
    drop-in for `output0`, so npu.yolo.postprocess takes it unchanged.

    conf_thres: if given, only anchors whose best class logit clears
        logit(conf_thres) are decoded. N shrinks from 8400 to ~dozens and the
        DFL softmax / sigmoid cost collapses. Results are identical because
        sigmoid is monotonic - the anchors dropped here are exactly the ones
        postprocess() would have dropped. Pass None for the full tensor.
    """
    box_f, cls_f = outs[:3], outs[3:]
    nc = cls_f[0].shape[1]

    box = np.concatenate([b.reshape(1, 4 * REG_MAX, -1) for b in box_f], 2)
    cls = np.concatenate([c.reshape(1, nc, -1) for c in cls_f], 2)
    anc, st = anchors_and_strides(imgsz, strides)

    if conf_thres is not None:
        # Clamped: conf_thres of exactly 1.0 is a ZeroDivisionError here and
        # 0.0 a divide-by-zero warning, and --conf 0 / --conf 1 are both
        # things a caller reasonably passes to mean keep-all / keep-none.
        c = min(max(float(conf_thres), 1e-12), 1.0 - 1e-12)
        t = np.log(c / (1.0 - c))  # inverse sigmoid
        keep = np.flatnonzero(cls[0].max(0) > t)
        if keep.size == 0:
            return np.zeros((1, 4 + nc, 0), np.float32)
        box = box[:, :, keep]
        cls = cls[:, :, keep]
        anc = anc[:, :, keep]
        st = st[:, keep]

    box = box.astype(np.float32, copy=False)
    n = box.shape[2]

    # DFL: 16 bins per box side -> expected value
    d = _softmax(box.reshape(1, 4, REG_MAX, n).transpose(0, 2, 1, 3).copy(), axis=1)
    bins = np.arange(REG_MAX, dtype=np.float32).reshape(1, REG_MAX, 1, 1)
    ltrb = (d * bins).sum(1)  # (1,4,n) distances left, top, right, bottom

    x1y1 = anc - ltrb[:, 0:2]
    x2y2 = anc + ltrb[:, 2:4]
    cxcy = (x1y1 + x2y2) * 0.5
    wh = x2y2 - x1y1
    xywh = np.concatenate([cxcy, wh], 1) * st[:, None]

    return np.concatenate([xywh, _sigmoid(cls.astype(np.float32, copy=False))], 1)


def head_order(session, imgsz=INPUT_SIZE):
    """Indices that reorder a session's outputs into HEAD_OUTS order.

    Quantization renames tensors (foo_output_0 ->
    foo_output_0_DequantizeLinear_Output), so fall back to matching by shape.

    The shape fallback is also the check that `imgsz` is right: pass the value
    npu.yolo.input_size read off this same session and a mismatch is impossible,
    but pass a stale 640 to a 512 model and the expected grids are not found and
    this raises with a message naming both. decode_heads would catch it too --
    anchor count is a function of imgsz, so a mismatch always breaks the
    broadcast -- but it surfaces as an opaque ValueError about (1,2,8400) versus
    (1,2,5376). Failing here instead is the difference between the two.
    """
    expected = head_shapes(imgsz)
    outs = session.get_outputs()
    # Shapes are checked before the name fast-path, not after it: an unquantized
    # cut model keeps the HEAD_OUTS names at every resolution, so matching on
    # names alone would return an order without ever noticing that imgsz is
    # wrong for this model.
    shapes = [tuple(o.shape) for o in outs]
    if sorted(shapes) != sorted(expected):
        raise SystemExit(f"unexpected head output shapes: {shapes}\n"
                         f"expected (in any order, for imgsz={imgsz}) {expected}")
    if [o.name for o in outs] == HEAD_OUTS:
        return list(range(len(HEAD_OUTS)))
    return [shapes.index(s) for s in expected]
