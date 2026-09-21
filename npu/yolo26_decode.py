"""YOLO26's anchor decode: the tail 1b_cut_head removes, in numpy.

YOLO26 dropped DFL. Its box branch regresses four values per anchor directly, where YOLOv8
emits 4 x REG_MAX bins reduced through a softmax-weighted sum, so this is
``npu.yolo_decode.decode_heads`` with that reduction taken out and nothing else changed - the
anchor grid, the stride vector and the output layout are identical, which is why
``npu.yolo.postprocess`` takes the result unchanged.

Transcribed from the ops the export actually contains, not from the paper:

    lt, rb = box[:, 0:2], box[:, 2:4]
    x1y1   = anchors - lt
    x2y2   = anchors + rb
    xywh   = concat([(x1y1 + x2y2) / 2, x2y2 - x1y1]) * strides
    out    = concat([xywh, sigmoid(cls)])

``tests`` compare this against the float export's own ``output0``; the anchor centres
(0.5, 1.5, ...) and strides (8, 16, 32) were read out of the graph's constants and match
``anchors_and_strides``.
"""
import numpy as np

from npu.yolo import INPUT_SIZE, NUM_CLASSES
from npu.yolo_decode import STRIDES, anchors_and_strides, _sigmoid

__all__ = ["decode_heads"]


def decode_heads(outs, imgsz=INPUT_SIZE, strides=STRIDES, nc=NUM_CLASSES, conf_thres=None):
    """
    outs: six arrays, box (1, 4, g, g) for each scale then cls (1, nc, g, g), the order
          the head cut writes them - all box, then all class, coarsest last.

    imgsz MUST be the size the input was letterboxed to: it sets the anchor grid, and a grid
    built for another resolution decodes every box to the wrong place instead of raising.

    Returns (1, 4 + nc, N) = concat([xywh in letterboxed pixels, class probabilities]), a
    drop-in for the export's ``output0``.

    conf_thres: decode only anchors whose best class logit clears logit(conf_thres). Identical
        results, because sigmoid is monotonic and the anchors dropped here are the ones
        postprocess would drop; N falls from 8400 to a few dozen.
    """
    if len(outs) != 6:
        raise ValueError(f"expected 6 head tensors, got {len(outs)}")
    box = np.concatenate([np.asarray(o, dtype=np.float32).reshape(1, 4, -1) for o in outs[:3]], axis=2)
    cls = np.concatenate([np.asarray(o, dtype=np.float32).reshape(1, nc, -1) for o in outs[3:]], axis=2)
    anchors, st = anchors_and_strides(imgsz, strides)
    if box.shape[2] != anchors.shape[2]:
        raise ValueError(f"{box.shape[2]} anchors from the heads, {anchors.shape[2]} from the "
                         f"grid for imgsz {imgsz} - the letterbox size and the heads disagree")
    if conf_thres is not None:
        keep = cls.max(axis=1)[0] >= float(np.log(conf_thres / (1.0 - conf_thres)))
        if not keep.any():
            return np.zeros((1, 4 + nc, 0), dtype=np.float32)
        box, cls, anchors, st = box[:, :, keep], cls[:, :, keep], anchors[:, :, keep], st[:, keep]
    x1y1 = anchors - box[:, 0:2]
    x2y2 = anchors + box[:, 2:4]
    xywh = np.concatenate([(x1y1 + x2y2) * 0.5, x2y2 - x1y1], axis=1) * st[:, None, :]
    return np.concatenate([xywh, _sigmoid(cls)], axis=1).astype(np.float32)