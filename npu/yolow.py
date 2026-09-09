"""
YOLO-World v2 constants, head-cut node names, and contrastive head decode.
Quark-free, like the rest of npu/.

YOLO-World v2 pairs an open-vocabulary vision backbone with text embeddings
(e.g. CLIP ViT-B/32) via text-guided cross-attention (C2fAttn with
MaxSigmoidAttnBlock at layers 12, 15, 18, 21).

The head cut extracts 6 tensors from the vision backbone:
  - 3 box distribution tensors (64ch) at strides 8, 16, 32
  - 3 visual feature tensors (512ch) at strides 8, 16, 32

The contrastive text projection (dot product between 512-ch visual features and
normalized text embeddings, followed by learned scale and bias) and the DFL
bounding box decoding run in NumPy on host CPU.
"""
from pathlib import Path
import numpy as np

from .paths import MODELS
from .yolo import (
    COCO_CLASSES,
    COCO_IDS,
    INPUT_SIZE,
    NUM_CLASSES,
    REG_MAX,
    STRIDES,
    draw,
    input_size,
    letterbox,
    postprocess,
)
from .yolo_decode import decode_heads

__all__ = [
    "BIASES",
    "BOX_OUTS",
    "COCO_CLASSES",
    "COCO_IDS",
    "HEAD_OUTS",
    "INPUT_SIZE",
    "NUM_CLASSES",
    "REG_MAX",
    "SCALES",
    "STRIDES",
    "TXT_FEAT_DIM",
    "VIS_OUTS",
    "decode_heads",
    "decode_yolow",
    "draw",
    "head_order",
    "head_shapes",
    "input_size",
    "letterbox",
    "load_coco_txt_feats",
    "postprocess",
]

TXT_FEAT_DIM = 512

# The six raw conv outputs of the cut YOLO-World v2 head in canonical order:
# box distributions (64ch) at strides 8/16/32, then visual features (512ch)
# at the same three strides.
BOX_OUTS = [f"/model.22/cv2.{i}/cv2.{i}.2/Conv_output_0" for i in range(3)]
VIS_OUTS = [f"/model.22/cv3.{i}/cv3.{i}.2/Conv_output_0" for i in range(3)]
HEAD_OUTS = BOX_OUTS + VIS_OUTS

# Learned logit scale (exp) and bias per detection level from yolov8s-worldv2
SCALES = (1.7862274646759033, 1.749911904335022, 1.8958474397659302)
BIASES = (-11.953125, -10.40625, -8.8671875)

COCO_TXT_FEATS_PATH = MODELS / "yolow_coco_txt_feats.npy"


def head_shapes(imgsz=INPUT_SIZE, strides=STRIDES, feat_dim=TXT_FEAT_DIM):
    """Expected shapes of HEAD_OUTS for a square input of `imgsz`."""
    grids = [imgsz // s for s in strides]
    return ([(1, 4 * REG_MAX, g, g) for g in grids]
            + [(1, feat_dim, g, g) for g in grids])


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


def load_coco_txt_feats(path=COCO_TXT_FEATS_PATH):
    """Load normalized COCO 80-class text embeddings (shape: [80, 512])."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"COCO text embeddings not found at {p}. "
            "Run pipelines/yolow/1_export.py first to generate them."
        )
    return np.load(p)


def decode_yolow(outs, txt_feats_norm, imgsz=INPUT_SIZE, scales=SCALES,
                 biases=BIASES, conf_thres=None):
    """Contrastive visual-text dot product and anchor decoding.

    outs: six arrays in HEAD_OUTS order -- box (1,64,g,g) then vis (1,512,g,g)
    txt_feats_norm: (K, 512) normalized text embeddings for K classes
    scales, biases: learned per-level scale and bias for contrastive logits

    Returns (1, 4 + K, N) compatible with npu.yolo.postprocess.
    """
    box_outs = outs[:3]
    vis_outs = outs[3:]

    cls_logits_list = []
    for i in range(3):
        vis_feat = vis_outs[i][0]  # (512, H, W)
        vis_hwc = np.transpose(vis_feat, (1, 2, 0))  # (H, W, 512)
        sim = vis_hwc @ txt_feats_norm.T  # (H, W, nc)
        logits = sim * scales[i] + biases[i]  # (H, W, nc)
        cls_logits = np.transpose(logits, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        cls_logits_list.append(cls_logits)

    yolo_outs = box_outs + cls_logits_list
    return decode_heads(yolo_outs, imgsz=imgsz, conf_thres=conf_thres)
