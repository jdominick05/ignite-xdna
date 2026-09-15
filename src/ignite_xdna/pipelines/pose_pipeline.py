"""
src/ignite_xdna/pipelines/pose_pipeline.py

Person keypoints (YOLOv8n-pose) on the Phoenix NPU from a graph-engine ``.ignite`` container
(``task == "pose"``). One dispatch per frame runs every convolution on the NPU; the host puts the
letterboxed frame into the input plane, reads the nine head tensors back (box distributions, person
score and 17 keypoints, at strides 8, 16 and 32) and decodes them:

* boxes: a softmax over 16 distance bins per side around each anchor (DFL);
* score: the sigmoid of the person logit;
* keypoints: ``x = (raw_x * 2 + grid_x) * stride`` (likewise y) and a sigmoid visibility, as
  ultralytics' ``Pose.kpts_decode``;
* one class, so NMS is class-agnostic, keeping at most ``max_det`` people.

The decode repeats ``npu/yolo_pose_decode.py`` and ``npu/yolo_pose.py`` operation for operation (same
dtypes, same order), so on the same heads it returns the same boxes, scores and keypoints, bit for bit,
as the numpy tail ``pipelines/yolov8n-pose`` runs after ONNX Runtime
(``tests/test_pose_pipeline_offline.py``). int8 heads are pruned on the dequantized score of each level
before their box and keypoint tensors are dequantized.

``ingress="native"`` (the default) letterboxes and quantizes the frame in one native pass straight into
the mapped input plane, as ``YoloPipeline`` does for detection. ``ingress="numpy"`` builds the input
``pipelines/yolov8n-pose/5_eval_map.py`` gives ONNX Runtime: ``npu/yolo.py``'s cv2 letterbox followed by
the model's QuantizeLinear. The two inputs differ by one code in some pixel values.

    with PosePipeline("build/yolov8n_pose.ignite") as pose:
        people, timings = pose.predict_sync(frame_bgr)
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

STRIDES = (8, 16, 32)
REG_MAX = 16
NUM_KPTS = 17
KPT_DIM = 3
# Head order of npu/yolo_pose.py's HEAD_OUTS: box P3-P5, score P3-P5, keypoints P3-P5.
HEAD_ORDER = ("p3_box", "p4_box", "p5_box", "p3_cls", "p4_cls", "p5_cls", "p3_kpt", "p4_kpt", "p5_kpt")

COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
# The 17-point skeleton ultralytics draws, 0-indexed into COCO_KEYPOINTS.
SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)


@dataclass
class PoseDetection:
    """One person in original image pixels; ``keypoints`` is float32 (17, 3): x, y, visibility."""
    x0: float
    y0: float
    w: float
    h: float
    score: float
    keypoints: np.ndarray


@dataclass
class PoseTimings:
    """Stage latencies of one frame in milliseconds.

    ``npu_forward_ms`` spans dispatch and head readback; ``dispatch_ms`` is the NPU dispatch alone and
    ``readback_ms`` the head syncs and int8 conversion.
    """
    preprocess_ms: float
    npu_forward_ms: float
    dispatch_ms: float
    readback_ms: float
    postprocess_ms: float
    glass_to_glass_ms: float
    source: str = "npu"


def _softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=axis, keepdims=True)
    return x


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def _dequantize(q: np.ndarray, scale_zero_point: Tuple[float, int]) -> np.ndarray:
    """float32 ``(q - zero_point) * scale``, the DequantizeLinear the cut model ends in."""
    scale, zero_point = scale_zero_point
    return (q.astype(np.float32) - np.float32(zero_point)) * np.float32(scale)


def anchors_and_strides(imgsz: int = 640, strides: Sequence[int] = STRIDES) -> Tuple[np.ndarray, np.ndarray]:
    """(1, 2, N) anchor centres in grid units and (1, N) strides, P3 anchors first."""
    pts, sts = [], []
    for s in strides:
        g = imgsz // s
        c = np.arange(g, dtype=np.float32) + 0.5
        yy, xx = np.meshgrid(c, c, indexing="ij")
        pts.append(np.stack([xx.ravel(), yy.ravel()], 0))
        sts.append(np.full(g * g, float(s), np.float32))
    return np.concatenate(pts, 1)[None], np.concatenate(sts)[None]


def letterbox(img_bgr: np.ndarray, size: int = 640):
    """``npu/yolo.py``'s letterbox: keep the aspect ratio, pad with grey 114 to ``size`` x ``size``.

    Returns the NCHW float32 RGB tensor in [0, 1], ``(pad_top, pad_left)`` and the scale.
    """
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


class PoseDecoder:
    """Device-free half of the pipeline: anchor grids, head decode and NMS (no NPU needed)."""

    def __init__(self, imgsz: int = 640, conf_thres: float = 0.25, iou_thres: float = 0.5, max_det: int = 300):
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self._anchors, self._strides = anchors_and_strides(imgsz)

    def decode(self, heads: Union[Mapping[str, np.ndarray], Sequence[np.ndarray]],
               scales: Optional[Mapping[str, Tuple[float, int]]] = None,
               conf_thres: Optional[float] = None) -> np.ndarray:
        """Heads -> (1, 56, N): xywh in letterboxed pixels, person probability, 17 x (x, y, visibility).

        ``heads`` is the nine float tensors in ``HEAD_ORDER`` (a sequence or a mapping by name), or a
        mapping of int8 views with ``scales`` ``{name: (scale, zero_point)}`` as
        ``GraphSession.run_yolo_monolithic`` returns them. With ``conf_thres`` only anchors whose score
        logit is above the threshold's logit are decoded.
        """
        if isinstance(heads, Mapping):
            outs = [heads[name] for name in HEAD_ORDER]
        else:
            outs = list(heads)
        nc = outs[3].shape[1]
        t = None
        if conf_thres is not None:
            c = min(max(float(conf_thres), 1e-12), 1.0 - 1e-12)
            t = np.log(c / (1.0 - c))
        boxes, classes, kpts, keeps = [], [], [], []
        offset = 0
        for level in range(3):
            b = outs[level].reshape(1, 4 * REG_MAX, -1)
            cl = outs[3 + level].reshape(1, nc, -1)
            k = outs[6 + level].reshape(1, NUM_KPTS * KPT_DIM, -1)
            if scales is not None:
                cl = _dequantize(cl, scales[HEAD_ORDER[3 + level]])
            n = cl.shape[2]
            if t is None:
                keep = np.arange(n)
            else:
                keep = np.flatnonzero(cl[0].max(0) > t)
            if keep.size:
                b, k = b[:, :, keep], k[:, :, keep]
                if scales is not None:
                    b = _dequantize(b, scales[HEAD_ORDER[level]])
                    k = _dequantize(k, scales[HEAD_ORDER[6 + level]])
                boxes.append(b)
                classes.append(cl[:, :, keep])
                kpts.append(k)
                keeps.append(keep + offset)
            offset += n
        if not keeps:
            return np.zeros((1, 4 + nc + NUM_KPTS * KPT_DIM, 0), np.float32)
        box = np.concatenate(boxes, 2)
        cls = np.concatenate(classes, 2)
        kpt = np.concatenate(kpts, 2)
        keep = np.concatenate(keeps)
        anc, st = self._anchors[:, :, keep], self._strides[:, keep]

        # From here on, npu/yolo_pose_decode.decode_heads line for line.
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
        grid_xy = (anc - 0.5)[:, None, :, :]
        kxy = (kpt[:, :, :2, :] * 2.0 + grid_xy) * st[:, None, None, :]
        kv = _sigmoid(kpt[:, :, 2:3, :])
        kpt_decoded = np.concatenate([kxy, kv], 2).reshape(1, NUM_KPTS * KPT_DIM, n)
        return np.concatenate([xywh, _sigmoid(cls.astype(np.float32, copy=False)), kpt_decoded], 1)

    def postprocess(self, output: np.ndarray, pad: Tuple[int, int], scale: float,
                    conf_thres: Optional[float] = None, iou_thres: Optional[float] = None,
                    max_det: Optional[int] = None) -> List[PoseDetection]:
        """(1, 56, N) from :meth:`decode` -> people in original image pixels (``npu/yolo_pose.py``'s
        postprocess: score filter, top ``10 * max_det``, class-agnostic NMS, at most ``max_det``)."""
        conf_thres = self.conf_thres if conf_thres is None else conf_thres
        iou_thres = self.iou_thres if iou_thres is None else iou_thres
        max_det = self.max_det if max_det is None else max_det
        out = output[0].T
        boxes = out[:, :4].copy()
        scores = out[:, 4]
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
        return [PoseDetection(x0=float(xywh[i, 0]), y0=float(xywh[i, 1]), w=float(xywh[i, 2]), h=float(xywh[i, 3]),
                              score=float(scores[i]), keypoints=kpts[i]) for i in idx]


class PosePipeline(PoseDecoder):
    """Owns one ``GraphSession`` on a ``pose`` container (one XRT hardware context) for its lifetime."""

    def __init__(self, model_path: Union[str, Path], device_index: int = 0, conf_thres: float = 0.25,
                 iou_thres: float = 0.5, max_det: int = 300, ingress: str = "native"):
        from ignite_xdna.runtime.graph_session import GraphSession
        if ingress not in ("native", "numpy"):
            raise ValueError(f"ingress must be 'native' or 'numpy', not {ingress!r}")
        self.model_path = Path(model_path)
        self.session: Optional[GraphSession] = GraphSession(self.model_path, device_index=device_index)
        if self.session.task != "pose":
            task = self.session.task
            self.close()
            raise ValueError(f"{self.model_path} is a {task} container, not a pose container")
        super().__init__(imgsz=int(self.session.input_placement["width"]), conf_thres=conf_thres,
                         iou_thres=iou_thres, max_det=max_det)
        if ingress == "native" and not self.session.direct_ingress:
            raise RuntimeError("native ingress needs the native preprocessor library; pass ingress='numpy'")
        self.ingress = ingress
        qs = self.session.ignite_manifest["quant_scales"]
        self._input_scale = float(qs["input_scale"])
        self._input_zero_point = int(qs.get("input_zero_point", 128))

    def stage(self, img_bgr: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """Put a BGR frame into the NPU input plane; returns the letterbox ``(pad, scale)``."""
        if self.ingress == "native":
            return self.session.stage_image(img_bgr)
        x, pad, scale = letterbox(img_bgr, self.imgsz)
        # The model's input QuantizeLinear (graph_reference.quantize_input): round half to even.
        q = np.clip(np.round(x[0].astype(np.float64) / self._input_scale).astype(np.int64) + self._input_zero_point,
                    0, 255).astype(np.uint8)
        self.session.stage_quantized(q)
        return pad, scale

    def predict_sync(self, img_bgr: np.ndarray, conf_thres: Optional[float] = None,
                     iou_thres: Optional[float] = None) -> Tuple[List[PoseDetection], PoseTimings]:
        """BGR uint8 frame -> people (boxes, scores, 17 keypoints each) and stage timings."""
        session = self.session
        if session is None:
            raise RuntimeError("PosePipeline is closed")
        conf = self.conf_thres if conf_thres is None else conf_thres
        t0 = time.perf_counter()
        pad, scale = self.stage(img_bgr)
        t1 = time.perf_counter()
        heads, ts = session.run_yolo_monolithic(None, return_timestamps=True)
        t2 = time.perf_counter()
        if not heads["heads_present"]:
            raise RuntimeError(f"{self.model_path}: the egress carries no pose heads ({heads['head_status']})")
        people = self.postprocess(self.decode(heads, heads["scales"], conf), pad, scale, conf, iou_thres)
        t3 = time.perf_counter()
        return people, PoseTimings(preprocess_ms=(t1 - t0) * 1e3, npu_forward_ms=(t2 - t1) * 1e3,
                                   dispatch_ms=ts["npu_ms"], readback_ms=ts["readback_ms"],
                                   postprocess_ms=(t3 - t2) * 1e3, glass_to_glass_ms=(t3 - t0) * 1e3)

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
