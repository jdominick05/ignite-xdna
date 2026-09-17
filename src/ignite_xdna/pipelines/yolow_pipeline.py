"""
src/ignite_xdna/pipelines/yolow_pipeline.py

Open-vocabulary detection (YOLO-World v2) on the Phoenix NPU, with the classes chosen at run time.

A YOLO-World v2 container (``pipelines/yolow/``: head cut, XINT8, ``3c_gptq_cv2.py``, compiled with host regions
``/model.{12,15,18,21}/attn/``) runs its four text cross-attention blocks (each with its projection convolution) on
the CPU between NPU segments and everything else on the NPU (63 of its 67 convolutions). The vocabulary enters only through those attention cores' text guides and the contrastive
decode (``yolow_text``), so ``set_classes`` swaps it on an open session: it embeds the names with CLIP's text
encoder, replaces the guides in the host segments (``EngineSession.set_host_constants``) and keeps the embeddings for
the decode. The NPU program never changes.

Per frame the host puts the letterboxed frame into the input plane, runs the segments, reads the six head tensors
back (box distributions and 512-channel visual features at strides 8, 16 and 32) and decodes them:

* class logits: each anchor's visual feature dotted with every class embedding, times the level's learned scale plus
  its bias (the export folds the contrastive head's BatchNorm into the last visual convolution);
* boxes: a softmax over 16 distance bins per side (DFL), as ``npu/yolo_decode.py``;
* per-class NMS, as ``npu/yolo.py``'s postprocess.

The decode repeats ``npu/yolow.py``'s ``decode_yolow`` and ``npu/yolo.py``'s ``postprocess`` operation for operation,
so on the same heads and constants it returns the same detections bit for bit (``tests/test_yolow_pipeline_offline.py``).

    with YoloWorldPipeline("build/yolow.ignite", "models/yolow_text_encoder.onnx", "models/yolow_text.npz",
                           classes=["person", "red backpack"]) as world:
        detections, timings = world.predict_sync(frame_bgr)
        world.set_classes(["dog", "frisbee"])
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from ignite_xdna.pipelines.pose_pipeline import REG_MAX, _dequantize, _sigmoid, _softmax, anchors_and_strides, letterbox

HEAD_ORDER = ("p3_box", "p4_box", "p5_box", "p3_cls", "p4_cls", "p5_cls")   # cls: the 512-channel visual features


def _pow2_zero_point_0(scale_zero_point: Tuple[float, int]) -> bool:
    scale, zero_point = scale_zero_point
    return int(zero_point) == 0 and scale > 0 and float(np.log2(scale)).is_integer()


@dataclass
class WorldDetection:
    """One detection in original image pixels."""
    x0: float
    y0: float
    w: float
    h: float
    score: float
    class_id: int
    name: str


@dataclass
class WorldTimings:
    """Stage latencies of one frame in milliseconds (see ``PoseTimings``); ``host_ms`` is the attention cores' part
    of ``npu_forward_ms``."""
    preprocess_ms: float
    npu_forward_ms: float
    dispatch_ms: float
    host_ms: float
    readback_ms: float
    postprocess_ms: float
    glass_to_glass_ms: float
    source: str = "npu"


class YoloWorldDecoder:
    """Device-free half: contrastive class logits, DFL boxes and per-class NMS for a given vocabulary."""

    def __init__(self, embeddings: np.ndarray, names: Sequence[str], contrastive_scales: Sequence[float],
                 contrastive_biases: Sequence[float], imgsz: int = 640, conf_thres: float = 0.25,
                 iou_thres: float = 0.5, max_det: int = 300):
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.contrastive_scales = tuple(contrastive_scales)
        self.contrastive_biases = tuple(contrastive_biases)
        self._anchors, self._strides = anchors_and_strides(imgsz)
        self.set_vocabulary(embeddings, names)

    def set_vocabulary(self, embeddings: np.ndarray, names: Sequence[str]) -> None:
        e = np.asarray(embeddings, dtype=np.float32)
        if e.ndim != 2 or e.shape[0] != len(names):
            raise ValueError(f"{len(names)} names but embeddings of shape {e.shape}")
        self.embeddings, self.names = e, list(names)

    def decode(self, heads: Union[Mapping[str, np.ndarray], Sequence[np.ndarray]],
               scales: Optional[Mapping[str, Tuple[float, int]]] = None,
               conf_thres: Optional[float] = None) -> np.ndarray:
        """Heads -> (1, 4 + K, N): xywh in letterboxed pixels and per-class probabilities.

        ``heads`` is the six float tensors in ``HEAD_ORDER``, or int8 views by name with ``scales``
        ``{name: (scale, zero_point)}`` as ``GraphSession.run_yolo_monolithic`` returns them. With ``conf_thres``
        only anchors whose best logit clears the threshold's logit are box-decoded (identical detections).
        """
        if isinstance(heads, Mapping) and scales is not None and all(_pow2_zero_point_0(scales[n]) for n in HEAD_ORDER):
            return self._decode_int8(heads, scales, conf_thres)
        outs = [heads[n] for n in HEAD_ORDER] if isinstance(heads, Mapping) else list(heads)
        if scales is not None:
            outs = [_dequantize(o, scales[n]) for o, n in zip(outs, HEAD_ORDER)]
        # npu/yolow.py decode_yolow: contrastive logits per level.
        cls_f = []
        for i in range(3):
            vis_hwc = np.transpose(outs[3 + i][0], (1, 2, 0))
            sim = vis_hwc @ self.embeddings.T
            logits = sim * self.contrastive_scales[i] + self.contrastive_biases[i]
            cls_f.append(np.transpose(logits, (2, 0, 1))[np.newaxis, ...].astype(np.float32))
        # npu/yolo_decode.py decode_heads.
        nc = cls_f[0].shape[1]
        box = np.concatenate([b.reshape(1, 4 * REG_MAX, -1) for b in outs[:3]], 2)
        cls = np.concatenate([c.reshape(1, nc, -1) for c in cls_f], 2)
        anc, st = self._anchors, self._strides
        if conf_thres is not None:
            c = min(max(float(conf_thres), 1e-12), 1.0 - 1e-12)
            t = np.log(c / (1.0 - c))
            keep = np.flatnonzero(cls[0].max(0) > t)
            if keep.size == 0:
                return np.zeros((1, 4 + nc, 0), np.float32)
            box, cls, anc, st = box[:, :, keep], cls[:, :, keep], anc[:, :, keep], st[:, keep]
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
        return np.concatenate([xywh, _sigmoid(cls.astype(np.float32, copy=False))], 1)

    def _decode_int8(self, heads: Mapping[str, np.ndarray], scales: Mapping[str, Tuple[float, int]],
                     conf_thres: Optional[float]) -> np.ndarray:
        """``decode`` on int8 heads whose scales are powers of two at zero point 0, bit for bit, in about half the time.

        The contrastive product runs on the int8 values with the head scale folded into the level's logit scale:
        ``((q * s) @ E) * S == (q @ E) * (s * S)`` exactly when ``s`` is a power of two, because scaling by a power of
        two is exact in floating point (the product and every partial sum are scaled, not rounded). Anchors are pruned
        level by level on the same per-anchor maxima, so no full logit tensor is concatenated, and only the kept
        anchors' box distributions are dequantized.
        """
        c = None if conf_thres is None else min(max(float(conf_thres), 1e-12), 1.0 - 1e-12)
        t = None if c is None else np.log(c / (1.0 - c))
        emb_t = self.embeddings.T
        nc = self.embeddings.shape[0]
        cls_kept, box_kept, keeps = [], [], []
        offset = 0
        for i in range(3):
            q = heads[HEAD_ORDER[3 + i]]
            s = float(scales[HEAD_ORDER[3 + i]][0])
            hw = q.shape[2] * q.shape[3]
            sim = np.transpose(q[0].astype(np.float32), (1, 2, 0)) @ emb_t
            logits = (sim * np.float32(s * self.contrastive_scales[i]) + np.float32(self.contrastive_biases[i]))
            logits = logits.reshape(hw, nc)
            keep = np.arange(hw) if t is None else np.flatnonzero(logits.max(1) > t)
            if keep.size:
                cls_kept.append(logits[keep])
                box_kept.append(_dequantize(heads[HEAD_ORDER[i]].reshape(1, 4 * REG_MAX, -1)[:, :, keep],
                                            scales[HEAD_ORDER[i]]))
                keeps.append(keep + offset)
            offset += hw
        if not keeps:
            return np.zeros((1, 4 + nc, 0), np.float32)
        keep = np.concatenate(keeps)
        cls = np.ascontiguousarray(np.concatenate(cls_kept, 0).T)[np.newaxis]
        box = np.concatenate(box_kept, 2)
        anc, st = self._anchors[:, :, keep], self._strides[:, keep]
        n = box.shape[2]
        d = _softmax(box.reshape(1, 4, REG_MAX, n).transpose(0, 2, 1, 3).copy(), axis=1)
        bins = np.arange(REG_MAX, dtype=np.float32).reshape(1, REG_MAX, 1, 1)
        ltrb = (d * bins).sum(1)
        x1y1 = anc - ltrb[:, 0:2]
        x2y2 = anc + ltrb[:, 2:4]
        cxcy = (x1y1 + x2y2) * 0.5
        wh = x2y2 - x1y1
        xywh = np.concatenate([cxcy, wh], 1) * st[:, None]
        return np.concatenate([xywh, _sigmoid(cls)], 1)

    def postprocess(self, output: np.ndarray, pad: Tuple[int, int], scale: float,
                    conf_thres: Optional[float] = None, iou_thres: Optional[float] = None,
                    max_det: Optional[int] = None) -> List[WorldDetection]:
        """(1, 4 + K, N) -> detections in original image pixels (``npu/yolo.py``'s per-class NMS postprocess)."""
        conf_thres = self.conf_thres if conf_thres is None else conf_thres
        iou_thres = self.iou_thres if iou_thres is None else iou_thres
        max_det = self.max_det if max_det is None else max_det
        out = output[0].T
        boxes = out[:, :4].copy()
        scores_all = out[:, 4:]
        cls = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(cls)), cls]
        keep = scores >= conf_thres
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        if len(boxes) == 0:
            return []
        if len(scores) > 10 * max_det:
            top = np.argpartition(-scores, 10 * max_det)[:10 * max_det]
            boxes, scores, cls = boxes[top], scores[top], cls[top]
        boxes[:, 0] -= pad[1]
        boxes[:, 1] -= pad[0]
        boxes /= scale
        xywh = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 2], boxes[:, 3]], axis=1)
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), cls.tolist(), conf_thres, iou_thres)
        if len(idx) == 0:
            return []
        idx = np.array(idx).reshape(-1)
        if len(idx) > max_det:
            idx = idx[np.argsort(-scores[idx])[:max_det]]
        return [WorldDetection(x0=float(xywh[i, 0]), y0=float(xywh[i, 1]), w=float(xywh[i, 2]), h=float(xywh[i, 3]),
                               score=float(scores[i]), class_id=int(cls[i]), name=self.names[int(cls[i])]) for i in idx]


class YoloWorldPipeline(YoloWorldDecoder):
    """Owns one ``GraphSession`` on a YOLO-World v2 container and the text encoder that sets its vocabulary."""

    def __init__(self, model_path: Union[str, Path], text_encoder: Union[str, Path], text_bundle: Union[str, Path],
                 classes: Sequence[str], device_index: int = 0, conf_thres: float = 0.25, iou_thres: float = 0.5,
                 max_det: int = 300, ingress: str = "native"):
        from ignite_xdna.pipelines.yolow_text import YoloWorldText
        from ignite_xdna.runtime.graph_session import GraphSession
        if ingress not in ("native", "numpy"):
            raise ValueError(f"ingress must be 'native' or 'numpy', not {ingress!r}")
        self.text = YoloWorldText(text_encoder, text_bundle)
        self.model_path = Path(model_path)
        self.session: Optional[GraphSession] = GraphSession(self.model_path, device_index=device_index)
        try:
            imgsz = int(self.session.input_placement["width"])
            if ingress == "native" and not self.session.direct_ingress:
                raise RuntimeError("native ingress needs the native preprocessor library; pass ingress='numpy'")
            embeddings, constants = self.text.vocabulary(list(classes))
            self.session.set_host_constants(constants)
        except Exception:
            self.close()
            raise
        super().__init__(embeddings, list(classes), self.text.contrastive_scales, self.text.contrastive_biases,
                         imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
        self.ingress = ingress
        qs = self.session.ignite_manifest["quant_scales"]
        self._input_scale = float(qs["input_scale"])
        self._input_zero_point = int(qs.get("input_zero_point", 128))

    def set_classes(self, classes: Sequence[str]) -> float:
        """Swap the vocabulary on the open session; returns the milliseconds it took (text encoder included)."""
        if self.session is None:
            raise RuntimeError("YoloWorldPipeline is closed")
        t0 = time.perf_counter()
        embeddings, constants = self.text.vocabulary(list(classes))
        self.session.set_host_constants(constants)
        self.set_vocabulary(embeddings, list(classes))
        return (time.perf_counter() - t0) * 1e3

    def stage(self, img_bgr: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """Put a BGR frame into the NPU input plane; returns the letterbox ``(pad, scale)``."""
        if self.ingress == "native":
            return self.session.stage_image(img_bgr)
        x, pad, scale = letterbox(img_bgr, self.imgsz)
        q = np.clip(np.round(x[0].astype(np.float64) / self._input_scale).astype(np.int64) + self._input_zero_point,
                    0, 255).astype(np.uint8)
        self.session.stage_quantized(q)
        return pad, scale

    def predict_sync(self, img_bgr: np.ndarray, conf_thres: Optional[float] = None,
                     iou_thres: Optional[float] = None) -> Tuple[List[WorldDetection], WorldTimings]:
        """BGR uint8 frame -> detections of the current vocabulary and stage timings."""
        session = self.session
        if session is None:
            raise RuntimeError("YoloWorldPipeline is closed")
        conf = self.conf_thres if conf_thres is None else conf_thres
        t0 = time.perf_counter()
        pad, scale = self.stage(img_bgr)
        t1 = time.perf_counter()
        heads, ts = session.run_yolo_monolithic(None, return_timestamps=True)
        t2 = time.perf_counter()
        if not heads["heads_present"]:
            raise RuntimeError(f"{self.model_path}: the egress carries no heads ({heads['head_status']})")
        detections = self.postprocess(self.decode(heads, heads["scales"], conf), pad, scale, conf, iou_thres)
        t3 = time.perf_counter()
        return detections, WorldTimings(preprocess_ms=(t1 - t0) * 1e3, npu_forward_ms=(t2 - t1) * 1e3,
                                        dispatch_ms=ts["npu_ms"], host_ms=float(getattr(session, "last_host_ms", 0.0)),
                                        readback_ms=ts["readback_ms"], postprocess_ms=(t3 - t2) * 1e3,
                                        glass_to_glass_ms=(t3 - t0) * 1e3)

    def close(self) -> None:
        if getattr(self, "session", None) is not None:
            self.session.close()
            self.session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
