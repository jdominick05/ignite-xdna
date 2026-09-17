# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/pipelines/yolo_pipeline.py

End-to-End Streaming Object Detection Pipeline for YOLOv8n on AMD Phoenix XDNA1 NPU.
Integrates:
  Stage 1: Zero-copy OpenCV letterboxing + INT8 quant-scaling (~1.88 ms)
  Stage 2: Monolithic 3-stage forward pass on Device 0 silicon (1.732 ms, 0 intermediate DDR bytes)
  Stage 3: CPU postprocessing with DFL decode + batched NMS (one native call for int8 heads)

Supports both synchronous execution (predict_sync) and 3-stage overlapped asynchronous
streaming pipelining (run_pipelined_stream) across bounded queues (maxsize=2).
"""

import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from ignite_xdna.runtime.session import InferenceSession
from ignite_xdna.runtime.driver import setup_xrt_environment, get_repo_root
from . import decode_native
from .preprocess import FusedPreprocessor

_log = logging.getLogger(__name__)

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

STRIDES = (8, 16, 32)
REG_MAX = 16
NUM_CLASSES = 80


@dataclass
class YoloDetection:
    """Represents a single detected object."""
    x0: float
    y0: float
    w: float
    h: float
    score: float
    class_id: int
    class_name: str


@dataclass
class PipelineTimings:
    """Detailed stage latencies for a single frame (in milliseconds).

    ``head_source`` says where the decoded boxes came from: ``"npu"`` (heads
    unpacked from the device egress), ``"oracle"`` (the ONNX Runtime CPU pass
    over the cut model) or ``"none"`` (no head tensors were available, so the
    frame has no detections by construction).
    """
    preprocess_ms: float
    npu_forward_ms: float
    postprocess_ms: float
    glass_to_glass_ms: float
    head_source: str = "none"


class SequencedQueue:
    """
    Thread-safe bounded priority queue guaranteeing strictly in-order FIFO consumption
    across concurrent multi-worker producers.
    """

    def __init__(self, maxsize: int = 4):
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._items: Dict[int, Any] = {}
        self._next_get = 0
        self._closed = False

    def put(self, seq_idx: int, item: Any):
        with self._cond:
            while seq_idx >= self._next_get + self.maxsize and not self._closed:
                self._cond.wait()
            self._items[seq_idx] = item
            self._cond.notify_all()

    def get(self) -> Optional[Tuple[int, Any]]:
        with self._cond:
            while self._next_get not in self._items and not self._closed:
                self._cond.wait()
            if self._next_get not in self._items:
                return None
            item = self._items.pop(self._next_get)
            seq = self._next_get
            self._next_get += 1
            self._cond.notify_all()
            return seq, item

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class YoloDecoder:
    """
    Device-free half of the pipeline: anchor grids plus the DFL decode + NMS
    postprocess. Instantiable without an NPU so the decode can be verified
    offline; ``YoloPipeline`` inherits it.
    """

    def __init__(self, imgsz: int = 640, conf_thres: float = 0.25, iou_thres: float = 0.50,
                 native_decode: bool = True):
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # Pre-cache anchor grids and strides for fast vectorized DFL decode
        self._anchors, self._strides = self._build_anchors_and_strides(self.imgsz, STRIDES)
        # int8 heads decode in one native call (decode_native.c, identical detections); float heads,
        # native_decode=False and a missing library keep the numpy path below.
        self._native = (decode_native.for_grid(self._anchors, self._strides, NUM_CLASSES)
                        if native_decode else None)

    @property
    def uses_native_decode(self) -> bool:
        """True when int8 heads are decoded by the native library rather than numpy."""
        return self._native is not None

    @staticmethod
    def _build_anchors_and_strides(imgsz: int, strides: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Precomputes (1, 2, N) anchor centers and (1, N) stride vectors."""
        pts, sts = [], []
        for s in strides:
            g = imgsz // s
            c = np.arange(g, dtype=np.float32) + 0.5
            yy, xx = np.meshgrid(c, c, indexing="ij")
            pts.append(np.stack([xx.ravel(), yy.ravel()], 0))
            sts.append(np.full(g * g, float(s), np.float32))
        return np.concatenate(pts, 1)[None], np.concatenate(sts)[None]

    def postprocess(
        self,
        heads: Union[Dict[str, np.ndarray], List[np.ndarray]],
        pad: Tuple[int, int],
        scale: float,
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
    ) -> List[YoloDetection]:
        """
        Stage 3: CPU postprocessing: DFL softmax decode and batched NMS.
        Decodes box coordinates, projects against anchor grids, prunes via inverse-sigmoid threshold,
        and applies non-maximum suppression. int8 heads with scales go through one native call
        (``decode_native``) that returns the same detections as the numpy code below, float for float;
        float heads, ``native_decode=False`` or a missing native library use numpy.

        ``heads`` is either the six float tensors (dict keyed ``p3_box … p5_cls``
        or a list ordered box P3/P4/P5 then cls P3/P4/P5) or, from
        ``InferenceSession.run_yolo_monolithic``, int8 egress views with a
        ``scales`` entry ``{name: (scale, zero_point)}``. int8 heads are pruned
        in the int8 domain (the confidence threshold is mapped to a quantized
        logit) and only the surviving candidates are dequantized. A head that is
        ``None`` means no head tensor exists for this frame: the result is empty.
        """
        conf_t = conf_thres if conf_thres is not None else self.conf_thres
        iou_t = iou_thres if iou_thres is not None else self.iou_thres

        # Unpack head tensors
        scales = None
        cls_max = None  # optional {p*_cls: int8 per-anchor class maxima} from a graph-engine session
        if isinstance(heads, dict):
            box_f = [heads.get("p3_box"), heads.get("p4_box"), heads.get("p5_box")]
            cls_f = [heads.get("p3_cls"), heads.get("p4_cls"), heads.get("p5_cls")]
            scales = heads.get("scales")
            cls_max = heads.get("cls_max")
        else:
            box_f = list(heads[:3])
            cls_f = list(heads[3:])

        if any(b is None for b in box_f) or any(c is None for c in cls_f):
            return []

        if self._native is not None and scales is not None:
            # One native call from the prune to NMS; None means these heads need the numpy path.
            decoded = self._native.decode(box_f, cls_f, cls_max, scales, pad, scale, conf_t, iou_t)
            if decoded is not None:
                boxes, scores, classes = decoded
                return [
                    YoloDetection(
                        x0=boxes[4 * i], y0=boxes[4 * i + 1], w=boxes[4 * i + 2], h=boxes[4 * i + 3],
                        score=scores[i], class_id=cid,
                        class_name=COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}",
                    )
                    for i, cid in enumerate(classes)
                ]

        # 1. Fast Inverse-Sigmoid Confidence Pruning per head (avoids 2.15 MB box concatenation)
        c_clamped = min(max(float(conf_t), 1e-12), 1.0 - 1e-12)
        logit_t = np.log(c_clamped / (1.0 - c_clamped))

        offsets = [0, 6400, 8000]
        head_names = (("p3_box", "p3_cls"), ("p4_box", "p4_cls"), ("p5_box", "p5_cls"))
        surviving_boxes = []
        surviving_cls = []
        surviving_indices = []

        for h_idx, (b, c) in enumerate(zip(box_f, cls_f)):
            if b is not None and b.dtype == np.uint8 and b.ndim == 3:
                # Fallback: convert uint8 C8 blocked heads to int8 NCHW if native decode is bypassed
                h_c = 4 * REG_MAX
                h_blocks = (h_c + 7) // 8
                n_anc = b.shape[1]
                b_chw = np.transpose(b[:h_blocks].reshape(h_blocks, n_anc, 8), (0, 2, 1)).reshape(-1, n_anc)[:h_c]
                b = (b_chw ^ 0x80).view(np.int8)
                c_blocks = (NUM_CLASSES + 7) // 8
                c_chw = np.transpose(c[:c_blocks].reshape(c_blocks, n_anc, 8), (0, 2, 1)).reshape(-1, n_anc)[:NUM_CLASSES]
                c = (c_chw ^ 0x80).view(np.int8)

            if c.dtype == np.int8:
                if scales is None:
                    raise ValueError("int8 head tensors need a 'scales' entry {name: (scale, zero_point)}")
                s_b, zp_b = scales[head_names[h_idx][0]]
                s_c, zp_c = scales[head_names[h_idx][1]]
                # logit > logit_t  <=>  (q - zp) * s > logit_t  <=>  q > logit_t / s + zp
                q_t = logit_t / float(s_c) + int(zp_c)
                c_flat = c.reshape(NUM_CLASSES, -1)
                c_max = cls_max.get(head_names[h_idx][1]) if cls_max else None
                keep_local = np.flatnonzero((c_max if c_max is not None else c_flat.max(0)) > q_t)
                if keep_local.size > 0:
                    b_flat = b.reshape(4 * REG_MAX, -1)
                    surviving_boxes.append(
                        (b_flat[:, keep_local].astype(np.float32) - np.float32(zp_b)) * np.float32(s_b))
                    surviving_cls.append(
                        (c_flat[:, keep_local].astype(np.float32) - np.float32(zp_c)) * np.float32(s_c))
                    surviving_indices.append(keep_local + offsets[h_idx])
                continue

            c_flat = c.reshape(NUM_CLASSES, -1)
            keep_local = np.flatnonzero(c_flat.max(0) > logit_t)
            if keep_local.size > 0:
                b_flat = b.reshape(4 * REG_MAX, -1)
                surviving_boxes.append(b_flat[:, keep_local])
                surviving_cls.append(c_flat[:, keep_local])
                surviving_indices.append(keep_local + offsets[h_idx])

        if not surviving_indices:
            return []

        box_kept = np.concatenate(surviving_boxes, axis=1)[None].astype(np.float32, copy=False)
        cls_kept = np.concatenate(surviving_cls, axis=1)[None].astype(np.float32, copy=False)
        keep = np.concatenate(surviving_indices)

        anc_kept = self._anchors[:, :, keep]
        strides_kept = self._strides[:, keep]

        # 3. DFL Softmax Projection (16 bins -> expected distance)
        n_cand = box_kept.shape[2]
        b_reshaped = box_kept.reshape(1, 4, REG_MAX, n_cand).transpose(0, 2, 1, 3).copy()
        b_reshaped -= b_reshaped.max(axis=1, keepdims=True)
        np.exp(b_reshaped, out=b_reshaped)
        b_reshaped /= b_reshaped.sum(axis=1, keepdims=True)

        bins = np.arange(REG_MAX, dtype=np.float32).reshape(1, REG_MAX, 1, 1)
        ltrb = (b_reshaped * bins).sum(1)  # (1, 4, n) distances: left, top, right, bottom

        # 4. Box reconstruction in letterboxed pixel coordinates
        x1y1 = anc_kept - ltrb[:, 0:2]
        x2y2 = anc_kept + ltrb[:, 2:4]
        cxcy = (x1y1 + x2y2) * 0.5
        wh = x2y2 - x1y1
        xywh = (np.concatenate([cxcy, wh], 1) * strides_kept[:, None])[0].T  # (n_cand, 4)

        # 5. Sigmoid class probabilities
        probs = 1.0 / (1.0 + np.exp(-cls_kept[0].T))  # (n_cand, 80)
        best_cls = probs.argmax(axis=1)
        scores = probs[np.arange(len(best_cls)), best_cls]

        # Filter by conf_thres
        mask = scores >= conf_t
        xywh = xywh[mask]
        scores = scores[mask]
        best_cls = best_cls[mask]
        if len(xywh) == 0:
            return []

        # Map boxes back to original image dimensions
        boxes_orig = xywh.copy()
        boxes_orig[:, 0] = (boxes_orig[:, 0] - boxes_orig[:, 2] / 2.0 - pad[1]) / scale
        boxes_orig[:, 1] = (boxes_orig[:, 1] - boxes_orig[:, 3] / 2.0 - pad[0]) / scale
        boxes_orig[:, 2] /= scale
        boxes_orig[:, 3] /= scale

        # 6. Batched NMS
        b_list = boxes_orig.tolist()
        s_list = scores.tolist()
        c_list = best_cls.tolist()
        indices = cv2.dnn.NMSBoxesBatched(b_list, s_list, c_list, conf_t, iou_t)
        if len(indices) == 0:
            return []

        indices = np.array(indices).reshape(-1)
        detections: List[YoloDetection] = []
        for i in indices:
            cid = int(best_cls[i])
            detections.append(
                YoloDetection(
                    x0=float(boxes_orig[i, 0]),
                    y0=float(boxes_orig[i, 1]),
                    w=float(boxes_orig[i, 2]),
                    h=float(boxes_orig[i, 3]),
                    score=float(scores[i]),
                    class_id=cid,
                    class_name=COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}",
                )
            )

        return sorted(detections, key=lambda d: -d.score)


class YoloPipeline(YoloDecoder):
    """
    High-Performance Streaming Pipeline for YOLOv8n Object Detection on AMD Phoenix Silicon.
    """

    def __init__(
        self,
        model_path_or_bundle: Optional[Union[str, Path]] = None,
        device_index: int = 0,
        conf_thres: float = 0.25,
        iou_thres: float = 0.50,
        imgsz: int = 640,
        enable_pipelining: bool = True,
        native_decode: bool = True,
    ):
        super().__init__(imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres, native_decode=native_decode)
        self.device_index = device_index
        self.enable_pipelining = enable_pipelining
        # Set by predict_sync: the session's HeadStatus reason for the last frame
        self.last_head_status: Optional[str] = None
        self._warned_heads_absent = False

        setup_xrt_environment()
        repo_root = get_repo_root()

        if model_path_or_bundle is None:
            ignite_cand = repo_root / "build" / "yolov8n_full.ignite"
            if not ignite_cand.exists():
                ignite_cand = repo_root / "build" / "yolov8n.ignite"
            if ignite_cand.exists():
                cand = ignite_cand
            else:
                cand = repo_root / "models" / "yolov8n_cut_xint8.onnx"
                if not cand.exists():
                    cand = repo_root / "models" / "yolov8n.onnx"
            self.model_path = cand
        else:
            self.model_path = Path(model_path_or_bundle)

        # 1. Initialize physical silicon monolithic session (single-dispatch fast-path)
        if str(self.model_path).endswith(".ignite"):
            self.session = InferenceSession.from_file(
                self.model_path,
                device_index=self.device_index,
                single_dispatch=True,
            )
        else:
            self.session = InferenceSession(
                model_path_or_bundle=self.model_path,
                device_index=self.device_index,
                enable_monolithic=True,
                full_yolo=True,
                single_dispatch=True,
            )

        # 2. Anchor grids and strides come from YoloDecoder.__init__

        # 3. Pre-load reference cut model for exact head outputs if required for visual verification
        # The oracle is the model the container was compiled from (manifest model_name), else yolov8n.
        compiled_from = (getattr(self.session, "ignite_manifest", None) or {}).get("model_name")
        cut_cand = repo_root / "models" / f"{compiled_from}.onnx" if compiled_from else None
        if cut_cand is None or not cut_cand.exists():
            cut_cand = repo_root / "models" / "yolov8n_cut_xint8.onnx"
        self._ort_cut_sess = None
        if cut_cand.exists():
            try:
                cm = onnx.load(str(cut_cand))
                self._ort_cut_sess = ort.InferenceSession(cm.SerializeToString(), providers=["CPUExecutionProvider"])
                self._ort_cut_input_name = self._ort_cut_sess.get_inputs()[0].name
            except Exception:
                self._ort_cut_sess = None

        # Pre-allocate scratch canvas and input buffer to enable zero-copy preprocessing
        self._canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        self._input_chw = np.empty((1, 3, self.imgsz, self.imgsz), dtype=np.int8)

        # Fused C/SIMD zero-copy preprocessor (< 0.80 ms)
        self.preprocessor = FusedPreprocessor(imgsz=self.imgsz)

    def preprocess(
        self,
        img_bgr: np.ndarray,
        out_buf: Optional[np.ndarray] = None,
        canvas: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """
        Stage 1: Fused Zero-Copy Ingress Preprocessing (~0.28 - 0.35 ms).
        Combines letterbox padding, bit-exact Q11 bilinear interpolation, BGR->RGB planar
        transposition, and uint8->int8 scale conversion in a single pass directly into DMA memory.
        Returns:
            quant_tensor: int8 [1, 3, imgsz, imgsz] ready for direct DMA ingress
            pad: (top, left) padding pixels
            scale: aspect scaling factor
        """
        target = self._input_chw if out_buf is None else out_buf
        return self.preprocessor.preprocess(img_bgr, out_buf=target)

    def forward_npu(
        self,
        quant_tensor: Optional[np.ndarray],
        return_timestamps: bool = False,
        unswizzle: bool = True,
    ) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Any]]:
        """
        Stage 2: Monolithic 3-Stage Silicon Forward Pass on Phoenix Device 0 (~1.732 ms).
        Chains Backbone (0.709 ms) -> Neck (0.407 ms) -> Detect Heads (0.616 ms).
        Zero intermediate host DDR traffic throughout the entire network.
        """
        return self.session.run_yolo_monolithic(
            quant_tensor,
            unswizzle=unswizzle,
            return_timestamps=return_timestamps,
        )

    def predict_sync(
        self,
        img_bgr: np.ndarray,
        use_oracle_for_boxes: bool = True,
    ) -> Tuple[List[YoloDetection], PipelineTimings]:
        """
        Synchronous single-frame inference returning detections and glass-to-glass timing breakdown.
        """
        # Oracle-free frames on a graph-engine session skip the int8 host tensor: the
        # native preprocessor quantizes straight into the NPU workspace input plane.
        direct = not use_oracle_for_boxes and bool(getattr(self.session, "direct_ingress", False))
        t0 = time.perf_counter()
        if direct:
            pad, scale = self.session.stage_image(img_bgr)
            quant_tensor = None
        else:
            quant_tensor, pad, scale = self.preprocess(img_bgr)
        t1 = time.perf_counter()

        # When not using the CPU oracle and native decode is available, we skip unswizzling
        # to decode directly from channel-blocked egress buffers.
        unswizzle = use_oracle_for_boxes or not self.uses_native_decode
        heads, hw_ts = self.forward_npu(
            quant_tensor,
            return_timestamps=True,
            unswizzle=unswizzle,
        )
        t2 = time.perf_counter()

        heads_present = bool(heads.get("heads_present", False))
        self.last_head_status = heads.get("head_status")

        if use_oracle_for_boxes and self._ort_cut_sess is not None:
            # Reference boxes from the ONNX Runtime CPU pass over the cut model (~43 ms)
            x_float = (quant_tensor.astype(np.float32) + 128.0) / 255.0
            cut_outs = self._ort_cut_sess.run(None, {self._ort_cut_input_name: x_float})
            dets = self.postprocess(cut_outs, pad, scale)
            head_source = "oracle"
        elif heads_present:
            dets = self.postprocess(heads, pad, scale)
            head_source = "npu"
        else:
            # No head tensors exist for this frame: report it once instead of decoding zeros
            dets = []
            head_source = "none"
            if not self._warned_heads_absent:
                self._warned_heads_absent = True
                _log.warning(
                    "NPU egress carries no detect heads, so predict_sync(use_oracle_for_boxes=False) "
                    "returns no detections: %s", self.last_head_status)
        t3 = time.perf_counter()

        timings = PipelineTimings(
            preprocess_ms=(t1 - t0) * 1000.0,
            npu_forward_ms=(t2 - t1) * 1000.0,
            postprocess_ms=(t3 - t2) * 1000.0,
            glass_to_glass_ms=(t3 - t0) * 1000.0,
            head_source=head_source,
        )
        return dets, timings

    def run_pipelined_stream(
        self,
        frames: List[np.ndarray],
        warmup: int = 50,
        iterations: int = 500,
        queue_size: int = 2,
        num_ingress_workers: int = 2,
        num_postprocess_workers: int = 2,
    ) -> Dict[str, Any]:
        """
        Executes sustained asynchronous streaming across dual-worker ingress and multi-stage overlap:
          Stage 1: Dual-worker Ingress Preprocessing (fused letterbox + INT8 quant-scaling)
          Stage 2: Monolithic NPU forward pass on physical Device 0 silicon
          Stage 3: Vectorized CPU postprocessing (DFL decode + batched NMS)
        """
        num_frames = len(frames)
        total_runs = warmup + iterations

        # Ping-pong double-buffering structures for dual-worker ingress overlap
        buf_0 = np.empty((1, 3, self.imgsz, self.imgsz), dtype=np.int8)
        buf_1 = np.empty((1, 3, self.imgsz, self.imgsz), dtype=np.int8)

        ready_0 = threading.Event()
        ready_1 = threading.Event()
        done_0 = threading.Event()
        done_1 = threading.Event()
        done_0.set()
        done_1.set()

        meta_0 = [None, None, 0.0]
        meta_1 = [None, None, 0.0]
        prep_times_0 = np.empty(total_runs, dtype=np.float64)
        prep_times_1 = np.empty(total_runs, dtype=np.float64)

        def worker_even():
            for idx in range(0, total_runs, 2):
                done_0.wait()
                done_0.clear()
                t0 = time.perf_counter()
                _, pad, scale = self.preprocess(frames[idx % num_frames], out_buf=buf_0)
                t1 = time.perf_counter()
                prep_times_0[idx] = (t1 - t0) * 1e6
                meta_0[0] = pad
                meta_0[1] = scale
                meta_0[2] = t0
                ready_0.set()

        def worker_odd():
            for idx in range(1, total_runs, 2):
                done_1.wait()
                done_1.clear()
                t0 = time.perf_counter()
                _, pad, scale = self.preprocess(frames[idx % num_frames], out_buf=buf_1)
                t1 = time.perf_counter()
                prep_times_1[idx] = (t1 - t0) * 1e6
                meta_1[0] = pad
                meta_1[1] = scale
                meta_1[2] = t0
                ready_1.set()

        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        old_switch = sys.getswitchinterval()
        try:
            sys.setswitchinterval(0.0005)

            t_even = threading.Thread(target=worker_even, name="Stage1_Ingress_0")
            t_odd = threading.Thread(target=worker_odd, name="Stage1_Ingress_1")
            t_even.start()
            t_odd.start()

            prep_latencies_us = np.empty(iterations, dtype=np.float64)
            npu_latencies_us = np.empty(iterations, dtype=np.float64)
            post_latencies_us = np.empty(iterations, dtype=np.float64)
            g2g_latencies_us = np.empty(iterations, dtype=np.float64)
            wall_g2g_latencies_us = np.empty(iterations, dtype=np.float64)

            t_stream_start = time.perf_counter()
            t_steady_start = 0.0

            for idx in range(total_runs):
                if idx == warmup:
                    t_steady_start = time.perf_counter()

                if idx % 2 == 0:
                    ready_0.wait()
                    ready_0.clear()
                    pad, scale, t_cap = meta_0[0], meta_0[1], meta_0[2]
                    t_npu_0 = time.perf_counter()
                    heads = self.forward_npu(buf_0, unswizzle=not self.uses_native_decode)
                    t_npu_1 = time.perf_counter()
                    done_0.set()
                    p_us = prep_times_0[idx]
                else:
                    ready_1.wait()
                    ready_1.clear()
                    pad, scale, t_cap = meta_1[0], meta_1[1], meta_1[2]
                    t_npu_0 = time.perf_counter()
                    heads = self.forward_npu(buf_1, unswizzle=not self.uses_native_decode)
                    t_npu_1 = time.perf_counter()
                    done_1.set()
                    p_us = prep_times_1[idx]

                t_post_0 = time.perf_counter()
                dets = self.postprocess(heads, pad, scale)
                t_post_1 = time.perf_counter()

                n_us = (t_npu_1 - t_npu_0) * 1e6
                post_us = (t_post_1 - t_post_0) * 1e6
                wall_g2g_us = (t_post_1 - t_cap) * 1e6

                if idx >= warmup:
                    out_idx = idx - warmup
                    prep_latencies_us[out_idx] = p_us
                    npu_latencies_us[out_idx] = n_us
                    post_latencies_us[out_idx] = post_us
                    g2g_latencies_us[out_idx] = p_us + n_us + post_us
                    wall_g2g_latencies_us[out_idx] = wall_g2g_us

            t_stream_end = time.perf_counter()
            t_even.join()
            t_odd.join()
        finally:
            sys.setswitchinterval(old_switch)

        elapsed_sec = t_stream_end - (t_steady_start if t_steady_start > 0 else t_stream_start)
        sustained_fps = iterations / elapsed_sec if elapsed_sec > 0 else 0.0

        g2g_arr = g2g_latencies_us / 1000.0  # to ms
        wall_g2g_arr = wall_g2g_latencies_us / 1000.0
        npu_arr = npu_latencies_us / 1000.0
        prep_arr = prep_latencies_us / 1000.0
        post_arr = post_latencies_us / 1000.0

        return {
            "iterations": iterations,
            "warmup": warmup,
            "elapsed_sec": elapsed_sec,
            "sustained_fps": sustained_fps,
            "glass_to_glass_ms": {
                "mean": float(np.mean(g2g_arr)),
                "median": float(np.median(g2g_arr)),
                "min": float(np.min(g2g_arr)),
                "max": float(np.max(g2g_arr)),
                "p95": float(np.percentile(g2g_arr, 95)),
                "p99": float(np.percentile(g2g_arr, 99)),
            },
            "wall_glass_to_glass_ms": {
                "mean": float(np.mean(wall_g2g_arr)),
                "median": float(np.median(wall_g2g_arr)),
                "p95": float(np.percentile(wall_g2g_arr, 95)),
            },
            "stage_breakdown_ms": {
                "preprocess_mean": float(np.mean(prep_arr)),
                "npu_mean": float(np.mean(npu_arr)),
                "postprocess_mean": float(np.mean(post_arr)),
            },
        }

    def close(self):
        """Releases hardware context and session buffers."""
        if hasattr(self, "session") and self.session is not None:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
