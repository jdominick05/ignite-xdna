# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/pipelines/decode_native.py

Native int8 head decode for ``YoloDecoder``: one ctypes call from the class-max prune to the
NMS-ordered detections, in place of about forty numpy calls and ``cv2.dnn.NMSBoxesBatched``.

``decode_native.c`` returns exactly what ``YoloDecoder.postprocess`` returns on int8 heads:
float32 operations run in numpy's order, every exp is read from a table filled here with
``np.exp`` (which gives an element the same bits whatever the array layout), and NMS follows
OpenCV's ``NMSBoxesBatched``. The library is built like ``preprocess_simd.c`` but without
``/fp:fast`` or OpenMP, either of which would change the boxes. When it is missing or cannot be
built, ``for_grid`` returns ``None`` and the decoder keeps its numpy path.
"""

import ctypes
import platform
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .preprocess import _compile_native_dll

ABI = 3
REG_MAX = 16
HEAD_NAMES = (("p3_box", "p3_cls"), ("p4_box", "p4_cls"), ("p5_box", "p5_cls"))
# Anchors per head and each head's first anchor, as YoloDecoder.postprocess indexes them.
HEAD_ANCHORS = (6400, 1600, 400)
HEAD_OFFSETS = (0, 6400, 8000)
TOTAL_ANCHORS = 8400


class _Head(ctypes.Structure):
    _fields_ = [
        ("box", ctypes.c_void_p),
        ("cls", ctypes.c_void_p),
        ("cls_max", ctypes.c_void_p),
        ("dfl_exp", ctypes.c_void_p),
        ("box_val", ctypes.c_void_p),
        ("sigmoid", ctypes.c_void_p),
        ("sigmoid_low", ctypes.c_void_p),
        ("q_threshold", ctypes.c_double),
        ("anchors", ctypes.c_int32),
        ("anchor_offset", ctypes.c_int32),
    ]


class _Decode(ctypes.Structure):
    _fields_ = [
        ("head", _Head * 3),
        ("anchor_x", ctypes.c_void_p),
        ("anchor_y", ctypes.c_void_p),
        ("strides", ctypes.c_void_p),
        ("reg_max", ctypes.c_int32),
        ("num_classes", ctypes.c_int32),
        ("conf", ctypes.c_float),
        ("iou", ctypes.c_float),
    ]


class _HeadC8(ctypes.Structure):
    _fields_ = [
        ("box_c8", ctypes.c_void_p),
        ("cls_c8", ctypes.c_void_p),
        ("cls_max", ctypes.c_void_p),
        ("dfl_exp", ctypes.c_void_p),
        ("box_val", ctypes.c_void_p),
        ("sigmoid", ctypes.c_void_p),
        ("sigmoid_low", ctypes.c_void_p),
        ("q_threshold", ctypes.c_double),
        ("anchors", ctypes.c_int32),
        ("anchor_offset", ctypes.c_int32),
    ]


class _DecodeC8(ctypes.Structure):
    _fields_ = [
        ("head", _HeadC8 * 3),
        ("anchor_x", ctypes.c_void_p),
        ("anchor_y", ctypes.c_void_p),
        ("strides", ctypes.c_void_p),
        ("reg_max", ctypes.c_int32),
        ("num_classes", ctypes.c_int32),
        ("conf", ctypes.c_float),
        ("iou", ctypes.c_float),
    ]


def _load_library() -> Optional[ctypes.CDLL]:
    here = Path(__file__).parent.resolve()
    dll = here / ("decode_native.dll" if platform.system() == "Windows" else "decode_native.so")
    if not dll.exists() and not _compile_native_dll(here / "decode_native.c", dll, "/O2 /fp:precise",
                                                    ["-O3", "-ffp-contract=off"]):
        return None
    try:
        lib = ctypes.CDLL(str(dll))
        lib.yolo_decode_abi.restype = ctypes.c_int
        if lib.yolo_decode_abi() != ABI:
            return None
        lib.yolo_decode_int8.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.yolo_decode_int8.restype = ctypes.c_int
        if hasattr(lib, "yolo_decode_c8_blocks"):
            lib.yolo_decode_c8_blocks.argtypes = [
                ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int32,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ]
            lib.yolo_decode_c8_blocks.restype = ctypes.c_int
        return lib
    except (OSError, AttributeError):
        return None


_LIB = _load_library()


def available() -> bool:
    """True when the native decode library is loaded."""
    return _LIB is not None


def _dequantized_logits(scale: float, zero_point: int) -> np.ndarray:
    return (np.arange(-128, 128).astype(np.float32) - np.float32(zero_point)) * np.float32(scale)


def dfl_exp_table(scale: float, zero_point: int) -> np.ndarray:
    """float32 [256][256]: ``np.exp(v(q) - v(q_max))`` at ``[q_max + 128][q + 128]``, v the dequantized logit."""
    v = _dequantized_logits(scale, zero_point)
    with np.errstate(over="ignore"):  # entries with q > q_max may overflow; the decode never reads them
        table = np.exp(v[None, :] - v[:, None])
    return np.ascontiguousarray(table, dtype=np.float32)


def sigmoid_table(scale: float, zero_point: int) -> np.ndarray:
    """float32 [256]: ``1 / (1 + np.exp(-v(q)))`` at ``[q + 128]``, computed as the numpy decode does."""
    v = _dequantized_logits(scale, zero_point)
    return np.ascontiguousarray(1.0 / (1.0 + np.exp(-v)), dtype=np.float32)


def sigmoid_low_table(sigmoid: np.ndarray) -> Optional[np.ndarray]:
    """int32 [256]: the smallest q whose sigmoid equals that of ``q`` at ``[q + 128]``, or ``None`` if not monotone.

    With the class maximum q_max known, numpy's argmax over float32 probabilities is the first class whose
    logit is at least this value: saturated logits share a probability, and the first of them wins.
    """
    if not np.all(sigmoid[1:] >= sigmoid[:-1]):
        return None
    return np.ascontiguousarray(np.searchsorted(sigmoid, sigmoid, side="left") - 128, dtype=np.int32)


class _Binding:
    """The native argument block for one set of head arrays, scales and thresholds."""

    __slots__ = ("key", "args", "address", "refs")

    def __init__(self, key: Tuple[Any, ...], args: _Decode, refs: List[Any]):
        self.key = key
        self.args = args
        self.address = ctypes.addressof(args)
        self.refs = refs  # keeps the head arrays and tables the pointers refer to alive


class NativeDecode:
    """Decodes int8 heads for one anchor grid; ``decode`` returns ``None`` when the numpy path must run."""

    def __init__(self, lib: ctypes.CDLL, anchors: np.ndarray, strides: np.ndarray, num_classes: int,
                 reg_max: int = REG_MAX):
        self._reg_max = int(reg_max)
        self._lib = lib
        self._fn = lib.yolo_decode_int8
        self._anchor_x = np.ascontiguousarray(anchors[0, 0], dtype=np.float32)
        self._anchor_y = np.ascontiguousarray(anchors[0, 1], dtype=np.float32)
        self._strides = np.ascontiguousarray(strides[0], dtype=np.float32)
        self._num_classes = int(num_classes)
        self._tables: Dict[Tuple[str, float, int], Optional[np.ndarray]] = {}
        self._binding: Optional[_Binding] = None
        self._binding_c8: Optional[_Binding] = None
        self._local = threading.local()

    def _table(self, kind: str, scale: float, zero_point: int) -> Optional[np.ndarray]:
        key = (kind, float(np.float32(scale)), int(zero_point))
        if key not in self._tables:
            if kind == "dfl":
                self._tables[key] = dfl_exp_table(scale, zero_point)
            elif kind == "sigmoid":
                self._tables[key] = sigmoid_table(scale, zero_point)
            elif kind == "box_val":
                # v(q) = (q - zero_point) * scale, the same expression the numpy path applies to
                # the box head. At reg_max 1 the four distances ARE these values, so sharing the
                # helper is what makes native and numpy agree bit for bit.
                self._tables[key] = np.ascontiguousarray(_dequantized_logits(scale, zero_point),
                                                         dtype=np.float32)
            else:
                self._tables[key] = sigmoid_low_table(self._table("sigmoid", scale, zero_point))
        return self._tables[key]

    def _bind(self, key: Tuple[Any, ...], box_f: Sequence[Any], cls_f: Sequence[Any],
              cls_max: Optional[Mapping[str, Any]], scales: Mapping[str, Any], conf_t: float,
              iou_t: float) -> Optional[_Binding]:
        args = _Decode()
        refs: List[Any] = [self._anchor_x, self._anchor_y, self._strides]
        c_clamped = min(max(float(conf_t), 1e-12), 1.0 - 1e-12)
        logit_t = np.log(c_clamped / (1.0 - c_clamped))
        for h, (box_name, cls_name) in enumerate(HEAD_NAMES):
            b, c = box_f[h], cls_f[h]
            for arr, channels in ((b, 4 * self._reg_max), (c, self._num_classes)):
                if (not isinstance(arr, np.ndarray) or arr.dtype != np.int8 or not arr.flags["C_CONTIGUOUS"]
                        or arr.size != channels * HEAD_ANCHORS[h]):
                    return None
            try:
                s_b, zp_b = scales[box_name]
                s_c, zp_c = scales[cls_name]
            except (KeyError, TypeError, ValueError):
                return None
            for s, zp in ((s_b, zp_b), (s_c, zp_c)):
                if not (float(s) > 0.0 and np.isfinite(float(s))) or int(zp) != zp or not -128 <= int(zp) <= 127:
                    return None
            m = cls_max.get(cls_name) if cls_max else None
            if m is not None and (not isinstance(m, np.ndarray) or m.dtype != np.int8
                                  or not m.flags["C_CONTIGUOUS"] or m.size != HEAD_ANCHORS[h]):
                return None
            dfl = self._table("dfl", s_b, int(zp_b))
            bval = self._table("box_val", s_b, int(zp_b))
            sig = self._table("sigmoid", s_c, int(zp_c))
            low = self._table("sigmoid_low", s_c, int(zp_c))
            head = args.head[h]
            head.box = b.ctypes.data
            head.cls = c.ctypes.data
            head.cls_max = m.ctypes.data if m is not None else None
            head.dfl_exp = dfl.ctypes.data
            head.box_val = bval.ctypes.data
            head.sigmoid = sig.ctypes.data
            head.sigmoid_low = low.ctypes.data if low is not None else None
            # The numpy prune's threshold, computed by the same expression.
            head.q_threshold = float(logit_t / float(s_c) + int(zp_c))
            head.anchors = HEAD_ANCHORS[h]
            head.anchor_offset = HEAD_OFFSETS[h]
            refs += [b, c, m, dfl, bval, sig, low]
        args.anchor_x = self._anchor_x.ctypes.data
        args.anchor_y = self._anchor_y.ctypes.data
        args.strides = self._strides.ctypes.data
        args.reg_max = self._reg_max
        args.num_classes = self._num_classes
        args.conf = float(np.float32(conf_t))
        args.iou = float(np.float32(iou_t))
        return _Binding(key, args, refs)

    def _bind_c8(self, key: Tuple[Any, ...], box_c8: Sequence[Any], cls_c8: Sequence[Any],
                 cls_max: Mapping[str, Any], scales: Mapping[str, Any], conf_t: float,
                 iou_t: float) -> Optional[_Binding]:
        args = _DecodeC8()
        refs: List[Any] = [self._anchor_x, self._anchor_y, self._strides]
        c_clamped = min(max(float(conf_t), 1e-12), 1.0 - 1e-12)
        logit_t = np.log(c_clamped / (1.0 - c_clamped))
        cls_blocks = (self._num_classes + 7) // 8
        for h, (box_name, cls_name) in enumerate(HEAD_NAMES):
            b, c = box_c8[h], cls_c8[h]
            n_anc = HEAD_ANCHORS[h]
            for arr, blocks in ((b, (4 * self._reg_max + 7) // 8), (c, cls_blocks)):
                if (not isinstance(arr, np.ndarray) or arr.dtype != np.uint8 or not arr.flags["C_CONTIGUOUS"]
                        or arr.size < blocks * n_anc * 8):
                    return None
            try:
                s_b, zp_b = scales[box_name]
                s_c, zp_c = scales[cls_name]
            except (KeyError, TypeError, ValueError):
                return None
            for s, zp in ((s_b, zp_b), (s_c, zp_c)):
                if not (float(s) > 0.0 and np.isfinite(float(s))) or int(zp) != zp or not -128 <= int(zp) <= 127:
                    return None
            m = cls_max.get(cls_name) if cls_max else None
            if m is None or not isinstance(m, np.ndarray) or m.dtype != np.int8 or not m.flags["C_CONTIGUOUS"] or m.size != n_anc:
                return None
            dfl = self._table("dfl", s_b, int(zp_b))
            bval = self._table("box_val", s_b, int(zp_b))
            sig = self._table("sigmoid", s_c, int(zp_c))
            low = self._table("sigmoid_low", s_c, int(zp_c))
            head = args.head[h]
            head.box_c8 = b.ctypes.data
            head.cls_c8 = c.ctypes.data
            head.cls_max = m.ctypes.data
            head.dfl_exp = dfl.ctypes.data
            head.box_val = bval.ctypes.data
            head.sigmoid = sig.ctypes.data
            head.sigmoid_low = low.ctypes.data if low is not None else None
            head.q_threshold = float(logit_t / float(s_c) + int(zp_c))
            head.anchors = n_anc
            head.anchor_offset = HEAD_OFFSETS[h]
            refs += [b, c, m, dfl, bval, sig, low]
        args.anchor_x = self._anchor_x.ctypes.data
        args.anchor_y = self._anchor_y.ctypes.data
        args.strides = self._strides.ctypes.data
        args.reg_max = self._reg_max
        args.num_classes = self._num_classes
        args.conf = float(np.float32(conf_t))
        args.iou = float(np.float32(iou_t))
        return _Binding(key, args, refs)

    def decode_c8(self, box_c8: Sequence[Any], cls_c8: Sequence[Any], cls_max: Mapping[str, Any],
                  scales: Mapping[str, Any], pad: Tuple[Any, Any], scale: float, conf_t: float,
                  iou_t: float) -> Optional[Tuple[List[float], List[float], List[int]]]:
        """Returns (boxes x0, y0, w, h flattened, scores, class ids) from channel-blocked heads, or ``None``."""
        if not hasattr(self._lib, "yolo_decode_c8_blocks"):
            return None
        maxima = (cls_max.get("p3_cls"), cls_max.get("p4_cls"), cls_max.get("p5_cls"))
        key = ("c8", id(box_c8[0]), id(box_c8[1]), id(box_c8[2]), id(cls_c8[0]), id(cls_c8[1]), id(cls_c8[2]),
               id(maxima[0]), id(maxima[1]), id(maxima[2]), id(scales), conf_t, iou_t)
        binding = self._binding_c8
        if binding is None or binding.key != key:
            binding = self._bind_c8(key, box_c8, cls_c8, cls_max, scales, conf_t, iou_t)
            if binding is None:
                return None
            binding.refs.append(scales)
            self._binding_c8 = binding
        local = self._local
        pointers = getattr(local, "pointers", None)
        if pointers is None:
            local.out = (np.empty(4 * TOTAL_ANCHORS, np.float32), np.empty(TOTAL_ANCHORS, np.float32),
                         np.empty(TOTAL_ANCHORS, np.int32))
            local.pointers = pointers = tuple(a.ctypes.data for a in local.out)
        n = self._lib.yolo_decode_c8_blocks(binding.address, pad[0], pad[1], scale, TOTAL_ANCHORS, *pointers)
        if n < 0:
            return None
        out = local.out
        return out[0][:4 * n].tolist(), out[1][:n].tolist(), out[2][:n].tolist()

    def decode(self, box_f: Sequence[Any], cls_f: Sequence[Any], cls_max: Optional[Mapping[str, Any]],
               scales: Mapping[str, Any], pad: Tuple[Any, Any], scale: float, conf_t: float,
               iou_t: float) -> Optional[Tuple[List[float], List[float], List[int]]]:
        """Returns (boxes x0, y0, w, h flattened, scores, class ids) in detection order, or ``None``."""
        if (len(box_f) == 3 and isinstance(box_f[0], np.ndarray) and box_f[0].dtype == np.uint8
                and cls_max is not None):
            return self.decode_c8(box_f, cls_f, cls_max, scales, pad, scale, conf_t, iou_t)
        maxima = (cls_max.get("p3_cls"), cls_max.get("p4_cls"), cls_max.get("p5_cls")) if cls_max else (None,) * 3
        # Identity of every array and of the scales (the binding holds them, so ids cannot be reused), thresholds.
        key = (id(box_f[0]), id(box_f[1]), id(box_f[2]), id(cls_f[0]), id(cls_f[1]), id(cls_f[2]),
               id(maxima[0]), id(maxima[1]), id(maxima[2]), id(scales), conf_t, iou_t)
        binding = self._binding
        if binding is None or binding.key != key:
            binding = self._bind(key, box_f, cls_f, cls_max, scales, conf_t, iou_t)
            if binding is None:
                return None
            binding.refs.append(scales)
            self._binding = binding
        local = self._local
        pointers = getattr(local, "pointers", None)
        if pointers is None:
            local.out = (np.empty(4 * TOTAL_ANCHORS, np.float32), np.empty(TOTAL_ANCHORS, np.float32),
                         np.empty(TOTAL_ANCHORS, np.int32))
            local.pointers = pointers = tuple(a.ctypes.data for a in local.out)
        n = self._fn(binding.address, pad[0], pad[1], scale, TOTAL_ANCHORS, *pointers)
        if n < 0:
            return None
        out = local.out
        return out[0][:4 * n].tolist(), out[1][:n].tolist(), out[2][:n].tolist()


def for_grid(anchors: np.ndarray, strides: np.ndarray, num_classes: int,
             reg_max: int = REG_MAX) -> Optional[NativeDecode]:
    """A ``NativeDecode`` for a decoder's anchor grid, or ``None`` without the library, for another
    grid, or for a ``reg_max`` the library does not implement (it does 16, the DFL head, and 1,
    a head that regresses the four distances directly)."""
    if _LIB is None or anchors.shape != (1, 2, TOTAL_ANCHORS) or strides.shape != (1, TOTAL_ANCHORS):
        return None
    if int(reg_max) not in (1, REG_MAX):
        return None
    return NativeDecode(_LIB, anchors, strides, num_classes, int(reg_max))
