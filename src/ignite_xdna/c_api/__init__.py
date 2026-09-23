# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignite_xdna/c_api/__init__.py

Python ctypes wrapper for libignite_xdna native shared library.
Provides zero-overhead native C-API bindings to run compiled .ignite models on AMD Phoenix NPU.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


class IgniteDetectionStruct(ctypes.Structure):
    _fields_ = [
        ("x0", ctypes.c_float),
        ("y0", ctypes.c_float),
        ("w", ctypes.c_float),
        ("h", ctypes.c_float),
        ("score", ctypes.c_float),
        ("class_id", ctypes.c_int),
        ("class_name", ctypes.c_char * 32),
    ]


class IgniteTimingsStruct(ctypes.Structure):
    _fields_ = [
        ("preprocess_ms", ctypes.c_double),
        ("npu_exec_ms", ctypes.c_double),
        ("postprocess_ms", ctypes.c_double),
        ("glass_to_glass_ms", ctypes.c_double),
    ]


class NativeIgniteEngine:
    """Wrapper around libignite_xdna shared library."""

    def __init__(
        self,
        model_path: str | Path,
        device_id: int = 0,
        dll_path: Optional[str | Path] = None,
        max_dets: int = 300,
    ):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if dll_path is None:
            # Search standard build locations
            candidates = [
                Path("build_native/Release/libignite_xdna.dll"),
                Path("build/Release/libignite_xdna.dll"),
                Path("build/libignite_xdna.dll"),
                Path("libignite_xdna.dll"),
            ]
            for cand in candidates:
                if cand.exists():
                    dll_path = cand.resolve()
                    break

        if dll_path is None or not Path(dll_path).exists():
            raise FileNotFoundError("Could not find libignite_xdna.dll")

        # Add XRT bin to DLL directory if on Windows
        xrt_bin = Path(r"C:\Xilinx\XRT\xrt_sdk\xrt\bin")
        if xrt_bin.exists() and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(xrt_bin))
            except Exception:
                pass

        self.lib = ctypes.CDLL(str(dll_path))

        # Declare C-ABI function signatures
        self.lib.ignite_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.ignite_load.restype = ctypes.c_void_p

        self.lib.ignite_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(IgniteDetectionStruct),
            ctypes.c_int,
        ]
        self.lib.ignite_run.restype = ctypes.c_int

        self.lib.ignite_free.argtypes = [ctypes.c_void_p]
        self.lib.ignite_free.restype = None

        self.lib.ignite_set_thresholds.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.lib.ignite_set_thresholds.restype = None

        self.lib.ignite_get_last_timings.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(IgniteTimingsStruct),
        ]
        self.lib.ignite_get_last_timings.restype = None

        self.lib.ignite_load_reference_heads.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.lib.ignite_load_reference_heads.restype = ctypes.c_int

        self.lib.ignite_run_async.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.ignite_run_async.restype = ctypes.c_int

        self.lib.ignite_wait.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(IgniteDetectionStruct),
            ctypes.c_int,
        ]
        self.lib.ignite_wait.restype = ctypes.c_int

        self.lib.ignite_get_last_error.argtypes = []
        self.lib.ignite_get_last_error.restype = ctypes.c_char_p

        # Initialize engine handle
        self.handle = self.lib.ignite_load(
            str(self.model_path).encode("utf-8"), device_id
        )
        if not self.handle:
            err = self.lib.ignite_get_last_error()
            err_msg = err.decode("utf-8") if err else "Unknown error"
            raise RuntimeError(f"ignite_load failed: {err_msg}")

        self._max_dets = max_dets
        self._det_arr = (IgniteDetectionStruct * self._max_dets)()

    def set_max_detections(self, max_dets: int) -> None:
        if max_dets > 0:
            self._max_dets = max_dets
            self._det_arr = (IgniteDetectionStruct * self._max_dets)()

    def set_thresholds(self, conf_thres: float = 0.25, iou_thres: float = 0.50) -> None:
        if self.handle:
            self.lib.ignite_set_thresholds(
                self.handle, ctypes.c_float(conf_thres), ctypes.c_float(iou_thres)
            )

    def load_reference_heads(self, heads_path: str | Path) -> None:
        if self.handle:
            rc = self.lib.ignite_load_reference_heads(
                self.handle, str(heads_path).encode("utf-8")
            )
            if rc != 0:
                err = self.lib.ignite_get_last_error()
                err_msg = err.decode("utf-8") if err else "Unknown error"
                raise RuntimeError(f"ignite_load_reference_heads failed: {err_msg}")

    def run(
        self, img_bgr: np.ndarray
    ) -> Tuple[List[dict], dict]:
        """Runs native inference on BGR image."""
        if not self.handle:
            raise RuntimeError("Engine has been closed.")

        if not img_bgr.flags["C_CONTIGUOUS"]:
            img_bgr = np.ascontiguousarray(img_bgr)

        h, w = img_bgr.shape[:2]
        stride = img_bgr.strides[0]
        data_ptr = img_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        num_dets = self.lib.ignite_run(
            self.handle,
            data_ptr,
            w,
            h,
            stride,
            self._det_arr,
            self._max_dets,
        )
        if num_dets < 0:
            err = self.lib.ignite_get_last_error()
            err_msg = err.decode("utf-8") if err else "Unknown error"
            raise RuntimeError(f"ignite_run failed ({num_dets}): {err_msg}")

        timings_struct = IgniteTimingsStruct()
        self.lib.ignite_get_last_timings(self.handle, ctypes.byref(timings_struct))

        timings = {
            "preprocess_ms": timings_struct.preprocess_ms,
            "npu_exec_ms": timings_struct.npu_exec_ms,
            "postprocess_ms": timings_struct.postprocess_ms,
            "glass_to_glass_ms": timings_struct.glass_to_glass_ms,
        }

        detections = []
        for i in range(num_dets):
            d = self._det_arr[i]
            detections.append({
                "x0": float(d.x0),
                "y0": float(d.y0),
                "w": float(d.w),
                "h": float(d.h),
                "score": float(d.score),
                "class_id": int(d.class_id),
                "class_name": d.class_name.decode("utf-8", errors="ignore"),
            })

        return detections, timings

    def run_async(self, img_bgr: np.ndarray) -> int:
        """Asynchronously enqueues an inference frame and returns the monotonic ticket."""
        if not self.handle:
            raise RuntimeError("Engine has been closed.")

        if not img_bgr.flags["C_CONTIGUOUS"]:
            img_bgr = np.ascontiguousarray(img_bgr)

        h, w = img_bgr.shape[:2]
        stride = img_bgr.strides[0]
        data_ptr = img_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        ticket = ctypes.c_uint64(0)
        rc = self.lib.ignite_run_async(
            self.handle,
            data_ptr,
            w,
            h,
            stride,
            ctypes.byref(ticket),
        )
        if rc != 0:
            err = self.lib.ignite_get_last_error()
            err_msg = err.decode("utf-8") if err else "Unknown error"
            raise RuntimeError(f"ignite_run_async failed ({rc}): {err_msg}")

        return int(ticket.value)

    def wait(self, ticket: int) -> Tuple[List[dict], dict]:
        """Waits for specified ticket to finish and returns its detections and timings."""
        if not self.handle:
            raise RuntimeError("Engine has been closed.")

        num_dets = self.lib.ignite_wait(
            self.handle,
            ctypes.c_uint64(ticket),
            self._det_arr,
            self._max_dets,
        )
        if num_dets < 0:
            err = self.lib.ignite_get_last_error()
            err_msg = err.decode("utf-8") if err else "Unknown error"
            raise RuntimeError(f"ignite_wait failed ({num_dets}): {err_msg}")

        timings_struct = IgniteTimingsStruct()
        self.lib.ignite_get_last_timings(self.handle, ctypes.byref(timings_struct))

        timings = {
            "preprocess_ms": timings_struct.preprocess_ms,
            "npu_exec_ms": timings_struct.npu_exec_ms,
            "postprocess_ms": timings_struct.postprocess_ms,
            "glass_to_glass_ms": timings_struct.glass_to_glass_ms,
        }

        detections = []
        for i in range(num_dets):
            d = self._det_arr[i]
            detections.append({
                "x0": float(d.x0),
                "y0": float(d.y0),
                "w": float(d.w),
                "h": float(d.h),
                "score": float(d.score),
                "class_id": int(d.class_id),
                "class_name": d.class_name.decode("utf-8", errors="ignore"),
            })

        return detections, timings

    def close(self) -> None:
        if self.handle:
            self.lib.ignite_free(self.handle)
            self.handle = None

    def __enter__(self) -> NativeIgniteEngine:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
