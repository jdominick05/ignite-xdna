# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
ignite_xdna.pipelines: Production end-to-end vision pipelines for AMD XDNA1 NPU.
"""

from .yolo_pipeline import YoloPipeline, YoloDetection, PipelineTimings

__all__ = ["YoloPipeline", "YoloDetection", "PipelineTimings"]
