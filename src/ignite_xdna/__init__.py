# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
ignite-xdna: Bare-metal AIE2 vector compute engine and compiler lowering for AMD XDNA1 NPU.
"""

__version__ = "0.2.0"

from . import compiler
from . import runtime
from . import pipelines
from .runtime.session import InferenceSession, RunHandle
from .pipelines.yolo_pipeline import YoloPipeline

__all__ = ["compiler", "runtime", "pipelines", "InferenceSession", "RunHandle", "YoloPipeline", "__version__"]
