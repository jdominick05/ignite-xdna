# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
ignite-xdna: Bare-metal AIE2 vector compute engine and compiler lowering for AMD XDNA1 NPU.
"""

__version__ = "0.3.1"

from . import compiler
from . import runtime
from . import pipelines
from .runtime.session import InferenceSession, RunHandle
from .runtime.loader import IgniteEngine, load
from .compiler.serializer import IgniteModelWriter, IgniteModelReader, IgniteHeader
from .pipelines.yolo_pipeline import YoloPipeline

__all__ = [
    "compiler",
    "runtime",
    "pipelines",
    "InferenceSession",
    "RunHandle",
    "IgniteEngine",
    "load",
    "IgniteModelWriter",
    "IgniteModelReader",
    "IgniteHeader",
    "YoloPipeline",
    "__version__",
]
