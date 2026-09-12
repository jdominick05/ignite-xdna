# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
ignite-xdna root forwarder: Exposes src/ignite_xdna directly in workspace context.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

_src_pkg = _src_dir / "ignite_xdna"
__path__ = [str(_src_pkg)]

_init_file = _src_pkg / "__init__.py"
with open(_init_file, "r", encoding="utf-8") as _f:
    exec(_f.read(), globals())
