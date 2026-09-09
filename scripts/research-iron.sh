#!/usr/bin/env bash
# Activate the existing mlir-aie venv for a research child, with worktree-local caches.
#   bash scripts/research-iron.sh kernels/memory_placement/probe.py <args>
# Invoke through research-lowlevel.sh --npu for contention checks and resource bounds.

set -euo pipefail
if [ "${1:-}" = --help ]; then sed -n '2,4s/^# //p' "${BASH_SOURCE[0]}"; exit 0; fi
: "${MLIR_AIE_ROOT:=$HOME/mlir-aie}"
source "$MLIR_AIE_ROOT/ironenv/Scripts/activate"
export MLIR_AIE_INSTALL_DIR="$(cygpath -w "$MLIR_AIE_ROOT/ironenv/Lib/site-packages/mlir_aie")"
export PEANO_INSTALL_DIR="$(cygpath -w "$MLIR_AIE_ROOT/ironenv/Lib/site-packages/llvm-aie")"
export PATH="$MLIR_AIE_ROOT/ironenv/Lib/site-packages/mlir_aie/bin:$MLIR_AIE_ROOT/ironenv/Lib/site-packages/mlir_aie/lib:/c/Xilinx/XRT/xrt_sdk/xrt/lib:/c/Xilinx/XRT/xrt_sdk/xrt:/c/Windows/System32/AMD:$PATH"
unset XILINX_XRT
export XRT_ROOT="C:/Xilinx/XRT/xrt_sdk/xrt"
export PYTHONPATH="$XRT_ROOT/python"
export NPU2=0
export NPU_CACHE_HOME="$(pwd -W)/scratch/iron-cache"
python "$@"
