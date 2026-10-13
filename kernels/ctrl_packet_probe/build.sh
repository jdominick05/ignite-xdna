#!/usr/bin/env bash
# Build one (arm, N) design: generate, then aiecc to aie.xclbin + insts.bin, with the control overlay.
#   bash build.sh w32|ctrl N
set -euo pipefail
ARM=$1; N=$2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MLIR_AIE_ROOT:=$HOME/mlir-aie}"
source "$MLIR_AIE_ROOT/ironenv/Scripts/activate"
export MLIR_AIE_INSTALL_DIR="$(cygpath -w "$MLIR_AIE_ROOT/ironenv/Lib/site-packages/mlir_aie")"
export PEANO_INSTALL_DIR="$(cygpath -w "$MLIR_AIE_ROOT/ironenv/Lib/site-packages/llvm-aie")"
export PATH="$MLIR_AIE_ROOT/ironenv/Lib/site-packages/mlir_aie/bin:$MLIR_AIE_ROOT/ironenv/Lib/site-packages/mlir_aie/lib:/c/Xilinx/XRT/xrt_sdk/xrt/lib:/c/Xilinx/XRT/xrt_sdk/xrt:$PATH"
unset XILINX_XRT
D="$HERE/build/${ARM}_${N}"
rm -rf "$D"; mkdir -p "$D"
python "$HERE/gen.py" --arm "$ARM" --n "$N" --out "$D"
cd "$D"
aiecc --peano="$PEANO_INSTALL_DIR" --alloc-scheme=basic-sequential --generate-ctrl-pkt-overlay \
  --get-xclbin --xclbin-name=aie.xclbin --get-npu-insts --npu-insts-name=insts.bin aie.mlir > aiecc.log 2>&1 \
  || { tail -20 aiecc.log; exit 1; }
ls -l aie.xclbin insts.bin | awk '{print $5, $NF}'
