#!/usr/bin/env bash
# Build kernels/dispatch_floor/dispatch_runner -- the C++ XRT host for the AIE dispatch floor -- on Linux.
#   bash scripts/build_dispatch_runner.sh          (from anywhere; XILINX_XRT defaults to /opt/xilinx/xrt)
# The Linux counterpart of scripts/build_dispatch_runner.bat; same source, same -O2, g++ in place of cl.exe.

set -euo pipefail
if [ "${1:-}" = --help ]; then sed -n '2,3s/^# //p' "${BASH_SOURCE[0]}"; exit 0; fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${XILINX_XRT:=/opt/xilinx/xrt}"
if [ ! -f "$XILINX_XRT/include/xrt/experimental/xrt_kernel.h" ]; then
  echo "[ERROR] no xrt::runlist header at $XILINX_XRT/include/xrt/experimental/xrt_kernel.h" >&2
  echo "        set XILINX_XRT to an XRT install that has it." >&2
  exit 1
fi
SRC="$REPO_ROOT/kernels/dispatch_floor/dispatch_runner.cpp"
OUT="$REPO_ROOT/kernels/dispatch_floor/dispatch_runner"
echo "[INFO] g++ $(g++ -dumpfullversion) kernels/dispatch_floor/dispatch_runner.cpp"
g++ -std=c++17 -O2 -Wall -I"$XILINX_XRT/include" "$SRC" -o "$OUT" \
    -L"$XILINX_XRT/lib" -Wl,-rpath,"$XILINX_XRT/lib" -lxrt_coreutil -pthread
echo "[SUCCESS] built kernels/dispatch_floor/dispatch_runner"
