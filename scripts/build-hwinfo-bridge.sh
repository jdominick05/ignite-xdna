#!/usr/bin/env bash
# Build tools/hwinfo_npu_bridge.exe (the XDNA1 NPU monitor / HWiNFO64 bridge) with MSVC, from Git Bash.
#
#   ./scripts/build-hwinfo-bridge.sh
#
# Thin wrapper over scripts/build_hwinfo_bridge.bat, which holds the real logic (vswhere ->
# vcvars64 -> cl.exe; headers from the npu_monitor_build conda env; XRT SDK when present).
# `cmd.exe //c` and not `/c`: Git Bash rewrites a bare `/c` into `C:/` before cmd sees it,
# and cmd then waits on stdin forever.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

cmd.exe //c "scripts\\build_hwinfo_bridge.bat"
