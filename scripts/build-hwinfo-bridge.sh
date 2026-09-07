#!/usr/bin/env bash
# Build the native HWiNFO64 NPU custom sensor bridge executable using MSVC.
#
#   ./scripts/build-hwinfo-bridge.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cmd.exe /c "call \"$SCRIPT_DIR\\build_hwinfo_bridge.bat\""
