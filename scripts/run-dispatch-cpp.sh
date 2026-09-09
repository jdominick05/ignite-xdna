#!/usr/bin/env bash
# Drive the dispatch-floor comparison: the SAME compiled design through the Python
# harness and through the C++ host, in one sitting.
#
#   ./scripts/run-dispatch-cpp.sh [--iters N] [--payload N] [--batch-sizes L]
#
# WHY BOTH IN ONE SITTING: CLAUDE.md's standing rule is that NPU latency on this shared
# machine drifts between sittings independent of any code change, so the committed
# 2026-09-09 Python figures are NOT a valid baseline for a C++ number measured today.
# This runs measure_runlist.py and dispatch_runner.exe back to back on the same cache
# entry, so the Python/C++ difference is the binding and not the day.
#
# The Python step is what compiles the passthrough through IRON and prints the cache
# entry it resolved; the C++ step is then pointed at THAT entry explicitly rather than
# at "the newest", because the newest entry in ~/.npu/cache/ is a different design with
# a different argument layout after any other design is compiled.
#
# Nothing is written into any of the repo's *cachekey/ compile-cache directories.
# ~/.npu/cache/ is IRON's own cache and is read here, plus whatever the IRON compile
# step adds for the passthrough itself.

set -euo pipefail

ITERS=100
PAYLOAD=4096
BATCHES="1,2,4,8,16,32,64"
while [ $# -gt 0 ]; do
    case "$1" in
        --iters) ITERS="$2"; shift 2 ;;
        --payload) PAYLOAD="$2"; shift 2 ;;
        --batch-sizes) BATCHES="$2"; shift 2 ;;
        --help|-h) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IRONENV="$HOME/mlir-aie/ironenv/Scripts"
export PATH="$IRONENV:$PATH"
export PYTHONPATH="/c/Xilinx/XRT/xrt_sdk/xrt/python${PYTHONPATH:+:$PYTHONPATH}"

XRT_SMI="/c/Windows/System32/AMD/xrt-smi.exe"

echo "=============================================================================="
echo "PRECONDITIONS"
echo "=============================================================================="
"$XRT_SMI" examine -r aie-partitions 2>&1 | tail -5
powershell -ExecutionPolicy Bypass -File tools/host_load.ps1 2>&1 | tail -8

echo
echo "=============================================================================="
echo "ARM 1: Python (measure_runlist.py) -- compiles the design and baselines it"
echo "=============================================================================="
PYOUT="$(mktemp -t dispatch_py.XXXXXX)"
trap 'rm -f "$PYOUT"' EXIT INT TERM
python3 kernels/dispatch_floor/measure_runlist.py \
    --iters "$ITERS" --payload "$PAYLOAD" --batch-sizes "$BATCHES" 2>&1 | tee "$PYOUT"

CACHE_ENTRY="$(grep -oE 'cache:  [0-9a-f]+' "$PYOUT" | head -1 | awk '{print $2}')"
if [ -z "$CACHE_ENTRY" ]; then
    echo "ERROR: could not read the resolved cache entry out of the Python run." >&2
    exit 1
fi
XCLBIN="$HOME/.npu/cache/$CACHE_ENTRY/final.xclbin"
INSTS="$HOME/.npu/cache/$CACHE_ENTRY/insts.bin"
echo
echo "Resolved design: $CACHE_ENTRY"

echo
echo "=============================================================================="
echo "ARM 2: C++ (dispatch_runner.exe) -- same xclbin, same insts, no Python"
echo "=============================================================================="
kernels/dispatch_floor/dispatch_runner.exe \
    --xclbin "$(cygpath -w "$XCLBIN")" --insts "$(cygpath -w "$INSTS")" \
    --iters "$ITERS" --payload "$PAYLOAD" --batch-sizes "$BATCHES"

echo
echo "=============================================================================="
echo "POST-RUN WITNESS"
echo "=============================================================================="
"$XRT_SMI" examine -r aie-partitions 2>&1 | tail -3
powershell -ExecutionPolicy Bypass -File tools/host_load.ps1 2>&1 | tail -8
