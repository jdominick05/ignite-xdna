#!/usr/bin/env bash
# A terminal launcher for the models in this repo: run one on your own input, or
# run one of the demos behind the findings.
#
#   ./scripts/tui.sh                # the menu
#   ./scripts/tui.sh --selftest     # validate the catalogue; opens no NPU context
#   ./scripts/tui.sh --lane task    # skip straight to the task list
#   ./scripts/tui.sh --lane demo
#
# Results go to outputs/, never results/ -- that tree is the tracked evidence base
# and its images are cited from the docs. Latencies the TUI prints are indicative
# (sess.run alone, one sitting) and are not doc-quotable; docs/BENCHMARKS.md has the
# measured numbers with their method and caveats.
#
# Arrow keys need a real Windows console, which mintty is not, so running through
# this wrapper gives numbered menus. For arrow keys, use PowerShell instead:
#
#   conda activate resnet_env17
#   $env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'
#   python -m tui
#
# Needs `rich` in resnet_env17 (it is already there on Desktop 2).

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         ARGS+=("$1") ;;
    esac
    shift
done

# npu_env does the whole preamble: activates resnet_env17, points
# RYZEN_AI_INSTALLATION_PATH at 1.7.1, exports XLNX_ONNX_EP_REPORT_FILE so the EP
# writes its report, and runs the contention and host-load checks. The TUI re-runs
# the last two before each launch, because a session-long menu can start twenty
# runs and a check taken once at startup says nothing about the twentieth.
npu_env

# Must be exported before any child imports cv2. OpenCV reads it once at videoio
# init, so setting it later is a silent no-op -- 90 s to open a camera instead of
# 0.22 s (results/cam_probe_backends.log, results/cam_probe_late_set.log).
export OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
export PYTHONUTF8=1

python -m tui "${ARGS[@]}"
