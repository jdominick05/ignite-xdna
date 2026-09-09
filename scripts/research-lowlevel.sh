#!/usr/bin/env bash
# Run a bounded Windows low-level research experiment with scrubbed UTF-8 evidence.
#
#   ./scripts/research-lowlevel.sh --log results/quant/<unique>.log [--npu] [--checks-only] [--seconds 300] [--rss-gib 8] -- python tools/<experiment>.py <args>
#   ./scripts/research-lowlevel.sh --log results/aie/<unique>.log --npu -- bash scripts/research-iron.sh kernels/memory_placement/probe.py <args>
#
# The parent uses resnet_env17, requires a clear host and (with --npu) an idle device,
# watches foreign processes, and bounds the entire child tree. --checks-only is for
# correctness runs without performance claims; --npu still requires an idle device.
# Existing log names are refused.
# --wait-clear <0..60> waits for an idle host before starting a timing run.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
LOG="" opts=()
while [ $# -gt 0 ]; do
    case "$1" in
        --log) LOG="$2"; shift 2 ;;
        --seconds|--rss-gib|--wait-clear) opts+=("$1" "$2"); shift 2 ;;
        --npu|--checks-only) opts+=("$1"); shift ;;
        --) shift; break ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
done
[ -n "$LOG" ] && [ $# -gt 0 ] || die "need --log and a command after --"
[ ! -e "$LOG" ] || die "log already exists"
use_env resnet_env17
export RYZEN_AI_INSTALLATION_PATH="$RYZEN_AI_PATH" XLNX_ONNX_EP_REPORT_FILE=vitisai_ep_report.json
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
run_logged "$LOG" python tools/research_process.py "${opts[@]}" -- "$@"
