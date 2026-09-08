#!/usr/bin/env bash
# Quantize folded ResNet with Ignition's exact-sample MinMSE, without Quark or torch.
#
#   ./scripts/quant-own.sh --out models/resnet50_own_nocle_c64.onnx --log results/quant/quant_resnet50_own_nocle_c64.log [--limit 64] [--scales-from reference.onnx]
#
# Uses resnet_env17. CLE is explicitly disabled; other graph families fail closed.
# Calibration guards disk from inferred tensor sizes and cleans its private spool.
# --scales-from selects Phase 1 replay and skips independent calibration.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
OUT="" LOG="" LIMIT=64 SCALES=""
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift ;;
        --log) LOG="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --scales-from) SCALES="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$OUT" ] && [ -n "$LOG" ] || die "need --out and --log"
[ ! -e "$OUT" ] && [ ! -e "$OUT.quant.json" ] && [ ! -e "$LOG" ] || die "output/sidecar/log already exists"
use_env resnet_env17
export PYTHONIOENCODING=utf-8
extra=()
[ -z "$SCALES" ] || extra=(--scales-from "$SCALES")
run_logged "$LOG" python pipelines/resnet50/3c_quantize_own.py --out "$OUT" --no-cle --limit "$LIMIT" "${extra[@]}"
