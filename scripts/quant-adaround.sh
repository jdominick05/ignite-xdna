#!/usr/bin/env bash
# Run Ignition's AdaRound (torch) on an emitted XINT8 file; the base's sidecar supplies the CLE flag and listing.
#
#   ./scripts/quant-adaround.sh --quant models/resnet50_ignition_cle_c64.onnx --out models/resnet50_ignition_cle_adaround_c64.onnx --log results/quant/quant_resnet50_ignition_cle_adaround_c64.log [--in-model models/resnet50_fp32.onnx] [--iters 1000] [--data-size 1000] [--seed 1705472343]
#
# Uses resnet_env deliberately: torch 2.4.1+cpu and the same ONNX Runtime (1.22.1) that
# the Quark oracle's data sessions use, so a graph diff between the two compares the
# algorithm, not the runtime. Quark stays import-blocked. About 16 minutes on Desktop 2
# for ResNet50's 54 layers; the log carries per-layer losses in Quark's line format so
# the two logs diff directly. Existing output/sidecar/log names are refused.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
QUANT="" OUT="" LOG="" IN=models/resnet50_fp32.onnx ITERS=1000 DATA=1000 SEED=1705472343
while [ $# -gt 0 ]; do
    case "$1" in
        --quant) QUANT="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --log) LOG="$2"; shift ;;
        --in-model) IN="$2"; shift ;;
        --iters) ITERS="$2"; shift ;;
        --data-size) DATA="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$QUANT" ] && [ -n "$OUT" ] && [ -n "$LOG" ] || die "need --quant, --out and --log"
[[ "$ITERS" =~ ^[1-9][0-9]*$ ]] && [[ "$DATA" =~ ^[1-9][0-9]*$ ]] || die "--iters and --data-size must be positive"
need_file "$QUANT"; need_file "$QUANT.quant.json" "the emitted model's Ignition sidecar"; need_file "$IN"
[ ! -e "$OUT" ] && [ ! -e "$OUT.quant.json" ] && [ ! -e "$LOG" ] || die "output/sidecar/log already exists"
use_env resnet_env
export PYTHONIOENCODING=utf-8
run_logged "$LOG" python -m quant adaround --in-model "$IN" --quant "$QUANT" --out "$OUT" \
    --iters "$ITERS" --data-size "$DATA" --seed "$SEED"
