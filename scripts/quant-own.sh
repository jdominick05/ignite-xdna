#!/usr/bin/env bash
# Run Ignition Alpha's calibration and emission (folded ResNet or head-cut YOLOv8), without Quark or torch.
#
#   ./scripts/quant-own.sh --out models/resnet50_own_nocle_c64.onnx --log results/quant/quant_resnet50_own_nocle_c64.log [--limit 64] [--cle] [--scales-from reference.onnx]
#   ./scripts/quant-own.sh --in-model models/yolov8n_cut.onnx --calib-dir data/coco_calib --out models/yolov8n_cut_ignition_cle_c64.onnx --log results/quant/quant_yolov8n_cut_ignition_cle_c64.log --cle
#
# Uses resnet_env17. CLE is off unless --cle is given; other graph families fail closed.
# --in-model selects the float export (family read from its operators); --calib-dir the listing folder.
# Calibration guards disk from inferred tensor sizes and cleans its private spool.
# --scales-from selects Phase 1 replay and skips independent calibration.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
OUT="" LOG="" LIMIT=64 SCALES="" CLE=--no-cle IN_MODEL="" CALIB_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift ;;
        --log) LOG="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --scales-from) SCALES="$2"; shift ;;
        --in-model) IN_MODEL="$2"; shift ;;
        --calib-dir) CALIB_DIR="$2"; shift ;;
        --cle) CLE=--cle ;;
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
[ -z "$SCALES" ] || extra+=(--scales-from "$SCALES")
[ -z "$IN_MODEL" ] || { need_file "$IN_MODEL"; extra+=(--in-model "$IN_MODEL"); }
[ -z "$CALIB_DIR" ] || { need_dir "$CALIB_DIR"; extra+=(--calib-dir "$CALIB_DIR"); }
run_logged "$LOG" python -m quant quantize --out "$OUT" "$CLE" --limit "$LIMIT" "${extra[@]}"
