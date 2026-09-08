#!/usr/bin/env bash
# Preparation disambiguator: Quark's pre-calibration graph (apply_pre_process with the static
# op types, the forced hardware-compatibility conversions and, with --cle, its CLE) against
# Ignition's prepared float graph, plus each Quark step in isolation and, with --replay-from,
# a re-emission of a committed artifact's positions through the owned emitter. No
# calibration, no hardware; a temporary replay model under this worktree's scratch/ is
# removed at exit and nothing is written but the log.
#
#   ./scripts/quant-prepare-probe.sh --log results/quant/prepare_probe_yolov8n_cut.log --in-model models/yolov8n_cut.onnx --replay-from models/yolov8n_cut_xint8.onnx --cle
#   ./scripts/quant-prepare-probe.sh --log results/quant/prepare_probe_resnet50_fp32.log --in-model models/resnet50_fp32.onnx --replay-from models/resnet50_xint8_c64.onnx --cle
#
# --calib-dir (data/coco_calib or data/calib by family) and --limit (default 2) size the
# reader Quark's pre-process carries; CLE reads no data. Runs in resnet_env because it
# imports Quark; the owned core never does.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
LOG="" MODEL=models/yolov8n_cut.onnx REPLAY="" CALIB_DIR="" LIMIT=2 CLE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --log) LOG="$2"; shift ;;
        --in-model) MODEL="$2"; shift ;;
        --replay-from) REPLAY="$2"; shift ;;
        --calib-dir) CALIB_DIR="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --cle) CLE=--cle ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$LOG" ] || die "--log is required"
[ ! -e "$LOG" ] || die "log exists: $LOG"
[[ "$LIMIT" =~ ^[0-9]+$ ]] && [ "$LIMIT" -gt 0 ] || die "--limit must be a positive integer"
need_file "$MODEL"
[ -z "$REPLAY" ] || need_file "$REPLAY"
use_env resnet_env
export PYTHONIOENCODING=utf-8
run_logged "$LOG" python tools/quant_prepare_probe.py --in-model "$MODEL" --limit "$LIMIT" $CLE \
    ${REPLAY:+--replay-from "$REPLAY"} ${CALIB_DIR:+--calib-dir "$CALIB_DIR"}
