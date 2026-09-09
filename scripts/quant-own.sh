#!/usr/bin/env bash
# Run Ignition Alpha's calibration and emission (folded ResNet, head-cut YOLOv8 or MODNet), without Quark or torch.
#
#   ./scripts/quant-own.sh --out models/resnet50_own_nocle_c64.onnx --log results/quant/quant_resnet50_own_nocle_c64.log [--limit 64] [--cle] [--scales-from reference.onnx] [--allow-busy]
#   ./scripts/quant-own.sh --in-model models/yolov8n_cut.onnx --calib-dir data/coco_calib --out models/yolov8n_cut_ignition_cle_c64.onnx --log results/quant/quant_yolov8n_cut_ignition_cle_c64.log --cle
#   ./scripts/quant-own.sh --in-model models/modnet/modnet_cut_fp32.onnx --calib-dir data/modnet_calib --cfg-path models/modnet/preprocess_config.json --out models/modnet/modnet_cut_ignition_cle_c64.onnx --log results/quant/quant_modnet_cut_ignition_cle_c64.log --cle
#
# Uses resnet_env17. CLE is off unless --cle is given; other graph families fail closed.
# --in-model selects the float export (family read from its operators); --calib-dir the listing
# folder; --cfg-path the preprocessing config ResNet's transform reads and MODNet's size is
# checked against (YOLO reads neither).
# Calibration guards disk from inferred tensor sizes and cleans its private spool.
# --scales-from selects Phase 1 replay and skips independent calibration.
# --cle-guard BITS skips a CLE pair whose per-channel scale exceeds BITS powers of two;
# off by default, which is the parity path.
# A contended machine is refused before anything starts. Another build or producer on
# the same 16 threads leaves parity untouched -- that is a fixed-seed computation -- but
# makes this run's wall time and peak working set uncomparable with any other's;
# --allow-busy measures anyway. The snapshot lands in the load_<log> witness either way.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
OUT="" LOG="" LIMIT=64 SCALES="" CLE=--no-cle IN_MODEL="" CALIB_DIR="" CFG_PATH="" ALLOW_BUSY=0 CLE_GUARD=""
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift ;;
        --log) LOG="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --scales-from) SCALES="$2"; shift ;;
        --in-model) IN_MODEL="$2"; shift ;;
        --calib-dir) CALIB_DIR="$2"; shift ;;
        --cfg-path) CFG_PATH="$2"; shift ;;
        --cle) CLE=--cle ;;
        --cle-guard) CLE_GUARD="$2"; shift ;;
        --allow-busy) ALLOW_BUSY=1 ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$OUT" ] && [ -n "$LOG" ] || die "need --out and --log"
[ ! -e "$OUT" ] && [ ! -e "$OUT.quant.json" ] && [ ! -e "$LOG" ] || die "output/sidecar/log already exists"
if [ "$ALLOW_BUSY" = 1 ]; then export HOST_LOAD_ALLOW_BUSY=1; fi
check_host_load refuse "$(load_witness "$LOG")"
use_env resnet_env17
export PYTHONIOENCODING=utf-8
extra=()
[ -z "$SCALES" ] || extra+=(--scales-from "$SCALES")
[ -z "$IN_MODEL" ] || { need_file "$IN_MODEL"; extra+=(--in-model "$IN_MODEL"); }
[ -z "$CALIB_DIR" ] || { need_dir "$CALIB_DIR"; extra+=(--calib-dir "$CALIB_DIR"); }
[ -z "$CFG_PATH" ] || { need_file "$CFG_PATH"; extra+=(--cfg-path "$CFG_PATH"); }
run_logged "$LOG" python -m quant quantize --out "$OUT" "$CLE" ${CLE_GUARD:+--cle-guard "$CLE_GUARD"} --limit "$LIMIT" "${extra[@]}"
