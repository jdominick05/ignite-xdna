#!/usr/bin/env bash
# Run Ignition's AdaRound (torch) on an emitted XINT8 file; the base's sidecar supplies the CLE flag and listing.
#
#   ./scripts/quant-adaround.sh --quant models/resnet50_ignition_cle_c64.onnx --out models/resnet50_ignition_cle_adaround_c64.onnx --log results/quant/quant_resnet50_ignition_cle_adaround_c64.log [--in-model models/resnet50_fp32.onnx] [--calib-dir data/calib] [--iters 1000] [--data-size 1000] [--seed 1705472343] [--allow-busy]
#   ./scripts/quant-adaround.sh --in-model models/yolov8n_cut.onnx --calib-dir data/coco_calib --quant models/yolov8n_cut_ignition_cle_c64.onnx --out models/yolov8n_cut_ignition_cle_adaround_c64.onnx --log results/quant/quant_yolov8n_cut_ignition_cle_adaround_c64.log
#
# Uses resnet_env deliberately: torch 2.4.1+cpu and the same ONNX Runtime (1.22.1) that
# the Quark oracle's data sessions use, so a graph diff between the two compares the
# algorithm, not the runtime. Quark stays import-blocked. The family is read from the
# base's sidecar; a head-cut YOLO base needs its COCO calibration folder and letterboxes
# to the graph input (--cfg-path is not read). About 16 minutes on Desktop 2 for
# ResNet50's 54 layers; the log carries per-layer losses in Quark's line format so the
# two logs diff directly. Existing output/sidecar/log names are refused.
# A contended machine is refused before anything starts. Another build or producer on
# the same 16 threads leaves parity untouched -- that is a fixed-seed computation -- but
# makes this run's wall time and peak working set uncomparable with any other's;
# --allow-busy measures anyway. The snapshot lands in the load_<log> witness either way.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
QUANT="" OUT="" LOG="" IN=models/resnet50_fp32.onnx CALIB_DIR="" ITERS=1000 DATA=1000 SEED=1705472343 ALLOW_BUSY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --quant) QUANT="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --log) LOG="$2"; shift ;;
        --in-model) IN="$2"; shift ;;
        --calib-dir) CALIB_DIR="$2"; shift ;;
        --iters) ITERS="$2"; shift ;;
        --data-size) DATA="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --allow-busy) ALLOW_BUSY=1 ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$QUANT" ] && [ -n "$OUT" ] && [ -n "$LOG" ] || die "need --quant, --out and --log"
[[ "$ITERS" =~ ^[1-9][0-9]*$ ]] && [[ "$DATA" =~ ^[1-9][0-9]*$ ]] || die "--iters and --data-size must be positive"
need_file "$QUANT"; need_file "$QUANT.quant.json" "the emitted model's Ignition sidecar"; need_file "$IN"
[ -z "$CALIB_DIR" ] || need_dir "$CALIB_DIR"
[ ! -e "$OUT" ] && [ ! -e "$OUT.quant.json" ] && [ ! -e "$LOG" ] || die "output/sidecar/log already exists"
if [ "$ALLOW_BUSY" = 1 ]; then export HOST_LOAD_ALLOW_BUSY=1; fi
check_host_load refuse "$(load_witness "$LOG")"
use_env resnet_env
export PYTHONIOENCODING=utf-8
extra=()
[ -z "$CALIB_DIR" ] || extra+=(--calib-dir "$CALIB_DIR")
run_logged "$LOG" python -m quant adaround --in-model "$IN" --quant "$QUANT" --out "$OUT" \
    --iters "$ITERS" --data-size "$DATA" --seed "$SEED" "${extra[@]}"
