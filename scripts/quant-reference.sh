#!/usr/bin/env bash
# Build a fresh Quark reference (no CLE unless --cle; --adaround for XINT8_ADAROUND), solely for the owned-emitter gates.
#
#   ./scripts/quant-reference.sh --out models/resnet50_quark_nocle_c64.onnx --log results/quant/quant_resnet50_quark_nocle_c64.log [--limit 64] [--cle] [--adaround]
#   ./scripts/quant-reference.sh --in-model models/yolov8n_cut.onnx --calib-dir data/coco_calib --out models/yolov8n_cut_quark_cle_c64.onnx --log results/quant/quant_yolov8n_cut_quark_cle_c64.log --cle
#
# Uses resnet_env and the shared preprocessing of the family read from the float export
# (classification config for ResNet, npu.yolo letterbox for a head-cut YOLO). --adaround runs Quark's
# CPU FastFinetune after calibration (about 16 minutes on Desktop 2 for 54 layers) and
# records its peak working set in the .reference.json sidecar. Calibration scratch
# is isolated under this worktree so cleanup cannot touch another session's cache.
# Existing model/log names are refused. The owned producer does not call this script.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
OUT="" LOG="" LIMIT=64 CLE="" ADAROUND="" IN_MODEL=models/resnet50_fp32.onnx CALIB_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift ;;
        --log) LOG="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --cle) CLE=--cle ;;
        --adaround) ADAROUND=--adaround ;;
        --in-model) IN_MODEL="$2"; shift ;;
        --calib-dir) CALIB_DIR="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$OUT" ] && [ -n "$LOG" ] || die "--out and --log are required"
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || die "--limit must be positive"
[ ! -e "$OUT" ] && [ ! -e "$LOG" ] || die "choose new model/log names"
need_file "$IN_MODEL"
[ -z "$CALIB_DIR" ] || need_dir "$CALIB_DIR"
use_env resnet_env
# Size this exact graph; the generic ResNet helper underestimates its All-mode spool.
NEEDED="$(python -c 'import sys,numpy as np; from quant.quantize import prepared_graph; from quant.qdq import quantizable_tensors; g,_=prepared_graph(sys.argv[2]); names,_,_=quantizable_tensors(g); size=sum(int(np.prod(g.value_shape(n)))*2 for n in names)*int(sys.argv[1]); print((size+1024**3-1)//1024**3+2)' "$LIMIT" "$IN_MODEL")"
require_disk "$NEEDED" "calibration of $IN_MODEL (inferred float16 activation bytes + reserve)"
mkdir -p scratch
PRIVATE_TMP="$(mktemp -d "$REPO_ROOT/scratch/quant-reference.XXXXXX")"
case "$PRIVATE_TMP" in "$REPO_ROOT"/scratch/quant-reference.*) ;; *) die "invalid private scratch path" ;; esac
export TEMP="$(cygpath -w "$PRIVATE_TMP")" TMP="$(cygpath -w "$PRIVATE_TMP")" PYTHONIOENCODING=utf-8
quark_guard
extra=(--in-model "$IN_MODEL")
[ -z "$CALIB_DIR" ] || extra+=(--calib-dir "$CALIB_DIR")
run_logged "$LOG" python pipelines/resnet50/3d_quantize_compare.py --out "$OUT" --limit "$LIMIT" "${extra[@]}" $CLE $ADAROUND
