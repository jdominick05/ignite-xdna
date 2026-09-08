#!/usr/bin/env bash
# Build a fresh ResNet50 Quark reference (no CLE unless --cle), solely for the owned-emitter gates.
#
#   ./scripts/quant-reference.sh --out models/resnet50_quark_nocle_c64.onnx --log results/quant/quant_resnet50_quark_nocle_c64.log [--limit 64] [--cle]
#
# Uses resnet_env and the shared classification preprocessing. Calibration scratch
# is isolated under this worktree so cleanup cannot touch another session's cache.
# Existing model/log names are refused. The owned producer does not call this script.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
OUT="" LOG="" LIMIT=64 CLE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift ;;
        --log) LOG="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --cle) CLE=--cle ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$OUT" ] && [ -n "$LOG" ] || die "--out and --log are required"
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || die "--limit must be positive"
[ ! -e "$OUT" ] && [ ! -e "$LOG" ] || die "choose new model/log names"
need_file models/resnet50_fp32.onnx
need_dir data/calib
use_env resnet_env
# Size this exact graph; the generic ResNet helper underestimates its All-mode spool.
NEEDED="$(python -c 'import sys,numpy as np; from quant.graph import Graph; from quant.qdq import quantizable_tensors; g=Graph.load("models/resnet50_fp32.onnx"); g.infer_shapes(); names,_,_=quantizable_tensors(g); size=sum(int(np.prod(g.value_shape(n)))*2 for n in names)*int(sys.argv[1]); print((size+1024**3-1)//1024**3+2)' "$LIMIT")"
require_disk "$NEEDED" "ResNet calibration (inferred float16 activation bytes + reserve)"
mkdir -p scratch
PRIVATE_TMP="$(mktemp -d "$REPO_ROOT/scratch/quant-reference.XXXXXX")"
case "$PRIVATE_TMP" in "$REPO_ROOT"/scratch/quant-reference.*) ;; *) die "invalid private scratch path" ;; esac
export TEMP="$(cygpath -w "$PRIVATE_TMP")" TMP="$(cygpath -w "$PRIVATE_TMP")" PYTHONIOENCODING=utf-8
quark_guard
run_logged "$LOG" python pipelines/resnet50/3d_quantize_compare.py --out "$OUT" --limit "$LIMIT" $CLE
