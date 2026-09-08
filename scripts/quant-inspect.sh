#!/usr/bin/env bash
# Inspect local ONNX model fingerprints; no model execution or NPU session.
#
#   ./scripts/quant-inspect.sh [--env resnet_env17|resnet_env] --log results/quant/<model_variant>.log MODEL [MODEL ...]
#
# The log is UTF-8 and must have a new name: existing evidence is never overwritten.
# Use both envs when checking quant/ core compatibility. The inspector reports any
# topological reordering in memory; model files are never rewritten.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

INSPECT_ENV=resnet_env17
INSPECT_LOG=""
MODELS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --env) [ $# -ge 2 ] || die "--env needs a value"; INSPECT_ENV="$2"; shift ;;
        --log) [ $# -ge 2 ] || die "--log needs a value"; INSPECT_LOG="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        --) shift; MODELS+=("$@"); break ;;
        -*) die "unknown flag $1" ;;
        *) MODELS+=("$1") ;;
    esac
    shift
done
case "$INSPECT_ENV" in resnet_env|resnet_env17) ;; *) die "use resnet_env or resnet_env17" ;; esac
[ -n "$INSPECT_LOG" ] || die "--log needs a new model/variant log name"
[ "${#MODELS[@]}" -gt 0 ] || die "supply at least one ONNX model"
[ ! -e "$INSPECT_LOG" ] || die "log already exists: $INSPECT_LOG"
for model in "${MODELS[@]}"; do need_file "$model"; done
use_env "$INSPECT_ENV"
export PYTHONIOENCODING=utf-8
run_logged "$INSPECT_LOG" python tools/quant_inspect.py "${MODELS[@]}"
