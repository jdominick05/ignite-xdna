#!/usr/bin/env bash
# CLE disambiguator: equalize the float export with Quark's cle_transforms and with
# Ignition's cross_layer_equalize, then compare the pattern list and every float
# initializer byte for byte. No calibration, no hardware, nothing written but the log.
#
#   ./scripts/quant-cle-probe.sh --log results/quant/cle_probe_resnet50_fp32.log [--in-model models/resnet50_fp32.onnx] [--steps 1]
#
# Runs in resnet_env because it imports Quark; the owned core never does.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
LOG="" MODEL=models/resnet50_fp32.onnx STEPS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --log) LOG="$2"; shift ;;
        --in-model) MODEL="$2"; shift ;;
        --steps) STEPS="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$LOG" ] || die "--log is required"
[ ! -e "$LOG" ] || die "log exists: $LOG"
[[ "$STEPS" =~ ^-?[0-9]+$ ]] || die "--steps must be an integer (-1 is the source's adaptive mode)"
need_file "$MODEL"
use_env resnet_env
export PYTHONIOENCODING=utf-8
run_logged "$LOG" python tools/quant_cle_probe.py --in-model "$MODEL" --steps "$STEPS"
