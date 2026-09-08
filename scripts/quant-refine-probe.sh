#!/usr/bin/env bash
# Refinement disambiguator: perturb the fresh no-CLE oracle's positions and diff Quark's
# adjust_quantize_info against Ignition's refine on byte-identical inputs.
#
#   ./scripts/quant-refine-probe.sh --log results/quant/refine_probe_resnet50_quark_nocle_c64.log [--reference models/resnet50_quark_nocle_c64.onnx] [--trials 200] [--seed 0] [--max-scales 6] [--max-shift 6] [--verbose]
#
# Runs in resnet_env because it imports Quark; the owned core never does. Directed
# cases violate one rule each, random trials perturb several scales, and the gate
# is identical final position tables. Writes nothing but the log, which it refuses
# to overwrite. No hardware is touched.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
LOG="" REFERENCE=models/resnet50_quark_nocle_c64.onnx TRIALS=200 SEED=0 SCALES=6 SHIFT=6 VERBOSE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --log) LOG="$2"; shift ;;
        --reference) REFERENCE="$2"; shift ;;
        --trials) TRIALS="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --max-scales) SCALES="$2"; shift ;;
        --max-shift) SHIFT="$2"; shift ;;
        --verbose) VERBOSE=--verbose ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$LOG" ] || die "--log is required"
[ ! -e "$LOG" ] || die "log exists: $LOG"
[[ "$TRIALS" =~ ^[0-9]+$ ]] && [[ "$SEED" =~ ^[0-9]+$ ]] || die "--trials and --seed must be nonnegative integers"
[[ "$SCALES" =~ ^[1-9][0-9]*$ ]] && [[ "$SHIFT" =~ ^[1-9][0-9]*$ ]] || die "--max-scales and --max-shift must be positive integers"
need_file "$REFERENCE"
need_file "${REFERENCE%.onnx}.reference.json" "the oracle sidecar binds the run to its hash"
use_env resnet_env
export PYTHONIOENCODING=utf-8
run_logged "$LOG" python tools/quant_refine_probe.py --reference "$REFERENCE" --trials "$TRIALS" --seed "$SEED" --max-scales "$SCALES" --max-shift "$SHIFT" $VERBOSE
