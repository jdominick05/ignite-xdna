#!/usr/bin/env bash
# Measure how Quark's finetune wrapper consumes the torch RNG, the fact Ignition's AdaRound transcription relies on.
#
#   ./scripts/quant-rng-probe.sh --log results/quant/adaround_rng_probe_resnet_env.log
#
# Uses resnet_env because the probe imports Quark (which prints its usual custom-op
# build warnings first). Writes only the log; refuses an existing log name.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
LOG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --log) LOG="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$LOG" ] || die "need --log"
[ ! -e "$LOG" ] || die "log already exists: $LOG"
use_env resnet_env
export PYTHONIOENCODING=utf-8
run_logged "$LOG" python tools/quant_adaround_rng_probe.py
