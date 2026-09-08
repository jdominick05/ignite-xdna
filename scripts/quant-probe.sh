#!/usr/bin/env bash
# Probe the XDNA1 EP with one-property changes to an owned ResNet graph.
#
#   ./scripts/quant-probe.sh --tag resnet50_accept_c64 [--n 32 | --full-eval] [--mutation NAME]
#
# Default: baseline plus all supported mutations, each in a separate process.
# Logs and models refuse overwrites. Each run checks CPU outputs, requires an idle
# NPU, uses the existing modelcachekey with a fresh compile, and logs EP placement
# and same-model CPU/NPU output error. Default is a diagnostic slice; --full-eval
# uses every labeled evaluation image and reports accuracy. Each child is bounded
# to 8 GiB RSS and 300 seconds (600 seconds for full evaluation).

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
TAG="" COUNT=32 MUTATION="" FULL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG="$2"; shift ;;
        --n) COUNT="$2"; shift ;;
        --mutation) MUTATION="$2"; shift ;;
        --full-eval) FULL=1 ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || die "need an alphanumeric/underscore/hyphen --tag"
[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || die "--n must be positive"
npu_env
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
if [ -n "$MUTATION" ]; then
    mutations=("$MUTATION")
else
    read -r -a mutations <<< "$(python -c 'from quant.probe import NOTES; print(" ".join(NOTES))')"
fi
baseline="models/${TAG}_baseline.onnx.probe.json"
for mutation in "${mutations[@]}"; do
    model="models/${TAG}_${mutation}.onnx"
    log="results/quant/probe_${TAG}_${mutation}.log"
    [ ! -e "$model" ] && [ ! -e "$log" ] || die "model/log exists for $mutation"
    extra=()
    seconds=300
    if [ "$FULL" = 1 ]; then
        extra+=(--full-eval)
        seconds=600
    fi
    if [ "$mutation" != baseline ]; then
        need_file "$baseline" "run the same tag's baseline first"
        extra+=(--baseline-result "$baseline")
    fi
    if run_logged "$log" python tools/quant_probe_bounded.py --seconds "$seconds" -- --mutation "$mutation" --out "$model" --n "$COUNT" "${extra[@]}"; then
        :
    else
        rc=$?
        [ "$rc" != 3 ] || die "NPU busy; resume the remaining mutations when idle"
        warn "probe process $mutation exited $rc; inspect its log before drawing a conclusion"
    fi
done
