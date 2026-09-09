#!/usr/bin/env bash
# COCO mAP for every YOLO model in models/, as one comparison table.
#
#   ./scripts/yolo-eval.sh                 # 500 images, every model found
#   ./scripts/yolo-eval.sh --n 5000        # the real number, slow
#   ./scripts/yolo-eval.sh --n 200 --quick # float models on CPU only
#   ./scripts/yolo-eval.sh --model models/yolov8n_cut_xint8.onnx   # just one
#   ./scripts/yolo-eval.sh --model models/yolov8l_cut_xint8.onnx --n 5000 \
#       --witness --progress 100 --tag witnessed   # the yolov8l DPU-timeout re-run
#
# --tag SUFFIX writes results/map_<stem>_<ep>_<SUFFIX>.log instead of the bare
# name. Required whenever the bare name is already committed: re-running a model
# that another machine already logged would otherwise silently replace evidence,
# and this script now refuses rather than doing that. Say what is different
# (--tag witnessed, --tag d2), not that it is a rerun.
#
# --witness runs tools/hwinfo_npu_bridge.exe alongside every NPU eval, one JSON
# sample per second into results/witness_<stem>_<ep>.jsonl, each carrying the
# active hardware contexts BY IDENTITY, the live clock and the power mode. It is
# for long runs that might hang: without it, a mid-run DPU timeout cannot be told
# apart from another session having been on the device, and "a crash alone is not
# a finding". --progress N (default 500) sets how precisely such a crash can be
# located; 100 is reasonable when actively hunting one.
#
# Float models (.onnx with no xint8 in the name) run on CPU; quantized ones run
# on the NPU. Uses eval conventions, not demo ones: conf 0.001, per-class NMS at
# IoU 0.7, up to 300 boxes. Evaluating at conf 0.25 understates mAP badly.
#
# Every NPU model shares one compile cache, so each is run with --fresh and
# recompiles. That is slow but it is the only safe way to switch models.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

N=500 QUICK=0 ONLY="" WITNESS=0 PROGRESS=500 TAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --n)        N="$2"; shift ;;
        --quick)    QUICK=1 ;;
        --model)    ONLY="$2"; shift ;;
        --witness)  WITNESS=1 ;;
        --progress) PROGRESS="$2"; shift ;;
        --tag)      TAG="$2"; shift ;;
        -h|--help)  usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)          die "unknown flag $1" ;;
    esac
    shift
done

need_dir  data/coco/val2017                          "run: pipelines/yolov8n/2_fetch_coco.py"
need_file data/coco/annotations/instances_val2017.json

npu_env
python -c "import pycocotools" 2>/dev/null \
    || die "pycocotools missing. In this env: pip install pycocotools"

if [ -n "$ONLY" ]; then
    MODELS=("$ONLY")
else
    MODELS=()
    for m in models/yolov8*.onnx; do
        [ -f "$m" ] || continue
        case "$m" in *_cut.onnx) continue ;; esac   # float cut == float full
        MODELS+=("$m")
    done
fi
[ ${#MODELS[@]} -gt 0 ] || die "no yolov8*.onnx in models/"

ROWS=()
for m in "${MODELS[@]}"; do
    stem=$(basename "$m" .onnx)
    case "$stem" in
        *xint8*) ep=npu ;;
        *)       ep=cpu ;;
    esac
    if [ "$QUICK" = 1 ] && [ "$ep" = npu ]; then
        info "skipping $stem (--quick)"
        continue
    fi
    step "$stem on ${ep^^}  ($N images)"
    log="results/map_${stem}_${ep}${TAG:+_$TAG}.log"
    # Refuse to overwrite a COMMITTED log. CLAUDE.md: "never rewrite or tidy an
    # existing log -- add a new one", and the derived name collides for any
    # re-run of the same model on a second machine. Measured the hard way: a
    # re-run of yolov8l here silently replaced the tracked 1532s laptop-era log
    # with a 273s one, destroying the very number it was being compared against.
    # An untracked log is fine to replace; it is scratch by definition.
    # Skip, not die: with no --model this loops over every yolov8*.onnx, and most
    # of their bare logs are committed, so dying would turn the documented
    # "every model found" default into a refusal on the first model. Refusing
    # loudly is right only when the caller named one model and meant it.
    if git ls-files --error-unmatch "$log" >/dev/null 2>&1; then
        if [ -n "$ONLY" ]; then
            die "$log is committed evidence and this run would overwrite it.
Pass --tag <what-is-different> (e.g. --tag witnessed) to write beside it instead."
        fi
        warn "skipping $stem: $log is committed evidence and would be overwritten
(pass --tag <what-is-different> to re-run every model beside its existing log)"
        continue
    fi
    fresh=()
    [ "$ep" = npu ] && fresh=(--fresh)
    [ "$WITNESS" = 1 ] && [ "$ep" = npu ] \
        && witness_start "results/witness_${stem}_${ep}${TAG:+_$TAG}.jsonl"
    run_logged "$log" python pipelines/yolov8n/5_eval_map.py \
        --model "$m" --ep "$ep" --n "$N" --log 3 \
        --progress-every "$PROGRESS" "${fresh[@]}" || {
            warn "$stem failed; see $log"
            [ "$WITNESS" = 1 ] && witness_stop
            continue; }
    [ "$WITNESS" = 1 ] && [ "$ep" = npu ] && witness_stop
    # || echo "" on every one: under `set -euo pipefail` a grep that matches
    # nothing fails the whole assignment and kills the script before the table
    # ever prints, which would make the ${map:-?} fallbacks below unreachable.
    map=$(grep -oE '^mAP@50-95 +[0-9.]+' "$log" | grep -oE '[0-9.]+$' || echo "")
    m50=$(grep -oE '^mAP@50 +[0-9.]+' "$log" | grep -oE '[0-9.]+$' || echo "")
    sml=$(grep -oE '^mAP small.*' "$log" | sed 's/.*  //' || echo "")
    lat=$(grep -oE '^latency +[0-9.]+' "$log" | grep -oE '[0-9.]+$' || echo "")
    ROWS+=("$(printf '%-34s %-4s %7s %7s   %-22s %8s' \
        "$stem" "$ep" "${map:-?}" "${m50:-?}" "${sml:-?}" "${lat:-?}")")
done

printf '\n%s====================== COCO mAP (%s images) ======================%s\n' \
    "$BOLD" "$N" "$OFF"
printf '%-34s %-4s %7s %7s   %-22s %8s\n' model ep mAP mAP50 "small/med/large" "ms"
printf '%s\n' "${ROWS[@]}"
echo
echo "  Latency here includes decode at conf 0.001, so it is much higher than the"
echo "  demo's: nearly all 8400 anchors survive the prefilter. Compare demo speed"
echo "  with ./scripts/yolo-cut.sh, not with this table."
echo "  logs: results/map_*.log   detections: results/dets_*.json"
