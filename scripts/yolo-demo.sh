#!/usr/bin/env bash
# Live detection from the webcam. Press q to quit.
#
#   ./scripts/yolo-demo.sh                 # best available model, NPU
#   ./scripts/yolo-demo.sh --ep cpu        # CPU, for comparison
#   ./scripts/yolo-demo.sh --source 1      # second camera
#   ./scripts/yolo-demo.sh --source clip.mp4
#   ./scripts/yolo-demo.sh --model models/yolov8n.onnx
#   ./scripts/yolo-demo.sh --seconds 15    # bounded, LOGGED run -- the evidence form
#
# Model preference, best first: cut+AdaRound, cut XINT8, full XINT8, FP32.
# The cut models decode in numpy and are the only ones with a shot at the NPU.
#
# --seconds N is what makes a webcam run quotable: it auto-quits after N seconds,
# tees a UTF-8 log to results/ under a name carrying the model and EP, appends the
# EP's own placement verdict to that same log, and leaves annotated snapshots in
# outputs/webcam/ (git-ignored -- camera frames must never land in tracked
# results/). Without it the demo is display-only and leaves nothing behind, which
# is why "does the single 4x4.xclbin session drive a live webcam" stayed open in
# RESEARCH.md long after every piece of code for it existed. A run with nobody in
# front of the camera verifies nothing; the summary says so when it sees 0 dets.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

EP=npu SOURCE=0 MODEL="" CONF=0.25 STALE=0 SECONDS_ARG="" LOG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --ep)       EP="$2"; shift ;;
        --source)   SOURCE="$2"; shift ;;
        --model)    MODEL="$2"; shift ;;
        --conf)     CONF="$2"; shift ;;
        --seconds)  SECONDS_ARG="$2"; shift ;;
        --log-file) LOG="$2"; shift ;;
        --stale)    STALE=1 ;;
        -h|--help)  usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)          die "unknown flag $1" ;;
    esac
    shift
done

if [ -z "$MODEL" ]; then
    for m in models/yolov8n_cut_xint8_adaround.onnx \
             models/yolov8n_cut_xint8.onnx \
             models/yolov8n_xint8.onnx \
             models/yolov8n_cut.onnx \
             models/yolov8n.onnx; do
        [ -f "$m" ] && { MODEL="$m"; break; }
    done
    [ -n "$MODEL" ] || die "no yolov8n model in models/ -- run ./scripts/yolo-cut.sh first"
    info "picked $MODEL"
fi
need_file "$MODEL"

# Allow-list the quantized names instead of listing the float ones: the float
# branch used to name yolov8n paths literally, so yolov8s_cut.onnx sailed past it.
case "$MODEL" in
    *int8*|*a8w8*) ;;
    *)  [ "$EP" = npu ] && die "$MODEL is float; the NPU needs a quantized model.
Run ./scripts/yolo-cut.sh, or pass --ep cpu." ;;
esac

# --fresh on every NPU launch. yolocutcachekey is shared by every cut model
# and this script auto-picks the best one present, which is routinely NOT the
# model yolo-cut.sh last compiled into that cache. Without --fresh the EP
# reuses the other model's artifact and the demo draws plausible-looking
# boxes from the wrong weights. Recompiling costs a minute; --stale skips it
# when you know the cache already matches.
FRESH=()
if [ "$EP" = npu ] && [ "$STALE" = 0 ]; then
    FRESH=(--fresh)
    info "recompiling (--fresh); pass --stale to reuse the existing cache"
fi

# A bounded run is a logged run. The name carries model and EP so two variants --
# or two machines -- can never write the same file, which is this repo's result-log
# naming rule, not a preference.
STEM="$(basename "$MODEL")"; STEM="${STEM%.onnx}"
if [ -n "$SECONDS_ARG" ] && [ -z "$LOG" ]; then
    LOG="results/webcam_single_4x4_${STEM}_${EP}.log"
fi

SECS=()
[ -n "$SECONDS_ARG" ] && SECS=(--max-seconds "$SECONDS_ARG")

# The witness goes BESIDE the log, never into it: npu_env appends the host-load
# sample to whatever path it is given, and run_logged's tee truncates the log
# afterwards -- so passing "$LOG" here silently loses the contention record.
# load_witness derives the name so two variants cannot share one witness.
WITNESS=""
[ -n "$LOG" ] && WITNESS="$(load_witness "$LOG")"
npu_env "$WITNESS"
step "webcam demo -- $MODEL on ${EP^^}, press q in the window to quit"

CMD=(python pipelines/yolov8n/4_detect.py
     --model "$MODEL" --ep "$EP" --source "$SOURCE" --conf "$CONF" --log 2
     "${FRESH[@]}" "${SECS[@]}")

if [ -z "$LOG" ]; then
    "${CMD[@]}"
    exit $?
fi

run_logged "$LOG" "${CMD[@]}"

# Append the EP's own verdict to the SAME log. A "_npu" in the filename means the
# EP was requested, never that it engaged -- vitisai_ep_report.json is the evidence.
if [ "$EP" = npu ]; then
    CACHE_KEY="$(sed -n 's/^cache key: //p' "$LOG" | head -1)"
    [ -n "$CACHE_KEY" ] || die "could not read the cache key back from $LOG"
    npu_verdict "$CACHE_KEY" 2>&1 | tee -a "$LOG"
fi
ok "log: $LOG"
