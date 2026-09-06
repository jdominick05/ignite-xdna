#!/usr/bin/env bash
# Live detection from the webcam. Press q to quit.
#
#   ./scripts/yolo-demo.sh                 # best available model, NPU
#   ./scripts/yolo-demo.sh --ep cpu        # CPU, for comparison
#   ./scripts/yolo-demo.sh --source 1      # second camera
#   ./scripts/yolo-demo.sh --source clip.mp4
#   ./scripts/yolo-demo.sh --model models/yolov8n.onnx
#
# Model preference, best first: cut+AdaRound, cut XINT8, full XINT8, FP32.
# The cut models decode in numpy and are the only ones with a shot at the NPU.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

EP=npu SOURCE=0 MODEL="" CONF=0.25 STALE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --ep)      EP="$2"; shift ;;
        --source)  SOURCE="$2"; shift ;;
        --model)   MODEL="$2"; shift ;;
        --conf)    CONF="$2"; shift ;;
        --stale)   STALE=1 ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         die "unknown flag $1" ;;
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

npu_env
step "webcam demo -- $MODEL on ${EP^^}, press q in the window to quit"
python pipelines/yolov8n/4_detect.py \
    --model "$MODEL" --ep "$EP" --source "$SOURCE" --conf "$CONF" --log 2 "${FRESH[@]}"
