#!/usr/bin/env bash
# OKS keypoint mAP for yolov8n-pose models in models/, as one comparison table.
#
#   ./scripts/pose-eval.sh                 # 500 images, every pose model found
#   ./scripts/pose-eval.sh --n 5000        # the real number, slow
#   ./scripts/pose-eval.sh --model models/yolov8n-pose_cut_xint8.onnx   # just one
#
# Same eval conventions as scripts/yolo-eval.sh: conf 0.001, IoU 0.7, up to 300
# boxes. See that script for why demo-conditions latency (conf 0.25) is not
# this table's latency number.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

N=500 ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --n)       N="$2"; shift ;;
        --model)   ONLY="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         die "unknown flag $1" ;;
    esac
    shift
done

need_dir  data/coco/val2017                                    "run: pipelines/yolov8n/2_fetch_coco.py"
need_file data/coco/annotations/person_keypoints_val2017.json  "run: pipelines/yolov8n-pose/2_fetch_coco_pose.py"

npu_env
python -c "import pycocotools" 2>/dev/null \
    || die "pycocotools missing. In this env: pip install pycocotools"

if [ -n "$ONLY" ]; then
    MODELS=("$ONLY")
else
    MODELS=()
    for m in models/yolov8n-pose*.onnx; do
        [ -f "$m" ] || continue
        case "$m" in *_cut.onnx) ;; *_cut_xint8*.onnx) ;; *) continue ;; esac
        MODELS+=("$m")
    done
fi
[ ${#MODELS[@]} -gt 0 ] || die "no yolov8n-pose*_cut*.onnx in models/"

ROWS=()
for m in "${MODELS[@]}"; do
    stem=$(basename "$m" .onnx)
    case "$stem" in
        *xint8*) ep=npu ;;
        *)       ep=cpu ;;
    esac
    step "$stem on ${ep^^}  ($N images)"
    log="results/map_kpts_${stem}_${ep}.log"
    fresh=()
    [ "$ep" = npu ] && fresh=(--fresh)
    run_logged "$log" python pipelines/yolov8n-pose/5_eval_map.py \
        --model "$m" --ep "$ep" --n "$N" --log 3 "${fresh[@]}" || {
            warn "$stem failed; see $log"; continue; }
    map=$(grep -oE '^OKS mAP@50-95 +[0-9.]+' "$log" | grep -oE '[0-9.]+$' || echo "")
    m50=$(grep -oE '^OKS mAP@50 +[0-9.]+' "$log" | grep -oE '[0-9.]+$' || echo "")
    lat=$(grep -oE '^latency +[0-9.]+' "$log" | grep -oE '[0-9.]+$' || echo "")
    ROWS+=("$(printf '%-34s %-4s %9s %9s %8s' \
        "$stem" "$ep" "${map:-?}" "${m50:-?}" "${lat:-?}")")
done

printf '\n%s================ OKS mAP (%s images) ================%s\n' \
    "$BOLD" "$N" "$OFF"
printf '%-34s %-4s %9s %9s %8s\n' model ep "OKS mAP" "OKS mAP50" ms
printf '%s\n' "${ROWS[@]}"
echo
echo "  logs: results/map_kpts_*.log   detections: results/dets_kpts_*.json"
