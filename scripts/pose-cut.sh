#!/usr/bin/env bash
# YOLOv8n-pose on the NPU, end to end: cut -> quantize -> CPU sanity -> NPU -> verdict.
#
# Same recipe as scripts/yolo-cut.sh (detect): the pose head ends in the same
# Reshape/Softmax/Transpose/Slice/Sub/Div arithmetic tail, cutting it is what
# lets the EP take the graph at all. See pipelines/yolov8n-pose/1b_cut_head.py.
#
#   ./scripts/pose-cut.sh                 # XINT8, 300 calib images
#   ./scripts/pose-cut.sh --limit 32      # faster; fine for "does the EP take it"
#   ./scripts/pose-cut.sh --adaround      # XINT8 + AdaRound (slow, many minutes)
#   ./scripts/pose-cut.sh --skip-quant    # reuse the existing quantized model
#
# Steps 0-2 need resnet_env (Quark); steps 3-5 need resnet_env17 (NPU).

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ADAROUND=0 SKIP_QUANT=0 LIMIT=300 SOURCE="assets/test_image.jpg" REQUANT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --adaround)   ADAROUND=1 ;;
        --skip-quant) SKIP_QUANT=1 ;;
        --recut)      REQUANT=1 ;;
        --limit)      LIMIT="$2"; shift ;;
        --source)     SOURCE="$2"; shift ;;
        -h|--help)    usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)            die "unknown flag $1" ;;
    esac
    shift
done

TAG=$([ "$ADAROUND" = 1 ] && echo xint8_adaround || echo xint8)
BASE="models/yolov8n-pose.onnx"
CUT="models/yolov8n-pose_cut.onnx"
QUANT="models/yolov8n-pose_cut_${TAG}.onnx"
CACHE_KEY="yoloposecutcachekey"

need_dir data/coco_calib "run: python pipelines/yolov8n/2_fetch_coco.py"

# --------------------------------------------------------------- 0. export
if [ ! -f "$BASE" ]; then
    step "0/5  exporting yolov8n-pose to ONNX (downloads weights if needed)"
    use_env resnet_env
    python pipelines/yolov8n-pose/1_export.py --out "$BASE"
    need_file "$BASE" "export did not produce $BASE"
fi

# ---------------------------------------------------------------- 1. cut head
if [ -f "$CUT" ] && [ "$REQUANT" = 0 ]; then
    step "1/5  head cut -- $CUT exists, skipping (--recut to redo)"
else
    step "1/5  cutting the DFL/kpt-decode tail off the graph"
    use_env resnet_env
    python pipelines/yolov8n-pose/1b_cut_head.py --in "$BASE" --out "$CUT"
fi
need_file "$CUT"

# ---------------------------------------------------------------- 2. quantize
if [ "$SKIP_QUANT" = 1 ]; then
    step "2/5  quantize -- skipped (--skip-quant)"
    need_file "$QUANT" "nothing to skip to; drop --skip-quant"
else
    step "2/5  quantizing to ${TAG}  (${LIMIT} calibration images)"
    require_disk $(( (LIMIT * MB_PER_CALIB_IMAGE_YOLO + 1023) / 1024 + 2 )) \
                 "Quark's calibration cache"
    quark_guard
    [ "$ADAROUND" = 1 ] && info "AdaRound is slow -- expect many minutes"
    use_env resnet_env
    ARGS=(--in "$CUT" --calib-dir data/coco_calib --limit "$LIMIT" --out "$QUANT")
    [ "$ADAROUND" = 1 ] && ARGS+=(--adaround)
    run_logged "results/pose_cut_quantize_${TAG}.log" \
        python pipelines/yolov8n-pose/3b_quantize_cut.py "${ARGS[@]}"
fi
need_file "$QUANT"

# ------------------------------------------------------------- 3. CPU sanity
step "3/5  CPU sanity check on the quantized model"
npu_env
run_logged "results/pose_cut_cpu.log" \
    python pipelines/yolov8n-pose/4_pose.py --model "$QUANT" --ep cpu \
        --source "$SOURCE" --runs 5

CPU_PEOPLE=$(grep -oE '^[0-9]+ people' "results/pose_cut_cpu.log" | head -1 | cut -d' ' -f1 || true)
[ "${CPU_PEOPLE:-0}" -gt 0 ] || die "the quantized model found nobody on CPU -- fix that before touching the NPU"
ok "$CPU_PEOPLE people found on CPU"

# ------------------------------------------------------------------ 4. NPU
step "4/5  NPU run  (--fresh: recompiling, this looks frozen but isn't)"
run_logged "results/pose_cut_npu.log" \
    python pipelines/yolov8n-pose/4_pose.py --model "$QUANT" --ep npu \
        --source "$SOURCE" --fresh --log 1

# ---------------------------------------------------------------- 5. verdict
step "5/5  what the EP actually took"
run_logged "results/pose_cut_diag.log" \
    python tools/diag_ep.py --model "$QUANT" --cache-key "$CACHE_KEY" || true

NPU_MS=$(grep -oE 'infer +mean +[0-9.]+' "results/pose_cut_npu.log" | head -1 \
         | grep -oE '[0-9.]+$' || echo "")

printf '\n%s================ RESULT ================%s\n' "$BOLD" "$OFF"
if npu_verdict "$CACHE_KEY"; then
    printf '%s  yolov8n-pose head-cut compiles on the NPU. %s ms/frame.%s\n' \
        "$GREEN" "${NPU_MS:-?}" "$OFF"
    echo "  Next: OKS keypoint mAP -- pipelines/yolov8n-pose/5_eval_map.py"
else
    printf '%s  Zero NPU nodes -- the decode tail was not the (whole) cause.%s\n' "$RED" "$OFF"
    echo "  Compare op types against results/yolo_cut_diag.log (the known-good detect cut)."
fi
echo
echo "  logs: results/pose_cut_{cpu,npu,diag}.log"
