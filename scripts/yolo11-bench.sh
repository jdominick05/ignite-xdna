#!/usr/bin/env bash
# YOLOv11n on Ryzen AI XDNA1 NPU: export -> cut -> quantize -> CPU/DML/NPU bench -> diag.
#
# Characterizes native NPU execution of YOLOv11n's C2PSA spatial self-attention
# block (4D MatMul + Softmax + Transpose) and decoupled DWConv detect head.
#
#   ./scripts/yolo11-bench.sh                 # full pipeline (64 calib images, 50 runs)
#   ./scripts/yolo11-bench.sh --skip-quant    # reuse existing quantized model
#   ./scripts/yolo11-bench.sh --adaround      # XINT8 + AdaRound
#   ./scripts/yolo11-bench.sh --runs 100      # 100 timed iterations
#

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ADAROUND=0 SKIP_QUANT=0 LIMIT=64 RUNS=50 SOURCE="assets/test_image.jpg"
while [ $# -gt 0 ]; do
    case "$1" in
        --adaround)   ADAROUND=1 ;;
        --skip-quant) SKIP_QUANT=1 ;;
        --limit)      LIMIT="$2"; shift ;;
        --runs)       RUNS="$2"; shift ;;
        --source)     SOURCE="$2"; shift ;;
        -h|--help)    usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)            die "unknown flag $1" ;;
    esac
    shift
done

TAG=$([ "$ADAROUND" = 1 ] && echo xint8_adaround || echo xint8)
BASE="models/yolo11n.onnx"
CUT="models/yolo11n_cut.onnx"
QUANT="models/yolo11n_cut_${TAG}.onnx"
CACHE_KEY="yolo11cutcachekey"

need_dir data/coco_calib "run: python pipelines/yolov8n/2_fetch_coco.py --n-calib 300"
need_file "$SOURCE"

# --------------------------------------------------------------- 0. export
if [ ! -f "$BASE" ]; then
    step "0/5  exporting yolo11n to ONNX"
    use_env resnet_env
    python pipelines/yolov11/1_export.py --size 640
    need_file "$BASE" "export did not produce $BASE"
fi

# --------------------------------------------------------------- 1. cut head
if [ ! -f "$CUT" ]; then
    step "1/5  cutting detect head decode tail -> $CUT"
    use_env resnet_env
    python pipelines/yolov11/1b_cut_head.py --in "$BASE" --out "$CUT"
    need_file "$CUT" "cut did not produce $CUT"
fi

# --------------------------------------------------------------- 2. quantize
if [ "$SKIP_QUANT" = 1 ]; then
    step "2/5  skipping quantize (--skip-quant)"
    need_file "$QUANT" "--skip-quant passed but $QUANT does not exist"
else
    step "2/5  quantizing with Quark XINT8 ($LIMIT images) -> $QUANT"
    # Quark disk safety check
    disk_guard $((LIMIT * 110))
    use_env resnet_env
    EXTRA=()
    [ "$ADAROUND" = 1 ] && EXTRA=(--adaround)
    python pipelines/yolov11/3b_quantize_cut.py --in "$CUT" --out "$QUANT" \
        --limit "$LIMIT" "${EXTRA[@]}"
    need_file "$QUANT" "quantization did not produce $QUANT"
fi

# --------------------------------------------------------------- 3. CPU sanity & FP32
step "3/5  CPU sanity: cut float model ($RUNS runs)"
use_env resnet_env17
LOG_CPU_FP32="results/lat_yolo11n_cut_fp32_cpu.log"
run_logged "$LOG_CPU_FP32" python pipelines/yolov11/4_detect.py \
    --model "$CUT" --ep cpu --source "$SOURCE" --runs "$RUNS"

# --------------------------------------------------------------- 4. DirectML (iGPU)
step "4/5  DirectML (iGPU) baseline ($RUNS runs)"
use_env resnet_env17
LOG_DML="results/lat_yolo11n_cut_fp32_dml.log"
run_logged "$LOG_DML" python pipelines/yolov11/4_detect.py \
    --model "$CUT" --ep dml --source "$SOURCE" --runs "$RUNS"

# --------------------------------------------------------------- 5. NPU XINT8
step "5/5  NPU inference: cut XINT8 ($RUNS runs, --fresh)"
npu_env
LOG_NPU="results/lat_yolo11n_cut_${TAG}_npu.log"
run_logged "$LOG_NPU" python pipelines/yolov11/4_detect.py \
    --model "$QUANT" --ep npu --source "$SOURCE" --runs "$RUNS" \
    --cache-key "$CACHE_KEY" --fresh --log 1

# --------------------------------------------------------------- 6. Diag EP
step "6/6  checking VitisAI EP partition report"
LOG_DIAG="results/diag_yolo11n_cut_${TAG}.log"
run_logged "$LOG_DIAG" python tools/diag_ep.py --cache-key "$CACHE_KEY"
npu_verdict "$CACHE_KEY" || true

# Extract latencies for summary
extract_lat() {
    grep -oE 'infer +mean +[0-9.]+ ms' "$1" | head -1 | grep -oE '[0-9.]+' || echo "?"
}

LAT_CPU=$(extract_lat "$LOG_CPU_FP32")
LAT_DML=$(extract_lat "$LOG_DML")
LAT_NPU=$(extract_lat "$LOG_NPU")

printf '\n%s================ RESULTS: YOLOv11n (640x640) ================%s\n' "$BOLD" "$OFF"
printf '%-20s %-10s %-15s %s\n' "Backend" "Precision" "Graph" "Latency"
printf '%-20s %-10s %-15s %s ms\n' "CPU (Zen 4)" "FP32" "head-cut" "$LAT_CPU"
printf '%-20s %-10s %-15s %s ms\n' "DirectML (780M)" "FP32" "head-cut" "$LAT_DML"
printf '%-20s %-10s %-15s %s ms\n' "XDNA1 NPU" "XINT8" "head-cut" "$LAT_NPU"
echo
echo "Logs written:"
echo "  CPU FP32:  $LOG_CPU_FP32"
echo "  DML FP32:  $LOG_DML"
echo "  NPU XINT8: $LOG_NPU"
echo "  EP Diag:   $LOG_DIAG"
