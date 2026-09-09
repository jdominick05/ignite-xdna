#!/usr/bin/env bash
# YOLO-World v2 on Ryzen AI XDNA1 NPU: export -> cut -> quantize -> CPU/DML/NPU bench -> diag.
#
# Characterizes native NPU execution of YOLO-World v2's text-guided cross-attention
# (C2fAttn with MaxSigmoidAttnBlock: 5D Einsum + 5D ReduceMax) and decoupled vision backbone.
#
#   ./scripts/yolow-bench.sh                 # full pipeline (200 calib images, 50 runs)
#   ./scripts/yolow-bench.sh --skip-quant    # reuse existing quantized models
#   ./scripts/yolow-bench.sh --runs 100      # 100 timed iterations
#

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SKIP_QUANT=0 LIMIT=200 RUNS=50 SOURCE="assets/test_image.jpg"
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-quant) SKIP_QUANT=1 ;;
        --limit)      LIMIT="$2"; shift ;;
        --runs)       RUNS="$2"; shift ;;
        --source)     SOURCE="$2"; shift ;;
        -h|--help)    usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)            die "unknown flag $1" ;;
    esac
    shift
done

BASE_STOCK="models/yolov8s-worldv2.onnx"
CUT_STOCK="models/yolov8s-worldv2_cut.onnx"
QUANT_STOCK="models/yolov8s-worldv2_cut_xint8.onnx"
CACHE_STOCK="yolowcutcachekey"

BASE_ABL="models/yolov8s-worldv2_no_attn.onnx"
CUT_ABL="models/yolov8s-worldv2_no_attn_cut.onnx"
QUANT_ABL="models/yolov8s-worldv2_no_attn_cut_xint8.onnx"
CACHE_ABL="yolownoattncachekey"

need_dir data/coco_calib "run: python pipelines/yolow/2_fetch_coco.py --n-calib 300"
need_file "$SOURCE"

# --------------------------------------------------------------- 0. export
if [ ! -f "$BASE_STOCK" ] || [ ! -f "$BASE_ABL" ]; then
    step "0/6  exporting YOLO-World v2 (stock and ablated) to ONNX"
    use_env resnet_env
    if [ ! -f "$BASE_STOCK" ]; then
        python pipelines/yolow/1_export.py --size 640
        need_file "$BASE_STOCK" "export did not produce $BASE_STOCK"
    fi
    if [ ! -f "$BASE_ABL" ]; then
        python pipelines/yolow/1_export.py --size 640 --no-text-attn
        need_file "$BASE_ABL" "export did not produce $BASE_ABL"
    fi
fi

# --------------------------------------------------------------- 1. cut head
if [ ! -f "$CUT_STOCK" ] || [ ! -f "$CUT_ABL" ]; then
    step "1/6  cutting detect head decode tail -> $CUT_STOCK and $CUT_ABL"
    use_env resnet_env
    if [ ! -f "$CUT_STOCK" ]; then
        python pipelines/yolow/1b_cut_head.py --in "$BASE_STOCK" --out "$CUT_STOCK"
        need_file "$CUT_STOCK" "cut did not produce $CUT_STOCK"
    fi
    if [ ! -f "$CUT_ABL" ]; then
        python pipelines/yolow/1b_cut_head.py --in "$BASE_ABL" --out "$CUT_ABL"
        need_file "$CUT_ABL" "cut did not produce $CUT_ABL"
    fi
fi

# --------------------------------------------------------------- 2. quantize
if [ "$SKIP_QUANT" = 1 ]; then
    step "2/6  skipping quantize (--skip-quant)"
    need_file "$QUANT_STOCK" "--skip-quant passed but $QUANT_STOCK does not exist"
    need_file "$QUANT_ABL" "--skip-quant passed but $QUANT_ABL does not exist"
else
    step "2/6  quantizing with Quark XINT8 ($LIMIT images)"
    disk_guard $((LIMIT * 200))
    use_env resnet_env
    if [ ! -f "$QUANT_STOCK" ]; then
        python pipelines/yolow/3b_quantize_cut.py --in "$CUT_STOCK" --out "$QUANT_STOCK" \
            --limit "$LIMIT"
        need_file "$QUANT_STOCK" "quantization did not produce $QUANT_STOCK"
    fi
    if [ ! -f "$QUANT_ABL" ]; then
        python pipelines/yolow/3b_quantize_cut.py --in "$CUT_ABL" --out "$QUANT_ABL" \
            --limit "$LIMIT"
        need_file "$QUANT_ABL" "quantization did not produce $QUANT_ABL"
    fi
fi

# --------------------------------------------------------------- 3. CPU sanity & FP32
step "3/6  CPU sanity: cut float model ($RUNS runs)"
use_env resnet_env17
LOG_CPU_FP32="results/lat_yolow_cut_fp32_cpu.log"
run_logged "$LOG_CPU_FP32" python pipelines/yolow/4_detect.py \
    --model "$CUT_STOCK" --ep cpu --source "$SOURCE" --runs "$RUNS"

# --------------------------------------------------------------- 4. DirectML (iGPU)
step "4/6  DirectML (iGPU) baseline ($RUNS runs)"
use_env resnet_env17
LOG_DML="results/lat_yolow_cut_fp32_dml.log"
run_logged "$LOG_DML" python pipelines/yolow/4_detect.py \
    --model "$CUT_STOCK" --ep dml --source "$SOURCE" --runs "$RUNS"

# --------------------------------------------------------------- 5. NPU XINT8 (Stock & Ablated)
step "5/6  NPU inference: cut XINT8 stock & ablated ($RUNS runs, --fresh)"
npu_env

LOG_NPU_STOCK="results/lat_yolow_cut_xint8_npu.log"
run_logged "$LOG_NPU_STOCK" python pipelines/yolow/4_detect.py \
    --model "$QUANT_STOCK" --ep npu --source "$SOURCE" --runs "$RUNS" \
    --cache-key "$CACHE_STOCK" --fresh --log 1

LOG_NPU_ABL="results/lat_yolow_no_attn_cut_xint8_npu.log"
run_logged "$LOG_NPU_ABL" python pipelines/yolow/4_detect.py \
    --model "$QUANT_ABL" --ep npu --source "$SOURCE" --runs "$RUNS" \
    --cache-key "$CACHE_ABL" --fresh --log 1

# --------------------------------------------------------------- 6. Diag EP
step "6/6  checking VitisAI EP partition reports"
LOG_DIAG_STOCK="results/diag_yolow_cut_xint8.log"
run_logged "$LOG_DIAG_STOCK" python tools/diag_ep.py --cache-key "$CACHE_STOCK"
npu_verdict "$CACHE_STOCK" || true

LOG_DIAG_ABL="results/diag_yolow_no_attn_cut_xint8.log"
run_logged "$LOG_DIAG_ABL" python tools/diag_ep.py --cache-key "$CACHE_ABL"
npu_verdict "$CACHE_ABL" || true

extract_lat() {
    grep -oE 'infer +mean +[0-9.]+ ms' "$1" | head -1 | grep -oE '[0-9.]+' || echo "?"
}

LAT_CPU=$(extract_lat "$LOG_CPU_FP32")
LAT_DML=$(extract_lat "$LOG_DML")
LAT_NPU_STOCK=$(extract_lat "$LOG_NPU_STOCK")
LAT_NPU_ABL=$(extract_lat "$LOG_NPU_ABL")

printf '\n%s================ YOLO-WORLD V2 LATENCY SUMMARY ================%s\n' "$BOLD" "$OFF"
printf '%-32s %-6s %-12s %s\n' "Model" "EP" "Infer Latency" "Output / Partition"
printf '%-32s %-6s %-12s %s\n' "YOLO-World v2 (Cut, FP32)" "CPU" "${LAT_CPU} ms" "Zen 4 host CPU"
printf '%-32s %-6s %-12s %s\n' "YOLO-World v2 (Cut, FP32)" "DML" "${LAT_DML} ms" "Radeon 780M iGPU"
printf '%-32s %-6s %-12s %s\n' "YOLO-World v2 (Stock, XINT8)" "NPU" "${LAT_NPU_STOCK} ms" "$CACHE_STOCK"
printf '%-32s %-6s %-12s %s\n' "YOLO-World v2 (No-Attn, XINT8)" "NPU" "${LAT_NPU_ABL} ms" "$CACHE_ABL"
echo
echo "Logs:"
echo "  $LOG_CPU_FP32"
echo "  $LOG_DML"
echo "  $LOG_NPU_STOCK"
echo "  $LOG_NPU_ABL"
echo "  $LOG_DIAG_STOCK"
echo "  $LOG_DIAG_ABL"
