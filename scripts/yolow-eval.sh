#!/usr/bin/env bash
# COCO mAP evaluation for YOLO-World v2 on val2017 (5000 images).
#
# Evaluates:
#   1. yolov8s-worldv2_cut.onnx (FP32 CPU baseline)
#   2. yolov8s-worldv2_cut_xint8.onnx (stock XINT8 on NPU)
#   3. yolov8s-worldv2_no_attn_cut_xint8.onnx (ablated XINT8 on NPU)
#
#   ./scripts/yolow-eval.sh           # full 5000 images
#   ./scripts/yolow-eval.sh --n 500   # quick slice
#

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

N=0
while [ $# -gt 0 ]; do
    case "$1" in
        --n)       N="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         die "unknown flag $1" ;;
    esac
    shift
done

need_dir  data/coco/val2017                          "run: pipelines/yolow/2_fetch_coco.py"
need_file data/coco/annotations/instances_val2017.json

npu_env
python -c "import pycocotools" 2>/dev/null \
    || die "pycocotools missing. In this env: pip install pycocotools"

# 1. FP32 CPU Baseline
step "1/3  Evaluating stock yolov8s-worldv2_cut.onnx (FP32, CPU)"
LOG_CPU="results/eval_yolow_cut_fp32_cpu.log"
run_logged "$LOG_CPU" python pipelines/yolow/5_eval_map.py \
    --model models/yolov8s-worldv2_cut.onnx --ep cpu --n "$N" \
    --dets results/dets_yolow_cut_fp32_cpu.json

# 2. Stock XINT8 on NPU
step "2/3  Evaluating stock yolov8s-worldv2_cut_xint8.onnx (XINT8, NPU)"
LOG_NPU_STOCK="results/eval_yolow_cut_xint8_npu.log"
run_logged "$LOG_NPU_STOCK" python pipelines/yolow/5_eval_map.py \
    --model models/yolov8s-worldv2_cut_xint8.onnx --ep npu --n "$N" \
    --cache-key yolowcutcachekey \
    --dets results/dets_yolow_cut_xint8_npu.json

# 3. Ablated XINT8 on NPU
step "3/3  Evaluating ablated yolov8s-worldv2_no_attn_cut_xint8.onnx (XINT8, NPU)"
LOG_NPU_ABL="results/eval_yolow_no_attn_cut_xint8_npu.log"
run_logged "$LOG_NPU_ABL" python pipelines/yolow/5_eval_map.py \
    --model models/yolov8s-worldv2_no_attn_cut_xint8.onnx --ep npu --n "$N" \
    --cache-key yolownoattncachekey \
    --dets results/dets_yolow_no_attn_cut_xint8_npu.json

parse_map() {
    grep -E 'Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all' "$1" | head -1 | awk '{print $NF}' || echo "?"
}
parse_map50() {
    grep -E 'Average Precision  \(AP\) @\[ IoU=0.50      \| area=   all' "$1" | head -1 | awk '{print $NF}' || echo "?"
}
parse_lat() {
    grep -oE 'mean +: +[0-9.]+ ms' "$1" | head -1 | grep -oE '[0-9.]+' || echo "?"
}

MAP_CPU=$(parse_map "$LOG_CPU")
MAP50_CPU=$(parse_map50 "$LOG_CPU")
LAT_CPU=$(parse_lat "$LOG_CPU")

MAP_NPU_STOCK=$(parse_map "$LOG_NPU_STOCK")
MAP50_NPU_STOCK=$(parse_map50 "$LOG_NPU_STOCK")
LAT_NPU_STOCK=$(parse_lat "$LOG_NPU_STOCK")

MAP_NPU_ABL=$(parse_map "$LOG_NPU_ABL")
MAP50_NPU_ABL=$(parse_map50 "$LOG_NPU_ABL")
LAT_NPU_ABL=$(parse_lat "$LOG_NPU_ABL")

printf '\n%s================ COCO val2017 EVALUATION RESULTS ================%s\n' "$BOLD" "$OFF"
printf '%-32s %-5s %-10s %-12s %-10s %s\n' "Model" "EP" "mAP@50-95" "mAP@50" "Latency" "NPU Placement"
printf '%-32s %-5s %-10s %-12s %-10s %s\n' "YOLO-World v2 (Stock, FP32)" "CPU" "$MAP_CPU" "$MAP50_CPU" "${LAT_CPU} ms" "n/a (Host CPU)"
printf '%-32s %-5s %-10s %-12s %-10s %s\n' "YOLO-World v2 (Stock, XINT8)" "NPU" "$MAP_NPU_STOCK" "$MAP50_NPU_STOCK" "${LAT_NPU_STOCK} ms" "yolowcutcachekey"
printf '%-32s %-5s %-10s %-12s %-10s %s\n' "YOLO-World v2 (No-Attn, XINT8)" "NPU" "$MAP_NPU_ABL" "$MAP50_NPU_ABL" "${LAT_NPU_ABL} ms" "yolownoattncachekey"
echo
echo "Logs written:"
echo "  $LOG_CPU"
echo "  $LOG_NPU_STOCK"
echo "  $LOG_NPU_ABL"
