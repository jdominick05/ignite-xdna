#!/usr/bin/env bash
# COCO mAP evaluation for YOLOv11n on val2017 (5000 images).
#
# Evaluates:
#   1. yolo11n_cut.onnx (FP32 CPU baseline)
#   2. yolo11n_cut_xint8.onnx (stock XINT8 on NPU, 6-node placement)
#   3. yolo11n_no_c2psa_cut_xint8.onnx (ablated monolithic DPU XINT8 on NPU)
#
#   ./scripts/yolo11-eval.sh           # full 5000 images
#   ./scripts/yolo11-eval.sh --n 500   # quick slice
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

need_dir  data/coco/val2017                          "run: pipelines/yolov8n/2_fetch_coco.py"
need_file data/coco/annotations/instances_val2017.json

npu_env
python -c "import pycocotools" 2>/dev/null \
    || die "pycocotools missing. In this env: pip install pycocotools"

# 1. FP32 CPU Baseline
step "1/3  Evaluating stock yolo11n_cut.onnx (FP32, CPU)"
LOG_CPU="results/eval_yolo11n_cut_fp32_cpu.log"
run_logged "$LOG_CPU" python pipelines/yolov11/5_eval_map.py \
    --model models/yolo11n_cut.onnx --ep cpu --n "$N" \
    --dets results/dets_yolo11n_cut_fp32_cpu.json

# 2. Stock XINT8 on NPU
step "2/3  Evaluating stock yolo11n_cut_xint8.onnx (XINT8, NPU)"
LOG_NPU_STOCK="results/eval_yolo11n_cut_xint8_npu.log"
run_logged "$LOG_NPU_STOCK" python pipelines/yolov11/5_eval_map.py \
    --model models/yolo11n_cut_xint8.onnx --ep npu --n "$N" \
    --cache-key yolo11cutcachekey \
    --dets results/dets_yolo11n_cut_xint8_npu.json

# 3. Ablated XINT8 on NPU
step "3/3  Evaluating ablated yolo11n_no_c2psa_cut_xint8.onnx (XINT8, NPU)"
LOG_NPU_ABLATED="results/eval_yolo11n_no_c2psa_cut_xint8_npu.log"
run_logged "$LOG_NPU_ABLATED" python pipelines/yolov11/5_eval_map.py \
    --model models/yolo11n_no_c2psa_cut_xint8.onnx --ep npu --n "$N" \
    --cache-key yolo11noc2psacachekey \
    --dets results/dets_yolo11n_no_c2psa_cut_xint8_npu.json

# Summary Table
parse_map() {
    grep -E '^mAP@50-95' "$1" | head -1 | awk '{print $2}' || echo "?"
}
parse_map50() {
    grep -E '^mAP@50 ' "$1" | head -1 | awk '{print $2}' || echo "?"
}
parse_lat() {
    grep -E '^latency' "$1" | head -1 | awk '{print $2}' || echo "?"
}

MAP_CPU=$(parse_map "$LOG_CPU")
MAP50_CPU=$(parse_map50 "$LOG_CPU")
LAT_CPU=$(parse_lat "$LOG_CPU")

MAP_NPU_STOCK=$(parse_map "$LOG_NPU_STOCK")
MAP50_NPU_STOCK=$(parse_map50 "$LOG_NPU_STOCK")
LAT_NPU_STOCK=$(parse_lat "$LOG_NPU_STOCK")

MAP_NPU_ABLATED=$(parse_map "$LOG_NPU_ABLATED")
MAP50_NPU_ABLATED=$(parse_map50 "$LOG_NPU_ABLATED")
LAT_NPU_ABLATED=$(parse_lat "$LOG_NPU_ABLATED")

printf '\n%s================ COCO val2017 EVALUATION RESULTS ================%s\n' "$BOLD" "$OFF"
printf '%-30s %-5s %-10s %-12s %-10s %s\n' "Model" "EP" "mAP@50-95" "mAP@50" "Latency" "NPU Placement"
printf '%-30s %-5s %-10s %-12s %-10s %s\n' "YOLOv11n (Stock, FP32)" "CPU" "$MAP_CPU" "$MAP50_CPU" "${LAT_CPU} ms" "n/a (Host CPU)"
printf '%-30s %-5s %-10s %-12s %-10s %s\n' "YOLOv11n (Stock, XINT8)" "NPU" "$MAP_NPU_STOCK" "$MAP50_NPU_STOCK" "${LAT_NPU_STOCK} ms" "6 / 1300 nodes (Fractured)"
printf '%-30s %-5s %-10s %-12s %-10s %s\n' "YOLOv11n (No-C2PSA, XINT8)" "NPU" "$MAP_NPU_ABLATED" "$MAP50_NPU_ABLATED" "${LAT_NPU_ABLATED} ms" "1173 / 1180 nodes (Monolithic)"
echo
echo "Logs written:"
echo "  $LOG_CPU"
echo "  $LOG_NPU_STOCK"
echo "  $LOG_NPU_ABLATED"
