#!/usr/bin/env bash
# BiSeNetV2 benchmark runner across CPU FP32, DirectML (Radeon 780M iGPU), and Ryzen AI NPU.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

EVAL_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --eval-only) EVAL_ONLY=1 ;;
        -h|--help)   usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)           die "unknown flag $1" ;;
    esac
    shift
done

if [ "$EVAL_ONLY" = 0 ]; then
    step "1. Export BiSeNetV2 FP32 ONNX graphs (nearest and bilinear)"
    use_env resnet_env
    run_logged "results/export_bisenetv2.log" python pipelines/bisenetv2/1_export.py

    step "2. Prepare calibration and validation data"
    run_logged "results/fetch_bisenetv2_data.log" python pipelines/bisenetv2/2_fetch_data.py

    step "3a. Quantize BiSeNetV2 (nearest) to XINT8"
    run_logged "results/quant_bisenetv2_xint8.log" python pipelines/bisenetv2/3_quantize.py \
        --src models/bisenetv2_fp32.onnx \
        --calib-dir data/bisenetv2_calib --limit 300

    step "3b. Quantize BiSeNetV2 (bilinear) to XINT8"
    run_logged "results/quant_bisenetv2_bilinear_xint8.log" python pipelines/bisenetv2/3_quantize.py \
        --src models/bisenetv2_bilinear_fp32.onnx \
        --calib-dir data/bisenetv2_calib --limit 300
fi

npu_env

step "4. Benchmark single-image latency on CPU FP32"
run_logged "results/lat_bisenetv2_cpu.log" python pipelines/bisenetv2/4_segment.py \
    --ep cpu --model models/bisenetv2_fp32.onnx --iters 50 --out-image results/bisenetv2_segment_cpu.jpg

step "5. Benchmark single-image latency on Radeon 780M iGPU (DirectML FP32)"
run_logged "results/lat_bisenetv2_dml.log" python pipelines/bisenetv2/4_segment.py \
    --ep dml --model models/bisenetv2_fp32.onnx --iters 50 --out-image results/bisenetv2_segment_dml.jpg

step "6. Benchmark single-image latency on Phoenix XDNA1 NPU (XINT8, nearest monolithic)"
run_logged "results/lat_bisenetv2_xint8_npu.log" python pipelines/bisenetv2/4_segment.py \
    --ep npu --model models/bisenetv2_fp32_xint8.onnx --iters 50 --fresh --cache-key bisenetv2cachekey --out-image results/bisenetv2_segment_npu.jpg

step "7. Benchmark single-image latency on Phoenix XDNA1 NPU (XINT8, stock bilinear)"
run_logged "results/lat_bisenetv2_bilinear_xint8_npu.log" python pipelines/bisenetv2/4_segment.py \
    --ep npu --model models/bisenetv2_bilinear_fp32_xint8.onnx --iters 50 --fresh --cache-key bisenetv2bilinearcache

step "8a. Diagnose VitisAI EP node placement (nearest)"
run_logged "results/diag_bisenetv2_xint8.log" python tools/diag_ep.py \
    --cache-key bisenetv2cachekey

step "8b. Diagnose VitisAI EP node placement (bilinear)"
run_logged "results/diag_bisenetv2_bilinear_xint8.log" python tools/diag_ep.py \
    --cache-key bisenetv2bilinearcache

step "9a. Quantitative segmentation fidelity evaluation on NPU (50 validation scenes)"
run_logged "results/eval_bisenetv2_xint8_npu.log" python pipelines/bisenetv2/5_eval.py \
    --model models/bisenetv2_fp32_xint8.onnx \
    --ref models/bisenetv2_fp32.onnx \
    --val-dir data/bisenetv2_val --ep npu --limit 50 --cache-key bisenetv2cachekey

step "9b. Quantitative segmentation fidelity evaluation on CPU (50 validation scenes)"
run_logged "results/eval_bisenetv2_xint8_cpu.log" python pipelines/bisenetv2/5_eval.py \
    --model models/bisenetv2_fp32_xint8.onnx \
    --ref models/bisenetv2_fp32.onnx \
    --val-dir data/bisenetv2_val --ep cpu --limit 50

ok "BiSeNetV2 pipeline benchmark complete. Logs written to results/."
