#!/usr/bin/env bash
# FastDepth benchmark runner across CPU FP32, DirectML (Radeon 780M iGPU), and Ryzen AI NPU.
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
    step "1. Export FastDepth FP32 ONNX graph"
    use_env resnet_env
    run_logged "results/export_fastdepth.log" python pipelines/fastdepth/1_export.py

    step "2. Prepare calibration and validation data"
    run_logged "results/fetch_fastdepth_data.log" python pipelines/fastdepth/2_fetch_data.py --calib-count 300 --val-count 50

    step "3. Quantize FastDepth to XINT8"
    run_logged "results/quant_fastdepth_xint8.log" python pipelines/fastdepth/3_quantize.py \
        --calib-dir data/fastdepth_calib --limit 300
fi

npu_env

step "4. Benchmark single-image latency on CPU FP32"
run_logged "results/lat_fastdepth_cpu.log" python pipelines/fastdepth/4_depth.py \
    --ep cpu --model models/fastdepth_fp32.onnx --n 50 --out results/fastdepth_depth_cpu.jpg

step "5. Benchmark single-image latency on Radeon 780M iGPU (DirectML FP32)"
run_logged "results/lat_fastdepth_dml.log" python pipelines/fastdepth/4_depth.py \
    --ep dml --model models/fastdepth_fp32.onnx --n 50 --out results/fastdepth_depth_dml.jpg

step "6. Benchmark single-image latency on Phoenix XDNA1 NPU (XINT8, 1 subgraph)"
run_logged "results/lat_fastdepth_xint8_npu.log" python pipelines/fastdepth/4_depth.py \
    --ep npu --model models/fastdepth_fp32_xint8.onnx --n 50 --fresh --out results/fastdepth_depth_npu.jpg

step "7. Diagnose VitisAI EP node placement"
run_logged "results/diag_fastdepth_xint8.log" python tools/diag_ep.py \
    --cache-key fastdepthcachekey

step "8a. Quantitative depth fidelity evaluation on NPU (50 validation scenes)"
run_logged "results/eval_fastdepth_xint8_npu.log" python pipelines/fastdepth/5_eval.py \
    --ref-model models/fastdepth_fp32.onnx \
    --test-model models/fastdepth_fp32_xint8.onnx \
    --val-dir data/fastdepth_val --ep npu --n 50

step "8b. Quantitative depth fidelity evaluation on CPU (50 validation scenes)"
run_logged "results/eval_fastdepth_xint8_cpu.log" python pipelines/fastdepth/5_eval.py \
    --ref-model models/fastdepth_fp32.onnx \
    --test-model models/fastdepth_fp32_xint8.onnx \
    --val-dir data/fastdepth_val --ep cpu --n 50

ok "FastDepth pipeline benchmark complete. Logs written to results/."
