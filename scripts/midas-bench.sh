#!/usr/bin/env bash
# MiDaS monocular depth estimation pipeline benchmark on AMD XDNA1 NPU vs CPU vs iGPU.
#
#   ./scripts/midas-bench.sh               # full benchmark: export, quantize, eval NPU & CPU
#   ./scripts/midas-bench.sh --eval-only   # evaluate existing compiled models
#
# Logs written to results/ (UTF-8). Run from Git Bash on Windows.

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
    step "1. Export MiDaS FP32 and head-cut ONNX graphs"
    use_env resnet_env
    run_logged "results/export_midas_small.log" python pipelines/midas/1_export.py

    step "2. Prepare calibration and validation data"
    run_logged "results/fetch_midas_data.log" python pipelines/midas/2_fetch_data.py --calib-count 300 --val-count 50

    step "3a. Quantize MiDaS stock bilinear to XINT8"
    run_logged "results/quant_midas_small_bilinear_xint8.log" python pipelines/midas/3_quantize.py \
        --in models/midas_small_cut.onnx --calib-dir data/midas_calib --limit 300

    step "3b. Quantize MiDaS nearest-neighbor to XINT8"
    run_logged "results/quant_midas_small_nearest_xint8.log" python pipelines/midas/3_quantize.py \
        --in models/midas_small_nearest_cut.onnx --calib-dir data/midas_calib --limit 300
fi

npu_env

step "4. Benchmark single-image latency on CPU FP32"
run_logged "results/lat_midas_small_cpu.log" python pipelines/midas/4_depth.py \
    --ep cpu --model models/midas_small_cut.onnx --n 50 --out results/midas_depth_cpu.jpg

step "5. Benchmark single-image latency on Radeon 780M iGPU (DirectML FP32)"
run_logged "results/lat_midas_small_dml.log" python pipelines/midas/4_depth.py \
    --ep dml --model models/midas_small_cut.onnx --n 50 --out results/midas_depth_dml.jpg

step "6a. Benchmark single-image latency on Phoenix XDNA1 NPU (XINT8 stock bilinear, 5 subgraphs)"
run_logged "results/lat_midas_small_bilinear_xint8_npu.log" python pipelines/midas/4_depth.py \
    --ep npu --model models/midas_small_cut_xint8.onnx --n 50 --fresh --out results/midas_depth_bilinear_npu.jpg

step "6b. Benchmark single-image latency on Phoenix XDNA1 NPU (XINT8 nearest, 1 subgraph)"
run_logged "results/lat_midas_small_nearest_xint8_npu.log" python pipelines/midas/4_depth.py \
    --ep npu --model models/midas_small_nearest_cut_xint8.onnx --n 50 --fresh --out results/midas_depth_npu.jpg

step "7a. Diagnose VitisAI EP node placement for stock bilinear"
run_logged "results/diag_midas_small_bilinear_xint8.log" python tools/diag_ep.py \
    --cache-key midascachekey

step "7b. Diagnose VitisAI EP node placement for nearest-neighbor"
run_logged "results/diag_midas_small_nearest_xint8.log" python tools/diag_ep.py \
    --cache-key midasnearestcache

step "8a. Quantitative depth fidelity evaluation for stock bilinear (50 validation scenes)"
run_logged "results/eval_midas_small_bilinear_xint8_npu.log" python pipelines/midas/5_eval.py \
    --ref-model models/midas_small_cut.onnx \
    --test-model models/midas_small_cut_xint8.onnx \
    --val-dir data/midas_val --ep npu --n 50

step "8b. Quantitative depth fidelity evaluation for nearest-neighbor (50 validation scenes)"
run_logged "results/eval_midas_small_nearest_xint8_npu.log" python pipelines/midas/5_eval.py \
    --ref-model models/midas_small_cut.onnx \
    --test-model models/midas_small_nearest_cut_xint8.onnx \
    --val-dir data/midas_val --ep npu --n 50

ok "MiDaS pipeline benchmark complete. Logs written to results/."
