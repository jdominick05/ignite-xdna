#!/usr/bin/env bash
# SESR super-resolution pipeline benchmark on AMD XDNA1 NPU vs CPU vs iGPU.
#
#   ./scripts/sesr-bench.sh               # full benchmark: export, quantize, eval NPU, CPU, DML
#   ./scripts/sesr-bench.sh --eval-only   # evaluate existing compiled models
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
    step "1. Export clean NCHW SESR-M7 FP32 ONNX graph"
    use_env resnet_env
    run_logged "results/export_sesr_m7.log" python pipelines/sesr/1_export.py

    step "2. Prepare calibration crops and fetch Set5/Set14 validation benchmarks"
    run_logged "results/fetch_sesr_data.log" python pipelines/sesr/2_fetch_data.py

    step "3a. Quantize SESR-M7 to plain XINT8"
    run_logged "results/quant_sesr_m7_xint8.log" python pipelines/sesr/3_quantize.py \
        --in models/sesr_m7_fp32.onnx --calib-dir data/sesr_calib --limit 100 \
        --out models/sesr_m7_xint8.onnx

    step "3b. Quantize SESR-M7 with AdaRound FastFinetune"
    run_logged "results/quant_sesr_m7_xint8_adaround.log" python pipelines/sesr/3_quantize.py \
        --in models/sesr_m7_fp32.onnx --calib-dir data/sesr_calib --limit 100 \
        --adaround --iters 200 --out models/sesr_m7_xint8_adaround.onnx
fi

npu_env

step "4. Benchmark single-tile latency on Zen 4 CPU (FP32)"
run_logged "results/lat_sesr_m7_cpu.log" python pipelines/sesr/4_upscale.py \
    --ep cpu --model models/sesr_m7_fp32.onnx --n 50

step "5. Benchmark single-tile latency on Radeon 780M iGPU (DirectML FP32)"
run_logged "results/lat_sesr_m7_dml.log" python pipelines/sesr/4_upscale.py \
    --ep dml --model models/sesr_m7_fp32.onnx --n 50

step "6. Benchmark single-tile latency on Phoenix XDNA1 NPU (plain XINT8)"
run_logged "results/lat_sesr_m7_xint8_npu.log" python pipelines/sesr/4_upscale.py \
    --ep npu --model models/sesr_m7_xint8.onnx --n 50 --fresh

step "7. Benchmark single-tile latency on Phoenix XDNA1 NPU (XINT8 + AdaRound)"
run_logged "results/lat_sesr_m7_adaround_npu.log" python pipelines/sesr/4_upscale.py \
    --ep npu --model models/sesr_m7_xint8_adaround.onnx --n 50 --fresh

step "8a. Diagnose VitisAI EP node placement for plain XINT8"
run_logged "results/diag_sesr_m7_xint8.log" python tools/diag_ep.py \
    --cache-key sesrcachekey

step "8b. Diagnose VitisAI EP node placement for AdaRound"
run_logged "results/diag_sesr_m7_adaround.log" python tools/diag_ep.py \
    --cache-key sesradaroundcachekey

step "9a. Quantitative fidelity evaluation on Set5 benchmark (NPU)"
run_logged "results/eval_sesr_m7_set5_npu.log" python pipelines/sesr/5_eval.py \
    --ep npu --dataset Set5

step "9b. Quantitative fidelity evaluation on Set14 benchmark (NPU)"
run_logged "results/eval_sesr_m7_set14_npu.log" python pipelines/sesr/5_eval.py \
    --ep npu --dataset Set14

step "10a. Quantitative fidelity evaluation on Set5 benchmark (DML)"
run_logged "results/eval_sesr_m7_set5_dml.log" python pipelines/sesr/5_eval.py \
    --ep dml --dataset Set5

step "10b. Quantitative fidelity evaluation on Set14 benchmark (DML)"
run_logged "results/eval_sesr_m7_set14_dml.log" python pipelines/sesr/5_eval.py \
    --ep dml --dataset Set14

ok "SESR super-resolution pipeline benchmark complete. Logs written to results/."
