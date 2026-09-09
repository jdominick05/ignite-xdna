#!/usr/bin/env bash
# Real-ESRGAN 4x super-resolution pipeline benchmark on AMD XDNA1 NPU vs CPU vs iGPU.
#
#   ./scripts/realesrgan-bench.sh               # full benchmark: export, quantize, eval NPU, CPU, DML
#   ./scripts/realesrgan-bench.sh --eval-only   # evaluate existing compiled models
#
# Logs written to results/ (UTF-8). Run from Git Bash on Windows.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

EVAL_ONLY=0
RES=64
ARCH="compact"

while [ $# -gt 0 ]; do
    case "$1" in
        --eval-only) EVAL_ONLY=1 ;;
        --res)       RES="$2"; shift ;;
        --arch)      ARCH="$2"; shift ;;
        -h|--help)   usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)           die "unknown flag $1" ;;
    esac
    shift
done

MODEL_PREFIX="realesrgan_${ARCH}_r${RES}"

if [ "$EVAL_ONLY" = 0 ]; then
    step "1. Export clean NCHW Real-ESRGAN (${ARCH} at ${RES}x${RES}) FP32 ONNX graph"
    use_env resnet_env
    run_logged "results/export_${MODEL_PREFIX}.log" python pipelines/realesrgan/1_export.py \
        --arch "${ARCH}" --res "${RES}"

    step "2. Prepare calibration crops and fetch Set5/Set14 validation benchmarks"
    run_logged "results/fetch_realesrgan_data.log" python pipelines/realesrgan/2_fetch_data.py \
        --crop-size "${RES}" --n-crops 100

    step "3a. Quantize Real-ESRGAN (${ARCH} r${RES}) to plain XINT8"
    run_logged "results/quant_${MODEL_PREFIX}_xint8.log" python pipelines/realesrgan/3_quantize.py \
        --in "models/${MODEL_PREFIX}_fp32.onnx" --calib-dir data/realesrgan_calib --limit 100 \
        --out "models/${MODEL_PREFIX}_xint8.onnx"

    step "3b. Quantize Real-ESRGAN (${ARCH} r${RES}) with AdaRound FastFinetune"
    run_logged "results/quant_${MODEL_PREFIX}_xint8_adaround.log" python pipelines/realesrgan/3_quantize.py \
        --in "models/${MODEL_PREFIX}_fp32.onnx" --calib-dir data/realesrgan_calib --limit 100 \
        --adaround --iters 200 --out "models/${MODEL_PREFIX}_xint8_adaround.onnx"
fi

npu_env

step "4. Benchmark single-tile latency on Zen 4 CPU (FP32)"
run_logged "results/lat_${MODEL_PREFIX}_cpu.log" python pipelines/realesrgan/4_upscale.py \
    --ep cpu --model "models/${MODEL_PREFIX}_fp32.onnx" --n 50

step "5. Benchmark single-tile latency on Radeon 780M iGPU (DirectML FP32)"
run_logged "results/lat_${MODEL_PREFIX}_dml.log" python pipelines/realesrgan/4_upscale.py \
    --ep dml --model "models/${MODEL_PREFIX}_fp32.onnx" --n 50

step "6. Benchmark single-tile latency on Phoenix XDNA1 NPU (plain XINT8)"
run_logged "results/lat_${MODEL_PREFIX}_xint8_npu.log" python pipelines/realesrgan/4_upscale.py \
    --ep npu --model "models/${MODEL_PREFIX}_xint8.onnx" --n 50 --fresh

step "7. Benchmark single-tile latency on Phoenix XDNA1 NPU (XINT8 + AdaRound)"
run_logged "results/lat_${MODEL_PREFIX}_adaround_npu.log" python pipelines/realesrgan/4_upscale.py \
    --ep npu --model "models/${MODEL_PREFIX}_xint8_adaround.onnx" --n 50 --fresh

CACHE_KEY="realesrgan_${ARCH}_${RES}_cache"

step "8a. Diagnose VitisAI EP node placement for plain XINT8"
run_logged "results/diag_${MODEL_PREFIX}_xint8.log" python tools/diag_ep.py \
    --cache-key "${CACHE_KEY}"

step "9a. Quantitative fidelity evaluation on Set5 benchmark (NPU)"
run_logged "results/eval_${MODEL_PREFIX}_set5_npu.log" python pipelines/realesrgan/5_eval.py \
    --ep npu --dataset Set5 --arch "${ARCH}" --res "${RES}"

step "9b. Quantitative fidelity evaluation on Set14 benchmark (NPU)"
run_logged "results/eval_${MODEL_PREFIX}_set14_npu.log" python pipelines/realesrgan/5_eval.py \
    --ep npu --dataset Set14 --arch "${ARCH}" --res "${RES}"

step "10a. Quantitative fidelity evaluation on Set5 benchmark (DML)"
run_logged "results/eval_${MODEL_PREFIX}_set5_dml.log" python pipelines/realesrgan/5_eval.py \
    --ep dml --dataset Set5 --arch "${ARCH}" --res "${RES}"

step "10b. Quantitative fidelity evaluation on Set14 benchmark (DML)"
run_logged "results/eval_${MODEL_PREFIX}_set14_dml.log" python pipelines/realesrgan/5_eval.py \
    --ep dml --dataset Set14 --arch "${ARCH}" --res "${RES}"

step "11. Visual reconstruction comparison on butterfly.png"
python pipelines/realesrgan/4_upscale.py --ep cpu --model "models/${MODEL_PREFIX}_fp32.onnx" \
    --image data/realesrgan_val/Set5_LR_x4/butterfly.png --out results/butterfly_realesr_cpu.png
python pipelines/realesrgan/4_upscale.py --ep dml --model "models/${MODEL_PREFIX}_fp32.onnx" \
    --image data/realesrgan_val/Set5_LR_x4/butterfly.png --out results/butterfly_realesr_dml.png
python pipelines/realesrgan/4_upscale.py --ep npu --model "models/${MODEL_PREFIX}_xint8_adaround.onnx" \
    --image data/realesrgan_val/Set5_LR_x4/butterfly.png --out results/butterfly_realesr_npu.png

ok "Real-ESRGAN super-resolution pipeline benchmark complete. Logs written to results/."
