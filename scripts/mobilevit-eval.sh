#!/usr/bin/env bash
# Reproduce the MobileViT-XXS quantization table: FP32 vs three XINT8 variants,
# all on CPU, all over the same eval images.
#
#   ./scripts/mobilevit-eval.sh            # 1000 images, the published numbers
#   ./scripts/mobilevit-eval.sh --n 200    # quick pass
#   ./scripts/mobilevit-eval.sh --slice    # also run the 100-image slice, to
#                                          # show why a slice is not the answer
#
# Every row is CPU. There is no NPU row here on purpose: the quantized models
# collapse on CPU already, so an NPU run would only measure a broken model
# faster. Fix the accuracy first, then add an --ep npu row.
#
# Expected (1000 images, Desktop 2 / Phoenix, Ryzen 7 8700G):
#   FP32                  68.30% / 88.20%
#   Full XINT8             0.00% /  0.00%
#   Hybrid XINT8           0.00% /  0.00%
#   Hybrid + AdaRound      0.80% /  2.50%
#
# The collapse is real and its mechanism is static -- see
# tools/audit_quant_grid.py, which reads it straight out of the .onnx files.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

N=1000 SLICE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --n)       N="$2"; shift ;;
        --slice)   SLICE=1 ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         die "unknown flag $1" ;;
    esac
    shift
done

CFG=models/preprocess_config_mobilevit_xxs.json

need_dir  data/eval "run: python pipelines/resnet50/2_fetch_imagenet.py"
need_file "$CFG"    "run: python pipelines/mobilevit/1_export.py"
need_file models/mobilevit_xxs_fp32.onnx
need_file models/mobilevit_xxs_xint8.onnx
need_file models/mobilevit_xxs_hybrid_xint8.onnx
need_file models/mobilevit_xxs_hybrid_adaround.onnx

npu_env
RESULTS=()

# bench <label> <slug> <model> <n>
bench() {
    local label="$1" slug="$2" model="$3" n="$4"
    step "$label  (cpu, $n images)"
    local log="results/mobilevit/eval_${slug}_cpu.log"
    run_logged "$log" python pipelines/resnet50/4_run.py \
        --ep cpu --images data/eval --n "$n" --model "$model" --cfg-path "$CFG"
    local acc lat
    acc=$(grep -oE 'top-1: *[0-9.]+% *top-5: *[0-9.]+%' "$log" | head -1 || echo "")
    lat=$(grep -oE 'latency/image +mean +[0-9.]+ ms' "$log" | grep -oE '[0-9.]+ ms' || echo "?")
    RESULTS+=("$(printf '%-24s %-6s %-34s %s' "$label" "$n" "${acc:-n/a}" "$lat")")
}

bench "FP32"              fp32            models/mobilevit_xxs_fp32.onnx           "$N"
bench "Full XINT8"        xint8           models/mobilevit_xxs_xint8.onnx          "$N"
bench "Hybrid XINT8"      hybrid_xint8    models/mobilevit_xxs_hybrid_xint8.onnx   "$N"
bench "Hybrid + AdaRound" hybrid_adaround models/mobilevit_xxs_hybrid_adaround.onnx "$N"

# The slice trap, measured rather than asserted: the same FP32 weights read
# ~7 points higher over the first 100 images than over the full 1000. This is
# the same failure as the yolov8s AdaRound 45.19-on-a-slice vs 39.98-on-5000.
if [ "$SLICE" = 1 ]; then
    bench "FP32 (100-image slice)" fp32_slice100 models/mobilevit_xxs_fp32.onnx 100
fi

printf '\n%s================ RESULTS ================%s\n' "$BOLD" "$OFF"
printf '%-24s %-6s %-34s %s\n' "MODEL" "N" "ACCURACY" "LATENCY/IMG"
printf '%s\n' "${RESULTS[@]}"
echo
echo "  Published mobilevit_xxs reference: ~69.0% top-1 (paper)"
echo "  logs: results/mobilevit/eval_*.log"
