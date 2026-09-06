#!/usr/bin/env bash
# Reproduce the ResNet50 results table: FP32 on CPU, XINT8 on NPU,
# XINT8+AdaRound on NPU. All three over the same eval images.
#
#   ./scripts/resnet-bench.sh              # 1000 images, the published numbers
#   ./scripts/resnet-bench.sh --n 200      # quick pass
#   ./scripts/resnet-bench.sh --cpu-int8   # also time INT8 on CPU (slow, ~40 ms)
#
# Expected (1000 images):
#   FP32           CPU   80.10% / 93.90%   19.7 ms
#   XINT8          NPU   71.70% / 88.40%    5.63 ms
#   XINT8+AdaRound NPU   79.80% / 92.50%    6.93 ms

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

N=1000 CPU_INT8=0
while [ $# -gt 0 ]; do
    case "$1" in
        --n)        N="$2"; shift ;;
        --cpu-int8) CPU_INT8=1 ;;
        -h|--help)  usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)          die "unknown flag $1" ;;
    esac
    shift
done

need_dir  data/eval                          "run: python pipelines/resnet50/2_fetch_imagenet.py"
need_file models/preprocess_config.json      "run: python pipelines/resnet50/1_export.py"
need_file models/resnet50_fp32.onnx
need_file models/resnet50_int8.onnx
need_file models/resnet50_xint8_adaround.onnx

npu_env
RESULTS=()

# bench <label> <ep> <model> <extra args...>
bench() {
    local label="$1" ep="$2" model="$3"; shift 3
    step "$label  ($ep, $N images)"
    local log="results/bench_$(echo "$label" | tr ' +' '__' | tr 'A-Z' 'a-z')_${ep}.log"
    run_logged "$log" python pipelines/resnet50/4_run.py \
        --ep "$ep" --images data/eval --n "$N" --model "$model" "$@"
    local acc lat
    acc=$(grep -oE 'top-1: *[0-9.]+% *top-5: *[0-9.]+%' "$log" | head -1 || echo "")
    lat=$(grep -oE 'mean +[0-9.]+ ms' "$log" | head -1 | grep -oE '[0-9.]+' || echo "?")
    RESULTS+=("$(printf '%-22s %-4s %-34s %s ms' "$label" "$ep" "${acc:-n/a}" "$lat")")
}

# --fresh on every NPU run: the cache is keyed by cacheKey, not by model hash,
# so the second NPU row would otherwise reuse the first one's compile.
bench "FP32"           cpu models/resnet50_fp32.onnx
[ "$CPU_INT8" = 1 ] && bench "XINT8" cpu models/resnet50_int8.onnx
bench "XINT8"          npu models/resnet50_int8.onnx --fresh
bench "XINT8+AdaRound" npu models/resnet50_xint8_adaround.onnx --fresh

printf '\n%s================ RESULTS ================%s\n' "$BOLD" "$OFF"
printf '%s\n' "${RESULTS[@]}"
echo
echo "  FP32 reference for resnet50.a1_in1k: 80.4% top-1"
echo "  logs: results/bench_*.log"
npu_verdict modelcachekey || true
