#!/usr/bin/env bash
# Reproduce the RegNetX-002 results table: FP32 on CPU, XINT8 on NPU.
#
#   ./scripts/regnet-bench.sh              # 1000 images
#   ./scripts/regnet-bench.sh --n 200      # quick pass
#

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

N=1000
while [ $# -gt 0 ]; do
    case "$1" in
        --n)        N="$2"; shift ;;
        -h|--help)  usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)          die "unknown flag $1" ;;
    esac
    shift
done

need_dir  data/eval                          "run: python pipelines/resnet50/2_fetch_imagenet.py"
need_file models/preprocess_config_regnetx002.json "run: python pipelines/resnet50/1_export.py --model regnetx_002.pycls_in1k --out models/regnetx_002_fp32.onnx --cfg-out models/preprocess_config_regnetx002.json"
need_file models/regnetx_002_fp32.onnx

npu_env
RESULTS=()

# bench <label> <ep> <model> <extra args...>
bench() {
    local label="$1" ep="$2" model="$3"; shift 3
    step "$label  ($ep, $N images)"
    local log="results/res/run_regnetx_002_$(echo "$label" | tr ' +' '__' | tr 'A-Z' 'a-z')_${ep}.log"
    run_logged "$log" python pipelines/resnet50/4_run.py \
        --ep "$ep" --images data/eval --n "$N" --model "$model" \
        --cfg-path models/preprocess_config_regnetx002.json \
        --cache-key regnetxcachekey "$@"
    local acc lat
    acc=$(grep -oE 'top-1: *[0-9.]+% *top-5: *[0-9.]+%' "$log" | head -1 || echo "")
    lat=$(grep -oE 'mean +[0-9.]+ ms' "$log" | head -1 | grep -oE '[0-9.]+' || echo "?")
    RESULTS+=("$(printf '%-22s %-4s %-34s %s ms' "$label" "$ep" "${acc:-n/a}" "$lat")")
}

bench "FP32" cpu models/regnetx_002_fp32.onnx
if [ -f models/regnetx_002_xint8.onnx ]; then
    bench "XINT8" npu models/regnetx_002_xint8.onnx --fresh
    run_logged "results/res/diag_regnetx_002_xint8.log" python tools/diag_ep.py --cache-key regnetxcachekey
    npu_verdict regnetxcachekey || true
fi

printf '\n%s================ RESULTS ================%s\n' "$BOLD" "$OFF"
printf '%s\n' "${RESULTS[@]}"
echo
echo "  FP32 reference for regnetx_002.pycls_in1k: 68.75% top-1"
echo "  logs: results/res/run_regnetx_002_*.log"
