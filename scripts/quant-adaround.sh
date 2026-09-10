#!/usr/bin/env bash
# Run Ignition's AdaRound (torch) on an emitted XINT8 file; the base's sidecar supplies the CLE flag and listing.
#
#   ./scripts/quant-adaround.sh --quant models/resnet50_ignition_cle_c64.onnx --out models/resnet50_ignition_cle_adaround_c64.onnx --log results/quant/quant_resnet50_ignition_cle_adaround_c64.log [--in-model models/resnet50_fp32.onnx] [--calib-dir data/calib] [--iters 1000] [--data-size 1000] [--seed 1705472343] [--allow-busy] [--env resnet_env] [--device cpu]
#   ./scripts/quant-adaround.sh --in-model models/yolov8n_cut.onnx --calib-dir data/coco_calib --quant models/yolov8n_cut_ignition_cle_c64.onnx --out models/yolov8n_cut_ignition_cle_adaround_c64.onnx --log results/quant/quant_yolov8n_cut_ignition_cle_adaround_c64.log
#   ./scripts/quant-adaround.sh --in-model models/modnet/modnet_cut_fp32.onnx --calib-dir data/modnet_calib --cfg-path models/modnet/preprocess_config.json --quant models/modnet/modnet_cut_ignition_cle_c64.onnx --out models/modnet/modnet_cut_ignition_cle_adaround_c64.onnx --log results/quant/quant_modnet_cut_ignition_cle_adaround_c64.log
#   ./scripts/quant-adaround.sh --env resnet_env_rocm --device cuda --quant models/resnet50_ignition_cle_c64.onnx --out models/resnet50_ignition_cle_adaround_c64_gpu_desktop1.onnx --log results/quant/quant_resnet50_ignition_cle_adaround_c64_gpu_desktop1.log
#
# Uses resnet_env deliberately: torch 2.4.1+cpu and the same ONNX Runtime (1.22.1) that
# the Quark oracle's data sessions use, so a graph diff between the two compares the
# algorithm, not the runtime. Quark stays import-blocked. The family is read from the
# base's sidecar; a head-cut YOLO base needs its COCO calibration folder and letterboxes
# to the graph input (--cfg-path is not read), and a MODNet base needs its own folder plus
# --cfg-path, which is only cross-checked against the graph's input size. About 16 minutes
# on Desktop 2 for ResNet50's 54 layers; the log carries per-layer losses in Quark's line
# format so the two logs diff directly. Existing output/sidecar/log names are refused.
# A contended machine is refused before anything starts. Another build or producer on
# the same 16 threads leaves parity untouched -- that is a fixed-seed computation -- but
# makes this run's wall time and peak working set uncomparable with any other's;
# --allow-busy measures anyway. The snapshot lands in the load_<log> witness either way.
#
# --env and --device exist to time the device, never for parity. Another env brings its
# own torch and ONNX Runtime, so its bytes do not compare with resnet_env's; --device off
# cpu also passes --accept-non-parity, since a GPU's float reduction order moves the
# emitted weights off the Quark oracle ("GPU AdaRound is opt-in and explicitly
# non-parity", docs/DECISIONS.md). Every log opens with tools/torch_device_info.py's
# DEVICE_INFO block (host, CPU, env, torch/HIP/ORT, the adapter "cuda" resolved to) and a
# GPU_LOAD snapshot, and closes with a second snapshot. check_host_load sees only the CPU,
# so a --device run also samples the GPU every 30 s into the gpuload_<log> witness.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
QUANT="" OUT="" LOG="" IN=models/resnet50_fp32.onnx CALIB_DIR="" CFG_PATH="" ITERS=1000 DATA=1000 SEED=1705472343 ALLOW_BUSY=0 ENV=resnet_env DEVICE=cpu
while [ $# -gt 0 ]; do
    case "$1" in
        --quant) QUANT="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --log) LOG="$2"; shift ;;
        --in-model) IN="$2"; shift ;;
        --calib-dir) CALIB_DIR="$2"; shift ;;
        --cfg-path) CFG_PATH="$2"; shift ;;
        --iters) ITERS="$2"; shift ;;
        --data-size) DATA="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --allow-busy) ALLOW_BUSY=1 ;;
        --env) ENV="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$QUANT" ] && [ -n "$OUT" ] && [ -n "$LOG" ] || die "need --quant, --out and --log"
[[ "$ITERS" =~ ^[1-9][0-9]*$ ]] && [[ "$DATA" =~ ^[1-9][0-9]*$ ]] || die "--iters and --data-size must be positive"
need_file "$QUANT"; need_file "$QUANT.quant.json" "the emitted model's Ignition sidecar"; need_file "$IN"
[ -z "$CALIB_DIR" ] || need_dir "$CALIB_DIR"
[ -z "$CFG_PATH" ] || need_file "$CFG_PATH"
[ ! -e "$OUT" ] && [ ! -e "$OUT.quant.json" ] && [ ! -e "$LOG" ] || die "output/sidecar/log already exists"
GPU_WITNESS="$(dirname "$LOG")/gpu$(basename "$(load_witness "$LOG")")"
[ "$DEVICE" = cpu ] || [ ! -e "$GPU_WITNESS" ] || die "$GPU_WITNESS already exists"
if [ "$ALLOW_BUSY" = 1 ]; then export HOST_LOAD_ALLOW_BUSY=1; fi
check_host_load refuse "$(load_witness "$LOG")"
use_env "$ENV"
export PYTHONIOENCODING=utf-8
extra=()
[ -z "$CALIB_DIR" ] || extra+=(--calib-dir "$CALIB_DIR")
[ -z "$CFG_PATH" ] || extra+=(--cfg-path "$CFG_PATH")
[ "$DEVICE" = cpu ] || extra+=(--device "$DEVICE" --accept-non-parity)

# The run between two GPU snapshots, so its log says what else held the GPU at either end.
adaround() {
    python tools/torch_device_info.py --gpu-load || warn "device probe failed; this log has no DEVICE_INFO"
    python -m quant adaround --in-model "$IN" --quant "$QUANT" --out "$OUT" \
        --iters "$ITERS" --data-size "$DATA" --seed "$SEED" "${extra[@]}"
    local rc=$?
    python tools/torch_device_info.py --load-only || true
    return $rc
}
if [ "$DEVICE" != cpu ]; then
    python tools/torch_device_info.py --watch 30 > "$GPU_WITNESS" 2>&1 &
    sampler=$!
    trap 'kill "$sampler" 2>/dev/null || true' EXIT
    info "gpu witness: $GPU_WITNESS (pid $sampler)"
fi
run_logged "$LOG" adaround
