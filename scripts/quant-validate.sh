#!/usr/bin/env bash
# Validate an owned/reference pair using the full eval for its family and EP reports.
#
#   ./scripts/quant-validate.sh --model models/resnet50_own_reemit_nocle_c64.onnx --reference models/resnet50_quark_nocle_c64.onnx --tag resnet50_reemit_nocle_c64 [--cpu-only | --npu-only]
#   ./scripts/quant-validate.sh --family yolo --model models/yolov8n_cut_ignition_cle_c64.onnx --reference models/yolov8n_cut_quark_cle_c64.onnx --tag yolov8n_cut_ignition_cle_c64 [--cpu-only | --npu-only]
#
# All logs use the supplied model/variant tag and refuse overwrites. ResNet CPU eval uses
# every labeled image in data/eval; --family yolo runs pipelines/yolov8n/5_eval_map.py on
# all 5000 val2017 images (conf 0.001, IoU 0.7, max_det 300) and writes the detections to
# results/dets_<tag>_<role>_<ep>.json. NPU runs check for foreign hardware contexts,
# always pass --fresh, and read the EP report immediately after each model.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
MODEL="" REFERENCE="" TAG="" CPU_ONLY=0 NPU_ONLY=0 FAMILY=resnet
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift ;;
        --reference) REFERENCE="$2"; shift ;;
        --tag) TAG="$2"; shift ;;
        --cpu-only) CPU_ONLY=1 ;;
        --npu-only) NPU_ONLY=1 ;;
        --family) FAMILY="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$MODEL" ] && [ -n "$REFERENCE" ] && [ -n "$TAG" ] || die "need --model, --reference and --tag"
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || die "tag must contain only letters/digits/_/-"
need_file "$MODEL"; need_file "$REFERENCE"
[ "$CPU_ONLY$NPU_ONLY" != 11 ] || die "choose only one of --cpu-only/--npu-only"
case "$FAMILY" in resnet|yolo) ;; *) die "--family must be resnet or yolo" ;; esac
npu_env
export PYTHONIOENCODING=utf-8
newlog() { [ ! -e "$1" ] || die "log exists: $1"; run_logged "$@"; }
if [ "$FAMILY" = yolo ]; then
    need_dir data/coco/val2017; need_file data/coco/annotations/instances_val2017.json
    python -c "import pycocotools" 2>/dev/null || die "pycocotools missing in resnet_env17"
    CACHE_KEY=yolocutcachekey
    # eval <role> <path> <ep> [extra...]: full 5000-image COCO mAP with the eval conventions.
    eval_model() { newlog "results/quant/map_${TAG}_$1_$3.log" python pipelines/yolov8n/5_eval_map.py --model "$2" --ep "$3" --n 0 --dets "results/dets_${TAG}_$1_$3.json" "${@:4}"; }
else
    # Assert every image used by the existing eval pipeline has a label.
    COUNT="$(python -c 'import glob,json,os; from npu.preprocess import IMG_EXTS; fs=sorted(p for ext in IMG_EXTS for p in glob.glob("data/eval/**/"+ext,recursive=True)); labels=json.load(open("data/eval/labels.json")); assert fs and all(os.path.basename(p) in labels for p in fs); print(len(fs))')"
    CACHE_KEY=modelcachekey
    eval_model() { newlog "results/quant/run_${TAG}_$1_$3.log" python pipelines/resnet50/4_run.py --model "$2" --ep "$3" --images data/eval --n "$COUNT" "${@:4}"; }
fi
if [ "$NPU_ONLY" = 0 ]; then
    newlog "results/quant/diff_${TAG}.log" python tools/quant_compare.py --model "$MODEL" --reference "$REFERENCE"
    for role in reference own; do
        path="$REFERENCE"; [ "$role" = reference ] || path="$MODEL"
        eval_model "$role" "$path" cpu
    done
fi
if [ "$CPU_ONLY" = 0 ]; then
    for role in reference own; do
        path="$REFERENCE"; [ "$role" = reference ] || path="$MODEL"
        witness="results/quant/contexts_${TAG}_${role}.log"
        newlog "$witness" /c/Windows/System32/AMD/xrt-smi.exe examine -r aie-partitions
        grep -q "No hardware contexts running" "$witness" || die "NPU has foreign contexts; wait for a clean paired run"
        eval_model "$role" "$path" npu --fresh
        newlog "results/quant/diag_${TAG}_${role}.log" python tools/diag_ep.py --cache-key "$CACHE_KEY"
    done
fi
