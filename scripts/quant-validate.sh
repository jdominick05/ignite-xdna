#!/usr/bin/env bash
# Validate an owned/reference ResNet pair using full classification eval and EP reports.
#
#   ./scripts/quant-validate.sh --model models/resnet50_own_reemit_nocle_c64.onnx --reference models/resnet50_quark_nocle_c64.onnx --tag resnet50_reemit_nocle_c64 [--cpu-only | --npu-only]
#
# All logs use the supplied model/variant tag and refuse overwrites. CPU eval uses
# every labeled image in data/eval. NPU runs check for foreign hardware contexts,
# always pass --fresh, and read the EP report immediately after each model.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
MODEL="" REFERENCE="" TAG="" CPU_ONLY=0 NPU_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift ;;
        --reference) REFERENCE="$2"; shift ;;
        --tag) TAG="$2"; shift ;;
        --cpu-only) CPU_ONLY=1 ;;
        --npu-only) NPU_ONLY=1 ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
    shift
done
[ -n "$MODEL" ] && [ -n "$REFERENCE" ] && [ -n "$TAG" ] || die "need --model, --reference and --tag"
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || die "tag must contain only letters/digits/_/-"
need_file "$MODEL"; need_file "$REFERENCE"
[ "$CPU_ONLY$NPU_ONLY" != 11 ] || die "choose only one of --cpu-only/--npu-only"
npu_env
export PYTHONIOENCODING=utf-8
# Assert every image used by the existing eval pipeline has a label.
COUNT="$(python -c 'import glob,json,os; from npu.preprocess import IMG_EXTS; fs=sorted(p for ext in IMG_EXTS for p in glob.glob("data/eval/**/"+ext,recursive=True)); labels=json.load(open("data/eval/labels.json")); assert fs and all(os.path.basename(p) in labels for p in fs); print(len(fs))')"
newlog() { [ ! -e "$1" ] || die "log exists: $1"; run_logged "$@"; }
if [ "$NPU_ONLY" = 0 ]; then
    newlog "results/quant/diff_${TAG}.log" python tools/quant_compare.py --model "$MODEL" --reference "$REFERENCE"
    for role in reference own; do
        path="$REFERENCE"; [ "$role" = reference ] || path="$MODEL"
        newlog "results/quant/run_${TAG}_${role}_cpu.log" python pipelines/resnet50/4_run.py --model "$path" --ep cpu --images data/eval --n "$COUNT"
    done
fi
if [ "$CPU_ONLY" = 0 ]; then
    for role in reference own; do
        path="$REFERENCE"; [ "$role" = reference ] || path="$MODEL"
        witness="results/quant/contexts_${TAG}_${role}.log"
        newlog "$witness" /c/Windows/System32/AMD/xrt-smi.exe examine -r aie-partitions
        grep -q "No hardware contexts running" "$witness" || die "NPU has foreign contexts; wait for a clean paired run"
        newlog "results/quant/run_${TAG}_${role}_npu.log" python pipelines/resnet50/4_run.py --model "$path" --ep npu --images data/eval --n "$COUNT" --fresh
        newlog "results/quant/diag_${TAG}_${role}.log" python tools/diag_ep.py --cache-key modelcachekey
    done
fi
