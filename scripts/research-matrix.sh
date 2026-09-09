#!/usr/bin/env bash
# Sequential, bounded low-level research matrices. No production options change.
#   bash scripts/research-matrix.sh memory <unique-tag>
#   bash scripts/research-matrix.sh arithmetic <unique-tag>
#   bash scripts/research-matrix.sh arithmetic-boundaries <unique-tag>
#   bash scripts/research-matrix.sh gemm <unique-tag>
#   bash scripts/research-matrix.sh calibration-dual <unique-tag> [selection]
#   bash scripts/research-matrix.sh calibration-time <unique-tag>
# Tags must include machine/date. RESEARCH_ASSET_ROOT can select models/data;
# otherwise use this checkout, then the parent main checkout for nested worktrees.
# calibration-dual selections: 'remaining' skips the completed ResNet50 CLE pair,
# or name one pair -- resnet50_cle, resnet50_no-cle, yolov8n_cle, modnet_cle -- to
# rerun just that family's legacy/dual pair into a fresh tag.
# Each case has its own immutable log, resource limit and host/device guard.
# calibration-time is the only timing-eligible calibration suite, so its cases wait
# up to 30 s for a clear host, as the memory and gemm cases already do.
# The first failure stops the matrix; keep that evidence and use a new tag to rerun.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
if [ "${1:-}" = --help ]; then usage "${BASH_SOURCE[0]}"; exit 0; fi
[ $# -ge 2 ] && [ $# -le 3 ] || die "need suite, unique machine/date tag and optional selection"
suite="$1" tag="$2"
selection="${3:-all}"
CALIB_PAIRS=(resnet50:cle resnet50:no-cle yolov8n:cle modnet:cle)
if [ "$selection" != all ]; then
    [ "$suite" = calibration-dual ] || die "a selection is only valid for calibration-dual"
    if [ "$selection" != remaining ]; then
        ok=0
        for pair in "${CALIB_PAIRS[@]}"; do
            [ "$selection" = "${pair%:*}_${pair#*:}" ] && ok=1
        done
        [ "$ok" = 1 ] || die "selection must be remaining or one of ${CALIB_PAIRS[*]//:/_}"
    fi
fi
[[ "$tag" =~ ^[a-zA-Z0-9_]+$ ]] || die "tag must use letters, digits and underscores"
base="scratch/research_$tag"
[ ! -e "$base" ] || die "output directory exists"
mkdir -p "$base"
assets="${RESEARCH_ASSET_ROOT:-.}"
if [ ! -d "$assets/models" ] && [ -z "${RESEARCH_ASSET_ROOT:-}" ]; then assets=../..; fi

memory_case() {
    local name="$1"; shift
    bash scripts/research-lowlevel.sh --wait-clear 30 --npu --seconds 240 --rss-gib 6 \
        --log "results/aie/memory_${tag}_${name}.log" -- bash scripts/research-iron.sh \
        kernels/memory_placement/probe.py --out "$base/$name" --iters 5 "$@"
}
calibration_case() {
    local family="$1" cle="$2" method="$3" block="$4" src data cfg flag name
    name="${family}_${cle}_${method}_${block}"
    flag="--$cle"
    case "$family" in
        resnet50)
            src="$assets/models/resnet50_fp32.onnx"; data="$assets/data/calib"
            cfg="$assets/models/preprocess_config.json" ;;
        yolov8n)
            src="$assets/models/yolov8n_cut.onnx"; data="$assets/data/coco_calib"
            cfg="$assets/models/preprocess_config.json" ;;
        modnet)
            src="$assets/models/modnet/modnet_cut_fp32.onnx"; data="$assets/data/modnet_calib"
            cfg="$assets/models/modnet/preprocess_config.json" ;;
    esac
    local checks=() reference_args=() gate=()
    if [ "$suite" = calibration-dual ]; then checks=(--checks-only); else gate=(--wait-clear 30); fi
    if [ "$method" != legacy ]; then
        reference_args=(--reference "$base/${family}_${cle}_legacy_1.onnx")
    fi
    bash scripts/research-lowlevel.sh "${checks[@]}" "${gate[@]}" --seconds 900 --rss-gib 16 \
        --log "results/quant/alphabet_${tag}_${name}.log" -- python tools/quant_calib_alphabet.py \
        --in-model "$src" --calib-dir "$data" --cfg-path "$cfg" "${reference_args[@]}" \
        --out "$base/$name.onnx" --method "$method" --limit 64 "$flag"
}
case "$suite" in
    gemm)
        reference=""
        for name in separate_1 same_1 same_2 separate_2 separate_alternate same_alternate; do
            opts=()
            if [[ "$name" = same* ]]; then opts+=(--layout same); else opts+=(--layout separate); fi
            if [[ "$name" = *alternate ]]; then opts+=(--alternate); fi
            if [ -n "$reference" ]; then opts+=(--reference "$reference"); fi
            bash scripts/research-lowlevel.sh --wait-clear 30 --npu --seconds 240 --rss-gib 6 \
                --log "results/aie/gemm_${tag}_${name}.log" -- bash scripts/research-iron.sh \
                kernels/memory_placement/gemm.py --targets 1,2,4,8 --out "$base/$name" "${opts[@]}"
            if [ -z "$reference" ]; then reference="$base/$name/result.json"; fi
        done
        ;;
    arithmetic-boundaries)
        for spec in 0:-1 16:15; do
            IFS=: read -r sc sb <<< "$spec"
            name="c32_sc${sc}_sb${sb}"
            bash scripts/research-lowlevel.sh --checks-only --npu --seconds 300 --rss-gib 6 \
                --log "results/quant/arithmetic_${tag}_${name}.log" -- python tools/xint8_arithmetic_probe.py \
                --fresh --shift-cut "$sc" --shift-bias "$sb" --out "$base/$name"
        done
        ;;
    memory)
        memory_case separate_1 --layout separate
        reference="$base/separate_1/result.json"
        memory_case same_1 --layout same --reference "$reference"
        memory_case same_2 --layout same --reference "$reference"
        memory_case separate_2 --layout separate --reference "$reference"
        memory_case separate_3 --layout separate --reference "$reference"
        memory_case same_3 --layout same --reference "$reference"
        for offset in 64 128 256; do
            memory_case "same_offset_$offset" --layout same --a-offset "$offset" --reference "$reference"
        done
        memory_case single_same --layout same --mode single --reference "$reference"
        memory_case single_separate --layout separate --mode single --reference "$reference"
        ;;
    arithmetic)
        for spec in 32:1:0 1:1:0 31:1:0 33:1:0 64:1:0 32:0:0 32:4:0 32:8:0 32:16:0 32:1:-1 32:1:3 32:1:15; do
            IFS=: read -r c sc sb <<< "$spec"
            name="c${c}_sc${sc}_sb${sb}"
            bash scripts/research-lowlevel.sh --checks-only --npu --seconds 300 --rss-gib 6 \
                --log "results/quant/arithmetic_${tag}_${name}.log" -- python tools/xint8_arithmetic_probe.py \
                --fresh --channels "$c" --shift-cut "$sc" --shift-bias "$sb" --out "$base/$name"
        done
        for c in 31 32 33; do
            name="c${c}_sc1_sb0_reverse"
            bash scripts/research-lowlevel.sh --checks-only --npu --seconds 300 --rss-gib 6 \
                --log "results/quant/arithmetic_${tag}_${name}.log" -- python tools/xint8_arithmetic_probe.py \
                --fresh --channels "$c" --reverse-channels --out "$base/$name"
        done
        ;;
    calibration-dual)
        for pair in "${CALIB_PAIRS[@]}"; do
            family="${pair%:*}" cle="${pair#*:}"
            case "$selection" in
                all) ;;
                remaining) [ "$pair" = resnet50:cle ] && continue || : ;;
                "${family}_${cle}") ;;
                *) continue ;;
            esac
            calibration_case "$family" "$cle" legacy 1
            calibration_case "$family" "$cle" dual 1
        done
        ;;
    calibration-time)
        for block in 1 2 3 4 5; do
            if (( block % 2 )); then methods="legacy alphabet"; else methods="alphabet legacy"; fi
            for method in $methods; do calibration_case resnet50 cle "$method" "$block"; done
        done
        ;;
    *) die "unknown suite $suite" ;;
esac
