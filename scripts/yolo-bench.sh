#!/usr/bin/env bash
# The full NPU benchmark matrix: {yolov8n, yolov8s} x {XINT8, XINT8+AdaRound},
# at ONE fixed calibration size, plus the float CPU baselines. Prints one table.
#
#   ./scripts/yolo-bench.sh                      # calib 200, mAP over all 5000
#   ./scripts/yolo-bench.sh --calib 100 --n 500  # quick pass
#   ./scripts/yolo-bench.sh --stage quant        # only build the models
#   ./scripts/yolo-bench.sh --stage table        # re-print from existing logs
#   ./scripts/yolo-bench.sh --variants "n s m"   # m needs ~120 GB free at 200
#   ./scripts/yolo-bench.sh --no-adaround        # plain XINT8 only, half the matrix
#
# --no-adaround exists because FastFinetune needs more RAM than the development
# machine has. Measured on 13.8 GB: AdaRound SIGSEGVs at layer 0 -- the only layer
# still at full 640x640 and therefore its memory high-water mark. It is not a
# Quark bug and a smaller --calib does not help (200 crashed where 300 had
# succeeded on an idle box). On a machine with more headroom, drop the flag.
#
# WHY A FIXED CALIBRATION SIZE. The models this repo already had vary two things
# at once: yolov8n_cut_xint8 was calibrated on 32 images, yolov8n_cut_xint8_
# adaround on 300. Their mAP gap (9.75 vs 2.9 against FP32) therefore cannot say
# how much came from AdaRound and how much from having 9x the calibration data.
# Holding calibration fixed and toggling only --adaround answers that, and the
# answer decides whether AdaRound's extra ~25 min per model is worth paying.
#
# DISK. Quark spools calibration activations to %TEMP% and does not stream them:
# ~105 MB/image for yolov8n but ~198 MB/image for yolov8s (measured 2026-09-05;
# see calib_mb_per_image in lib.sh). At --calib 200 that is 23 GB for n and 41 GB
# for s. Runs are strictly serial and each cleans up after itself, including on
# Ctrl-C, so peak usage is one model's worth -- never the sum.
#
# Steps that quantize need resnet_env; everything else needs resnet_env17.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CALIB=200 NIMG=0 STAGE=all VARIANTS="n s" FORCE=0 ADAROUND=1
SOURCE="assets/test_image.jpg" RUNS=20
while [ $# -gt 0 ]; do
    case "$1" in
        --calib)    CALIB="$2"; shift ;;
        --n)        NIMG="$2"; shift ;;
        --stage)    STAGE="$2"; shift ;;
        --variants) VARIANTS="$2"; shift ;;
        --runs)     RUNS="$2"; shift ;;
        --force)    FORCE=1 ;;
        --no-adaround) ADAROUND=0 ;;
        -h|--help)  usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)          die "unknown flag $1" ;;
    esac
    shift
done
case "$STAGE" in all|quant|lat|map|table) ;; *) die "--stage must be all|quant|lat|map|table" ;; esac

# The quantization modes each variant is built in. "" is plain XINT8; the empty
# string is a real element, so MODES is never empty and `${MODES[@]}` is safe
# under `set -u`. Every later stage iterates this same list, so --no-adaround
# keeps AdaRound models out of the table even if one is left over on disk.
MODES=("")
[ "$ADAROUND" = 1 ] && MODES+=(adaround)

BENCH=results/bench
FAILED=()
mkdir -p "$BENCH"
need_dir data/coco_calib "run: python pipelines/yolov8n/2_fetch_coco.py --n-calib 300"

# free_ram_gb -- AdaRound is the memory-hungry step; the reference machine has 13.8 GB.
# Windows only; anything unparseable reports 0 and the caller just warns.
free_ram_gb() {
    powershell -NoProfile -Command \
        "[math]::Floor((Get-CIMInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" \
        2>/dev/null | tr -d '\r' | grep -oE '^[0-9]+' || echo 0
}

# name of the quantized model for a variant + mode, tagged with the calibration
# size so two calibrations of the same variant can coexist and be compared.
qname() { echo "models/yolov8${1}_cut_xint8_c${CALIB}${2:+_$2}.onnx"; }

# ---------------------------------------------------------------- 1. quantize
if [ "$STAGE" = all ] || [ "$STAGE" = quant ]; then
    for v in $VARIANTS; do
        need_file "models/yolov8${v}_cut.onnx" \
            "run: ./scripts/yolo-cut.sh --variant $v --limit 32   (cuts the head)"
        for mode in "${MODES[@]}"; do
            out=$(qname "$v" "$mode")
            if [ -f "$out" ] && [ "$FORCE" = 0 ]; then
                info "have $(basename "$out") -- skipping (--force to redo)"
                continue
            fi
            mb=$(calib_mb_per_image "$v")
            step "quantize yolov8${v} ${mode:-xint8}  (${CALIB} images, ~$((CALIB * mb / 1024)) GB of spool)"
            require_disk $(( (CALIB * mb + 1023) / 1024 + 2 )) \
                         "Quark's calibration cache (yolov8${v}: ~${mb} MB/image)"
            quark_guard
            if [ -n "$mode" ]; then
                info "AdaRound: expect this one to take many minutes"
                ram=$(free_ram_gb)
                [ "$ram" -lt 6 ] 2>/dev/null && warn \
                    "only ${ram} GB RAM free -- FastFinetune segfaults rather than
raising when it runs out. Close other work before trusting this run."
            fi
            use_env resnet_env
            ARGS=(--in "models/yolov8${v}_cut.onnx" --calib-dir data/coco_calib
                  --limit "$CALIB" --out "$out")
            [ -n "$mode" ] && ARGS+=(--adaround)
            # One model failing must not cost the other three. AdaRound has been
            # seen to take SIGSEGV rather than raise when the box runs out of
            # memory (13.8 GB here, and FastFinetune holds the float and the
            # quantized model plus a layer's cached activations at once), so a
            # crash mid-matrix is a live possibility on an unattended run.
            if run_logged "$BENCH/quant_$(basename "$out" .onnx).log" \
                   python pipelines/yolov8n/3b_quantize_cut.py "${ARGS[@]}" \
               && [ -f "$out" ]; then
                ok "$(basename "$out")  ($(du -h "$out" | cut -f1))"
            else
                warn "quantizing $(basename "$out") FAILED -- continuing with the rest"
                warn "  see $BENCH/quant_$(basename "$out" .onnx).log"
                FAILED+=("$(basename "$out" .onnx)")
            fi
            # Release the spool before the next model rather than at script exit,
            # so peak disk is one model's worth and not the whole matrix's.
            quark_cleanup || true
        done
    done
fi

# The models the later stages measure: the matrix, then the float baselines.
MODELS=()
for v in $VARIANTS; do
    for mode in "${MODES[@]}"; do
        m=$(qname "$v" "$mode"); [ -f "$m" ] && MODELS+=("$m")
    done
done
for v in $VARIANTS; do
    [ -f "models/yolov8${v}.onnx" ] && MODELS+=("models/yolov8${v}.onnx")
done
[ ${#MODELS[@]} -gt 0 ] || die "no models to benchmark; run --stage quant first"

ep_for() { case "$1" in *xint8*) echo npu ;; *) echo cpu ;; esac; }

# ----------------------------------------------------------------- 2. latency
# Demo conditions: conf 0.25, one image, --runs timed iterations. Every NPU model
# shares yolocutcachekey, so each needs --fresh or it silently runs the previous
# model's compiled artifact.
if [ "$STAGE" = all ] || [ "$STAGE" = lat ]; then
    need_file "$SOURCE"
    npu_env
    for m in "${MODELS[@]}"; do
        stem=$(basename "$m" .onnx); ep=$(ep_for "$m")
        log="$BENCH/lat_${stem}_${ep}.log"
        if [ -f "$log" ] && [ "$FORCE" = 0 ]; then info "have $log"; continue; fi
        step "latency  $stem on ${ep^^}"
        fresh=(); [ "$ep" = npu ] && fresh=(--fresh)
        run_logged "$log" python pipelines/yolov8n/4_detect.py \
            --model "$m" --ep "$ep" --source "$SOURCE" --runs "$RUNS" \
            --log 2 "${fresh[@]}" || { warn "$stem latency failed"; continue; }
        if [ "$ep" = npu ]; then
            run_logged "$BENCH/diag_${stem}.log" \
                python tools/diag_ep.py --model "$m" --cache-key yolocutcachekey || true
        fi
    done
fi

# --------------------------------------------------------------------- 3. mAP
# Eval conditions, NOT demo conditions: conf 0.001, per-class NMS at IoU 0.7,
# 300 boxes. The decode alone costs ~34 ms at conf 0.001 versus 0.12 ms at 0.25,
# so latency measured here is not comparable to the latency column above.
if [ "$STAGE" = all ] || [ "$STAGE" = map ]; then
    need_dir  data/coco/val2017
    need_file data/coco/annotations/instances_val2017.json
    npu_env
    python -c "import pycocotools" 2>/dev/null \
        || die "pycocotools missing. In this env: pip install pycocotools"
    for m in "${MODELS[@]}"; do
        stem=$(basename "$m" .onnx); ep=$(ep_for "$m")
        log="$BENCH/map_${stem}_${ep}.log"
        if [ -f "$log" ] && [ "$FORCE" = 0 ]; then info "have $log"; continue; fi
        step "mAP  $stem on ${ep^^}  ($([ "$NIMG" = 0 ] && echo 5000 || echo "$NIMG") images)"
        fresh=(); [ "$ep" = npu ] && fresh=(--fresh)
        run_logged "$log" python pipelines/yolov8n/5_eval_map.py \
            --model "$m" --ep "$ep" --n "$NIMG" --log 3 "${fresh[@]}" \
            || { warn "$stem mAP failed; see $log"; continue; }
    done
fi

# ------------------------------------------------------------------- 4. table
# Every grep ends in `|| echo ""`: under `set -euo pipefail` a grep that matches
# nothing would kill the script before the table prints, making the ${x:-?}
# fallbacks unreachable -- the same trap already documented in yolo-eval.sh.
ROWS=()
for m in "${MODELS[@]}"; do
    stem=$(basename "$m" .onnx); ep=$(ep_for "$m")
    lat=$(grep -aoE 'infer +mean +[0-9.]+' "$BENCH/lat_${stem}_${ep}.log" 2>/dev/null \
          | head -1 | grep -oE '[0-9.]+$' || echo "")
    map=$(grep -aoE '^mAP@50-95 +[0-9.]+' "$BENCH/map_${stem}_${ep}.log" 2>/dev/null \
          | grep -oE '[0-9.]+$' || echo "")
    m50=$(grep -aoE '^mAP@50 +[0-9.]+' "$BENCH/map_${stem}_${ep}.log" 2>/dev/null \
          | grep -oE '[0-9.]+$' || echo "")
    nodes=$(grep -aoE '^ +[0-9]+ nodes on the NPU' "$BENCH/diag_${stem}.log" 2>/dev/null \
            | grep -oE '[0-9]+' | head -1 || echo "")
    ROWS+=("$(printf '%-38s %-4s %8s %8s %8s %9s' \
        "$stem" "$ep" "${lat:-?}" "${map:-?}" "${m50:-?}" "${nodes:-—}")")
done

printf '\n%s==================== YOLOv8 NPU benchmark ====================%s\n' "$BOLD" "$OFF"
printf '%s  calibration %s images | mAP over %s val2017 images%s\n' \
    "$DIM" "$CALIB" "$([ "$NIMG" = 0 ] && echo 5000 || echo "$NIMG")" "$OFF"
printf '%-38s %-4s %8s %8s %8s %9s\n' model ep "ms" "mAP" "mAP50" "NPU nodes"
printf '%s\n' "${ROWS[@]}"
echo
echo "  ms  = demo conditions (conf 0.25), inference only, mean of $RUNS runs."
echo "        NOT the mAP run's latency: decode at conf 0.001 costs ~34 ms more."
echo "  logs: $BENCH/{quant,lat,map,diag}_*.log"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo
    warn "these models are MISSING from the table because quantization failed:"
    printf '      %s\n' "${FAILED[@]}"
    warn "re-run just those with: ./scripts/yolo-bench.sh --stage quant"
fi
