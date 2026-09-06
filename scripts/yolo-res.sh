#!/usr/bin/env bash
# Input-resolution sweep: the same yolov8 model at 640, 512 and 416, on the NPU.
#
#   ./scripts/yolo-res.sh                       # yolov8n at 640/512/416, calib 32
#   ./scripts/yolo-res.sh --variant s           # same sweep on yolov8s
#   ./scripts/yolo-res.sh --sizes "640 320"     # any multiple of 32
#   ./scripts/yolo-res.sh --map --n 500         # add COCO mAP per size
#   ./scripts/yolo-res.sh --stage table         # re-print from existing logs
#
# WHAT THIS IS FOR. The n-vs-s benchmark showed yolov8s costs 1.75x the latency
# of n for 3.2x the arithmetic, and that yolov8n fills only ~1.1 of the array's
# 16 TOPS. Both say the same thing: at 640 this NPU is not compute-bound by a
# model this small. Width was one probe of that. Resolution is the other, and a
# cleaner one -- it changes the amount of work WITHOUT changing the graph at
# all, so the node count, the operator mix and the partitioning all stay fixed
# and the only variable is how many pixels flow through.
#
# The question it answers: does latency track pixel count, or is there a floor?
# Convolution FLOPs are exactly proportional to input area, so a compute-bound
# accelerator would run 416x416 at (416/640)^2 = 42% of the 640 time. Anything
# above that is fixed cost -- weight streaming, DMA in and out, per-layer
# invocation overhead -- and the table below reports it as a fitted constant.
# That constant is a property of THE HARDWARE, reusable for any future model,
# which is the point: it says how small a model has to get before shrinking it
# stops buying anything.
#
# CALIBRATION IS DELIBERATELY SMALL (32). Calibration size affects accuracy, not
# latency and not what the EP accepts, and latency is the deliverable here. It
# also keeps the disk honest: 32 images is ~3.4 GB of Quark spool at 640 for n,
# against ~23 GB at the 200 the accuracy benchmark uses. Pass --map to get mAP
# too, but read it as "mAP at calib 32", not as this repo's headline accuracy.
#
# Steps that quantize need resnet_env; everything else needs resnet_env17.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

VARIANT=n SIZES="640 512 416" CALIB=32 RUNS=20 STAGE=all FORCE=0 DOMAP=0 NIMG=0
SOURCE="assets/test_image.jpg"
while [ $# -gt 0 ]; do
    case "$1" in
        --variant) VARIANT="$2"; shift ;;
        --sizes)   SIZES="$2"; shift ;;
        --calib)   CALIB="$2"; shift ;;
        --runs)    RUNS="$2"; shift ;;
        --stage)   STAGE="$2"; shift ;;
        --source)  SOURCE="$2"; shift ;;
        --map)     DOMAP=1 ;;
        --n)       NIMG="$2"; DOMAP=1; shift ;;
        --force)   FORCE=1 ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         die "unknown flag $1" ;;
    esac
    shift
done

case "$VARIANT" in n|s|m|l|x) ;; *) die "--variant must be one of n s m l x" ;; esac
case "$STAGE" in all|build|lat|map|table) ;; *) die "--stage must be all|build|lat|map|table" ;; esac
for sz in $SIZES; do
    [ "$sz" -gt 0 ] 2>/dev/null || die "--sizes takes integers, got '$sz'"
    # The stride-32 head grid must be an integer, and npu.yolo.input_size
    # refuses anything else -- fail here instead of 20 minutes into a quantize.
    [ $((sz % 32)) -eq 0 ] || die "size $sz is not a multiple of 32"
done

RES=results/res
mkdir -p "$RES"
need_dir data/coco_calib "run: python pipelines/yolov8n/2_fetch_coco.py --n-calib 300"

# 640 keeps the unsuffixed names the rest of the repo already uses, so the
# sweep reuses models/yolov8n.onnx and its cut instead of rebuilding them.
# Other sizes get an _r<size> tag so they can all coexist on disk.
stem_for() { if [ "$1" = 640 ]; then echo "yolov8${VARIANT}"; else echo "yolov8${VARIANT}_r$1"; fi; }
base_for() { echo "models/$(stem_for "$1").onnx"; }
cut_for()  { echo "models/$(stem_for "$1")_cut.onnx"; }
q_for()    { echo "models/$(stem_for "$1")_cut_xint8_c${CALIB}.onnx"; }

# onnx_size <model> -- the H of an NCHW input, or empty if it can't be read.
# Used to check that a model already on disk is at the size we think it is:
# "the file exists" is not the same as "the file is 512x512", and reusing a
# 640 graph for the 512 row would quietly put a wrong number in the table.
onnx_size() {
    python - "$1" <<'PY' 2>/dev/null || true
import sys, onnx
d = onnx.load(sys.argv[1], load_external_data=False)
dims = [x.dim_value for x in d.graph.input[0].type.tensor_type.shape.dim]
print(dims[2] if len(dims) == 4 else "")
PY
}

FAILED=()

# ------------------------------------------------------------------- 1. build
if [ "$STAGE" = all ] || [ "$STAGE" = build ]; then
    use_env resnet_env
    for sz in $SIZES; do
        base=$(base_for "$sz"); cut=$(cut_for "$sz"); q=$(q_for "$sz")

        have=""; [ -f "$base" ] && have="$(onnx_size "$base" | tr -d '[:space:]')"
        if [ "$have" = "$sz" ] && [ "$FORCE" = 0 ]; then
            info "have $(basename "$base") at ${sz}px"
        else
            if [ -n "$have" ] && [ "$have" != "$sz" ]; then
                warn "$(basename "$base") is ${have}px, not ${sz}px -- re-exporting"
            fi
            step "export yolov8${VARIANT} at ${sz}x${sz}"
            if run_logged "$RES/export_$(stem_for "$sz").log" \
                   python pipelines/yolov8n/1_export.py \
                       --weights "models/yolov8${VARIANT}.pt" --size "$sz" --out "$base"
            then
                rm -f "$cut" "$q"   # both are downstream of a base that just changed
            else
                warn "export at ${sz} failed"; FAILED+=("export ${sz}"); continue
            fi
        fi

        if [ -f "$cut" ] && [ "$FORCE" = 0 ]; then
            info "have $(basename "$cut")"
        else
            step "cut the decode tail at ${sz}px"
            if run_logged "$RES/cut_$(stem_for "$sz").log" \
                   python pipelines/yolov8n/1b_cut_head.py --in "$base" --out "$cut"
            then
                rm -f "$q"
            else
                warn "cut at ${sz} failed"; FAILED+=("cut ${sz}"); continue
            fi
        fi

        if [ -f "$q" ] && [ "$FORCE" = 0 ]; then
            info "have $(basename "$q")"
        else
            mb=$(calib_mb_per_image "$VARIANT" "$sz")
            step "quantize ${sz}px to XINT8  (${CALIB} images, ~$((CALIB * mb / 1024)) GB of spool)"
            require_disk $(( (CALIB * mb + 1023) / 1024 + 2 )) \
                         "Quark's calibration cache (yolov8${VARIANT} at ${sz}px: ~${mb} MB/image)"
            quark_guard
            if run_logged "$RES/quant_$(basename "$q" .onnx).log" \
                   python pipelines/yolov8n/3b_quantize_cut.py \
                       --in "$cut" --calib-dir data/coco_calib \
                       --limit "$CALIB" --out "$q" \
               && [ -f "$q" ]; then
                ok "$(basename "$q")  ($(du -h "$q" | cut -f1))"
            else
                warn "quantizing ${sz}px FAILED -- continuing with the other sizes"
                FAILED+=("quantize ${sz}")
            fi
            # Free the spool between sizes, so peak disk is one model's worth.
            quark_cleanup || true
        fi
    done
fi

# The sizes that actually have a quantized model to measure.
READY=()
for sz in $SIZES; do [ -f "$(q_for "$sz")" ] && READY+=("$sz"); done
[ ${#READY[@]} -gt 0 ] || die "nothing quantized to measure; run --stage build first"

# ----------------------------------------------------------------- 2. latency
# Demo conditions (conf 0.25), inference only, mean of --runs. Every size shares
# yolocutcachekey, so each run needs --fresh or it silently serves the previous
# size's compiled artifact -- which would look like perfect scaling.
if [ "$STAGE" = all ] || [ "$STAGE" = lat ]; then
    need_file "$SOURCE"
    npu_env
    for sz in "${READY[@]}"; do
        log="$RES/lat_$(stem_for "$sz")_npu.log"
        if [ -f "$log" ] && [ "$FORCE" = 0 ]; then info "have $log"; continue; fi
        step "latency at ${sz}px on NPU  (--fresh: recompiling)"
        if ! run_logged "$log" python pipelines/yolov8n/4_detect.py \
                 --model "$(q_for "$sz")" --ep npu --source "$SOURCE" \
                 --runs "$RUNS" --fresh --log 2; then
            warn "${sz}px latency failed"; FAILED+=("latency ${sz}"); continue
        fi
        run_logged "$RES/diag_$(stem_for "$sz").log" \
            python tools/diag_ep.py --model "$(q_for "$sz")" \
                --cache-key yolocutcachekey || true
    done
fi

# --------------------------------------------------------------------- 3. mAP
if [ "$DOMAP" = 1 ] && { [ "$STAGE" = all ] || [ "$STAGE" = map ]; }; then
    need_dir  data/coco/val2017
    need_file data/coco/annotations/instances_val2017.json
    npu_env
    python -c "import pycocotools" 2>/dev/null \
        || die "pycocotools missing. In this env: pip install pycocotools"
    for sz in "${READY[@]}"; do
        log="$RES/map_$(stem_for "$sz")_npu.log"
        if [ -f "$log" ] && [ "$FORCE" = 0 ]; then info "have $log"; continue; fi
        step "mAP at ${sz}px  ($([ "$NIMG" = 0 ] && echo 5000 || echo "$NIMG") images)"
        run_logged "$log" python pipelines/yolov8n/5_eval_map.py \
            --model "$(q_for "$sz")" --ep npu --n "$NIMG" --fresh --log 3 \
            || { warn "${sz}px mAP failed; see $log"; FAILED+=("mAP ${sz}"); }
    done
fi

# ------------------------------------------------------------------- 4. table
# Every grep ends in `|| echo ""`: under `set -euo pipefail` a grep that matches
# nothing kills the script before the table prints, making the ${x:-?} fallbacks
# unreachable. Same trap as yolo-bench.sh and yolo-eval.sh.
SZ=() LAT=() NODES=() TOTAL=() GF=() MAP=()
for sz in "${READY[@]}"; do
    st=$(stem_for "$sz")
    SZ+=("$sz")
    LAT+=("$(grep -aoE 'infer +mean +[0-9.]+' "$RES/lat_${st}_npu.log" 2>/dev/null \
             | head -1 | grep -oE '[0-9.]+$' || echo "")")
    # diag_ep.py prints "  922 nodes on the NPU, 7 elsewhere." -- two numbers on
    # one line, not a ratio, so the total is the sum.
    dl=$(grep -aoE '[0-9]+ nodes on the NPU, [0-9]+ elsewhere' \
         "$RES/diag_${st}.log" 2>/dev/null | head -1 || echo "")
    n1=$(echo "$dl" | grep -oE '^[0-9]+' || echo "")
    n2=$(echo "$dl" | grep -oE ', [0-9]+' | grep -oE '[0-9]+' || echo "")
    NODES+=("$n1")
    if [ -n "$n1" ] && [ -n "$n2" ]; then TOTAL+=("$((n1 + n2))"); else TOTAL+=(""); fi
    # Ultralytics prints "YOLOv8n summary ... 8.7 GFLOPs" during export, at the
    # exported resolution. Taking it from the log beats hardcoding a constant.
    GF+=("$(grep -aoE '[0-9.]+ GFLOPs' "$RES/export_${st}.log" 2>/dev/null \
            | tail -1 | grep -oE '^[0-9.]+' || echo "")")
    MAP+=("$(grep -aoE '^mAP@50-95 +[0-9.]+' "$RES/map_${st}_npu.log" 2>/dev/null \
             | grep -oE '[0-9.]+$' || echo "")")
done

# Fill in GFLOPs the logs don't have. Convolution FLOPs are exactly proportional
# to input area -- same graph, same channels, fewer spatial positions -- so one
# measured value gives every other size. This is the normal case for the 640
# row: the sweep reuses the model the repo already had, so there is no export
# log to read. Derived values are marked with a trailing *.
REF=-1
for i in "${!SZ[@]}"; do
    if [ -n "${GF[$i]}" ]; then REF=$i; break; fi
done
if [ "$REF" -ge 0 ]; then
    for i in "${!SZ[@]}"; do
        if [ -z "${GF[$i]}" ]; then
            GF[$i]="$(awk -v g="${GF[$REF]}" -v a="${SZ[$i]}" -v b="${SZ[$REF]}" \
                      'BEGIN{printf "%.1f*", g * a * a / (b * b)}')"
        fi
    done
fi

ROWS=() FITDATA=()
for i in "${!SZ[@]}"; do
    ROWS+=("$(printf '%-6s %8s %9s %11s %9s %8s' \
        "${SZ[$i]}" "$(awk -v s="${SZ[$i]}" 'BEGIN{printf "%.3f", s*s/1000000}')" \
        "${GF[$i]:-?}" "${NODES[$i]:-—}/${TOTAL[$i]:-—}" \
        "${LAT[$i]:-?}" "${MAP[$i]:-—}")")
    [ -n "${LAT[$i]}" ] && FITDATA+=("${SZ[$i]} ${LAT[$i]}")
done

printf '\n%s============== YOLOv8%s input-resolution sweep ==============%s\n' \
    "$BOLD" "$VARIANT" "$OFF"
printf '%s  XINT8 head-cut on NPU | calibration %s images | mean of %s runs%s\n' \
    "$DIM" "$CALIB" "$RUNS" "$OFF"
printf '%-6s %8s %9s %11s %9s %8s\n' size Mpixels GFLOPs "NPU nodes" ms mAP
printf '%s\n' "${ROWS[@]}"
echo

# The actual finding: split measured latency into a part that scales with pixels
# and a part that does not. Needs two sizes; three or more makes the fit mean
# something. Least squares on ms = a + b*Mpixels.
#
# The data goes in as ARGV, not on stdin: `python - <<'PY'` already uses stdin
# for the script itself, so a pipe into it would silently replace the program.
if [ ${#FITDATA[@]} -ge 2 ]; then
    python - "${FITDATA[@]}" <<'PY'
import sys

pts = sorted(((int(s) ** 2 / 1e6, float(ms))
              for s, ms in (a.split() for a in sys.argv[1:])), reverse=True)

n = len(pts)
sx = sum(p for p, _ in pts); sy = sum(t for _, t in pts)
sxx = sum(p * p for p, _ in pts); sxy = sum(p * t for p, t in pts)
den = n * sxx - sx * sx
if abs(den) < 1e-12:      # every size identical; nothing to fit
    sys.exit(0)
b = (n * sxy - sx * sy) / den      # ms per megapixel
a = (sy - b * sx) / n              # ms that does not scale with pixels

side = lambda px: int(round(px ** 0.5 * 1000))
ref_px, ref_ms = pts[0]

print("  If latency were pure compute it would scale with pixel count:")
print(f"  {'size':>6} {'measured':>10} {'area-scaled':>12} {'excess':>9}")
for px, ms in pts:
    pred = ref_ms * px / ref_px
    print(f"  {side(px):>6} {ms:>9.2f}  {pred:>11.2f}  {ms - pred:>+8.2f}")

share = 100 * a / ref_ms if ref_ms else 0
print(f"\n  least-squares fit:  ms = {a:.2f} + {b:.2f} x Mpixels")
print(f"  fixed cost per inference : {a:.2f} ms  ({share:.0f}% of the "
      f"{side(ref_px)}px run)")
print(f"  marginal cost            : {b:.2f} ms per megapixel\n")

if share < 5:
    print("  VERDICT: latency tracks pixel count almost exactly. At this model")
    print("  size the NPU is compute-bound, and cutting resolution buys latency")
    print("  in proportion to the pixels removed.")
elif share < 25:
    print("  VERDICT: mostly compute-bound, with a real fixed floor. Shrinking")
    print(f"  the input keeps paying, but never below ~{a:.1f} ms per inference")
    print("  no matter how small the input gets.")
else:
    print("  VERDICT: dominated by fixed cost, not by pixels. Most of the time")
    print("  is weight streaming / DMA / per-layer overhead that a smaller input")
    print("  does not touch, so shrinking resolution is a poor lever here --")
    print("  spend the pixels, or spend the width, and get accuracy for free.")
PY
else
    warn "need at least two measured sizes to fit; got ${#FITDATA[@]}"
fi

echo
echo "  ms = demo conditions (conf 0.25), inference only. NOT the mAP run's"
echo "       latency: decode at conf 0.001 costs far more."
echo "  *  = GFLOPs derived by area from another size, not read from an export log."
echo "  logs: $RES/{export,cut,quant,lat,diag,map}_*.log"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo
    warn "these steps failed and are missing from the table:"
    printf '      %s\n' "${FAILED[@]}"
fi
