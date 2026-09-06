#!/usr/bin/env bash
# Input-resolution sweep: ResNet50 (XINT8, no AdaRound) at several sizes, on
# the NPU. Same question yolo-res.sh answered for YOLOv8 -- does latency track
# pixel count, or is there a fixed floor -- but this time top-1/top-5 accuracy
# comes along for free, because 4_run.py already reports both in one pass.
# Unlike detection, a classifier trained at one resolution (224, crop_pct 0.95
# for this checkpoint) is NOT expected to hold accuracy away from it, so the
# sweep exists to find the actual speed/accuracy tradeoff, not just the floor.
#
#   ./scripts/resnet-res.sh                       # 128/160/192/224/256/288, calib 64
#   ./scripts/resnet-res.sh --sizes "160 224 320"
#   ./scripts/resnet-res.sh --n 1000              # eval on more images
#   ./scripts/resnet-res.sh --stage table         # re-print from existing logs
#
# CALIBRATION IS SMALL (64) and AdaRound is OFF: this is plain XINT8 at every
# size, not the 79.8% headline number (which used AdaRound at 224 only and
# does not exist at other sizes). Reading this table as "ResNet50 gets X% at
# size Y" is right for relative comparisons across size, wrong as an absolute
# ceiling -- AdaRound recovers a few points at 224 and would at every size.
#
# Steps that quantize need resnet_env; the run step needs resnet_env17.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SIZES="128 160 192 224 256 288" CALIB=64 NIMG=1000 STAGE=all FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --sizes)  SIZES="$2"; shift ;;
        --calib)  CALIB="$2"; shift ;;
        --n)      NIMG="$2"; shift ;;
        --stage)  STAGE="$2"; shift ;;
        --force)  FORCE=1 ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)        die "unknown flag $1" ;;
    esac
    shift
done

case "$STAGE" in all|build|run|table) ;; *) die "--stage must be all|build|run|table" ;; esac
for sz in $SIZES; do
    [ "$sz" -gt 0 ] 2>/dev/null || die "--sizes takes integers, got '$sz'"
done

RES=results/res
mkdir -p "$RES"
need_dir data/calib "run: python pipelines/resnet50/2_fetch_imagenet.py --n-calib 300"
need_dir data/eval  "run: python pipelines/resnet50/2_fetch_imagenet.py --n-eval 1000"

# 224 keeps the unsuffixed names the rest of the repo already uses.
stem_for()   { if [ "$1" = 224 ]; then echo "resnet50"; else echo "resnet50_r$1"; fi; }
base_for()   { echo "models/$(stem_for "$1")_fp32.onnx"; }
cfg_for()    { if [ "$1" = 224 ]; then echo "models/preprocess_config.json"; \
                else echo "models/preprocess_config_r$1.json"; fi; }
q_for()      { echo "models/$(stem_for "$1")_xint8_c${CALIB}.onnx"; }

# onnx_size <model> -- the H of an NCHW input, or empty if unreadable. Guards
# against silently reusing a stale-resolution model already on disk.
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
        base=$(base_for "$sz"); cfg=$(cfg_for "$sz"); q=$(q_for "$sz")

        have=""; [ -f "$base" ] && have="$(onnx_size "$base" | tr -d '[:space:]')"
        if [ "$have" = "$sz" ] && [ -f "$cfg" ] && [ "$FORCE" = 0 ]; then
            info "have $(basename "$base") at ${sz}px"
        else
            [ -n "$have" ] && [ "$have" != "$sz" ] \
                && warn "$(basename "$base") is ${have}px, not ${sz}px -- re-exporting"
            step "export resnet50 at ${sz}x${sz}"
            if run_logged "$RES/export_$(stem_for "$sz").log" \
                   python pipelines/resnet50/1_export.py \
                       --size "$sz" --out "$base" --cfg-out "$cfg"
            then
                rm -f "$q"   # downstream of a base that just changed
            else
                warn "export at ${sz} failed"; FAILED+=("export ${sz}"); continue
            fi
        fi

        if [ -f "$q" ] && [ "$FORCE" = 0 ]; then
            info "have $(basename "$q")"
        else
            mb=$(calib_mb_per_image_resnet "$sz")
            step "quantize ${sz}px to XINT8  (${CALIB} images, ~$((CALIB * mb / 1024 + 1)) GB of spool)"
            require_disk $(( (CALIB * mb + 1023) / 1024 + 2 )) \
                         "Quark's calibration cache (resnet50 at ${sz}px: ~${mb} MB/image)"
            quark_guard
            if run_logged "$RES/quant_$(basename "$q" .onnx).log" \
                   python pipelines/resnet50/3_quantize.py \
                       --calib-dir data/calib --limit "$CALIB" --config XINT8 \
                       --in-model "$base" --cfg-path "$cfg" --out "$q" \
               && [ -f "$q" ]; then
                ok "$(basename "$q")  ($(du -h "$q" | cut -f1))"
            else
                warn "quantizing ${sz}px FAILED -- continuing with the other sizes"
                FAILED+=("quantize ${sz}")
            fi
            quark_cleanup || true
        fi
    done
fi

READY=()
for sz in $SIZES; do [ -f "$(q_for "$sz")" ] && READY+=("$sz"); done
[ ${#READY[@]} -gt 0 ] || die "nothing quantized to measure; run --stage build first"

# --------------------------------------------------------- 2. run (lat + acc)
# Every size shares RESNET_CACHE_KEY (npu/paths.py), so each run needs --fresh
# or it silently serves the previous size's compiled artifact.
if [ "$STAGE" = all ] || [ "$STAGE" = run ]; then
    npu_env
    for sz in "${READY[@]}"; do
        log="$RES/run_$(stem_for "$sz")_npu.log"
        if [ -f "$log" ] && [ "$FORCE" = 0 ]; then info "have $log"; continue; fi
        step "latency + top-1/top-5 at ${sz}px on NPU  (--fresh: recompiling, $NIMG images)"
        if ! run_logged "$log" python pipelines/resnet50/4_run.py \
                 --ep npu --images data/eval --n "$NIMG" \
                 --model "$(q_for "$sz")" --cfg-path "$(cfg_for "$sz")" --fresh; then
            warn "${sz}px run failed"; FAILED+=("run ${sz}"); continue
        fi
        run_logged "$RES/diag_$(stem_for "$sz").log" \
            python tools/diag_ep.py --model "$(q_for "$sz")" \
                --cache-key modelcachekey || true
    done
fi

# ------------------------------------------------------------------- 3. table
SZ=() LAT=() TOP1=() TOP5=() NODES=() TOTAL=()
for sz in "${READY[@]}"; do
    st=$(stem_for "$sz")
    log="$RES/run_${st}_npu.log"
    SZ+=("$sz")
    LAT+=("$(grep -aoE 'mean +[0-9.]+ ms' "$log" 2>/dev/null | head -1 \
             | grep -oE '[0-9.]+' || echo "")")
    acc=$(grep -aoE 'top-1: *[0-9.]+% *top-5: *[0-9.]+%' "$log" 2>/dev/null | head -1 || echo "")
    TOP1+=("$(echo "$acc" | grep -oE 'top-1: *[0-9.]+' | grep -oE '[0-9.]+$' || echo "")")
    TOP5+=("$(echo "$acc" | grep -oE 'top-5: *[0-9.]+' | grep -oE '[0-9.]+$' || echo "")")
    dl=$(grep -aoE '[0-9]+ nodes on the NPU, [0-9]+ elsewhere' \
         "$RES/diag_${st}.log" 2>/dev/null | head -1 || echo "")
    n1=$(echo "$dl" | grep -oE '^[0-9]+' || echo "")
    n2=$(echo "$dl" | grep -oE ', [0-9]+' | grep -oE '[0-9]+' || echo "")
    NODES+=("$n1")
    if [ -n "$n1" ] && [ -n "$n2" ]; then TOTAL+=("$((n1 + n2))"); else TOTAL+=(""); fi
done

ROWS=() FITDATA=()
for i in "${!SZ[@]}"; do
    ROWS+=("$(printf '%-6s %8s %11s %9s %8s %8s' \
        "${SZ[$i]}" "$(awk -v s="${SZ[$i]}" 'BEGIN{printf "%.3f", s*s/1000000}')" \
        "${NODES[$i]:-—}/${TOTAL[$i]:-—}" \
        "${LAT[$i]:-?}" "${TOP1[$i]:-—}" "${TOP5[$i]:-—}")")
    [ -n "${LAT[$i]}" ] && FITDATA+=("${SZ[$i]} ${LAT[$i]}")
done

printf '\n%s============== ResNet50 input-resolution sweep ==============%s\n' "$BOLD" "$OFF"
printf '%s  XINT8, no AdaRound | calibration %s images | eval %s images%s\n' \
    "$DIM" "$CALIB" "$NIMG" "$OFF"
printf '%-6s %8s %11s %9s %8s %8s\n' size Mpixels "NPU nodes" ms top-1 top-5
printf '%s\n' "${ROWS[@]}"
echo

if [ ${#FITDATA[@]} -ge 2 ]; then
    python - "${FITDATA[@]}" <<'PY'
import sys

pts = sorted(((int(s) ** 2 / 1e6, float(ms))
              for s, ms in (a.split() for a in sys.argv[1:])), reverse=True)

n = len(pts)
sx = sum(p for p, _ in pts); sy = sum(t for _, t in pts)
sxx = sum(p * p for p, _ in pts); sxy = sum(p * t for p, t in pts)
den = n * sxx - sx * sx
if abs(den) < 1e-12:
    sys.exit(0)
b = (n * sxy - sx * sy) / den
a = (sy - b * sx) / n

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
    print("  VERDICT: latency tracks pixel count almost exactly here.")
elif share < 25:
    print("  VERDICT: mostly compute-bound, with a real fixed floor.")
else:
    print("  VERDICT: dominated by fixed cost, not pixels -- shrinking input")
    print("  buys little latency. Weigh that against the accuracy column above:")
    print("  if top-1 drops faster than latency, smaller is a bad trade here.")
PY
else
    warn "need at least two measured sizes to fit; got ${#FITDATA[@]}"
fi

echo
echo "  logs: $RES/{export,quant,run,diag}_*.log"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo
    warn "these steps failed and are missing from the table:"
    printf '      %s\n' "${FAILED[@]}"
fi
