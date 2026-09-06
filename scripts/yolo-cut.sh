#!/usr/bin/env bash
# YOLOv8n on the NPU, end to end: cut -> quantize -> CPU sanity -> NPU -> verdict.
#
# The float decode tail is what made the VitisAI EP refuse the whole graph.
# Removing it takes the EP from 0 to 922/929 nodes and inference from 39 ms to
# ~9 ms. See "The YOLOv8 partitioning failure" in docs/DECISIONS.md.
#
#   ./scripts/yolo-cut.sh                # yolov8n, XINT8, 100 calib images
#   ./scripts/yolo-cut.sh --variant s    # yolov8s: ~3x the compute, +7 mAP
#   ./scripts/yolo-cut.sh --variant m --limit 300 --adaround   # best accuracy
#   ./scripts/yolo-cut.sh --limit 32     # faster; fine for the "does the EP
#                                        # take it" question, which is what
#                                        # this script is really for
#   ./scripts/yolo-cut.sh --adaround     # XINT8 + AdaRound (slow, many minutes)
#   ./scripts/yolo-cut.sh --skip-quant   # reuse the existing quantized model
#
# Steps 1-2 need resnet_env (Quark); steps 3-5 need resnet_env17 (NPU).
#
# Watch the disk: Quark spools calibration activations PER IMAGE at 640x640
# and does not clean up if it dies -- ~105 MB/image for yolov8n (300 images
# ~31 GB) but ~198 MB/image for yolov8s (300 images ~58 GB, more than most
# machines have free). The script sizes the check per variant and cleans up
# after itself, including on Ctrl-C.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ADAROUND=0 SKIP_QUANT=0 LIMIT=100 SOURCE="assets/test_image.jpg" REQUANT=0
VARIANT=n
while [ $# -gt 0 ]; do
    case "$1" in
        --adaround)   ADAROUND=1 ;;
        --skip-quant) SKIP_QUANT=1 ;;
        --recut)      REQUANT=1 ;;
        --limit)      LIMIT="$2"; shift ;;
        --variant)    VARIANT="$2"; shift ;;
        --source)     SOURCE="$2"; shift ;;
        -h|--help)    usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)            die "unknown flag $1" ;;
    esac
    shift
done

case "$VARIANT" in n|s|m|l|x) ;; *) die "--variant must be one of n s m l x" ;; esac

TAG=$([ "$ADAROUND" = 1 ] && echo xint8_adaround || echo xint8)
BASE="models/yolov8${VARIANT}.onnx"
CUT="models/yolov8${VARIANT}_cut.onnx"
QUANT="models/yolov8${VARIANT}_cut_${TAG}.onnx"
# One cache key for every variant, so a variant switch MUST recompile. Step 4
# always passes --fresh, which is what makes that safe.
CACHE_KEY="yolocutcachekey"

need_dir data/coco_calib "run: python pipelines/yolov8n/2_fetch_coco.py --n-calib 300"

# --------------------------------------------------------------- 0. export
if [ ! -f "$BASE" ]; then
    step "0/5  exporting yolov8${VARIANT} to ONNX (downloads weights if needed)"
    use_env resnet_env
    python pipelines/yolov8n/1_export.py --weights "models/yolov8${VARIANT}.pt" --size 640
    need_file "$BASE" "export did not produce $BASE"
fi

# ---------------------------------------------------------------- 1. cut head
if [ -f "$CUT" ] && [ "$REQUANT" = 0 ]; then
    step "1/5  head cut -- $CUT exists, skipping (--recut to redo)"
else
    step "1/5  cutting the decode tail off the graph"
    use_env resnet_env
    python pipelines/yolov8n/1b_cut_head.py --in "$BASE" --out "$CUT"
fi
need_file "$CUT"

# ---------------------------------------------------------------- 2. quantize
if [ "$SKIP_QUANT" = 1 ]; then
    step "2/5  quantize -- skipped (--skip-quant)"
    need_file "$QUANT" "nothing to skip to; drop --skip-quant"
else
    step "2/5  quantizing to ${TAG}  (${LIMIT} calibration images)"
    MB_PER_IMG=$(calib_mb_per_image "$VARIANT")
    require_disk $(( (LIMIT * MB_PER_IMG + 1023) / 1024 + 2 )) \
                 "Quark's calibration cache (yolov8${VARIANT}: ~${MB_PER_IMG} MB/image)"
    quark_guard
    [ "$ADAROUND" = 1 ] && info "AdaRound is slow -- expect many minutes"
    use_env resnet_env
    ARGS=(--in "$CUT" --calib-dir data/coco_calib --limit "$LIMIT" --out "$QUANT")
    [ "$ADAROUND" = 1 ] && ARGS+=(--adaround)
    run_logged "results/yolo_cut_quantize_${TAG}.log" \
        python pipelines/yolov8n/3b_quantize_cut.py "${ARGS[@]}"
fi
need_file "$QUANT"

# ------------------------------------------------------------- 3. CPU sanity
# If the quantized cut model is broken on CPU, an NPU run tells you nothing.
step "3/5  CPU sanity check on the quantized model"
npu_env
run_logged "results/yolo_cut_cpu.log" \
    python pipelines/yolov8n/4_detect.py --model "$QUANT" --ep cpu \
        --source "$SOURCE" --runs 5

CPU_DETS=$(grep -oE '^[0-9]+ detections' "results/yolo_cut_cpu.log" | head -1 | cut -d' ' -f1 || true)
[ "${CPU_DETS:-0}" -gt 0 ] || die "the quantized model found nothing on CPU -- fix that before touching the NPU"
ok "$CPU_DETS detections on CPU"

# ------------------------------------------------------------------ 4. NPU
step "4/5  NPU run  (--fresh: recompiling, this looks frozen but isn't)"
run_logged "results/yolo_cut_npu.log" \
    python pipelines/yolov8n/4_detect.py --model "$QUANT" --ep npu \
        --source "$SOURCE" --fresh --log 1

# ---------------------------------------------------------------- 5. verdict
step "5/5  what the EP actually took"
run_logged "results/yolo_cut_diag.log" \
    python tools/diag_ep.py --model "$QUANT" --cache-key "$CACHE_KEY" || true

NPU_MS=$(grep -oE 'infer +mean +[0-9.]+' "results/yolo_cut_npu.log" | head -1 \
         | grep -oE '[0-9.]+$' || echo "")

# "Fast enough" is per variant, not a single number. Measured 2026-09-05 on the
# full benchmark: n 8.94 ms, s 15.63 ms -- a flat 15 ms budget called a perfectly
# healthy yolov8s run slow. Latency scales well below FLOPs on this hardware
# (s is 3.2x the arithmetic of n for 1.75x the time), so these budgets are set
# from measurement for n and s, and extrapolated generously for m/l/x.
case "$VARIANT" in
    n) BUDGET_MS=13 ;;   # measured 8.94
    s) BUDGET_MS=22 ;;   # measured 15.63
    m) BUDGET_MS=45 ;;   # extrapolated -- replace with a measurement
    l) BUDGET_MS=75 ;;
    x) BUDGET_MS=110 ;;
esac

printf '\n%s================ RESULT ================%s\n' "$BOLD" "$OFF"
if npu_verdict "$CACHE_KEY"; then
    if [ -n "$NPU_MS" ] && awk "BEGIN{exit !($NPU_MS < $BUDGET_MS)}"; then
        printf '%s  The float tail was the problem. %s ms/frame -- ship it.%s\n' \
            "$GREEN" "$NPU_MS" "$OFF"
        echo "  Next: COCO mAP eval, then ./scripts/yolo-demo.sh for the webcam."
    else
        printf '%s  EP engaged but still slow (%s ms).%s\n' "$YELLOW" "${NPU_MS:-?}" "$OFF"
        echo "  A real partitioning problem -- but you now have a compile log and an"
        echo "  operator table for the first time. Read results/yolo_cut_npu.log and the"
        echo "  per-node split in results/yolo_cut_diag.log."
    fi
else
    printf '%s  Still zero NPU nodes: the decode tail was never the cause.%s\n' "$RED" "$OFF"
    echo "  The cause is in the backbone. The cut graph still has all 57 SiLU"
    echo "  (-> HardSigmoid) patterns, all 8 Split, both Resize, 13 Concat."
    echo "  Next: bisect by feeding the EP shorter prefixes of the model until one"
    echo "  compiles -- extract up to the end of /model.9/ (SPPF, no Resize), then"
    echo "  /model.4/. That isolates the op class in about three runs."
fi
echo
echo "  logs: results/yolo_cut_{cpu,npu,diag}.log"
