#!/usr/bin/env bash
# W4A8 accuracy gate: COCO val2017 mAP@50-95 of the shipped W8A8 YOLOv8n/s against 4-bit-weight
# emulations of them, on the CPU. No NPU. The kill line and the verdict rule live in
# tools/w4a8_verdict.py and are pre-registered (committed) before any evaluation runs.
#
#   ./scripts/w4a8-eval.sh controls   # emit every variant into scratch/int4_w4a8/ and check it with
#                                     # ONNX Runtime, no mAP: tools/w4a8_emulate.py, resnet_env17
#   ./scripts/w4a8-eval.sh gate       # the engine's integer oracle against ONNX Runtime on every
#                                     # form-b file: tools/w4a8_engine_gate.py, mlir-aie-iron
#   ./scripts/w4a8-eval.sh prereg     # the pre-registration log; commit it before anything below
#   ./scripts/w4a8-eval.sh smoke      # B8 optimized on 500 images (the committed CPU figures are
#                                     # 30.25 / 40.77) and B8 unoptimized on 20 (timing)
#   ./scripts/w4a8-eval.sh matrix     # all 5000 images, serial: n then s for B8, E1, E2, then
#                                     # E1h, U, and B8 with ONNX Runtime's optimizations
#   ./scripts/w4a8-eval.sh verdict    # tools/w4a8_verdict.py over the matrix logs
#
# Options: --threads N pins ONNX Runtime's intra-op threads for every row (default 8; the float
# summation order of an unoptimized QDQ Conv depends on it). --machine TAG names the logs
# (default: desktop2 on DESKTOP-CBL5NUA, else required).
#
# Every evaluation is --ep cpu and never --fresh (that clears the NPU compile cache). The
# decision rows use --ort-opt disable_all: a QDQ file whose weights were rewritten needs the
# unoptimized reference, which the engine's integer oracle matches layer for layer.
#
# Logs go to results/int4/ (UTF-8, profile path scrubbed), detections to the git-ignored
# results/dets_<model>_<arm>_cpu_<unopt|opt>.json. The script refuses to replace either; delete by
# hand to re-run. Variant files stay in the git-ignored scratch/int4_w4a8/. A variant with a
# per-channel weight scale is *_ortonly.onnx and must never reach ignite-compile.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

STAGE="" THREADS=8 MACHINE=""
while [ $# -gt 0 ]; do
    case "$1" in
        controls|gate|prereg|smoke|matrix|verdict) STAGE="$1" ;;
        --threads) THREADS="$2"; shift ;;
        --machine) MACHINE="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         die "unknown argument $1" ;;
    esac
    shift
done
[ -n "$STAGE" ] || { usage "${BASH_SOURCE[0]}"; die "name a stage"; }
if [ -z "$MACHINE" ]; then
    case "$(hostname)" in
        DESKTOP-CBL5NUA) MACHINE=desktop2 ;;
        *) die "unknown host $(hostname): pass --machine TAG" ;;
    esac
fi

DATE="$(date +%Y%m%d)"
W4=scratch/int4_w4a8
RAW="$W4/raw"
OUT=results/int4
MODELS=(yolov8n yolov8s)

need_dir  data/coco/val2017                          "run: pipelines/yolov8n/2_fetch_coco.py"
need_file data/coco/annotations/instances_val2017.json
for m in "${MODELS[@]}"; do
    need_file "models/${m}_cut.onnx"
    need_file "models/${m}_cut_xint8.onnx"
done
mkdir -p "$W4" "$RAW" "$OUT"

# refuse <path>... -- stop before any work if a target already exists.
refuse() {
    local p
    for p in "$@"; do
        [ ! -e "$p" ] || die "$p exists; this script never replaces a log or detections. Delete it by hand to re-run."
    done
}

# scrub <raw> <log> -- copy with the local profile path replaced by C:\Users\<user>.
scrub() {
    python -c 'import pathlib,re,sys; s=pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"); s=re.sub(r"(?i)([A-Z]:[\\/]Users[\\/])[^\\/\s\"<>]+",r"\1<user>",s); s=re.sub(r"(?i)(/[A-Z]/Users/)[^/\s\"<>]+",r"\1<user>",s); pathlib.Path(sys.argv[2]).write_text(s,encoding="utf-8")' "$1" "$2"
}

# logged <log> <cmd...> -- run through run_logged into scratch, then scrub into results/int4/.
logged() {
    local log="$1" rc=0; shift
    local raw="$RAW/$(basename "$log")"
    rm -f "$raw"
    run_logged "$raw" "$@" || rc=$?
    scrub "$raw" "$log"
    return $rc
}

# arm_model <model> <arm> -- the file of one arm: exactly one of <name>.onnx, <name>_ortonly.onnx.
arm_model() {
    local m="$1" arm="$2" a b
    if [ "$arm" = b8 ]; then echo "models/${m}_cut_xint8.onnx"; return 0; fi
    a="$W4/${m}_w4a8_${arm}.onnx"; b="$W4/${m}_w4a8_${arm}_ortonly.onnx"
    if [ -f "$a" ] && [ ! -f "$b" ]; then echo "$a"
    elif [ -f "$b" ] && [ ! -f "$a" ]; then echo "$b"
    else die "want exactly one of $a and $b -- run: $0 controls"
    fi
}

EMIT_VARIANTS=(w8check e1 e2 e1h u)

controls_emit() {   # controls_emit <model>: every variant, then the ONNX Runtime comparisons
    local m="$1" v
    for v in "${EMIT_VARIANTS[@]}"; do
        printf '\n### emit %s %s\n' "$m" "$v"
        python tools/w4a8_emulate.py emit --float "models/${m}_cut.onnx" --xint8 "models/${m}_cut_xint8.onnx" \
            --variant "$v" --out "$W4/" --sidecar "$OUT/w4a8_sidecar_${m}_${v}_${MACHINE}_${DATE}.json" || return 1
    done
    printf '\n### emit %s e1 --form a (control: must match form b bit for bit)\n' "$m"
    python tools/w4a8_emulate.py emit --float "models/${m}_cut.onnx" --xint8 "models/${m}_cut_xint8.onnx" \
        --variant e1 --form a --out "$W4/" --sidecar "$OUT/w4a8_sidecar_${m}_e1forma_${MACHINE}_${DATE}.json" \
        || return 1
    printf '\n### compare shipped vs w8check (must be bit-identical)\n'
    python tools/w4a8_emulate.py compare "models/${m}_cut_xint8.onnx" "$W4/${m}_w8check.onnx" --n 16 \
        --threads "$THREADS" || return 1
    printf '\n### compare E1 form b vs E1 form a (must be bit-identical)\n'
    python tools/w4a8_emulate.py compare "$(arm_model "$m" e1)" "$W4/${m}_w4a8_e1_forma_ortonly.onnx" --n 16 \
        --threads "$THREADS" || return 1
    printf '\n### compare shipped vs E1 (negative control: must differ)\n'
    local out rc=0
    out="$(python tools/w4a8_emulate.py compare "models/${m}_cut_xint8.onnx" "$(arm_model "$m" e1)" --n 2 \
        --threads "$THREADS" 2>&1)" || rc=$?
    printf '%s\n' "$out"
    if [ "$rc" != 1 ] || ! grep -q "outputs DIFFER" <<< "$out"; then
        echo "[controls] FAIL: the negative control did not report differing outputs (exit $rc)"
        return 1
    fi
    echo "[controls] negative control OK: compare tells the shipped file from E1"
    echo "[controls] PASS"
}

controls_gate() {   # controls_gate <model>: the engine gate on every form-b file
    local m="$1" arm f
    for arm in b8 e1 e1h e2 u; do
        f="$(arm_model "$m" "$arm")" || return 1
        printf '\n### engine gate %s %s\n' "$m" "$arm"
        case "$f" in
            *_ortonly.onnx) echo "[gate] $f holds a per-channel (form-a) conv: no engine lowering, no gate row" ;;
            *) python tools/w4a8_engine_gate.py "$f" --coco 4 || return 1 ;;
        esac
    done
    echo "[gate] every form-b file PASS"
}

stage_controls() {
    local m logs=()
    for m in "${MODELS[@]}"; do
        logs+=("$OUT/w4a8_controls_${m}_${MACHINE}_${DATE}.log")
        for v in "${EMIT_VARIANTS[@]}" e1forma; do logs+=("$OUT/w4a8_sidecar_${m}_${v}_${MACHINE}_${DATE}.json"); done
    done
    refuse "${logs[@]}"
    use_env resnet_env17
    for m in "${MODELS[@]}"; do
        step "controls $m (emit + ONNX Runtime comparisons)"
        logged "$OUT/w4a8_controls_${m}_${MACHINE}_${DATE}.log" controls_emit "$m" || die "controls $m failed"
    done
    ok "controls pass; next: $0 gate"
}

stage_gate() {
    local m logs=()
    for m in "${MODELS[@]}"; do logs+=("$OUT/w4a8_engine_gate_${m}_${MACHINE}_${DATE}.log"); done
    refuse "${logs[@]}"
    use_env mlir-aie-iron
    PYTHONPATH="$(cygpath -w "$REPO_ROOT/src")"
    export PYTHONPATH
    # the repo-root ignite_xdna/ forwards to src/; what matters is where the compiler modules load from
    python -c 'import sys, pathlib; from ignite_xdna.compiler import graph_ir; p = pathlib.Path(graph_ir.__file__).resolve(); r = pathlib.Path(sys.argv[1]).resolve(); print(f"graph_ir from {p}"); sys.exit(0 if r in p.parents else f"not under {r}")' "$(cygpath -w "$REPO_ROOT/src")" \
        || die "ignite_xdna.compiler does not come from this checkout's src/"
    for m in "${MODELS[@]}"; do
        step "engine gate $m"
        logged "$OUT/w4a8_engine_gate_${m}_${MACHINE}_${DATE}.log" controls_gate "$m" || die "engine gate $m failed"
    done
    ok "engine gate passes; next: $0 prereg, then commit"
}

stage_prereg() {
    local log="$OUT/w4a8_accuracy_prereg_${MACHINE}_${DATE}.log" files=() m arm
    refuse "$log"
    for m in "${MODELS[@]}"; do
        files+=("models/${m}_cut.onnx" "models/${m}_cut_xint8.onnx")
        for arm in e1 e2 e1h u; do files+=("$(arm_model "$m" "$arm")"); done
    done
    use_env resnet_env17
    logged "$log" python tools/w4a8_verdict.py --prereg --threads "$THREADS" --models "${files[@]}" \
        || die "prereg failed"
    ok "commit $log (and the controls) before: $0 smoke"
}

# eval_row <model> <arm> <unopt|opt> <n> -- one 5_eval_map.py run; n 0 = all 5000 images.
eval_row() {
    local m="$1" arm="$2" opt="$3" n="$4" f tag o t0
    f="$(arm_model "$m" "$arm")"
    tag="${m}_${arm}_cpu_${opt}"
    [ "$n" = 0 ] || tag="${tag}_n${n}"
    o=default; [ "$opt" = unopt ] && o=disable_all
    step "$([ "$n" = 0 ] && echo eval || echo smoke) $tag  ($f)"
    # the snapshot names processes, so it is taken raw and scrubbed like the logs
    printf '\n### %s\n' "$tag" >> "$RAW/$(basename "$WITNESS")"
    check_host_load warn "$RAW/$(basename "$WITNESS")"
    scrub "$RAW/$(basename "$WITNESS")" "$WITNESS"
    t0=$SECONDS
    logged "$(row_log "$m" "$arm" "$opt" "$n")" python pipelines/yolov8n/5_eval_map.py --model "$f" --ep cpu \
        --n "$n" --ort-opt "$o" --ort-threads "$THREADS" --dets "results/dets_${tag}.json" \
        || die "$tag failed"
    info "$tag: $((SECONDS - t0)) s"
}

row_log() {   # row_log <model> <arm> <opt> <n>
    if [ "$4" = 0 ]; then echo "$OUT/eval_${1}_${2}_cpu_${3}_${MACHINE}_${DATE}.log"
    else echo "$OUT/smoke_${1}_${2}_cpu_${3}_n${4}_${MACHINE}_${DATE}.log"; fi
}

row_dets() {  # row_dets <model> <arm> <opt> <n>
    if [ "$4" = 0 ]; then echo "results/dets_${1}_${2}_cpu_${3}.json"
    else echo "results/dets_${1}_${2}_cpu_${3}_n${4}.json"; fi
}

run_rows() {  # run_rows <witness> "<model> <arm> <opt> <n>"...
    WITNESS="$1"; shift
    local r targets=("$WITNESS")
    for r in "$@"; do
        # shellcheck disable=SC2086
        targets+=("$(row_log $r)" "$(row_dets $r)")
    done
    refuse "${targets[@]}"
    rm -f "$RAW/$(basename "$WITNESS")"
    use_env resnet_env17
    python -c "import pycocotools" 2>/dev/null || die "pycocotools missing in resnet_env17"
    for r in "$@"; do
        # shellcheck disable=SC2086
        eval_row $r
    done
}

stage_smoke() {
    ls "$OUT"/w4a8_accuracy_prereg_*.log >/dev/null 2>&1 || die "no pre-registration log: run $0 prereg and commit it"
    run_rows "$OUT/load_w4a8_smoke_${MACHINE}_${DATE}.log" \
        "yolov8n b8 opt 500" "yolov8s b8 opt 500" "yolov8n b8 unopt 20" "yolov8s b8 unopt 20"
    info "expected from results/aie/silu_epilogue/: yolov8n 30.25, yolov8s 40.77 (500 images, CPU, optimized)"
    grep -H "^mAP@50-95" "$OUT"/smoke_*_b8_cpu_opt_n500_"${MACHINE}"_"${DATE}".log || true
}

stage_matrix() {
    ls "$OUT"/w4a8_accuracy_prereg_*.log >/dev/null 2>&1 || die "no pre-registration log: run $0 prereg and commit it"
    run_rows "$OUT/load_w4a8_matrix_${MACHINE}_${DATE}.log" \
        "yolov8n b8 unopt 0" "yolov8n e1 unopt 0" "yolov8n e2 unopt 0" \
        "yolov8s b8 unopt 0" "yolov8s e1 unopt 0" "yolov8s e2 unopt 0" \
        "yolov8n e1h unopt 0" "yolov8s e1h unopt 0" \
        "yolov8n u unopt 0" "yolov8s u unopt 0" \
        "yolov8n b8 opt 0" "yolov8s b8 opt 0"
    ok "matrix done; next: $0 verdict"
}

stage_verdict() {
    local log="$OUT/w4a8_accuracy_verdict_${MACHINE}_${DATE}.log"
    refuse "$log"
    use_env resnet_env17
    # shellcheck disable=SC2046
    logged "$log" python tools/w4a8_verdict.py --logs $(ls "$OUT"/eval_yolov8*_cpu_*_"${MACHINE}"_*.log) \
        || die "verdict incomplete, see $log"
}

"stage_$STAGE"
