#!/usr/bin/env bash
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# W4V: post-training W4 recovery on the shipped W8A8 YOLOv8n/s, judged on COCO val2017 mAP@50-95 on the
# CPU. No NPU. The arms, the cut, K-D and every hyperparameter live in tools/w4v_verdict.py (the accepted
# pre-registration text, its protocol JSON and the verdict code: the frozen triple every log prints).
#
#   ./scripts/w4v-eval.sh selftest   # both tools' synthetic selftests (light)
#   ./scripts/w4v-eval.sh controls   # C0-C5 on the RTN arms, C1-C4 byte for byte against gate D
#   ./scripts/w4v-eval.sh parity     # C6: quant/'s int8 parity after the additive grid (a, b, c)
#   ./scripts/w4v-eval.sh repro      # C9: B8 and RTN E1, E2, U at 5,000, both models; a mismatch is a STOP
#   ./scripts/w4v-eval.sh emit1      # the MX sensitivity and y_f statistics, then every other tier-1 arm
#   ./scripts/w4v-eval.sh prereg     # pins the models, the captures and every tier-1 arm; commit it next
#   ./scripts/w4v-eval.sh bias1      # C5b on the tier-1 int32-bias arms not yet audited (v2.5 C; a FAIL is a STOP)
#   ./scripts/w4v-eval.sh screen1    # B8 and the tier-1 arms at 500, then the cut (tier 1)
#   ./scripts/w4v-eval.sh emit2      # the tier-2 arms over the pinned tier-1 files, then their pins log
#   ./scripts/w4v-eval.sh bias2      # C5b on the tier-2 +BC arms (v2.5 C; a FAIL is a STOP)
#   ./scripts/w4v-eval.sh screen2    # the tier-2 arms at 500, then the cut (tier 2, against tier 1's B8)
#   ./scripts/w4v-eval.sh gate       # C8: the engine's integer oracle on every advancing form-b file
#   ./scripts/w4v-eval.sh decisive   # B8 and every advancing arm at 5,000
#   ./scripts/w4v-eval.sh verdict    # K-D on all 5,000 and on H, from the repro and decisive logs
#
# Options: --machine TAG names the logs (default: desktop2 on DESKTOP-CBL5NUA, else required).
# The ONNX Runtime threads are the protocol's 8 and are not an option.
#
# Every evaluation is 5_eval_map.py --ep cpu --ort-opt disable_all --ort-threads 8, never --fresh, with
# --dets into the git-ignored scratch/w4v/dets/; a row's log is the frozen triple, the eval and the
# DETS_SHA256 line. A row whose eval ends in v2.5 B.1's "no detections at all" exit gets a w4v_zerodet_ record
# from --zerodet, citing its arm's audit, or the stage stops (B.3, B.4). The screen and the decisive rows refuse a
# file that no committed prereg or pins log pins. Logs go to results/int4/w4v_* (UTF-8, profile path scrubbed),
# sidecars to results/int4/w4v_sidecars/; the script refuses to replace either (delete by hand to re-run). Arm
# files, captures and detections stay in scratch/w4v/. A file with a per-channel weight scale is *_ortonly.onnx and
# must never reach ignite-compile.
# models/, data/ and build/ are links to the main checkout's, as in the int4-study worktree.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

STAGE="" MACHINE=""
while [ $# -gt 0 ]; do
    case "$1" in
        selftest|controls|parity|repro|emit1|prereg|bias1|screen1|emit2|bias2|screen2|gate|decisive|verdict) STAGE="$1" ;;
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

THREADS=8
# the prereg text is not ASCII, and a Windows pipe is cp1252 without this
export PYTHONIOENCODING=utf-8
# the profile directory's name, read at run time (USERPROFILE's last component, else HOME's) and never written into
# a committed file; a name too short or too generic to grep for safely stops the run before anything is made
PROFILE_DIR="${USERPROFILE:-$HOME}"
PROFILE_DIR="${PROFILE_DIR//\\//}"
PROFILE_DIR="${PROFILE_DIR%/}"
PROFILE_NAME="${PROFILE_DIR##*/}"
[ "${#PROFILE_NAME}" -ge 3 ] || die "the profile name is under 3 characters; the scrub and its check cannot be trusted"
[ "$(printf '%s' "$PROFILE_NAME" | tr '[:upper:]' '[:lower:]')" != user ] \
    || die "the profile name is \"user\", which the logs hold anyway; the scrub and its check cannot be trusted"
DATE="$(date +%Y%m%d)"
TAG="${MACHINE}_${DATE}"
W=scratch/w4v
ARMS="$W/arms"
CAP="$W/captures"
DETS="$W/dets"
PAR="$W/parity"
RAW="$W/raw"
OUT=results/int4
SIDE="$OUT/w4v_sidecars"
GATE_D_DETS=../int4-study/results   # gate D's surviving detections (w4a8-eval.sh:217-219), report-only
V24_DETS=""   # screen1 only: the archived v2.4 detections (v2.5 D, option A), for the report-only V24_DETS_SHA256 line
MODELS=(yolov8n yolov8s)
C9_ARMS=(rtn_e1 rtn_e2 rtn_u)

if [ "$STAGE" != selftest ]; then
    need_dir  data/coco/val2017                          "link data/ to the main checkout's"
    need_file data/coco/annotations/instances_val2017.json
    need_dir  data/coco_calib
    for m in "${MODELS[@]}"; do
        need_file "models/${m}_cut.onnx"                 "link models/ to the main checkout's"
        need_file "models/${m}_cut_xint8.onnx"
    done
fi
mkdir -p "$W" "$ARMS" "$CAP" "$DETS" "$PAR" "$RAW" "$OUT" "$SIDE"

# refuse <path>... -- stop before any work if a target already exists.
refuse() {
    local p
    for p in "$@"; do
        [ ! -e "$p" ] || die "$p exists; this script never replaces a log or detections. Delete it by hand to re-run."
    done
}

# scrub <raw> <log> -- copy with the local profile path replaced by <user>: C:\Users\<user>, with any run of
# separators (a repr or JSON escape doubles them), /c/Users/<user>, and the name as a path component anywhere.
scrub() {
    python -c 'import pathlib, re, sys
s = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
s = re.sub(r"(?i)([A-Z]:[\\/]+Users[\\/]+)[^\\/\s\"<>]+", r"\1<user>", s)
s = re.sub(r"(?i)(/[A-Z]/+Users/+)[^/\s\"<>]+", r"\1<user>", s)
s = re.sub(r"(?im)(?<=[\\/])" + re.escape(sys.argv[3]) + r"(?=[\\/\s\"<>]|$)", "<user>", s)
pathlib.Path(sys.argv[2]).write_text(s, encoding="utf-8")' "$1" "$2" "$PROFILE_NAME" \
        || die "the scrub of $1 into $2 failed"
}

# no_profile <file>... -- die if a log or a sidecar still holds the profile directory's name, if it is missing, or
# if grep cannot read it (exit 2 must not pass as "no match").
no_profile() {
    local f rc
    for f in "$@"; do
        [ -f "$f" ] || die "$f is missing, so it cannot be checked for the profile name"
        rc=0
        grep -q -i -F -- "$PROFILE_NAME" "$f" || rc=$?
        case "$rc" in
            0) die "$f still holds the profile name after the scrub; fix it by hand" ;;
            1) ;;
            *) die "grep could not read $f (exit $rc), so it is not checked for the profile name" ;;
        esac
    done
}

# logged <log> <cmd...> -- run through run_logged into scratch, then scrub into results/int4/ and check it.
logged() {
    local log="$1" rc=0; shift
    local raw="$RAW/$(basename "$log")"
    rm -f "$raw"
    run_logged "$raw" "$@" || rc=$?
    scrub "$raw" "$log"
    no_profile "$log"
    return $rc
}

# sidecars_clean -- the sidecars are written by w4_recover.py, not scrubbed: they must hold relative paths only.
sidecars_clean() {
    local f
    for f in "$SIDE"/*.json; do [ -e "$f" ] && no_profile "$f"; done
    return 0
}

# arm_file <model> <arm> -- b8 is the shipped file; an arm is exactly one of <name>.onnx, <name>_ortonly.onnx.
arm_file() {
    local m="$1" arm="$2" a b
    if [ "$arm" = b8 ]; then echo "models/${m}_cut_xint8.onnx"; return 0; fi
    a="$ARMS/${m}_w4v_${arm}.onnx"; b="$ARMS/${m}_w4v_${arm}_ortonly.onnx"
    if [ -f "$a" ] && [ ! -f "$b" ]; then echo "$a"
    elif [ -f "$b" ] && [ ! -f "$a" ]; then echo "$b"
    else die "want exactly one of $a and $b"
    fi
}

# arms_of <tier> -- the tier's arms in section 3.7's row order, from the frozen matrix.
arms_of() {
    python -c 'import sys; sys.path.insert(0, "tools"); import w4v_verdict as v; print(" ".join(v.arms(int(sys.argv[1]))))' "$1"
}

# tier_arms <tier> -- TIER_ARMS := arms_of <tier>, or die: `for arm in $(arms_of ...)` ignores a failed or empty
# read, runs no arm, and still reports success.
tier_arms() {
    TIER_ARMS="$(arms_of "$1")" || die "cannot read tier $1's arms from the frozen matrix"
    [ -n "$TIER_ARMS" ] || die "the frozen matrix gives tier $1 no arms"
}

# ready <stage> -- no stage runs until every earlier stage's logs are in HEAD and unchanged (`git add` alone is not
# committed), carry their success lines, hold no W4V_CONTROL FAIL, and hold only hash blocks equal to the tools'
# (tools/w4v_verdict.py --ready: its STAGES and stage_logs); nor while any log of the stage itself exists, of any date,
# committed or not (a restart needs the gate's ruling).
ready() {
    python tools/w4v_verdict.py --ready "$1" \
        || die "$1 waits: an earlier stage's logs are not committed, not complete or not frozen (PROBLEM lines above)"
}

# advanced -- "<model> <arm>" per arm advanced by the one committed cut log of each tier (tools/w4v_verdict.py
# --advanced: SCREEN COMPLETE, every pinned arm screened, the frozen state holds).
advanced() {
    local out rc=0
    out="$(python tools/w4v_verdict.py --advanced)" || rc=$?
    if [ "$rc" != 0 ]; then printf '%s\n' "$out" >&2; return 1; fi
    printf '%s\n' "$out" | sed -n 's/^ADVANCED //p'
}

# ------------------------------------------------------------------------------------------------ eval rows

row_name() { echo "w4v_${1}_${2}_${3}_n${4}"; }   # row_name <stage> <model> <arm> <n>

# v24_dets <dets> <archived dets> -- v2.5 D's report-only line: the archived v2.4 detections' sha256 and whether this
# row's are byte-identical. A row the held v2.4 run never reached has no archived file and gets "none (no v2.4 row)",
# which is no comparison, so every "byte-identical False" is a real mismatch. Printed, not required; row_body prints
# it before it returns, so a zero-detection row's log holds it before --zerodet pins that log's sha.
v24_dets() {
    python -c 'import hashlib, pathlib, sys
a, b = (pathlib.Path(p) for p in sys.argv[1:3])
sa = hashlib.sha256(a.read_bytes()).hexdigest() if a.is_file() else "missing"
if b.is_file():
    sb = hashlib.sha256(b.read_bytes()).hexdigest()
    print(f"V24_DETS_SHA256 {sb} (byte-identical {sa == sb}) {sys.argv[2]} (report-only; this row: {sa})")
else:
    print(f"V24_DETS_SHA256 none (no v2.4 row) {sys.argv[2]} (report-only; this row: {sa})")' "$1" "$2"
}

# row_body <stage> <model> <arm> <model file> <n flag> <dets> -- one row's log. A failed eval returns the eval's own
# exit code, after the report-only V24 line, so eval_row can look for v2.5 B.1's exit (1, "no detections at all").
row_body() {
    local stage="$1" m="$2" arm="$3" f="$4" n="$5" dets="$6" g rc=0
    python tools/w4v_verdict.py --hashes || return 1
    if [ "$stage" != repro ]; then python tools/w4v_verdict.py --pinned "$f" || return 1; fi
    python pipelines/yolov8n/5_eval_map.py --model "$f" --ep cpu --n "$n" --ort-opt disable_all \
        --ort-threads "$THREADS" --dets "$dets" || rc=$?
    if [ -n "$V24_DETS" ]; then v24_dets "$dets" "$V24_DETS/$(basename "$dets")" || return 1; fi
    [ "$rc" = 0 ] || return "$rc"
    python tools/w4v_verdict.py --dets-sha "$dets" || return 1
    if [ "$stage" = repro ]; then
        g="$GATE_D_DETS/dets_${m}_${arm#rtn_}_cpu_unopt.json"
        python -c 'import hashlib, pathlib, sys
a, b = (pathlib.Path(p) for p in sys.argv[1:3])
sa = hashlib.sha256(a.read_bytes()).hexdigest()
sb = hashlib.sha256(b.read_bytes()).hexdigest() if b.is_file() else "missing"
print(f"GATE_D_DETS_SHA256 {sb} {sys.argv[2]} (report-only; this row: {sa}, byte-identical {sa == sb})")' "$dets" "$g"
    fi
}

# audit_of <model> <arm> -- the audit log a zero-detection record cites: v2.5 v2_5.audits, through
# tools/w4v_verdict.py audit_for (the committed GENUINE log for bc_e1, else the one committed bias<tier> log).
audit_of() {
    python -c 'import sys
sys.path.insert(0, "tools")
import w4v_verdict as v
p, problems = v.audit_for(sys.argv[1], sys.argv[2])
for x in problems:
    print(f"PROBLEM {x}", file=sys.stderr)
if problems or p is None:
    sys.exit(1)
print(v.shown(p))' "$1" "$2"
}

# zerodet_name <model> <arm> <n> -- the record's log (v2.5 B.4; it names no stage).
zerodet_name() { echo "$OUT/w4v_zerodet_${1}_${2}_n${3}_${TAG}.log"; }

# eval_row <stage> <model> <arm> <n> -- n is 500 or 5000 (5_eval_map.py --n 0 means all 5,000). A row that ends in
# B.1's exit gets its record, or the stage stops (v2.5 B.3, B.4); the row log itself is never touched again. A refused
# record stays in results/int4/ with its PROBLEM lines, like any refused log: the stage's --ready then refuses a
# restart (own_logs, zerodet_globs) until the gate rules, and nothing here deletes it.
eval_row() {
    local stage="$1" m="$2" arm="$3" n="$4" f row nflag t0 rc=0 log audit rec
    f="$(arm_file "$m" "$arm")" || die "no file for $m $arm"
    row="$(row_name "$stage" "$m" "$arm" "$n")"
    log="$OUT/${row}_${TAG}.log"
    nflag="$n"; [ "$n" = 5000 ] && nflag=0
    step "$stage $m $arm on $n images  ($f)"
    # the snapshot names processes, so it is taken raw and scrubbed like the logs
    printf '\n### %s\n' "$row" >> "$RAW/$(basename "$WITNESS")"
    check_host_load warn "$RAW/$(basename "$WITNESS")"
    scrub "$RAW/$(basename "$WITNESS")" "$WITNESS"
    no_profile "$WITNESS"
    t0=$SECONDS
    logged "$log" row_body "$stage" "$m" "$arm" "$f" "$nflag" "$DETS/${row}.json" || rc=$?
    if [ "$rc" != 0 ]; then
        [ "$rc" = 1 ] && grep -q "^no detections at all -- something is broken upstream" "$log" \
            || die "$row failed; the stage stops here and goes to the gate as it stands"
        audit="$(audit_of "$m" "$arm")" \
            || die "$row: zero detections, and no audit covers $m $arm; STOP (v2.5 B.3), to the gate"
        rec="$(zerodet_name "$m" "$arm" "$n")"
        logged "$rec" python tools/w4v_verdict.py --zerodet --log "$log" --dets "$DETS/${row}.json" --audit "$audit" \
            || die "$row: zero detections, and --zerodet refused the record; STOP (v2.5 B.3), see $rec"
        info "$row: zero detections, recorded as AP 0.00 by rule B.2 in $rec"
    fi
    info "$row: $((SECONDS - t0)) s"
}

# run_rows <stage> <witness> "<model> <arm> <n>"... -- refuse every target first, check every pin, then run.
run_rows() {
    local stage="$1" r m arm n f targets=()
    WITNESS="$2"; shift 2
    targets+=("$WITNESS")
    for r in "$@"; do
        read -r m arm n <<< "$r"
        targets+=("$OUT/$(row_name "$stage" "$m" "$arm" "$n")_${TAG}.log" "$DETS/$(row_name "$stage" "$m" "$arm" "$n").json"
                  "$(zerodet_name "$m" "$arm" "$n")")
    done
    refuse "${targets[@]}"
    use_env resnet_env17
    python -c "import pycocotools" 2>/dev/null || die "pycocotools missing in resnet_env17"
    if [ "$stage" != repro ]; then
        for r in "$@"; do
            read -r m arm n <<< "$r"
            f="$(arm_file "$m" "$arm")" || die "no file for $m $arm; nothing ran"
            python tools/w4v_verdict.py --pinned "$f" >/dev/null \
                || die "$m $arm: its file is not the pinned one; nothing ran"
        done
    fi
    # the witness log opens with the frozen hash block, as every W4V log does
    python tools/w4v_verdict.py --hashes > "$RAW/$(basename "$WITNESS")" || die "cannot start $WITNESS"
    for r in "$@"; do
        read -r m arm n <<< "$r"
        eval_row "$stage" "$m" "$arm" "$n"
    done
}

# ----------------------------------------------------------------------------------------------- the stages

selftest_body() {
    python tools/w4v_verdict.py --selftest || return 1
    python tools/w4_recover.py selftest
}

stage_selftest() {
    local log="$OUT/w4v_selftest_${TAG}.log"
    refuse "$log"
    use_env resnet_env17
    logged "$log" selftest_body || die "selftest failed, see $log"
    ok "selftest pass"
}

stage_controls() {
    local m logs=()
    for m in "${MODELS[@]}"; do logs+=("$OUT/w4v_controls_${m}_${TAG}.log"); done
    refuse "${logs[@]}"
    use_env resnet_env17
    ready controls
    for m in "${MODELS[@]}"; do
        step "controls $m (C0-C5 on the RTN arms)"
        logged "$OUT/w4v_controls_${m}_${TAG}.log" python tools/w4_recover.py controls --model "$m" \
            --out-dir "$ARMS" --captures "$CAP" --sidecar-dir "$SIDE" --tag "$TAG" || die "controls $m failed"
    done
    sidecars_clean
    ok "controls pass; next: commit the controls logs, then $0 parity"
}

# marker_of <name> -- W4V_CONTROL, or W4V_REPORT for a report-only check ("report:<name>"), which
# tools/w4v_verdict.py does not read: a report-only FAIL decides nothing.
marker_of() { case "$1" in report:*) echo "W4V_REPORT ${1#report:}" ;; *) echo "W4V_CONTROL $1" ;; esac; }

# marked <name> <scope> <pattern> <cmd...> -- run, then PASS iff it exits 0 and prints a line starting with
# the pattern (no "$" anchor: the Python lines end in CRLF here).
marked() {
    local name="$1" scope="$2" pattern="$3" out="$RAW/marked_$$.txt" rc=0
    shift 3
    { "$@" 2>&1 | tee "$out"; rc=${PIPESTATUS[0]}; } || true
    if [ "$rc" = 0 ] && grep -q "$pattern" "$out"; then
        echo "$(marker_of "$name") $scope PASS"
    else
        echo "$(marker_of "$name") $scope FAIL (exit $rc)"
        rm -f "$out"
        return 1
    fi
    rm -f "$out"
}

parity_a() {   # C6(a): the replay from the shipped scales, both models
    local m="$1"
    python tools/w4v_verdict.py --hashes || return 1
    marked C6a "$m" "^GRAPH_DIFF_PASS True" python -m quant quantize --in-model "models/${m}_cut.onnx" \
        --out "$PAR/${m}_cut_replay_cle.onnx" --cle --scales-from "models/${m}_cut_xint8.onnx" --calib-method exact
}

parity_b() {   # C6(b): exact c64 CLE calibration against the recorded Quark oracle
    python tools/w4v_verdict.py --hashes || return 1
    python -m quant quantize --in-model models/yolov8n_cut.onnx --out "$PAR/yolov8n_cut_ignition_cle_c64.onnx" \
        --cle --calib-dir data/coco_calib --limit 64 --calib-method exact \
        || { echo "$(marker_of C6b) yolov8n FAIL (the quantize)"; return 1; }
    marked C6b yolov8n "^INT8_EXACT True" python tools/quant_compare.py \
        --model "$PAR/yolov8n_cut_ignition_cle_c64.onnx" --reference models/yolov8n_cut_quark_cle_c64.onnx
}

parity_c() {   # C6(c) <name> <suffix>: AdaRound W8 on (b)'s file against the recorded Quark AdaRound oracle
    local name="$1" out="$PAR/yolov8n_cut_ignition_cle_adaround_c64$2.onnx"
    python tools/w4v_verdict.py --hashes || return 1
    python -m quant adaround --in-model models/yolov8n_cut.onnx --quant "$PAR/yolov8n_cut_ignition_cle_c64.onnx" \
        --out "$out" --calib-dir data/coco_calib || { echo "$(marker_of "$name") yolov8n FAIL (the AdaRound)"; return 1; }
    marked "$name" yolov8n "^INT8_EXACT True" python tools/quant_compare.py --model "$out" \
        --reference models/yolov8n_cut_quark_cle_adaround_c64.onnx
}

stage_parity() {
    local m logs=()
    need_file models/yolov8n_cut_quark_cle_c64.onnx
    need_file models/yolov8n_cut_quark_cle_adaround_c64.onnx
    for m in "${MODELS[@]}"; do logs+=("$OUT/w4v_parity_c6a_${m}_${TAG}.log"); done
    logs+=("$OUT/w4v_parity_c6b_yolov8n_${TAG}.log" "$OUT/w4v_parity_c6c_yolov8n_${TAG}.log"
           "$OUT/w4v_parity_c6c_env17_yolov8n_${TAG}.log")
    refuse "${logs[@]}" "$PAR/yolov8n_cut_ignition_cle_c64.onnx"
    info "disk: $(free_gb) GB free; (b)'s calibration spools about 7.1 GB (quant_yolov8n_cut_ignition_cle_c64.log:3)"
    use_env resnet_env17
    ready parity
    for m in "${MODELS[@]}"; do
        step "C6(a) $m: replay from the shipped scales"
        logged "$OUT/w4v_parity_c6a_${m}_${TAG}.log" parity_a "$m" || die "C6(a) $m failed"
    done
    step "C6(b) yolov8n: exact c64 CLE calibration against the Quark oracle"
    logged "$OUT/w4v_parity_c6b_yolov8n_${TAG}.log" parity_b || die "C6(b) failed"
    # the recorded AdaRound parity ran with ORT 1.22.1 (diff_yolov8n_cut_ignition_cle_adaround_c64.log:1233, :1272)
    use_env resnet_env
    step "C6(c) yolov8n: AdaRound W8 against the Quark oracle, resnet_env (the recorded condition)"
    logged "$OUT/w4v_parity_c6c_yolov8n_${TAG}.log" parity_c C6c "" || die "C6(c) failed"
    use_env resnet_env17
    step "C6(c) yolov8n in resnet_env17, report-only (W4V's AdaRound runs here)"
    logged "$OUT/w4v_parity_c6c_env17_yolov8n_${TAG}.log" parity_c report:C6c_env17 _env17 \
        || warn "the report-only env17 AdaRound parity did not hold; it decides nothing, see its log"
    ok "parity pass; next: commit the parity logs, then $0 repro on the gate's go"
}

stage_repro() {
    local m arm rows=() log="$OUT/w4v_repro_c9_${TAG}.log"
    use_env resnet_env17
    ready repro
    for m in "${MODELS[@]}"; do
        rows+=("$m b8 5000")
        for arm in "${C9_ARMS[@]}"; do rows+=("$m $arm 5000"); done
    done
    refuse "$log"
    run_rows repro "$OUT/load_w4v_repro_${TAG}.log" "${rows[@]}"
    # shellcheck disable=SC2046
    logged "$log" python tools/w4v_verdict.py --c9 --logs $(ls "$OUT"/w4v_repro_yolov8*_n5000_"${TAG}".log) \
        || die "C9: B8 or RTN does not reproduce gate D; STOP, see $log"
    ok "C9 pass; next: commit the repro logs, then $0 emit1 on the gate's go"
}

# emit_tier <tier> <model> -- every arm of the tier but the RTN arms the controls made; C3 twins on AdaRound and
# GPTQ E1; C7 repeats on YOLOv8n's GPTQ, BC, VC and AdaRound E1.
emit_tier() {
    local tier="$1" m="$2" arm extra
    tier_arms "$tier"   # a die here ends run_logged's subshell: logged returns 1 and no W4V_DONE is printed
    for arm in $TIER_ARMS; do
        case "$arm" in rtn_e1|rtn_e2|rtn_u) continue ;; esac
        extra=()
        case "$arm" in ar_e1|gptq_e1) extra+=(--twin) ;; esac
        if [ "$m" = yolov8n ]; then case "$arm" in gptq_e1|bc_e1|vc_e1|ar_e1) extra+=(--repeat) ;; esac; fi
        python tools/w4_recover.py emit --model "$m" --method "${arm%_*}" --form "${arm##*_}" --tier "$tier" \
            --out-dir "$ARMS" --captures "$CAP" --sidecar-dir "$SIDE" --tag "$TAG" ${extra[@]+"${extra[@]}"} \
            || return 1
    done
    echo "W4V_DONE emit${tier} $m"   # the stage's success line: every arm above exited 0 (tools/w4v_verdict.py ready)
}

stage_emit1() {
    local m logs=()
    for m in "${MODELS[@]}"; do logs+=("$OUT/w4v_sensitivity_${m}_${TAG}.log" "$OUT/w4v_emit1_${m}_${TAG}.log"); done
    refuse "${logs[@]}"
    use_env resnet_env17
    # C9 comes before any heavy stage: its logs committed, with a PASS on both models
    ready emit1
    for m in "${MODELS[@]}"; do
        step "sensitivity $m (the MX W8 sets and the y_f statistics, once per model)"
        logged "$OUT/w4v_sensitivity_${m}_${TAG}.log" python tools/w4_recover.py sensitivity --model "$m" \
            --captures "$CAP" || die "sensitivity $m failed"
    done
    for m in "${MODELS[@]}"; do
        step "emit tier 1 $m"
        logged "$OUT/w4v_emit1_${m}_${TAG}.log" emit_tier 1 "$m" || die "emit tier 1 $m failed"
    done
    sidecars_clean
    ok "tier 1 emitted; next: commit the sensitivity and emit logs, then $0 prereg"
}

stage_prereg() {
    local log="$OUT/w4v_prereg_${TAG}.log" files=() m arm f
    refuse "$log"
    use_env resnet_env17
    # every stage before it ready; --prereg re-checks that itself, and refuses unless the tools are committed and
    # clean and the models are MODEL_PIN's
    ready prereg
    tier_arms 1
    for m in "${MODELS[@]}"; do
        files+=("models/${m}_cut.onnx" "models/${m}_cut_xint8.onnx" "$CAP/${m}_yf.npz" "$CAP/${m}_mx.json")
        for arm in $TIER_ARMS; do f="$(arm_file "$m" "$arm")" || die "no file for $m $arm"; files+=("$f"); done
    done
    logged "$log" python tools/w4v_verdict.py --prereg --files "${files[@]}" || die "prereg refused, see $log"
    ok "commit $log before: $0 bias1"
}

# bias_arms_of <tier> -- the arms C5b audits in stage bias<tier> (tools/w4v_verdict.py bias_arms, v2.5 C).
bias_arms_of() {
    python -c 'import sys; sys.path.insert(0, "tools"); import w4v_verdict as v; print(" ".join(v.bias_arms(int(sys.argv[1]))))' "$1"
}

# bias_body <tier> <model> -- C5b (v2.5 C) on each of the stage's arms, against its pinned file: a W4V_CONTROL C5b
# <model> <arm> PASS|FAIL line each, keyed on the tool's own "C5B <model> <arm> PASS" line. A FAIL is a DEFECT and a
# STOP (B.3): no later arm runs and no W4V_DONE is printed.
bias_body() {
    local tier="$1" m="$2" arm arms
    python tools/w4v_verdict.py --hashes || return 1
    arms="$(bias_arms_of "$tier")" || return 1
    [ -n "$arms" ] || return 1
    for arm in $arms; do
        printf '\n### C5b %s %s\n' "$m" "$arm"
        marked C5b "$m $arm" "^C5B $m $arm PASS" python tools/w4v_bias_audit.py recompute "$m" "$arm" || return 1
    done
    echo "W4V_DONE bias${tier} $m"   # the stage's success line (tools/w4v_verdict.py ready)
}

# stage_bias <tier> -- bias1 after the prereg, bias2 after emit2 and its pins, each before its screen (v2.5 C).
stage_bias() {
    local tier="$1" m logs=()
    for m in "${MODELS[@]}"; do logs+=("$OUT/w4v_bias${tier}_${m}_${TAG}.log"); done
    refuse "${logs[@]}"
    use_env resnet_env17
    ready "bias${tier}"
    for m in "${MODELS[@]}"; do
        step "C5b bias${tier} $m: the independent bias check"
        logged "$OUT/w4v_bias${tier}_${m}_${TAG}.log" bias_body "$tier" "$m" \
            || die "C5b bias${tier} $m did not pass: a DEFECT is a STOP (v2.5 B.3); see its log and go to the gate"
    done
}

stage_bias1() {
    stage_bias 1
    ok "C5b bias1 pass; commit the bias1 logs, then $0 screen1 on the gate's go"
}

stage_screen1() {
    local m arm arms rows=() logs=()
    use_env resnet_env17
    ready screen1
    V24_DETS="$W/dets_v24_screen1_stop"
    [ -d "$V24_DETS" ] || warn "$V24_DETS is missing: every V24_DETS_SHA256 line will read none (report-only)"
    tier_arms 1
    arms="$TIER_ARMS"
    for m in "${MODELS[@]}"; do
        rows+=("$m b8 500")
        for arm in $arms; do rows+=("$m $arm 500"); done
    done
    refuse "$OUT/w4v_cut_tier1_${TAG}.log"
    run_rows screen "$OUT/load_w4v_screen1_${TAG}.log" "${rows[@]}"
    for m in "${MODELS[@]}"; do
        logs+=("$OUT/$(row_name screen "$m" b8 500)_${TAG}.log")
        for arm in $arms; do logs+=("$OUT/$(row_name screen "$m" "$arm" 500)_${TAG}.log"); done
    done
    logged "$OUT/w4v_cut_tier1_${TAG}.log" python tools/w4v_verdict.py --screen --tier 1 --logs "${logs[@]}" \
        || die "the tier-1 screen is incomplete, see $OUT/w4v_cut_tier1_${TAG}.log"
    ok "commit the screen and cut logs and any w4v_zerodet_ records; next: $0 emit2 on the gate's go"
}

stage_emit2() {
    local m arm f logs=() files=()
    for m in "${MODELS[@]}"; do logs+=("$OUT/w4v_emit2_${m}_${TAG}.log"); done
    refuse "${logs[@]}" "$OUT/w4v_pins_${TAG}.log"
    use_env resnet_env17
    ready emit2
    for m in "${MODELS[@]}"; do
        step "emit tier 2 $m"
        logged "$OUT/w4v_emit2_${m}_${TAG}.log" emit_tier 2 "$m" || die "emit tier 2 $m failed"
    done
    sidecars_clean
    tier_arms 2
    for m in "${MODELS[@]}"; do
        for arm in $TIER_ARMS; do f="$(arm_file "$m" "$arm")" || die "no file for $m $arm"; files+=("$f"); done
    done
    logged "$OUT/w4v_pins_${TAG}.log" python tools/w4v_verdict.py --pins --files "${files[@]}" \
        || die "pins refused, see $OUT/w4v_pins_${TAG}.log"
    ok "commit the emit2 logs and $OUT/w4v_pins_${TAG}.log before: $0 bias2"
}

stage_bias2() {
    stage_bias 2
    ok "C5b bias2 pass; commit the bias2 logs, then $0 screen2 on the gate's go"
}

stage_screen2() {
    local m arm arms rows=() logs=()
    use_env resnet_env17
    ready screen2
    tier_arms 2
    arms="$TIER_ARMS"
    for m in "${MODELS[@]}"; do
        for arm in $arms; do rows+=("$m $arm 500"); done
    done
    refuse "$OUT/w4v_cut_tier2_${TAG}.log"
    run_rows screen "$OUT/load_w4v_screen2_${TAG}.log" "${rows[@]}"
    # tier 2 is cut against tier 1's B8 rows
    logs+=($(ls "$OUT"/w4v_screen_yolov8[ns]_b8_n500_"${MACHINE}"_*.log))
    for m in "${MODELS[@]}"; do
        for arm in $arms; do logs+=("$OUT/$(row_name screen "$m" "$arm" 500)_${TAG}.log"); done
    done
    logged "$OUT/w4v_cut_tier2_${TAG}.log" python tools/w4v_verdict.py --screen --tier 2 --logs "${logs[@]}" \
        || die "the tier-2 screen is incomplete, see $OUT/w4v_cut_tier2_${TAG}.log"
    ok "commit the screen and cut logs and any w4v_zerodet_ records; next: $0 gate"
}

# gate_body -- SRC_TREE (the compiler C8 judges with, as HEAD records it; src/ clean), then C8 on every advancing
# form-b file: a W4V_GATE <sha> PASS|FAIL line each (a FAIL costs that arm its eligibility, whatever its form), then
# tools/w4v_verdict.py --c8 prints the hash block, the int32 coverage marker and what stopped each int32-bias arm
# that was not checked; W4V_DONE gate only after a C8 PASS.
gate_body() {
    local m arm f out rc sha adv results="$RAW/gate_results_$$.txt"
    python tools/w4v_verdict.py --src-tree || return 1
    adv="$(advanced)" || return 1
    : > "$results"
    while read -r m arm; do
        [ -n "$m" ] || continue
        f="$(arm_file "$m" "$arm")" || return 1
        case "$f" in
            *_ortonly.onnx)
                echo "[gate] $m $arm: $f holds a per-channel (form-a) conv: no engine lowering, no gate row"
                continue ;;
        esac
        printf '\n### engine gate %s %s\n' "$m" "$arm"
        out="$RAW/gate_$$.txt"
        rc=0
        { python tools/w4a8_engine_gate.py "$f" --coco 4 2>&1 | tee "$out"; rc=${PIPESTATUS[0]}; } || true
        sha="$(python -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
            "$f")" || return 1
        if [ "$rc" = 0 ] && grep -q "^\[gate\] PASS" "$out"; then
            echo "W4V_GATE $sha PASS" | tee -a "$results"
        else
            echo "W4V_GATE $sha FAIL" | tee -a "$results"
            echo "[gate] $m $arm: this form-b file fails C8, so the arm is not decision-eligible"
        fi
        rm -f "$out"
    done <<< "$adv"
    python tools/w4v_verdict.py --c8 --logs "$results" || return 1
    rm -f "$results"
    echo "W4V_DONE gate"   # the stage's success line (tools/w4v_verdict.py ready)
}

stage_gate() {
    local log="$OUT/w4v_gate_${TAG}.log"
    refuse "$log"
    use_env mlir-aie-iron
    ready gate
    # src/ is not in TOOLS_JSON: C8 judges with the compiler HEAD records, so src/ must be clean against HEAD
    python tools/w4v_verdict.py --src-tree \
        || die "src/ is not clean against HEAD, so no commit records the compiler C8 would judge with; nothing ran"
    PYTHONPATH="$(cygpath -w "$REPO_ROOT/src")"
    export PYTHONPATH
    # the repo-root ignite_xdna/ forwards to src/; what matters is where the compiler modules load from
    python -c 'import sys, pathlib; from ignite_xdna.compiler import graph_ir; p = pathlib.Path(graph_ir.__file__).resolve(); r = pathlib.Path(sys.argv[1]).resolve(); print(f"graph_ir from {p}"); sys.exit(0 if r in p.parents else f"not under {r}")' "$(cygpath -w "$REPO_ROOT/src")" \
        || die "ignite_xdna.compiler does not come from this checkout's src/"
    logged "$log" gate_body || die "the engine gate stage failed, see $log"
    ok "engine gate done; commit $log; next: $0 decisive on the gate's go"
}

stage_decisive() {
    local m arm adv rows=()
    use_env resnet_env17
    ready decisive
    for m in "${MODELS[@]}"; do rows+=("$m b8 5000"); done
    adv="$(advanced)" || die "cannot read the cut logs"
    while read -r m arm; do
        [ -n "$m" ] || continue
        case "$arm" in rtn_e1|rtn_e2|rtn_u) continue ;; esac   # their 5,000-image rows are C9's
        rows+=("$m $arm 5000")
    done <<< "$adv"
    run_rows decisive "$OUT/load_w4v_decisive_${TAG}.log" "${rows[@]}"
    ok "decisive rows done; commit the logs; next: $0 verdict on the gate's go"
}

stage_verdict() {
    local log="$OUT/w4v_verdict_${TAG}.log"
    refuse "$log"
    use_env resnet_env17
    # every earlier stage ready, the decisive rows included; --verdict re-checks that itself before it scores
    ready verdict
    # shellcheck disable=SC2046
    logged "$log" python tools/w4v_verdict.py --verdict \
        --logs $(ls "$OUT"/w4v_repro_yolov8*_n5000_"${MACHINE}"_*.log "$OUT"/w4v_decisive_yolov8*_n5000_"${MACHINE}"_*.log) \
        --controls $(ls "$OUT"/w4v_controls_*_"${MACHINE}"_*.log "$OUT"/w4v_parity_*_"${MACHINE}"_*.log \
                        "$OUT"/w4v_repro_c9_"${MACHINE}"_*.log "$OUT"/w4v_emit[12]_*_"${MACHINE}"_*.log \
                        "$OUT"/w4v_gate_"${MACHINE}"_*.log) \
        || die "verdict refused or incomplete, see $log"
}

"stage_$STAGE"
