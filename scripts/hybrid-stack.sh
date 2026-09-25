#!/usr/bin/env bash
# Hybrid Gemma 3 4B stack on the 8700G (NPU prompt, 780M generation). S1: the NPU numerics gate, CPU
# only. tools/hybrid_s1.py holds the plan (its PREREG text), the rules and the predictions; the
# pre-registration log is committed before anything heavy runs.
#
#   ./scripts/hybrid-stack.sh s1-selftest   # synthetic checks: no model, no chip
#   ./scripts/hybrid-stack.sh s1-prereg     # the plan, the text pins, PROTOCOL_JSON, the frozen hashes (commit it)
#   ./scripts/hybrid-stack.sh s1-build      # the five models from (c)'s C0-H16, C3, one profiled pass each
#   ./scripts/hybrid-stack.sh s1-check      # C1, C2 (the form), C4, C5, C6
#   ./scripts/hybrid-stack.sh s1-states     # the five models' hidden states, then band C (commit with the two above)
#   ./scripts/hybrid-stack.sh s1-verdict    # the metrics and the frozen verdict
#
# S1b (tools/hybrid_s1b.py): N2 serves the prompt's KV, and R0's continuation decides. Same pattern as S1.
#   ./scripts/hybrid-stack.sh s1b-selftest  # synthetic checks: no model, no chip
#   ./scripts/hybrid-stack.sh s1b-prereg    # the plan, the 84 windows, PROTOCOL_JSON, the frozen hashes (commit it)
#   ./scripts/hybrid-stack.sh s1b-check     # the model pins, C5' and the R0 control (RAM-heavy)
#   ./scripts/hybrid-stack.sh s1b-states    # the arms' prompts and R0's continuations, in batches (RAM-heavy)
#   ./scripts/hybrid-stack.sh s1b-verdict   # the metrics and the frozen verdict
#
# S0 (tools/hybrid_s0.py), the light steps: a re-read, the publishers' asset list, verified downloads.
#   ./scripts/hybrid-stack.sh s0-baseline
#   ./scripts/hybrid-stack.sh s0-assets
#   ./scripts/hybrid-stack.sh s0-fetch COMPONENT   # its log is hybrid_s0_fetch-COMPONENT_...
#
# Before s1-build, copy (c)'s C0-H16 model.onnx and model.onnx.data to scratch/llm/hybrid_s1/models/ as
# c0h16.onnx and base.onnx.data, and text_only's config.json and tokenizer files to scratch/llm/gemma/text_only/;
# s1-build checks every copy against (c)'s pins. build, check and states are RAM-heavy: announce them
# first. Each heavy child refuses to start below 15 GB available.
#
# Options: --machine TAG names the logs (default: desktop2 on DESKTOP-CBL5NUA, else required).
# --tag TAG adds _TAG after the date, for a re-run on the same day (a memory refusal or a crash); the
# tool reads the latest log of each kind, and the suffix sorts after the untagged name.
#
# Logs go to results/llm/ (UTF-8, profile path scrubbed). The script refuses to replace a log; delete by
# hand to re-run. Model and state files stay in the git-ignored scratch/llm/hybrid_s1/.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

STAGE="" MACHINE="" TAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        s1-selftest|s1-prereg|s1-build|s1-check|s1-states|s1-verdict) STAGE="${1//-/_}" ;;
        s1b-selftest|s1b-prereg|s1b-check|s1b-states|s1b-verdict) STAGE="${1//-/_}" ;;
        s0-baseline|s0-assets) STAGE="${1//-/_}" ;;
        s0-fetch)  STAGE=s0_fetch; COMP="${2:-}"; shift ;;
        --machine) MACHINE="$2"; shift ;;
        --tag)     TAG="_$2"; shift ;;
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
OUT=results/llm
RAW=scratch/llm/raw
T=tools/hybrid_s1.py
mkdir -p "$OUT" "$RAW"
export HF_HUB_OFFLINE=1          # every file S1 reads is already in the local cache and pinned by hash

refuse() {
    local p
    for p in "$@"; do
        [ ! -e "$p" ] || die "$p exists; this script never replaces a log. Delete it by hand to re-run."
    done
}

# scrub <raw> <log> -- copy with the local profile path replaced by C:\Users\<user>.
scrub() {
    python -c 'import pathlib,re,sys; s=pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"); s=re.sub(r"(?i)([A-Z]:[\\/]Users[\\/])[^\\/\s\"<>]+",r"\1<user>",s); s=re.sub(r"(?i)(/[A-Z]/Users/)[^/\s\"<>]+",r"\1<user>",s); pathlib.Path(sys.argv[2]).write_text(s,encoding="utf-8")' "$1" "$2"
}

# logged <log> <cmd...> -- run through run_logged into scratch, then scrub into results/llm/.
logged() {
    local log="$1" rc=0; shift
    local raw="$RAW/$(basename "$log")"
    rm -f "$raw"
    run_logged "$raw" "$@" || rc=$?
    scrub "$raw" "$log"
    return $rc
}

need() {
    ls "$OUT"/hybrid_s1_"$1"_*.log >/dev/null 2>&1 || die "no S1 $1 log: run $0 s1-$1 first"
}

stage() {
    local name="$1" log="$OUT/hybrid_s1_$1_${MACHINE}_${DATE}${TAG}.log"
    refuse "$log"
    use_env resnet_env17
    logged "$log" python "$T" "$name" || die "S1 $name did not finish OK, see $log"
    ok "S1 $name done: $log"
}

stage_s1_selftest() { stage selftest; }
stage_s1_prereg()   { stage prereg; }
stage_s1_build()    { need prereg; stage build; }
stage_s1_check()    { need build; stage check; }
stage_s1_states()   { need check; stage states; }
stage_s1_verdict()  { need states; stage verdict; }

# stage_for <tool> <log kind> <mode> [args...] -- stage, for S1b's and S0's tools and log names.
stage_for() {
    local tool="$1" kind="$2" mode="$3"; shift 3
    local log="$OUT/${kind}_${MACHINE}_${DATE}${TAG}.log"
    refuse "$log"
    use_env resnet_env17
    logged "$log" python "$tool" "$mode" "$@" || die "$kind did not finish OK, see $log"
    ok "$kind done: $log"
}

need1b() {
    ls "$OUT"/hybrid_s1b_"$1"_*.log >/dev/null 2>&1 || die "no S1b $1 log: run $0 s1b-$1 first"
}

T1B=tools/hybrid_s1b.py
stage_s1b_selftest() { stage_for "$T1B" hybrid_s1b_selftest selftest; }
stage_s1b_prereg()   { stage_for "$T1B" hybrid_s1b_prereg prereg; }
stage_s1b_check()    { need1b prereg; stage_for "$T1B" hybrid_s1b_check check; }
stage_s1b_states()   { need1b check; stage_for "$T1B" hybrid_s1b_states states; }
stage_s1b_verdict()  { need1b states; stage_for "$T1B" hybrid_s1b_verdict verdict; }

T0=tools/hybrid_s0.py
stage_s0_baseline() { stage_for "$T0" hybrid_s0_baseline baseline; }
stage_s0_assets()   { stage_for "$T0" hybrid_s0_assets assets; }
stage_s0_fetch() {
    [[ "${COMP:-}" =~ ^[a-z0-9][a-z0-9.-]*$ ]] || die "s0-fetch needs a COMPONENT name"
    stage_for "$T0" "hybrid_s0_fetch-$COMP" fetch "$COMP"
}

"stage_$STAGE"
