#!/usr/bin/env bash
# LLM study, Phase 1: the CPU and DirectML yardsticks for M = 1 decode, on this APU. No NPU. The
# arms, the rules and the predictions live in tools/llm_decode_verdict.py and are pre-registered
# (committed) before any timing runs.
#
#   ./scripts/llm-study.sh controls   # MatMulNBits packing against float64, and the DirectML placement
#                                     # negative control; resnet_env17 (cpu, dml) and mlir-aie-iron (cpu)
#   ./scripts/llm-study.sh adapter    # which adapter DirectML's device_id 0 is: DXGI's list, and the
#                                     # process's GPU memory per adapter LUID with a session open
#   ./scripts/llm-study.sh build      # the int4 and dense weight files in scratch/llm/ (no timing)
#   ./scripts/llm-study.sh prereg     # the pre-registration log; commit it before anything below
#   ./scripts/llm-study.sh read       # read bandwidth: numpy sums, ONNX Runtime ReduceSum on cpu and dml
#   ./scripts/llm-study.sh gemv       # the 36 decisive int4 rows, then the dense and thread-sweep context
#   ./scripts/llm-study.sh dml16      # re-run of the 6 DirectML fp16 int4 rows only: sitting 1's streamed
#                                     # 0.90-0.98 GiB (a builder defect, commit 4620b53); run build first
#   ./scripts/llm-study.sh verdict    # tools/llm_decode_verdict.py over the read and gemv logs
#
# Stage 1 of the re-scoped study (NPU): a clean read-only DRAM bandwidth test, the decode reopen
# condition. tools/npu_read_bw_probe.py holds the design, the rules and the predictions.
#   ./scripts/llm-study.sh npuread-prereg    # its pre-registration log; commit it before the sitting
#   ./scripts/llm-study.sh npuread           # the sitting, recorded by tools/silicon_probe_record.py
#                                            # (xrt-smi idle before and after every child)
#   ./scripts/llm-study.sh npuread-verdict   # the mechanical verdict over the sitting's log
#   (build the xclbins first: bash scripts/research-iron.sh tools/npu_read_bw_probe.py build)
#
# Stage 2 (CPU, DirectML and NPU together): do the chips' DDR reads add? tools/concurrent_read_bw.py
# holds the design, the rules and the predictions. The three readers overlap by design.
#   ./scripts/llm-study.sh concurrent-prereg    # its pre-registration log; commit it before the sitting
#   ./scripts/llm-study.sh concurrent           # the sitting, recorded by tools/silicon_probe_record.py
#   ./scripts/llm-study.sh concurrent-verdict   # the mechanical verdict over the sitting's log
#
# The R5 follow-up (the user: "measure the sync first"): what a DirectML + NPU split pays to join
# the chips on every GEMV. tools/split_sync_cost.py holds the design, the kill line and predictions.
#   ./scripts/llm-study.sh sync-prereg    # its pre-registration log; commit it before the sitting
#   ./scripts/llm-study.sh sync           # the sitting (NPU and iGPU), recorded by silicon_probe_record.py
#   ./scripts/llm-study.sh sync-verdict   # the mechanical verdict
#   (build first: bash scripts/research-iron.sh tools/split_sync_cost.py build)
#
# Stage 3: prefill GEMM at Llama-2-7B's shapes on the CPU, DirectML and the NPU.
# tools/llm_prefill_bench.py holds the arms, the tiles, the rule and the predictions.
#   ./scripts/llm-study.sh prefill-build     # the fixed inputs in scratch/llm/prefill/ and the NPU
#                                            # configs in build/llm_prefill/ (compile only, no chip)
#   ./scripts/llm-study.sh prefill-prereg    # its pre-registration log; commit it before the sitting
#   ./scripts/llm-study.sh prefill           # the sitting: NPU, then CPU, then DirectML, one after
#                                            # another, each recorded by silicon_probe_record.py
#   ./scripts/llm-study.sh prefill-verdict   # the mechanical verdict over the three logs
#
# The noise study (the user: "study the noise"): stage 3's two unattributed noise sources, the CPU's
# 8-thread bimodality and DirectML's level shifts. tools/measure_noise.py holds the arms, the rules
# and the predictions. A finding about measuring on this APU; it re-scores nothing.
#   ./scripts/llm-study.sh noise-prereg    # its pre-registration log; commit it before the sitting
#   ./scripts/llm-study.sh noise           # the sitting: CPU, then DirectML, each recorded by
#                                          # silicon_probe_record.py
#   ./scripts/llm-study.sh noise-verdict   # the mechanical verdict over the two logs
#
# Stage 3b (the user: "Re-run prefill, pinned"): stage 3 again with the CPU's threads pinned and
# DirectML timed over fresh sessions; nothing else changed. tools/llm_prefill_3b.py holds the
# changes, the repeat rule and the predictions; the NPU runs stage 3's own code.
#   ./scripts/llm-study.sh prefill3b-prereg    # its pre-registration log (re-verifies the NPU pins);
#                                              # commit it before the sitting
#   ./scripts/llm-study.sh prefill3b           # the sitting: NPU, then CPU, then DirectML
#   ./scripts/llm-study.sh prefill3b-verdict   # the mechanical verdict over 3b's three logs (the
#                                              # newest sitting date, so a sitting past midnight works)
#
# Options: --machine TAG names the logs (default: desktop2 on DESKTOP-CBL5NUA, else required).
# --tag TAG adds _TAG to the verdict log's name, for a second verdict on the same day.
#
# read and gemv are timing sittings: announce them to the other sessions first, and never run them
# beside another session's NPU, CPU or GPU measurement. Each group refuses to start while another
# heavy job holds the CPU, and its witness records the host load, xrt-smi (the NPU must be idle) and,
# for DirectML, the GPU engines' utilization.
#
# Logs go to results/llm/ (UTF-8, profile path scrubbed). The script refuses to replace a log;
# delete by hand to re-run. Weight files stay in the git-ignored scratch/llm/.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

STAGE="" MACHINE="" TAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        controls|adapter|build|prereg|read|gemv|dml16|verdict) STAGE="$1" ;;
        npuread-prereg|npuread|npuread-verdict) STAGE="${1//-/_}" ;;
        concurrent-prereg|concurrent|concurrent-verdict) STAGE="${1//-/_}" ;;
        sync-prereg|sync|sync-verdict) STAGE="${1//-/_}" ;;
        prefill-build|prefill-prereg|prefill|prefill-verdict) STAGE="${1//-/_}" ;;
        noise-prereg|noise|noise-verdict) STAGE="${1//-/_}" ;;
        prefill3b-prereg|prefill3b|prefill3b-verdict) STAGE="${1//-/_}" ;;
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
B=tools/llm_gemv_bench.py
SHAPES=("4096 4096" "4096 11008" "11008 4096")
BLOCKS=(32 128)
mkdir -p "$OUT" "$RAW"

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

need_prereg() {
    ls "$OUT"/llm_decode_prereg_*.log >/dev/null 2>&1 || die "no pre-registration log: run $0 prereg and commit it"
}

# witness <file> <label> [gpu] -- host load (refuse on a heavy peer), xrt-smi, and GPU engines.
witness() {
    local w="$RAW/$(basename "$1")" label="$2"
    printf '\n### %s  %s\n' "$label" "$(date -Iseconds)" >> "$w"
    check_host_load refuse "$w"
    printf 'XRT_SMI_BEGIN\n' >> "$w"
    /c/Windows/System32/AMD/xrt-smi.exe examine -r aie-partitions >> "$w" 2>&1 || printf 'xrt-smi failed\n' >> "$w"
    printf 'XRT_SMI_END\n' >> "$w"
    if [ "${3:-}" = gpu ]; then
        powershell -NoProfile -Command "\$s = (Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | Where-Object CookedValue -gt 1; 'GPU_ENGINES_OVER_1PCT ' + @(\$s).Count; \$s | ForEach-Object { 'GPU_ENGINE ' + \$_.InstanceName + ' ' + [math]::Round(\$_.CookedValue, 1) }" >> "$w" 2>&1
    fi
    scrub "$w" "$1"
}

# row <desc> <cmd...> -- one row; a failure is written into the log and the group goes on.
row() {
    local desc="$1"; shift
    printf '\n### %s\n' "$desc"
    "$@" || echo "ROW_FAILED $desc"
}

stage_controls() {
    local l1="$OUT/llm_controls_resnet_env17_${MACHINE}_${DATE}.log" l2="$OUT/llm_controls_mlir-aie-iron_${MACHINE}_${DATE}.log"
    refuse "$l1" "$l2"
    use_env resnet_env17
    logged "$l1" bash -c "python $B controls --ep cpu && python $B controls --ep dml" || die "controls failed (resnet_env17)"
    use_env mlir-aie-iron
    logged "$l2" python $B controls --ep cpu || die "controls failed (mlir-aie-iron)"
    ok "controls pass; next: $0 build"
}

stage_adapter() {
    local log="$OUT/llm_dml_adapter_${MACHINE}_${DATE}.log"
    refuse "$log"
    use_env resnet_env17
    logged "$log" python $B adapter || die "the DirectML adapter is not the Radeon 780M, see $log"
    ok "DirectML device_id 0 is the Radeon 780M"
}

stage_build() {
    local s b
    use_env resnet_env17
    for s in "${SHAPES[@]}"; do
        for b in "${BLOCKS[@]}"; do
            # shellcheck disable=SC2086
            python $B build --shape $s --block "$b" || die "build $s $b failed"
        done
    done
    ok "weights built in scratch/llm/; next: $0 prereg"
}

stage_prereg() {
    local log="$OUT/llm_decode_prereg_${MACHINE}_${DATE}.log"
    refuse "$log"
    use_env resnet_env17
    logged "$log" python tools/llm_decode_verdict.py --prereg || die "prereg failed"
    ok "commit $log (with the tools and the controls) before: $0 read"
}

read_group() {
    python tools/cpu_mem_bw.py --gib 4 --threads 1 2 4 8 16 --reps 5
}

stage_read() {
    need_prereg
    local w="$OUT/load_llm_read_${MACHINE}_${DATE}.log"
    local l1="$OUT/llm_read_cpu_numpy_${MACHINE}_${DATE}.log" l2="$OUT/llm_read_cpu_ort_fp32_${MACHINE}_${DATE}.log"
    local l3="$OUT/llm_read_dml_fp16_${MACHINE}_${DATE}.log" l4="$OUT/llm_read_dml_fp32_${MACHINE}_${DATE}.log"
    refuse "$w" "$l1" "$l2" "$l3" "$l4"
    use_env resnet_env17
    witness "$w" "numpy"
    logged "$l1" read_group || die "numpy read failed"
    witness "$w" "ort cpu fp32"
    logged "$l2" python $B readbw --ep cpu --dtype fp32 --threads 8 || die "cpu readbw failed"
    witness "$w" "dml fp16" gpu
    logged "$l3" python $B readbw --ep dml --dtype fp16 || die "dml fp16 readbw failed"
    witness "$w" "dml fp32" gpu
    logged "$l4" python $B readbw --ep dml --dtype fp32 || die "dml fp32 readbw failed"
    witness "$w" "after read"
    ok "read done; next: $0 gemv"
}

group_cpu() {   # the four int4 cpu configurations of one ONNX Runtime build (accuracy_level 0 and 4)
    local s b a
    for a in 0 4; do
        for b in "${BLOCKS[@]}"; do
            for s in "${SHAPES[@]}"; do
                # shellcheck disable=SC2086
                row "cpu nbits acc $a block $b $s" python $B gemv --ep cpu --arm nbits --t1 fp32 --acc "$a" --block "$b" --threads 8 --shape $s
            done
        done
    done
}

group_dml() {   # group_dml [t1...]: the DirectML int4 rows (default fp32 then fp16)
    local s b t ts=("$@")
    [ ${#ts[@]} -gt 0 ] || ts=(fp32 fp16)
    for t in "${ts[@]}"; do
        for b in "${BLOCKS[@]}"; do
            for s in "${SHAPES[@]}"; do
                # shellcheck disable=SC2086
                row "dml nbits $t block $b $s" python $B gemv --ep dml --arm nbits --t1 "$t" --block "$b" --shape $s
            done
        done
    done
}

group_dense() {
    local s
    for s in "${SHAPES[@]}"; do
        # shellcheck disable=SC2086
        row "cpu dense fp32 $s" python $B gemv --ep cpu --arm dense --t1 fp32 --threads 8 --shape $s
    done
    for s in "${SHAPES[@]}"; do
        # shellcheck disable=SC2086
        row "dml dense fp16 $s" python $B gemv --ep dml --arm dense --t1 fp16 --shape $s
    done
}

group_sweep() {
    local th
    for th in 1 2 4 16; do
        row "cpu nbits acc 4 block 32 4096 11008, $th threads" python $B gemv --ep cpu --arm nbits --t1 fp32 --acc 4 \
            --block 32 --threads "$th" --shape 4096 11008
    done
}

stage_gemv() {
    need_prereg
    local w="$OUT/load_llm_gemv_${MACHINE}_${DATE}.log"
    local l1="$OUT/llm_gemv_cpu_ort123_${MACHINE}_${DATE}.log" l2="$OUT/llm_gemv_cpu_ort130_${MACHINE}_${DATE}.log"
    local l3="$OUT/llm_gemv_dml_ort123_${MACHINE}_${DATE}.log" l4="$OUT/llm_gemv_dense_ort123_${MACHINE}_${DATE}.log"
    local l5="$OUT/llm_gemv_sweep_ort123_${MACHINE}_${DATE}.log"
    refuse "$w" "$l1" "$l2" "$l3" "$l4" "$l5"
    use_env resnet_env17
    witness "$w" "cpu ort 1.23"
    logged "$l1" group_cpu
    witness "$w" "dml ort 1.23" gpu
    logged "$l3" group_dml
    witness "$w" "dense ort 1.23" gpu
    logged "$l4" group_dense
    witness "$w" "sweep ort 1.23"
    logged "$l5" group_sweep
    use_env mlir-aie-iron
    witness "$w" "cpu ort 1.30"
    logged "$l2" group_cpu
    witness "$w" "after gemv"
    grep -h "^ROW_FAILED" "$l1" "$l2" "$l3" "$l4" "$l5" && warn "rows failed, see above" || true
    ok "gemv done; next: $0 verdict"
}

stage_dml16() {
    need_prereg
    local w="$OUT/load_llm_gemv_dml16_rerun_${MACHINE}_${DATE}.log" l="$OUT/llm_gemv_dml16_rerun_ort123_${MACHINE}_${DATE}.log"
    refuse "$w" "$l"
    use_env resnet_env17
    witness "$w" "dml fp16 re-run" gpu
    logged "$l" group_dml fp16
    witness "$w" "after dml fp16 re-run"
    grep -h "^ROW_FAILED" "$l" && warn "rows failed, see above" || true
    ok "re-run done; next: $0 verdict --tag rerun"
}

stage_verdict() {
    local log="$OUT/llm_decode_verdict${TAG}_${MACHINE}_${DATE}.log"
    refuse "$log"
    use_env resnet_env17
    # shellcheck disable=SC2046
    logged "$log" python tools/llm_decode_verdict.py --logs $(ls "$OUT"/llm_read_*_"${MACHINE}"_*.log \
        "$OUT"/llm_gemv_*_"${MACHINE}"_*.log "$OUT"/load_llm_*_"${MACHINE}"_*.log) || die "verdict incomplete, see $log"
}

NPUREAD=tools/npu_read_bw_probe.py

stage_npuread_prereg() {
    local log="$OUT/npu_read_bw_prereg_${MACHINE}_${DATE}.log"
    refuse "$log"
    use_env mlir-aie-iron
    logged "$log" python $NPUREAD prereg || die "prereg failed"
    ok "commit $log (with $NPUREAD) before: $0 npuread"
}

stage_npuread() {
    ls "$OUT"/npu_read_bw_prereg_*.log >/dev/null 2>&1 || die "no pre-registration log: run $0 npuread-prereg and commit it"
    [ -f build/npu_read_bw_probe/c4x2_m1024/insts.bin ] || die "no xclbins: bash scripts/research-iron.sh $NPUREAD build"
    local log="$OUT/npu_read_bw_suite${TAG}_${MACHINE}_${DATE}.log"
    refuse "$log"
    use_env mlir-aie-iron
    # the recorder writes UTC, machine, command and commit, witnesses xrt-smi idle before and after,
    # scrubs the profile path, and never replaces a log
    python tools/silicon_probe_record.py --log "$log" --device --seconds 1800 -- \
        bash scripts/research-iron.sh $NPUREAD suite || die "the sitting failed, see $log"
    ok "sitting done; next: $0 npuread-verdict"
}

stage_npuread_verdict() {
    local suite="$OUT/npu_read_bw_suite${TAG}_${MACHINE}_${DATE}.log" log="$OUT/npu_read_bw_verdict${TAG}_${MACHINE}_${DATE}.log"
    [ -f "$suite" ] || die "no sitting log $suite"
    refuse "$log"
    use_env mlir-aie-iron
    logged "$log" python $NPUREAD verdict "$suite" || die "verdict incomplete, see $log"
}

CONC=tools/concurrent_read_bw.py

stage_concurrent_prereg() {
    # --tag rerun writes the re-run rule (written after sitting 1) instead of the original prereg
    local log="$OUT/concurrent_read_prereg${TAG}_${MACHINE}_${DATE}.log" extra=()
    [ "$TAG" = "_rerun" ] && extra=(--rerun)
    refuse "$log"
    use_env resnet_env17
    logged "$log" python $CONC prereg "${extra[@]}" || die "prereg failed"
    ok "commit $log (with $CONC) before: $0 concurrent"
}

stage_concurrent() {
    ls "$OUT"/concurrent_read_prereg_*.log >/dev/null 2>&1 || die "no pre-registration log: run $0 concurrent-prereg and commit it"
    [ -f build/npu_read_bw_probe/c4x2_m1024/insts.bin ] || die "no NPU reader: bash scripts/research-iron.sh $NPUREAD build"
    [ -f scratch/llm/readbw_fp32_65536x4096/model.onnx ] || die "no readbw model: $0 read builds it (Phase 1)"
    local log="$OUT/concurrent_read_suite${TAG}_${MACHINE}_${DATE}.log"
    refuse "$log"
    # the coordinator runs in resnet_env17: its interpreter starts the CPU and DirectML readers,
    # and the NPU reader starts through scripts/research-iron.sh
    use_env resnet_env17
    python tools/silicon_probe_record.py --log "$log" --device --seconds 1800 -- \
        python $CONC suite || die "the sitting failed, see $log"
    ok "sitting done; next: $0 concurrent-verdict"
}

stage_concurrent_verdict() {
    local suite="$OUT/concurrent_read_suite${TAG}_${MACHINE}_${DATE}.log" log="$OUT/concurrent_read_verdict${TAG}_${MACHINE}_${DATE}.log"
    [ -f "$suite" ] || die "no sitting log $suite"
    refuse "$log"
    use_env resnet_env17
    logged "$log" python $CONC verdict "$suite" || die "verdict incomplete, see $log"
}

SYNC=tools/split_sync_cost.py

stage_sync_prereg() {
    local log="$OUT/split_sync_prereg_${MACHINE}_${DATE}.log"
    refuse "$log"
    use_env resnet_env17
    logged "$log" python $SYNC prereg || die "prereg failed"
    ok "commit $log (with $SYNC) before: $0 sync"
}

stage_sync() {
    ls "$OUT"/split_sync_prereg_*.log >/dev/null 2>&1 || die "no pre-registration log: run $0 sync-prereg and commit it"
    [ -f build/split_sync_probe/passthrough_w8192_c1024/insts.bin ] || die "not built: bash scripts/research-iron.sh $SYNC build"
    local log="$OUT/split_sync_suite${TAG}_${MACHINE}_${DATE}.log"
    refuse "$log"
    # the DirectML worker runs in resnet_env17; its interpreter goes through the environment so the
    # logged command stays relative, and the log names only the environment
    use_env resnet_env17
    export SPLIT_WORKER_PYTHON="$(cygpath -w "$CONDA_PREFIX")\\python.exe" SPLIT_WORKER_ENV=resnet_env17
    python tools/silicon_probe_record.py --log "$log" --device --seconds 900 -- \
        bash scripts/research-iron.sh $SYNC suite || die "the sitting failed, see $log"
    ok "sitting done; next: $0 sync-verdict"
}

stage_sync_verdict() {
    local suite="$OUT/split_sync_suite${TAG}_${MACHINE}_${DATE}.log" log="$OUT/split_sync_verdict${TAG}_${MACHINE}_${DATE}.log"
    [ -f "$suite" ] || die "no sitting log $suite"
    refuse "$log"
    use_env resnet_env17
    logged "$log" python $SYNC verdict "$suite" || die "verdict incomplete, see $log"
}

PRE=tools/llm_prefill_bench.py

stage_prefill_build() {
    use_env resnet_env17
    python $PRE inputs || die "the inputs failed"
    bash scripts/research-iron.sh $PRE build || die "the NPU build failed"
    ok "inputs and NPU configs built, no chip; next: $0 prefill-prereg"
}

stage_prefill_prereg() {
    # --tag rerun writes the re-run rule (written after sitting 1) instead of the original prereg
    local log="$OUT/llm_prefill_prereg${TAG}_${MACHINE}_${DATE}.log" extra=()
    [ "$TAG" = "_rerun" ] && extra=(--rerun)
    refuse "$log"
    use_env resnet_env17
    logged "$log" python $PRE prereg "${extra[@]}" || die "prereg failed"
    ok "commit $log (with $PRE) before: $0 prefill${TAG:+ --tag ${TAG#_}}"
}

stage_prefill() {
    ls "$OUT"/llm_prefill_prereg_*.log >/dev/null 2>&1 || die "no pre-registration log: run $0 prefill-prereg and commit it"
    if [ "$TAG" = "_rerun" ]; then
        ls "$OUT"/llm_prefill_prereg_rerun_*.log >/dev/null 2>&1 || die "no re-run rule: run $0 prefill-prereg --tag rerun and commit it"
    fi
    local npu="$OUT/llm_prefill_npu${TAG}_${MACHINE}_${DATE}.log" cpu="$OUT/llm_prefill_cpu${TAG}_${MACHINE}_${DATE}.log"
    local dml="$OUT/llm_prefill_dml${TAG}_${MACHINE}_${DATE}.log"
    refuse "$npu" "$cpu" "$dml"
    use_env resnet_env17
    # one chip at a time; the NPU goes first, so a failed smoke stops the sitting before the others run
    python tools/silicon_probe_record.py --log "$npu" --device --seconds 1800 -- \
        bash scripts/research-iron.sh $PRE npu || die "the NPU rows failed, see $npu"
    python tools/silicon_probe_record.py --log "$cpu" --device --seconds 1800 -- \
        python $PRE ort --ep cpu || die "the CPU rows failed, see $cpu"
    python tools/silicon_probe_record.py --log "$dml" --device --seconds 1800 -- \
        python $PRE ort --ep dml || die "the DirectML rows failed, see $dml"
    ok "sitting done; next: $0 prefill-verdict"
}

stage_prefill_verdict() {
    local npu="$OUT/llm_prefill_npu${TAG}_${MACHINE}_${DATE}.log" cpu="$OUT/llm_prefill_cpu${TAG}_${MACHINE}_${DATE}.log"
    local dml="$OUT/llm_prefill_dml${TAG}_${MACHINE}_${DATE}.log" log="$OUT/llm_prefill_verdict${TAG}_${MACHINE}_${DATE}.log"
    local f
    for f in "$npu" "$cpu" "$dml"; do [ -f "$f" ] || die "no sitting log $f"; done
    refuse "$log"
    use_env resnet_env17
    logged "$log" python $PRE verdict "$npu" "$cpu" "$dml" || die "verdict incomplete, see $log"
}

NOISE=tools/measure_noise.py

stage_noise_prereg() {
    local log="$OUT/llm_noise_prereg_${MACHINE}_${DATE}.log"
    refuse "$log"
    use_env resnet_env17
    logged "$log" python $NOISE prereg || die "prereg failed"
    ok "commit $log (with $NOISE) before: $0 noise"
}

stage_noise() {
    ls "$OUT"/llm_noise_prereg_*.log >/dev/null 2>&1 || die "no pre-registration log: run $0 noise-prereg and commit it"
    local cpu="$OUT/llm_noise_cpu_${MACHINE}_${DATE}.log" dml="$OUT/llm_noise_dml_${MACHINE}_${DATE}.log"
    refuse "$cpu" "$dml"
    use_env resnet_env17
    python tools/silicon_probe_record.py --log "$cpu" --device --seconds 1200 -- \
        python $NOISE cpu || die "the CPU arms failed, see $cpu"
    python tools/silicon_probe_record.py --log "$dml" --device --seconds 1200 -- \
        python $NOISE dml || die "the DirectML sessions failed, see $dml"
    ok "sitting done; next: $0 noise-verdict"
}

stage_noise_verdict() {
    local cpu="$OUT/llm_noise_cpu_${MACHINE}_${DATE}.log" dml="$OUT/llm_noise_dml_${MACHINE}_${DATE}.log"
    local log="$OUT/llm_noise_verdict_${MACHINE}_${DATE}.log" f
    for f in "$cpu" "$dml"; do [ -f "$f" ] || die "no sitting log $f"; done
    refuse "$log"
    use_env resnet_env17
    logged "$log" python $NOISE verdict "$cpu" "$dml" || die "verdict incomplete, see $log"
}

P3B=tools/llm_prefill_3b.py

stage_prefill3b_prereg() {
    local log="$OUT/llm_prefill3b_prereg_${MACHINE}_${DATE}.log"
    refuse "$log"
    use_env resnet_env17
    logged "$log" python $P3B prereg || die "prereg failed or the NPU pins differ from stage 3's, see $log"
    ok "commit $log (with $P3B) before: $0 prefill3b"
}

stage_prefill3b() {
    ls "$OUT"/llm_prefill3b_prereg_*.log >/dev/null 2>&1 || die "no 3b pre-registration log: run $0 prefill3b-prereg and commit it"
    local npu="$OUT/llm_prefill3b_npu_${MACHINE}_${DATE}.log" cpu="$OUT/llm_prefill3b_cpu_${MACHINE}_${DATE}.log"
    local dml="$OUT/llm_prefill3b_dml_${MACHINE}_${DATE}.log"
    refuse "$npu" "$cpu" "$dml"
    use_env resnet_env17
    # one chip at a time, as in stage 3; the NPU rows are stage 3's own code, unchanged
    python tools/silicon_probe_record.py --log "$npu" --device --seconds 1800 -- \
        bash scripts/research-iron.sh $PRE npu || die "the NPU rows failed, see $npu"
    python tools/silicon_probe_record.py --log "$cpu" --device --seconds 1800 -- \
        python $P3B ort --ep cpu || die "the CPU rows failed, see $cpu"
    python tools/silicon_probe_record.py --log "$dml" --device --seconds 2400 -- \
        python $P3B ort --ep dml || die "the DirectML rows failed, see $dml"
    ok "sitting done; next: $0 prefill3b-verdict"
}

stage_prefill3b_verdict() {
    local d
    d="$(ls "$OUT"/llm_prefill3b_npu_"${MACHINE}"_*.log 2>/dev/null | sed -n 's/.*_\([0-9]\{8\}\)\.log$/\1/p' | sort | tail -1)"
    [ -n "$d" ] || die "no 3b sitting logs"
    local npu="$OUT/llm_prefill3b_npu_${MACHINE}_${d}.log" cpu="$OUT/llm_prefill3b_cpu_${MACHINE}_${d}.log"
    local dml="$OUT/llm_prefill3b_dml_${MACHINE}_${d}.log" log="$OUT/llm_prefill3b_verdict_${MACHINE}_${d}.log" f
    for f in "$npu" "$cpu" "$dml"; do [ -f "$f" ] || die "no sitting log $f"; done
    refuse "$log"
    use_env resnet_env17
    logged "$log" python $P3B verdict "$npu" "$cpu" "$dml" || die "verdict incomplete, see $log"
}

"stage_$STAGE"
