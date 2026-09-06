#!/usr/bin/env bash
# Shared setup for the pipeline scripts. Source this, don't run it.
#
# Runs under Git Bash on Windows. NOT WSL: the XDNA1 NPU has no Linux userspace,
# so a WSL run would silently be CPU-only.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Windows-style path: this is read by Python's os.path.join, not by bash.
: "${RYZEN_AI_PATH:=C:\\Program Files\\RyzenAI\\1.7.1}"

BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
BLUE=$'\033[34m'; DIM=$'\033[2m'; OFF=$'\033[0m'

step() { printf '\n%s==> %s%s\n' "$BOLD$BLUE" "$*" "$OFF"; }
info() { printf '%s    %s%s\n' "$DIM" "$*" "$OFF"; }
warn() { printf '%s!!  %s%s\n' "$YELLOW" "$*" "$OFF" >&2; }
ok()   { printf '%s OK %s%s\n' "$GREEN" "$*" "$OFF"; }
die()  { printf '\n%sERROR: %s%s\n' "$RED$BOLD" "$*" "$OFF" >&2; exit 1; }

# Quark needs ninja.exe from the env's Scripts\, which only reaches PATH on
# activation. Invoking envs\<name>\python.exe by path breaks `import quark`
# with "RuntimeError: Ninja is required to load C++ extensions".
_conda_ready=0
conda_init() {
    [ "$_conda_ready" = 1 ] && return 0
    local base
    base="$(conda info --base 2>/dev/null || true)"
    for c in "$base/etc/profile.d/conda.sh" \
             "$HOME/miniforge3/etc/profile.d/conda.sh" \
             "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "$c" ]; then
            # conda.sh trips over `set -u`
            set +u; . "$c"; set -u
            _conda_ready=1
            return 0
        fi
    done
    die "could not find conda.sh. Is miniforge installed?"
}

# use_env <name>  -- activate a conda env and confirm it took.
use_env() {
    conda_init
    set +u; conda activate "$1"; set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "$1" ] || die "failed to activate $1"
    info "env: $1  ($(python -c 'import sys;print(sys.version.split()[0])'))"
}

# npu_env -- the 1.7.1 inference env, with the vars every NPU run needs.
# 1.8.0 ships no Phoenix xclbin, and existing shells often still point at it.
npu_env() {
    use_env resnet_env17
    export RYZEN_AI_INSTALLATION_PATH="$RYZEN_AI_PATH"
    export XLNX_ONNX_EP_REPORT_FILE="vitisai_ep_report.json"
    info "RYZEN_AI_INSTALLATION_PATH = $RYZEN_AI_INSTALLATION_PATH"
}

# usage "${BASH_SOURCE[0]}" -- print the script's leading comment block as help,
# so it can never drift out of sync with a hardcoded line range.
usage() {
    awk 'NR==1 && /^#!/ {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$1"
}

need_file() { [ -f "$1" ] || die "missing $1${2:+ -- $2}"; }
need_dir()  { [ -d "$1" ] || die "missing directory $1${2:+ -- $2}"; }

# run_logged <logfile> <cmd...> -- tee to results/, keep the exit status.
# Logging through bash gives UTF-8; PowerShell's `*>` writes UTF-16, which
# makes every later grep silently match nothing.
run_logged() {
    local log="$1"; shift
    mkdir -p "$(dirname "$log")"
    info "log: $log"
    set +e
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    return $rc
}

# --- Quark's calibration cache ------------------------------------------
# Quark spools every calibration activation to %TEMP%\quark_onnx.calib.*, and
# it is NOT cleaned up if the process is killed. Measured 2026-09-05, after
# filling the C: drive from 30 GB free to 9 GB. ResNet at 224x224 is a fraction
# of that, which is why --limit 300 was always fine there.
#
# The rate scales with ACTIVATION WIDTH, not node count. yolov8n and yolov8s
# have the same 929 nodes (both are depth 0.33; s is only wider), yet s spools
# 6.18 GB for 32 images -- 198 MB/image against n's 105. Both measured at
# 640x640 on 2026-09-05. Using n's number for s underestimates a 300-image run
# by 25 GB, which is enough to fill the disk while require_disk says fine.
MB_PER_CALIB_IMAGE_YOLO=105

# calib_mb_per_image <variant> [size] -- MB of %TEMP% spool per calibration
# image. n and s are measured at 640. m/l/x are deliberately pessimistic
# extrapolations from width x depth: over-estimating only makes require_disk
# refuse sooner, which is the safe direction. Measure before trusting them with
# a large --limit.
#
# `size` scales the answer by activation AREA. Every spooled tensor is an
# activation map, so halving the input side quarters the spool: yolov8n at 416
# is ~44 MB/image, not 105. Using the 640 number for a 416 sweep would refuse
# runs that fit comfortably -- the opposite failure to the one that filled the
# disk, but still a wrong check. Rounds up, and never returns 0.
calib_mb_per_image() {
    local base
    case "$1" in
        n) base=105 ;;   # measured at 640
        s) base=198 ;;   # measured at 640
        m) base=600 ;;   # extrapolated: 1.5x width, 2x depth
        l) base=1000 ;;  # extrapolated
        x) base=1500 ;;  # extrapolated
        *) base=198 ;;
    esac
    awk -v b="$base" -v s="${2:-640}" \
        'BEGIN{ v = b * s * s / (640 * 640); printf "%d", (v < 1 ? 1 : int(v + 0.5)) }'
}

# calib_mb_per_image_resnet [size] -- MB of %TEMP% spool per calibration image
# for ResNet50. ~12 MB/image at 224x224 is why --limit 300 was always fine
# there; scales by activation area for other sizes, same reasoning as the
# yolo helper above.
calib_mb_per_image_resnet() {
    awk -v s="${1:-224}" \
        'BEGIN{ v = 12 * s * s / (224 * 224); printf "%d", (v < 1 ? 1 : int(v + 0.5)) }'
}

tmp_dir() {
    local t="${TEMP:-${TMP:-}}"
    if [ -n "$t" ] && command -v cygpath >/dev/null 2>&1; then
        cygpath -u "$t"
    else
        echo "/c/Users/${USERNAME:-${USER:-Default}}/AppData/Local/Temp"
    fi
}

free_gb() { df -k "${1:-/c}" | awk 'NR==2{printf "%d", $4/1048576}'; }

# require_disk <gb_needed> <what> -- refuse rather than fill the disk.
require_disk() {
    local need="$1" what="$2" have
    have="$(free_gb "$(tmp_dir)")"
    info "disk: ${have} GB free, ~${need} GB needed for $what"
    if [ "$have" -lt "$need" ]; then
        die "not enough free disk: ${have} GB free, ~${need} GB needed for $what.
Quark spools calibration activations to $(tmp_dir) and does not stream them.
Lower --limit (roughly ${MB_PER_CALIB_IMAGE_YOLO} MB per image at 640x640), or free some space."
    fi
    [ "$have" -lt $((need * 2)) ] && warn "this will leave about $((have - need)) GB free"
    return 0
}

# quark_guard -- arm cleanup of Quark temp dirs this script creates, so an
# interrupt doesn't strand tens of GB. Only touches dirs newer than the marker,
# so a quantization running in another shell is left alone.
_quark_marker=""
quark_guard() {
    _quark_marker="$(tmp_dir)/.quark_guard.$$"
    : > "$_quark_marker"
    trap quark_cleanup EXIT INT TERM
}

quark_cleanup() {
    local rc=$? t; t="$(tmp_dir)"
    [ -n "$_quark_marker" ] || return $rc
    # -print0 and one path per iteration. %TEMP% lives under the user profile,
    # so a profile name with a space (C:\Users\John Doe) would make an unquoted
    # `rm -rf $stale` word-split into /c/Users/John and Doe/AppData/... .
    local found=0 p
    while IFS= read -r -d '' p; do
        [ $found -eq 0 ] && warn "cleaning up Quark temp caches in $t"
        found=1
        info "  rm -rf $p ($(du -sh "$p" 2>/dev/null | cut -f1))"
        rm -rf "$p" 2>/dev/null || true
    done < <(find "$t" -maxdepth 1 -type d -name 'quark_onnx.*' \
             -newer "$_quark_marker" -print0 2>/dev/null || true)
    rm -f "$_quark_marker"
    return $rc
}

# npu_verdict <cache_key> -- read the EP's own assignment report and say
# whether the NPU actually took anything.
npu_verdict() {
    # Declare separately: under `set -u`, referring to `key` from a later
    # assignment in the same `local` statement reads it as unbound.
    local key="$1"
    local rep="$REPO_ROOT/$key/vitisai_ep_report.json"
    if [ ! -f "$rep" ]; then
        warn "no $key/vitisai_ep_report.json -- the EP never ran, or that run
did not export XLNX_ONNX_EP_REPORT_FILE (scripts/*.sh always do)"
        return 1
    fi
    python - "$rep" <<'PY'
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1]))
c = Counter(n["device"] for n in d.get("nodeStat", []))
npu = c.get("NPU", 0)
total = sum(c.values())
if npu:
    print(f"\n  VERDICT: {npu}/{total} nodes on the NPU. The EP engaged.")
else:
    print(f"\n  VERDICT: 0/{total} nodes on the NPU -- the EP claimed nothing.")
    sys.exit(2)
PY
}
