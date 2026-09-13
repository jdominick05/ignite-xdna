#!/usr/bin/env bash
# Compile/verify the standalone Phoenix Conv+Residual+SiLU epilogue.
#   ./scripts/fused-epilogue.sh --compile --cin 512
#   ./scripts/fused-epilogue.sh --hardware --cin 512 --iters 10
#   ./scripts/fused-epilogue.sh --compile --hardware --cin 32 --iters 3
#   ./scripts/fused-epilogue.sh --offline
# Every invocation creates a new UTF-8, profile-scrubbed evidence log.
# Uses the existing ironenv and bounds the child with host/NPU witnesses.
# The <8% qualification is for Cin=512, Cout=32, 3x3, eight spatial outputs.
# Cin=32 is a diagnostic comparison; hardware always enforces the target.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
if [ "${1:-}" = --help ]; then usage "${BASH_SOURCE[0]}"; exit 0; fi
cin=512
next=""
for arg in "$@"; do
    if [ "$next" = cin ]; then cin="$arg"; next=""; continue; fi
    case "$arg" in --cin) next=cin ;; --cin=*) cin="${arg#*=}" ;; esac
done
case "$cin" in 32|512) ;; *) die "--cin must be 32 or 512" ;; esac
tag="$(date -u +%Y%m%dT%H%M%SZ)_${RANDOM}"
log="results/aie/fused_epilogue_phoenix_cin${cin}_${tag}.log"
bash scripts/research-lowlevel.sh --log "$log" --npu --seconds 1800 --rss-gib 10 -- \
    bash scripts/research-iron.sh tests/test_fused_epilogue_kernel.py "$@"
