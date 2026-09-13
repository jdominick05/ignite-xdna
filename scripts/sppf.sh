#!/usr/bin/env bash
# Compile and verify 20x20x256 INT8 SPPF on Phoenix Device 0.
#   ./scripts/sppf.sh --compile
#   ./scripts/sppf.sh --hardware --iters 100
#   ./scripts/sppf.sh --offline
# New scrubbed UTF-8 evidence, with host/device contention and resource witnesses.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
if [ "${1:-}" = --help ]; then usage "${BASH_SOURCE[0]}"; exit 0; fi
tag="$(date -u +%Y%m%dT%H%M%SZ)_${RANDOM}"
bash scripts/research-lowlevel.sh --log "results/aie/sppf_20x20x256_phoenix_${tag}.log" \
    --npu --seconds 1800 --rss-gib 10 -- \
    bash scripts/research-iron.sh tests/test_sppf_kernel.py "$@"
