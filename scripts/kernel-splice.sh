#!/usr/bin/env bash
# Native Phoenix Conv2D -> bf16 GroupNorm(32) -> Conv2D BO-splice validation.
#   ./scripts/kernel-splice.sh --compile --L 1024 --chunk 256 --iters 30
#   ./scripts/kernel-splice.sh --L 1024 --chunk 256 --iters 30  # reuse explicit artifacts
# Additional flags go to tests/test_kernel_splice.py. Requires mlir-aie ironenv.
# Writes a new timestamped, UTF-8/profile-scrubbed evidence log for each invocation.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
if [ "${1:-}" = --help ]; then usage "${BASH_SOURCE[0]}"; exit 0; fi
tag="$(date -u +%Y%m%dT%H%M%SZ)_${RANDOM}"
log="results/aie/kernel_splice_conv_gn32_phoenix_${tag}.log"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
# Tee the raw stream only into ignored scratch; the checked-in evidence receives
# redaction before its first write, including compiler diagnostics and tracebacks.
mkdir -p scratch
raw="scratch/kernel_splice_${tag}.log"
if run_logged "$raw" bash scripts/research-iron.sh tests/test_kernel_splice.py --hardware "$@"; then
    rc=0
else
    rc=$?
fi
if grep -Ei 'Allocating local copy|Reverting to host copy|KDMA not supported' "$raw"; then
    printf '\nWRAPPER_VERDICT FAIL: XRT host-copy fallback detected\n' >> "$raw"
    rc=1
fi
printf '\nWRAPPER_EXIT_CODE %s\n' "$rc" >> "$raw"
python -c 'import pathlib,re,sys; s=pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"); s=re.sub(r"(?i)([A-Z]:[\\/]Users[\\/])[^\\/\s\"<>]+",r"\1<user>",s); s=re.sub(r"(?i)(/[A-Z]/Users/)[^/\s\"<>]+",r"\1<user>",s); pathlib.Path(sys.argv[2]).write_text(s,encoding="utf-8")' "$raw" "$log"
info "evidence: $log (exit $rc)"
exit "$rc"
