#!/usr/bin/env bash
# What did the VitisAI EP actually take? Reads every compile cache's
# vitisai_ep_report.json, which the EP writes on every session build -- unlike
# the compile log, which prints only on a real compile.
#
#   ./scripts/diag.sh                       # all caches, one line each
#   ./scripts/diag.sh --full                # + per-node op/device tables
#   ./scripts/diag.sh --model models/yolov8n_cut_xint8.onnx   # + static QDQ audit
#
# modelcachekey is the known-good control: 393 NPU / 2 CPU.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

FULL=0 MODEL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --full)    FULL=1 ;;
        --model)   MODEL="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         die "unknown flag $1" ;;
    esac
    shift
done

use_env resnet_env17

if [ -n "$MODEL" ]; then
    need_file "$MODEL"
    step "static audit -- $MODEL"
    python tools/diag_ep.py --model "$MODEL"
fi

FOUND=0
for key in modelcachekey yolocachekey yolocutcachekey; do
    # Not on the report alone: a run that did not export XLNX_ONNX_EP_REPORT_FILE
    # leaves a compiled artifact and no report, and that cache is precisely the
    # one worth reporting -- skipping it made it look like it never existed.
    if [ ! -f "$key/vitisai_ep_report.json" ] && ! compgen -G "$key/compiled.*.xmodel" >/dev/null; then
        continue
    fi
    FOUND=1
    step "$key"
    if [ "$FULL" = 1 ]; then
        python tools/diag_ep.py --cache-key "$key"
    else
        npu_verdict "$key" || true
        # compgen, not [ -e glob ]: `test` takes one argument, so two or more
        # compiled.*.xmodel files make it fail and report "never compiled".
        if compgen -G "$key/compiled.*.xmodel" >/dev/null; then
            ok "compiled xmodel present"
        else
            warn "no compiled.*.xmodel -- nothing was ever compiled for the NPU"
        fi
    fi
done

[ "$FOUND" = 1 ] || warn "no EP reports found. Run something on --ep npu first."
