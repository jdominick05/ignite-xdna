#!/usr/bin/env bash
# End-to-end MobileViT XXS Fused Attention and Heterogeneous Spliced Pipeline Demo.
#
#   ./scripts/attention-demo.sh               # default: Stage 3, 8 AIE cores, 5 iters
#   ./scripts/attention-demo.sh --stage 4      # Stage 4 attention (N=16, D=24)
#   ./scripts/attention-demo.sh --stage 2      # Stage 2 attention (N=256, D=16)
#   ./scripts/attention-demo.sh --cores 4      # Run on 4 AIE cores
#   ./scripts/attention-demo.sh --iters 10     # Benchmark 10 iterations
#
# Demonstrates:
#   1. Fused BF16 Multi-Head Attention kernel running across 8 physical AIE2 cores
#      on the AMD XDNA1 NPU (Ryzen 7 8700G Phoenix).
#   2. Cut CNN backbone running 100% on NPU via VitisAI EP (1 single subgraph, 407 nodes, 1.71 ms).
#   3. Numerical verification against real calibration golden tensors and Top-1 prediction.
#   4. Heterogeneous spliced pipeline executing in 3.28 ms (33x faster than stock VitisAI EP).

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

STAGE=3 CORES=8 ITERS=5 IMAGE="data/calib/000000.jpg"
while [ $# -gt 0 ]; do
    case "$1" in
        --stage)   STAGE="$2"; shift ;;
        --cores)   CORES="$2"; shift ;;
        --iters)   ITERS="$2"; shift ;;
        --image)   IMAGE="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         die "unknown flag $1" ;;
    esac
    shift
done

need_file "models/mobilevit_cut_backbone_xint8.onnx" "run: python scratch/export_cut_backbone.py"
need_dir  "data/golden/attn_s${STAGE}_l0" "run: python kernels/attention_bf16/extract_golden.py --stage $STAGE"

npu_env

step "Running MobileViT Fused Attention & Spliced Pipeline Demo"
python tools/demo_attention.py --stage "$STAGE" --num-cores "$CORES" --iters "$ITERS" --image "$IMAGE"
