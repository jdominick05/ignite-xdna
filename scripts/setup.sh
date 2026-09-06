#!/usr/bin/env bash
# One-time build: export the models, fetch the datasets, quantize.
# Everything it produces is git-ignored and reproducible, which is why it is
# not in the repo. Budget ~2 GB of downloads and a while for AdaRound.
#
#   ./scripts/setup.sh              # both pipelines
#   ./scripts/setup.sh --resnet     # ResNet50 only
#   ./scripts/setup.sh --yolo       # YOLOv8n only
#   ./scripts/setup.sh --no-data    # skip the downloads, build models only
#
# ImageNet is gated: you need a Hugging Face account approved for
# ILSVRC/imagenet-1k plus `hf auth login`. COCO is ungated.
#
# Existing artifacts are left alone; delete one to rebuild it.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

DO_RESNET=1 DO_YOLO=1 DO_DATA=1 N_CALIB=300 N_EVAL=1000
while [ $# -gt 0 ]; do
    case "$1" in
        --resnet)  DO_YOLO=0 ;;
        --yolo)    DO_RESNET=0 ;;
        --no-data) DO_DATA=0 ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         die "unknown flag $1" ;;
    esac
    shift
done

have() { [ -f "$1" ] && { info "have $1"; return 0; } || return 1; }

if [ "$DO_RESNET" = 1 ]; then
    step "ResNet50"
    use_env resnet_env
    have models/resnet50_fp32.onnx || python pipelines/resnet50/1_export.py
    if [ "$DO_DATA" = 1 ] && [ ! -d data/eval ]; then
        info 'ImageNet is gated -- needs an approved HF account and `hf auth login`'
        python pipelines/resnet50/2_fetch_imagenet.py \
            --n-calib "$N_CALIB" --n-eval "$N_EVAL" --shards 0 7
    fi
    need_dir data/calib "ImageNet fetch did not produce calibration images"
    # --out is NOT optional: 3_quantize.py defaults to
    # models/resnet50_<config-lower>.onnx == resnet50_xint8.onnx, but 4_run.py's
    # default and resnet-bench.sh both want resnet50_int8.onnx. Without it the
    # `have` check never becomes true and resnet-bench.sh dies on need_file.
    have models/resnet50_int8.onnx || \
        python pipelines/resnet50/3_quantize.py --calib-dir data/calib --config XINT8 \
            --out models/resnet50_int8.onnx
    have models/resnet50_xint8_adaround.onnx || {
        info "AdaRound -- slow"
        python pipelines/resnet50/3_quantize.py --calib-dir data/calib --config XINT8_ADAROUND
    }
fi

if [ "$DO_YOLO" = 1 ]; then
    step "YOLOv8n"
    use_env resnet_env
    have models/yolov8n.onnx || python pipelines/yolov8n/1_export.py --size 640
    if [ "$DO_DATA" = 1 ] && [ ! -d data/coco_calib ]; then
        python pipelines/yolov8n/2_fetch_coco.py --n-calib "$N_CALIB"
    fi
    need_dir data/coco_calib "COCO fetch did not produce calibration images"
    have models/yolov8n_cut.onnx || python pipelines/yolov8n/1b_cut_head.py
    if ! have models/yolov8n_cut_xint8.onnx; then
        # The same trap yolo-cut.sh guards: Quark spools ~105 MB per 640x640
        # image into %TEMP% and strands it if killed, so 300 images peaks near
        # 31 GB. Checked here too, because setup.sh is the first thing a fresh
        # clone runs and it calibrates on 300 images by default.
        require_disk $(( (N_CALIB * MB_PER_CALIB_IMAGE_YOLO + 1023) / 1024 + 2 )) \
                     "Quark's calibration cache"
        quark_guard
        python pipelines/yolov8n/3b_quantize_cut.py --calib-dir data/coco_calib \
            --limit "$N_CALIB"
    fi
fi

step "done"
echo "  ./scripts/resnet-bench.sh    the ResNet50 results table"
echo "  ./scripts/yolo-cut.sh        YOLOv8n on the NPU, end to end"
echo "  ./scripts/yolo-demo.sh       webcam"
echo "  ./scripts/diag.sh            what the EP took"
