"""Achieved ops/s from measured fps, replacing the retracted xrt-smi GOPS path
(RESEARCH.md finding 3: GOPS is a per-context notional figure, not delivered
compute). Convention: AMD's 16 TOPS nameplate is INT8 with 2 ops per MAC, so
TOPS = MACs_per_inference * 2 * fps. MACs come from onnx-tool's static graph
analysis of the actual on-NPU graph (the head-cut model, not Ultralytics'
published FLOPs, which include the decode tail this repo runs on CPU instead).

Every (model, fps) pair below is a hardcoded citation of a specific measured
number from results/ or README.md -- this script does the arithmetic, not the
measuring. Update the table when a new config is measured; do not extrapolate.

    python tools/estimate_tops.py
"""
import sys
from pathlib import Path

import onnx_tool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path
from npu.paths import MODELS

NAMEPLATE_TOPS = 16.0

# (label, model file, fps, source)
ROWS = [
    ("resnet50 XINT8 (solo, shared 4x4)", "resnet50_xint8_c64.onnx", 1000 / 5.68,
     "README.md:318, 5.68 ms"),
    ("wide_resnet50_2 XINT8 (solo, shared 4x4)", "wide_resnet50_2_xint8_c64.onnx", 1000 / 9.66,
     "README.md:393, 9.66 ms"),
    ("yolov8n cut XINT8 (solo, shared 4x4)", "yolov8n_cut_xint8.onnx", 1000 / 8.94,
     "README.md:85, 8.94 ms"),
    ("yolov8s cut XINT8 (solo, shared 4x4)", "yolov8s_cut_xint8.onnx", 1000 / 15.63,
     "README.md:86, 15.63 ms"),
    ("yolov8m cut XINT8 (solo, shared 4x4)", "yolov8m_cut_xint8.onnx", 1000 / 30.80,
     "README.md:136, 30.80 ms"),
    ("yolov8l cut XINT8 (solo, shared 4x4)", "yolov8l_cut_xint8.onnx", 1000 / 49.67,
     "README.md:172, 49.67 ms"),
    ("yolov8x cut XINT8 (solo, shared 4x4)", "yolov8x_cut_xint8.onnx", 1000 / 117.11,
     "README.md:173, 117.11 ms"),
    ("yolov8n cut XINT8 (4x independent 1x4 columns)", "yolov8n_cut_xint8.onnx", 245.2,
     "results/multi_partition_yolov8n.log, combined fps @ N=4"),
    ("yolov8m cut XINT8 (4x independent 1x4 columns)", "yolov8m_cut_xint8.onnx", 65.3,
     "results/multi_partition_yolov8m.log, combined fps @ N=4"),
    ("yolov8l cut XINT8 (4x independent 1x4 columns)", "yolov8l_cut_xint8.onnx", 37.3,
     "results/multi_partition_yolov8l.log, combined fps @ N=4"),
    ("yolov8x cut XINT8 (4x independent 1x4 columns)", "yolov8x_cut_xint8.onnx", 23.2,
     "results/multi_partition_yolov8x.log, combined fps @ N=4"),
    ("yolov8x@1280 cut XINT8, throughput-only (4x independent 1x4 columns)",
     "yolov8x_r1280_cut_xint8.onnx", 6.0,
     "results/multi_partition_yolov8x_r1280.log, combined fps @ N=4"),
    ("wide_resnet101_2 XINT8 (solo, shared 4x4)", "wide_resnet101_2_xint8_c64.onnx",
     1000 / 17.90, "README.md:320, 17.90 ms"),
    ("wide_resnet101_2 XINT8 (4x independent 1x4 columns)",
     "wide_resnet101_2_xint8_c64.onnx", 98.3,
     "results/multi_partition_wide_resnet101_2.log, combined fps @ N=4"),
]

_mac_cache = {}


def macs_for(model_file):
    if model_file not in _mac_cache:
        m = onnx_tool.Model(str(MODELS / model_file))
        m.graph.shape_infer()
        m.graph.profile()
        _mac_cache[model_file] = m.graph.macs[0]
    return _mac_cache[model_file]


def main():
    print(f"{'config':<48} {'MACs':>14} {'fps':>8} {'TOPS':>8} {'% of 16':>8}   source")
    for label, model_file, fps, source in ROWS:
        macs = macs_for(model_file)
        tops = macs * 2 * fps / 1e12
        pct = 100 * tops / NAMEPLATE_TOPS
        print(f"{label:<48} {macs:>14.4g} {fps:>8.2f} {tops:>8.3f} {pct:>7.1f}%   {source}")


if __name__ == "__main__":
    main()
