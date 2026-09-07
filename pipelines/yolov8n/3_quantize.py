"""
YOLO step 3: quantize yolov8n to XINT8 with the post-processing subgraph kept
in float (so the box/score merge doesn't wreck the scales), optional AdaRound.

    python pipelines/yolov8n/3_quantize.py --calib-dir data\\coco_calib
    python pipelines/yolov8n/3_quantize.py --calib-dir data\\coco_calib --adaround

Run in the env with Quark (resnet_env).
"""
import argparse
import copy
import glob
import os

import cv2
import onnx
from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS
from npu.yolo import input_size, letterbox

IN_MODEL = MODELS / "yolov8n.onnx"

# Post-processing subgraph: from the box/class concats to the final merge.
# Verified against the onnxslim'd export in y1. Kept in float.
HEAD_START = ["/model.22/Concat", "/model.22/Concat_1"]
HEAD_END = ["/model.22/Concat_3"]

# AMD's own YOLOv8 AdaRound params
ADAROUND = {
    "DataSize": 1000, "FixedSeed": 1705472343, "BatchSize": 2,
    "NumIterations": 1000, "LearningRate": 0.1, "OptimAlgorithm": "adaround",
    "OptimDevice": "cpu", "InferDevice": "cpu", "EarlyStop": True,
}


class CocoCalibReader(CalibrationDataReader):
    def __init__(self, folder, limit, imgsz, input_name="images"):
        self.files = sorted(glob.glob(os.path.join(folder, "*.jpg")))[:limit]
        if not self.files:
            raise SystemExit(f"no jpgs in {folder}")
        print(f"calibrating on {len(self.files)} images at {imgsz}x{imgsz}")
        self.input_name = input_name
        self.imgsz = imgsz
        self.i = 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        img = cv2.imread(self.files[self.i])
        self.i += 1
        x, _, _ = letterbox(img, self.imgsz)
        return {self.input_name: x}

    def rewind(self):
        self.i = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--adaround", action="store_true")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=None,
                    help="number of CPU threads for PyTorch/OpenMP (default: respects OMP_NUM_THREADS or 4)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    threads = args.threads
    if threads is None:
        threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass
    print(f"CPU threads: {threads}")

    tag = "xint8_adaround" if args.adaround else "xint8"
    out = args.out or str(MODELS / f"yolov8n_{tag}.onnx")

    qc = copy.deepcopy(get_default_config("XINT8"))
    qc.subgraphs_to_exclude = [(HEAD_START, HEAD_END)]
    if args.adaround:
        qc.include_fast_ft = True
        params = dict(ADAROUND)
        params["NumIterations"] = args.iters
        params["DataSize"] = min(args.limit, params["DataSize"])
        qc.extra_options["FastFinetune"] = params

    print(f"config: XINT8{' + AdaRound' if args.adaround else ''}  ->  {out}")
    print(f"float subgraph: {HEAD_START} .. {HEAD_END}")
    # Calibrate at whatever resolution this model was exported at, not a fixed
    # 640: re-exporting yolov8n.onnx smaller would otherwise silently calibrate
    # on 640px activations for a 512px graph.
    imgsz = input_size([d.dim_value for d in onnx.load(str(IN_MODEL))
                        .graph.input[0].type.tensor_type.shape.dim], str(IN_MODEL))
    dr = CocoCalibReader(args.calib_dir, args.limit, imgsz)
    ModelQuantizer(Config(global_quant_config=qc)).quantize_model(str(IN_MODEL), out, dr)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
