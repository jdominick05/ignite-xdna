"""
Step 3: Quantize exported MODNet ONNX to INT8 (XINT8) with AMD Quark.

Feeds real portrait images from data/modnet_calib for power-of-two scale calibration.

Run in resnet_env (has Quark):
    python pipelines/modnet/3_quantize.py --calib-dir data/modnet_calib --config XINT8
    python pipelines/modnet/3_quantize.py --calib-dir data/modnet_calib --config XINT8_ADAROUND --iters 500
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np

from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from npu.modnet import preprocess
IN_MODEL = ROOT / "models" / "modnet" / "modnet_fp32.onnx"
CFG_PATH = ROOT / "models" / "modnet" / "preprocess_config.json"
CALIB_DIR = ROOT / "data" / "modnet_calib"


class PortraitCalibReader(CalibrationDataReader):
    def __init__(self, calib_dir, cfg, limit=200, input_name="input"):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPEG"):
            files.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))
        files = sorted(files)[:limit]
        if not files:
            raise SystemExit(f"No calibration images found under {calib_dir}")
        print(f"[3_quantize] Using {len(files)} calibration images")

        self.files = files
        self.input_name = input_name
        self.height = cfg.get("height", 512)
        self.width = cfg.get("width", 512)
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.files):
            return None
        img_path = self.files[self.idx]
        self.idx += 1

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return self.get_next()
        arr, _ = preprocess(bgr, self.width)
        return {self.input_name: arr}

    def rewind(self):
        self.idx = 0


def main():
    ap = argparse.ArgumentParser(description="Quantize MODNet with Quark.")
    ap.add_argument("--calib-dir", default=str(CALIB_DIR), help=f"Calibration images (default: {CALIB_DIR})")
    ap.add_argument("--limit", type=int, default=100, help="Calibration sample limit (default: 100)")
    ap.add_argument("--config", default="XINT8", choices=["XINT8", "XINT8_ADAROUND", "A8W8"],
                    help="Quark config (default: XINT8)")
    ap.add_argument("--in-model", default=str(IN_MODEL), help=f"FP32 model path (default: {IN_MODEL})")
    ap.add_argument("--out", default=None, help="Output quantized model path")
    ap.add_argument("--iters", type=int, default=500, help="NumIterations for FastFinetune (default: 500)")
    ap.add_argument("--device", default="cpu", help="Device for FastFinetune (default: cpu)")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads for PyTorch/OpenMP (default: 8)")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

    try:
        import torch
        torch.set_num_threads(args.threads)
    except ImportError:
        pass

    out_model = args.out or str(Path(args.in_model).parent / f"modnet_{args.config.lower()}.onnx")
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dr = PortraitCalibReader(args.calib_dir, cfg, limit=args.limit, input_name=cfg.get("input_name", "input"))

    print(f"[3_quantize] Config: {args.config} -> {out_model}")
    quant_config = get_default_config(args.config)
    if "FastFinetune" in quant_config.extra_options:
        quant_config.extra_options["FastFinetune"]["OptimDevice"] = args.device
        quant_config.extra_options["FastFinetune"]["InferDevice"] = args.device
        if args.iters:
            quant_config.extra_options["FastFinetune"]["NumIterations"] = args.iters
        print(f"[3_quantize] FastFinetune device: {args.device}, iters: {args.iters}")

    config = Config(global_quant_config=quant_config)
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(args.in_model, out_model, dr)
    print(f"[3_quantize] Successfully written {out_model}")


if __name__ == "__main__":
    main()
