"""
Step 2: Quantize exported MobileViT to XINT8 with AMD Quark on real calibration images.

Usage:
    python pipelines/mobilevit/2_quantize.py --calib-dir data/calib --limit 300
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS
from npu.preprocess import IMG_EXTS, build_transform

IN_MODEL = MODELS / "mobilevit_xxs_fp32.onnx"
CFG_PATH = MODELS / "preprocess_config_mobilevit_xxs.json"


class ImageCalibReader(CalibrationDataReader):
    def __init__(self, calib_dir, cfg, limit=300, input_name="input", batch=1):
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))
        files = sorted(files)[:limit]
        if not files:
            raise SystemExit(f"No images found under {calib_dir}")
        n_batches = len(files) // batch
        files = files[: n_batches * batch]
        print(f"Using {len(files)} real calibration images ({n_batches} batches of {batch})")

        self.transform = build_transform(cfg)
        self.input_name = input_name
        self.files = files
        self.batch = batch
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.files):
            return None
        chunk = self.files[self.idx: self.idx + self.batch]
        self.idx += self.batch
        import numpy as np
        x = np.concatenate([self.transform(p) for p in chunk], axis=0)
        return {self.input_name: x}

    def rewind(self):
        self.idx = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", default="data/calib", help="folder of calibration images")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--config", default="XINT8", help="Quark default config name (default: XINT8)")
    ap.add_argument("--out", default=None, help="output path")
    ap.add_argument("--in-model", default=None, help="FP32 model to quantize")
    ap.add_argument("--cfg-path", default=None, help="preprocess config")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    in_model = args.in_model or str(IN_MODEL)
    out_model = args.out or str(MODELS / f"mobilevit_xxs_{args.config.lower()}.onnx")
    cfg_path = args.cfg_path or str(CFG_PATH)

    with open(cfg_path) as f:
        cfg_json = json.load(f)

    reader = ImageCalibReader(args.calib_dir, cfg_json, limit=args.limit,
                              input_name="input", batch=args.batch)

    print(f"Configuring Quark quantizer with {args.config}...")
    qc = get_default_config(args.config)
    cfg = Config(global_quant_config=qc)
    quantizer = ModelQuantizer(cfg)

    print(f"Quantizing {in_model} -> {out_model}...")
    quantizer.quantize_model(in_model, out_model, reader)

    print(f"Quantization complete! Output written to {out_model}")


if __name__ == "__main__":
    main()
