"""
Step 2: Quantize the exported timm ResNet50 to INT8 (XINT8) with AMD Quark.

Requires a folder of real calibration images. See notes at the bottom
for where to get them. Roughly 200-500 images is plenty.

    python pipelines/resnet50/3_quantize.py --calib-dir data/calib --config XINT8_ADAROUND

Produces:
    models/resnet50_<config>.onnx

Run in resnet_env (the only env with Quark).
"""

import argparse
import glob
import json
import os


from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS
from npu.preprocess import IMG_EXTS, build_transform

IN_MODEL = MODELS / "resnet50_fp32.onnx"
CFG_PATH = MODELS / "preprocess_config.json"


class ImageCalibReader(CalibrationDataReader):
    """Feeds calibration images to Quark one batch at a time. `batch` stacks
    that many images per sample to match a static multi-batch export -- the
    graph's input shape is fixed at export time, so calibration must feed the
    same shape or Quark's session creation fails immediately."""

    def __init__(self, calib_dir, cfg, limit=300, input_name="input", batch=1):
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))
        files = sorted(files)[:limit]
        if not files:
            raise SystemExit(f"No images found under {calib_dir}")
        n_batches = len(files) // batch
        files = files[: n_batches * batch]  # drop remainder; every sample must be full
        print(f"Using {len(files)} calibration images ({n_batches} batches of {batch})")

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
    ap.add_argument("--calib-dir", required=True, help="folder of calibration images")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--config", default="XINT8",
                    help="Quark default config name. XINT8 = power-of-two scales "
                         "(coarse, guaranteed on XDNA1). A8W8 = float scales "
                         "(more accurate; may not compile on the X1 backend). "
                         "XINT8_ADAROUND / XINT8_ADAQUANT = power-of-two plus "
                         "fast-finetune for accuracy recovery (slow).")
    ap.add_argument("--out", default=None,
                    help="output path (default: models/resnet50_<config>.onnx)")
    ap.add_argument("--in-model", default=None,
                    help="FP32 model to quantize (default: models/resnet50_fp32.onnx)")
    ap.add_argument("--cfg-path", default=None,
                    help="preprocess config (default: models/preprocess_config.json) -- "
                         "must match the one written by step 1 for this --in-model")
    ap.add_argument("--batch", type=int, default=1,
                    help="must match the static batch --in-model was exported with")
    args = ap.parse_args()
    out_model = args.out or str(MODELS / f"resnet50_{args.config.lower()}.onnx")
    in_model = args.in_model or str(IN_MODEL)
    cfg_path = args.cfg_path or str(CFG_PATH)

    with open(cfg_path) as f:
        cfg = json.load(f)

    dr = ImageCalibReader(args.calib_dir, cfg, limit=args.limit, batch=args.batch)

    print(f"config: {args.config}  ->  {out_model}")
    quant_config = get_default_config(args.config)
    config = Config(global_quant_config=quant_config)

    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(in_model, out_model, dr)

    print(f"\nWrote {out_model}")


if __name__ == "__main__":
    main()
