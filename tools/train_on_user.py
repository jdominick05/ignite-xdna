"""
Offline Deep Personalization: Quantize MODNet on user webcam frames using Quark MinMSE.

Usage:
    conda activate resnet_env
    python tools/train_on_user.py
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npu.modnet import preprocess
from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config


class UserDataReader(CalibrationDataReader):
    def __init__(self, calib_files, target_size=512):
        self.files = calib_files
        self.target_size = target_size
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.files):
            return None
        fpath = self.files[self.idx]
        self.idx += 1
        bgr = cv2.imread(str(fpath))
        if bgr is None:
            return self.get_next()
        arr, _ = preprocess(bgr, self.target_size)
        return {"input": arr}


def main():
    user_calib_dir = ROOT / "data" / "user_calib"
    user_files = sorted(list(user_calib_dir.glob("*.jpg")))

    if not user_files:
        print(f"[train_on_user] ERROR: No calibration frames found in {user_calib_dir}")
        print("[train_on_user] Please run 'python demos/portrait_matting_demo.py' and press [T] to capture your webcam frames first!")
        sys.exit(1)

    print(f"[train_on_user] Found {len(user_files)} user webcam frames in {user_calib_dir}")
    fp32_model = ROOT / "models" / "modnet" / "modnet_zero_concat_fp32.onnx"
    output_xint8 = ROOT / "models" / "modnet" / "modnet_user_xint8.onnx"

    if not fp32_model.exists():
        print(f"[train_on_user] ERROR: FP32 model not found at {fp32_model}")
        sys.exit(1)

    print("[train_on_user] Calibrating INT8 quantization with Quark MinMSE on your camera feed...")
    reader = UserDataReader(user_files)
    quant_config = get_default_config("XINT8")
    config = Config(global_quant_config=quant_config)
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(str(fp32_model), str(output_xint8), reader)

    print("\n=======================================================")
    print(f" Personalized model saved to: {output_xint8}")
    print(" To launch the live demo with your tailored model:")
    print("   conda activate resnet_env17")
    print("   $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'")
    print(f"   python demos/portrait_matting_demo.py --model {output_xint8}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
