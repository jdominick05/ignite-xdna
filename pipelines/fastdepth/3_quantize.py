"""
FastDepth step 3: Quantize FastDepth to XINT8 (or XINT8 + AdaRound).

Uses npu.fastdepth.preprocess, ensuring calibration and inference are byte-identical.
Defaults to 8 CPU threads and 8 physical cores (0x5555 affinity mask on Zen 4)
to prevent SMT thrashing during calibration / FastFinetune.

    conda activate resnet_env
    python pipelines/fastdepth/3_quantize.py --calib-dir data/fastdepth_calib
    python pipelines/fastdepth/3_quantize.py --calib-dir data/fastdepth_calib --adaround
"""
import argparse
import copy
import glob
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import onnx
from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.fastdepth import INPUT_SIZE, input_size, preprocess
from npu.paths import MODELS

DEFAULT_IN = MODELS / "fastdepth_fp32.onnx"

ADAROUND = {
    "DataSize": 1000,
    "FixedSeed": 1705472343,
    "BatchSize": 2,
    "NumIterations": 1000,
    "LearningRate": 0.1,
    "OptimAlgorithm": "adaround",
    "OptimDevice": "cpu",
    "InferDevice": "cpu",
    "EarlyStop": True,
}


class FastDepthCalibReader(CalibrationDataReader):
    """Feeds calibration images using npu.fastdepth.preprocess."""

    def __init__(self, folder, limit, target_size=INPUT_SIZE, input_name="image"):
        self.files = sorted(glob.glob(os.path.join(folder, "*.jpg")))[:limit]
        if not self.files:
            raise SystemExit(f"No JPG images found in {folder}")
        print(f"Calibrating on {len(self.files)} images at {target_size}x{target_size}")
        self.input_name = input_name
        self.target_size = target_size
        self.i = 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        path = self.files[self.i]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read {path}")
        self.i += 1
        if self.i % 50 == 0:
            print(f"  calib {self.i}/{len(self.files)}")
        tensor, _ = preprocess(img, self.target_size)
        return {self.input_name: tensor}

    def rewind(self):
        self.i = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    ap.add_argument("--calib-dir", default=str(Path("data") / "fastdepth_calib"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--adaround", action="store_true")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    threads = args.threads
    if threads is None:
        threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass

    # Zen 4 8-core physical affinity pinning
    if sys.platform == "win32" and (os.cpu_count() or 0) >= 16:
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.windll.kernel32
            k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
            k32.SetProcessAffinityMask.restype = wintypes.BOOL
            mask = 0x5555 if threads == 8 else (0x55 if threads == 4 else None)
            if mask is not None:
                if k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
                    print(f"Pinned process affinity mask to 0x{mask:X} ({threads} real cores, no SMT)")
        except Exception:
            pass

    print(f"CPU threads: {threads}")

    if not os.path.isfile(args.src):
        raise SystemExit(f"{args.src} missing - run 1_export.py first")

    m = onnx.load(args.src)
    target_size = input_size([d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim], args.src)
    print(f"Input model: {args.src}, {len(m.graph.node)} nodes, {target_size}x{target_size} resolution")

    qc = copy.deepcopy(get_default_config("XINT8"))
    tag = "xint8"
    if args.adaround:
        tag = "xint8_adaround"
        params = dict(ADAROUND)
        params["NumIterations"] = args.iters
        params["DataSize"] = min(args.limit, params["DataSize"])
        params["OptimDevice"] = args.device
        params["InferDevice"] = args.device
        qc.include_fast_ft = True
        qc.extra_options["FastFinetune"] = params
        print(f"AdaRound: {args.iters} iterations on {args.device}")

    out = args.out or str(MODELS / f"{Path(args.src).stem}_{tag}.onnx")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    input_name = m.graph.input[0].name
    dr = FastDepthCalibReader(args.calib_dir, args.limit, target_size=target_size, input_name=input_name)
    print(f"Quantizing with XINT8{' + AdaRound' if args.adaround else ''} -> {out}")
    ModelQuantizer(Config(global_quant_config=qc)).quantize_model(args.src, out, dr)

    q = onnx.load(out)
    ops = Counter(n.op_type for n in q.graph.node)
    print(f"Wrote {out}: {len(q.graph.node)} nodes")
    print("  Ops:", ops.most_common(12))


if __name__ == "__main__":
    main()
