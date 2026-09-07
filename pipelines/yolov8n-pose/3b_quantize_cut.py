"""
YOLO-pose step 3b: quantize the head-cut model (from 1b_cut_head.py) to XINT8.

Same recipe as pipelines/yolov8n/3b_quantize_cut.py -- no float tail left in
the graph, so nothing to exclude, every node is a candidate for the NPU.
Calibration reuses data/coco_calib (plain COCO images, no keypoint labels
needed): calibration only ever looks at images, not annotations.

    conda activate resnet_env
    python pipelines/yolov8n-pose/3b_quantize_cut.py --calib-dir data/coco_calib
    python pipelines/yolov8n-pose/3b_quantize_cut.py --calib-dir data/coco_calib --adaround

Writes models/yolov8n-pose_cut_xint8.onnx (or _xint8_adaround.onnx).
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

from npu.paths import MODELS
from npu.yolo import input_size
from npu.yolo_pose import HEAD_OUTS, letterbox

IN_MODEL = MODELS / "yolov8n-pose_cut.onnx"

# Same AdaRound params as pipelines/yolov8n/3b_quantize_cut.py (AMD's own
# yolov8m example set) -- unverified for pose specifically, since AdaRound has
# not been run on this model yet.
ADAROUND = {
    "DataSize": 1000, "FixedSeed": 1705472343, "BatchSize": 2,
    "NumIterations": 1000, "LearningRate": 0.1, "OptimAlgorithm": "adaround",
    "OptimDevice": "cpu", "InferDevice": "cpu", "EarlyStop": True,
}


class CocoCalibReader(CalibrationDataReader):
    """Uses npu.yolo_pose.letterbox (== npu.yolo.letterbox), the same
    transform inference uses. Do not inline a second copy."""

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
        if img is None:
            raise RuntimeError(f"cv2 could not read {self.files[self.i]}")
        self.i += 1
        if self.i % 50 == 0:
            print(f"  calib {self.i}/{len(self.files)}")
        x, _, _ = letterbox(img, self.imgsz)
        return {self.input_name: x}

    def rewind(self):
        self.i = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(IN_MODEL))
    ap.add_argument("--calib-dir", default=str(Path("data") / "coco_calib"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--adaround", action="store_true")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--device", default="cpu",
                    help="OptimDevice/InferDevice for AdaRound's FastFinetune")
    ap.add_argument("--threads", type=int, default=None,
                    help="number of CPU threads for PyTorch/OpenMP (default: respects OMP_NUM_THREADS or 4)")
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

    # On Windows with SMT (e.g. 8 cores / 16 threads), pin to physical cores
    # to avoid SMT thread thrashing on OpenMP barriers.
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
        raise SystemExit(f"{args.src} missing - run 1b_cut_head.py first")

    m = onnx.load(args.src)
    got = [o.name for o in m.graph.output]
    if got != HEAD_OUTS:
        raise SystemExit(f"expected the 9 head outputs in HEAD_OUTS order, got {got}")
    imgsz = input_size([d.dim_value for d in
                        m.graph.input[0].type.tensor_type.shape.dim], args.src)
    print(f"input model: {args.src}, {len(m.graph.node)} nodes, {len(got)} outputs, "
          f"{imgsz}x{imgsz} input")

    qc = copy.deepcopy(get_default_config("XINT8"))
    if not hasattr(qc, "calibrate_method"):
        raise SystemExit(f"get_default_config('XINT8') returned {type(qc)}, not the "
                         "legacy QuantizationConfig - check the Quark version")

    tag = "xint8"
    if args.adaround:
        tag = "xint8_adaround"
        params = dict(ADAROUND)
        params["NumIterations"] = args.iters
        params["DataSize"] = min(args.limit, params["DataSize"])
        qc.include_fast_ft = True
        qc.extra_options["FastFinetune"] = params
        print(f"AdaRound: {args.iters} iterations (slow - expect many minutes)")

    out = args.out or str(MODELS / f"{Path(args.src).stem}_{tag}.onnx")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    dr = CocoCalibReader(args.calib_dir, args.limit, imgsz)
    print(f"quantizing with XINT8{' + AdaRound' if args.adaround else ''} -> {out}")
    ModelQuantizer(Config(global_quant_config=qc)).quantize_model(args.src, out, dr)

    q = onnx.load(out)
    ops = Counter(n.op_type for n in q.graph.node)
    print(f"\nwrote {out}")
    print("ops:", ops.most_common())
    print(f"QuantizeLinear {ops['QuantizeLinear']}  DequantizeLinear {ops['DequantizeLinear']}")
    first = q.graph.node[0]
    print(f"first node: {first.op_type} {first.name}")
    if first.op_type != "QuantizeLinear":
        print("WARNING: graph does not start with QuantizeLinear - check that the "
              "input tensor got quantized.")
    print("\nNext: tools/diag_ep.py --model <out>, then 4_pose.py --ep npu --fresh")


if __name__ == "__main__":
    main()
