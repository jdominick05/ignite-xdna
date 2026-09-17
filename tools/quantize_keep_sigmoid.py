"""resnet_env (Quark): XINT8 exactly as pipelines/<p>/3b_quantize_cut.py quantizes a head-cut model (same default
config, same CocoCalibReader, 200 calibration images from data/coco_calib), except
extra_options["ConvertSigmoidToHardSigmoid"] = False, so SiLU keeps a real Sigmoid instead of the HardSigmoid that
SimulateDPU writes. Not a model the graph engine lowers today (the compiler matches only HardSigmoid + Mul); it is the
accuracy a real quantized sigmoid keeps under Quark's own calibration. Prints the op counts.

    python tools/quantize_keep_sigmoid.py models/yolov8n_cut.onnx out.onnx --pipeline yolov8n
    python tools/quantize_keep_sigmoid.py models/yolov8n-pose_cut.onnx out.onnx --pipeline yolov8n-pose
"""
import argparse
import copy
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--pipeline", default="yolov8n", help="pipelines/<name>/3b_quantize_cut.py supplies the reader")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("q3b", ROOT / "pipelines" / args.pipeline / "3b_quantize_cut.py")
    q3b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q3b)

    m = onnx.load(args.src)
    imgsz = q3b.input_size([d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim], args.src)
    qc = copy.deepcopy(q3b.get_default_config("XINT8"))
    qc.extra_options["ConvertSigmoidToHardSigmoid"] = False
    print(f"XINT8 with ConvertSigmoidToHardSigmoid=False; extra_options {qc.extra_options}; {imgsz}x{imgsz}")
    dr = q3b.CocoCalibReader(str(ROOT / "data" / "coco_calib"), args.limit, imgsz)
    q3b.ModelQuantizer(q3b.Config(global_quant_config=qc)).quantize_model(args.src, args.out, dr)
    ops = Counter(n.op_type for n in onnx.load(args.out).graph.node)
    print(f"wrote {Path(args.out).name}: Sigmoid {ops['Sigmoid']}, HardSigmoid {ops['HardSigmoid']}, Mul {ops['Mul']}, "
          f"QuantizeLinear {ops['QuantizeLinear']}, DequantizeLinear {ops['DequantizeLinear']}")


if __name__ == "__main__":
    main()
