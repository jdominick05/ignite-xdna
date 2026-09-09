"""
FastDepth step 1: Export FastDepth monocular depth estimation to ONNX for Ryzen AI.

FastDepth (Wofk et al., ICRA 2019, MIT) uses a MobileNet encoder and a
depthwise-separable convolutional decoder (NNConv5dw-skipadd).
All 5 upsampling layers are nearest-neighbor resize, avoiding the multi-subgraph
fragmentation observed in stock bilinear MiDaS.

Exports:
  models/fastdepth_fp32.onnx               -- Clean ONNX export (opset 17, batch=1, static 1x3x256x256)
  models/preprocess_config_fastdepth.json  -- Preprocessing config metadata

Run in resnet_env (has onnx + onnxslim):
    python pipelines/fastdepth/1_export.py
"""
import argparse
import json
import os
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import onnx
from onnx import version_converter
import onnxslim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.fastdepth import INPUT_SIZE
from npu.paths import MODELS

WASABI_URL = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/146_FastDepth/resources.tar.gz"


def ensure_base_model(out_dir):
    base_path = out_dir / "fast_depth_256x256.onnx"
    if base_path.is_file():
        return base_path

    print(f"Base model {base_path} not found. Downloading from PINTO Model Zoo S3...")
    res = urllib.request.urlopen(WASABI_URL)
    tf = tarfile.open(fileobj=res, mode="r|gz")
    target_member = None
    for member in tf:
        if "256x256" in member.name and member.name.endswith("fast_depth_256x256.onnx"):
            target_member = member
            break

    if target_member is None:
        raise SystemExit(f"Could not locate 256x256 ONNX model in {WASABI_URL}")

    f = tf.extractfile(target_member)
    with open(base_path, "wb") as out:
        out.write(f.read())
    print(f"Downloaded and saved base model to {base_path}")
    return base_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=INPUT_SIZE, help="Input square resolution (default: 256)")
    ap.add_argument("--out-dir", default=str(MODELS), help="Directory to save exported ONNX models")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = ensure_base_model(out_dir)
    print(f"Loading base FastDepth model from {base_path}...")
    m = onnx.load(str(base_path))

    print(f"Upgrading opset to 17...")
    m17 = version_converter.convert_version(m, 17)

    old_in = m17.graph.input[0].name
    old_out = m17.graph.output[0].name
    print(f"Normalizing tensor names: {old_in} -> image, {old_out} -> depth")

    m17.graph.input[0].name = "image"
    for node in m17.graph.node:
        for i, inp in enumerate(node.input):
            if inp == old_in:
                node.input[i] = "image"

    m17.graph.output[0].name = "depth"
    for node in m17.graph.node:
        for i, out in enumerate(node.output):
            if out == old_out:
                node.output[i] = "depth"

    onnx.checker.check_model(m17)

    print("Simplifying exported graph with onnxslim...")
    slim_model = onnxslim.slim(m17)
    onnx.checker.check_model(slim_model)

    fp32_path = out_dir / "fastdepth_fp32.onnx"
    onnx.save(slim_model, str(fp32_path))
    print(f"Wrote {fp32_path}: {len(slim_model.graph.node)} nodes")
    print("  Ops:", Counter(n.op_type for n in slim_model.graph.node).most_common())

    cfg = {
        "architecture": "fastdepth_mobilenet_nnconv5dw",
        "input_name": "image",
        "input_size": [1, 3, args.size, args.size],
        "scale": "1/255.0",
        "output_name": "depth",
        "output_shape": [1, 1, args.size, args.size],
    }
    cfg_path = out_dir / "preprocess_config_fastdepth.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote preprocessing config: {cfg_path}")


if __name__ == "__main__":
    main()
