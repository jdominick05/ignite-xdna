"""
Step 1: Export timm ResNet50 to ONNX for Ryzen AI (Phoenix/Hawk Point, XDNA1).

Produces:
  models/resnet50_fp32.onnx   - the FP32 graph, opset 17, static shapes
  models/preprocess_config.json - timm's preprocessing params (used by step 2 + 3)

Run in resnet_env (the Ryzen AI 1.8.0 clone, which has torch + timm):
    python pipelines/resnet50/1_export.py
    python pipelines/resnet50/1_export.py --size 192 \
        --out models/resnet50_r192_fp32.onnx --cfg-out models/preprocess_config_r192.json
    python pipelines/resnet50/1_export.py --model wide_resnet50_2.racm_in1k \
        --out models/wide_resnet50_2_fp32.onnx --cfg-out models/preprocess_config_wide.json

--size overrides timm's default input_size (224 for resnet50.a1_in1k) while
keeping every other preprocessing param (mean/std/interpolation/crop_pct) from
the checkpoint's own config, so a resolution sweep changes only the pixel
count -- not the normalization recipe, which would confound the comparison.

--model swaps the timm checkpoint entirely (e.g. wide_resnet50_2.racm_in1k, the
2x-width sibling of resnet50, to test whether the "width is nearly free" finding
from the YOLOv8n-vs-s comparison holds on a classifier too). Its own
resolve_data_config is pulled regardless -- never assume another checkpoint
shares resnet50.a1_in1k's mean/std/crop_pct.
"""

import argparse
import json
import os

import timm
import torch
from timm.data import resolve_data_config

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS

MODEL_NAME = "resnet50.a1_in1k"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=MODEL_NAME,
                help=f"timm model name (default: {MODEL_NAME})")
ap.add_argument("--size", type=int, default=None,
                help="override input side (default: timm's own, 224 for this checkpoint)")
ap.add_argument("--out", default=None, help="default: models/resnet50_fp32.onnx")
ap.add_argument("--cfg-out", default=None, help="default: models/preprocess_config.json")
ap.add_argument("--batch", type=int, default=1,
                help="STATIC batch size baked into the graph (default 1, the only "
                     "value used anywhere else in this repo). Not dynamic_axes -- "
                     "this is a fixed-shape export testing whether the VitisAI EP "
                     "accepts N>1 as a static dimension at all, distinct from the "
                     "dynamic-batch path that's rejected in docs/DECISIONS.md.")
args = ap.parse_args()

ONNX_PATH = Path(args.out) if args.out else MODELS / "resnet50_fp32.onnx"
CFG_PATH = Path(args.cfg_out) if args.cfg_out else MODELS / "preprocess_config.json"

MODELS.mkdir(parents=True, exist_ok=True)
ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
CFG_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Load model -----------------------------------------------------------
model = timm.create_model(args.model, pretrained=True)
model.eval()

# --- Pull timm's OWN preprocessing config --------------------------------
# Do not hardcode ImageNet defaults. Each timm checkpoint has its own
# crop_pct / interpolation, and calibration must match inference exactly.
cfg = resolve_data_config({}, model=model)
if args.size is not None:
    c, _, _ = cfg["input_size"]
    cfg["input_size"] = (c, args.size, args.size)
print("timm data config:", cfg)

c, h, w = cfg["input_size"]
dummy = torch.randn(args.batch, c, h, w)

# --- Export ---------------------------------------------------------------
# dynamo=False  -> use the legacy TorchScript exporter. The new dynamo
#                  exporter silently emits opset 18 even when you ask for 17.
# no dynamic_axes -> batch is a fixed STATIC number (--batch, default 1), never
#                  a dynamic dimension. A dynamic batch injects Shape/Concat/
#                  Reshape nodes that are prime CPU-fallback candidates -- that
#                  is what's rejected, not a static batch >1 as such.
torch.onnx.export(
    model,
    dummy,
    str(ONNX_PATH),
    export_params=True,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamo=False,
)

# Save preprocessing config for the next steps
serializable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
with open(CFG_PATH, "w") as f:
    json.dump(serializable, f, indent=2)

# --- Verify ---------------------------------------------------------------
import onnx
from collections import Counter

m = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(m)

opsets = [(o.domain or "ai.onnx", o.version) for o in m.opset_import]
in_shape = [d.dim_value or d.dim_param for d in m.graph.input[0].type.tensor_type.shape.dim]
ops = Counter(n.op_type for n in m.graph.node)

print(f"\nWrote {ONNX_PATH}")
print(f"  opset:       {opsets}")
print(f"  input shape: {in_shape}")
print(f"  op types:    {ops.most_common()}")

assert any(v == 17 for _, v in opsets), "opset is not 17 - check your torch version"
assert in_shape[0] == args.batch, f"batch must be static {args.batch}"
print("\nOK - ready for step 2 (quantization)")
