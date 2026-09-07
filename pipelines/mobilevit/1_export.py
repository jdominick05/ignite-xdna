"""
Step 1: Export timm MobileViT to ONNX for Ryzen AI (Phoenix/Hawk Point, XDNA1).

Produces:
  models/mobilevit_xxs_fp32.onnx   - the FP32 graph, opset 17, static shapes (1, 3, 256, 256)
  models/preprocess_config_mobilevit_xxs.json - timm's preprocessing params (used by step 2 + 3)

Run in resnet_env (has torch + timm):
    python pipelines/mobilevit/1_export.py
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import timm
import torch
from timm.data import resolve_data_config

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS

MODEL_NAME = "mobilevit_xxs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME,
                    help=f"timm model name (default: {MODEL_NAME})")
    ap.add_argument("--size", type=int, default=None,
                    help="override input side (default: timm's own, 256 for mobilevit_xxs)")
    ap.add_argument("--out", default=None, help="default: models/mobilevit_xxs_fp32.onnx")
    ap.add_argument("--cfg-out", default=None, help="default: models/preprocess_config_mobilevit_xxs.json")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    tag = args.model.replace(".", "_")
    onnx_path = Path(args.out) if args.out else MODELS / f"{tag}_fp32.onnx"
    cfg_path = Path(args.cfg_out) if args.cfg_out else MODELS / f"preprocess_config_{tag}.json"

    MODELS.mkdir(parents=True, exist_ok=True)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} with pretrained=True...")
    model = timm.create_model(args.model, pretrained=True)
    model.eval()

    cfg = resolve_data_config({}, model=model)
    if args.size is not None:
        c, _, _ = cfg["input_size"]
        cfg["input_size"] = (c, args.size, args.size)
    print("timm data config:", cfg)

    c, h, w = cfg["input_size"]
    dummy = torch.randn(args.batch, c, h, w)

    print(f"Exporting to {onnx_path} (opset 17, static shape)...")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    serializable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    with open(cfg_path, "w") as f:
        json.dump(serializable, f, indent=2)

    import onnx
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)

    opsets = [(o.domain or "ai.onnx", o.version) for o in m.opset_import]
    in_shape = [d.dim_value or d.dim_param for d in m.graph.input[0].type.tensor_type.shape.dim]
    ops = Counter(n.op_type for n in m.graph.node)

    print(f"\nWrote {onnx_path}")
    print(f"  opset:       {opsets}")
    print(f"  input shape: {in_shape}")
    print(f"  nodes:       {len(m.graph.node)}")
    print(f"  op types:    {ops.most_common(10)}")

    assert any(v == 17 for _, v in opsets), "opset is not 17"
    assert in_shape[0] == args.batch, f"batch must be static {args.batch}"
    print("\nOK - ready for step 2 (quantization)")


if __name__ == "__main__":
    main()
