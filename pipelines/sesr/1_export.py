"""
SESR step 1: Export clean NCHW SESR-M7 to ONNX (opset 17, static shape).

Takes the pre-trained weights from AMD's Hugging Face release
(amd/sesr-m7-256x256-tiles-amdnpu) and maps them into a clean NCHW PyTorch model:
  Conv2d(3, 16, 5) -> 7 x [Conv2d(16, 16, 3) + ReLU] + Skip -> Conv2d(16, 12, 5) -> PixelShuffle(2)

Exported with static shape (1, 3, 256, 256) -> (1, 3, 512, 512), opset 17.

    conda activate resnet_env
    python pipelines/sesr/1_export.py
"""
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import onnx
import onnx.numpy_helper as nh
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from npu.paths import MODELS
from npu.sesr import INPUT_SIZE, SCALE

AMD_SESR_URL = (
    "https://huggingface.co/amd/sesr-m7-256x256-tiles-amdnpu/resolve/main/onnx-models/sesr_nhwc_fp32_256x256.onnx"
)
RAW_AMD_MODEL = MODELS / "sesr_m7_amd_fp32.onnx"
DEFAULT_OUT = MODELS / "sesr_m7_fp32.onnx"


class SESRM7(nn.Module):
    """Clean NCHW SESR-M7 (Collapsible Linear Blocks for Super-Efficient Super Resolution).

    Bhardwaj et al., MLSys 2022 (arXiv:2103.09404).
    Linear reparameterized inference graph:
      head: 5x5 Conv (3 -> 16 channels, pad 2, no bias)
      body: 7 blocks of 3x3 Conv (16 -> 16 channels, pad 1, no bias) + ReLU
      skip: head output added to body output
      tail: 5x5 Conv (16 -> 3 * scale^2 channels, pad 2, no bias)
      ps:   PixelShuffle(scale) -> (1, 3, H*scale, W*scale)
    """

    def __init__(self, channels=16, m=7, scale=SCALE):
        super().__init__()
        self.head = nn.Conv2d(3, channels, kernel_size=5, padding=2, bias=False)
        self.body = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
            )
            for _ in range(m)
        ])
        self.tail = nn.Conv2d(channels, 3 * (scale ** 2), kernel_size=5, padding=2, bias=False)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        h = self.head(x)
        feat = h
        for block in self.body:
            feat = block(feat)
        out = feat + h
        out = self.tail(out)
        return self.ps(out)


def download_raw_model():
    MODELS.mkdir(parents=True, exist_ok=True)
    if not RAW_AMD_MODEL.exists():
        print(f"Downloading raw AMD SESR model from Hugging Face -> {RAW_AMD_MODEL}")
        urlretrieve(AMD_SESR_URL, RAW_AMD_MODEL)
        print("Download complete.")


def load_amd_weights(net: SESRM7):
    m = onnx.load(str(RAW_AMD_MODEL))
    inits = {t.name: torch.from_numpy(nh.to_array(t).copy()) for t in m.graph.initializer}

    net.head.weight.data.copy_(inits["head.body_reparam.weight"])
    for i in range(7):
        net.body[i][0].weight.data.copy_(inits[f"body.{i}.body.0.body_reparam.weight"])
    net.tail.weight.data.copy_(inits["tail.body_reparam.weight"])
    print("Loaded 9 weight tensors into SESRM7 PyTorch model.")


def main():
    download_raw_model()

    net = SESRM7(channels=16, m=7, scale=SCALE)
    load_amd_weights(net)
    net.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        out_py = net(dummy)
    expected_out_size = INPUT_SIZE * SCALE
    assert out_py.shape == (1, 3, expected_out_size, expected_out_size), f"Unexpected shape {out_py.shape}"

    out_path = DEFAULT_OUT
    print(f"Exporting clean NCHW model to {out_path} (opset 17, static shape)...")
    torch.onnx.export(
        net,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )

    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    ops = [n.op_type for n in m.graph.node]
    print(f"Exported {out_path}: {len(m.graph.node)} nodes")
    print(f"  Input:  {[d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]}")
    print(f"  Output: {[d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]}")
    print(f"  Ops:    {set(ops)}")


if __name__ == "__main__":
    main()
