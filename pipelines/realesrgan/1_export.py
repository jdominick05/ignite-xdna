"""Real-ESRGAN step 1: Export clean NCHW Real-ESRGAN models to ONNX (opset 17, static shape).

Exports both 4x super-resolution architectures:
  1. AMD 10-RRDB (realesrgan_amd): 10 RRDB blocks, 120 dense concatenations, 32 base channels.
  2. Real-ESRGAN Compact (SRVGGNet-v3): 16 plain convs, PixelShuffle(4), zero dense concats.

Supports configurable static spatial resolution (default: 64x64 -> 256x256; also 128x128 and 256x256).

    conda activate resnet_env
    python pipelines/realesrgan/1_export.py --arch both --res 64
"""
import argparse
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import onnx
import onnx.numpy_helper as nh
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from npu.paths import MODELS
from npu.realesrgan import DEFAULT_INPUT_SIZE, SCALE

AMD_RRDB_URL = (
    "https://huggingface.co/amd/realesrgan-256x256-tiles-amdnpu/resolve/main/onnx-models/realesrgan_nchw_256x256_fp32.onnx"
)
COMPACT_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
)

RAW_AMD_MODEL = MODELS / "realesrgan_amd_fp32.onnx"
RAW_COMPACT_WEIGHTS = MODELS / "realesr-general-x4v3.pth"


# ---------------------------------------------------------------------------
# Architecture 1: AMD 10-RRDB
# ---------------------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=32, num_grow_ch=16):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat=32, num_grow_ch=16):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class AMD10RRDBNet(nn.Module):
    """AMD's 10-RRDB Real-ESRGAN architecture.

    Features 10 RRDB blocks (each with 3 RDBs of 5 convs), 120 dense Concats,
    and 4x upsampling via two 2x nearest-neighbor Resize stages.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=32, num_block=10, num_grow_ch=16):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList([RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = feat
        for b in self.body:
            body_feat = b(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        # 4x upsampling via two 2x nearest resize + conv
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def load_amd_rrdb_weights(net: AMD10RRDBNet):
    if not RAW_AMD_MODEL.exists():
        print(f"Downloading raw AMD Real-ESRGAN model -> {RAW_AMD_MODEL}")
        urlretrieve(AMD_RRDB_URL, RAW_AMD_MODEL)
    m = onnx.load(str(RAW_AMD_MODEL), load_external_data=False)
    inits = {init.name: nh.to_array(init).copy() for init in m.graph.initializer}
    state_dict = {name: torch.from_numpy(inits[name]) for name, _ in net.named_parameters()}
    net.load_state_dict(state_dict, strict=True)
    print(f"Loaded {len(state_dict)} weight tensors into AMD10RRDBNet.")


# ---------------------------------------------------------------------------
# Architecture 2: Real-ESRGAN Compact (SRVGGNet-v3)
# ---------------------------------------------------------------------------
class SRVGGNetCompact(nn.Module):
    """SRVGGNet-v3 Compact architecture (realesr-general-x4v3).

    A linear feedforward chain of 34 conv layers with PReLU activations,
    upsampling via PixelShuffle(4), and residual skip with nearest interpolation.
    Zero dense concatenations.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def load_compact_weights(net: SRVGGNetCompact):
    if not RAW_COMPACT_WEIGHTS.exists():
        print(f"Downloading Real-ESRGAN Compact weights -> {RAW_COMPACT_WEIGHTS}")
        urlretrieve(COMPACT_WEIGHTS_URL, RAW_COMPACT_WEIGHTS)
    sd = torch.load(str(RAW_COMPACT_WEIGHTS), map_location="cpu", weights_only=False)
    sd = sd.get("params_ema", sd.get("params", sd))
    net.load_state_dict(sd, strict=True)
    print(f"Loaded {len(sd)} weight tensors into SRVGGNetCompact.")


# ---------------------------------------------------------------------------
# Export Logic
# ---------------------------------------------------------------------------
def export_onnx(model: nn.Module, out_path: Path, res: int):
    model.eval()
    dummy_input = torch.randn(1, 3, res, res, dtype=torch.float32)
    with torch.no_grad():
        expected_output = model(dummy_input)

    out_res = res * SCALE
    print(f"Exporting {model.__class__.__name__} static ({1}, 3, {res}, {res}) -> (1, 3, {out_res}, {out_res})")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )

    # Validate exported model
    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    print(f"Exported successfully to {out_path} ({out_path.stat().st_size:,} bytes, {len(m.graph.node)} nodes).")


def main():
    parser = argparse.ArgumentParser(description="Export Real-ESRGAN models to static NCHW ONNX.")
    parser.add_argument(
        "--arch",
        choices=["both", "rrdb", "compact"],
        default="both",
        help="Model architecture to export (default: both).",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help=f"Input square spatial dimension (default: {DEFAULT_INPUT_SIZE}).",
    )
    args = parser.parse_args()

    res = args.res
    MODELS.mkdir(parents=True, exist_ok=True)

    if args.arch in ("both", "rrdb"):
        rrdb_net = AMD10RRDBNet()
        load_amd_rrdb_weights(rrdb_net)
        rrdb_out = MODELS / f"realesrgan_rrdb_r{res}_fp32.onnx"
        export_onnx(rrdb_net, rrdb_out, res)

    if args.arch in ("both", "compact"):
        compact_net = SRVGGNetCompact()
        load_compact_weights(compact_net)
        compact_out = MODELS / f"realesrgan_compact_r{res}_fp32.onnx"
        export_onnx(compact_net, compact_out, res)


if __name__ == "__main__":
    main()
