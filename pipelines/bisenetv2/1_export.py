"""Export BiSeNetV2 to static batch-1 ONNX (opset 17).

Exports:
  1. models/bisenetv2_fp32.onnx (nearest-neighbor upsample in SegmentHead for monolithic DPU compilation)
  2. models/bisenetv2_bilinear_fp32.onnx (stock bilinear upsample in SegmentHead for DPU fragmentation ablation)
  3. models/preprocess_config_bisenetv2.json (metadata)
"""
import argparse
import json
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import onnx
import onnxslim

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from npu.bisenetv2 import INPUT_SIZE, CITYSCAPES_CLASSES
from npu.paths import MODELS

# Self-contained PyTorch BiSeNetV2 architecture matching CoinCheung/BiSeNet
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DetailBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):
        return self.S3(self.S2(self.S1(x)))


class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        return self.fuse(feat)


class CEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        return self.conv_last(feat)


class GELayerS1(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super().__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv2(self.dwconv(self.conv1(x)))
        return self.relu(feat + x)


class GELayerS2(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super().__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=2, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=2, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv2(self.dwconv2(self.dwconv1(self.conv1(x))))
        shortcut = self.shortcut(x)
        return self.relu(feat + shortcut)


class SegmentBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(GELayerS2(16, 32), GELayerS1(32, 32))
        self.S4 = nn.Sequential(GELayerS2(32, 64), GELayerS1(64, 64))
        self.S5_4 = nn.Sequential(GELayerS2(64, 128), GELayerS1(128, 128), GELayerS1(128, 128), GELayerS1(128, 128))
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat5_5


class BGALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_d, x_s):
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.up1(self.right1(x_s))
        right2 = self.right2(x_s)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        return self.conv(left + right)


class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_mode='nearest'):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, 1, 1, 0, bias=True)
        if up_mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        else:
            self.up = nn.Upsample(scale_factor=8, mode='nearest')

    def forward(self, x):
        feat = self.conv(x)
        feat = self.conv_out(feat)
        return self.up(feat)


class BiSeNetV2(nn.Module):
    def __init__(self, n_classes=19, up_mode='nearest'):
        super().__init__()
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()
        self.head = SegmentHead(128, 1024, n_classes, up_mode=up_mode)

    def forward(self, x):
        feat_d = self.detail(x)
        feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        return self.head(feat_head)


def export_variant(weights_path, up_mode, out_path, size=INPUT_SIZE):
    print(f"\n--- Exporting BiSeNetV2 ({up_mode} upsample) -> {out_path} ---")
    model = BiSeNetV2(n_classes=19, up_mode=up_mode)
    sd = torch.load(weights_path, map_location="cpu")
    # Filter out aux keys
    filtered_sd = {k: v for k, v in sd.items() if not k.startswith("aux")}
    model.load_state_dict(filtered_sd, strict=False)
    model.eval()

    dummy = torch.randn(1, 3, size, size)
    raw_path = out_path.replace(".onnx", "_raw.onnx")
    torch.onnx.export(
        model,
        dummy,
        raw_path,
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        do_constant_folding=True
    )

    print("Optimizing graph with onnxslim...")
    m = onnxslim.slim(raw_path)
    onnx.save(m, out_path)
    if os.path.isfile(raw_path):
        os.remove(raw_path)

    m = onnx.load(out_path)
    print(f"Exported {out_path}: {len(m.graph.node)} nodes")
    op_counts = {}
    for n in m.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {cnt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(MODELS / "model_final_v2_city.pth"))
    ap.add_argument("--size", type=int, default=INPUT_SIZE)
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        raise SystemExit(f"Weights file {args.weights} not found. Download it first.")

    out_nearest = str(MODELS / "bisenetv2_fp32.onnx")
    out_bilinear = str(MODELS / "bisenetv2_bilinear_fp32.onnx")

    export_variant(args.weights, "nearest", out_nearest, args.size)
    export_variant(args.weights, "bilinear", out_bilinear, args.size)

    cfg_path = MODELS / "preprocess_config_bisenetv2.json"
    with open(cfg_path, "w") as f:
        json.dump({
            "model_type": "bisenetv2",
            "classes": CITYSCAPES_CLASSES,
            "num_classes": len(CITYSCAPES_CLASSES),
            "input_size": args.size,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "linear"
        }, f, indent=2)
    print(f"Saved metadata to {cfg_path}")


if __name__ == "__main__":
    main()
