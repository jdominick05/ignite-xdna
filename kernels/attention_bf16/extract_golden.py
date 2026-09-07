"""Dump real attention inputs, weights, and reference outputs from mobilevit_xxs to .npy.

Runs in resnet_env (needs torch + timm).
Loads mobilevit_xxs, preprocessed on a real calibration/eval image, runs inference,
and dumps:
    <out_dir>/x.npy              Attention input tensor [BP, N, C]
    <out_dir>/w_qkv.npy          QKV linear weight [3*C, C]
    <out_dir>/b_qkv.npy          QKV linear bias [3*C]
    <out_dir>/q.npy              Query tensor [BP, H, N, D]
    <out_dir>/k.npy              Key tensor [BP, H, N, D]
    <out_dir>/v.npy              Value tensor [BP, H, N, D]
    <out_dir>/scores.npy         Scaled Q @ K^T [BP, H, N, N]
    <out_dir>/attn_weights.npy   Softmax(scores, dim=-1) [BP, H, N, N]
    <out_dir>/context.npy        Attn @ V [BP, H, N, D]
    <out_dir>/w_proj.npy         Output projection weight [C, C]
    <out_dir>/b_proj.npy         Output projection bias [C]
    <out_dir>/y.npy              Attention block output [BP, N, C]
    <out_dir>/meta.json          Shapes, scale, stage, block, image path

Usage:
    conda activate resnet_env
    python kernels/attention_bf16/extract_golden.py --stage 3 --layer 0 --out-dir data/golden/attn_s3_l0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms


def main():
    p = argparse.ArgumentParser(description="Extract golden attention tensors from mobilevit_xxs")
    p.add_argument("--image", default=r"data/calib/000000.jpg", help="input image path")
    p.add_argument("--stage", type=int, default=3, choices=[2, 3, 4], help="mobilevit stage")
    p.add_argument("--layer", type=int, default=0, help="transformer layer index inside stage")
    p.add_argument("--out-dir", default=r"data/golden/attn_s3_l0", help="output directory")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading mobilevit_xxs for stage {args.stage}, layer {args.layer}...")
    model = timm.create_model("mobilevit_xxs", pretrained=True).eval()

    # Find the target attention module
    # stages[stage][1] is MobileVitBlock, which has .transformer
    block = model.stages[args.stage][1]
    if not hasattr(block, "transformer"):
        raise ValueError(f"Stage {args.stage} block 1 has no transformer")
    transformer_layers = block.transformer
    if args.layer >= len(transformer_layers):
        raise ValueError(f"Layer {args.layer} out of range (stage has {len(transformer_layers)} layers)")
    attn_mod = transformer_layers[args.layer].attn

    # Intercept intermediate tensors using forward hooks
    captured = {}

    def hook_attn_forward(module, inp, out):
        x = inp[0]  # [BP, N, C]
        captured["x"] = x.detach().cpu().numpy()
        captured["y"] = out.detach().cpu().numpy()

        B, N, C = x.shape
        num_heads = module.num_heads
        head_dim = module.head_dim
        scale = module.scale

        qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        captured["q"] = q.detach().cpu().numpy()
        captured["k"] = k.detach().cpu().numpy()
        captured["v"] = v.detach().cpu().numpy()

        q_scaled = q * scale
        attn = q_scaled @ k.transpose(-2, -1)
        captured["scores"] = attn.detach().cpu().numpy()

        attn_weights = attn.softmax(dim=-1)
        captured["attn_weights"] = attn_weights.detach().cpu().numpy()

        context = attn_weights @ v
        captured["context"] = context.detach().cpu().numpy()

    handle = attn_mod.register_forward_hook(hook_attn_forward)

    # Preprocess image
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    if os.path.exists(args.image):
        print(f"Loading real image from {args.image}...")
        img = Image.open(args.image).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
    else:
        print(f"Warning: {args.image} not found, using dummy random input")
        img_tensor = torch.randn(1, 3, 256, 256)

    # Run model forward pass
    with torch.no_grad():
        _ = model(img_tensor)

    handle.remove()

    if "x" not in captured:
        raise RuntimeError("Failed to capture attention tensors")

    # Extract weights
    w_qkv = attn_mod.qkv.weight.detach().cpu().numpy()
    b_qkv = attn_mod.qkv.bias.detach().cpu().numpy() if attn_mod.qkv.bias is not None else None
    w_proj = attn_mod.proj.weight.detach().cpu().numpy()
    b_proj = attn_mod.proj.bias.detach().cpu().numpy() if attn_mod.proj.bias is not None else None

    # Save to disk
    np.save(os.path.join(args.out_dir, "x.npy"), captured["x"])
    np.save(os.path.join(args.out_dir, "w_qkv.npy"), w_qkv)
    if b_qkv is not None:
        np.save(os.path.join(args.out_dir, "b_qkv.npy"), b_qkv)
    np.save(os.path.join(args.out_dir, "q.npy"), captured["q"])
    np.save(os.path.join(args.out_dir, "k.npy"), captured["k"])
    np.save(os.path.join(args.out_dir, "v.npy"), captured["v"])
    np.save(os.path.join(args.out_dir, "scores.npy"), captured["scores"])
    np.save(os.path.join(args.out_dir, "attn_weights.npy"), captured["attn_weights"])
    np.save(os.path.join(args.out_dir, "context.npy"), captured["context"])
    np.save(os.path.join(args.out_dir, "w_proj.npy"), w_proj)
    if b_proj is not None:
        np.save(os.path.join(args.out_dir, "b_proj.npy"), b_proj)
    np.save(os.path.join(args.out_dir, "y.npy"), captured["y"])

    meta = {
        "model": "mobilevit_xxs",
        "stage": args.stage,
        "layer": args.layer,
        "image": args.image,
        "x_shape": list(captured["x"].shape),
        "q_shape": list(captured["q"].shape),
        "k_shape": list(captured["k"].shape),
        "v_shape": list(captured["v"].shape),
        "scores_shape": list(captured["scores"].shape),
        "y_shape": list(captured["y"].shape),
        "num_heads": attn_mod.num_heads,
        "head_dim": attn_mod.head_dim,
        "scale": float(attn_mod.scale),
    }

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Successfully extracted golden tensors to {args.out_dir}:")
    print(f"  Input x:      {captured['x'].shape}")
    print(f"  Q/K/V:        {captured['q'].shape} (BP={captured['q'].shape[0]}, H={captured['q'].shape[1]}, N={captured['q'].shape[2]}, D={captured['q'].shape[3]})")
    print(f"  Scores:       {captured['scores'].shape}")
    print(f"  Attn weights: {captured['attn_weights'].shape}")
    print(f"  Context:      {captured['context'].shape}")
    print(f"  Output y:     {captured['y'].shape}")


if __name__ == "__main__":
    main()
