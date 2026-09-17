"""
YOLO-World v2 step 6: export the text side, so a container can take any vocabulary at run time.

Why:
  A YOLO-World v2 container runs the four text cross-attention regions on the host, and the vocabulary reaches the
  network only through their text guides (initializers the runtime can replace, EngineSession.set_host_constants)
  and through the contrastive decode. Both are functions of the class names' CLIP ViT-B/32 text embeddings. This
  step writes what the runtime needs to compute them without torch:
    models/yolow_text_encoder.onnx   CLIP ViT-B/32's text encoder: int64 tokens [K, 77] -> L2-normalized [K, 512]
    models/yolow_text.npz            the tokenizer's BPE merges, the four guide projections (model.{b}.attn.gl) and
                                     the contrastive head's per-level logit scale (exp) and bias
  ignite_xdna.pipelines.yolow_text reads them.

Checks, printed (the script exits 1 if one fails):
  - ignite_xdna's tokenizer gives clip.tokenize's tokens on every test phrase (COCO's names and awkward text);
  - the ONNX encoder's embeddings against PyTorch's (max |diff|);
  - the guides computed from the bundle for COCO's names against the exported model's initializers;
  - the contrastive constants against npu.yolow.SCALES and BIASES.

    conda activate resnet_env      (torch, ultralytics and its clip; downloads ViT-B-32.pt on first use)
    python pipelines/yolow/6_text_encoder.py
"""
import argparse
import gzip
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

from npu.paths import MODELS
from npu.yolo import COCO_CLASSES
import npu.yolow as yw

TEST_PHRASES = ["a photo of a cat", "red backpack", "traffic cone", "  person   on a bike ", "hot-dog", "t-shirt",
                "don't walk sign", "it's a dog's life", "100 dollar bill", "3d printer", "ice cream (vanilla)",
                "rock &amp; roll", "a.b.c", "under_score", "semi;colon", "UPPER Case", "x" * 200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=str(MODELS / "yolov8s-worldv2.pt"))
    ap.add_argument("--cut-onnx", default=str(MODELS / "yolov8s-worldv2_cut.onnx"),
                    help="the exported cut model whose text guides the bundle must reproduce")
    ap.add_argument("--clip-dir", default=str(MODELS / "clip"), help="where clip.load keeps ViT-B-32.pt")
    ap.add_argument("--out-dir", default=str(MODELS))
    args = ap.parse_args()

    import clip
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import numpy_helper
    from ultralytics import YOLOWorld

    from ignite_xdna.pipelines import yolow_text as yt

    out_dir = Path(args.out_dir)
    onnx_path, npz_path = out_dir / "yolow_text_encoder.onnx", out_dir / "yolow_text.npz"
    failures = []

    model, _ = clip.load("ViT-B/32", device="cpu", download_root=args.clip_dir)
    model = model.float().eval()

    class TextEncoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, tokens):
            f = self.m.encode_text(tokens)
            return f / f.norm(p=2, dim=-1, keepdim=True)

    enc = TextEncoder(model).eval()
    sample = clip.tokenize(["a photo of a cat", "dog"], truncate=True).long()   # this clip fork returns int32
    torch.onnx.export(enc, (sample,), str(onnx_path), input_names=["tokens"], output_names=["embeddings"],
                      dynamic_axes={"tokens": {0: "classes"}, "embeddings": {0: "classes"}}, opset_version=17)
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size:,} B)")

    bpe_path = os.path.join(os.path.dirname(clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
    merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")[1:49152 - 256 - 2 + 1]
    world = YOLOWorld(args.pt).model.eval()
    arrays = {"bpe_merges": np.array(merges)}
    for b in yt.GUIDE_BLOCKS:
        attn = world.model[b].attn
        arrays[f"model.{b}.attn.gl.weight"] = attn.gl.weight.detach().numpy()
        arrays[f"model.{b}.attn.gl.bias"] = attn.gl.bias.detach().numpy()
        arrays[f"model.{b}.attn.hc"] = np.array(attn.hc)
    head = world.model[22]
    arrays["contrastive_scale"] = np.array([float(h.logit_scale.exp()) for h in head.cv4], dtype=np.float64)
    arrays["contrastive_bias"] = np.array([float(h.bias) for h in head.cv4], dtype=np.float64)
    np.savez(npz_path, **arrays)
    print(f"wrote {npz_path} ({npz_path.stat().st_size:,} B): {len(merges)} merges, guides for blocks {yt.GUIDE_BLOCKS}")

    text = yt.YoloWorldText(onnx_path, npz_path)
    phrases = list(COCO_CLASSES) + TEST_PHRASES
    ours = text.tokenizer(phrases)
    theirs = clip.tokenize(phrases, truncate=True).numpy().astype(np.int64)
    bad = [p for p, a, b in zip(phrases, ours, theirs) if not np.array_equal(a, b)]
    print(f"tokenizer: {len(phrases) - len(bad)}/{len(phrases)} phrases give clip.tokenize's tokens" + (f"; differ: {bad}" if bad else ""))
    if bad:
        failures.append("tokenizer")

    with torch.no_grad():
        ref = enc(torch.from_numpy(theirs)).numpy()
    got = text.embed(phrases)
    diff = float(np.abs(got - ref).max())
    print(f"text encoder: ONNX against PyTorch on {len(phrases)} phrases, max |diff| {diff:.3e}")
    if diff > 1e-4:
        failures.append("encoder")

    coco = text.host_constants(text.embed(list(COCO_CLASSES)))
    inits = {t.name: numpy_helper.to_array(t) for t in onnx.load(args.cut_onnx).graph.initializer}
    for name, g in coco.items():
        d = float(np.abs(g - inits[name]).max())
        print(f"guide {name} {g.shape}: max |diff| against the exported initializer {d:.3e}")
        if d > 1e-4:
            failures.append(name)
    # npu.yolow's scales were written by hand with the first YOLO-World pipeline and differ from the checkpoint's
    # exp(logit_scale) by up to 2.8e-5 relative, for a reason not recorded; the bundle keeps the checkpoint's.
    rel = float(np.max(np.abs(arrays["contrastive_scale"] / np.asarray(yw.SCALES) - 1)))
    print(f"contrastive scale {arrays['contrastive_scale'].tolist()} (npu.yolow {list(yw.SCALES)}, largest relative "
          f"difference {rel:.1e}), bias {arrays['contrastive_bias'].tolist()} (npu.yolow {list(yw.BIASES)})")
    if rel > 1e-4 or not np.allclose(arrays["contrastive_bias"], yw.BIASES, atol=1e-5):
        failures.append("contrastive constants")
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
