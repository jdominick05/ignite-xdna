"""Dump one InstanceNormalization node's real input, params and ORT output to .npy.

Runs in resnet_env17 (needs onnx + onnxruntime + the repo's npu package), NOT in
the mlir-aie ironenv. Picks the first InstanceNormalization node in
models/resnetv2_50x3_xint8.onnx whose input is [1, 32, L] for the requested L,
adds that node's input and output tensors to the graph outputs, runs the model
on the CPU EP for one real preprocessed image, and writes:

    <out-dir>/x.npy       node input,  float32 [1, 32, L]  (what CPU ORT sees)
    <out-dir>/y_ort.npy   node output, float32 [1, 32, L]  (what CPU ORT produces)
    <out-dir>/scale.npy   float32 [32]
    <out-dir>/bias.npy    float32 [32]
    <out-dir>/meta.json   node name, epsilon, image path

groupnorm.py --golden-dir <out-dir> then feeds x (rounded to bf16) through the
AIE kernel and compares against both a float64 reference and y_ort.

Usage (Git Bash, repo root):
    conda activate resnet_env17
    python kernels/groupnorm_bf16/extract_golden.py --L 301056 \
        --out-dir data/golden/instancenorm_L301056
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import numpy_helper  # noqa: E402

import npu.preprocess as preprocess  # noqa: E402
from npu.paths import MODELS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=str(MODELS / "resnetv2_50x3_xint8.onnx"))
    p.add_argument("--L", type=int, default=301056, help="flattened group length")
    p.add_argument("--image", default=str(ROOT / "data" / "eval" / "000000.jpg"))
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--input-size",
        type=int,
        default=448,
        help="model input resolution (resnetv2_50x3_bit is 448, not the 224 in "
        "models/preprocess_config.json)",
    )
    args = p.parse_args()

    model = onnx.load(args.model)
    g = model.graph
    shapes = {
        vi.name: tuple(d.dim_value for d in vi.type.tensor_type.shape.dim)
        for vi in list(g.value_info) + list(g.input) + list(g.output)
    }
    inits = {t.name: t for t in g.initializer}

    node = None
    for n in g.node:
        if n.op_type != "InstanceNormalization":
            continue
        if shapes.get(n.input[0], ()) == (1, 32, args.L):
            node = n
            break
    if node is None:
        sys.exit(f"no InstanceNormalization node with input shape (1, 32, {args.L})")

    eps = None
    for a in node.attribute:
        if a.name == "epsilon":
            eps = a.f
    # scale/bias are either plain initializers or (in this Quark-quantized
    # model) the outputs of DequantizeLinear nodes; fetch the latter from the
    # session by exposing them as graph outputs alongside the node's own
    # input and output.
    x_name, y_name = node.input[0], node.output[0]
    s_name, b_name = node.input[1], node.input[2]
    fetch = [x_name, y_name]
    for name, dims in ((x_name, [1, 32, args.L]), (y_name, [1, 32, args.L])):
        if name not in {o.name for o in g.output}:
            g.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, dims))
    for name in (s_name, b_name):
        if name in inits:
            continue
        fetch.append(name)
        if name not in {o.name for o in g.output}:
            g.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [32]))

    with open(MODELS / "preprocess_config.json") as f:
        cfg = json.load(f)
    cfg["input_size"] = [3, args.input_size, args.input_size]
    transform = preprocess.build_transform(cfg)
    img = transform(args.image)

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0].name
    outs = dict(zip(fetch, sess.run(fetch, {inp: img})))
    x, y = outs[x_name].astype(np.float32), outs[y_name].astype(np.float32)

    def const(name):
        if name in inits:
            return numpy_helper.to_array(inits[name]).astype(np.float32)
        return np.asarray(outs[name], dtype=np.float32).reshape(-1)

    scale, bias = const(s_name), const(b_name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "x.npy", x)
    np.save(out_dir / "y_ort.npy", y)
    np.save(out_dir / "scale.npy", scale)
    np.save(out_dir / "bias.npy", bias)
    meta = {
        "model": args.model,
        "node": node.name,
        "input_tensor": x_name,
        "output_tensor": y_name,
        "scale_tensor": s_name,
        "bias_tensor": b_name,
        "epsilon": eps,
        "image": args.image,
        "input_size": args.input_size,
        "shape": [1, 32, args.L],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Cancellation check for the kernel's E[x^2] - mean^2 variance: how big is
    # |mean| relative to std per group on this real tensor?
    xg = x.reshape(32, args.L).astype(np.float64)
    mean = xg.mean(axis=1)
    std = xg.std(axis=1)
    ratio = np.abs(mean) / np.maximum(std, 1e-12)
    print(f"node {node.name}: input {x_name} {x.shape}, epsilon {eps}")
    print(f"  |x| max {np.abs(xg).max():.4f}, per-group |mean|/std max {ratio.max():.3f}")
    print(f"  scale range [{scale.min():.4f}, {scale.max():.4f}], "
          f"bias range [{bias.min():.4f}, {bias.max():.4f}]")
    print(f"  ORT output |y| max {np.abs(y).max():.4f}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
