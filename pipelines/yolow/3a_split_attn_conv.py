"""
YOLO-World v2 step 3a: split each C2fAttn output convolution in two, so XINT8 gives each half its own weight scale.

Why:
  /model.{12,15,18,21}/cv2/conv/Conv is a 1x1 convolution over Concat(a, b, c, attn). Its weights reading
  the text attention branch are up to about 4x larger than the rest, and one per-tensor power-of-two weight
  scale then rounds the small weights away: that is the whole XINT8 collapse (docs/BENCHMARKS.md, "YOLO-World
  v2 on the graph engine"). A 1x1 convolution is linear in its input channels, so

      Conv(Concat(a, b, c, attn); W, B) == Conv(Concat(a, b, c); W[:, :k], B) + Conv(attn; W[:, k:])

  with k the channels of a, b and c. The Add feeds the original activation. This is an exact FP32 rewrite;
  the script checks it by running both models on the same input in ONNX Runtime.

  On the graph engine the two halves are ordinary convolutions and the Add with the activation after it is a
  residual packet carrying HardSwish, so the attention core is the only host region left.

    conda activate resnet_env
    python pipelines/yolow/3a_split_attn_conv.py
    python pipelines/yolow/3b_quantize_cut.py --in models/yolov8s-worldv2_cut_split.onnx

Writes models/yolov8s-worldv2_cut_split.onnx.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS

BLOCKS = ("12", "15", "18", "21")


def split(m: onnx.ModelProto):
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    by_out = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    inferred = onnx.shape_inference.infer_shapes(m)
    shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
              for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)}
    new_nodes, done = [], []
    for n in g.node:
        blk = next((b for b in BLOCKS if n.name == f"/model.{b}/cv2/conv/Conv"), None)
        if blk is None:
            new_nodes.append(n)
            continue
        cat = by_out[n.input[0]]
        if cat.op_type != "Concat" or len(consumers[cat.output[0]]) != 1:
            raise ValueError(f"{n.name}: input is not a Concat read only by it")
        attn = cat.input[-1]
        if not attn.startswith(f"/model.{blk}/attn/"):
            raise ValueError(f"{n.name}: last Concat input {attn} is not the attention output")
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if list(attrs.get("kernel_shape", [])) != [1, 1] or int(attrs.get("group", 1)) != 1:
            raise ValueError(f"{n.name}: the split is exact only for a 1x1, group 1 convolution")
        k = sum(shapes[i][1] for i in cat.input[:-1])
        w = numpy_helper.to_array(inits[n.input[1]])
        pre = f"/model.{blk}/cv2/split"
        t_w_abc, t_w_attn = f"{pre}.abc.weight", f"{pre}.attn.weight"
        g.initializer.extend([numpy_helper.from_array(np.ascontiguousarray(w[:, :k]), t_w_abc),
                              numpy_helper.from_array(np.ascontiguousarray(w[:, k:]), t_w_attn)])
        cat_abc = helper.make_node("Concat", list(cat.input[:-1]), [f"{pre}/Concat_abc_output_0"],
                                   name=f"{pre}/Concat_abc", axis=1)
        conv_abc = helper.make_node("Conv", [cat_abc.output[0], t_w_abc, n.input[2]], [f"{pre}/abc/Conv_output_0"],
                                    name=f"{pre}/abc/Conv", **attrs)
        conv_attn = helper.make_node("Conv", [attn, t_w_attn], [f"{pre}/attn/Conv_output_0"],
                                     name=f"{pre}/attn/Conv", **attrs)
        add = helper.make_node("Add", [conv_abc.output[0], conv_attn.output[0]], [n.output[0]], name=f"{pre}/Add")
        new_nodes = [x for x in new_nodes if x is not cat]
        new_nodes.extend([cat_abc, conv_abc, conv_attn, add])
        done.append((blk, k, w.shape[1] - k, float(np.abs(w[:, :k]).max()), float(np.abs(w[:, k:]).max())))
    if len(done) != len(BLOCKS):
        raise ValueError(f"split {len(done)} of {len(BLOCKS)} convolutions")
    del g.node[:]
    g.node.extend(new_nodes)
    used = {i for n in g.node for i in n.input}
    keep = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(MODELS / "yolov8s-worldv2_cut.onnx"))
    ap.add_argument("--out", dest="dst", default=str(MODELS / "yolov8s-worldv2_cut_split.onnx"))
    args = ap.parse_args()

    m = onnx.load(args.src)
    for blk, k, ka, wa, wb in split(m):
        print(f"/model.{blk}/cv2: {k} + {ka} input channels, max |w| {wa:.4f} and {wb:.4f}")
    onnx.checker.check_model(m)
    onnx.save(m, args.dst)
    print(f"wrote {args.dst}")

    import onnxruntime as ort
    a = ort.InferenceSession(args.src, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(args.dst, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)
    feed = {a.get_inputs()[0].name: x}
    for o, ya, yb in zip(a.get_outputs(), a.run(None, feed), b.run(None, feed)):
        err = float(np.sum((ya.astype(np.float64) - yb) ** 2))
        sqnr = 10 * np.log10(float(np.sum(ya.astype(np.float64) ** 2)) / err) if err else float("inf")
        print(f"  {o.name}: SQNR {sqnr:.1f} dB against the unsplit model")


if __name__ == "__main__":
    main()
