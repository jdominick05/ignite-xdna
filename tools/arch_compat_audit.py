#!/usr/bin/env python3
"""Which current detection architectures this engine could take, by op and by conv shape.

    python tools/arch_compat_audit.py yolo26n yolo12n yolo11n yolov8n

Builds each architecture from the YAML ultralytics ships, exports ONNX at 640 and scores every
node against what the graph engine accepts. STRUCTURAL only: the weights are random, so this
says what would have to lower, never how well it would score. Run it in resnet_env, which is
the environment with ultralytics.

What the engine takes (compiler/engine_schedule.conv_chunk_kind and graph_ir):

    1x1 stride 1 pad 0 | 3x3 stride 1 pad 1 | 3x3 stride 2 pad 1 | 5x5 stride 1 pad 2
    MaxPool, Add, Resize (x2), Concat, and the Mul/Sigmoid/HardSigmoid epilogues
    depthwise convolution (group == Cin == Cout); any other group, and any dilation, is refused

Everything else is a declared host segment - which costs an ONNX Runtime call per frame, as
YOLO11n's attention core does - or a compiler change.

The Div, Gather, Shape, Sub and one Softmax that every YOLO here reports are its DFL and
anchor decode tail, which the head cut removes before quantization; YOLOv8n lowers today with
exactly that set. So read a family's extra MatMul and Softmax as its attention, and judge it
against YOLO11n, which the engine already runs with one host call per frame.
"""
import argparse
import collections
import contextlib
import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ACCEPTED_CONV = {(1, 1, 0), (3, 1, 1), (3, 2, 1), (5, 1, 2)}
ENGINE_OPS = {"Conv", "MaxPool", "Add", "Resize", "Concat", "Mul", "Sigmoid", "HardSigmoid",
              "HardSwish", "Split", "Relu", "QuantizeLinear", "DequantizeLinear", "Constant",
              "Identity", "Reshape", "Transpose", "Slice"}
DECODE_TAIL = {"Div", "Gather", "Shape", "Sub"}      # removed by the head cut


def export(name, out_dir):
    import onnx
    from ultralytics import YOLO
    f = out_dir / f"{name}.onnx"
    if not f.exists():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            p = YOLO(f"{name}.yaml").export(format="onnx", imgsz=640, opset=17,
                                            simplify=False, dynamic=False, verbose=False)
        Path(p).replace(f)
    return onnx.load(str(f)).graph


def audit(graph):
    ops = collections.Counter(n.op_type for n in graph.node)
    convs, bad, groups, dilated = collections.Counter(), collections.Counter(), 0, 0
    for n in graph.node:
        if n.op_type != "Conv":
            continue
        a = {x.name: x for x in n.attribute}
        k = tuple(a["kernel_shape"].ints)[0] if "kernel_shape" in a else 0
        s = tuple(a["strides"].ints)[0] if "strides" in a else 1
        p0 = tuple(a["pads"].ints)[0] if "pads" in a else 0
        convs[(k, s, p0)] += 1
        groups += (a["group"].i if "group" in a else 1) != 1
        dilated += (tuple(a["dilations"].ints)[0] if "dilations" in a else 1) != 1
        if (k, s, p0) not in ACCEPTED_CONV:
            bad[(k, s, p0)] += 1
    foreign = {k: v for k, v in ops.items() if k not in ENGINE_OPS}
    attention = {k: v for k, v in foreign.items() if k not in DECODE_TAIL}
    return dict(sorted(convs.items())), dict(sorted(bad.items())), groups, dilated, \
        dict(sorted(foreign.items())), dict(sorted(attention.items())), len(graph.node)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="ultralytics YAML stems, e.g. yolo26n")
    ap.add_argument("--out", type=Path, default=Path("scratch") / "arch_audit")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        try:
            g = export(name, args.out)
        except Exception as e:
            print(f"{name:10} EXPORT FAILED: {type(e).__name__}: {str(e)[:90]}")
            continue
        convs, bad, groups, dilated, foreign, attn, total = audit(g)
        print(f"\n=== {name}: {total} nodes, {sum(convs.values())} convs")
        print(f"    conv (k, stride, pad)  : {convs}")
        print(f"    UNSUPPORTED conv shapes: {bad or 'none'}")
        print(f"    grouped {groups}, dilated {dilated}")
        print(f"    outside the vocabulary : {foreign or 'none'}")
        print(f"    of which NOT decode    : {attn or 'none'}   <- the real host burden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())