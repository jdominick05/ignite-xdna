#!/usr/bin/env python3
"""Cut a detector's decode tail: the graph's outputs become its raw head convolutions.

    python tools/cut_detect_head.py IN.onnx OUT.onnx --heads A,B,C,D,E,F
    python tools/cut_detect_head.py IN.onnx OUT.onnx --find

``--find`` walks back from the graph output through the decode ops and prints the producers it
stops at, which are the head convolutions to pass to ``--heads``. Order matters: a decoder
reads the heads positionally.

pipelines/yolov8n/1b_cut_head.py does this for YOLOv8 with the names hard-coded in
npu.yolo.HEAD_OUTS. This is the same operation for a family whose head is shaped differently -
YOLO26, for instance, has no DFL, so its box branch is four channels rather than sixty-four and
its tail is anchor decode alone.
"""
import argparse
import collections
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import onnx
from onnx import shape_inference
from onnx.utils import Extractor

TAIL = {"Reshape", "Softmax", "Transpose", "Slice", "Div", "Sub", "Add", "Mul", "Concat",
        "Sigmoid", "Shape", "Gather", "Unsqueeze", "Squeeze", "Split", "Constant", "Cast",
        "Range", "Expand", "Tile", "ScatterND", "TopK", "ReduceMax", "ArgMax", "Where",
        "Greater", "Less", "Equal", "Not", "Pad", "Flatten", "Identity", "Exp", "Clip", "Neg",
        # An end-to-end, NMS-free head (YOLO26 exported with nms=False still has one) selects
        # its top-k boxes in the graph: TopK picks the indices, GatherElements applies them and
        # Mod turns a flat index back into a class. All three are decode, not computation.
        "GatherElements", "Mod"}


def find_heads(g):
    producer = {o: n for n in g.node for o in n.output}
    seen, heads, stack = set(), [], [g.output[0].name]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        n = producer.get(t)
        if n is None:
            continue
        if n.op_type in TAIL:
            stack.extend(n.input)
        else:
            heads.append((n.op_type, n.name, t))
    return sorted(heads, key=lambda h: h[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path, nargs="?")
    ap.add_argument("--heads", default=None, help="comma-separated head output tensors, in decode order")
    ap.add_argument("--input-name", default="images")
    ap.add_argument("--find", action="store_true", help="print the head producers and stop")
    args = ap.parse_args()

    m = shape_inference.infer_shapes(onnx.load(str(args.src)))
    if args.find or not args.heads:
        for op, name, t in find_heads(m.graph):
            print(f"{op:8} {name:48} -> {t}")
        if args.find:
            return 0
        raise SystemExit("pass --heads with the tensors above, in decode order")

    if args.dst is None:
        raise SystemExit("an output path is required unless --find")
    cut = Extractor(m).extract_model([args.input_name], args.heads.split(","))
    onnx.save(cut, str(args.dst))
    g = shape_inference.infer_shapes(onnx.load(str(args.dst))).graph
    ops = collections.Counter(n.op_type for n in g.node)
    vi = {v.name: v for v in list(g.value_info) + list(g.output)}
    print(f"{len(g.node)} nodes (was {len(m.graph.node)})")
    for o in g.output:
        d = [x.dim_value for x in vi[o.name].type.tensor_type.shape.dim]
        print(f"  out {o.name:52} {d}")
    print(f"  ops: {dict(sorted(ops.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())