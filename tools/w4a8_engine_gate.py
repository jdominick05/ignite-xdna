#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline engine gate for a W4A8 form-b file from tools/w4a8_emulate.py (no device).

The model is lowered with the engine's default lowering (``graph_ir.lower_yolov8n``). For bus.jpg,
the first ``--coco`` images of ``--images`` and one uniform random input, every layer tensor of the
integer oracle ``graph_reference.run_direct`` must equal ONNX Runtime's uint8 intermediate of the
same file under ORT_DISABLE_ALL. This is the same comparison tools/silu_sigmoid_gates.py makes, with
the model itself as the reference.

Refuses any Conv whose weight DequantizeLinear has a scale of more than one element: graph_ir.scale_zp
reads element 0 of a vector and would silently lower a per-channel (form-a, *_ortonly) file wrong.

Exit 0 only if every layer is exact on every input.

    python tools/w4a8_engine_gate.py scratch/int4_w4a8/yolov8n_w4a8_e1.onnx --coco 4
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402

from ignite_xdna.compiler import graph_ir, graph_reference as gr  # noqa: E402


def float_input(ir, image: Path) -> np.ndarray:
    t = ir.tensors[ir.input]
    img = cv2.resize(cv2.imread(str(image)), (t.width, t.height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


def refuse_per_channel(model: onnx.ModelProto) -> int:
    inits = {t.name: t for t in model.graph.initializer}
    by_output = {o: n for n in model.graph.node for o in n.output}
    convs = [n for n in model.graph.node if n.op_type == "Conv"]
    for conv in convs:
        dq = by_output.get(conv.input[1])
        if dq is None or dq.op_type != "DequantizeLinear":
            raise SystemExit(f"{conv.name}: weight is not fed by a DequantizeLinear")
        for name in dq.input[1:]:
            if numpy_helper.to_array(inits[name]).size != 1:
                raise SystemExit(f"refusing {conv.name}: weight DQ {name} has more than one element "
                                 f"(per-channel); graph_ir.scale_zp would read only element 0")
    return len(convs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--image", default=str(ROOT / "assets" / "bus.jpg"))
    ap.add_argument("--images", default=str(ROOT / "data" / "coco" / "val2017"),
                    help="a COCO image directory for --coco more inputs")
    ap.add_argument("--coco", type=int, default=4)
    args = ap.parse_args()

    shown = args.model.replace("\\", "/")
    digest = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    n_conv = refuse_per_channel(onnx.load(args.model))
    t0 = time.perf_counter()
    ir = graph_ir.lower_yolov8n(args.model)
    shifts = [L.shift_out for L in ir.layers if isinstance(L, graph_ir.ConvLayer)]
    print(f"[gate] {shown}  sha256 {digest}")
    print(f"[gate] {n_conv} Convs, every weight DQ per-tensor; lowered to {len(ir.layers)} layers "
          f"(shift_out {min(shifts)}..{max(shifts)} over {len(shifts)} ConvLayers) in "
          f"{time.perf_counter() - t0:.1f} s", flush=True)

    inputs = [(Path(args.image).name, float_input(ir, Path(args.image)))]
    if args.coco:
        found = sorted(Path(args.images).glob("*.jpg"))[:args.coco]
        if len(found) != args.coco:
            raise SystemExit(f"found {len(found)} images, wanted {args.coco}")
        inputs += [(p.name, float_input(ir, p)) for p in found]
    t = ir.tensors[ir.input]
    rng = np.random.default_rng(0)
    inputs.append(("uniform random", rng.random((3, t.height, t.width), dtype=np.float32)))
    names = [L.output for L in ir.layers]
    ok = True
    for label, x in inputs:
        q = gr.quantize_input(x, t.scale, t.zero_point)
        direct = gr.run_direct(ir, q)
        ort = gr.ort_intermediates(args.model, x[None], names)
        bad = [L.name for L in ir.layers
               if not np.array_equal(direct[L.output][:ir.tensors[L.output].channels], ort[L.output])]
        print(f"[gate] {label}: run_direct == ONNX Runtime (ORT_DISABLE_ALL) for "
              f"{len(ir.layers) - len(bad)}/{len(ir.layers)} layers{'' if not bad else ' MISMATCH ' + str(bad[:5])}",
              flush=True)
        ok &= not bad
    print(f"[gate] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
