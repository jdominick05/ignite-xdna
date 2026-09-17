#!/usr/bin/env python3
"""Offline gates of the sigmoid SiLU epilogue on one QDQ model (no device).

1. Lower the model with ``silu_sigmoid=True`` and build ``silu_sigmoid.reference_model``.
2. For bus.jpg, the first ``--coco`` images of ``--images`` and one uniform random input, every layer tensor of
   ``graph_reference.run_direct`` must equal ONNX Runtime's uint8 intermediate of the reference model.
3. Every layer's packets, replayed through the core emulator from the reference tensors of bus.jpg, must rebuild that
   layer's output byte for byte (``engine_schedule.emulate_layer``).

Exit 0 only if every comparison is exact.

    python tools/silu_sigmoid_gates.py models/yolov8n_cut_xint8.onnx --images data/coco/val2017 --coco 4
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402

from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import graph_ir, graph_reference as gr, silu_sigmoid  # noqa: E402


def float_input(ir, image: Path) -> np.ndarray:
    t = ir.tensors[ir.input]
    img = cv2.resize(cv2.imread(str(image)), (t.width, t.height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--image", default=str(ROOT / "assets" / "bus.jpg"))
    ap.add_argument("--images", default=None, help="a COCO image directory for --coco more inputs")
    ap.add_argument("--coco", type=int, default=0)
    ap.add_argument("--no-emulate", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    ir = graph_ir.lower_yolov8n(args.model, silu_sigmoid=True)
    ref = silu_sigmoid.reference_model(onnx.load(args.model))
    sig = [L for L in ir.layers if isinstance(L, graph_ir.ConvLayer) and L.sigmoid is not None]
    hs = [L for L in ir.layers if isinstance(L, graph_ir.ConvLayer) and L.hswish is not None]
    errs = Counter((L.sigmoid.max_error, round(L.sigmoid.mean_error, 3)) for L in sig)
    print(f"[gates] {Path(args.model).name}: {len(ir.layers)} layers, {len(sig)} with the sigmoid epilogue, "
          f"{len(hs)} with the HardSwish epilogue; per-layer output-LSB error against the exact quantized SiLU "
          f"(max, mean): {dict(sorted(errs.items()))}; lowered and reference model built in "
          f"{time.perf_counter() - t0:.1f} s", flush=True)
    if hs:
        print("[gates] FAIL: a SiLU kept the HardSwish epilogue")
        return 1

    inputs = [(Path(args.image).name, float_input(ir, Path(args.image)))]
    if args.images and args.coco:
        for p in sorted(Path(args.images).glob("*.jpg"))[:args.coco]:
            inputs.append((p.name, float_input(ir, p)))
    t = ir.tensors[ir.input]
    rng = np.random.default_rng(0)
    inputs.append(("uniform random", rng.random((3, t.height, t.width), dtype=np.float32)))
    names = [L.output for L in ir.layers]
    ok = True
    direct_bus = None
    for label, x in inputs:
        q = gr.quantize_input(x, t.scale, t.zero_point)
        direct = gr.run_direct(ir, q)
        ort = gr.ort_intermediates(ref, x[None], names)
        bad = [L.name for L in ir.layers
               if not np.array_equal(direct[L.output][:ir.tensors[L.output].channels], ort[L.output])]
        print(f"[gates] {label}: run_direct == ONNX Runtime on the reference model for "
              f"{len(ir.layers) - len(bad)}/{len(ir.layers)} layers{'' if not bad else ' MISMATCH ' + str(bad[:5])}",
              flush=True)
        ok &= not bad
        if direct_bus is None:
            direct_bus = direct

    if not args.no_emulate:
        t1 = time.perf_counter()
        ws = es.plan_workspace(ir)
        scheds, store = es.schedule_graph(ir, ws)
        ws_arr = ws.halo_fill()
        for name, arr in direct_bus.items():
            if name in ws.placements:
                ws.write_tensor(ws_arr, name, arr)
        bad = []
        for s in scheds:
            L = ir.layers[s.layer_index]
            c = ir.tensors[L.output].channels
            ws.write_tensor(ws_arr, L.output, np.zeros_like(direct_bus[L.output]))
            es.emulate_layer(s, store, ws_arr)
            n = int(np.sum(ws.read_tensor(ws_arr, L.output)[:c] != direct_bus[L.output][:c]))
            if n:
                bad.append((L.name, n))
        print(f"[gates] packet emulation of every layer == run_direct ({inputs[0][0]}): "
              f"{len(scheds) - len(bad)}/{len(scheds)} layers in {time.perf_counter() - t1:.0f} s"
              f"{'' if not bad else ' MISMATCH ' + str(bad[:5])}", flush=True)
        ok &= not bad
    print(f"[gates] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
