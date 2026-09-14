#!/usr/bin/env python3
"""Run a graph-engine .ignite container on Device 0 and compare every layer tensor with the integer reference.

    bash scripts/research-iron.sh tools/verify_engine_container.py --container build/yolov8s.ignite \
        --model models/yolov8s_cut_xint8.onnx [--image assets/bus.jpg] [--iters 20]

The reference is ``graph_reference.run_direct`` on the same quantized input (itself checked
against ONNX Runtime's uint8 intermediates by the compiler tests). The input is staged into the
container's input plane already quantized, so the comparison isolates the device: every conv,
pool and residual tensor the workspace holds after one dispatch is read back and compared byte
for byte. ``--iters`` then times further dispatches of the same frame. Exit 0 only if every
layer is exact.

Detection containers take the image resized to the model input, RGB / 255; super-resolution
containers take it resized to the model input, RGB minus 128 (the models' own QuantizeLinear
then gives the uint8 input).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import graph_ir, graph_reference as gr  # noqa: E402
from ignite_xdna.compiler.serializer import IgniteModelReader  # noqa: E402


def quantized_input(ir, image: Path, task: str) -> np.ndarray:
    t = ir.tensors[ir.input]
    img = cv2.resize(cv2.imread(str(image)), (t.width, t.height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = rgb - 128.0 if task == "super_resolution" else rgb / 255.0
    q = gr.quantize_input(np.transpose(x, (2, 0, 1)), t.scale, t.zero_point)
    full = np.full((t.blocks * 8, t.height, t.width), t.zero_point, dtype=np.uint8)
    full[:t.channels] = q
    return full[:t.channels]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", required=True)
    ap.add_argument("--model", required=True, help="the QDQ ONNX model the container was compiled from")
    ap.add_argument("--image", default=str(ROOT / "assets" / "bus.jpg"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20, help="extra timed dispatches after the checked one")
    ap.add_argument("--json", default=None, help="write per-layer results and timings here")
    args = ap.parse_args()

    with IgniteModelReader(args.container) as reader:
        manifest = dict(reader.manifest)
    task = manifest.get("task", "detect")
    ir = graph_ir.lower_yolov8n(args.model)
    ws = es.plan_workspace(ir)
    ge = manifest["graph_engine"]
    if ws.nbytes != int(ge["workspace_bytes"]) or ir.input != ge["input_tensor"]:
        print(f"[verify] container plan differs from {args.model}: workspace {ge['workspace_bytes']} vs {ws.nbytes}")
        return 2
    q_in = quantized_input(ir, Path(args.image), task)
    t0 = time.perf_counter()
    direct = gr.run_direct(ir, q_in)
    print(f"[verify] {task} container {Path(args.container).name}: {len(ir.layers)} layers, reference in "
          f"{time.perf_counter() - t0:.1f} s", flush=True)

    from ignite_xdna.runtime.graph_session import GraphSession, is_graph_container  # noqa: E402
    session_cls = GraphSession
    if task != "detect":
        from ignite_xdna.runtime.graph_session import DenseGraphSession  # noqa: E402
        session_cls = DenseGraphSession
    sess = session_cls(args.container, device_index=args.device)
    results, timings = [], []
    try:
        p = sess.input_placement
        h = int(p["halo"])
        plane = sess._input_plane
        plane[h:h + p["height"], h:h + p["width"], :q_in.shape[0]] = np.transpose(q_in, (1, 2, 0))
        base = int(p["base"])
        if sess._ws_map is None:
            sess.bo_ws.write(plane, base)
        sess.bo_ws.sync(sess.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, sess._input_bytes, base)
        first_ms = sess.dispatch()
        exact = 0
        for L in ir.layers:
            c = ir.tensors[L.output].channels
            got = sess.read_tensor(L.output)[:c]
            nd = int(np.sum(got != direct[L.output][:c]))
            exact += nd == 0
            results.append({"index": L.index, "name": L.name, "mismatches": nd, "size": int(got.size)})
            print(f"  L{L.index:2d} {L.name:44s} {'EXACT' if nd == 0 else f'MISMATCH {nd}/{got.size}'}", flush=True)
        for _ in range(args.iters):
            timings.append(sess.dispatch())
    finally:
        sess.close()
    arr = np.asarray(timings) if timings else np.asarray([first_ms])
    print(f"[verify] {exact}/{len(ir.layers)} layers exact | first dispatch {first_ms:.3f} ms | dispatch mean "
          f"{arr.mean():.3f} ms, min {arr.min():.3f}, max {arr.max():.3f} over {arr.size}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"container": Path(args.container).name, "task": task,
                                               "layers_exact": exact, "layers": len(ir.layers),
                                               "first_dispatch_ms": first_ms, "dispatch_ms": arr.tolist(),
                                               "results": results}, indent=1), encoding="utf-8")
    print("[verify] PASS" if exact == len(ir.layers) else "[verify] FAIL", flush=True)
    return 0 if exact == len(ir.layers) else 1


if __name__ == "__main__":
    raise SystemExit(main())
