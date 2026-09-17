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
    ap.add_argument("--host-constants", default=None, metavar="NPZ",
                    help="replace these initializers (npz keys are their names) in the host segments and in the "
                         "reference model, e.g. YOLO-World text guides for another vocabulary")
    args = ap.parse_args()

    with IgniteModelReader(args.container) as reader:
        manifest = dict(reader.manifest)
    task = manifest.get("task", "detect")
    ge = manifest["graph_engine"]
    host_regions = [s["name"] for s in ge.get("segments", []) if s["kind"] == "host"]
    model = args.model
    constants = {}
    if args.host_constants:
        import onnx
        from onnx import numpy_helper
        with np.load(args.host_constants) as npz:
            constants = {k: npz[k] for k in npz.files}
        model = onnx.load(args.model)
        found = []
        for t in model.graph.initializer:
            if t.name in constants:
                t.CopyFrom(numpy_helper.from_array(constants[t.name], t.name))
                found.append(t.name)
        if sorted(found) != sorted(constants):
            print(f"[verify] {args.model} has no initializers {sorted(set(constants) - set(found))}")
            return 2
        # The lowering reads the stored intermediate shapes; ONNX Runtime only warns that the attention outputs'
        # stored class dimension no longer matches.
        print(f"[verify] host constants from {Path(args.host_constants).name}: "
              f"{', '.join(f'{k} {constants[k].shape}' for k in sorted(constants))}", flush=True)
    silu = ge.get("silu", "hardsigmoid")
    if silu not in ("hardsigmoid", "sigmoid4"):
        print(f"[verify] container SiLU form {silu!r} is unknown to this verifier")
        return 2
    ir = graph_ir.lower_yolov8n(model, host_regions=host_regions, silu_sigmoid=silu == "sigmoid4")
    if silu == "sigmoid4":
        print("[verify] SiLU through the sigmoid epilogue: the reference is silu_sigmoid.reference_model of the model")
    ws = es.plan_workspace(ir)
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
    if task == "super_resolution":
        from ignite_xdna.runtime.graph_session import DenseGraphSession  # noqa: E402
        session_cls = DenseGraphSession
    sess = session_cls(args.container, device_index=args.device)
    results, timings, host_timings = [], [], []
    try:
        if constants:
            print(f"[verify] replaced in host segments: {sess.set_host_constants(constants)}", flush=True)
        p = sess.input_placement
        h = int(p["halo"])
        plane = sess._input_plane
        plane[h:h + p["height"], h:h + p["width"], :q_in.shape[0]] = np.transpose(q_in, (1, 2, 0))
        base = int(p["base"])
        if sess._ws_map is None:
            sess.bo_ws.write(plane, base)
        sess.bo_ws.sync(sess.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, sess._input_bytes, base)
        first_ms = sess.dispatch()
        first_host_ms = sess.last_host_ms
        exact = 0
        for L in ir.layers:
            c = ir.tensors[L.output].channels
            got = sess.read_tensor(L.output)[:c]
            nd = int(np.sum(got != direct[L.output][:c]))
            exact += nd == 0
            host = isinstance(L, graph_ir.HostLayer)
            results.append({"index": L.index, "name": L.name, "mismatches": nd, "size": int(got.size),
                            **({"host": True} if host else {})})
            print(f"  L{L.index:2d} {'HOST ' if host else ''}{L.name:44s} "
                  f"{'EXACT' if nd == 0 else f'MISMATCH {nd}/{got.size}'}", flush=True)
        for _ in range(args.iters):
            timings.append(sess.dispatch())
            host_timings.append(sess.last_host_ms)
    finally:
        sess.close()
    arr = np.asarray(timings) if timings else np.asarray([first_ms])
    harr = np.asarray(host_timings) if host_timings else np.asarray([first_host_ms])
    print(f"[verify] {exact}/{len(ir.layers)} layers exact | first dispatch {first_ms:.3f} ms | dispatch mean "
          f"{arr.mean():.3f} ms, min {arr.min():.3f}, max {arr.max():.3f} over {arr.size}", flush=True)
    if host_regions:
        print(f"[verify] {len(host_regions)} host segment(s) {host_regions}: host mean {harr.mean():.3f} ms, "
              f"min {harr.min():.3f}, max {harr.max():.3f} (NPU dispatch above excludes it)", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"container": Path(args.container).name, "task": task,
                                               "layers_exact": exact, "layers": len(ir.layers),
                                               "first_dispatch_ms": first_ms, "dispatch_ms": arr.tolist(),
                                               "host_regions": host_regions, "host_ms": harr.tolist(),
                                               "results": results}, indent=1), encoding="utf-8")
    print("[verify] PASS" if exact == len(ir.layers) else "[verify] FAIL", flush=True)
    return 0 if exact == len(ir.layers) else 1


if __name__ == "__main__":
    raise SystemExit(main())
