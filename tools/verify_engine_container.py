#!/usr/bin/env python3
"""Run a graph-engine .ignite container on Device 0 and compare its layer tensors with the integer reference.

    bash scripts/research-iron.sh tools/verify_engine_container.py --container build/yolov8s.ignite \
        --model models/yolov8s_cut_xint8.onnx [--image assets/bus.jpg] [--iters 20]

The reference is ``graph_reference.run_direct`` on the same quantized input (itself checked
against ONNX Runtime's uint8 intermediates by the compiler tests). The input is staged into the
container's input plane already quantized, so the comparison isolates the device: every conv,
pool and residual tensor the workspace holds after one dispatch is read back and compared byte
for byte. ``--iters`` then times further dispatches of the same frame.

Not every layer can be read back. ``plan_workspace`` reuses a workspace slot as soon as its last
reader has run, giving co-tenant tensors the same base, so after one dispatch only the last tenant
of each slot still holds its own bytes. Each layer is therefore reported as EXACT/MISMATCH when it
owns its slot or SLOT ok/BAD when it does not (its bytes checked against the tensor that legitimately
wrote the slot last), and exit 0 means every readable layer is exact and every reused slot holds its
planned final tenant. On yolov8n that is 25 readable of 66 layers, all exact.

To check every layer of a lowering, build and verify with workspace reuse off -- ``reuse=False`` gives
each tensor its own slot, so all of them are readable and all of them must read EXACT.

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
from ignite_xdna.compiler.engine_compile import check_kernel_covers_packets  # noqa: E402
from ignite_xdna.compiler import graph_ir, graph_reference as gr  # noqa: E402
from ignite_xdna.compiler.serializer import IgniteModelReader  # noqa: E402


def quantized_input(ir, image: Path, task: str) -> np.ndarray:
    t = ir.tensors[ir.input]
    if task == "classify" and t.channels > 4:
        # A head-only container takes an already-pooled feature vector, which the harness supplies at
        # (0, 0). A whole classifier's input is an image (3 channels), so it goes through the path below.
        full = np.full((t.blocks * 8, t.height, t.width), t.zero_point, dtype=np.uint8)
        rng = np.random.RandomState(42)
        full[:t.channels, 0, 0] = rng.randint(0, 256, size=t.channels, dtype=np.uint8)
        return full[:t.channels]
    img = cv2.resize(cv2.imread(str(image)), (t.width, t.height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = rgb - 128.0 if task == "super_resolution" else rgb / 255.0
    q = gr.quantize_input(np.transpose(x, (2, 0, 1)), t.scale, t.zero_point)
    full = np.full((t.blocks * 8, t.height, t.width), t.zero_point, dtype=np.uint8)
    full[:t.channels] = q
    return full[:t.channels]


def slot_final_tenants(ws, ir) -> dict:
    """``slot base -> (layer index, tensor)`` of the last layer to write each workspace slot.

    ``plan_workspace`` with liveness reuse gives co-tenant tensors the same base and every tenant writes
    from the slot start, so after one dispatch a slot holds only its last writer. A tensor that is not its
    slot's final tenant cannot be read back at all -- that is the readback method's limit, not a device
    defect, and calling it MISMATCH is how a correct device gets reported as a regression."""
    return {ws.placements[L.output].base: (L.index, L.output) for L in ir.layers}


def session_for(task: str, container: str, device_index: int = 0):
    """The session that runs this container: dense planes for super-resolution, a pooled-vector egress
    for classification, the channel-block layout for everything else."""
    from ignite_xdna.runtime import graph_session as gs  # noqa: PLC0415 - opens the device
    if task == "super_resolution":
        return gs.DenseGraphSession(container, device_index=device_index)
    if task == "classify":
        return gs.ClassificationSession(container, device_index=device_index)
    return gs.GraphSession(container, device_index=device_index)


def stage_input(sess, task: str, q_in) -> None:
    """Place one already-quantized frame where the container expects it, without dispatching."""
    if task == "classify":
        sess.stage_quantized(q_in)
        return
    p = sess.input_placement
    h, base = int(p["halo"]), int(p["base"])
    plane = sess._input_plane
    plane[h:h + p["height"], h:h + p["width"], :q_in.shape[0]] = np.transpose(q_in, (1, 2, 0))
    if sess._ws_map is None:
        sess.bo_ws.write(plane, base)
    sess.bo_ws.sync(sess.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                    sess._input_bytes, base)


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
        wpackets = reader.get_blob_bytes("wpackets.bin")
    task = manifest.get("task", "detect")
    ge = manifest["graph_engine"]
    try:
        check_kernel_covers_packets(ge, wpackets)
    except ValueError as exc:
        raise SystemExit(f"[verify] REFUSED: {exc}")
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
    if ge.get("fuse_stencils"):
        # Re-apply the pass the container was built with, or the plan check below compares a fused
        # container against an unfused lowering and refuses it - which is how a wrong fused layer could
        # never be localized on silicon. The reference for each layer stays the model's own semantics:
        # graph_reference runs a fused pair as the two convolutions it replaces.
        from ignite_xdna.compiler import passes
        ir = passes.match_stencil_fusion(ir)
        print(f"[verify] container declares fuse_stencils: re-lowered to {len(ir.layers)} layers "
              f"({sum(1 for L in ir.layers if isinstance(L, graph_ir.FusedConvLayer))} fused pairs)")
    if silu == "sigmoid4":
        print("[verify] SiLU through the sigmoid epilogue: the reference is silu_sigmoid.reference_model of the model")
    # Plan the workspace the way this container was built. A reuse-free container gives every tensor its own
    # slot, so every layer stays readable after one dispatch and all of them are checkable; the default plan
    # recycles slots and only each slot's final tenant is readable. See ignite-compile --no-workspace-reuse.
    reuse = bool(ge.get("workspace_reuse", True))
    ws = es.plan_workspace(ir, reuse=reuse)
    if not reuse:
        print("[verify] workspace reuse off: every tensor owns a slot, so every layer below is readable")
    if ws.nbytes != int(ge["workspace_bytes"]) or ir.input != ge["input_tensor"]:
        print(f"[verify] container plan differs from {args.model}: workspace {ge['workspace_bytes']} vs {ws.nbytes}")
        return 2
    q_in = quantized_input(ir, Path(args.image), task)
    t0 = time.perf_counter()
    direct = gr.run_direct(ir, q_in)
    print(f"[verify] {task} container {Path(args.container).name}: {len(ir.layers)} layers, reference in "
          f"{time.perf_counter() - t0:.1f} s", flush=True)

    sess = session_for(task, args.container, args.device)
    results, timings, host_timings = [], [], []
    try:
        if constants:
            print(f"[verify] replaced in host segments: {sess.set_host_constants(constants)}", flush=True)
        stage_input(sess, task, q_in)
        first_ms = sess.dispatch()
        first_host_ms = sess.last_host_ms
        # plan_workspace's liveness reuse hands co-tenant tensors the same slot base and every tenant
        # writes from the slot start. This tool reads all tensors after ONE dispatch, so a tensor whose
        # slot a later layer reused cannot read back as its own value, however correctly the device
        # computed it -- only the last tenant of a slot is still there. Classify before judging: a
        # "MISMATCH" on a reused slot is this method's blind spot, not a device defect. Measured on
        # yolov8n: 25 of 66 layers own their slot, and all 41 remaining read back as their successor.
        final_of = slot_final_tenants(ws, ir)
        exact = observable = slot_ok = reused = 0
        for L in ir.layers:
            name, t = L.output, ir.tensors[L.output]
            got = sess.read_tensor(name)[:t.channels]
            host = isinstance(L, graph_ir.HostLayer)
            owner_idx, owner = final_of[ws.placements[name].base]
            if owner == name:
                observable += 1
                nd = int(np.sum(got != direct[name][:t.channels]))
                exact += nd == 0
                verdict = "EXACT" if nd == 0 else f"MISMATCH {nd}/{got.size}"
            else:
                reused += 1
                c = min(t.channels, ir.tensors[owner].channels)
                nd = int(np.sum(got[:c] != direct[owner][:c]))
                slot_ok += nd == 0
                verdict = ("SLOT ok" if nd == 0 else f"SLOT BAD {nd}/{c}") + f", holds L{owner_idx} now"
            results.append({"index": L.index, "name": L.name, "mismatches": nd, "size": int(got.size),
                            "readable": owner == name, "slot_owner": owner,
                            **({"host": True} if host else {})})
            print(f"  L{L.index:2d} {'HOST ' if host else ''}{L.name:44s} {verdict}", flush=True)
        for _ in range(args.iters):
            timings.append(sess.dispatch())
            host_timings.append(sess.last_host_ms)
    finally:
        sess.close()
    arr = np.asarray(timings) if timings else np.asarray([first_ms])
    harr = np.asarray(host_timings) if host_timings else np.asarray([first_host_ms])
    if reused:
        print(f"[verify] {exact}/{observable} readable layers exact; {slot_ok}/{reused} reused slots hold "
              f"their planned final tenant | {len(ir.layers)} layers total | first dispatch {first_ms:.3f} ms | "
              f"dispatch mean {arr.mean():.3f} ms, min {arr.min():.3f}, max {arr.max():.3f} over {arr.size}",
              flush=True)
        print(f"[verify] {reused} of {len(ir.layers)} layers are NOT checkable by post-dispatch readback: "
              f"workspace reuse let a later layer write their slot. That is this tool's coverage limit, "
              f"not a device finding -- build with ignite-compile --no-workspace-reuse to check "
              f"every layer of the lowering.", flush=True)
    else:
        print(f"[verify] {exact}/{len(ir.layers)} layers exact | first dispatch {first_ms:.3f} ms | dispatch mean "
              f"{arr.mean():.3f} ms, min {arr.min():.3f}, max {arr.max():.3f} over {arr.size}", flush=True)
    if host_regions:
        print(f"[verify] {len(host_regions)} host segment(s) {host_regions}: host mean {harr.mean():.3f} ms, "
              f"min {harr.min():.3f}, max {harr.max():.3f} (NPU dispatch above excludes it)", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"container": Path(args.container).name, "task": task,
                                               "layers_exact": exact, "layers": len(ir.layers),
                                               "layers_readable": observable, "slots_reused": reused,
                                               "slots_reused_ok": slot_ok,
                                               "first_dispatch_ms": first_ms, "dispatch_ms": arr.tolist(),
                                               "host_regions": host_regions, "host_ms": harr.tolist(),
                                               "results": results}, indent=1), encoding="utf-8")
    ok = exact == observable and slot_ok == reused
    print("[verify] PASS" if ok else "[verify] FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
