"""Run the first N YOLOv8n layers through the convolution engine on silicon.

  python tests/test_engine_graph.py --compile  --layers 3     build insts for layers 0..2 (ironenv)
  python tests/test_engine_graph.py --hardware --layers 3     run on Device 0, compare every layer's tensor
  python tests/test_engine_graph.py --offline  --layers 3     packet-emulate the same schedule (any env)

The workspace, the static weight packets and the expected tensors come from
the compiler modules; the hardware run leaves every intermediate tensor in the
workspace, so each layer is checked byte for byte against the direct integer
reference (which itself matches ONNX Runtime's uint8 intermediates).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.compiler import engine_emulator as em  # noqa: E402
from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import graph_ir, graph_reference as gr  # noqa: E402

MODEL = ROOT / "models" / "yolov8n_cut_xint8.onnx"
BUS = ROOT / "assets" / "bus.jpg"
COLS = 4


def load_image_chw(path=BUS, size=640):
    import cv2
    img = cv2.imread(str(path))
    img = cv2.resize(img, (size, size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


OPTIONS = {"merge_fills": True, "pair_drains": False, "weight_runs": False, "retire_batch": 1, "w_depth": 1}


def prepare(n_layers: int):
    ir = graph_ir.lower_yolov8n(MODEL)
    ws = es.plan_workspace(ir)
    scheds, store = es.schedule_graph(ir, ws, merge_fills=OPTIONS["merge_fills"], pair_drains=OPTIONS["pair_drains"],
                                      weight_runs=OPTIONS["weight_runs"])
    chw = load_image_chw()
    q_in = gr.quantize_input(chw, ir.tensors[ir.input].scale)
    ws_arr = ws.halo_fill()
    ws.write_tensor(ws_arr, ir.input, q_in)
    return ir, ws, scheds[:n_layers], store, ws_arr, q_in


def sequence_body_for(scheds):
    from ignite_xdna.compiler.engine_sequence import SequenceEmitter
    from kernels.aie2.conv_engine import design as eng

    def body(ws, wp):
        emitter = SequenceEmitter(ws, wp, eng.WS_BYTES, eng.WP_BYTES, {c: eng.fifo_names(c) for c in range(COLS)})
        for s in scheds:
            # run_column_programs retires every task at its end: a layer barrier.
            emitter.run_column_programs(s.programs, bd_budget=14, retire_batch=OPTIONS["retire_batch"])
    return body


def compile_graph(build_dir: Path, n_layers: int):
    import aie.iron as iron
    from aie.iron.device import NPU1
    from aie.utils.compile.utils import compile_mlir_module
    from kernels.aie2.conv_engine import design as eng

    iron.set_current_device(NPU1())
    ir, ws, scheds, store, ws_arr, q_in = prepare(n_layers)
    build_dir.mkdir(parents=True, exist_ok=True)
    np.save(build_dir / "ws_init.npy", ws_arr)
    np.save(build_dir / "wp.npy", store.blob())
    meta = {"layers": [{"index": s.layer_index, "name": s.name, "output": ir.layers[s.layer_index].output,
                        "rounds": s.rounds, "packets": s.packets, "w_fills": s.w_fills} for s in scheds],
            "ws_bytes": int(ws.nbytes), "wp_bytes": int(store.nbytes),
            "placements": {n: {"base": p.base, "halo": p.halo, "height": p.height, "width": p.width,
                               "blocks": p.blocks, "planes": p.planes} for n, p in ws.placements.items()}}
    t0 = time.perf_counter()
    program = eng.build_program(iron.get_current_device(), sequence_body_for(scheds), w_depth=OPTIONS["w_depth"])
    module = program.resolve_program()
    t_resolve = time.perf_counter() - t0
    meta["options"] = dict(OPTIONS)
    work = build_dir / "design.prj"
    if work.exists():
        import shutil
        shutil.rmtree(work)  # never reuse a stale engine.o
    work.mkdir(exist_ok=True)
    (build_dir / "design.mlir").write_text(str(module), encoding="utf-8")
    compile_mlir_module(module, insts_path=build_dir / "insts.bin", xclbin_path=build_dir / "design.xclbin",
                        work_dir=work, device=iron.get_current_device())
    dt = time.perf_counter() - t0
    meta["compile_seconds"] = round(dt, 1)
    meta["resolve_seconds"] = round(t_resolve, 1)
    meta["insts_bytes"] = (build_dir / "insts.bin").stat().st_size
    meta["xclbin_sha256"] = hashlib.sha256((build_dir / "design.xclbin").read_bytes()).hexdigest()
    (build_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[compile] {n_layers} layers: {sum(s.rounds for s in scheds)} rounds, insts {meta['insts_bytes']} B, "
          f"resolve {t_resolve:.1f} s, total {dt:.1f} s")
    return meta


def compare_layers(ir, ws, scheds, ws_got, direct):
    results = []
    for s in scheds:
        L = ir.layers[s.layer_index]
        got = ws.read_tensor(ws_got, L.output)[:ir.tensors[L.output].channels]
        ref = direct[L.output]
        nd = int(np.sum(got != ref))
        results.append((s.layer_index, s.name, nd, ref.size))
    return results


def run_offline(n_layers: int):
    ir, ws, scheds, store, ws_arr, q_in = prepare(n_layers)
    direct = gr.run_direct(ir, q_in, stop_after=n_layers - 1)
    for s in scheds:
        es.emulate_layer(s, store, ws_arr)
    ok = True
    for idx, name, nd, total in compare_layers(ir, ws, scheds, ws_arr, direct):
        print(f"  L{idx:2d} {name:38s} {'EXACT' if nd == 0 else f'MISMATCH {nd}/{total}'}")
        ok &= nd == 0
    return ok


def run_hardware(build_dir: Path, n_layers: int, device_idx: int = 0, iters: int = 2, per_layer: bool = False,
                 timeout_ms: int = 20000):
    from ignite_xdna.compiler.engine_sequence import program_task_count, split_instruction_stream
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    ir, ws, scheds, store, ws_init, q_in = prepare(n_layers)
    direct = gr.run_direct(ir, q_in, stop_after=n_layers - 1)
    wp = store.blob()
    insts = (build_dir / "insts.bin").read_bytes()
    harness = XrtSiliconHarness(device_idx=device_idx)
    all_ok = True
    try:
        harness.load_xclbin(str(build_dir / "design.xclbin"))
        bo_ws = harness.create_host_bo(int(ws.nbytes), 3)
        bo_wp = harness.create_host_bo(int(wp.size), 4)
        bo_wp.write(wp.tobytes(), 0)
        bo_wp.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        if per_layer:
            counts = [sum(program_task_count(p) for p in s.programs) for s in scheds]
            pieces = split_instruction_stream(insts, counts)
            streams = [harness.create_instruction_bo_from_bytes(p) for p in pieces]
        else:
            streams = [harness.create_instruction_bo_from_bytes(insts)]
        for it in range(iters):
            bo_ws.write(ws_init.tobytes(), 0)
            bo_ws.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            per_layer_ms = []
            t_all = time.perf_counter()
            for li, (bo_instr, n_instr) in enumerate(streams):
                t0 = time.perf_counter()
                run, state = harness.dispatch_kernel(bo_instr, n_instr, bo_ws, bo_wp, timeout_ms=timeout_ms)
                dt = (time.perf_counter() - t0) * 1e3
                per_layer_ms.append(dt)
                if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
                    where = f"layer {scheds[li].layer_index} ({scheds[li].name})" if per_layer else "whole stream"
                    print(f"[hardware] iter {it}: dispatch state {state} at {where} after {dt:.1f} ms")
                    return False
            total_ms = (time.perf_counter() - t_all) * 1e3
            bo_ws.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            got = np.frombuffer(bo_ws.read(int(ws.nbytes), 0), dtype=np.uint8).copy()
            print(f"[hardware] iter {it}: {len(streams)} dispatch(es), {total_ms:.3f} ms for {n_layers} layers")
            for k, (idx, name, nd, total) in enumerate(compare_layers(ir, ws, scheds, got, direct)):
                tms = f" {per_layer_ms[k]:7.3f} ms" if per_layer else ""
                print(f"  L{idx:2d} {name:38s}{tms} {'EXACT' if nd == 0 else f'MISMATCH {nd}/{total}'}")
                all_ok &= nd == 0
        bo_ws = bo_wp = None
        streams = None
    finally:
        harness.close()
    return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--per-layer", action="store_true", help="split the stream and dispatch one layer at a time")
    parser.add_argument("--timeout-ms", type=int, default=20000)
    parser.add_argument("--no-merge-fills", action="store_true")
    parser.add_argument("--pair-drains", action="store_true", help="experimental: drain two rounds per task")
    parser.add_argument("--weight-runs", action="store_true", help="experimental: stream a round's weights in one task")
    parser.add_argument("--retire-batch", type=int, default=1)
    parser.add_argument("--w-depth", type=int, default=1)
    args = parser.parse_args()
    OPTIONS.update(merge_fills=not args.no_merge_fills, pair_drains=args.pair_drains,
                   weight_runs=args.weight_runs, retire_batch=args.retire_batch, w_depth=args.w_depth)
    build_dir = Path(args.build_dir or (ROOT / "build" / "conv_engine" / f"graph{args.layers}")).resolve()
    ok = True
    if args.offline:
        ok &= run_offline(args.layers)
    if args.compile:
        compile_graph(build_dir, args.layers)
    if args.hardware:
        ok &= run_hardware(build_dir, args.layers, args.device, args.iters, per_layer=args.per_layer,
                           timeout_ms=args.timeout_ms)
    print("[graph] PASS" if ok else "[graph] FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
