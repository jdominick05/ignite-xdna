#!/usr/bin/env python3
"""Per-layer NPU time of a graph-engine YOLO container, by dispatching its instruction stream one layer at a time.

    bash scripts/research-iron.sh tools/engine_layer_timing.py --onnx models/yolov8s_cut_xint8.onnx \\
        --container build/yolov8s.ignite --image ../Ignition/examples/assets/bus.jpg [--rounds 30] [--nop] [--json out]

The engine has no device timer in its stream, so this times from the host. The container's ONNX is lowered again
and must reproduce the container (workspace size and every weight-packet byte), or the tool refuses. Each layer's
task count then cuts `insts.bin` at its layer barrier (`engine_sequence.split_instruction_stream`; the emitter
retires every task at each barrier, so the pieces are position independent). Each piece, the whole stream and an
empty stream (header only, no ops) get one pre-built XRT run each, the same start/wait path `GraphSession.dispatch`
uses.

One round = the whole stream once, then every layer piece in order, then the empty stream `--empty` times; rounds
repeat `--rounds` times after `--warmup` rounds, on the same staged frame. Reported per layer: the median dispatch
ms, and that minus the empty-dispatch median (`net_ms`, DERIVED: it assumes each dispatch pays the empty one's
fixed cost once). Check: the sum of net layer times against the whole stream less one empty dispatch.

Exactness gate: after the per-layer sequence of each round, the heads read back must equal the heads of the
whole-stream dispatch of the same round, byte for byte.

`--nop` repeats everything on a copy whose weight packets are all NOP (`tools/engine_dispatch_floor.py`): the
same stream and data movement with the arithmetic skipped. Per layer, real minus NOP is that layer's core compute,
and NOP net is its transport plus task overhead.

Run so far: YOLOv8s merged container with --silu-sigmoid (Desktop 2, 2026-10-13); the same and its --stream-share
build, same sitting (Desktop 2, 2026-10-13, branch block5-stream-share).
"""
from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def relower(onnx: Path, manifest: dict):
    from ignite_xdna.compiler.engine_compile import profile_for_manifest
    from ignite_xdna.compiler.graph_ir import lower_yolov8n
    ge = manifest["graph_engine"]
    sched = profile_for_manifest(manifest).schedule
    ir = lower_yolov8n(onnx, silu_sigmoid=ge.get("silu") == "sigmoid4")
    ws = sched.plan_workspace(ir, reuse=ge.get("workspace_reuse", True))
    if ge.get("stream_share"):
        # Stream-shared containers (compiler/stream_share.py). The layer pieces still run in frame order, so the
        # MemTile chains' period phase is the whole frame's; the frame's period padding is in the last piece.
        from ignite_xdna.compiler.stream_share import schedule_stream_shared
        _, scheds, store, _ = schedule_stream_shared(ir, ws)
    else:
        scheds, store = sched.schedule_graph(ir, ws)
    return ir, ws, scheds, store


def pieces_for(container: Path, onnx: Path):
    from ignite_xdna.compiler.engine_sequence import program_task_count, split_instruction_stream
    from ignite_xdna.compiler.serializer import IgniteModelReader
    with IgniteModelReader(container) as r:
        manifest = dict(r.manifest)
        insts = r.get_blob_bytes("insts.bin")
        wp = r.get_blob_bytes("wpackets.bin")
    ge = manifest["graph_engine"]
    if ge.get("segments"):
        raise ValueError("container has host segments; this tool times monolithic containers only")
    ir, ws, scheds, store = relower(onnx, manifest)
    if ws.nbytes != manifest.get("workspace_bytes", ge.get("workspace_bytes")):
        raise ValueError(f"re-lowered workspace {ws.nbytes} differs from the container's")
    if store.blob().tobytes() != wp:
        raise ValueError("re-lowered weight packets differ from the container's")
    counts = [sum(program_task_count(p) for p in s.programs) for s in scheds]
    pieces = split_instruction_stream(insts, counts)
    names = [(s.layer_index, s.name) for s in scheds]
    return manifest, insts, pieces, names


def empty_stream(insts: bytes) -> bytes:
    major, minor, _, _ = struct.unpack("<4I", insts[:16])
    return struct.pack("<4I", major, minor, 0, 16)


def time_container(container: Path, pieces, insts: bytes, image: Path, rounds: int, warmup: int, n_empty: int):
    from ignite_xdna.runtime.graph_session import GraphSession
    s = GraphSession(container, device_index=0)
    try:
        pyxrt = s.harness.pyxrt
        done = getattr(getattr(pyxrt, "ert_cmd_state", None), "ERT_CMD_STATE_COMPLETED", None)
        keep = []

        def mk(blob: bytes):
            bo, n = s.harness.create_instruction_bo_from_bytes(blob)
            run = pyxrt.run(s.harness.kernel)
            for i, arg in enumerate((3, bo, n, s.bo_ws, s.bo_wp)):
                run.set_arg(i, arg)
            keep.append(bo)
            return run

        def go(run) -> float:
            t0 = time.perf_counter()
            run.start()
            st = run.wait(10000)
            dt = (time.perf_counter() - t0) * 1e3
            if st != done and str(st) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
                raise RuntimeError(f"dispatch ended in {st}")
            return dt

        whole = mk(insts)
        runs = [mk(p) for p in pieces]
        empty = mk(empty_stream(insts))
        s.stage_image(cv2.imread(str(image)))
        t_whole, t_empty, t_layer = [], [], [[] for _ in runs]
        mismatches = 0
        for k in range(warmup + rounds):
            w = go(whole)
            ref = np.array(s.read_heads(unswizzle=False), copy=True)
            lt = [go(r) for r in runs]
            got = np.array(s.read_heads(unswizzle=False), copy=True)
            e = [go(empty) for _ in range(n_empty)]
            if not np.array_equal(ref, got):
                mismatches += 1
            if k >= warmup:
                t_whole.append(w)
                t_empty.extend(e)
                for i, v in enumerate(lt):
                    t_layer[i].append(v)
        return {"whole": t_whole, "empty": t_empty, "layers": t_layer, "mismatched_rounds": mismatches,
                "rounds": warmup + rounds}
    finally:
        keep.clear()
        s.close()


def summarise(tag, names, res, manifest):
    med = statistics.median
    e = med(res["empty"])
    w = med(res["whole"])
    rows = []
    mlayers = {L["index"]: L for L in manifest["graph_engine"]["layers"]}
    for (idx, name), v in zip(names, res["layers"]):
        m = med(v)
        L = mlayers.get(idx, {})
        rows.append({"arm": tag, "index": idx, "name": name, "ms": round(m, 4), "net_ms": round(m - e, 4),
                     "spread_ms": round(float(np.percentile(v, 90) - np.percentile(v, 10)), 4),
                     "rounds_in_layer": L.get("rounds"), "packets": L.get("packets"), "w_fills": L.get("w_fills")})
    net = sum(r["net_ms"] for r in rows)
    summ = {"arm": tag, "whole_ms_median": round(w, 4), "empty_ms_median": round(e, 4),
            "empty_ms_p10_p90": [round(float(np.percentile(res["empty"], 10)), 4),
                                 round(float(np.percentile(res["empty"], 90)), 4)],
            "sum_layer_ms": round(sum(r["ms"] for r in rows), 4), "sum_net_ms": round(net, 4),
            "whole_minus_empty_ms": round(w - e, 4), "net_over_whole_less_empty": round(net / (w - e), 4),
            "mismatched_rounds": res["mismatched_rounds"], "rounds_total": res["rounds"], "layers": len(rows)}
    return rows, summ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--container", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--empty", type=int, default=5, help="empty dispatches per round")
    ap.add_argument("--nop", action="store_true", help="also time an all-NOP-weights copy of the container")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    manifest, insts, pieces, names = pieces_for(a.container, a.onnx)
    print(f"[layers] {a.container.name}: {len(pieces)} layer pieces, {len(insts)} B stream, "
          f"pieces sum {sum(len(p) - 16 for p in pieces) + 16} B", flush=True)
    arms = [("real", a.container)]
    if a.nop:
        sys.path.insert(0, str(ROOT / "tools"))
        from engine_dispatch_floor import write_nop_copy
        nop = a.container.with_name(a.container.stem + "_nop.ignite")
        write_nop_copy(a.container, nop)
        arms.append(("nop", nop))
    out = {"container": a.container.name, "arms": {}}
    for tag, path in arms:
        res = time_container(path, pieces, insts, a.image, a.rounds, a.warmup, a.empty)
        rows, summ = summarise(tag, names, res, manifest)
        for r in rows:
            print("LAYER_TIME " + json.dumps(r), flush=True)
        print("SUMMARY " + json.dumps(summ), flush=True)
        out["arms"][tag] = {"summary": summ, "layers": rows}
    if a.json:
        a.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
