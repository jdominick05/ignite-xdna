#!/usr/bin/env python3
"""Ask the device, not the schedule: after one dispatch, does each reused workspace slot hold the tensor that
was supposed to write it last?

    bash scripts/research-iron.sh tools/prove_slot_cotenancy.py --container build/yolov8n.ignite \
        --model models/yolov8n_cut_xint8.onnx [--image assets/bus.jpg]

``plan_workspace`` reuses a freed workspace slot for the next tensor whose geometry matches
(ignite-xdna 6a620f0), giving co-tenants the same base while every tenant writes from the slot start.
So after a single dispatch only the last writer of a slot can still be read back, and
``tools/verify_engine_container.py`` reading every layer would call the rest MISMATCH.

This tool states which case it is, per slot. For every slot with more than one tenant it reads the
slot back and compares the earlier tenants' byte range against the *reference value of the slot's
final tenant*. Bytes equal to the successor means the device wrote what the plan said and the earlier
layer is simply unreadable; bytes matching neither means a real defect. Prints the counts and exits
non-zero if any slot holds neither.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

import verify_engine_container as verify  # noqa: E402  (reuses its input staging verbatim)
from ignite_xdna.compiler import engine_schedule as es, graph_ir, graph_reference as gr  # noqa: E402
from ignite_xdna.runtime.graph_session import GraphSession  # noqa: E402


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--container", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default=str(ROOT / "assets" / "bus.jpg"))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)

    ir = graph_ir.lower_yolov8n(args.model)
    ws = es.plan_workspace(ir)
    from ignite_xdna.compiler.serializer import IgniteModelReader  # noqa: E402
    task = IgniteModelReader(args.container).manifest.get("task", "detect")
    q_in = verify.quantized_input(ir, Path(args.image), task)
    direct = gr.run_direct(ir, q_in)

    tenants = {}
    for L in ir.layers:
        tenants.setdefault(ws.placements[L.output].base, []).append((L.index, L.output))
    shared = {b: t for b, t in tenants.items() if len(t) > 1}
    print(f"[cotenancy] {task} container: {len(tenants)} slot bases hold "
          f"{sum(len(t) for t in tenants.values())} layer tensors; {len(shared)} of those slots have "
          f"more than one tenant")

    bad = proven = 0
    sess = verify.session_for(task, args.container, args.device)
    try:
        verify.stage_input(sess, task, q_in)
        sess.dispatch()
        for b, tl in sorted(shared.items()):
            owner_idx, owner = tl[-1]
            c_owner = ir.tensors[owner].channels
            for idx, name in tl[:-1]:
                c = min(ir.tensors[name].channels, c_owner)
                got = sess.read_tensor(name)[:c]
                as_successor = int(np.sum(got != direct[owner][:c]))
                as_itself = int(np.sum(got != direct[name][:c]))
                hit = as_successor == 0
                proven += hit
                bad += not hit
                print(f"  L{idx:2d} {name[:44]:44s} slot last written by L{owner_idx:2d}  "
                      f"diff-as-successor {as_successor:8d}/{got.size}  "
                      f"diff-as-itself {as_itself:8d}  {'AS PLANNED' if hit else 'NEITHER'}", flush=True)
    finally:
        sess.close()

    checked = proven + bad
    print(f"[cotenancy] {proven}/{checked} reused tensors read back exactly as their slot's final tenant"
          f"{' -> every one, the earlier layers are unreadable, not wrong' if bad == 0 else ''}")
    if bad:
        print(f"[cotenancy] {bad} slots hold neither their own value nor their successor's", flush=True)
    print("[cotenancy] PASS" if bad == 0 else "[cotenancy] FAIL", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
