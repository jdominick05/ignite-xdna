"""Fingerprint a model's graph-engine schedule under one checkout's code: placements, every column program
(DMA patterns, weight items) and the weight-packet store, with workspace reuse on and off.

    python tools/schedule_fingerprint.py <checkout root> <model.onnx> [<model.onnx> ...]

Run once per checkout and compare the printed digests: equal digests mean the two versions lay out and
schedule the model identically, which is everything the instruction stream is emitted from.
"""
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root / "src"), str(root)]
import ignite_xdna  # noqa: E402

assert str(Path(ignite_xdna.__file__).resolve()).startswith(str(root)), ignite_xdna.__file__
from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import graph_ir  # noqa: E402


def digest(model: str, reuse: bool) -> str:
    ir = graph_ir.lower_yolov8n(model)
    ws = es.plan_workspace(ir, reuse=reuse)
    scheds, store = es.schedule_graph(ir, ws)
    hsh = hashlib.sha256()
    hsh.update(repr(ws.nbytes).encode())
    for name in sorted(ws.placements):
        p = ws.placements[name]
        hsh.update(repr((name, p.base, p.halo, p.height, p.width, p.blocks, p.planes, p.halo_value, p.dtype,
                         p.band_rows, p.nbytes)).encode())
    for s in scheds:
        hsh.update(repr((s.layer_index, s.name, s.rounds, s.packets, s.w_fills)).encode())
        for prog in s.programs:
            hsh.update(repr(prog).encode())
    hsh.update(store.blob().tobytes())
    return f"{hsh.hexdigest()[:16]} ({len(ir.layers)} layers, ws {ws.nbytes} B)"


print(f"code from {ignite_xdna.__file__}")
for m in sys.argv[2:]:
    for reuse in (True, False):
        print(f"{Path(m).name} reuse={reuse}: {digest(m, reuse)}", flush=True)
