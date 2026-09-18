#!/usr/bin/env python3
"""Compile model-zoo XINT8 classifiers with the graph engine and report where each one stops.

    python tools/sweep_model_zoo_classifiers.py [models/a.onnx models/b.onnx ...]

With no arguments it sweeps the XINT8 classification models present under ``models/``. Each model
goes through the real compiler stages -- ``graph_ir.lower_yolov8n``, then ``plan_workspace`` and
``schedule_graph`` -- and the first refusal is printed verbatim. No device, no IRON build, no
container is written: this measures which graphs the compiler accepts, which is a different question
from what any of them costs or how accurate it is.

A model is "SCHEDULED" when a transaction schedule exists for it. ``--build`` instead runs the full
``ignite-compile`` for each survivor, which needs the mlir-aie-iron environment.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ignite_xdna.compiler import graph_ir, engine_schedule as es  # noqa: E402
from ignite_xdna.compiler.engine_compile import graph_task  # noqa: E402

# Families that carry a terminal classification head, in the order a reader would ask about them.
DEFAULT_GLOBS = ("resnet*_xint8*.onnx", "resnetv2*_xint8*.onnx", "wide_resnet*_xint8*.onnx",
                 "resnext*_xint8*.onnx", "densenet*_xint8.onnx", "regnetx*_xint8.onnx",
                 "mobilenetv2*_xint8*.onnx", "mobilevit*_xint8.onnx", "*-cls*_xint8.onnx")
SKIP_WORDS = ("adaround", "b2", "b4", "r128", "r160", "r224", "r288", "r384", "accept",
              "hybrid", "ignition", "quark", "fp32", "int8", "a8w8", "a16w8", "stock")


def _tokens(stem: str) -> set:
    # Token, not substring: "xint8" must not be read as "int8", or the filter would drop every model.
    return {t for t in stem.replace("-", "_").split("_") if t}


def pick_models(globs) -> list:
    out = []
    for pattern in globs:
        for p in sorted((ROOT / "models").glob(pattern)):
            toks = _tokens(p.stem.lower())
            if any(w in toks for w in SKIP_WORDS):
                continue
            if p not in out:
                out.append(p)
    return out


def try_compile(path: Path) -> tuple:
    """(stage reached, layer count, verdict string). stage is one of lower/schedule/task/ok."""
    try:
        ir = graph_ir.lower_yolov8n(str(path))
    except Exception as exc:  # noqa: BLE001 - the refusal IS the measurement
        return ("lower", 0, f"{type(exc).__name__}: {' '.join(str(exc).split())[:150]}")
    try:
        ws = es.plan_workspace(ir)
        scheds, store = es.schedule_graph(ir, ws)
    except Exception as exc:  # noqa: BLE001
        return ("schedule", len(ir.layers), f"{type(exc).__name__}: {' '.join(str(exc).split())[:150]}")
    try:
        task = graph_task(ir)
    except Exception as exc:  # noqa: BLE001
        task = f"task? {type(exc).__name__}"
    hosts = sum(1 for L in ir.layers if isinstance(L, graph_ir.HostLayer))
    return ("ok", len(ir.layers), f"SCHEDULED task={task} {hosts} host layer(s) "
                                  f"workspace {ws.nbytes / 1e6:.1f} MB")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile model-zoo classifiers and report refusals.")
    parser.add_argument("models", nargs="*", type=Path, help="ONNX files (default: the sweep list)")
    args = parser.parse_args(argv)

    models = args.models or pick_models(DEFAULT_GLOBS)
    if not models:
        print("no classifier models found under models/", file=sys.stderr)
        return 1
    print(f"# graph-engine compile sweep over {len(models)} XINT8 classifiers (offline, no device)")
    print(f"# host {platform.node()}  python {sys.version.split()[0]}  onnxruntime-free lowering only")
    print(f"# stages: lower_yolov8n -> plan_workspace -> schedule_graph\n")
    width = max(len(p.name) for p in models) + 2
    ok = 0
    for path in models:
        t0 = time.time()
        stage, layers, verdict = try_compile(path)
        ok += stage == "ok"
        print(f"{path.name:{width}s} {stage:9s} {layers:3d}L  {verdict}  ({time.time() - t0:.0f}s)",
              flush=True)
    print(f"\n# {ok}/{len(models)} reached a schedulable graph")
    return 0


if __name__ == "__main__":
    sys.exit(main())
