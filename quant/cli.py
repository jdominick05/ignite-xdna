"""Ignition Alpha: inspect ONNX graphs or quantize folded ResNet without CLE."""
import argparse
from dataclasses import asdict
import importlib.abc
import json
from pathlib import Path
import sys

from . import __version__


class BlockProducerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in ("quark", "torch"):
            raise ModuleNotFoundError(f"Ignition core forbids importing {fullname}")
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m quant", description=__doc__)
    parser.add_argument("--version", action="version", version=f"Ignition {__version__} (Alpha)")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Print a static fingerprint; does not run a model")
    inspect.add_argument("models", type=Path, nargs="+")
    emit = commands.add_parser("quantize", help="Calibrate and emit XINT8 QDQ for folded ResNet")
    emit.add_argument("--in-model", type=Path, default=Path("models/resnet50_fp32.onnx"))
    emit.add_argument("--out", type=Path, required=True)
    emit.add_argument("--no-cle", action="store_true", required=True,
                      help="Required acknowledgement: Alpha has no cross-layer equalization")
    emit.add_argument("--calib-dir", type=Path, default=Path("data/calib"))
    emit.add_argument("--cfg-path", type=Path, default=Path("models/preprocess_config.json"))
    emit.add_argument("--limit", type=int, default=64)
    emit.add_argument("--scratch", type=Path, default=Path("scratch"))
    emit.add_argument("--scales-from", type=Path,
                      help="Replay a reference position table; skips independent calibration")
    args = parser.parse_args(argv)
    # Install only while the command runs; importing the CLI has no global side effects.
    guard = BlockProducerImports()
    sys.meta_path.insert(0, guard)
    try:
        from onnx.checker import ValidationError
        from .graph import Graph
        from .quantize import file_hash, quantize
        from .sources import ImageFolderSource
        from .verify import graph_diff

        try:
            if args.command == "inspect":
                reports = []
                for path in args.models:
                    graph = Graph.load(path)
                    reports.append({"model": path.as_posix(), "sha256": file_hash(path),
                                    "node_order_changed_in_memory": graph.node_order_changed,
                                    "fingerprint": graph.fingerprint()})
                print(json.dumps({"producer": "Ignition", "version": __version__,
                                  "method": "Static inspection; no EP acceptance verdict",
                                  "models": reports}, indent=2))
                return
            if args.limit < 1:
                parser.error("--limit must be positive")
            if args.out.exists() or Path(str(args.out) + ".quant.json").exists():
                parser.error("Choose a new output/sidecar name; existing artifacts are never overwritten")
            source, cfg = None, None
            if args.scales_from is None:
                cfg = json.loads(args.cfg_path.read_text(encoding="utf-8"))
                graph = Graph.load(args.in_model)
                input_name = graph.model.graph.input[0].name
                if graph.value_shape(input_name) != (1, *cfg["input_size"]):
                    parser.error("Preprocessing config does not match model input shape")
                source = ImageFolderSource(args.calib_dir, cfg, args.limit, input_name)
            print(f"Ignition {__version__}: folded ResNet / no CLE", flush=True)
            print("IMPORT_BLOCK_ACTIVE quark torch", flush=True)
            report = quantize(args.in_model, args.out, scales_from=args.scales_from,
                              source=source, preprocess=cfg, scratch=args.scratch)
            print(json.dumps({k: v for k, v in report.items() if k not in ("positions", "calibration")}, indent=2))
            if "calibration" in report:
                print("CALIBRATION_REPORT", json.dumps({k: v for k, v in report["calibration"].items()
                                                        if k not in ("tensors", "emission_positions")}, indent=2))
            if args.scales_from:
                difference = graph_diff(Graph.load(args.out), Graph.load(args.scales_from))
                print("GRAPH_DIFF", json.dumps(asdict(difference), indent=2))
                print("GRAPH_DIFF_PASS", difference.ok())
                if not difference.ok():
                    raise SystemExit(1)
        except (OSError, ValueError, RuntimeError, KeyError, ValidationError) as exc:
            parser.exit(1, f"Ignition: {exc}\n")
    finally:
        sys.meta_path.remove(guard)
