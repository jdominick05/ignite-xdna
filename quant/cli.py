"""Ignition Alpha: inspect ONNX graphs, quantize folded ResNet with or without CLE, or AdaRound an emitted file."""
import argparse
from dataclasses import asdict
import importlib.abc
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

from . import __version__


class BlockProducerImports(importlib.abc.MetaPathFinder):
    """Quark is always refused; torch only outside the adaround subcommand."""

    def __init__(self, blocked=("quark", "torch")):
        self.blocked = tuple(blocked)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise ModuleNotFoundError(f"Ignition core forbids importing {fullname}")
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m quant", description=__doc__)
    parser.add_argument("--version", action="version", version=f"Ignition {__version__} (Alpha)")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Print a static fingerprint of any ONNX file; does not run a model")
    inspect.add_argument("models", type=Path, nargs="+")
    emit = commands.add_parser("quantize", help="Calibrate and emit XINT8 QDQ for folded ResNet")
    emit.add_argument("--in-model", type=Path, default=Path("models/resnet50_fp32.onnx"))
    emit.add_argument("--out", type=Path, required=True)
    cle = emit.add_mutually_exclusive_group(required=True)
    cle.add_argument("--cle", action="store_true", help="Apply the transcribed cross-layer equalization first")
    cle.add_argument("--no-cle", action="store_true", help="Calibrate the float export as exported")
    emit.add_argument("--calib-dir", type=Path, default=Path("data/calib"))
    emit.add_argument("--cfg-path", type=Path, default=Path("models/preprocess_config.json"))
    emit.add_argument("--limit", type=int, default=64)
    emit.add_argument("--scratch", type=Path, default=Path("scratch"))
    emit.add_argument("--scales-from", type=Path,
                      help="Replay a reference position table; skips independent calibration")
    ada = commands.add_parser("adaround", help="AdaRound weight rounding for an emitted XINT8 file (torch; Quark stays blocked)")
    ada.add_argument("--in-model", type=Path, default=Path("models/resnet50_fp32.onnx"),
                     help="The float export the base was quantized from (hash-checked against its sidecar)")
    ada.add_argument("--quant", type=Path, required=True, help="Emitted XINT8 model with its .quant.json sidecar")
    ada.add_argument("--out", type=Path, required=True)
    ada.add_argument("--calib-dir", type=Path, default=Path("data/calib"))
    ada.add_argument("--cfg-path", type=Path, default=Path("models/preprocess_config.json"))
    ada.add_argument("--data-size", type=int, default=1000, help="Quark's DataSize (images used, capped by the listing)")
    ada.add_argument("--iters", type=int, default=1000, help="Quark's NumIterations per layer")
    ada.add_argument("--seed", type=int, default=1705472343, help="Quark's FixedSeed")
    args = parser.parse_args(argv)
    # Install only while the command runs; importing the CLI has no global side effects.
    guard = BlockProducerImports(("quark",) if args.command == "adaround" else ("quark", "torch"))
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
                    graph = Graph.load(path, strict=False)
                    reports.append({"model": path.as_posix(), "sha256": file_hash(path),
                                    "export_contract": graph.contract_error or "ok",
                                    "onnx_checker": graph.checker_error or "ok",
                                    "node_order_changed_in_memory": graph.node_order_changed,
                                    "fingerprint": graph.fingerprint()})
                print(json.dumps({"producer": "Ignition", "version": __version__,
                                  "method": "Static inspection; contract violations reported, not enforced; no EP acceptance verdict",
                                  "models": reports}, indent=2))
                return
            if args.command == "adaround":
                from .adaround import FastFinetuneConfig, finetune
                from .cle import cross_layer_equalize
                from .qdq import read_pos_table
                if args.out.exists() or Path(str(args.out) + ".quant.json").exists():
                    parser.error("Choose a new output/sidecar name; existing artifacts are never overwritten")
                base_sidecar = Path(str(args.quant) + ".quant.json")
                provenance = json.loads(base_sidecar.read_text(encoding="utf-8"))
                if provenance["output_sha256"] != file_hash(args.quant):
                    parser.error("Base sidecar hash does not match the quantized model")
                if provenance["input_sha256"] != file_hash(args.in_model):
                    parser.error("Float model differs from the one the base was quantized from")
                if "calibration" not in provenance:
                    parser.error("AdaRound needs a base with an independent calibration listing in its sidecar")
                if "adaround" in provenance:
                    parser.error("The base was already finetuned; start from the emitted XINT8 file")
                cfg = json.loads(args.cfg_path.read_text(encoding="utf-8"))
                if cfg != provenance["preprocess"]:
                    parser.error("Preprocessing config differs from the base sidecar")
                float_graph = Graph.load(args.in_model)
                input_name = float_graph.model.graph.input[0].name
                listing = provenance["calibration"]["listing"]
                source = ImageFolderSource(args.calib_dir, cfg, len(listing), input_name)
                if [p.as_posix() for p in source.listing()] != listing:
                    parser.error("Calibration listing differs from the base sidecar")
                if provenance["cle"]:
                    # The float reference is the equalized float graph, as in Quark's post-process.
                    cross_layer_equalize(float_graph)
                float_graph.infer_shapes()
                quant_graph = Graph.load(args.quant)
                config = FastFinetuneConfig(DataSize=args.data_size, FixedSeed=args.seed, NumIterations=args.iters)
                print(f"Ignition {__version__}: AdaRound on {args.quant.as_posix()} "
                      f"({'CLE' if provenance['cle'] else 'no CLE'} base, {len(listing)} images)", flush=True)
                print("IMPORT_BLOCK_ACTIVE quark", flush=True)
                start = time.perf_counter()
                adaround_report = finetune(float_graph, quant_graph, source, config)
                quant_graph.model.producer_name = "Ignition"
                quant_graph.model.producer_version = __version__
                quant_graph.save(args.out)
                report = dict(provenance)
                report.update({
                    "producer": "Ignition", "producer_version": __version__,
                    "scope": provenance["scope"] + "_adaround", "mode": provenance["mode"] + "+adaround",
                    "base_model": args.quant.as_posix(), "base_sha256": file_hash(args.quant),
                    "adaround": asdict(adaround_report),
                    "positions": {name: asdict(tq) for name, tq in read_pos_table(quant_graph).items()},
                    "output_sha256": file_hash(args.out), "wall_seconds": time.perf_counter() - start,
                    "quark_imported": any(name == "quark" or name.startswith("quark.") for name in sys.modules),
                    "torch_imported": "torch" in sys.modules,
                })
                # This process's own versions; the base's stay under base_versions.
                report["base_versions"] = provenance["versions"]
                report["versions"] = {p: metadata.version(p) for p in ("numpy", "onnx", "onnxruntime", "torch")}
                Path(str(args.out) + ".quant.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                layers = asdict(adaround_report)["layers"]
                print("ADAROUND_REPORT", json.dumps({k: v for k, v in asdict(adaround_report).items() if k != "layers"}, indent=2))
                print("ADAROUND_LAYERS index name iterations early_stop changed max_lsb recons_before recons_after")
                for layer in layers:
                    print("ADAROUND_LAYER", layer["index"], layer["name"], layer["iterations"], layer["early_stop"],
                          layer["changed_elements"], layer["max_lsb"], f"{layer['recons_before']:f}", f"{layer['recons_after']:f}")
                print("ADAROUND_COMPLETE", json.dumps({"output_sha256": report["output_sha256"],
                                                       "wall_seconds": report["wall_seconds"],
                                                       "peak_rss_bytes": adaround_report.peak_rss_bytes,
                                                       "changed_elements": sum(l["changed_elements"] for l in layers)}))
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
            print(f"Ignition {__version__}: folded ResNet / {'CLE' if args.cle else 'no CLE'}", flush=True)
            print("IMPORT_BLOCK_ACTIVE quark torch", flush=True)
            report = quantize(args.in_model, args.out, scales_from=args.scales_from,
                              source=source, preprocess=cfg, scratch=args.scratch, cle=args.cle)
            print(json.dumps({k: v for k, v in report.items() if k not in ("positions", "calibration", "cle_report")}, indent=2))
            if "cle_report" in report:
                print("CLE_REPORT", json.dumps({k: v for k, v in report["cle_report"].items() if k != "scaled"}, indent=2))
            if "calibration" in report:
                print("CALIBRATION_REPORT", json.dumps({k: v for k, v in report["calibration"].items()
                                                        if k not in ("tensors", "emission_positions")}, indent=2))
            if args.scales_from:
                difference = graph_diff(Graph.load(args.out), Graph.load(args.scales_from))
                print("GRAPH_DIFF", json.dumps(asdict(difference), indent=2))
                print("GRAPH_DIFF_PASS", difference.ok())
                if not difference.ok():
                    raise SystemExit(1)
        except (OSError, ValueError, RuntimeError, KeyError, NotImplementedError, ValidationError) as exc:
            parser.exit(1, f"Ignition: {exc}\n")
    finally:
        sys.meta_path.remove(guard)
