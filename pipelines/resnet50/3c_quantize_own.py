"""Quantize a folded ResNet using the owned producer (no Quark import)."""
import argparse
from dataclasses import asdict
import importlib.abc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class BlockProducerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in ("quark", "torch"):
            raise ModuleNotFoundError(f"Owned core forbids importing {fullname}")
        return None


sys.meta_path.insert(0, BlockProducerImports())

from quant.graph import Graph
from quant.quantize import quantize
from quant.sources import ImageFolderSource
from quant.verify import graph_diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-model", type=Path, default=Path("models/resnet50_fp32.onnx"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scales-from", type=Path, help="Phase 1 replay; omit for independent calibration")
    parser.add_argument("--no-cle", action="store_true", required=True,
                        help="Required: this implementation supports folded ResNet without CLE")
    parser.add_argument("--calib-dir", type=Path, default=Path("data/calib"))
    parser.add_argument("--cfg-path", type=Path, default=Path("models/preprocess_config.json"))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--scratch", type=Path, default=ROOT / "scratch")
    args = parser.parse_args()
    source, cfg = None, None
    if args.scales_from is None:
        cfg = json.loads(args.cfg_path.read_text(encoding="utf-8"))
        graph = Graph.load(args.in_model)
        input_name = graph.model.graph.input[0].name
        if graph.value_shape(input_name) != (1, *cfg["input_size"]):
            parser.error("Preprocessing config does not match model input shape")
        source = ImageFolderSource(args.calib_dir, cfg, args.limit, input_name)
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


if __name__ == "__main__":
    main()
