"""Build the fresh no-CLE Quark oracle for the owned quantizer's Phase 1 gate.

Quark is imported only in this reference process. The owned producer runs in a
separate process and never imports Quark. Use scripts/quant-reference.sh for guards.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quant.graph import Graph
from quant.sources import ImageFolderSource, as_reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-model", type=Path, default=Path("models/resnet50_fp32.onnx"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, default=Path("data/calib"))
    parser.add_argument("--cfg-path", type=Path, default=Path("models/preprocess_config.json"))
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()
    sidecar = args.out.with_suffix(".reference.json")
    if args.out.exists() or sidecar.exists():
        parser.error("Reference output or sidecar already exists; choose a new name")
    graph = Graph.load(args.in_model)
    cfg = json.loads(args.cfg_path.read_text(encoding="utf-8"))
    input_name = graph.model.graph.input[0].name
    if graph.value_shape(input_name) != (1, *cfg["input_size"]):
        parser.error("Preprocessing config does not match model input shape")
    source = ImageFolderSource(args.calib_dir, cfg, args.limit, input_name)
    with args.in_model.open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "machine": "Desktop 2 / Ryzen 7 8700G", "include_cle": False,
        "float_model_sha256": model_hash, "preprocess": cfg,
        "listing": [p.as_posix() for p in source.listing()],
        "versions": {p: metadata.version(p) for p in ("amd-quark", "onnx", "onnxruntime", "numpy")},
    }
    print(json.dumps(report, indent=2), flush=True)
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, get_default_config
    quant_config = get_default_config("XINT8")
    quant_config.include_cle = False
    start = time.perf_counter()
    ModelQuantizer(Config(global_quant_config=quant_config)).quantize_model(
        str(args.in_model), str(args.out), as_reader(source))
    report["wall_seconds"] = time.perf_counter() - start
    if not args.out.is_file():
        raise RuntimeError("Quark returned without writing the reference")
    with args.out.open("rb") as stream:
        report["model_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("REFERENCE COMPLETE", json.dumps({k: report[k] for k in ("model_sha256", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
