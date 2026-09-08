"""Build a fresh Quark oracle (no CLE by default, --cle for the default preset, --adaround for XINT8_ADAROUND) for the owned quantizer's gates.

Quark is imported only in this reference process. The owned producer runs in a
separate process and never imports Quark. Use scripts/quant-reference.sh for guards.
"""
import argparse
import copy
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

import onnx

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
    parser.add_argument("--cle", action="store_true", help="Keep Quark's default include_cle=True")
    parser.add_argument("--adaround", action="store_true",
                        help="Use the XINT8_ADAROUND preset: FastFinetune (AdaRound, cpu) after the same calibration")
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
        "machine": "Desktop 2 / Ryzen 7 8700G", "include_cle": args.cle, "include_fast_ft": args.adaround,
        "float_model_sha256": model_hash, "preprocess": cfg,
        "listing": [p.as_posix() for p in source.listing()],
        "versions": {p: metadata.version(p) for p in ("amd-quark", "onnx", "onnxruntime", "numpy")},
    }
    print(json.dumps(report, indent=2), flush=True)
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, get_default_config
    quant_config = get_default_config("XINT8_ADAROUND" if args.adaround else "XINT8")
    quant_config.include_cle = args.cle
    if args.adaround:
        # Provenance: Quark's DEFAULT_ADAROUND_PARAMS is a shared module-level dict; copy, never mutate it.
        report["fast_finetune"] = copy.deepcopy(quant_config.extra_options["FastFinetune"])
        report["versions"]["torch"] = metadata.version("torch")
    if args.cle:
        # Provenance only: the ordered pattern list Quark's matcher produces on this export.
        from quark.onnx.algorithm.cle.equalization import Equalization
        from quark.onnx.quantizers.registry import NPUCnnRegistry, QDQRegistry, QLinearOpsRegistry
        op_types = sorted(set(QLinearOpsRegistry) | set(QDQRegistry) | set(NPUCnnRegistry))
        patterns = Equalization(onnx.load(args.in_model), op_types, [], []).get_cle_pattern_pair()
        report["cle_patterns"] = [[p[1].name, p[-1].name, list(p[0])] for p in patterns]
        print("CLE_PATTERNS", len(patterns), flush=True)
    start = time.perf_counter()
    ModelQuantizer(Config(global_quant_config=quant_config)).quantize_model(
        str(args.in_model), str(args.out), as_reader(source))
    report["wall_seconds"] = time.perf_counter() - start
    import psutil
    memory = psutil.Process().memory_info()
    report["peak_rss_bytes"] = getattr(memory, "peak_wset", None)
    if not args.out.is_file():
        raise RuntimeError("Quark returned without writing the reference")
    with args.out.open("rb") as stream:
        report["model_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("REFERENCE COMPLETE", json.dumps({k: report[k] for k in ("model_sha256", "wall_seconds", "peak_rss_bytes")}), flush=True)


if __name__ == "__main__":
    main()
