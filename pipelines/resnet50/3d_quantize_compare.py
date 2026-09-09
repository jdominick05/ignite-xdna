"""Build a fresh Quark oracle (no CLE by default, --cle for the default preset, --adaround for XINT8_ADAROUND) for the owned quantizer's gates.

The family is read from the float export: folded ResNet uses the classification
preprocessing config, a head-cut YOLO export letterboxes through npu.yolo like
pipelines/yolov8n/3b_quantize_cut.py, and MODNet reads through npu.modnet.preprocess like
pipelines/modnet/3_quantize.py. The recorded listing is what the owned producer must replay.

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
from quant.quantize import graph_family, simplify_for
from quant.sources import CocoSource, ImageFolderSource, ModnetSource, as_reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-model", type=Path, default=Path("models/resnet50_fp32.onnx"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, default=None,
                        help="data/calib for ResNet, data/coco_calib for YOLO, data/modnet_calib for MODNet unless given")
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
    family = graph_family(graph)
    input_name = graph.model.graph.input[0].name
    if family == "yolo_cut":
        from npu.yolo import input_size
        imgsz = input_size(list(graph.value_shape(input_name) or ()), str(args.in_model))
        source = CocoSource(args.calib_dir or Path("data/coco_calib"), args.limit, imgsz, input_name)
    elif family == "modnet":
        shape = graph.value_shape(input_name)
        if shape is None or len(shape) != 4 or shape[2] != shape[3]:
            parser.error(f"MODNet expects a square NCHW input, got {shape}")
        cfg = json.loads(args.cfg_path.read_text(encoding="utf-8"))
        if (cfg.get("height"), cfg.get("width")) != (shape[2], shape[3]):
            parser.error("Preprocessing config size does not match the model input")
        source = ModnetSource(args.calib_dir or Path("data/modnet_calib"), args.limit, shape[2], input_name)
    else:
        cfg = json.loads(args.cfg_path.read_text(encoding="utf-8"))
        if graph.value_shape(input_name) != (1, *cfg["input_size"]):
            parser.error("Preprocessing config does not match model input shape")
        source = ImageFolderSource(args.calib_dir or Path("data/calib"), cfg, args.limit, input_name)
    cfg = source.preprocess()
    with args.in_model.open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "machine": "Desktop 2 / Ryzen 7 8700G", "family": family, "include_cle": args.cle, "include_fast_ft": args.adaround,
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
        # Record the patterns on the graph Quark actually equalizes: its SimplifyModel step
        # runs first, and on MODNet it removes a Resize and reorders the node list the
        # matcher walks. simplify_for is a no-op for the other families.
        equalized = Graph(onnx.load(args.in_model))
        simplify_for(equalized, family)
        patterns = Equalization(equalized.model, op_types, [], []).get_cle_pattern_pair()
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
