"""Ignition XINT8 producer for folded ResNet, with optional CLE and position-table replay."""
from dataclasses import asdict
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

from . import __version__
from .cle import cross_layer_equalize
from .graph import Graph
from .passes import avgpool_dpu_scale
from .qdq import emit, read_pos_table, quantizable_tensors
from .refine import refine
from .sources import ImageFolderSource


def _check_imports() -> None:
    if any(name.split(".", 1)[0] in ("quark", "torch") for name in sys.modules):
        raise RuntimeError("Owned core must run without Quark or torch imported")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quantize(model_in: Path, model_out: Path, *, scales_from: Path | None = None,
             source: ImageFolderSource | None = None, preprocess: dict | None = None,
             scratch: Path | None = None, cle: bool = False) -> dict:
    _check_imports()
    if (scales_from is None) == (source is None):
        raise ValueError("Supply exactly one of a calibration source or scales_from")
    model_in, model_out = Path(model_in), Path(model_out)
    sidecar = Path(str(model_out) + ".quant.json")
    if model_out.exists() or sidecar.exists():
        raise FileExistsError("Choose a new model/sidecar name")
    start = time.perf_counter()
    graph = Graph.load(model_in)
    # Reject unsupported graphs before any costly calibration or output write.
    quantizable_tensors(graph)
    graph.infer_shapes()
    if len(graph.model.graph.input) != 1 or len(graph.model.graph.output) != 1:
        raise ValueError("Ignition Alpha supports one image input and one classifier output")
    for node in graph.nodes():
        if node.op_type == "GlobalAveragePool":
            shape = graph.value_shape(node.input[0])
            if shape is None or len(shape) != 4 or shape[-2:] != (7, 7):
                raise ValueError(f"Ignition Alpha supports only 7x7 GAP, got {shape}")
    report = {
        "producer": "Ignition", "producer_version": __version__,
        "scope": "folded_resnet_cle" if cle else "folded_resnet_nocle", "format": "XINT8_QDQ",
        "mode": "reemit_from_positions" if scales_from else "own_minmse_cle" if cle else "own_minmse_nocle",
        "cle": cle,
        "input_sha256": file_hash(model_in),
        "versions": {p: metadata.version(p) for p in ("numpy", "onnx")},
    }
    if cle:
        # Quark equalizes the float model before calibration (preproc.py apply_pre_process);
        # calibration and weight quantization then see the equalized initializers.
        report["cle_report"] = asdict(cross_layer_equalize(graph))
        graph.infer_shapes()
    if scales_from is not None:
        scales_from = Path(scales_from)
        reference = Graph.load(scales_from)
        positions = read_pos_table(reference)
        reference_refine = refine(reference)
        if reference_refine.log:
            raise ValueError("Reference positions are not a refinement fixed point")
        report["scales_from_sha256"] = file_hash(scales_from)
        report["reference_refine"] = asdict(reference_refine)
    else:
        if scratch is None or preprocess is None:
            raise ValueError("Independent calibration requires scratch and preprocessing metadata")
        from .calib import collect_and_choose
        positions, report["calibration"] = collect_and_choose(graph, source, scratch)
        report["preprocess"] = preprocess
    report["emit"] = asdict(emit(graph, positions))
    graph.infer_shapes()
    report["gap_mul"] = avgpool_dpu_scale(graph)
    report["refine"] = asdict(refine(graph))
    if not report["refine"]["converged"]:
        raise ValueError("Position refinement did not converge")
    graph.model.producer_name = "Ignition"
    graph.model.producer_version = __version__
    _check_imports()
    graph.save(model_out)
    report["positions"] = {name: asdict(tq) for name, tq in read_pos_table(graph).items()}
    report["output_sha256"] = file_hash(model_out)
    report["wall_seconds"] = time.perf_counter() - start
    report["quark_imported"] = any(name == "quark" or name.startswith("quark.") for name in sys.modules)
    report["torch_imported"] = "torch" in sys.modules
    sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
