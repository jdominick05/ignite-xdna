"""Ignition XINT8 producer for folded ResNet, head-cut YOLOv8 and MODNet, with optional CLE and position replay."""
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
from .passes import avgpool_dpu_scale, hardsigmoid_dpu_scale, sigmoid_to_hardsigmoid, simplify, split_to_slice
from .qdq import emit, read_pos_table, quantizable_tensors
from .refine import refine

YOLO_MARKERS = {"Sigmoid", "Split", "Slice", "Concat", "Resize"}
MODNET_MARKERS = {"Clip"}  # neither measured classifier nor head-cut YOLO export carries one


def _check_imports() -> None:
    if any(name.split(".", 1)[0] in ("quark", "torch") for name in sys.modules):
        raise RuntimeError("Owned core must run without Quark or torch imported")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def graph_family(g: Graph) -> str:
    """folded_resnet, yolo_cut or modnet, decided by the operators present; anything else fails later."""
    ops = {n.op_type for n in g.nodes()}
    if ops & MODNET_MARKERS:
        return "modnet"
    return "yolo_cut" if ops & YOLO_MARKERS else "folded_resnet"


def check_family(g: Graph, family: str) -> None:
    if len(g.model.graph.input) != 1:
        raise ValueError("Ignition supports one image input")
    if family == "folded_resnet":
        if len(g.model.graph.output) != 1:
            raise ValueError("Folded ResNet support expects one classifier output")
        for node in g.nodes():
            if node.op_type == "GlobalAveragePool":
                shape = g.value_shape(node.input[0])
                if shape is None or len(shape) != 4 or shape[-2:] != (7, 7):
                    raise ValueError(f"Ignition Alpha supports only 7x7 GAP, got {shape}")
    elif family == "modnet":
        if len(g.model.graph.output) != 1:
            raise ValueError("MODNet support expects the single matte-logits output")
        for node in g.nodes():
            if node.op_type == "GlobalAveragePool":
                shape = g.value_shape(node.input[0])
                if shape is None or len(shape) != 4:
                    raise ValueError(f"MODNet GlobalAveragePool needs a rank-4 input, got {shape}")
    elif not g.model.graph.output:
        raise ValueError("Head-cut YOLO support expects the head outputs")


def simplify_for(g: Graph, family: str) -> dict:
    """The vendor's SimplifyModel step, which precedes CLE and so precedes prepare.

    Only MODNet needs it: on the two gated exports onnxslim is measured to be a structural
    no-op (`tools/quant_prepare_probe.py`'s isolated steps), so running it there would
    change nothing that is already gated, and it is not run.
    """
    return {"simplify": simplify(g)} if family == "modnet" else {}


def prepare(g: Graph, family: str) -> dict:
    """The vendor's hardware-compatibility pre-process for this family, after CLE."""
    report = {}
    if family == "yolo_cut":
        report["split_to_slice"] = split_to_slice(g)
    g.infer_shapes()
    # Reject unsupported graphs before any costly calibration or output write.
    quantizable_tensors(g)
    return report


def prepared_graph(model_in: Path) -> tuple[Graph, str]:
    """Load, prepare and shape-infer a float export; the wrappers size calibration from it."""
    graph = Graph.load(Path(model_in))
    family = graph_family(graph)
    graph.infer_shapes()
    check_family(graph, family)
    simplify_for(graph, family)
    graph.infer_shapes()
    prepare(graph, family)
    return graph, family


def quantize(model_in: Path, model_out: Path, *, scales_from: Path | None = None,
             source=None, preprocess: dict | None = None,
             scratch: Path | None = None, cle: bool = False,
             cle_guard: float | None = 2.0,
             calib_method: str = "hist", hist_bins: int = 2048) -> dict:
    _check_imports()
    if (scales_from is None) == (source is None):
        raise ValueError("Supply exactly one of a calibration source or scales_from")
    model_in, model_out = Path(model_in), Path(model_out)
    sidecar = Path(str(model_out) + ".quant.json")
    if model_out.exists() or sidecar.exists():
        raise FileExistsError("Choose a new model/sidecar name")
    start = time.perf_counter()
    graph = Graph.load(model_in)
    family = graph_family(graph)
    graph.infer_shapes()
    check_family(graph, family)
    report = {
        "producer": "Ignition", "producer_version": __version__, "family": family,
        "scope": f"{family}_{'cle' if cle else 'nocle'}", "format": "XINT8_QDQ",
        "mode": "reemit_from_positions" if scales_from else "own_minmse_cle" if cle else "own_minmse_nocle",
        "cle": cle,
        "input_sha256": file_hash(model_in),
        "versions": {p: metadata.version(p) for p in ("numpy", "onnx")},
    }
    if family == "modnet":
        # onnxslim's output defines this family's prepared graph, so the artifact records
        # which version produced it; a later release could simplify differently.
        report["versions"]["onnxslim"] = metadata.version("onnxslim")
    # SimplifyModel is the vendor's first pre-process step and precedes CLE, whose pattern
    # walk reads the node list it leaves behind.
    report["simplify"] = simplify_for(graph, family)
    if report["simplify"]:
        graph.infer_shapes()
    if cle:
        # Quark equalizes the float model before calibration (preproc.py apply_pre_process);
        # calibration and weight quantization then see the equalized initializers.
        report["cle_report"] = asdict(cross_layer_equalize(graph, max_scale_log2=cle_guard))
        graph.infer_shapes()
    # Quark's after-algorithm optimizations (Split to Slice) follow CLE and precede calibration.
    report["prepare"] = prepare(graph, family)
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
        if preprocess is None:
            raise ValueError("Independent calibration requires preprocessing metadata")
        if calib_method == "exact" and scratch is None:
            raise ValueError("Exact calibration requires scratch directory")
        from .calib import collect_and_choose
        positions, report["calibration"] = collect_and_choose(
            graph, source, scratch, method=calib_method, num_bins=hist_bins
        )
        report["preprocess"] = preprocess
    report["emit"] = asdict(emit(graph, positions))
    graph.infer_shapes()
    # The vendor's DPU simulation runs on the quantized graph, before refinement.
    report["gap_mul"] = avgpool_dpu_scale(graph)
    report["hardsigmoid"] = {"replaced": sigmoid_to_hardsigmoid(graph), "scaled": hardsigmoid_dpu_scale(graph)}
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
