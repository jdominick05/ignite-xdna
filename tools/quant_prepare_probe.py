"""Preparation disambiguator: Quark's pre-calibration graph vs Ignition's prepared graph.

Runs on the float export before any calibration, so a mismatch is attributable to graph
preparation alone. Three questions, each logged and gated:

1. Whole pre-process. Quark's apply_pre_process with the resolved static op types, the
   audited XINT8 defaults, the hardware-compatibility conversions its quantize() forces on
   for enable_npu_cnn and, with --cle, its own CLE, against Ignition's load -> CLE ->
   prepare in the order quantize() applies them. The gate is graph_diff: ordered structure
   and every initializer byte for byte.
2. Steps in isolation. Each of Quark's preparation steps applied alone to the float export
   (onnxslim, ORT basic optimization as Quark runs it, BatchNorm folding, Split to Slice),
   reporting the node and initializer delta each one makes. This is what backs a claim that
   a step is a no-op on a given export.
3. Replay. With --replay-from, the committed artifact's positions are re-emitted through
   Ignition (scales_from) and graph_diff'd against the artifact: refinement fixed point on
   both sides, exact positions, int8 initializers byte for byte.

Only this reference process imports Quark, after the owned modules. The replay writes a
temporary model under a scratch directory that is removed at exit; nothing else is written
except the log.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.cle import cross_layer_equalize
from quant.graph import Graph
from quant.quantize import check_family, graph_family, prepare, quantize
from quant.sources import CocoSource, ImageFolderSource, as_reader
from quant.verify import graph_diff


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clone(model: onnx.ModelProto) -> onnx.ModelProto:
    return onnx.ModelProto.FromString(model.SerializeToString())


def delta(before: onnx.ModelProto, after: onnx.ModelProto) -> dict:
    """What a step did to a graph: operator counts, node names, initializer names and bytes."""
    ops_before = Counter(n.op_type for n in before.graph.node)
    ops_after = Counter(n.op_type for n in after.graph.node)
    names_before = {n.name for n in before.graph.node}
    names_after = {n.name for n in after.graph.node}
    init_before = {t.name: t.SerializeToString() for t in before.graph.initializer}
    init_after = {t.name: t.SerializeToString() for t in after.graph.initializer}
    changed = sorted(k for k in init_before if k in init_after and init_before[k] != init_after[k])
    diff = graph_diff(Graph(clone(before)), Graph(clone(after)))
    return {
        "nodes": [len(before.graph.node), len(after.graph.node)],
        "op_delta": {op: [ops_before.get(op, 0), ops_after.get(op, 0)]
                     for op in sorted(ops_before.keys() | ops_after.keys()) if ops_before.get(op, 0) != ops_after.get(op, 0)},
        "node_names_removed": sorted(names_before - names_after)[:8],
        "node_names_added": len(names_after - names_before),
        "initializers": [len(init_before), len(init_after)],
        "initializers_removed": sorted(set(init_before) - set(init_after))[:8],
        "initializers_added": len(set(init_after) - set(init_before)),
        "initializer_bytes_changed": changed[:8],
        "value_info": [len(before.graph.value_info), len(after.graph.value_info)],
        "structure_equal": not diff.node_delta and not diff.init_exact_mismatch and not diff.weight_lsb,
    }


def source_for(graph: Graph, family: str, calib_dir: Path | None, limit: int, cfg_path: Path):
    """The CLI's reader for the family: letterbox at the graph's input size, or the timm config."""
    input_name = graph.model.graph.input[0].name
    if family == "yolo_cut":
        from npu.yolo import input_size
        imgsz = input_size(list(graph.value_shape(input_name) or ()), input_name)
        return CocoSource(calib_dir or ROOT / "data" / "coco_calib", limit, imgsz, input_name)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if graph.value_shape(input_name) != (1, *cfg["input_size"]):
        raise ValueError("Preprocess config input_size does not match the graph input")
    return ImageFolderSource(calib_dir or ROOT / "data" / "calib", cfg, limit, input_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-model", type=Path, default=Path("models/yolov8n_cut.onnx"))
    parser.add_argument("--replay-from", type=Path, default=None,
                        help="Committed XINT8 artifact of the same export to re-emit from its positions")
    parser.add_argument("--calib-dir", type=Path, default=None, help="data/coco_calib or data/calib by family")
    parser.add_argument("--limit", type=int, default=2, help="Images Quark's pre-process reader carries (CLE needs none)")
    parser.add_argument("--cfg-path", type=Path, default=Path("models/preprocess_config.json"),
                        help="ResNet only: the classification preprocess config the CLI reads")
    parser.add_argument("--cle", action="store_true", help="Keep Quark's default include_cle=True on both sides")
    args = parser.parse_args()
    if any(name.split(".", 1)[0] in ("quark", "torch") for name in sys.modules):
        raise RuntimeError("Quark must be imported only after the owned modules")

    base = onnx.load(args.in_model)
    graph = Graph(clone(base))
    family = graph_family(graph)
    graph.infer_shapes()
    check_family(graph, family)
    source = source_for(graph, family, args.calib_dir, args.limit, args.cfg_path)
    setup = {
        "in_model": args.in_model.as_posix(), "sha256": file_hash(args.in_model), "family": family,
        "replay_from": args.replay_from.as_posix() if args.replay_from else None,
        "replay_sha256": file_hash(args.replay_from) if args.replay_from else None,
        "cle": args.cle, "reader_images": len(source), "preprocess": source.preprocess(),
        "versions": {p: metadata.version(p) for p in ("amd-quark", "onnx", "onnxruntime", "numpy", "onnxslim")},
        "ops": dict(sorted(Counter(n.op_type for n in base.graph.node).items())),
    }
    print("PREPARE_PROBE_SETUP", json.dumps(setup, indent=2), flush=True)

    # Ignition, in quantize()'s order: CLE on the float graph, then the family's preparation.
    start = time.perf_counter()
    ignition = {}
    if args.cle:
        ignition["cle"] = asdict(cross_layer_equalize(graph))
        graph.infer_shapes()
    ignition["prepare"] = prepare(graph, family)
    ignition["seconds"] = round(time.perf_counter() - start, 3)
    ignition["delta_from_export"] = delta(base, graph.model)

    replay = None
    scratch = Path(tempfile.mkdtemp(prefix="prepare_probe_", dir=str(ROOT / "scratch")))
    try:
        if args.replay_from:
            out = scratch / "replay.onnx"
            start = time.perf_counter()
            report = quantize(args.in_model, out, scales_from=args.replay_from, cle=args.cle)
            reference = Graph.load(args.replay_from)
            diff = graph_diff(Graph.load(out), reference)
            replay = {
                "seconds": round(time.perf_counter() - start, 3),
                "reference_refine_moves": len(report["reference_refine"]["log"]),
                "replay_refine_moves": len(report["refine"]["log"]),
                "replay_refine_loops": report["refine"]["loops"],
                "emit": report["emit"], "gap_mul": report["gap_mul"], "hardsigmoid": report["hardsigmoid"],
                "graph_diff": asdict(diff),
                "exact": diff.ok(weight_tol_lsb=0),
                "positions": len(report["positions"]),
            }
            print("PREPARE_REPLAY", json.dumps(replay, indent=2), flush=True)

        # Quark, only from here on.
        from onnxruntime.quantization.onnx_model import ONNXModel
        from quark.onnx.calibration import CachedDataReader
        from quark.onnx.optimizations import optimize_model, optimize_model_using_onnxrt, optimize_model_using_onnxslim
        from quark.onnx.preprocess import apply_pre_process
        from quark.onnx.quantization.config import get_default_config
        from quark.onnx.quantization.quantize import get_static_op_types

        qc = get_default_config("XINT8")
        qc.include_cle = args.cle
        extra = dict(qc.extra_options)
        # quantize() forces these on for enable_npu_cnn before the pre-process sees extra_options.
        for key in ("ConvertSplitToSlice", "ConvertBNToConv", "ConvertReduceMeanToGlobalAvgPool", "SplitLargeKernelPool"):
            extra.setdefault(key, True)
        op_types = get_static_op_types(clone(base), qc.op_types_to_quantize, [], qc.enable_npu_cnn, False,
                                       qc.quant_format, extra)
        quark_setup = {
            "op_types_to_quantize": sorted(op_types), "extra_options": extra,
            "optimize_model": qc.optimize_model, "include_cle": qc.include_cle, "enable_npu_cnn": qc.enable_npu_cnn,
        }
        print("PREPARE_PROBE_QUARK", json.dumps(quark_setup, indent=2, default=str), flush=True)

        steps = {}
        start = time.perf_counter()
        steps["onnxslim"] = delta(base, optimize_model_using_onnxslim(clone(base)))
        work = scratch / "float.onnx"
        shutil.copy(args.in_model, work)
        steps["ort_basic"] = delta(base, optimize_model_using_onnxrt(work, scratch / "optimized_model.onnx"))
        steps["fold_batch_norm"] = delta(base, optimize_model(
            clone(base), op_types, [], [], convert_bn_to_conv=False, convert_reduce_mean_to_global_avg_pool=False,
            split_large_kernel_pool=False, convert_split_to_slice=False, fuse_instance_norm=False, fuse_l2_norm=False,
            fuse_gelu=False, fuse_layer_norm=False, fold_batch_norm=True, convert_clip_to_relu=False,
            fold_batch_norm_after_concat=True, dedicate_dq_node=False))
        steps["hardware_compat"] = delta(base, optimize_model(
            clone(base), op_types, [], [], convert_bn_to_conv=extra["ConvertBNToConv"],
            convert_reduce_mean_to_global_avg_pool=extra["ConvertReduceMeanToGlobalAvgPool"],
            split_large_kernel_pool=extra["SplitLargeKernelPool"], convert_split_to_slice=extra["ConvertSplitToSlice"],
            fuse_instance_norm=False, fuse_l2_norm=False, fuse_gelu=False, fuse_layer_norm=False,
            fold_batch_norm=False, convert_clip_to_relu=False, fold_batch_norm_after_concat=False, dedicate_dq_node=False))
        steps["seconds"] = round(time.perf_counter() - start, 3)
        print("PREPARE_PROBE_STEPS", json.dumps(steps, indent=2), flush=True)

        work = scratch / "quark_float.onnx"
        shutil.copy(args.in_model, work)
        start = time.perf_counter()
        quark_model = apply_pre_process(
            onnx.load(work), work, CachedDataReader(as_reader(source), None, False, False),
            calibrate_method=qc.calibrate_method, activation_type=qc.activation_type, weight_type=qc.weight_type,
            nodes_to_quantize=[], nodes_to_exclude=[], op_types_to_quantize=op_types,
            use_external_data_format=False, convert_fp16_to_fp32=False, convert_nchw_to_nhwc=False,
            optimize_model_flag=qc.optimize_model, include_cle=qc.include_cle, include_sq=False,
            include_rotation=False, extra_options=extra)
        topo = ONNXModel(quark_model)
        topo.topological_sort()
        quark_model = onnx.shape_inference.infer_shapes(topo.model)
        quark_seconds = round(time.perf_counter() - start, 3)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    whole = graph_diff(Graph(clone(quark_model)), Graph(clone(graph.model)))
    quark_names = [n.name for n in quark_model.graph.node]
    ignition_names = [n.name for n in graph.model.graph.node]
    result = {
        "quark_seconds": quark_seconds, "ignition": ignition,
        "quark_delta_from_export": delta(base, quark_model),
        "node_names_equal_as_sets": set(quark_names) == set(ignition_names),
        "node_order_equal": quark_names == ignition_names,
        "graph_diff": asdict(whole),
        "whole_pre_process_equal": not whole.node_delta and not whole.init_exact_mismatch and not whole.weight_lsb,
    }
    print("PREPARE_PROBE_RESULT", json.dumps(result, indent=2), flush=True)
    passed = result["whole_pre_process_equal"] and (replay is None or (
        replay["exact"] and replay["reference_refine_moves"] == 0 and replay["replay_refine_moves"] == 0))
    print("PREPARE_PROBE_PASS", passed, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
