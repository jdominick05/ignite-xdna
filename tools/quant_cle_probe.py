"""CLE disambiguator: equalize the float export with Quark and with Ignition, compare bytes.

Runs before any calibration so a mismatch is attributable to cross-layer equalization
alone. Quark's cle_transforms receives the resolved default op-type list (the union of
its three registries, as its pipeline resolves it) and the audited XINT8 defaults;
Ignition's cross_layer_equalize receives the same float graph. The gate is the ordered
pattern list and every float initializer byte for byte. Only this reference process
imports Quark. Nothing is written except the log.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.cle import cross_layer_equalize
from quant.graph import Graph


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clone(model: onnx.ModelProto) -> onnx.ModelProto:
    return onnx.ModelProto.FromString(model.SerializeToString())


def float_initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer
            if t.data_type == onnx.TensorProto.FLOAT}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-model", type=Path, default=Path("models/resnet50_fp32.onnx"))
    parser.add_argument("--steps", type=int, default=1, help="CLESteps; the audited default is 1")
    args = parser.parse_args()
    if any(name.startswith("quark") for name in sys.modules):
        raise RuntimeError("Quark must be imported only after the owned modules")
    base = onnx.load(args.in_model)
    original = float_initializers(base)

    from quark.onnx.algorithm.cle.equalization import Equalization, cle_transforms
    from quark.onnx.quantizers.registry import NPUCnnRegistry, QDQRegistry, QLinearOpsRegistry
    op_types = sorted(set(QLinearOpsRegistry) | set(QDQRegistry) | set(NPUCnnRegistry))
    records = []
    logger = logging.getLogger("quark.onnx.algorithm.cle.equalization_screen")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    convs = [n for n in base.graph.node if n.op_type == "Conv"]
    setup = {
        "in_model": args.in_model.as_posix(), "sha256": file_hash(args.in_model),
        "versions": {p: metadata.version(p) for p in ("amd-quark", "onnx", "numpy")},
        "op_types_to_quantize": op_types, "steps": args.steps,
        "convs": len(convs), "convs_with_group_attribute": sum(any(a.name == "group" for a in n.attribute) for n in convs),
        "float_initializers": len(original),
    }
    print("CLE_PROBE_SETUP", json.dumps(setup, indent=2), flush=True)

    quark_patterns = Equalization(clone(base), op_types, [], []).get_cle_pattern_pair()
    quark_pairs = [[p[1].name, p[-1].name, list(p[0])] for p in quark_patterns]
    start = time.perf_counter()
    quark_model = cle_transforms(clone(base), op_types, [], [], cle_steps=args.steps, cle_balance_method="max",
                                 cle_weight_threshold=0.5, cle_scale_append_bias=True,
                                 cle_scale_use_threshold=True, cle_total_layer_diff_threshold=2e-7)
    quark_seconds = time.perf_counter() - start
    quark_values = float_initializers(quark_model)

    graph = Graph(clone(base))
    graph.topo_sort()
    start = time.perf_counter()
    report = cross_layer_equalize(graph, steps=args.steps)
    ignition_seconds = time.perf_counter() - start
    ignition_values = float_initializers(graph.model)
    ignition_pairs = [[p["head"], p["tail"], p["chain"]] for p in report.pairs]

    names = sorted(original)
    changed_quark = [n for n in names if not np.array_equal(original[n], quark_values[n])]
    changed_ignition = [n for n in names if not np.array_equal(original[n], ignition_values[n])]
    mismatch = {}
    for name in names:
        q, i = quark_values.get(name), ignition_values.get(name)
        if q is None or i is None or q.dtype != i.dtype or q.shape != i.shape:
            mismatch[name] = "missing/dtype/shape"
        elif q.tobytes() != i.tobytes():
            mismatch[name] = {"max_abs": float(np.max(np.abs(q - i))), "elements": int(np.count_nonzero(q != i))}
    result = {
        "quark_log": records,
        "quark_pattern_count": len(quark_pairs), "ignition_pattern_count": report.pattern_count,
        "unique_pairs": report.unique_pairs, "patterns_equal": quark_pairs == ignition_pairs,
        "steps": [sum("Total CrossLayerEqualization steps" in m for m in records), report.steps],
        "changed_initializers": [len(changed_quark), len(changed_ignition)],
        "changed_equal": changed_quark == changed_ignition,
        "byte_mismatches": mismatch, "seconds": [round(quark_seconds, 3), round(ignition_seconds, 3)],
        "ignition_last_diff": report.last_diff,
    }
    print("CLE_PROBE_RESULT", json.dumps(result, indent=2), flush=True)
    print("CLE_PAIRS", json.dumps(ignition_pairs), flush=True)
    print("CLE_SCALES", json.dumps(report.scaled, indent=1), flush=True)
    passed = result["patterns_equal"] and not mismatch and result["changed_equal"]
    print("CLE_PROBE_PASS", passed, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
