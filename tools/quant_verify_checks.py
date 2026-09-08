"""Exercise graph-diff rejection on mutations of a real emitted model (no NPU)."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant.graph import Graph
from quant.pow2 import quantize
from quant.verify import graph_diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    original = Graph.load(args.model)

    def clone():
        return Graph(onnx.load_model_from_string(original.model.SerializeToString()))

    checks = {}
    renamed = clone()
    anchors = {v.name for v in (*renamed.model.graph.input, *renamed.model.graph.output,
                                *renamed.model.graph.initializer)}
    names = {out: "renamed_" + out for n in renamed.nodes() for out in n.output if out not in anchors}
    for node in renamed.nodes():
        node.name = "node_" + node.name
        for ports in (node.input, node.output):
            for i, name in enumerate(ports):
                ports[i] = names.get(name, name)
    for value in renamed.model.graph.value_info:
        value.name = names.get(value.name, value.name)
    nodes = renamed.nodes()[::-1]
    del renamed.model.graph.node[:]
    renamed.model.graph.node.extend(nodes)
    renamed.topo_sort()
    checks["rename_and_resort_accepted"] = graph_diff(original, renamed).ok(0)
    del renamed

    rewired = clone()
    residual = next(n for n in rewired.nodes() if n.op_type == "Add")
    residual.input[1] = residual.input[0]
    checks["residual_rewire_rejected"] = bool(graph_diff(original, rewired).node_delta)
    del rewired

    scaled = clone()
    name = next(t.name for t in scaled.model.graph.initializer if t.name.endswith("_scale"))
    scaled.set_initializer(name, scaled.initializer(name) * np.float32(2))
    checks["changed_scale_rejected"] = not graph_diff(original, scaled).ok()
    del scaled

    modified = clone()
    name = next(t.name for t in modified.model.graph.initializer if t.name.endswith("_quantized"))
    array = modified.initializer(name).copy()
    first = int(array.flat[0])
    direction = 1 if first <= 125 else -1
    array.flat[0] = first + direction
    modified.set_initializer(name, array)
    difference = graph_diff(original, modified)
    checks["one_lsb_accepted_and_counted"] = (difference.ok(1) and not difference.ok(0) and
                                               difference.weight_changed_elements == {name: 1})
    array.flat[0] = first + 2 * direction
    modified.set_initializer(name, array)
    checks["two_lsb_rejected"] = not graph_diff(original, modified).ok(1)
    values = np.array([-128, -127, -1.5, -0.5, 0.5, 1.5, 127, 128], dtype=np.float32)
    checks["ties_even_symmetric_clip"] = np.array_equal(quantize(values, 0, 0, "int8"),
                                                        [-127, -127, -2, 0, 0, 2, 127, 127])
    print("REAL_MODEL_MUTATION_CHECKS", json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)
    print("QUANT VERIFY CHECKS PASS")


if __name__ == "__main__":
    main()
