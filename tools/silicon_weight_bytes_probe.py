"""Measure the head-cut artifact's parameters and actual initializer storage."""
import hashlib
import platform
from collections import defaultdict
from pathlib import Path

import onnx
import onnx_tool
from onnx import numpy_helper


def main():
    path = Path(__file__).resolve().parents[1] / "models/yolov8n_cut_xint8.onnx"
    print("machine:", platform.node(), platform.processor())
    print("artifact:", path.name)
    print("sha256:", hashlib.sha256(path.read_bytes()).hexdigest())
    model = onnx_tool.Model(str(path))
    model.graph.shape_infer()
    model.graph.profile()
    print("onnx_tool.graph.params:", model.graph.params)
    graph = onnx.load(path).graph
    groups = defaultdict(lambda: [0, 0, 0])
    for tensor in graph.initializer:
        array = numpy_helper.to_array(tensor)
        counts = groups[str(array.dtype)]
        counts[0] += 1
        counts[1] += array.size
        counts[2] += array.nbytes
    for dtype, counts in sorted(groups.items()):
        print(f"initializer dtype={dtype} tensors={counts[0]} elements={counts[1]} bytes={counts[2]}")
    print("initializer_total_bytes:", sum(c[2] for c in groups.values()))
    initializers = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    producers = {o: n for n in graph.node for o in n.output}
    weights, biases = {}, {}
    convs = [n for n in graph.node if n.op_type == "Conv"]
    for node in convs:
        for index, tensors in ((1, weights), (2, biases)):
            if len(node.input) <= index:
                continue
            key = node.input[index]
            producer = producers.get(key)
            if producer is not None and producer.op_type == "DequantizeLinear":
                key = producer.input[0]
            tensors[key] = initializers[key]
    print("Conv_nodes:", len(convs))
    for label, tensors in (("Conv_weights", weights), ("Conv_biases", biases)):
        print(label, "tensors:", len(tensors), "elements:", sum(a.size for a in tensors.values()),
              "bytes:", sum(a.nbytes for a in tensors.values()), "dtypes:", sorted({str(a.dtype) for a in tensors.values()}))
    reachable_bytes = 16 * 65536 + 4 * 524288
    print("DERIVED_reachable_L1_plus_MemTile_bytes:", reachable_bytes)
    print("DERIVED_initializer_excess_bytes:", sum(c[2] for c in groups.values()) - reachable_bytes)
    print("VERDICT: total artifact initializers exceed reachable SRAM before activation/workspace allocation;")
    print("whole-artifact residency is closed; compact packing and partial residency remain open experiments.")
    print("Scope: artifact storage only; graph.params counts elements, not dtype-aware bytes.")
    print("No NPU context opened; no latency or runtime-residency claim.")


if __name__ == "__main__":
    main()
