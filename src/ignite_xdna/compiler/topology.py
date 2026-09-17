"""Model-side topology checks for the AIE2 convolution engine.

These checks do not rewrite a model. They make two traffic-heavy shapes
explicit before compilation: 5x5 convolutions and a first spatial convolution
that does not downsample. The ``refuse`` policy is opt-in because existing
benchmarks intentionally include both shapes.
"""
from __future__ import annotations

from dataclasses import dataclass
import onnx


@dataclass(frozen=True)
class TopologyFinding:
    kind: str
    node_name: str
    detail: str


def _ints(node: onnx.NodeProto, name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    for attr in node.attribute:
        if attr.name == name:
            return tuple(int(v) for v in attr.ints)
    return default


def audit_model(model: onnx.ModelProto) -> tuple[TopologyFinding, ...]:
    """Return traffic-related topology findings in graph order."""
    findings: list[TopologyFinding] = []
    first_spatial_conv = True
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Conv":
            continue
        kernel = _ints(node, "kernel_shape", (3, 3))
        strides = _ints(node, "strides", (1, 1))
        name = node.name or f"Conv_{index}"
        if kernel == (5, 5):
            findings.append(TopologyFinding(
                "five_by_five",
                name,
                "5x5 convolution retains the engine's 3x3 packet footprint with 73% unread slack; "
                "replace it with stacked 3x3 layers when model accuracy permits",
            ))
        elif len(kernel) == 2 and kernel[0] > 3 and kernel[1] > 3:
            findings.append(TopologyFinding(
                "wide_kernel",
                name,
                f"{kernel[0]}x{kernel[1]} convolution is wider than the native 3x3 path",
            ))
        if first_spatial_conv:
            first_spatial_conv = False
            if strides[0] == 1 and strides[1] == 1:
                findings.append(TopologyFinding(
                    "high_resolution_entry",
                    name,
                    "first spatial convolution is stride 1; stride 2 or 4 patchification could reduce "
                    "the high-resolution fill and activation traffic",
                ))
    return tuple(findings)


def enforce(model: onnx.ModelProto, policy: str = "report") -> tuple[TopologyFinding, ...]:
    """Audit ``model`` and optionally reject traffic-heavy topology findings."""
    if policy not in {"report", "refuse"}:
        raise ValueError(f"unknown topology policy {policy!r}; expected 'report' or 'refuse'")
    findings = audit_model(model)
    if policy == "refuse" and findings:
        summary = "; ".join(f"{f.node_name}: {f.kind}" for f in findings)
        raise ValueError(f"model violates topology policy: {summary}")
    return findings
