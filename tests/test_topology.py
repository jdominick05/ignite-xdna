import onnx
from onnx import TensorProto, helper
import pytest

from ignite_xdna.compiler.topology import audit_model, enforce


def model_with_convs(*specs):
    nodes = []
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])]
    previous = "x"
    for i, (kernel, stride) in enumerate(specs):
        weight = f"w{i}"
        output = f"y{i}"
        nodes.append(helper.make_node(
            "Conv", [previous, weight], [output], name=f"conv{i}",
            kernel_shape=[kernel, kernel], strides=[stride, stride],
        ))
        nodes[-1].input[1] = weight
        previous = output
    nodes.append(helper.make_node("Identity", [previous], ["out"]))
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, None)
    initializers = [
        helper.make_tensor(f"w{i}", TensorProto.FLOAT, [4, 3 if i == 0 else 4, k, k], [0.0] * (4 * (3 if i == 0 else 4) * k * k))
        for i, (k, _) in enumerate(specs)
    ]
    return helper.make_model(helper.make_graph(nodes, "topology", inputs, [output], initializers))


def test_audit_reports_wide_kernel_and_missing_entry_patchification():
    findings = audit_model(model_with_convs((5, 1)))
    assert [(f.kind, f.node_name) for f in findings] == [
        ("five_by_five", "conv0"),
        ("high_resolution_entry", "conv0"),
    ]


def test_stride_two_entry_has_no_high_resolution_finding():
    findings = audit_model(model_with_convs((3, 2)))
    assert [f.kind for f in findings] == []


def test_refuse_policy_is_explicit_and_report_is_non_destructive():
    model = model_with_convs((5, 1))
    assert len(enforce(model, "report")) == 2
    with pytest.raises(ValueError, match="topology policy"):
        enforce(model, "refuse")


def test_unknown_policy_fails_closed():
    with pytest.raises(ValueError, match="unknown topology policy"):
        enforce(model_with_convs((3, 2)), "ignore")
