"""Check Alpha entry points and fail-closed boundaries using actual ResNet artifacts.

No NPU is mocked or requested here. Full execution remains a separate logged gate.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant import __version__
from quant.cli import main as cli
from quant.quantize import file_hash, quantize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Fresh Alpha output to check")
    args = parser.parse_args()
    original = Path("models/resnet50_fp32.onnx")
    reference = Path("models/resnet50_quark_nocle_c64.onnx")
    digest = file_hash(args.model)
    report = json.loads(Path(str(args.model) + ".quant.json").read_text(encoding="utf-8"))
    model = onnx.load(args.model)
    assert model.producer_name == report["producer"] == "Ignition"
    assert model.producer_version == report["producer_version"] == __version__
    assert report["output_sha256"] == digest
    assert report["scope"] == "folded_resnet_nocle" and report["cle"] is False
    assert report["mode"] == "own_minmse_nocle" and report["calibration"]["images"] == 64
    assert not report["quark_imported"] and not report["torch_imported"]
    print("ALPHA_ARTIFACT", json.dumps({"model_sha256": digest, "version": __version__,
                                       "calibration_images": report["calibration"]["images"]}))
    version = subprocess.run(["python", "-m", "quant", "--version"], capture_output=True, text=True, check=True)
    assert version.stdout.strip() == f"Ignition {__version__} (Alpha)"
    print(version.stdout.strip())
    legacy = subprocess.run(["python", "pipelines/resnet50/3c_quantize_own.py", "--help"],
                            capture_output=True, text=True, check=True)
    assert "--no-cle" in legacy.stdout and "--scales-from" in legacy.stdout
    before = list(sys.meta_path)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        cli(["inspect", str(args.model)])
    assert json.loads(output.getvalue())["models"][0]["sha256"] == digest
    assert sys.meta_path == before
    for extra in ([], ["--no-cle", "--limit", "0"], ["--no-cle"]):
        result = subprocess.run(["python", "-m", "quant", "quantize", "--out", str(args.model), *extra],
                                capture_output=True, text=True)
        assert result.returncode == 2, result.stderr
        print("CLI_REJECT", result.stderr.splitlines()[-1])
    assert file_hash(args.model) == digest

    with tempfile.TemporaryDirectory(prefix="ignition-alpha-checks-", dir=ROOT / "scratch") as directory:
        tmp = Path(directory)
        for case, expected in (("opset", "opset"), ("batch", "batch 1"),
                               ("dynamic", "static"), ("operator", "Unsupported float operator"),
                               ("gap", "7x7 GAP")):
            candidate = onnx.load(original)
            dims = candidate.graph.input[0].type.tensor_type.shape.dim
            if case == "opset":
                candidate.opset_import[0].version = 19
            elif case == "batch":
                dims[0].dim_value = 2
            elif case == "dynamic":
                dims[0].dim_param = "batch"
            elif case == "operator":
                next(n for n in candidate.graph.node if n.op_type == "Relu").op_type = "Sigmoid"
            elif case == "gap":
                dims[2].dim_value = dims[3].dim_value = 256
                del candidate.graph.value_info[:]
            path, out = tmp / (case + ".onnx"), tmp / (case + "_quant.onnx")
            onnx.save(candidate, path)
            try:
                quantize(path, out, scales_from=reference)
            except ValueError as exc:
                assert expected in str(exc), (case, str(exc))
                print("GRAPH_REJECT", case, str(exc))
            else:
                raise AssertionError(f"Unsupported graph passed: {case}")
            assert not out.exists() and not Path(str(out) + ".quant.json").exists()
    assert not any(n.split(".", 1)[0] in ("quark", "torch") for n in sys.modules)
    print("IGNITION ALPHA CHECKS PASS; no NPU sessions requested")


if __name__ == "__main__":
    main()
