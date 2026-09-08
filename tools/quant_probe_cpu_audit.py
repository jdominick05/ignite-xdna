"""Check recorded probes against CPU ORT with all graph optimizations disabled."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npu.preprocess import build_transform
from quant.probe import output_difference
from quant.quantize import file_hash


def reference(path, images):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    key = session.get_inputs()[0].name
    return np.concatenate([session.run(None, {key: image})[0] for image in images])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--base", type=Path, default=Path("models/resnet50_own_nocle_c64.onnx"))
    args = parser.parse_args()
    records = sorted(Path("models").glob(args.tag + "_*.onnx.probe.json"))
    if not records:
        parser.error("No probe records found")
    rows, cache = [], {}
    for path in records:
        report = json.loads(path.read_text(encoding="utf-8"))
        model = Path(str(path).removesuffix(".probe.json"))
        row = {"mutation": report["mutation"]["name"], "recorded_status": report["status"],
               "model_sha256": report["model_sha256"], "inputs_sha256": report["inputs_sha256"]}
        if report["status"] != "completed":
            row["note"] = "No completed NPU output; no placement or numerical verdict"
            rows.append(row)
            continue
        if file_hash(model) != report["model_sha256"] or file_hash(args.base) != report["base_sha256"]:
            raise ValueError("Model hash changed since the measured run")
        transform = build_transform(report["preprocess"])
        images = [transform(Path(p)) for p in report["listing"]]
        digest = hashlib.sha256(b"".join(i.tobytes() for i in images)).hexdigest()
        if digest != report["inputs_sha256"]:
            raise ValueError("Preprocessed inputs changed since the measured run")
        if digest not in cache:
            cache[digest] = reference(args.base, images)
        result = reference(model, images)
        with np.load(str(model) + ".outputs.npz") as stored:
            row["unoptimized_vs_baseline_unoptimized"] = output_difference(result, cache[digest])
            row["optimized_cpu_vs_unoptimized_cpu"] = output_difference(stored["cpu"], result)
            row["npu_vs_unoptimized_cpu"] = output_difference(stored["npu"], result)
        row["npu_nodes"] = report["ep"]["npu"]
        row["total_nodes"] = report["ep"]["total"]
        rows.append(row)
        print("CPU_REFERENCE_AUDIT_ROW", json.dumps(row), flush=True)
    print("CPU_REFERENCE_AUDIT", json.dumps({"ort_version": ort.__version__,
                                             "reference": "ORT_DISABLE_ALL CPUExecutionProvider",
                                             "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
